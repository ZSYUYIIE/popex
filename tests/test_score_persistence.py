from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import sqlite3
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import _run_score_job, create_app
from app.score_artifacts import (
    ScoreArtifactError,
    load_score_artifact,
    remove_unowned_score_artifacts,
)
from app.score_pipeline import (
    ScorePipelineError,
    ScorePipelineResult,
    construct_score,
)
from app.score_sources import current_score_fingerprint
from app.transcription_draft import write_transcription_draft
from app.transcription_events import write_raw_transcription
from test_harmony_api import (
    RAW_CREATED_AT,
    create_job,
    make_settings,
    publish_harmony,
    raw_events,
    raw_payload,
)
from test_interpretation_api import draft_payload

ATTEMPT_FILE_RE = re.compile(r"score-document\.[a-f0-9]{32}\.json")


def client_for(settings, **kwargs) -> TestClient:
    return TestClient(create_app(settings, **kwargs))


def job_json(client: TestClient, job_id: str) -> dict:
    return next(job for job in client.get("/api/jobs").json() if job["id"] == job_id)


def score_files(settings, job_id: str) -> list[str]:
    directory = settings.exports_dir / job_id / "score"
    return sorted(path.name for path in directory.iterdir()) if directory.exists() else []


def upstream_digests(settings, job_id: str) -> dict[str, str]:
    job_dir = settings.exports_dir / job_id
    return {
        str(path.relative_to(job_dir)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(job_dir.rglob("*"))
        if path.is_file() and "score" not in path.relative_to(job_dir).parts
    }


def publish_matching_interpretation(settings, job_id: str) -> dict:
    """Publish an editable draft whose one part covers the fixture's raw events."""
    payload = draft_payload()
    events = raw_events()
    ids = [event["id"] for event in events]
    payload["sourceTranscription"]["sourceEventIndex"] = [
        {
            "id": event["id"],
            "eventType": "pitched",
            "sourceKind": event["sourceKind"],
            "rawStartSeconds": event["startSeconds"],
            "rawEndSeconds": event["endSeconds"],
            "confidence": event["confidence"],
            "midiPitch": event["midiPitch"],
        }
        for event in events
    ]
    payload["parts"][0]["sourceEventIds"] = ids
    payload["voices"][0]["sourceEventIds"] = ids
    payload["phrases"][0].update(
        sourceEventIds=ids, rawStartSeconds=0.0, rawEndSeconds=0.9
    )
    template = payload["pitchedItems"][0]
    payload["pitchedItems"] = []
    for event in events:
        item = copy.deepcopy(template)
        item.update(
            id=f"pitched_{event['id']}",
            sourceEventIds=[event["id"]],
            rawStartSeconds=event["startSeconds"],
            rawEndSeconds=event["endSeconds"],
            pitch={
                "midiNote": event["midiNote"],
                "midiPitch": event["midiPitch"],
                "frequencyHz": event["frequencyHz"],
                "noteName": event["noteName"],
            },
        )
        payload["pitchedItems"].append(item)
    evidence = payload["interpretationEvidence"]
    evidence["pitchedPartInference"]["assignments"] = [
        {"eventId": event_id, "sourceEventIds": [event_id]} for event_id in ids
    ]
    evidence["rhythmInterpretation"]["event_interpretations"] = [
        {"eventId": event_id, "sourceEventIds": [event_id]} for event_id in ids
    ]
    write_transcription_draft(job_id, settings, payload)
    db.update_job(
        settings.database_path,
        job_id,
        interpretation_status="completed",
        interpretation_stage="completed",
        interpretation_progress=100,
        interpretation_version=payload["draftVersion"],
        interpretation_artifact_file_name="interpretation/draft.json",
        interpreted_at=payload["createdAt"],
        interpretation_part_count=1,
        interpretation_phrase_count=1,
        interpretation_pitched_item_count=len(ids),
        interpretation_percussion_item_count=0,
        interpretation_warning_count=len(payload["warnings"]),
    )
    return payload


def build(client: TestClient, job_id: str, *, force: bool = False):
    suffix = "?force=true" if force else ""
    return client.post(f"/api/jobs/{job_id}/score/construct{suffix}")


# ---------------------------------------------------------------------------
# Contract and happy path


def test_score_contract_appears_only_after_raw_transcription(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    pending = create_job(settings, transcribed=False)
    ready = create_job(settings)
    with client_for(settings) as client:
        assert "score" not in job_json(client, pending)
        assert build(client, pending).status_code == 409
        job = job_json(client, ready)
    score = job["score"]
    assert score["status"] == "not_started"
    assert score["canStart"] is True and score["canRebuild"] is False
    assert score["available"] is False and score["downloadUrls"] is None
    assert not any(key.startswith("score_") or key == "scored_at" for key in job)


def test_construct_query_and_conflicts_are_strict(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct?force=yes").status_code == 422
        assert build(client, "f" * 32).status_code == 404
        assert build(client, job_id).status_code == 202
        conflict = build(client, job_id)
        assert conflict.status_code == 409
        assert "force=true" in conflict.json()["detail"]
        assert build(client, job_id, force=True).status_code == 202


def test_successful_construction_persists_reviewable_document(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    before = upstream_digests(settings, job_id)
    with client_for(settings) as client:
        response = build(client, job_id)
        assert response.status_code == 202
        assert response.json()["score"]["status"] == "processing"
        record = db.get_job(settings.database_path, job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        summary = client.get(f"/api/jobs/{job_id}/score/saved").json()
        midi = client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi")
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
        exported = client.get(f"/api/jobs/{job_id}/score/saved/download?format=json")
        job = job_json(client, job_id)

    assert record["score_status"] == "completed"
    assert record["score_attempt_id"] is None
    assert record["score_source_fingerprint"] == current_score_fingerprint(record)
    assert score_files(settings, job_id) == [record["score_artifact_file_name"].split("/")[1]]
    assert job["score"]["available"] is True and job["score"]["stale"] is False
    assert job["score"]["canRebuild"] is True
    assert job["score"]["counts"] == {
        "measures": details["counts"]["measures"],
        "notes": 3,
        "chordSymbols": 0,
        "warnings": len(details["warnings"]),
    }

    assert details["version"] == "score-pipeline-v1"
    assert details["builderVersion"] == "score-construction-v1"
    assert details["sources"]["transcription"] == {
        "version": "raw-transcription-v1",
        "createdAt": RAW_CREATED_AT,
    }
    assert details["sources"]["harmony"] is None
    assert details["sources"]["interpretation"] is None
    assert details["layers"]["pitchedNotes"]["status"] == "included"
    for layer in ("chordSymbols", "partLabels", "percussion", "tablature"):
        assert details["layers"][layer]["status"] == "omitted"
        assert details["layers"][layer]["note"]
    notes = [note for measure in details["measures"] for note in measure["notes"]]
    assert sorted(note["id"] for note in notes) == ["p_c", "p_e", "p_g"]
    assert all(note["partId"] is None for note in notes)
    assert {note["rawMidiPitch"] for note in notes} == {60.12, 63.94, 67.04}
    assert "measures" not in summary

    assert midi.status_code == 200 and midi.content.startswith(b"MThd")
    assert 'filename="draft-score.mid"' in midi.headers["content-disposition"]
    root = ET.fromstring(musicxml.content)
    assert root.tag == "score-partwise"
    public = exported.json()
    text = exported.text
    assert "fileName" not in text
    assert record["score_attempt_id"] is None
    attempt = record["score_artifact_file_name"].split(".")[1]
    assert attempt not in text
    assert str(settings.data_dir) not in text
    assert public["sourceFingerprint"] == record["score_source_fingerprint"]
    assert upstream_digests(settings, job_id) == before


def test_harmony_becomes_measure_chord_symbols_with_evidence(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert details["layers"]["chordSymbols"]["status"] == "included"
    assert details["sources"]["harmony"]["version"]
    first = details["measures"][0]
    assert first["chordSymbol"] == "C"
    assert first["harmony"] and all(
        entry["unresolved"] is False and entry["symbol"] == "C"
        for entry in first["harmony"]
    )
    assert details["counts"]["chordSymbols"] >= 1
    words = [element.text for element in ET.fromstring(musicxml.content).iter("words")]
    assert "C" in words


def test_interpretation_parts_label_notes_without_changing_exports(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_matching_interpretation(settings, job_id)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert details["layers"]["partLabels"]["status"] == "included"
    assert details["parts"] == [
        {
            "id": "part_full_mix",
            "role": "pitched",
            "instrumentKind": "source_pitched_line",
            "sourceKind": "full_mix",
            "noteCount": 3,
        }
    ]
    notes = [note for measure in details["measures"] for note in measure["notes"]]
    assert {note["partId"] for note in notes} == {"part_full_mix"}
    assert details["counts"]["notesWithPart"] == 3
    assert len(list(ET.fromstring(musicxml.content).iter("score-part"))) == 1


def test_mismatched_interpretation_is_omitted_not_combined(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_matching_interpretation(settings, job_id)
    db.update_job(
        settings.database_path,
        job_id,
        interpretation_version="some-other-draft-v9",
    )
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
    assert details["layers"]["partLabels"]["status"] == "omitted"
    assert "does not match" in details["layers"]["partLabels"]["note"]
    assert details["parts"] == []


# ---------------------------------------------------------------------------
# Staleness, failures, retries and restarts


def test_later_harmony_marks_saved_score_stale_but_keeps_it(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        publish_harmony(settings, job_id)
        job = job_json(client, job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved").json()
        midi = client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi")
        assert job["score"]["stale"] is True
        assert job["score"]["canRebuild"] is True
        assert details["stale"] is True
        assert "rebuild" in details["warnings"][0].lower()
        assert details["layers"]["chordSymbols"]["status"] == "omitted"
        assert midi.status_code == 200
        assert build(client, job_id, force=True).status_code == 202
        rebuilt = client.get(f"/api/jobs/{job_id}/score/saved").json()
    assert rebuilt["stale"] is False
    assert rebuilt["layers"]["chordSymbols"]["status"] == "included"


def _failing_processor(message: str):
    def processor(*args, **kwargs):
        raise ScorePipelineError(message)

    return processor


def test_failed_rebuild_preserves_previous_score_and_sanitizes_error(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    previous = db.get_job(settings.database_path, job_id)
    previous_details = None
    failing = _failing_processor(
        f"Could not read {settings.data_dir / 'secret' / 'file.json'} token=abc123"
    )
    with client_for(settings, score_processor=failing) as client:
        previous_details = client.get(f"/api/jobs/{job_id}/score/saved").json()
        assert build(client, job_id, force=True).status_code == 202
        job = job_json(client, job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved").json()
        download = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    record = db.get_job(settings.database_path, job_id)
    assert record["score_status"] == "failed"
    assert record["score_artifact_file_name"] == previous["score_artifact_file_name"]
    assert record["scored_at"] == previous["scored_at"]
    assert str(settings.data_dir) not in record["score_error"]
    assert "abc123" not in record["score_error"]
    assert job["score"]["available"] is True
    assert job["score"]["canStart"] is True
    assert details == previous_details | {"status": "failed"}
    assert download.status_code == 200
    assert score_files(settings, job_id) == [
        previous["score_artifact_file_name"].split("/")[1]
    ]


def test_unexpected_processor_error_is_generic(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)

    def explode(*args, **kwargs):
        raise RuntimeError("Traceback /home/private/secret")

    with client_for(settings, score_processor=explode) as client:
        assert build(client, job_id).status_code == 202
        job = job_json(client, job_id)
        missing = client.get(f"/api/jobs/{job_id}/score/saved")
    assert job["score"]["status"] == "failed"
    assert job["score"]["error"] == "Unexpected score-construction failure. Check server logs."
    assert job["score"]["available"] is False
    assert missing.status_code == 404


def test_invalid_processor_result_is_rejected_and_cleaned(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)

    def lying(job_id, settings, record, progress, *, attempt_id, expected_fingerprint):
        result = construct_score(
            job_id,
            settings,
            record,
            progress,
            attempt_id=attempt_id,
            expected_fingerprint=expected_fingerprint,
        )
        return ScorePipelineResult(
            **{
                **{field: getattr(result, field) for field in result.__slots__},
                "note_count": result.note_count + 1,
            }
        )

    with client_for(settings, score_processor=lying) as client:
        assert build(client, job_id).status_code == 202
        job = job_json(client, job_id)
    assert job["score"]["status"] == "failed"
    assert job["score"]["error"] == "Score construction returned an invalid result."
    assert score_files(settings, job_id) == []


def test_evidence_change_during_construction_discards_result(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    durable = db.get_job(settings.database_path, job_id)["score_artifact_file_name"]

    def racing(job_id, settings, record, progress, **kwargs):
        result = construct_score(job_id, settings, record, progress, **kwargs)
        publish_harmony(settings, job_id)  # evidence changes before completion
        return result

    with client_for(settings, score_processor=racing) as client:
        assert build(client, job_id, force=True).status_code == 202
        job = job_json(client, job_id)
    record = db.get_job(settings.database_path, job_id)
    assert record["score_status"] == "failed"
    assert "changed" in record["score_error"]
    assert record["score_artifact_file_name"] == durable
    assert job["score"]["stale"] is True
    assert score_files(settings, job_id) == [durable.split("/")[1]]


def test_duplicate_and_superseded_workers_cannot_publish(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    settings.ensure_directories()
    calls: list[str] = []

    def counting(job_id, settings, record, progress, **kwargs):
        calls.append(kwargs["attempt_id"])
        return construct_score(job_id, settings, record, progress, **kwargs)

    first = db.claim_score_attempt(settings.database_path, job_id, score_version="score-pipeline-v1")
    assert first is not None
    assert db.claim_score_attempt(settings.database_path, job_id, score_version="score-pipeline-v1") is None
    db.fail_incomplete_jobs(settings.database_path)  # simulated restart
    second = db.claim_score_attempt(settings.database_path, job_id, score_version="score-pipeline-v1")
    assert second is not None and second != first

    _run_score_job(job_id, settings, counting, first)  # superseded: never starts
    assert calls == []
    _run_score_job(job_id, settings, counting, second)
    _run_score_job(job_id, settings, counting, second)  # duplicate: no second run
    assert calls == [second]
    record = db.get_job(settings.database_path, job_id)
    assert record["score_status"] == "completed"
    assert record["score_artifact_file_name"] == f"score/score-document.{second}.json"
    assert not db.fail_score_attempt(
        settings.database_path, job_id, attempt_id=first, error="late failure"
    )
    assert db.get_job(settings.database_path, job_id)["score_status"] == "completed"


def test_restart_during_rebuild_keeps_previous_score(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    durable = db.get_job(settings.database_path, job_id)["score_artifact_file_name"]
    attempt = db.claim_score_attempt(
        settings.database_path, job_id, score_version="score-pipeline-v1", force=True
    )
    assert attempt
    with client_for(settings) as client:  # lifespan runs restart recovery
        job = job_json(client, job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved")
    assert job["score"]["status"] == "failed"
    assert "restart" in job["score"]["error"]
    assert job["score"]["available"] is True
    assert details.status_code == 200
    assert db.get_job(settings.database_path, job_id)["score_artifact_file_name"] == durable


def test_worker_start_removes_orphaned_attempt_files_only(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    durable = db.get_job(settings.database_path, job_id)["score_artifact_file_name"]
    score_dir = settings.exports_dir / job_id / "score"
    orphan = score_dir / f"score-document.{'a' * 32}.json"
    orphan.write_text("{}", encoding="utf-8")
    temporary = score_dir / f".score-document.{'b' * 32}.{'c' * 32}.tmp"
    temporary.write_text("partial", encoding="utf-8")
    unrelated = score_dir / "notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    removed = remove_unowned_score_artifacts(
        job_id,
        settings,
        lambda: db.score_cleanup_lease(settings.database_path, job_id),
    )
    assert removed == 2
    assert sorted(score_files(settings, job_id)) == sorted(
        [durable.split("/")[1], "notes.txt"]
    )


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_cleanup_never_follows_a_linked_attempt_file(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    target = tmp_path / "outside.json"
    target.write_text("outside", encoding="utf-8")
    link = settings.exports_dir / job_id / "score" / f"score-document.{'d' * 32}.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted on this host")
    remove_unowned_score_artifacts(
        job_id,
        settings,
        lambda: db.score_cleanup_lease(settings.database_path, job_id),
    )
    assert not os.path.lexists(link)
    assert target.read_text(encoding="utf-8") == "outside"


# ---------------------------------------------------------------------------
# Confined reads and downloads


def _saved(settings, client, job_id):
    assert build(client, job_id).status_code == 202
    record = db.get_job(settings.database_path, job_id)
    leaf = record["score_artifact_file_name"].split("/")[1]
    return settings.exports_dir / job_id / "score" / leaf


def test_tampered_or_foreign_saved_score_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    other_id = create_job(settings)
    with client_for(settings) as client:
        path = _saved(settings, client, job_id)
        other_path = _saved(settings, client, other_id)
        original = path.read_bytes()

        document = json.loads(original)
        document["counts"]["notes"] += 1
        path.write_text(json.dumps(document), encoding="utf-8")
        assert client.get(f"/api/jobs/{job_id}/score/saved").status_code == 500

        path.write_bytes(original.replace(b'"tempoBpm":', b'"tempoBpm":NaN,"x":'))
        assert client.get(f"/api/jobs/{job_id}/score/saved").status_code == 500

        path.write_bytes(other_path.read_bytes())
        response = client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi")
        assert response.status_code == 500
        assert str(settings.data_dir) not in response.text

        path.unlink()
        assert client.get(f"/api/jobs/{job_id}/score/saved").status_code == 404


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlinks unavailable")
def test_linked_saved_score_or_directory_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        path = _saved(settings, client, job_id)
        outside = tmp_path / "outside.json"
        outside.write_bytes(path.read_bytes())
        path.unlink()
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("symlink creation is not permitted on this host")
        assert client.get(f"/api/jobs/{job_id}/score/saved").status_code == 500
        path.unlink()
        score_dir = path.parent
        moved = tmp_path / "moved-score"
        path.parent.rename(moved)
        (moved / path.name).write_bytes(outside.read_bytes())
        score_dir.symlink_to(moved, target_is_directory=True)
        assert client.get(f"/api/jobs/{job_id}/score/saved").status_code == 500
        with pytest.raises(ScoreArtifactError):
            load_score_artifact(
                job_id,
                settings,
                artifact_file_name=f"score/{path.name}",
            )


def test_download_format_and_availability_are_validated(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert client.get(f"/api/jobs/{job_id}/score/saved/download?format=pdf").status_code == 422
        assert client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi").status_code == 404
        assert client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=1").status_code == 422
        assert client.get(f"/api/jobs/{'e' * 32}/score/saved").status_code == 404


def test_chord_symbol_needs_a_dominant_uncontested_candidate() -> None:
    from app.score_pipeline import _map_harmony

    def segment(segment_id, start, end, symbol=None, confidence=0.7):
        return {
            "id": segment_id,
            "rawStartSeconds": start,
            "rawEndSeconds": end,
            "unresolved": symbol is None,
            "primaryCandidate": None
            if symbol is None
            else {"symbol": symbol, "confidence": confidence},
        }

    measures = [
        {"measureIndex": index, "startSeconds": index * 4.0, "endSeconds": (index + 1) * 4.0}
        for index in range(3)
    ]
    stats = _map_harmony(
        measures,
        [
            segment("seg_c", 0.0, 3.0, "C"),
            segment("seg_g", 3.0, 4.0, "G"),  # 25% competitor blocks bar 1
            segment("seg_u", 4.0, 8.0),  # unresolved bar 2
            segment("seg_am", 8.0, 10.5, "Am"),  # 62.5% uncontested in bar 3
            segment("seg_late", 13.0, 14.0, "F"),  # after the last bar
        ],
        seconds_per_beat=1.0,
        beats_per_measure=4,
    )
    assert [measure["chordSymbol"] for measure in measures] == [None, None, "Am"]
    assert [entry["symbol"] for entry in measures[0]["harmony"]] == ["C", "G"]
    assert measures[0]["harmony"][1]["startBeat"] == 3.0
    assert measures[1]["harmony"] == [
        {
            "segmentId": "seg_u",
            "startBeat": 0.0,
            "endBeat": 4.0,
            "symbol": None,
            "confidence": None,
            "unresolved": True,
        }
    ]
    assert stats == {
        "windows": 4,
        "unresolved": 1,
        "ambiguous": 1,
        "beyond": 1,
        "truncated": 0,
    }



def test_evidence_change_before_worker_start_fails_instead_of_sticking(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    settings.ensure_directories()
    attempt = db.claim_score_attempt(
        settings.database_path, job_id, score_version="score-pipeline-v1"
    )
    publish_harmony(settings, job_id)  # harmony completes before the worker runs
    calls = []
    _run_score_job(job_id, settings, lambda *a, **k: calls.append(1), attempt)
    record = db.get_job(settings.database_path, job_id)
    assert calls == []
    assert record["score_status"] == "failed"
    assert record["score_attempt_id"] is None
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        assert job_json(client, job_id)["score"]["status"] == "completed"


def test_database_error_at_completion_fails_attempt_and_keeps_previous(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
    durable = db.get_job(settings.database_path, job_id)["score_artifact_file_name"]

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(db, "complete_score_attempt", locked)
    with client_for(settings) as client:
        assert build(client, job_id, force=True).status_code == 202
        job = job_json(client, job_id)
    record = db.get_job(settings.database_path, job_id)
    assert record["score_status"] == "failed"
    assert record["score_attempt_id"] is None
    assert record["score_error"] == "The draft score could not be saved safely."
    assert record["score_artifact_file_name"] == durable
    assert job["score"]["available"] is True
    assert score_files(settings, job_id) == [durable.split("/")[1]]


def test_harmony_with_changed_raw_evidence_is_omitted(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    changed = raw_payload()
    changed["pitchedNoteEvents"][0]["midiPitch"] = 60.4  # same IDs, new pitch
    write_raw_transcription(job_id, settings, changed)
    with client_for(settings) as client:
        assert build(client, job_id).status_code == 202
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
    assert details["layers"]["chordSymbols"]["status"] == "omitted"
    assert "does not match" in details["layers"]["chordSymbols"]["note"]
    assert all(measure["chordSymbol"] is None for measure in details["measures"])


def test_overlong_chord_symbol_is_not_placed_and_score_still_saves() -> None:
    from app.score_pipeline import _map_harmony

    measures = [{"measureIndex": 0, "startSeconds": 0.0, "endSeconds": 4.0}]
    stats = _map_harmony(
        measures,
        [
            {
                "id": "seg_long",
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 4.0,
                "unresolved": False,
                "primaryCandidate": {"symbol": "C" * 200, "confidence": 0.9},
            }
        ],
        seconds_per_beat=1.0,
        beats_per_measure=4,
    )
    assert measures[0]["chordSymbol"] is None
    assert measures[0]["harmony"][0]["unresolved"] is True
    assert measures[0]["harmony"][0]["symbol"] is None
    assert stats["unresolved"] == 1
