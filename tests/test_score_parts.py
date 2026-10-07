"""Instrument parts by source line and per-part exports (Cycle 13)."""

from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.score_artifacts import (
    ScoreArtifactValidationError,
    load_score_artifact,
    validate_score_artifact,
)
from app.score_construction import (
    ScoreConstructionError,
    build_score_document,
    plan_score_parts,
    score_to_musicxml_text,
)
from app.score_corrections import apply_corrections, new_operation
from app.score_pipeline import score_outdated_reason
from app.transcription_events import write_raw_transcription
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings, raw_payload
from test_score_percussion import drum_raw_payload


def note(note_id: str, midi: int, source: str, start: float = 0.0) -> dict:
    return {"id": note_id, "midiNote": midi, "sourceKind": source, "quantizedBeat": start}


@pytest.mark.parametrize(
    "notes,expected",
    [
        ([note("a", 72, "vocals"), note("b", 67, "vocals")], ("lead-vocal", "treble")),
        ([note("a", 52, "vocals"), note("b", 55, "vocals")], ("lead-vocal", "treble-8vb")),
        ([note("a", 43, "vocals")], ("lead-vocal", "bass")),
        ([note("a", 48, "other"), note("b", 64, "other")], ("accompaniment", "grand")),
        ([note("a", 64, "other")], ("accompaniment", "treble")),
        ([note("a", 33, "bass"), note("b", 40, "bass")], ("bass-line", "bass-8vb")),
        ([note("a", 50, "bass")], ("bass-line", "bass")),
        ([note("a", 40, "full_mix"), note("b", 60, "full_mix"), note("c", 76, "full_mix")], ("pitched-lines", "grand")),
        ([note("a", 50, "full_mix"), note("b", 52, "full_mix")], ("pitched-lines", "bass")),
        ([note("a", 70, "synth_pad")], ("other-lines", "treble")),
    ],
)
def test_parts_and_clefs_follow_source_lines_and_range(notes, expected) -> None:
    plan = plan_score_parts(notes)
    assert [(item["id"], item["clef"]) for item in plan] == [expected]
    assert plan[0]["noteCount"] == len(notes)


def test_part_order_is_fixed_and_names_are_honest() -> None:
    plan = plan_score_parts(
        [note("b", 40, "bass"), note("o", 64, "other"), note("v", 70, "vocals"), note("f", 72, "full_mix")]
    )
    assert [item["id"] for item in plan] == ["lead-vocal", "pitched-lines", "accompaniment", "bass-line"]
    assert dict((item["id"], item["name"]) for item in plan)["accompaniment"] == "Accompaniment reduction"


def stem_events() -> list[dict]:
    events = []
    for index, (midi, source) in enumerate(
        [(72, "vocals"), (74, "vocals"), (48, "other"), (64, "other"), (67, "other"),
         (36, "bass"), (43, "bass"), (60, "full_mix")]
    ):
        events.append(
            {"id": f"e_{index}", "sourceKind": source, "startSeconds": (index % 4) * 0.5,
             "endSeconds": (index % 4) * 0.5 + 0.45, "midiNote": midi, "midiPitch": float(midi),
             "frequencyHz": 440.0 * 2 ** ((midi - 69) / 12), "noteName": "X", "confidence": 0.85,
             "warnings": []}
        )
    return events


def create_parts_job(settings) -> str:
    job_id = create_job(settings)
    payload = raw_payload()
    drums = drum_raw_payload()
    payload["pitchedNoteEvents"] = stem_events()
    payload["percussionEvents"] = drums["percussionEvents"]
    payload["alignmentCandidates"] = [
        {"eventId": e["id"], "eventType": "pitched", "rawTimeSeconds": e["startSeconds"], "confidence": 0.0}
        for e in payload["pitchedNoteEvents"]
    ] + [c for c in drums["alignmentCandidates"] if c["eventType"] == "percussion"]
    write_raw_transcription(job_id, settings, payload)
    db.update_job(
        settings.database_path, job_id,
        pitched_event_count=len(payload["pitchedNoteEvents"]),
        percussion_event_count=len(payload["percussionEvents"]),
    )
    return job_id


def built(tmp_path: Path) -> tuple:
    settings = make_settings(tmp_path)
    job_id = create_parts_job(settings)
    client = TestClient(create_app(settings))
    client.__enter__()
    assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
    return settings, job_id, client


def test_score_exports_one_part_per_source_line(tmp_path: Path) -> None:
    settings, job_id, client = built(tmp_path)
    try:
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        full = ET.fromstring(client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml").content)
        downloads = {
            part["id"]: client.get(part["downloadUrl"]) for part in details["exportParts"]
        }
        missing = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml&part=guitar-tab")
        unknown = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml&part=../x")
        midi_part = client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi&part=drums")
    finally:
        client.__exit__(None, None, None)

    assert details["version"] == "score-pipeline-v5"
    assert [item["id"] for item in details["scoreParts"]] == ["lead-vocal", "pitched-lines", "accompaniment"]
    assert {part["id"] for part in details["exportParts"]} == {
        "lead-vocal", "pitched-lines", "accompaniment", "bass-tab", "drums"
    }
    assert any("accompaniment reduction" in warning.lower() for warning in details["warnings"])
    names = [part.findtext("part-name") for part in full.findall("part-list/score-part")]
    assert names[:3] == ["Lead vocal (draft)", "Pitched lines (full mix, draft)", "Accompaniment reduction"]
    accompaniment = full.find("part[@id='P6']")
    assert accompaniment.findtext("measure/attributes/staves") == "2"
    staffs = {n.findtext("staff") for n in accompaniment.iter("note")}
    assert staffs == {"1", "2"}
    assert full.find("part[@id='P1']//metronome") is not None  # directions on the top part
    assert full.find("part[@id='P6']//metronome") is None
    # Every pitched note appears exactly once outside TAB staves (no ties here).
    pitched_parts = [full.find(f"part[@id='{xml_id}']") for xml_id in ("P1", "P5", "P6")]
    on_staves = sum(len([n for n in part.iter("note") if n.find("pitch") is not None]) for part in pitched_parts)
    bass_standard = [n for n in full.find("part[@id='P3']").iter("note") if n.findtext("staff") == "1"]
    assert on_staves + len(bass_standard) == details["counts"]["notes"]

    for part_id, response in downloads.items():
        assert response.status_code == 200, part_id
        assert f'filename="draft-score-{part_id}.musicxml"' in response.headers["content-disposition"]
        single = ET.fromstring(response.content)
        assert len(single.findall("part")) == 1
        assert single.find(".//metronome") is not None
    assert missing.status_code == 404 and unknown.status_code == 404 and midi_part.status_code == 404


def test_validator_and_corrections_keep_parts_consistent(tmp_path: Path) -> None:
    settings, job_id, client = built(tmp_path)
    client.__exit__(None, None, None)
    record = db.get_job(settings.database_path, job_id)
    document = load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])
    tampered = copy.deepcopy(document)
    tampered["scoreParts"][0]["noteCount"] += 1
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(tampered)
    renamed = copy.deepcopy(document)
    renamed["scoreParts"][0]["name"] = "Lead guitar"
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(renamed)

    corrected, _ = apply_corrections(
        document,
        [new_operation({"op": "delete_note", "target": {"noteId": "e_0"}}),
         new_operation({"op": "delete_note", "target": {"noteId": "e_1"}})],
    )
    validate_score_artifact(corrected)
    assert [item["id"] for item in corrected["scoreParts"]] == ["pitched-lines", "accompaniment"]


def test_older_documents_keep_one_combined_part(tmp_path: Path) -> None:
    document = build_score_document(
        [{"id": "v", "startSeconds": 0, "endSeconds": 1, "midiNote": 72, "confidence": 0.9, "sourceKind": "vocals"},
         {"id": "o", "startSeconds": 0, "endSeconds": 1, "midiNote": 48, "confidence": 0.9, "sourceKind": "other"}],
        tempo_bpm=120.0,
    )
    root = ET.fromstring(score_to_musicxml_text(document))
    assert [p.findtext("part-name") for p in root.findall("part-list/score-part")] == ["Draft Pitched Events"]
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(document, only_part="lead-vocal")
    assert score_to_musicxml_text(document, only_part="combined")


def test_scores_from_before_parts_are_out_of_date(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        client.post(f"/api/jobs/{job_id}/score/construct")
        db.update_job(settings.database_path, job_id, score_version="score-pipeline-v4")
        summary = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)["score"]
    assert summary["stale"] is True and summary["staleReason"] == "parts"
    assert score_outdated_reason({"score_version": "score-pipeline-v5", "score_note_count": 3}) is None


def test_parts_list_renders_with_downloads() -> None:
    detail = {
        "available": True, "createdAt": SUMMARY["createdAt"], "layers": {}, "counts": {}, "warnings": [],
        "parts": [],
        "exportParts": [
            {"id": "lead-vocal", "name": "Lead vocal (draft)", "kind": "pitched", "clef": "treble-8vb",
             "noteCount": 12, "downloadUrl": "/api/jobs/done/score/saved/download?format=musicxml&part=lead-vocal"},
            {"id": "accompaniment", "name": "Accompaniment reduction", "kind": "pitched", "clef": "grand",
             "noteCount": 40, "downloadUrl": "/api/jobs/done/score/saved/download?format=musicxml&part=accompaniment"},
            {"id": "drums", "name": "Drum kit", "kind": "percussion", "clef": "percussion", "noteCount": 1,
             "downloadUrl": "https://evil.example/x"},
        ],
    }
    result = _run_node(
        f"""
t.setDetail("done", {json.dumps(detail)});
const html=t.renderScore({{id:"done",score:{json.dumps(SUMMARY)}}});
const stale=t.renderScore({{id:"s",score:{json.dumps({**SUMMARY, "stale": True, "staleReason": "parts"})}}});
console.log(JSON.stringify({{html,stale}}));
"""
    )
    html = result["html"]
    assert "<strong>Parts</strong>" in html
    assert "<strong>Lead vocal (draft)</strong> <span class=\"detail-note\">12 notes · treble clef, sounding an octave lower</span>" in html
    assert 'aria-label="Download Accompaniment reduction as MusicXML"' in html
    assert "not a specific instrument" in html
    assert "1 hit · percussion staff" in html and "evil.example" not in html
    assert "built before separate instrument parts" in result["stale"]


_XSD = os.environ.get("POPEX_MUSICXML_XSD")


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_part_exports_validate_against_musicxml_schema() -> None:
    from lxml import etree  # required whenever the schema path is configured

    schema = etree.XMLSchema(etree.parse(_XSD))
    document = build_score_document(
        [{"id": f"n{i}", "startSeconds": i * 0.4, "endSeconds": i * 0.4 + 1.1, "midiNote": midi,
          "confidence": 0.9, "sourceKind": source}
         for i, (midi, source) in enumerate(
             [(72, "vocals"), (48, "other"), (64, "other"), (36, "bass"), (40, "full_mix"), (79, "full_mix")]
         )],
        tempo_bpm=110.0,
    )
    notes = [n for m in document["measures"] for n in m["notes"]]
    document["scoreParts"] = plan_score_parts(notes)
    for only in (None, "lead-vocal", "accompaniment", "bass-line", "pitched-lines"):
        tree = etree.fromstring(score_to_musicxml_text(document, only_part=only).encode("utf-8"))
        assert schema.validate(tree), (only, schema.error_log)
