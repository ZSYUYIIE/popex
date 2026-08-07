from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    InterpretationPipelineError,
    InterpretationPipelineResult,
)
from app.main import create_app
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    write_transcription_draft,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    write_raw_transcription,
)


CREATED_AT = "2026-08-07T10:30:00+00:00"


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("example.invalid",),
        max_duration_seconds=60,
        max_filesize_mb=16,
        max_upload_mb=16,
        audio_quality="192",
        ffmpeg_binary="missing-test-ffmpeg",
        ffprobe_binary="missing-test-ffprobe",
        audio_analysis_enabled=True,
    )


def raw_payload() -> dict:
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "raw-transcription-v1",
        "createdAt": "2026-08-07T10:00:00+00:00",
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": "baseline-librosa-v1",
        },
        "algorithms": {"testRaw": {"version": "raw-transcription-v1"}},
        "pitchedNoteEvents": [
            {
                "id": "p1",
                "sourceKind": "full_mix",
                "startSeconds": 0.0,
                "endSeconds": 0.5,
                "midiNote": 69,
                "midiPitch": 69.2,
                "frequencyHz": 445.1,
                "noteName": "A4",
                "confidence": 0.9,
            }
        ],
        "percussionEvents": [],
        "alignmentCandidates": [
            {
                "eventId": "p1",
                "eventType": "pitched",
                "rawTimeSeconds": 0.0,
                "confidence": 0.0,
            }
        ],
        "warnings": ["Raw candidate remains editable."],
    }


def draft_payload() -> dict:
    return {
        "schemaVersion": 1,
        "draftVersion": INTERPRETATION_PIPELINE_VERSION,
        "createdAt": CREATED_AT,
        "sourceTranscription": {
            "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
            "schemaVersion": 1,
            "transcriptionVersion": "raw-transcription-v1",
            "provenance": {"pipelineVersion": "raw-transcription-v1"},
            "sourceEventIndex": [
                {
                    "id": "p1",
                    "eventType": "pitched",
                    "sourceKind": "full_mix",
                    "rawStartSeconds": 0.0,
                    "rawEndSeconds": 0.5,
                    "confidence": 0.9,
                    "midiPitch": 69.2,
                }
            ],
        },
        "algorithms": {
            "pitchedPartInference": {"version": "source-phrase-v1"},
            "percussionInterpretation": {"version": "broad-drum-structure-v1"},
            "rhythmInterpretation": {"version": "conservative-grid-v1"},
        },
        "interpretationEvidence": {
            "pitchedPartInference": {
                "version": "source-phrase-v1",
                "parts": [],
                "voices": [],
                "phrases": [],
                "assignments": [{"eventId": "p1", "sourceEventIds": ["p1"]}],
                "unassignedEventIds": [],
                "warnings": [],
                "diagnostics": {},
            },
            "percussionInterpretation": {
                "version": "broad-drum-structure-v1",
                "parts": [],
                "voices": [],
                "groups": [],
                "assignments": [],
                "unresolved_event_ids": [],
                "warnings": [],
                "diagnostics": {},
            },
            "rhythmInterpretation": {
                "version": "conservative-grid-v1",
                "meter_candidates": [],
                "measures": [],
                "event_interpretations": [
                    {"eventId": "p1", "sourceEventIds": ["p1"]}
                ],
                "rest_candidates": [],
                "warnings": [],
                "diagnostics": {"rawTimingAuthoritative": True},
            },
        },
        "parts": [
            {
                "id": "part_full_mix",
                "sourceKind": "full_mix",
                "role": "pitched",
                "instrumentKind": "source_pitched_line",
                "voiceIds": ["voice_full_mix"],
                "sourceEventIds": ["p1"],
                "confidence": 0.9,
            }
        ],
        "voices": [
            {
                "id": "voice_full_mix",
                "partId": "part_full_mix",
                "voiceKind": "monophonic",
                "sourceEventIds": ["p1"],
                "confidence": 0.9,
            }
        ],
        "measures": [],
        "phrases": [
            {
                "id": "phrase_full_mix",
                "partId": "part_full_mix",
                "voiceId": "voice_full_mix",
                "sourceEventIds": ["p1"],
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 0.5,
                "confidence": 0.8,
            }
        ],
        "pitchedItems": [
            {
                "id": "pitched_p1",
                "interpretationType": "note",
                "placementStatus": "unassigned",
                "partId": "part_full_mix",
                "voiceId": "voice_full_mix",
                "sourceEventIds": ["p1"],
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 0.5,
                "sourceKind": "full_mix",
                "pitch": {
                    "midiNote": 69,
                    "midiPitch": 69.2,
                    "frequencyHz": 445.1,
                    "noteName": "A4",
                },
                "confidence": 0.8,
            }
        ],
        "percussionItems": [],
        "alternatives": [],
        "warnings": ["Editable warning."],
    }


def create_job(
    settings: Settings,
    *,
    transcribed: bool = True,
    interpretation_status: str = "not_started",
) -> str:
    job_id = uuid4().hex
    db.create_job(
        settings.database_path,
        job_id,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = settings.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    fields = {
        "status": "completed",
        "stage": "completed",
        "progress": 100,
        "preparation_status": "completed",
        "normalized_file_name": "analysis.wav",
        "analysis_status": "completed",
        "analysis_version": "baseline-librosa-v1",
        "analysis_json_file_name": ANALYSIS_JSON_RELATIVE_PATH,
        "transcription_status": "completed" if transcribed else "not_started",
        "transcription_stage": "completed" if transcribed else "not_started",
        "transcription_progress": 100 if transcribed else 0,
        "transcription_version": "raw-transcription-v1" if transcribed else None,
        "transcription_artifact_file_name": (
            RAW_TRANSCRIPTION_RELATIVE_PATH if transcribed else None
        ),
        "transcribed_at": "2026-08-07T10:00:00+00:00" if transcribed else None,
        "pitched_event_count": 1 if transcribed else None,
        "percussion_event_count": 0 if transcribed else None,
        "aligned_event_count": 0 if transcribed else None,
        "interpretation_status": interpretation_status,
        "interpretation_stage": interpretation_status,
    }
    db.update_job(settings.database_path, job_id, **fields)
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    return job_id


def publish_previous_draft(settings: Settings, job_id: str) -> dict:
    payload = draft_payload()
    write_transcription_draft(job_id, settings, payload)
    db.update_job(
        settings.database_path,
        job_id,
        interpretation_status="completed",
        interpretation_stage="completed",
        interpretation_progress=100,
        interpretation_message="Editable interpretation complete.",
        interpretation_version=INTERPRETATION_PIPELINE_VERSION,
        interpretation_artifact_file_name=INTERPRETATION_DRAFT_RELATIVE_PATH,
        interpreted_at=CREATED_AT,
        interpretation_part_count=1,
        interpretation_phrase_count=1,
        interpretation_pitched_item_count=1,
        interpretation_percussion_item_count=0,
        interpretation_warning_count=1,
        interpretation_error=None,
    )
    return payload


def successful_processor(job_id, settings, stage_callback):
    stage_callback(
        "interpreting_pitched_parts",
        "Grouping pitched candidates.",
        30,
    )
    stage_callback(
        "interpreting_rhythm",
        "Building rhythm hypotheses.",
        70,
    )
    payload = draft_payload()
    write_transcription_draft(job_id, settings, payload)
    return InterpretationPipelineResult(
        version=INTERPRETATION_PIPELINE_VERSION,
        draft_file_name=INTERPRETATION_DRAFT_RELATIVE_PATH,
        created_at=CREATED_AT,
        part_count=1,
        phrase_count=1,
        pitched_item_count=1,
        percussion_item_count=0,
        warning_count=1,
        warnings=("Editable warning.",),
        payload=payload,
    )


def expected_failure(job_id, settings, stage_callback):
    stage_callback("interpreting_rhythm", "Building rhythm hypotheses.", 66)
    raise InterpretationPipelineError(
        f"Interpretation failed near {settings.data_dir / 'private' / 'draft.json'}; "
        "token=super-secret https://private.invalid/log"
    )


def unexpected_failure(job_id, settings, stage_callback):
    stage_callback("assembling_interpretation_draft", "Assembling draft.", 80)
    raise RuntimeError(f"private failure at {settings.data_dir / 'secret.txt'}")


def invalid_processor(job_id, settings, stage_callback):
    payload = publish_previous_draft(settings, job_id)
    return InterpretationPipelineResult(
        version=INTERPRETATION_PIPELINE_VERSION,
        draft_file_name="interpretation/wrong.json",
        created_at=CREATED_AT,
        part_count=1,
        phrase_count=1,
        pitched_item_count=1,
        percussion_item_count=0,
        warning_count=1,
        warnings=("Editable warning.",),
        payload=payload,
    )


def test_contract_appears_only_after_raw_transcription(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings, transcribed=False)
        before = client.get(f"/api/jobs/{job_id}").json()
        assert "interpretation" not in before

        write_raw_transcription(job_id, settings, raw_payload())
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_stage="completed",
            transcription_progress=100,
            transcription_version="raw-transcription-v1",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
            transcribed_at="2026-08-07T10:00:00+00:00",
            pitched_event_count=1,
            percussion_event_count=0,
            aligned_event_count=0,
        )
        after = client.get(f"/api/jobs/{job_id}").json()

    contract = after["interpretation"]
    assert contract["status"] == "not_started"
    assert contract["canStart"] is True
    assert contract["canReinterpret"] is False
    assert contract["available"] is False
    assert contract["startUrl"] == f"/api/jobs/{job_id}/interpret"
    assert contract["detailsUrl"].endswith("?includeItems=false")
    assert contract["fullDetailsUrl"].endswith("?includeItems=true")
    assert contract["downloadUrl"] is None
    for internal in (
        "interpretation_status",
        "interpretation_stage",
        "interpretation_artifact_file_name",
        "interpretation_part_count",
    ):
        assert internal not in after


def test_start_succeeds_without_stems_or_demucs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        db.update_job(
            settings.database_path,
            job_id,
            separation_status="failed",
            separation_stage="failed",
            separation_error="Optional separation unavailable.",
        )
        response = client.post(f"/api/jobs/{job_id}/interpret")
        job = client.get(f"/api/jobs/{job_id}").json()

    assert response.status_code == 202
    interpretation = job["interpretation"]
    assert interpretation["status"] == "completed"
    assert interpretation["available"] is True
    assert interpretation["counts"] == {
        "parts": 1,
        "phrases": 1,
        "pitched": 1,
        "percussion": 0,
        "warnings": 1,
    }
    assert interpretation["canReinterpret"] is True
    assert "demucs" not in json.dumps(interpretation).lower()


def test_start_conflicts_and_force_query_are_strict(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{'f' * 32}/interpret").status_code == 404

        incomplete = create_job(settings, transcribed=False)
        assert client.post(f"/api/jobs/{incomplete}/interpret").status_code == 409

        processing = create_job(settings, interpretation_status="processing")
        assert client.post(f"/api/jobs/{processing}/interpret").status_code == 409

        completed = create_job(settings)
        publish_previous_draft(settings, completed)
        assert client.post(f"/api/jobs/{completed}/interpret").status_code == 409
        forced = client.post(f"/api/jobs/{completed}/interpret?force=true")
        assert forced.status_code == 202
        assert client.post(f"/api/jobs/{completed}/interpret?force=1").status_code == 422


def test_invalid_or_missing_raw_artifact_blocks_start(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        raw_path = settings.exports_dir / job_id / RAW_TRANSCRIPTION_RELATIVE_PATH
        raw_path.unlink()
        missing = client.post(f"/api/jobs/{job_id}/interpret")
        assert missing.status_code == 409

        write_raw_transcription(job_id, settings, raw_payload())
        raw_path.write_text("{broken", encoding="utf-8")
        corrupt = client.post(f"/api/jobs/{job_id}/interpret")
        assert corrupt.status_code == 409


def test_atomic_claim_conflict_schedules_no_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    calls = []

    def processor(*args):
        calls.append(args)
        return successful_processor(*args)

    app = create_app(settings=settings, interpretation_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        monkeypatch.setattr(db, "claim_interpretation_attempt", lambda *a, **k: False)
        response = client.post(f"/api/jobs/{job_id}/interpret")

    assert response.status_code == 409
    assert calls == []


def test_expected_retry_failure_preserves_previous_draft_and_sanitizes_error(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=expected_failure)
    with TestClient(app) as client:
        job_id = create_job(settings)
        previous = publish_previous_draft(settings, job_id)
        db.update_job(
            settings.database_path,
            job_id,
            interpretation_status="failed",
            interpretation_stage="failed",
            interpretation_progress=50,
            interpretation_error="Earlier retry failed.",
        )
        response = client.post(f"/api/jobs/{job_id}/interpret")
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/interpretation").json()

    assert response.status_code == 202
    interpretation = job["interpretation"]
    assert interpretation["status"] == "failed"
    assert interpretation["available"] is True
    assert interpretation["counts"] == {
        "parts": 1,
        "phrases": 1,
        "pitched": 1,
        "percussion": 0,
        "warnings": 1,
    }
    serialized = json.dumps(interpretation)
    assert str(settings.data_dir) not in serialized
    assert "super-secret" not in serialized
    assert "private.invalid" not in serialized
    assert details["available"] is True
    assert details["version"] == previous["draftVersion"]
    assert details["counts"]["parts"] == 1


def test_unexpected_failure_is_generic_and_preserves_previous_draft(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=unexpected_failure)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_previous_draft(settings, job_id)
        db.update_job(settings.database_path, job_id, interpretation_status="failed")
        response = client.post(f"/api/jobs/{job_id}/interpret")
        job = client.get(f"/api/jobs/{job_id}").json()

    assert response.status_code == 202
    interpretation = job["interpretation"]
    assert interpretation["status"] == "failed"
    assert interpretation["available"] is True
    assert interpretation["error"] == (
        "Unexpected editable interpretation failure. Check server logs."
    )


def test_invalid_processor_result_marks_failed_without_erasing_previous_draft(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=invalid_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_previous_draft(settings, job_id)
        db.update_job(settings.database_path, job_id, interpretation_status="failed")
        response = client.post(f"/api/jobs/{job_id}/interpret")
        job = client.get(f"/api/jobs/{job_id}").json()

    assert response.status_code == 202
    assert job["interpretation"]["status"] == "failed"
    assert job["interpretation"]["available"] is True
    assert job["interpretation"]["error"] == (
        "Editable interpretation returned an invalid result."
    )


def test_details_default_to_summary_and_full_mode_preserves_items(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        expected = publish_previous_draft(settings, job_id)
        summary = client.get(f"/api/jobs/{job_id}/interpretation")
        explicit_summary = client.get(
            f"/api/jobs/{job_id}/interpretation?includeItems=false"
        )
        full = client.get(f"/api/jobs/{job_id}/interpretation?includeItems=true")
        invalid = client.get(f"/api/jobs/{job_id}/interpretation?includeItems=1")

    assert summary.status_code == explicit_summary.status_code == full.status_code == 200
    for payload in (summary.json(), explicit_summary.json()):
        assert payload["available"] is True
        assert payload["counts"]["parts"] == 1
        assert "parts" not in payload
        assert "interpretationEvidence" not in payload
    complete = full.json()
    assert complete["parts"] == expected["parts"]
    assert complete["voices"] == expected["voices"]
    assert complete["phrases"] == expected["phrases"]
    assert complete["pitchedItems"] == expected["pitchedItems"]
    assert complete["interpretationEvidence"] == expected["interpretationEvidence"]
    assert invalid.status_code == 422


def test_missing_pointer_returns_unavailable_summary(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.get(f"/api/jobs/{job_id}/interpretation")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["counts"]["parts"] == 0


def test_canonical_download_and_corrupt_artifact_failure(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        expected = publish_previous_draft(settings, job_id)
        download = client.get(f"/api/jobs/{job_id}/interpretation/download")
        assert download.status_code == 200
        assert download.json() == expected
        disposition = download.headers.get("content-disposition", "")
        assert "editable-interpretation.json" in disposition

        path = settings.exports_dir / job_id / INTERPRETATION_DRAFT_RELATIVE_PATH
        stored = json.loads(path.read_text(encoding="utf-8"))
        stored["voices"][0]["partId"] = "missing_part"
        path.write_text(json.dumps(stored), encoding="utf-8")
        assert client.get(f"/api/jobs/{job_id}/interpretation").status_code == 500
        assert client.get(f"/api/jobs/{job_id}/interpretation/download").status_code == 500


def test_download_without_pointer_is_404(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, interpretation_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.get(f"/api/jobs/{job_id}/interpretation/download")
    assert response.status_code == 404
