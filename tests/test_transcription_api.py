from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.main import create_app
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    validate_raw_transcription,
    write_raw_transcription,
)
from app.transcription_pipeline import (
    TRANSCRIPTION_VERSION,
    TranscriptionPipelineError,
    TranscriptionPipelineResult,
)


CREATED_AT = "2026-08-06T04:00:00+00:00"


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


def artifact_payload() -> dict:
    return {
        "schemaVersion": 1,
        "transcriptionVersion": TRANSCRIPTION_VERSION,
        "createdAt": CREATED_AT,
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": "baseline-librosa-v1",
        },
        "algorithms": {
            "pitchTracking": {"version": "baseline-pyin-v1"},
            "percussionDetection": {"version": "baseline-onset-bands-v1"},
            "eventAlignment": {"version": "advisory-beat-grid-v1"},
            "transcriptionPipeline": {"version": TRANSCRIPTION_VERSION},
        },
        "pitchedNoteEvents": [
            {
                "id": "p000001",
                "sourceKind": "full_mix",
                "startSeconds": 0.25,
                "endSeconds": 0.75,
                "midiNote": 69,
                "midiPitch": 69.1,
                "frequencyHz": 442.55,
                "noteName": "A4",
                "confidence": 0.9,
            }
        ],
        "percussionEvents": [
            {
                "id": "r000001",
                "sourceKind": "full_mix",
                "timeSeconds": 0.5,
                "strength": 0.8,
                "hits": [
                    {"kind": "unknown_percussion", "confidence": 0.55}
                ],
                "rawFeatureSummary": {"spectralCentroidHz": 1500.0},
            }
        ],
        "alignmentCandidates": [
            {
                "eventId": "p000001",
                "eventType": "pitched",
                "rawTimeSeconds": 0.25,
                "beatIndex": 0,
                "subdivision": 4,
                "subdivisionIndex": 2,
                "alignedTimeSeconds": 0.25,
                "offsetSeconds": 0.0,
                "confidence": 0.95,
                "measureIndex": 0,
                "beatInMeasure": 1,
            },
            {
                "eventId": "r000001",
                "eventType": "percussion",
                "rawTimeSeconds": 0.5,
                "confidence": 0.0,
                "warnings": ["No safe local alignment candidate was accepted."],
            },
        ],
        "warnings": ["Raw events remain candidates for review."],
    }


def create_job(
    settings: Settings,
    *,
    analyzed: bool = True,
    transcription_status: str = "not_started",
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
    (job_dir / "analysis.wav").write_bytes(b"RIFF synthetic")
    fields = {
        "status": "completed" if analyzed else "processing",
        "stage": "completed" if analyzed else "importing",
        "progress": 100 if analyzed else 50,
        "preparation_status": "completed",
        "normalized_file_name": "analysis.wav",
        "analysis_status": "completed" if analyzed else "processing",
        "analysis_version": "baseline-librosa-v1" if analyzed else None,
        "analysis_json_file_name": (
            ANALYSIS_JSON_RELATIVE_PATH if analyzed else None
        ),
        "transcription_status": transcription_status,
        "transcription_stage": transcription_status,
    }
    db.update_job(settings.database_path, job_id, **fields)
    return job_id


def publish_previous_result(settings: Settings, job_id: str) -> dict:
    payload = validate_raw_transcription(artifact_payload())
    write_raw_transcription(job_id, settings, payload)
    db.update_job(
        settings.database_path,
        job_id,
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Raw transcription complete.",
        transcription_version=payload["transcriptionVersion"],
        transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        transcribed_at=payload["createdAt"],
        pitched_event_count=1,
        percussion_event_count=1,
        aligned_event_count=1,
        transcription_error=None,
    )
    return payload


def successful_processor(job_id, settings, stage_callback):
    stage_callback(
        "selecting_transcription_inputs",
        "Selecting safe transcription audio.",
        5,
    )
    stage_callback(
        "detecting_pitched_events",
        "Detecting raw pitched-note candidates.",
        25,
    )
    stage_callback(
        "detecting_percussion_events",
        "Detecting raw percussion candidates.",
        55,
    )
    stage_callback(
        "aligning_transcription_events",
        "Aligning raw events to saved timing evidence.",
        75,
    )
    payload = validate_raw_transcription(artifact_payload())
    write_raw_transcription(job_id, settings, payload)
    return TranscriptionPipelineResult(
        transcription_version=TRANSCRIPTION_VERSION,
        artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        transcribed_at=CREATED_AT,
        pitched_event_count=1,
        percussion_event_count=1,
        aligned_event_count=1,
        input_mode="full_mix",
        warnings=tuple(payload["warnings"]),
        payload=payload,
    )


def expected_failure(job_id, settings, stage_callback):
    stage_callback(
        "detecting_pitched_events",
        "Detecting raw pitched-note candidates.",
        35,
    )
    raise TranscriptionPipelineError(
        f"Detector failed near {settings.data_dir / 'private' / 'audio.wav'}; "
        "token=super-secret https://private.invalid/log"
    )


def unexpected_failure(job_id, settings, stage_callback):
    stage_callback("detecting_percussion_events", "Detecting percussion.", 60)
    raise RuntimeError(f"private failure at {settings.data_dir / 'secret.txt'}")


def test_contract_is_absent_before_analysis_and_live_after_completion(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings, analyzed=False)
        before = client.get(f"/api/jobs/{job_id}").json()
        assert "transcription" not in before
        db.update_job(
            settings.database_path,
            job_id,
            status="completed",
            stage="completed",
            progress=100,
            analysis_status="completed",
            analysis_version="baseline-librosa-v1",
            analysis_json_file_name=ANALYSIS_JSON_RELATIVE_PATH,
        )
        after = client.get(f"/api/jobs/{job_id}").json()

    contract = after["transcription"]
    assert contract["status"] == "not_started"
    assert contract["canStart"] is True
    assert contract["available"] is False
    assert contract["startUrl"] == f"/api/jobs/{job_id}/transcribe"
    assert contract["detailsUrl"].endswith("?includeEvents=false")
    assert contract["downloadUrl"] is None
    for internal in (
        "transcription_status",
        "transcription_stage",
        "transcription_artifact_file_name",
        "pitched_event_count",
    ):
        assert internal not in after


def test_full_mix_start_succeeds_without_working_separation(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        db.update_job(
            settings.database_path,
            job_id,
            separation_status="failed",
            separation_stage="failed",
            separation_error="Optional separation unavailable.",
        )
        response = client.post(f"/api/jobs/{job_id}/transcribe")
        payload = client.get(f"/api/jobs/{job_id}").json()

    assert response.status_code == 202
    transcription = payload["transcription"]
    assert transcription["status"] == "completed"
    assert transcription["available"] is True
    assert transcription["counts"] == {
        "pitched": 1,
        "percussion": 1,
        "aligned": 1,
    }
    assert transcription["downloadUrl"].startswith("/api/jobs/")
    assert "demucs" not in json.dumps(transcription).lower()


def test_start_conflicts_are_safe_and_force_is_strict(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{'f' * 32}/transcribe").status_code == 404
        incomplete = create_job(settings, analyzed=False)
        assert client.post(f"/api/jobs/{incomplete}/transcribe").status_code == 409

        processing = create_job(settings, transcription_status="processing")
        assert client.post(f"/api/jobs/{processing}/transcribe").status_code == 409

        completed = create_job(settings)
        publish_previous_result(settings, completed)
        assert client.post(f"/api/jobs/{completed}/transcribe").status_code == 409
        forced = client.post(f"/api/jobs/{completed}/transcribe?force=true")
        assert forced.status_code == 202
        assert client.post(f"/api/jobs/{completed}/transcribe?force=1").status_code == 422


def test_atomic_claim_conflict_schedules_no_background_task(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    calls = []

    def processor(*args):
        calls.append(args)
        return successful_processor(*args)

    app = create_app(settings=settings, transcription_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        monkeypatch.setattr(db, "claim_transcription_attempt", lambda *a, **k: False)
        response = client.post(f"/api/jobs/{job_id}/transcribe")

    assert response.status_code == 409
    assert calls == []


def test_expected_retry_failure_preserves_previous_artifact_and_sanitizes_error(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=expected_failure)
    with TestClient(app) as client:
        job_id = create_job(settings)
        previous = publish_previous_result(settings, job_id)
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="failed",
            transcription_stage="failed",
            transcription_progress=50,
            transcription_error="Earlier retry failed.",
        )
        response = client.post(f"/api/jobs/{job_id}/transcribe")
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/transcription").json()

    assert response.status_code == 202
    transcription = job["transcription"]
    assert transcription["status"] == "failed"
    assert transcription["available"] is True
    assert transcription["counts"] == {
        "pitched": 1,
        "percussion": 1,
        "aligned": 1,
    }
    serialized = json.dumps(transcription)
    assert str(settings.data_dir) not in serialized
    assert "super-secret" not in serialized
    assert "private.invalid" not in serialized
    assert details["available"] is True
    assert details["version"] == previous["transcriptionVersion"]
    assert details["counts"]["aligned"] == 1


def test_unexpected_failure_is_generic_and_preserves_previous_result(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=unexpected_failure)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_previous_result(settings, job_id)
        db.update_job(settings.database_path, job_id, transcription_status="failed")
        response = client.post(f"/api/jobs/{job_id}/transcribe")
        job = client.get(f"/api/jobs/{job_id}").json()

    assert response.status_code == 202
    transcription = job["transcription"]
    assert transcription["status"] == "failed"
    assert transcription["available"] is True
    assert transcription["error"] == (
        "Unexpected raw transcription failure. Check server logs."
    )
    assert str(settings.data_dir) not in json.dumps(transcription)


def test_details_default_to_summary_and_full_mode_preserves_events(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        expected = publish_previous_result(settings, job_id)
        summary = client.get(f"/api/jobs/{job_id}/transcription")
        explicit_summary = client.get(
            f"/api/jobs/{job_id}/transcription?includeEvents=false"
        )
        full = client.get(
            f"/api/jobs/{job_id}/transcription?includeEvents=true"
        )
        invalid = client.get(
            f"/api/jobs/{job_id}/transcription/includeEvents=1"
        )

    assert summary.status_code == explicit_summary.status_code == full.status_code == 200
    for payload in (summary.json(), explicit_summary.json()):
        assert payload["available"] is True
        assert payload["counts"] == {
            "pitched": 1,
            "percussion": 1,
            "aligned": 1,
        }
        assert "pitchedNoteEvents" not in payload
        assert "percussionEvents" not in payload
        assert "alignmentCandidates" not in payload
    complete = full.json()
    assert complete["pitchedNoteEvents"] == expected["pitchedNoteEvents"]
    assert complete["percussionEvents"] == expected["percussionEvents"]
    assert complete["alignmentCandidates"] == expected["alignmentCandidates"]
    assert invalid.status_code == 422



def test_missing_pointer_returns_unavailable_summary(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.get(f"/api/jobs/{job_id}/transcription")

    assert response.status_code == 200
    assert response.json()["available"] is False
    assert response.json()["counts"] == {
        "pitched": 0,
        "percussion": 0,
        "aligned": 0,
    }


def test_canonical_download_and_bounded_artifact_failures(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        expected = publish_previous_result(settings, job_id)
        download = client.get(f"/api/jobs/{job_id}/transcription/download")
        assert download.status_code == 200
        assert download.json() == expected
        assert "raw-transcription.json" in download.headers["content-disposition"]

        missing = create_job(settings)
        assert (
            client.get(f"/api/jobs/{missing}/transcription/download").status_code
            == 404
        )

        corrupt = create_job(settings)
        path = settings.exports_dir / corrupt / RAW_TRANSCRIPTION_RELATIVE_PATH
        path.parent.mkdir(parents=True)
        path.write_text("{not-json", encoding="utf-8")
        db.update_job(
            settings.database_path,
            corrupt,
            transcription_status="completed",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        )
        failure = client.get(f"/api/jobs/{corrupt}/transcription/download")

    assert failure.status_code == 500
    assert str(settings.data_dir) not in failure.text
    assert "not-json" not in failure.text


def test_symlinked_download_is_rejected_without_reading_outside(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, transcription_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        directory = settings.exports_dir / job_id / "transcription"
        directory.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps(artifact_payload()), encoding="utf-8")
        target = directory / "raw-events.json"
        try:
            target.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        )
        response = client.get(f"/api/jobs/{job_id}/transcription/download")

    assert response.status_code == 500
    assert str(outside) not in response.text
    assert outside.read_text(encoding="utf-8") == json.dumps(artifact_payload())


def test_invalid_processor_result_fails_without_publishing_pointer(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    def invalid_processor(job_id, settings, stage_callback):
        return TranscriptionPipelineResult(
            transcription_version=TRANSCRIPTION_VERSION,
            artifact_file_name="../outside.json",
            transcribed_at=CREATED_AT,
            pitched_event_count=1,
            percussion_event_count=1,
            aligned_event_count=1,
            input_mode="full_mix",
            warnings=(),
            payload={},
        )

    app = create_app(settings=settings, transcription_processor=invalid_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.post(f"/api/jobs/{job_id}/transcribe")
        payload = client.get(f"/api/jobs/{job_id}").json()["transcription"]

    assert response.status_code == 202
    assert payload["status"] == "failed"
    assert payload["available"] is False
    assert payload["downloadUrl"] is None
    assert payload["error"] == "Raw transcription returned an invalid result."
