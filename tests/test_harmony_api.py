from __future__ import annotations

import copy
import json
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    build_harmony_artifact,
    write_harmony_artifact,
)
from app.harmony_inference import infer_harmony
from app.harmony_pipeline import (
    HARMONY_PIPELINE_VERSION,
    HarmonyPipelineError,
    HarmonyPipelineResult,
)
from app.main import create_app
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    write_raw_transcription,
)


RAW_CREATED_AT = "2026-08-14T04:00:00+00:00"
HARMONY_CREATED_AT = "2026-08-14T04:15:00+00:00"
ANALYSIS_VERSION = "baseline-librosa-v1"


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


def analysis_payload() -> dict:
    return {
        "schemaVersion": 1,
        "analysisVersion": ANALYSIS_VERSION,
        "createdAt": "2026-08-14T03:55:00+00:00",
        "sourceAsset": "analysis.wav",
        "libraries": {},
        "audio": {
            "durationSeconds": 2.0,
            "sampleRate": 44100,
            "channels": 1,
            "peakAmplitude": 0.8,
            "rms": 0.2,
            "rmsDbfs": -13.9,
            "silent": False,
        },
        "timing": {
            "tempoBpm": 120.0,
            "tempoConfidence": 0.9,
            "tempoStable": True,
            "beatsSeconds": [0.0, 1.0, 2.0],
            "beatConfidence": 0.9,
            "downbeatsSeconds": [0.0],
            "meter": 4,
            "meterConfidence": 0.8,
        },
        "tonality": {
            "tonalCenter": "C",
            "primaryCandidate": {
                "tonalCenter": "C",
                "collection": "ionian",
                "displayName": "C major",
                "confidence": 0.8,
                "supportedByBaseline": True,
            },
            "candidates": [],
            "localRegions": [],
            "chromaticismScore": None,
            "baselineCollections": ["ionian", "aeolian"],
            "key": "C",
            "mode": "major",
            "symbol": "C major",
            "confidence": 0.8,
            "scoreMargin": 0.1,
            "tuningOffsetCents": 0.0,
            "chromaMean": [0.0] * 12,
            "alternatives": [],
        },
        "warnings": [],
    }


def raw_events() -> list[dict]:
    return [
        {
            "id": "p_c",
            "sourceKind": "full_mix",
            "startSeconds": 0.0,
            "endSeconds": 0.9,
            "midiNote": 60,
            "midiPitch": 60.12,
            "frequencyHz": 264.0,
            "noteName": "C4",
            "confidence": 0.92,
            "warnings": [],
        },
        {
            "id": "p_e",
            "sourceKind": "full_mix",
            "startSeconds": 0.05,
            "endSeconds": 0.9,
            "midiNote": 64,
            "midiPitch": 63.94,
            "frequencyHz": 328.5,
            "noteName": "E4",
            "confidence": 0.9,
            "warnings": [],
        },
        {
            "id": "p_g",
            "sourceKind": "full_mix",
            "startSeconds": 0.1,
            "endSeconds": 0.9,
            "midiNote": 67,
            "midiPitch": 67.04,
            "frequencyHz": 393.0,
            "noteName": "G4",
            "confidence": 0.87,
            "warnings": [],
        },
    ]


def raw_payload() -> dict:
    events = raw_events()
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "raw-transcription-v1",
        "createdAt": RAW_CREATED_AT,
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": ANALYSIS_VERSION,
        },
        "algorithms": {"testRaw": {"version": "raw-transcription-v1"}},
        "pitchedNoteEvents": events,
        "percussionEvents": [],
        "alignmentCandidates": [
            {
                "eventId": event["id"],
                "eventType": "pitched",
                "rawTimeSeconds": event["startSeconds"],
                "confidence": 0.0,
            }
            for event in events
        ],
        "warnings": ["Raw candidates remain editable."],
    }


def harmony_inference_result():
    return infer_harmony(
        raw_events(),
        {
            "beatsSeconds": [0.0, 1.0, 2.0],
            "beatConfidence": 0.9,
        },
        {
            "primaryCandidate": {
                "tonalCenter": "C",
                "collection": "ionian",
                "displayName": "C major",
                "confidence": 0.8,
            }
        },
    )


def harmony_artifact(*, created_at: str = HARMONY_CREATED_AT) -> dict:
    return build_harmony_artifact(
        harmony_inference_result(),
        harmony_version=HARMONY_PIPELINE_VERSION,
        created_at=created_at,
        transcription_version="raw-transcription-v1",
        analysis_version=ANALYSIS_VERSION,
    )


def result_from_artifact(payload: dict) -> HarmonyPipelineResult:
    diagnostics = payload["diagnostics"]
    warnings = tuple(payload["warnings"])
    return HarmonyPipelineResult(
        version=payload["harmonyVersion"],
        artifact_file_name=HARMONY_ARTIFACT_RELATIVE_PATH,
        created_at=payload["createdAt"],
        event_count=diagnostics["eventCount"],
        segment_count=diagnostics["segmentCount"],
        resolved_segment_count=diagnostics["resolvedSegmentCount"],
        unresolved_segment_count=diagnostics["unresolvedSegmentCount"],
        unresolved_event_count=diagnostics["unresolvedEventCount"],
        used_interpretation_context=False,
        interpretation_version=None,
        warning_count=len(warnings),
        warnings=warnings,
        payload=copy.deepcopy(payload),
    )


def create_job(
    settings: Settings,
    *,
    transcribed: bool = True,
    harmony_status: str = "not_started",
    separation_status: str = "failed",
    interpretation_status: str = "not_started",
) -> str:
    settings.ensure_directories()
    db.init_database(settings.database_path)
    job_id = uuid4().hex
    db.create_job(
        settings.database_path,
        job_id,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = settings.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(analysis_payload(), allow_nan=False),
        encoding="utf-8",
    )
    db.update_job(
        settings.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        normalized_file_name="analysis.wav",
        analysis_status="completed",
        analysis_version=ANALYSIS_VERSION,
        analysis_json_file_name=ANALYSIS_JSON_RELATIVE_PATH,
        analyzed_at="2026-08-14T03:55:00+00:00",
        separation_status=separation_status,
        separation_stage=(
            "failed" if separation_status == "failed" else separation_status
        ),
        separation_error=(
            "Optional separation unavailable."
            if separation_status == "failed"
            else None
        ),
        transcription_status="completed" if transcribed else "not_started",
        transcription_stage="completed" if transcribed else "not_started",
        transcription_progress=100 if transcribed else 0,
        transcription_version="raw-transcription-v1" if transcribed else None,
        transcription_artifact_file_name=(
            RAW_TRANSCRIPTION_RELATIVE_PATH if transcribed else None
        ),
        transcribed_at=RAW_CREATED_AT if transcribed else None,
        pitched_event_count=3 if transcribed else None,
        percussion_event_count=0 if transcribed else None,
        aligned_event_count=0 if transcribed else None,
        interpretation_status=interpretation_status,
        interpretation_stage=interpretation_status,
        harmony_status=harmony_status,
        harmony_stage=harmony_status,
    )
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    return job_id


def publish_harmony(
    settings: Settings,
    job_id: str,
    *,
    created_at: str = HARMONY_CREATED_AT,
) -> dict:
    payload = harmony_artifact(created_at=created_at)
    write_harmony_artifact(job_id, settings, payload)
    diagnostics = payload["diagnostics"]
    db.update_job(
        settings.database_path,
        job_id,
        harmony_status="completed",
        harmony_stage="completed",
        harmony_progress=100,
        harmony_message="Harmonic context complete.",
        harmony_attempt_version=HARMONY_PIPELINE_VERSION,
        harmony_version=HARMONY_PIPELINE_VERSION,
        harmony_artifact_file_name=HARMONY_ARTIFACT_RELATIVE_PATH,
        harmonized_at=payload["createdAt"],
        harmony_source_transcription_version="raw-transcription-v1",
        harmony_source_transcription_artifact_file_name=(
            RAW_TRANSCRIPTION_RELATIVE_PATH
        ),
        harmony_source_transcribed_at=RAW_CREATED_AT,
        harmony_event_count=diagnostics["eventCount"],
        harmony_segment_count=diagnostics["segmentCount"],
        harmony_resolved_segment_count=diagnostics["resolvedSegmentCount"],
        harmony_unresolved_segment_count=diagnostics[
            "unresolvedSegmentCount"
        ],
        harmony_unresolved_event_count=diagnostics["unresolvedEventCount"],
        harmony_warning_count=diagnostics["warningCount"],
        harmony_used_interpretation_context=False,
        harmony_error=None,
    )
    return payload


def successful_processor(job_id, settings, stage_callback):
    stage_callback(
        "loading_analysis_context",
        "Loading matching timing and tonal evidence.",
        18,
    )
    stage_callback(
        "inferring_harmonic_context",
        "Inferring conservative local harmonic candidates.",
        48,
    )
    payload = harmony_artifact()
    write_harmony_artifact(job_id, settings, payload)
    stage_callback(
        "saving_harmonic_context",
        "Saving the canonical harmonic-context artifact.",
        92,
    )
    return result_from_artifact(payload)


def expected_failure(job_id, settings, stage_callback):
    stage_callback(
        "inferring_harmonic_context",
        "Inferring conservative local harmonic candidates.",
        48,
    )
    raise HarmonyPipelineError(
        f"Harmony failed near {settings.data_dir / 'private' / 'context.json'}; "
        "token=super-secret https://private.invalid/log"
    )


def unexpected_failure(job_id, settings, stage_callback):
    stage_callback(
        "validating_harmonic_context",
        "Validating harmonic evidence.",
        76,
    )
    raise RuntimeError(f"private failure at {settings.data_dir / 'secret.txt'}")


def invalid_processor(job_id, settings, stage_callback):
    payload = harmony_artifact()
    write_harmony_artifact(job_id, settings, payload)
    result = result_from_artifact(payload)
    return HarmonyPipelineResult(
        version=result.version,
        artifact_file_name="harmony/wrong.json",
        created_at=result.created_at,
        event_count=result.event_count,
        segment_count=result.segment_count,
        resolved_segment_count=result.resolved_segment_count,
        unresolved_segment_count=result.unresolved_segment_count,
        unresolved_event_count=result.unresolved_event_count,
        used_interpretation_context=result.used_interpretation_context,
        interpretation_version=result.interpretation_version,
        warning_count=result.warning_count,
        warnings=result.warnings,
        payload=result.payload,
    )


def upstream_snapshot(settings: Settings, job_id: str) -> dict:
    record = db.get_job(settings.database_path, job_id)
    assert record is not None
    fields = (
        "status",
        "stage",
        "progress",
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "analysis_json_file_name",
        "separation_status",
        "separation_stage",
        "separation_error",
        "transcription_status",
        "transcription_version",
        "transcription_artifact_file_name",
        "transcribed_at",
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
        "interpretation_status",
        "interpretation_stage",
        "interpretation_artifact_file_name",
        "interpreted_at",
    )
    return {field: record[field] for field in fields}


def test_contract_appears_only_after_raw_transcription(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings, transcribed=False)
        before = client.get(f"/api/jobs/{job_id}").json()
        assert "harmony" not in before

        write_raw_transcription(job_id, settings, raw_payload())
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_stage="completed",
            transcription_progress=100,
            transcription_version="raw-transcription-v1",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
            transcribed_at=RAW_CREATED_AT,
            pitched_event_count=3,
            percussion_event_count=0,
            aligned_event_count=0,
        )
        after = client.get(f"/api/jobs/{job_id}").json()

    contract = after["harmony"]
    assert contract["status"] == "not_started"
    assert contract["canStart"] is True
    assert contract["canReharmonize"] is False
    assert contract["available"] is False
    assert contract["startUrl"] == f"/api/jobs/{job_id}/harmonize"
    assert contract["detailsUrl"].endswith("?includeSegments=false")
    assert contract["fullDetailsUrl"].endswith("?includeSegments=true")
    assert contract["downloadUrl"] is None
    assert contract["counts"] == {
        "events": 0,
        "segments": 0,
        "resolved": 0,
        "unresolved": 0,
        "unresolvedEvents": 0,
        "warnings": 0,
    }
    for key in after:
        assert not key.startswith("harmony_")


def test_start_succeeds_without_stems_demucs_or_interpretation(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(
            settings,
            separation_status="failed",
            interpretation_status="failed",
        )
        before = upstream_snapshot(settings, job_id)
        response = client.post(f"/api/jobs/{job_id}/harmonize")
        job = client.get(f"/api/jobs/{job_id}").json()
        after = upstream_snapshot(settings, job_id)

    assert response.status_code == 202
    harmony = job["harmony"]
    assert harmony["status"] == "completed"
    assert harmony["available"] is True
    assert harmony["version"] == HARMONY_PIPELINE_VERSION
    assert harmony["attemptVersion"] == HARMONY_PIPELINE_VERSION
    assert harmony["counts"]["events"] == 3
    assert harmony["counts"]["segments"] >= 1
    assert harmony["canReharmonize"] is True
    assert harmony["usedInterpretationContext"] is False
    assert before == after
    assert "demucs" not in json.dumps(harmony).lower()


def test_start_conflicts_and_force_query_are_strict(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        assert client.post(f"/api/jobs/{'f' * 32}/harmonize").status_code == 404

        incomplete = create_job(settings, transcribed=False)
        assert client.post(f"/api/jobs/{incomplete}/harmonize").status_code == 409

        processing = create_job(settings, harmony_status="processing")
        assert client.post(f"/api/jobs/{processing}/harmonize").status_code == 409

        completed = create_job(settings)
        publish_harmony(settings, completed)
        assert client.post(f"/api/jobs/{completed}/harmonize").status_code == 409
        forced = client.post(f"/api/jobs/{completed}/harmonize?force=true")
        assert forced.status_code == 202
        assert client.post(f"/api/jobs/{completed}/harmonize?force=1").status_code == 422
        assert client.post(f"/api/jobs/{completed}/harmonize?force=false").status_code == 409


def test_invalid_or_missing_raw_or_analysis_blocks_start(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    calls: list[tuple] = []

    def processor(*args):
        calls.append(args)
        return successful_processor(*args)

    app = create_app(settings=settings, harmony_processor=processor)
    with TestClient(app) as client:
        missing_raw = create_job(settings)
        raw_path = settings.exports_dir / missing_raw / RAW_TRANSCRIPTION_RELATIVE_PATH
        raw_path.unlink()
        assert client.post(f"/api/jobs/{missing_raw}/harmonize").status_code == 409

        corrupt_raw = create_job(settings)
        raw_path = settings.exports_dir / corrupt_raw / RAW_TRANSCRIPTION_RELATIVE_PATH
        raw_path.write_text("{broken", encoding="utf-8")
        assert client.post(f"/api/jobs/{corrupt_raw}/harmonize").status_code == 409

        missing_analysis = create_job(settings)
        analysis_path = settings.exports_dir / missing_analysis / ANALYSIS_JSON_RELATIVE_PATH
        analysis_path.unlink()
        assert client.post(f"/api/jobs/{missing_analysis}/harmonize").status_code == 409

        stale_analysis = create_job(settings)
        analysis_path = settings.exports_dir / stale_analysis / ANALYSIS_JSON_RELATIVE_PATH
        payload = analysis_payload()
        payload["analysisVersion"] = "different-v2"
        analysis_path.write_text(json.dumps(payload), encoding="utf-8")
        assert client.post(f"/api/jobs/{stale_analysis}/harmonize").status_code == 409

    assert calls == []


def test_atomic_claim_conflict_schedules_no_processor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    calls = []

    def processor(*args):
        calls.append(args)
        return successful_processor(*args)

    app = create_app(settings=settings, harmony_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        monkeypatch.setattr(db, "claim_harmony_attempt", lambda *_a, **_k: False)
        response = client.post(f"/api/jobs/{job_id}/harmonize")

    assert response.status_code == 409
    assert calls == []


def test_successful_background_progress_and_completion_are_durable(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    observed: list[tuple[str, float]] = []

    def processor(job_id, app_settings, stage_callback):
        def capture(stage, message, progress):
            stage_callback(stage, message, progress)
            record = db.get_job(app_settings.database_path, job_id)
            assert record is not None
            observed.append((record["harmony_stage"], record["harmony_progress"]))

        return successful_processor(job_id, app_settings, capture)

    app = create_app(settings=settings, harmony_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.post(f"/api/jobs/{job_id}/harmonize")

    assert response.status_code == 202
    assert observed == [
        ("loading_analysis_context", 18.0),
        ("inferring_harmonic_context", 48.0),
        ("saving_harmonic_context", 92.0),
    ]
    record = db.get_job(settings.database_path, job_id)
    assert record is not None
    assert record["harmony_status"] == "completed"
    assert record["harmony_progress"] == 100
    assert record["harmony_artifact_file_name"] == HARMONY_ARTIFACT_RELATIVE_PATH
    assert record["harmony_segment_count"] == (
        record["harmony_resolved_segment_count"]
        + record["harmony_unresolved_segment_count"]
    )


@pytest.mark.parametrize(
    ("processor", "expected_fragment"),
    [
        (expected_failure, "harmony"),
        (unexpected_failure, "unexpected"),
        (invalid_processor, "invalid"),
    ],
)
def test_processor_failures_are_bounded_and_preserve_upstream(
    tmp_path: Path,
    processor,
    expected_fragment: str,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(
            settings,
            separation_status="failed",
            interpretation_status="failed",
        )
        before = upstream_snapshot(settings, job_id)
        response = client.post(f"/api/jobs/{job_id}/harmonize")
        after_job = client.get(f"/api/jobs/{job_id}").json()
        after = upstream_snapshot(settings, job_id)

    assert response.status_code == 202
    harmony = after_job["harmony"]
    assert harmony["status"] == "failed"
    error = harmony["error"]
    assert isinstance(error, str) and error
    assert len(error) <= 500
    assert "/" not in error
    assert "\\" not in error
    assert "super-secret" not in error
    assert "private.invalid" not in error
    assert expected_fragment in error.lower()
    assert before == after


def test_previous_artifact_remains_visible_during_recomputation(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        previous = publish_harmony(settings, job_id)
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version="harmonic-context-v2",
            force=True,
        )
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/harmony")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    contract = job["harmony"]
    assert contract["status"] == "processing"
    assert contract["available"] is True
    assert contract["version"] == HARMONY_PIPELINE_VERSION
    assert contract["attemptVersion"] == "harmonic-context-v2"
    assert details.status_code == 200
    assert details.json()["harmonyVersion"] == previous["harmonyVersion"]
    assert download.status_code == 200


def test_failed_forced_retry_preserves_previous_details_and_download(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=expected_failure)
    with TestClient(app) as client:
        job_id = create_job(settings)
        previous = publish_harmony(settings, job_id)
        before = upstream_snapshot(settings, job_id)
        response = client.post(f"/api/jobs/{job_id}/harmonize?force=true")
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")
        after = upstream_snapshot(settings, job_id)

    assert response.status_code == 202
    contract = job["harmony"]
    assert contract["status"] == "failed"
    assert contract["available"] is True
    assert contract["version"] == HARMONY_PIPELINE_VERSION
    assert details.status_code == 200
    assert details.json()["segments"] == previous["segments"]
    assert download.status_code == 200
    assert json.loads(download.content) == previous
    assert before == after


def test_stale_source_worker_cannot_complete_or_fail_new_source(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)

    def stale_processor(job_id, app_settings, stage_callback):
        stage_callback(
            "inferring_harmonic_context",
            "Inferring conservative local harmonic candidates.",
            48,
        )
        payload = harmony_artifact()
        write_harmony_artifact(job_id, app_settings, payload)
        db.update_job(
            app_settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_stage="completed",
            transcription_progress=100,
            transcription_version="raw-transcription-v2",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
            transcribed_at="2026-08-14T05:00:00+00:00",
            pitched_event_count=3,
            percussion_event_count=0,
            aligned_event_count=0,
        )
        return result_from_artifact(payload)

    app = create_app(settings=settings, harmony_processor=stale_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.post(f"/api/jobs/{job_id}/harmonize")
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/harmony")

    assert response.status_code == 202
    contract = job["harmony"]
    assert contract["status"] == "not_started"
    assert contract["available"] is False
    assert details.status_code == 404
    record = db.get_job(settings.database_path, job_id)
    assert record is not None
    assert record["transcription_version"] == "raw-transcription-v2"
    assert record["harmony_error"] is None


def test_summary_full_details_and_download_use_validated_artifact_truth(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        artifact = publish_harmony(settings, job_id)
        summary = client.get(f"/api/jobs/{job_id}/harmony")
        explicit_summary = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=false"
        )
        full = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        invalid = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=1")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert summary.status_code == 200
    assert explicit_summary.json() == summary.json()
    summary_payload = summary.json()
    assert summary_payload["available"] is True
    assert summary_payload["status"] == "completed"
    assert summary_payload["segmentsIncluded"] is False
    assert summary_payload["counts"] == {
        "events": artifact["diagnostics"]["eventCount"],
        "segments": artifact["diagnostics"]["segmentCount"],
        "resolved": artifact["diagnostics"]["resolvedSegmentCount"],
        "unresolved": artifact["diagnostics"]["unresolvedSegmentCount"],
        "unresolvedEvents": artifact["diagnostics"]["unresolvedEventCount"],
        "warnings": artifact["diagnostics"]["warningCount"],
    }
    assert "rawEvidence" not in summary_payload
    assert "segments" not in summary_payload
    assert "unresolvedEventIds" not in summary_payload

    assert full.status_code == 200
    full_payload = full.json()
    assert full_payload["segmentsIncluded"] is True
    assert full_payload["rawEvidence"] == artifact["rawEvidence"]
    assert full_payload["segments"] == artifact["segments"]
    assert full_payload["unresolvedEventIds"] == artifact[
        "unresolvedEventIds"
    ]
    assert invalid.status_code == 422

    assert download.status_code == 200
    assert download.headers["content-type"].startswith("application/json")
    assert "harmonic-context.json" in download.headers["content-disposition"]
    assert json.loads(download.content) == artifact


def test_unavailable_corrupt_and_metadata_mismatch_are_not_exposed(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        unavailable = create_job(settings)
        assert client.get(f"/api/jobs/{unavailable}/harmony").status_code == 404
        assert (
            client.get(f"/api/jobs/{unavailable}/harmony/download").status_code
            == 404
        )

        corrupt = create_job(settings)
        publish_harmony(settings, corrupt)
        artifact_path = settings.exports_dir / corrupt / HARMONY_ARTIFACT_RELATIVE_PATH
        artifact_path.write_text("{broken", encoding="utf-8")
        assert client.get(f"/api/jobs/{corrupt}/harmony").status_code == 500
        assert (
            client.get(f"/api/jobs/{corrupt}/harmony/download").status_code
            == 500
        )

        mismatch = create_job(settings)
        publish_harmony(settings, mismatch)
        db.update_job(
            settings.database_path,
            mismatch,
            harmony_event_count=999,
        )
        assert client.get(f"/api/jobs/{mismatch}/harmony").status_code == 500
        assert (
            client.get(f"/api/jobs/{mismatch}/harmony/download").status_code
            == 500
        )


def test_symlinked_artifact_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_harmony(settings, job_id)
        path = settings.exports_dir / job_id / HARMONY_ARTIFACT_RELATIVE_PATH
        outside = tmp_path / "outside.json"
        path.replace(outside)
        try:
            path.symlink_to(outside)
        except OSError:
            pytest.skip("symlinks unavailable")
        assert client.get(f"/api/jobs/{job_id}/harmony").status_code == 500
        assert (
            client.get(f"/api/jobs/{job_id}/harmony/download").status_code
            == 500
        )


def test_restart_failure_preserves_previous_artifact_contract(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    settings.ensure_directories()
    db.init_database(settings.database_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    assert db.claim_harmony_attempt(
        settings.database_path,
        job_id,
        harmony_version="harmonic-context-v2",
        force=True,
    )
    db.fail_incomplete_jobs(settings.database_path)

    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/harmony")

    contract = job["harmony"]
    assert contract["status"] == "failed"
    assert contract["available"] is True
    assert contract["canStart"] is True
    assert details.status_code == 200


def test_internal_harmony_columns_and_hostile_values_do_not_leak(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_harmony(settings, job_id)
        db.update_job(
            settings.database_path,
            job_id,
            harmony_status="failed",
            harmony_stage="failed",
            harmony_message="Safe retry message.",
            harmony_error="Harmonic-context processing failed.",
        )
        payload = client.get(f"/api/jobs/{job_id}").json()

    assert payload["harmony"]["error"] == "Harmonic-context processing failed."
    for key in payload:
        assert not key.startswith("harmony_")
    encoded = json.dumps(payload).lower()
    assert "sqlite" not in encoded
    assert "token=" not in encoded


def test_no_final_notation_or_export_claims() -> None:
    source = (Path(__file__).resolve().parents[1] / "app" / "main.py").read_text(
        encoding="utf-8"
    ).lower()
    harmony_section = source[source.index("def _run_harmony_job") :]
    for forbidden in (
        "musicxml",
        "midi export",
        "tablature",
        "engraving",
        "publication-ready",
        "final chord chart",
        "exact voicing",
    ):
        assert forbidden not in harmony_section
