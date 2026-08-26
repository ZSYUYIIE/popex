from __future__ import annotations

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import db
from app import harmony_artifacts as harmony_artifacts_module
from app import main as main_module
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    build_harmony_artifact,
    harmony_attempt_artifact_file_name,
    load_harmony_artifact,
    write_harmony_artifact,
)
from app.harmony_inference import infer_harmony
from app.harmony_pipeline import (
    HARMONY_PIPELINE_VERSION,
    HarmonyPipelineError,
    HarmonyPipelineResult,
)
from app.main import _run_harmony_job, create_app
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


def result_from_artifact(
    payload: dict,
    *,
    artifact_file_name: str = HARMONY_ARTIFACT_RELATIVE_PATH,
) -> HarmonyPipelineResult:
    diagnostics = payload["diagnostics"]
    warnings = tuple(payload["warnings"])
    return HarmonyPipelineResult(
        version=payload["harmonyVersion"],
        artifact_file_name=artifact_file_name,
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
    )
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    if harmony_status == "processing":
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
    elif harmony_status != "not_started":
        db.update_job(
            settings.database_path,
            job_id,
            harmony_status=harmony_status,
            harmony_stage=harmony_status,
        )
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
        harmony_warning_count=len(payload["warnings"]),
        harmony_used_interpretation_context=False,
        harmony_error=None,
    )
    return payload


def _attempt_target(attempt_id: str | None) -> str:
    return (
        HARMONY_ARTIFACT_RELATIVE_PATH
        if attempt_id is None
        else harmony_attempt_artifact_file_name(attempt_id)
    )


def successful_processor(
    job_id,
    settings,
    stage_callback,
    *,
    attempt_id=None,
):
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
    target = _attempt_target(attempt_id)
    write_harmony_artifact(
        job_id,
        settings,
        payload,
        artifact_file_name=target,
    )
    stage_callback(
        "saving_harmonic_context",
        "Saving the canonical harmonic-context artifact.",
        92,
    )
    return result_from_artifact(payload, artifact_file_name=target)


def expected_failure(
    job_id,
    settings,
    stage_callback,
    *,
    attempt_id=None,
):
    stage_callback(
        "inferring_harmonic_context",
        "Inferring conservative local harmonic candidates.",
        48,
    )
    raise HarmonyPipelineError(
        f"Harmony failed near {settings.data_dir / 'private' / 'context.json'}; "
        "token=super-secret https://private.invalid/log"
    )


def unexpected_failure(
    job_id,
    settings,
    stage_callback,
    *,
    attempt_id=None,
):
    stage_callback(
        "validating_harmonic_context",
        "Validating harmonic evidence.",
        76,
    )
    raise RuntimeError(f"private failure at {settings.data_dir / 'secret.txt'}")


def invalid_processor(
    job_id,
    settings,
    stage_callback,
    *,
    attempt_id=None,
):
    payload = harmony_artifact()
    target = _attempt_target(attempt_id)
    write_harmony_artifact(
        job_id,
        settings,
        payload,
        artifact_file_name=target,
    )
    result = result_from_artifact(payload, artifact_file_name=target)
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
        forced_processing = client.post(
            f"/api/jobs/{processing}/harmonize?force=true"
        )
        assert forced_processing.status_code == 409

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

    def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return successful_processor(*args, **kwargs)

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

        mismatched_raw = create_job(settings)
        payload = raw_payload()
        payload["createdAt"] = "2026-08-14T04:01:00+00:00"
        write_raw_transcription(mismatched_raw, settings, payload)
        assert client.post(f"/api/jobs/{mismatched_raw}/harmonize").status_code == 409

        mismatched_counts = create_job(settings)
        db.update_job(
            settings.database_path,
            mismatched_counts,
            pitched_event_count=999,
        )
        assert client.post(f"/api/jobs/{mismatched_counts}/harmonize").status_code == 409

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

    def processor(*args, **kwargs):
        calls.append((args, kwargs))
        return successful_processor(*args, **kwargs)

    app = create_app(settings=settings, harmony_processor=processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        monkeypatch.setattr(db, "claim_harmony_attempt", lambda *_a, **_k: None)
        response = client.post(f"/api/jobs/{job_id}/harmonize")

    assert response.status_code == 409
    assert calls == []


def test_request_schedules_only_the_attempt_identity_returned_by_its_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    scheduled_tasks: list[tuple[object, tuple[object, ...], dict[str, object]]] = []
    claimed_attempts: list[str] = []
    processor_calls: list[str | None] = []
    first_claimed = Event()
    release_first = Event()
    state_lock = Lock()
    real_claim = db.claim_harmony_attempt

    def paused_first_claim(*args, **kwargs):
        claimed = real_claim(*args, **kwargs)
        assert isinstance(claimed, str)
        with state_lock:
            claimed_attempts.append(claimed)
            is_first = len(claimed_attempts) == 1
        if is_first:
            first_claimed.set()
            assert release_first.wait(5)
        return claimed

    def capture_task(_background_tasks, function, *args, **kwargs):
        with state_lock:
            scheduled_tasks.append((function, args, kwargs))

    def tracked_processor(*args, attempt_id=None, **kwargs):
        processor_calls.append(attempt_id)
        return successful_processor(
            *args,
            attempt_id=attempt_id,
            **kwargs,
        )

    monkeypatch.setattr(db, "claim_harmony_attempt", paused_first_claim)
    monkeypatch.setattr(main_module.BackgroundTasks, "add_task", capture_task)
    app = create_app(settings=settings, harmony_processor=tracked_processor)

    with TestClient(app) as client:
        job_id = create_job(settings)
        with ThreadPoolExecutor(max_workers=1) as executor:
            first_response_future = executor.submit(
                client.post,
                f"/api/jobs/{job_id}/harmonize",
            )
            assert first_claimed.wait(5)
            try:
                replacement = raw_payload()
                write_raw_transcription(job_id, settings, replacement)
                db.update_job(
                    settings.database_path,
                    job_id,
                    transcription_status="completed",
                    transcription_stage="completed",
                    transcription_progress=100,
                    transcription_message="Replacement raw transcription complete.",
                    transcription_version="raw-transcription-v1",
                    transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
                    transcribed_at=RAW_CREATED_AT,
                    pitched_event_count=3,
                    percussion_event_count=0,
                    aligned_event_count=0,
                    transcription_error=None,
                )
                second_response = client.post(f"/api/jobs/{job_id}/harmonize")
            finally:
                release_first.set()
            first_response = first_response_future.result(timeout=5)

        assert first_response.status_code == second_response.status_code == 202
        assert len(claimed_attempts) == 2
        first_attempt, second_attempt = claimed_attempts
        assert first_attempt != second_attempt
        assert len(scheduled_tasks) == 2
        scheduled_attempts = [task[1][-1] for task in scheduled_tasks]
        assert scheduled_attempts.count(first_attempt) == 1
        assert scheduled_attempts.count(second_attempt) == 1

        second_task = next(
            task for task in scheduled_tasks if task[1][-1] == second_attempt
        )
        second_task[0](*second_task[1], **second_task[2])
        completed = db.get_job(settings.database_path, job_id)
        assert completed is not None
        pointer = completed["harmony_artifact_file_name"]
        assert pointer == harmony_attempt_artifact_file_name(second_attempt)
        path = settings.exports_dir / job_id / pointer
        before_bytes = path.read_bytes()
        before_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        before_download = client.get(f"/api/jobs/{job_id}/harmony/download")

        first_task = next(
            task for task in scheduled_tasks if task[1][-1] == first_attempt
        )
        first_task[0](*first_task[1], **first_task[2])
        after = db.get_job(settings.database_path, job_id)
        after_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        after_download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert processor_calls == [second_attempt]
    assert after == completed
    assert path.read_bytes() == before_bytes
    assert before_details.status_code == after_details.status_code == 200
    assert before_details.json() == after_details.json()
    assert before_download.status_code == after_download.status_code == 200
    assert before_download.content == after_download.content == before_bytes
    assert not (
        settings.exports_dir
        / job_id
        / harmony_attempt_artifact_file_name(first_attempt)
    ).exists()


def test_successful_background_progress_and_completion_are_durable(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    observed: list[tuple[str, float]] = []

    def processor(
        job_id,
        app_settings,
        stage_callback,
        *,
        attempt_id=None,
    ):
        def capture(stage, message, progress):
            stage_callback(stage, message, progress)
            record = db.get_job(app_settings.database_path, job_id)
            assert record is not None
            observed.append((record["harmony_stage"], record["harmony_progress"]))

        return successful_processor(
            job_id,
            app_settings,
            capture,
            attempt_id=attempt_id,
        )

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
    assert record["harmony_attempt_id"] is None
    pointer = record["harmony_artifact_file_name"]
    assert isinstance(pointer, str)
    assert pointer.startswith("harmony/harmonic-context.")
    assert pointer.endswith(".json")
    assert pointer != HARMONY_ARTIFACT_RELATIVE_PATH
    assert load_harmony_artifact(
        job_id,
        settings,
        artifact_file_name=pointer,
    ) is not None
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

    def stale_processor(
        job_id,
        app_settings,
        stage_callback,
        *,
        attempt_id=None,
    ):
        stage_callback(
            "inferring_harmonic_context",
            "Inferring conservative local harmonic candidates.",
            48,
        )
        payload = harmony_artifact()
        target = _attempt_target(attempt_id)
        write_harmony_artifact(
            job_id,
            app_settings,
            payload,
            artifact_file_name=target,
        )
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
        return result_from_artifact(payload, artifact_file_name=target)

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


def test_old_worker_cannot_complete_reclaimed_same_version_attempt(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    attempt_ids: list[str] = []

    def stale_reclaim_processor(
        job_id,
        app_settings,
        stage_callback,
        *,
        attempt_id=None,
    ):
        record = db.get_job(app_settings.database_path, job_id)
        assert record is not None
        old_attempt_id = record["harmony_attempt_id"]
        assert isinstance(old_attempt_id, str)
        assert old_attempt_id == attempt_id
        attempt_ids.append(old_attempt_id)
        stage_callback(
            "inferring_harmonic_context",
            "Inferring conservative local harmonic candidates.",
            48,
        )
        old_payload = harmony_artifact()
        old_target = harmony_attempt_artifact_file_name(old_attempt_id)
        write_harmony_artifact(
            job_id,
            app_settings,
            old_payload,
            artifact_file_name=old_target,
        )

        replacement = raw_payload()
        replacement["transcriptionVersion"] = "raw-transcription-v2"
        replacement["createdAt"] = "2026-08-14T05:00:00+00:00"
        write_raw_transcription(job_id, app_settings, replacement)
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
        assert db.claim_harmony_attempt(
            app_settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        reclaimed = db.get_job(app_settings.database_path, job_id)
        assert reclaimed is not None
        new_attempt_id = reclaimed["harmony_attempt_id"]
        assert isinstance(new_attempt_id, str)
        attempt_ids.append(new_attempt_id)
        return result_from_artifact(
            old_payload,
            artifact_file_name=old_target,
        )

    app = create_app(settings=settings, harmony_processor=stale_reclaim_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.post(f"/api/jobs/{job_id}/harmonize")
        details = client.get(f"/api/jobs/{job_id}/harmony")

    assert response.status_code == 202
    assert len(attempt_ids) == 2
    assert attempt_ids[0] != attempt_ids[1]
    record = db.get_job(settings.database_path, job_id)
    assert record is not None
    assert record["harmony_status"] == "processing"
    assert record["harmony_attempt_id"] == attempt_ids[1]
    assert record["harmony_artifact_file_name"] is None
    assert details.status_code == 404
    old_path = (
        settings.exports_dir
        / job_id
        / harmony_attempt_artifact_file_name(attempt_ids[0])
    )
    assert old_path.is_file()


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
        "warnings": len(artifact["warnings"]),
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


def test_raw_event_warnings_survive_default_pipeline_details_and_download(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    warning_list = [
        "Dominant line only; review this pitch.",
        "Pitch boundary remains uncertain.",
    ]
    with TestClient(app) as client:
        job_id = create_job(settings)
        payload = raw_payload()
        payload["pitchedNoteEvents"][0]["warnings"] = warning_list
        write_raw_transcription(job_id, settings, payload)

        response = client.post(f"/api/jobs/{job_id}/harmonize")
        full = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert response.status_code == 202
    assert full.status_code == 200
    evidence = {item["id"]: item for item in full.json()["rawEvidence"]}
    assert evidence["p_c"]["warnings"] == warning_list
    assert evidence["p_e"]["warnings"] == []
    assert evidence["p_g"]["warnings"] == []
    assert download.status_code == 200
    downloaded = json.loads(download.content)
    downloaded_evidence = {item["id"]: item for item in downloaded["rawEvidence"]}
    assert downloaded_evidence["p_c"]["warnings"] == warning_list


def test_legacy_missing_warning_field_remains_readable_with_current_raw_warnings(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        legacy = publish_harmony(settings, job_id)
        for item in legacy["rawEvidence"]:
            item.pop("warnings", None)
        write_harmony_artifact(job_id, settings, legacy)

        current_raw = raw_payload()
        current_raw["pitchedNoteEvents"][0]["warnings"] = ["Review this pitch."]
        write_raw_transcription(job_id, settings, current_raw)

        full = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert full.status_code == 200
    evidence = {item["id"]: item for item in full.json()["rawEvidence"]}
    assert "warnings" not in evidence["p_c"]
    assert download.status_code == 200
    downloaded = json.loads(download.content)
    downloaded_evidence = {item["id"]: item for item in downloaded["rawEvidence"]}
    assert "warnings" not in downloaded_evidence["p_c"]


def test_legacy_explicit_empty_warnings_do_not_mask_current_warning_change(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_harmony(settings, job_id)
        current_raw = raw_payload()
        current_raw["pitchedNoteEvents"][0]["warnings"] = ["Review this pitch."]
        write_raw_transcription(job_id, settings, current_raw)
        details = client.get(f"/api/jobs/{job_id}/harmony")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert details.status_code == 500
    assert download.status_code == 500


def test_warning_only_raw_mutation_invalidates_new_attempt_artifact(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings)
    with TestClient(app) as client:
        job_id = create_job(settings)
        response = client.post(f"/api/jobs/{job_id}/harmonize")
        assert response.status_code == 202
        completed = db.get_job(settings.database_path, job_id)
        assert completed is not None
        pointer = completed["harmony_artifact_file_name"]
        assert isinstance(pointer, str)
        assert pointer != HARMONY_ARTIFACT_RELATIVE_PATH

        current_raw = raw_payload()
        current_raw["pitchedNoteEvents"][0]["warnings"] = ["Review this pitch."]
        write_raw_transcription(job_id, settings, current_raw)
        details = client.get(f"/api/jobs/{job_id}/harmony")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert details.status_code == 500
    assert download.status_code == 500


def test_unscoped_injected_processor_is_redirected_and_failed_retry_preserves_legacy(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        previous = publish_harmony(settings, job_id)
        legacy_path = settings.exports_dir / job_id / HARMONY_ARTIFACT_RELATIVE_PATH
        legacy_bytes = legacy_path.read_bytes()
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
            force=True,
        )
        claimed = db.get_job(settings.database_path, job_id)
        assert claimed is not None
        attempt_id = claimed["harmony_attempt_id"]
        assert isinstance(attempt_id, str)
        target = harmony_attempt_artifact_file_name(attempt_id)

        def legacy_writer(
            worker_job_id,
            app_settings,
            _stage_callback,
            *,
            attempt_id=None,
        ):
            payload = harmony_artifact(created_at="2026-08-14T04:30:00+00:00")
            redirected = write_harmony_artifact(
                worker_job_id,
                app_settings,
                payload,
            )
            assert redirected.name == Path(target).name
            assert legacy_path.read_bytes() == legacy_bytes
            return result_from_artifact(payload)

        _run_harmony_job(job_id, settings, legacy_writer, attempt_id)
        record = db.get_job(settings.database_path, job_id)
        assert record is not None
        details = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert record["harmony_status"] == "failed"
    assert record["harmony_artifact_file_name"] == HARMONY_ARTIFACT_RELATIVE_PATH
    assert legacy_path.read_bytes() == legacy_bytes
    attempt_path = settings.exports_dir / job_id / target
    assert attempt_path.exists() is (
        not harmony_artifacts_module._descriptor_relative_cleanup_supported()
    )
    assert details.status_code == 200
    assert details.json()["createdAt"] == previous["createdAt"]
    assert download.status_code == 200
    assert download.content == legacy_bytes


def test_late_old_worker_cannot_replace_or_hide_newer_success(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)

    with TestClient(app) as client:
        job_id = create_job(settings)
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        old_record = db.get_job(settings.database_path, job_id)
        assert old_record is not None
        old_attempt = old_record["harmony_attempt_id"]
        assert isinstance(old_attempt, str)

        replacement = raw_payload()
        write_raw_transcription(job_id, settings, replacement)
        db.update_job(
            settings.database_path,
            job_id,
            transcription_status="completed",
            transcription_stage="completed",
            transcription_progress=100,
            transcription_message="Replacement raw transcription complete.",
            transcription_version="raw-transcription-v1",
            transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
            transcribed_at=RAW_CREATED_AT,
            pitched_event_count=3,
            percussion_event_count=0,
            aligned_event_count=0,
            transcription_error=None,
        )
        invalidated = db.get_job(settings.database_path, job_id)
        assert invalidated is not None
        assert invalidated["harmony_status"] == "not_started"
        assert invalidated["harmony_attempt_id"] is None

        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        new_record = db.get_job(settings.database_path, job_id)
        assert new_record is not None
        new_attempt = new_record["harmony_attempt_id"]
        assert isinstance(new_attempt, str) and new_attempt != old_attempt

        def new_processor(
            worker_job_id,
            app_settings,
            _stage_callback,
            *,
            attempt_id=None,
        ):
            assert attempt_id == new_attempt
            payload = harmony_artifact(created_at="2026-08-14T04:20:00+00:00")
            target = harmony_attempt_artifact_file_name(new_attempt)
            write_harmony_artifact(
                worker_job_id,
                app_settings,
                payload,
                artifact_file_name=target,
            )
            return result_from_artifact(payload, artifact_file_name=target)

        _run_harmony_job(job_id, settings, new_processor, new_attempt)
        completed = db.get_job(settings.database_path, job_id)
        assert completed is not None
        new_pointer = completed["harmony_artifact_file_name"]
        assert new_pointer == harmony_attempt_artifact_file_name(new_attempt)

        def old_processor(
            worker_job_id,
            app_settings,
            _stage_callback,
            *,
            attempt_id=None,
        ):
            assert attempt_id == old_attempt
            payload = harmony_artifact(created_at="2026-08-14T04:30:00+00:00")
            target = harmony_attempt_artifact_file_name(old_attempt)
            write_harmony_artifact(
                worker_job_id,
                app_settings,
                payload,
                artifact_file_name=target,
            )
            return result_from_artifact(payload, artifact_file_name=target)

        _run_harmony_job(job_id, settings, old_processor, old_attempt)
        final_record = db.get_job(settings.database_path, job_id)
        details = client.get(f"/api/jobs/{job_id}/harmony?includeSegments=true")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert final_record == completed
    assert details.status_code == 200
    assert details.json()["createdAt"] == "2026-08-14T04:20:00+00:00"
    assert download.status_code == 200
    assert json.loads(download.content)["createdAt"] == "2026-08-14T04:20:00+00:00"
    old_path = (
        settings.exports_dir
        / job_id
        / harmony_attempt_artifact_file_name(old_attempt)
    )
    new_path = settings.exports_dir / job_id / new_pointer
    assert not old_path.exists()
    assert new_path.is_file()


def test_late_duplicate_worker_cannot_remove_the_durable_winner(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    processor_calls: list[str | None] = []

    def tracked_processor(*args, attempt_id=None, **kwargs):
        processor_calls.append(attempt_id)
        return successful_processor(*args, attempt_id=attempt_id, **kwargs)

    app = create_app(settings=settings, harmony_processor=tracked_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        claimed_attempt = db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        record = db.get_job(settings.database_path, job_id)
        assert record is not None
        attempt_id = record["harmony_attempt_id"]
        assert isinstance(attempt_id, str)
        assert claimed_attempt

        _run_harmony_job(job_id, settings, tracked_processor, attempt_id)
        completed = db.get_job(settings.database_path, job_id)
        assert completed is not None
        pointer = completed["harmony_artifact_file_name"]
        assert isinstance(pointer, str)
        path = settings.exports_dir / job_id / pointer
        before_bytes = path.read_bytes()
        before_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        before_download = client.get(f"/api/jobs/{job_id}/harmony/download")

        _run_harmony_job(job_id, settings, tracked_processor, attempt_id)

        after = db.get_job(settings.database_path, job_id)
        after_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        after_download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert processor_calls == [attempt_id]
    assert after == completed
    assert path.read_bytes() == before_bytes
    assert before_details.status_code == after_details.status_code == 200
    assert before_details.json() == after_details.json()
    assert before_download.status_code == after_download.status_code == 200
    assert before_download.content == after_download.content == before_bytes


def test_only_one_duplicate_worker_can_consume_and_complete_an_attempt(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    winner_entered = Event()
    release_winner = Event()
    processor_calls: list[str] = []

    def paused_winner(
        worker_job_id,
        app_settings,
        stage_callback,
        *,
        attempt_id=None,
    ):
        assert isinstance(attempt_id, str)
        processor_calls.append(attempt_id)
        winner_entered.set()
        assert release_winner.wait(5)
        return successful_processor(
            worker_job_id,
            app_settings,
            stage_callback,
            attempt_id=attempt_id,
        )

    def forbidden_duplicate(*_args, **_kwargs):
        raise AssertionError("a duplicate worker must not enter the processor")

    with TestClient(app) as client:
        job_id = create_job(settings)
        attempt_id = db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        assert isinstance(attempt_id, str)

        with ThreadPoolExecutor(max_workers=1) as executor:
            winner = executor.submit(
                _run_harmony_job,
                job_id,
                settings,
                paused_winner,
                attempt_id,
            )
            assert winner_entered.wait(5)
            try:
                _run_harmony_job(
                    job_id,
                    settings,
                    forbidden_duplicate,
                    attempt_id,
                )
            finally:
                release_winner.set()
            winner.result(timeout=5)

        completed = db.get_job(settings.database_path, job_id)
        assert completed is not None
        pointer = completed["harmony_artifact_file_name"]
        assert isinstance(pointer, str)
        path = settings.exports_dir / job_id / pointer
        before_bytes = path.read_bytes()
        before_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        before_download = client.get(f"/api/jobs/{job_id}/harmony/download")

        _run_harmony_job(
            job_id,
            settings,
            forbidden_duplicate,
            attempt_id,
        )
        after = db.get_job(settings.database_path, job_id)
        after_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        after_download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert processor_calls == [attempt_id]
    assert after == completed
    assert path.read_bytes() == before_bytes
    assert before_details.status_code == after_details.status_code == 200
    assert before_details.json() == after_details.json()
    assert before_download.status_code == after_download.status_code == 200
    assert before_download.content == after_download.content == before_bytes


def test_stale_reconciliation_cannot_remove_a_newer_durable_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    entered_reconciliation = Event()
    release_reconciliation = Event()
    real_reconcile = main_module.reconcile_harmony_attempt_artifacts
    old_attempt: str | None = None
    old_state_reads = 0

    def pause_old_reconciliation(*args, **kwargs):
        nonlocal old_state_reads
        if kwargs.get("active_attempt_id") == old_attempt:
            real_reader = kwargs["protection_state_reader"]
            first_read = True

            def stale_then_current_state():
                nonlocal first_read, old_state_reads
                old_state_reads += 1
                current = real_reader()
                if first_read:
                    first_read = False
                    entered_reconciliation.set()
                    assert release_reconciliation.wait(5)
                return current

            kwargs = {
                **kwargs,
                "protection_state_reader": stale_then_current_state,
            }
        return real_reconcile(*args, **kwargs)

    monkeypatch.setattr(
        main_module,
        "reconcile_harmony_attempt_artifacts",
        pause_old_reconciliation,
    )

    with TestClient(app) as client:
        job_id = create_job(settings)
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
        old_record = db.get_job(settings.database_path, job_id)
        assert old_record is not None
        old_attempt = old_record["harmony_attempt_id"]
        assert isinstance(old_attempt, str)

        with ThreadPoolExecutor(max_workers=1) as executor:
            old_worker = executor.submit(
                _run_harmony_job,
                job_id,
                settings,
                successful_processor,
                old_attempt,
            )
            assert entered_reconciliation.wait(5)
            try:
                replacement = raw_payload()
                write_raw_transcription(job_id, settings, replacement)
                db.update_job(
                    settings.database_path,
                    job_id,
                    transcription_status="completed",
                    transcription_stage="completed",
                    transcription_progress=100,
                    transcription_message="Replacement raw transcription complete.",
                    transcription_version="raw-transcription-v1",
                    transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
                    transcribed_at=RAW_CREATED_AT,
                    pitched_event_count=3,
                    percussion_event_count=0,
                    aligned_event_count=0,
                    transcription_error=None,
                )
                assert db.claim_harmony_attempt(
                    settings.database_path,
                    job_id,
                    harmony_version=HARMONY_PIPELINE_VERSION,
                )
                new_record = db.get_job(settings.database_path, job_id)
                assert new_record is not None
                new_attempt = new_record["harmony_attempt_id"]
                assert isinstance(new_attempt, str) and new_attempt != old_attempt

                _run_harmony_job(
                    job_id,
                    settings,
                    successful_processor,
                    new_attempt,
                )
                completed = db.get_job(settings.database_path, job_id)
                assert completed is not None
                pointer = completed["harmony_artifact_file_name"]
                assert isinstance(pointer, str)
                path = settings.exports_dir / job_id / pointer
                before_bytes = path.read_bytes()
                before_details = client.get(
                    f"/api/jobs/{job_id}/harmony?includeSegments=true"
                )
                before_download = client.get(
                    f"/api/jobs/{job_id}/harmony/download"
                )
            finally:
                release_reconciliation.set()
            old_worker.result(timeout=5)

        after = db.get_job(settings.database_path, job_id)
        after_details = client.get(
            f"/api/jobs/{job_id}/harmony?includeSegments=true"
        )
        after_download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert after == completed
    assert path.read_bytes() == before_bytes
    assert before_details.status_code == after_details.status_code == 200
    assert before_details.json() == after_details.json()
    assert before_download.status_code == after_download.status_code == 200
    assert before_download.content == after_download.content == before_bytes
    if harmony_artifacts_module._descriptor_relative_cleanup_supported():
        assert old_state_reads >= 2


def test_worker_reconciles_repeated_crash_orphans_without_deleting_live_or_durable(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    publish_harmony(settings, job_id)
    legacy_path = settings.exports_dir / job_id / HARMONY_ARTIFACT_RELATIVE_PATH
    legacy_bytes = legacy_path.read_bytes()
    cleanup_supported = (
        harmony_artifacts_module._descriptor_relative_cleanup_supported()
    )
    orphan_targets: list[str] = []

    for cycle in range(3):
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
            force=(cycle == 0),
        )
        record = db.get_job(settings.database_path, job_id)
        assert record is not None
        attempt_id = record["harmony_attempt_id"]
        assert isinstance(attempt_id, str)
        target = harmony_attempt_artifact_file_name(attempt_id)
        orphan_targets.append(target)
        write_harmony_artifact(
            job_id,
            settings,
            harmony_artifact(),
            artifact_file_name=target,
        )
        db.fail_incomplete_jobs(settings.database_path)
        restarted = db.get_job(settings.database_path, job_id)
        assert restarted is not None
        assert restarted["harmony_status"] == "failed"
        assert restarted["harmony_attempt_id"] is None
        assert restarted["harmony_artifact_file_name"] == HARMONY_ARTIFACT_RELATIVE_PATH

    for target in orphan_targets:
        assert (settings.exports_dir / job_id / target).is_file()
    assert legacy_path.read_bytes() == legacy_bytes

    assert db.claim_harmony_attempt(
        settings.database_path,
        job_id,
        harmony_version=HARMONY_PIPELINE_VERSION,
    )
    live_record = db.get_job(settings.database_path, job_id)
    assert live_record is not None
    live_attempt = live_record["harmony_attempt_id"]
    assert isinstance(live_attempt, str)
    live_target = harmony_attempt_artifact_file_name(live_attempt)
    live_payload = harmony_artifact(created_at="2026-08-14T04:40:00+00:00")
    live_path = write_harmony_artifact(
        job_id,
        settings,
        live_payload,
        artifact_file_name=live_target,
    )

    def already_published_processor(
        worker_job_id,
        app_settings,
        _stage_callback,
        *,
        attempt_id=None,
    ):
        assert worker_job_id == job_id
        assert app_settings is settings
        assert attempt_id == live_attempt
        assert legacy_path.read_bytes() == legacy_bytes
        assert live_path.is_file()
        for target in orphan_targets:
            assert (settings.exports_dir / job_id / target).exists() is (
                not cleanup_supported
            )
        return result_from_artifact(
            live_payload,
            artifact_file_name=live_target,
        )

    _run_harmony_job(job_id, settings, already_published_processor, live_attempt)
    completed = db.get_job(settings.database_path, job_id)
    assert completed is not None
    assert completed["harmony_status"] == "completed"
    assert completed["harmony_artifact_file_name"] == live_target
    assert live_path.is_file()
    for target in orphan_targets:
        assert (settings.exports_dir / job_id / target).exists() is (
            not cleanup_supported
        )


def test_same_metadata_raw_evidence_replacement_invalidates_details_and_download(
    tmp_path: Path,
) -> None:
    settings = make_settings(tmp_path)
    app = create_app(settings=settings, harmony_processor=successful_processor)
    with TestClient(app) as client:
        job_id = create_job(settings)
        publish_harmony(settings, job_id)
        replacement = raw_payload()
        replacement["pitchedNoteEvents"][0].update(
            midiNote=62,
            midiPitch=62.12,
            frequencyHz=296.3,
            noteName="D4",
        )
        write_raw_transcription(job_id, settings, replacement)
        details = client.get(f"/api/jobs/{job_id}/harmony")
        download = client.get(f"/api/jobs/{job_id}/harmony/download")

    assert details.status_code == 500
    assert download.status_code == 500
    assert "validated" in details.json()["detail"].lower()
    assert str(settings.data_dir) not in json.dumps(details.json())


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
    assert "harmony_attempt_id" not in payload
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
