from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import soundfile as sf

from app import db
from app.config import Settings
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_NAME,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    REQUIRED_STEM_KINDS,
    STEM_MANIFEST_RELATIVE_PATH,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    load_raw_transcription,
)
from app import transcription_pipeline as pipeline

JOB_ID = "a" * 32
SAMPLE_RATE = 22_050
FIXED_TIME = "2026-08-06T02:00:00+00:00"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(
        data_dir=tmp_path / "data",
        allowed_hosts=("example.invalid",),
        max_duration_seconds=60,
        max_filesize_mb=32,
        max_upload_mb=32,
        audio_quality="192",
        audio_analysis_enabled=True,
        audio_analysis_version="baseline-librosa-v1",
    )
    value.ensure_directories()
    db.init_database(value.database_path)
    return value


def _tone(frequency: float, duration: float, amplitude: float = 0.18) -> np.ndarray:
    time = np.arange(int(SAMPLE_RATE * duration), dtype=np.float64) / SAMPLE_RATE
    return (amplitude * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


def _full_mix(duration: float = 2.0) -> np.ndarray:
    signal = _tone(440.0, duration)
    width = max(8, int(0.012 * SAMPLE_RATE))
    pulse = np.hanning(width).astype(np.float32) * 0.75
    for second in (0.25, 0.75, 1.25, 1.75):
        start = int(second * SAMPLE_RATE)
        end = min(signal.size, start + width)
        if start < signal.size:
            signal[start:end] += pulse[: end - start]
    return np.clip(signal, -0.95, 0.95)


def _timing() -> dict[str, Any]:
    return {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "beatConfidence": 0.9,
        "downbeatsSeconds": [0.0, 2.0],
        "meter": 4,
        "meterConfidence": 0.9,
    }


def _create_analyzed_job(
    settings: Settings,
    *,
    audio: np.ndarray | None = None,
    separation_status: str = "not_started",
) -> Path:
    db.create_job(
        settings.database_path,
        JOB_ID,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = settings.exports_dir / JOB_ID
    job_dir.mkdir(parents=True, exist_ok=True)
    signal = _full_mix() if audio is None else np.asarray(audio, dtype=np.float32)
    sf.write(job_dir / "analysis.wav", signal, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir()
    analysis = {
        "schemaVersion": 1,
        "analysisVersion": "baseline-librosa-v1",
        "createdAt": FIXED_TIME,
        "sourceAsset": "analysis.wav",
        "libraries": {},
        "audio": {
            "durationSeconds": len(signal) / SAMPLE_RATE,
            "sampleRate": SAMPLE_RATE,
            "channels": 1,
            "peakAmplitude": float(np.max(np.abs(signal))),
            "rms": float(np.sqrt(np.mean(np.square(signal), dtype=np.float64))),
            "rmsDbfs": -12.0,
            "silent": False,
        },
        "timing": _timing(),
        "tonality": {},
        "warnings": [],
    }
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(analysis, allow_nan=False), encoding="utf-8"
    )
    db.update_job(
        settings.database_path,
        JOB_ID,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        normalized_file_name="analysis.wav",
        analysis_status="completed",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at=FIXED_TIME,
        separation_status=separation_status,
    )
    return job_dir


def _publish_stems(settings: Settings, job_dir: Path) -> None:
    run_id = "b" * 32
    run_dir = job_dir / "stems" / "runs" / run_id
    run_dir.mkdir(parents=True)
    labels = {"vocals": "Vocals", "bass": "Bass", "drums": "Drums", "other": "Other"}
    frequencies = {"vocals": 440.0, "bass": 110.0, "drums": 180.0, "other": 330.0}
    stems = []
    for kind in REQUIRED_STEM_KINDS:
        path = run_dir / f"{kind}.wav"
        signal = _tone(frequencies[kind], 1.0)
        if kind == "drums":
            signal = _full_mix(1.0)
        sf.write(path, signal, SAMPLE_RATE, format="WAV", subtype="PCM_16")
        info = sf.info(path)
        stems.append(
            {
                "kind": kind,
                "label": labels[kind],
                "fileName": f"stems/runs/{run_id}/{kind}.wav",
                "durationSeconds": float(info.frames / info.samplerate),
                "sampleRate": int(info.samplerate),
                "channels": int(info.channels),
                "sizeBytes": path.stat().st_size,
            }
        )
    model = {
        "name": AUDITED_MODEL_NAME,
        "packageVersion": AUDITED_DEMUCS_VERSION,
        "runtimeProfile": "linux-x86_64-cpu-cpython313",
        "workerVersion": "1.0.0",
        "torchVersion": "2.13.0+cpu",
        "huggingfaceHubVersion": "1.26.0",
        "repository": AUDITED_MODEL_REPOSITORY,
        "revision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "weightsIdentifier": f"sha256:{AUDITED_CHECKPOINT_SHA256}",
        "device": "cpu",
    }
    manifest = {
        "schemaVersion": 3,
        "separationVersion": "demucs-worker-v3",
        "createdAt": FIXED_TIME,
        "sourceAsset": "analysis.wav",
        "runId": run_id,
        "model": model,
        "stems": stems,
        "warnings": [],
    }
    manifest_path = job_dir / STEM_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, allow_nan=False), encoding="utf-8")
    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="completed",
        separation_stage="completed",
        separation_progress=100,
        separation_version="demucs-worker-v3",
        separation_model=AUDITED_MODEL_NAME,
        stem_manifest_file_name=STEM_MANIFEST_RELATIVE_PATH,
        separated_at=FIXED_TIME,
    )


def _pitch_result(source_kind: str, *, start: float | None = None) -> dict[str, Any]:
    starts = {"vocals": 0.25, "bass": 0.25, "other": 0.25, "full_mix": 0.4}
    value = starts[source_kind] if start is None else start
    midi = {"vocals": 69, "bass": 45, "other": 64, "full_mix": 69}[source_kind]
    frequency = float(440.0 * (2.0 ** ((midi - 69) / 12.0)))
    return {
        "algorithmVersion": "baseline-pyin-v1",
        "sourceKind": source_kind,
        "events": [
            {
                "id": "p000001",
                "sourceKind": source_kind,
                "startSeconds": value,
                "endSeconds": value + 0.3,
                "midiNote": midi,
                "midiPitch": midi + 0.125,
                "frequencyHz": frequency,
                "noteName": {69: "A4", 45: "A2", 64: "E4"}[midi],
                "confidence": 0.88,
                "velocity": 90,
                "warnings": [],
                "rawFeatureSummary": {"existingEvidence": source_kind},
            }
        ],
        "warnings": [f"{source_kind} detector warning"],
        "diagnostics": {
            "pyin": {"frameCount": 12, "voicedFrameCount": 10},
            "eventing": {
                "eventEvidence": [
                    {
                        "eventId": "p000001",
                        "firstFrameIndex": 2,
                        "lastFrameIndex": 10,
                        "voicedFrameCount": 9,
                        "meanVoicedProbability": 0.91,
                        "pitchMadCents": 4.0,
                    }
                ]
            },
        },
    }


def _percussion_result(source_kind: str) -> dict[str, Any]:
    return {
        "algorithmVersion": "baseline-onset-bands-v1",
        "sourceKind": source_kind,
        "events": [
            {
                "id": "r000001",
                "sourceKind": source_kind,
                "timeSeconds": 0.5,
                "strength": 0.9,
                "hits": [
                    {"kind": "kick", "confidence": 0.8},
                    {"kind": "closed_hat", "confidence": 0.7},
                ],
                "rawFeatureSummary": {"spectralFlux": 0.75},
                "warnings": [],
            }
        ],
        "warnings": [],
        "diagnostics": {"eventsReturned": 1, "labelsAreBroad": True},
    }


def _empty_result(kind: str, event_type: str) -> dict[str, Any]:
    return {
        "algorithmVersion": (
            "baseline-pyin-v1" if event_type == "pitched" else "baseline-onset-bands-v1"
        ),
        "sourceKind": kind,
        "events": [],
        "warnings": ["No reliable events were detected."],
        "diagnostics": {"eventsReturned": 0},
    }


def test_no_stem_job_runs_real_detectors_alignment_and_publication(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    before = (job_dir / "analysis.wav").read_bytes()
    stages: list[tuple[str, float]] = []

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda stage, message, progress: stages.append((stage, progress)),
    )

    assert result.input_mode == "full_mix"
    assert result.artifact_file_name == RAW_TRANSCRIPTION_RELATIVE_PATH
    assert result.pitched_event_count >= 1
    assert result.percussion_event_count == len(result.payload["percussionEvents"])
    percussion_algorithm = result.payload["algorithms"]["percussionDetection"]
    assert percussion_algorithm["sources"]["full_mix"]["version"] == "baseline-onset-bands-v1"
    if result.percussion_event_count == 0:
        assert any("No reliable percussion onsets" in item for item in result.warnings)
    assert "sourceSeparation" not in result.payload
    assert result.payload["algorithms"]["transcriptionPipeline"]["demucsRequired"] is False
    assert (job_dir / "analysis.wav").read_bytes() == before
    assert load_raw_transcription(JOB_ID, settings) == result.payload
    ids = [item["id"] for item in result.payload["pitchedNoteEvents"]]
    ids += [item["id"] for item in result.payload["percussionEvents"]]
    assert len(ids) == len(set(ids))
    assert all("detectorEventId" in item["rawFeatureSummary"] for item in result.payload["pitchedNoteEvents"])
    assert [value for _, value in stages] == sorted(value for _, value in stages)
    assert max(value for _, value in stages) < 100


def test_valid_stems_use_all_sources_and_deterministic_unique_ids(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    monkeypatch.setattr(pipeline, "_utc_now", lambda: FIXED_TIME)
    pitch_calls: list[tuple[str, str]] = []
    percussion_calls: list[tuple[str, str]] = []

    def pitch_processor(path: Path, *, source_kind: str):
        pitch_calls.append((source_kind, path.name))
        return _pitch_result(source_kind)

    def percussion_processor(path: Path, *, source_kind: str):
        percussion_calls.append((source_kind, path.name))
        return _percussion_result(source_kind)

    first = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=pitch_processor,
        percussion_processor=percussion_processor,
    )
    second = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=pitch_processor,
        percussion_processor=percussion_processor,
    )

    assert first.input_mode == "separated_stems"
    assert first.payload == second.payload
    assert [item["sourceKind"] for item in first.payload["pitchedNoteEvents"]] == [
        "vocals",
        "bass",
        "other",
    ]
    assert [item["id"] for item in first.payload["pitchedNoteEvents"]] == [
        "p000001",
        "p000002",
        "p000003",
    ]
    for item in first.payload["pitchedNoteEvents"]:
        summary = item["rawFeatureSummary"]
        assert summary["detectorEventId"] == "p000001"
        assert summary["existingEvidence"] == item["sourceKind"]
        assert summary["firstFrameIndex"] == 2
    percussion = first.payload["percussionEvents"][0]
    assert percussion["id"] == "r000001"
    assert len(percussion["hits"]) == 2
    assert percussion["rawFeatureSummary"]["detectorEventId"] == "r000001"
    assert first.payload["sourceSeparation"]["model"]["name"] == AUDITED_MODEL_NAME
    assert pitch_calls[:3] == [("vocals", "vocals.wav"), ("bass", "bass.wav"), ("other", "other.wav")]
    assert percussion_calls[0] == ("drums", "drums.wav")


def test_partial_pitched_stem_failure_preserves_successes(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    calls: list[str] = []

    def pitch_processor(path: Path, *, source_kind: str):
        calls.append(source_kind)
        if source_kind == "vocals":
            raise RuntimeError("private failure /tmp/secret.wav")
        return _pitch_result(source_kind)

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=pitch_processor,
        percussion_processor=lambda path, *, source_kind: _percussion_result(source_kind),
    )

    assert calls == ["vocals", "bass", "other"]
    assert {item["sourceKind"] for item in result.payload["pitchedNoteEvents"]} == {"bass", "other"}
    assert any("vocals stem failed" in item for item in result.warnings)
    assert "/tmp" not in json.dumps(result.payload)


def test_all_pitched_stems_fail_then_full_mix_is_attempted_once(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    calls: list[str] = []

    def pitch_processor(path: Path, *, source_kind: str):
        calls.append(source_kind)
        if source_kind != "full_mix":
            raise RuntimeError("stem failed")
        return _pitch_result(source_kind)

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=pitch_processor,
        percussion_processor=lambda path, *, source_kind: _percussion_result(source_kind),
    )

    assert calls == ["vocals", "bass", "other", "full_mix"]
    assert result.input_mode == "stems_with_full_mix_fallback"
    assert [item["sourceKind"] for item in result.payload["pitchedNoteEvents"]] == ["full_mix"]
    assert "sourceSeparation" in result.payload


def test_drums_failure_falls_back_to_full_mix_once(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    calls: list[str] = []

    def percussion_processor(path: Path, *, source_kind: str):
        calls.append(source_kind)
        if source_kind == "drums":
            raise RuntimeError("drums failed")
        return _percussion_result(source_kind)

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=lambda path, *, source_kind: _pitch_result(source_kind),
        percussion_processor=percussion_processor,
    )

    assert calls == ["drums", "full_mix"]
    assert result.input_mode == "stems_with_full_mix_fallback"
    assert result.payload["percussionEvents"][0]["sourceKind"] == "full_mix"


def test_failed_separation_with_stale_manifest_uses_full_mix(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    db.update_job(settings.database_path, JOB_ID, separation_status="failed")
    pitch_calls: list[str] = []
    percussion_calls: list[str] = []

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=lambda path, *, source_kind: (
            pitch_calls.append(source_kind) or _pitch_result(source_kind)
        ),
        percussion_processor=lambda path, *, source_kind: (
            percussion_calls.append(source_kind) or _percussion_result(source_kind)
        ),
    )

    assert result.input_mode == "full_mix"
    assert pitch_calls == ["full_mix"]
    assert percussion_calls == ["full_mix"]
    assert "sourceSeparation" not in result.payload


def test_corrupt_published_manifest_falls_back_safely(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    _publish_stems(settings, job_dir)
    (job_dir / STEM_MANIFEST_RELATIVE_PATH).write_text("{broken", encoding="utf-8")

    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=lambda path, *, source_kind: _pitch_result(source_kind),
        percussion_processor=lambda path, *, source_kind: _percussion_result(source_kind),
    )

    assert result.input_mode == "full_mix"
    assert "sourceSeparation" not in result.payload
    assert any("failed current safety" in item for item in result.warnings)


def test_honest_empty_detector_results_publish_empty_artifact(settings: Settings) -> None:
    _create_analyzed_job(settings)
    result = pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=lambda path, *, source_kind: _empty_result(source_kind, "pitched"),
        percussion_processor=lambda path, *, source_kind: _empty_result(source_kind, "percussion"),
    )
    assert result.pitched_event_count == 0
    assert result.percussion_event_count == 0
    assert result.payload["alignmentCandidates"] == []
    assert any("without pitched-note" in item for item in result.warnings)


def test_all_detector_boundaries_failing_prevents_publication(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)

    def fail(*args, **kwargs):
        raise RuntimeError("private /home/user/recording.wav")

    with pytest.raises(pipeline.TranscriptionPipelineError, match="could not produce"):
        pipeline.transcribe_job(
            JOB_ID,
            settings,
            lambda *_: None,
            pitch_processor=fail,
            percussion_processor=fail,
        )
    assert not (job_dir / RAW_TRANSCRIPTION_RELATIVE_PATH).exists()


def test_missing_or_stale_required_analysis_fails_safely(settings: Settings) -> None:
    job_dir = _create_analyzed_job(settings)
    (job_dir / "analysis.wav").unlink()
    with pytest.raises(pipeline.TranscriptionPipelineError, match="Analysis audio is missing"):
        pipeline.transcribe_job(JOB_ID, settings, lambda *_: None)

    sf.write(job_dir / "analysis.wav", _tone(440.0, 1.0), SAMPLE_RATE)
    db.update_job(settings.database_path, JOB_ID, analysis_version="stale-v1")
    with pytest.raises(pipeline.TranscriptionPipelineError, match="provenance is stale"):
        pipeline.transcribe_job(JOB_ID, settings, lambda *_: None)


def test_processor_outputs_do_not_mutate_caller_owned_events(settings: Settings) -> None:
    _create_analyzed_job(settings)
    pitched = _pitch_result("full_mix")
    percussion = _percussion_result("full_mix")
    pitched_snapshot = copy.deepcopy(pitched)
    percussion_snapshot = copy.deepcopy(percussion)

    pipeline.transcribe_job(
        JOB_ID,
        settings,
        lambda *_: None,
        pitch_processor=lambda path, *, source_kind: pitched,
        percussion_processor=lambda path, *, source_kind: percussion,
    )

    assert pitched == pitched_snapshot
    assert percussion == percussion_snapshot
