from __future__ import annotations

import json
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable

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
    SeparationOptions,
    StemSeparationError,
    separate_stems,
)
from app.separation_artifacts import load_stem_details, resolve_stem_artifact
from app.separation_capability import (
    STATE_DOWNLOAD_REQUIRED,
    STATE_READY,
    STATE_RUNTIME_MISSING,
    STATE_UNAVAILABLE,
    probe_separation_capability,
)
from app.separation_runtime import (
    CHECKPOINT_SIZE_BYTES,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    RuntimeMissingError,
    SeparationRuntimeClient,
    WorkerCommandError,
    WorkerErrorDetail,
)

JOB_ID = "a" * 32
PROFILE = "linux-x86_64-cpu-cpython313"
WORKER_VERSION = "1.0.0"
TORCH_VERSION = "2.13.0+cpu"
HUB_VERSION = "1.26.0"
VERSIONS = {
    "demucs": AUDITED_DEMUCS_VERSION,
    "torch": TORCH_VERSION,
    "huggingface_hub": HUB_VERSION,
    "safetensors": "0.8.0",
    "PyYAML": "6.0.3",
}
SEPARATION_VERSION = "demucs-worker-v3"
SAMPLE_RATE = 8_000


@dataclass
class Completed:
    returncode: int
    stdout: bytes
    stderr: bytes = b""
    stdout_overflow: bool = False
    stderr_overflow: bool = False


@dataclass
class RuntimeFixture:
    client: SeparationRuntimeClient
    runner: "ProtocolRunner"
    worker: Path
    cache: Path
    runtime_lock: Path


class ProtocolRunner:
    """Queued protocol-v1 process runner with synthetic WAV side effects."""

    def __init__(self, *outcomes: Completed | BaseException | Callable[..., Completed]):
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []
        self.output_bytes: dict[str, dict[str, bytes]] = {}
        self._lock = threading.Lock()

    def __call__(self, argv, **kwargs):
        with self._lock:
            if not self.outcomes:
                raise AssertionError("unexpected worker invocation")
            outcome = self.outcomes.pop(0)
        values = list(argv)
        command = values[3] if len(values) > 3 else ""
        self.calls.append(
            {
                "argv": values,
                "command": command,
                "env": dict(kwargs["env"]),
                "timeout": kwargs["timeout"],
                "shell": kwargs["shell"],
                "capture_output": kwargs["capture_output"],
                "text": kwargs["text"],
                "check": kwargs["check"],
            }
        )
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(values, kwargs, self)
        return outcome


def _ok(command: str, result: dict[str, Any], warnings=()) -> Completed:
    return Completed(
        0,
        json.dumps(
            {
                "protocolVersion": 1,
                "command": command,
                "status": "ok",
                "result": result,
                "warnings": list(warnings),
            },
            allow_nan=False,
        ).encode("utf-8"),
    )


def _failure(
    exit_code: int,
    command: str,
    worker_code: str,
    *,
    message: str = "The worker command failed.",
    retryable: bool = True,
    stderr: str = "",
) -> Completed:
    return Completed(
        exit_code,
        json.dumps(
            {
                "protocolVersion": 1,
                "command": command,
                "status": "error",
                "error": {
                    "code": worker_code,
                    "message": message,
                    "retryable": retryable,
                },
                "warnings": [],
            },
            allow_nan=False,
        ).encode("utf-8"),
        stderr.encode("utf-8"),
    )


def _runtime_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtimeProfile": PROFILE,
        "workerVersion": WORKER_VERSION,
        "pythonVersion": "3.13.14",
        "runtimeLockSource": "profile",
        "installedVersions": dict(VERSIONS),
        "lockedVersions": dict(VERSIONS),
        "compatible": True,
    }
    result.update(overrides)
    return result


def _model_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtimeProfile": PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": AUDITED_DEMUCS_VERSION,
        "torchVersion": TORCH_VERSION,
        "huggingfaceHubVersion": HUB_VERSION,
        "modelRepository": AUDITED_MODEL_REPOSITORY,
        "modelRevision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "verifiedAt": "2026-08-04T00:00:00+00:00",
        "offlineReady": True,
        "readinessManifest": "readiness/htdemucs-bf35a81b-v1.json",
    }
    result.update(overrides)
    return result


def _separation_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtimeProfile": PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": AUDITED_DEMUCS_VERSION,
        "torchVersion": TORCH_VERSION,
        "huggingfaceHubVersion": HUB_VERSION,
        "modelRepository": AUDITED_MODEL_REPOSITORY,
        "modelRevision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "device": "cpu",
        "outputs": ["vocals.wav", "bass.wav", "drums.wav", "other.wav"],
    }
    result.update(overrides)
    return result


def _arg(argv: list[str], name: str) -> str:
    return argv[argv.index(name) + 1]


def _write_wav(path: Path, *, frequency: float) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(SAMPLE_RATE // 20, dtype=np.float32) / SAMPLE_RATE
    mono = 0.1 * np.sin(2 * np.pi * frequency * time)
    stereo = np.column_stack((mono, mono))
    sf.write(path, stereo, SAMPLE_RATE, format="WAV", subtype="PCM_16")
    return path.read_bytes()


def _successful_separation(
    result: dict[str, Any] | None = None,
) -> Callable[..., Completed]:
    def outcome(argv: list[str], kwargs: dict[str, Any], runner: ProtocolRunner) -> Completed:
        workspace = Path(_arg(argv, "--workspace-root"))
        output_relative = _arg(argv, "--output-relative")
        output = workspace.joinpath(*PurePosixPath(output_relative).parts)
        run_id = PurePosixPath(output_relative).parts[2]
        written: dict[str, bytes] = {}
        for index, kind in enumerate(REQUIRED_STEM_KINDS):
            written[kind] = _write_wav(
                output / f"{kind}.wav",
                frequency=220.0 + (index * 55.0),
            )
        runner.output_bytes[run_id] = written
        return _ok("separate", dict(result or _separation_result()))

    return outcome


def _settings(tmp_path: Path) -> Settings:
    value = Settings(
        data_dir=tmp_path / "data",
        allowed_hosts=("example.com",),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=500,
        audio_quality="192",
    )
    value.ensure_directories()
    return value


def _runtime(tmp_path: Path, *outcomes) -> RuntimeFixture:
    cache = tmp_path / "runtime-cache"
    cache.mkdir(parents=True, exist_ok=True)
    worker = tmp_path / "runtime" / "bin" / "popex-demucs-worker"
    runtime_lock = tmp_path / "runtime" / "worker-runtime-lock.json"
    runtime_lock.parent.mkdir(parents=True, exist_ok=True)
    runtime_lock.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "runtimeProfile": PROFILE,
                "workerVersion": WORKER_VERSION,
                "packages": VERSIONS,
            }
        ),
        encoding="utf-8",
    )
    runner = ProtocolRunner(*outcomes)
    client = SeparationRuntimeClient(
        worker,
        cache,
        runtime_lock_path=runtime_lock,
        expected_runtime_profile=PROFILE,
        process_runner=runner,
    )
    return RuntimeFixture(client, runner, worker, cache, runtime_lock)


def _prepare_job(settings: Settings) -> dict[str, Any]:
    db.init_database(settings.database_path)
    db.create_job(
        settings.database_path,
        JOB_ID,
        source_type="upload",
        original_filename="synthetic.wav",
    )
    job_dir = settings.exports_dir / JOB_ID
    source = job_dir / "source" / "synthetic.wav"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"synthetic source")
    _write_wav(job_dir / "analysis.wav", frequency=110.0)
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
    (job_dir / "analysis").mkdir()
    (job_dir / "analysis" / "audio-analysis.json").write_text(
        "{}", encoding="utf-8"
    )
    db.update_job(
        settings.database_path,
        JOB_ID,
        status="completed",
        stage="completed",
        progress=100,
        message="Prepared source and analysis are available.",
        source_file_name="source/synthetic.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        preparation_status="completed",
        analysis_status="completed",
        analysis_version="baseline-librosa-v1",
        tempo_bpm=120.0,
        tempo_confidence=0.8,
        key_symbol="C major",
        key_confidence=0.7,
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-04T00:00:00+00:00",
        analysis_error=None,
    )
    job = db.get_job(settings.database_path, JOB_ID)
    assert job is not None
    return job


def _options(runtime: RuntimeFixture) -> SeparationOptions:
    return SeparationOptions(
        separation_version=SEPARATION_VERSION,
        worker_runner=runtime.client,
        cache_root=runtime.cache,
        expected_model_repository=AUDITED_MODEL_REPOSITORY,
        expected_model_revision=AUDITED_MODEL_REVISION,
        expected_checkpoint_file=AUDITED_CHECKPOINT_FILE,
        expected_checkpoint_sha256=AUDITED_CHECKPOINT_SHA256,
        expected_demucs_version=AUDITED_DEMUCS_VERSION,
        expected_runtime_profile=PROFILE,
        device="cpu",
        timeout_seconds=15,
    )


def _claim(settings: Settings) -> bool:
    return db.claim_separation_attempt(
        settings.database_path,
        JOB_ID,
        separation_version=SEPARATION_VERSION,
        separation_model=AUDITED_MODEL_NAME,
    )


def _persist_success(settings: Settings, result) -> dict[str, Any]:
    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="completed",
        separation_stage="completed",
        separation_progress=100,
        separation_message="Stem separation completed.",
        separation_version=result.separation_version,
        separation_model=result.model_name,
        stem_manifest_file_name=result.manifest_file_name,
        separated_at=result.created_at,
        separation_error=None,
    )
    job = db.get_job(settings.database_path, JOB_ID)
    assert job is not None
    return job


def _persist_failure(settings: Settings, error: BaseException) -> dict[str, Any]:
    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="failed",
        separation_stage="failed",
        separation_message="Stem separation failed; prepared audio remains available.",
        separation_error=str(error),
    )
    job = db.get_job(settings.database_path, JOB_ID)
    assert job is not None
    return job


def _run_and_persist(
    settings: Settings,
    runtime: RuntimeFixture,
    *,
    callback=None,
):
    stages: list[tuple[str, str, float]] = []

    def record(stage: str, message: str, progress: float) -> None:
        stages.append((stage, message, progress))
        db.update_job(
            settings.database_path,
            JOB_ID,
            separation_stage=stage,
            separation_progress=progress,
            separation_message=message,
        )
        if callback is not None:
            callback(stage, message, progress)

    result = separate_stems(
        JOB_ID,
        settings,
        _options(runtime),
        stage_callback=record,
    )
    return result, _persist_success(settings, result), stages


def _assert_runtime_lock_boundary(runtime: RuntimeFixture) -> None:
    lock_payload = json.loads(runtime.runtime_lock.read_text(encoding="utf-8"))
    assert lock_payload == {
        "schemaVersion": 1,
        "runtimeProfile": PROFILE,
        "workerVersion": WORKER_VERSION,
        "packages": VERSIONS,
    }
    for call in runtime.runner.calls:
        assert call["argv"][0] == str(runtime.worker)
        assert call["argv"][1:3] == ["--protocol-version", "1"]
        assert call["shell"] is False
        assert call["capture_output"] is True
        assert call["text"] is False
        assert call["check"] is False
        assert call["env"]["POPEX_DEMUCS_RUNTIME_LOCK"] == str(
            runtime.runtime_lock
        )
        assert call["env"]["HF_HOME"] == str(runtime.cache)


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _assert_no_private_paths(value: object, runtime: RuntimeFixture, workspace: Path) -> None:
    encoded = json.dumps(value, default=str, ensure_ascii=False)
    for path in (
        runtime.worker,
        runtime.cache,
        runtime.runtime_lock,
        workspace,
    ):
        assert str(path) not in encoded
    assert "readiness/htdemucs" not in encoded


def test_ready_pipeline_composes_through_artifact_resolution(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    baseline = _prepare_job(settings)
    runtime = _runtime(
        tmp_path,
        _ok("runtime-probe", _runtime_result()),
        _ok("model-probe", _model_result()),
        _successful_separation(),
    )

    capability = probe_separation_capability(
        runtime.client,
        enabled=True,
        device="cpu",
    )
    assert capability is not None
    assert capability.state == STATE_READY
    assert capability.actionable is True
    assert capability.network_required is False
    assert capability.profile == PROFILE
    assert _claim(settings) is True
    assert _claim(settings) is False

    result, job, stages = _run_and_persist(settings, runtime)

    assert [item[0] for item in stages] == [
        "preparing_separation",
        "separating_stems",
        "validating_stems",
        "saving_stems",
    ]
    assert all(0 < item[2] < 100 for item in stages)
    assert [call["command"] for call in runtime.runner.calls] == [
        "runtime-probe",
        "model-probe",
        "separate",
    ]
    _assert_runtime_lock_boundary(runtime)

    separation_call = runtime.runner.calls[-1]
    assert _arg(separation_call["argv"], "--cache-root") == str(runtime.cache)
    workspace = (settings.exports_dir / JOB_ID).resolve()
    assert _arg(separation_call["argv"], "--workspace-root") == str(workspace)
    assert _arg(separation_call["argv"], "--input-relative") == "analysis.wav"
    assert _arg(separation_call["argv"], "--output-relative") == (
        f"stems/runs/{result.run_id}/worker-output"
    )
    assert separation_call["env"]["HF_HUB_OFFLINE"] == "1"

    assert result.payload["schemaVersion"] == 3
    assert result.payload["model"]["repository"] == AUDITED_MODEL_REPOSITORY
    assert result.payload["model"]["revision"] == AUDITED_MODEL_REVISION
    assert result.payload["model"]["checkpointFile"] == AUDITED_CHECKPOINT_FILE
    assert result.payload["model"]["checkpointSha256"] == AUDITED_CHECKPOINT_SHA256
    assert result.payload["model"]["runtimeProfile"] == PROFILE
    assert job["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH
    assert job["separated_at"] == result.created_at

    for stem in result.stems:
        stored = workspace / stem.file_name
        assert stored.read_bytes() == runtime.runner.output_bytes[result.run_id][stem.kind]

    details = load_stem_details(JOB_ID, settings, job)
    assert details.available is True
    assert details.status == "completed"
    assert details.model == AUDITED_MODEL_NAME
    assert details.version == SEPARATION_VERSION
    assert len(details.stems) == 4
    for kind in REQUIRED_STEM_KINDS:
        resolved = resolve_stem_artifact(JOB_ID, kind, settings, job)
        assert resolved.path.read_bytes() == runtime.runner.output_bytes[result.run_id][kind]
        assert resolved.file_name == f"stems/runs/{result.run_id}/{kind}.wav"

    for key in (
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "tempo_bpm",
        "key_symbol",
        "analysis_json_file_name",
    ):
        assert job[key] == baseline[key]
    _assert_no_private_paths(capability.runtime_payload(), runtime, workspace)
    _assert_no_private_paths(result.payload, runtime, workspace)
    _assert_no_private_paths(details.payload(), runtime, workspace)
