from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier, Lock

import numpy as np
import pytest
import soundfile as sf

from app import db
from app.config import Settings
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    STEM_MANIFEST_RELATIVE_PATH,
    StemSeparationError,
)
from app.separation_runtime import (
    CHECKPOINT_SIZE_BYTES,
    ModelPreparationResult,
    ModelProbeResult,
    RuntimeProbeResult,
    WorkerCommandError,
    WorkerErrorDetail,
)
from app.separation_service import (
    SeparationService,
    SeparationStartConflict,
)

JOB_ID = "a" * 32
SECOND_JOB_ID = "b" * 32
SAMPLE_RATE = 8_000


def make_settings(tmp_path: Path, *, enabled: bool = True) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com",),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=500,
        audio_quality="192",
        ffmpeg_binary="missing-test-ffmpeg",
        ffprobe_binary="missing-test-ffprobe",
        stem_separation_enabled=enabled,
        stem_separation_worker_executable=tmp_path / "bin" / "worker",
        stem_separation_runtime_lock=tmp_path / "runtime-lock.json",
        stem_separation_cache_dir=tmp_path / "runtime-cache",
        stem_separation_runtime_profile="linux-cpu-v1",
        stem_separation_device="cpu",
        stem_separation_timeout_seconds=30,
    )


def create_ready_job(settings: Settings, job_id: str = JOB_ID) -> dict:
    settings.ensure_directories()
    db.init_database(settings.database_path)
    record = db.create_job(
        settings.database_path,
        job_id,
        source_type="upload",
        original_filename="song.wav",
    )
    job_dir = settings.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    sf.write(
        job_dir / "analysis.wav",
        np.zeros((SAMPLE_RATE // 10, 2), dtype=np.float32),
        SAMPLE_RATE,
        subtype="PCM_16",
    )
    (job_dir / "source.wav").write_bytes(b"source")
    db.update_job(
        settings.database_path,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        message="Audio analysis complete.",
        source_file_name="source.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        preparation_status="completed",
        analysis_status="completed",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-04T00:00:00+00:00",
    )
    return db.get_job(settings.database_path, job_id) or record


class FakeRuntimeClient:
    def __init__(
        self,
        *,
        ready: bool = True,
        fail_separation: bool = False,
        operation_delay: float = 0,
    ):
        self.ready = ready
        self.fail_separation = fail_separation
        self.operation_delay = operation_delay
        self.runtime_probe_calls = 0
        self.model_probe_calls = 0
        self.prepare_calls = 0
        self.separate_calls = 0
        self.active_operations = 0
        self.max_active_operations = 0
        self._counter_lock = Lock()

    def _enter_operation(self) -> None:
        with self._counter_lock:
            self.active_operations += 1
            self.max_active_operations = max(
                self.max_active_operations,
                self.active_operations,
            )

    def _leave_operation(self) -> None:
        with self._counter_lock:
            self.active_operations -= 1

    def runtime_probe(self) -> RuntimeProbeResult:
        with self._counter_lock:
            self.runtime_probe_calls += 1
        return RuntimeProbeResult(
            runtime_profile="linux-cpu-v1",
            worker_version="1.0.0",
            python_version="3.13.14",
            runtime_lock_source="profile",
            demucs_version=AUDITED_DEMUCS_VERSION,
            torch_version="2.13.0+cpu",
            huggingface_hub_version="1.16.1",
            safetensors_version="0.6.2",
            pyyaml_version="6.0.3",
        )

    def model_probe(self) -> ModelProbeResult:
        with self._counter_lock:
            self.model_probe_calls += 1
            ready = self.ready
        if not ready:
            raise WorkerCommandError(
                WorkerErrorDetail(
                    code="MODEL_DOWNLOAD_REQUIRED",
                    message="The verified model is not prepared.",
                    retryable=True,
                    exit_code=20,
                    worker_code="MODEL_DOWNLOAD_REQUIRED",
                )
            )
        return self._model_result(ModelProbeResult)

    def prepare_model(self, *, allow_model_download: bool) -> ModelPreparationResult:
        assert allow_model_download is True
        self._enter_operation()
        try:
            with self._counter_lock:
                self.prepare_calls += 1
            if self.operation_delay:
                time.sleep(self.operation_delay)
            with self._counter_lock:
                self.ready = True
            return self._model_result(ModelPreparationResult)
        finally:
            self._leave_operation()

    def __call__(
        self,
        *,
        workspace_root: Path,
        cache_root: Path,
        input_relative: str,
        output_relative: str,
        device: str,
        timeout_seconds: float,
    ):
        self._enter_operation()
        try:
            with self._counter_lock:
                self.separate_calls += 1
            assert input_relative == "analysis.wav"
            assert cache_root.is_dir()
            assert device == "cpu"
            assert timeout_seconds == 30
            if self.operation_delay:
                time.sleep(self.operation_delay)
            if self.fail_separation:
                raise RuntimeError(f"worker failed at {cache_root}")
            output = workspace_root / output_relative
            output.mkdir(parents=True, exist_ok=True)
            for index, kind in enumerate(("vocals", "bass", "drums", "other"), 1):
                sf.write(
                    output / f"{kind}.wav",
                    np.full(
                        (SAMPLE_RATE // 20, 2),
                        index / 10,
                        dtype=np.float32,
                    ),
                    SAMPLE_RATE,
                    subtype="PCM_16",
                )
            return {
                "runtimeProfile": "linux-cpu-v1",
                "workerVersion": "1.0.0",
                "demucsVersion": AUDITED_DEMUCS_VERSION,
                "torchVersion": "2.13.0+cpu",
                "huggingfaceHubVersion": "1.16.1",
                "modelRepository": AUDITED_MODEL_REPOSITORY,
                "modelRevision": AUDITED_MODEL_REVISION,
                "checkpointFile": AUDITED_CHECKPOINT_FILE,
                "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
                "device": "cpu",
                "outputs": ["vocals.wav", "bass.wav", "drums.wav", "other.wav"],
            }
        finally:
            self._leave_operation()

    @staticmethod
    def _model_result(result_type):
        return result_type(
            runtime_profile="linux-cpu-v1",
            worker_version="1.0.0",
            demucs_version=AUDITED_DEMUCS_VERSION,
            torch_version="2.13.0+cpu",
            huggingface_hub_version="1.16.1",
            model_repository=AUDITED_MODEL_REPOSITORY,
            model_revision=AUDITED_MODEL_REVISION,
            checkpoint_file=AUDITED_CHECKPOINT_FILE,
            checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
            checkpoint_sha256=AUDITED_CHECKPOINT_SHA256,
            verified_at="2026-08-04T00:00:00+00:00",
            offline_ready=True,
        )


def run_scheduled(scheduled: list[tuple]) -> None:
    function, *arguments = scheduled[0]
    function(*arguments)


def test_disabled_service_omits_summary_and_performs_zero_runtime_calls(tmp_path: Path):
    settings = make_settings(tmp_path, enabled=False)
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)
    job = create_ready_job(settings)

    assert service.initialize() is None
    assert service.serialize_job(job) is None
    assert runtime.runtime_probe_calls == 0
    assert runtime.model_probe_calls == 0


def test_ready_summary_is_cached_and_uses_exact_frontend_shape(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)
    job = create_ready_job(settings)

    service.initialize()
    first = service.serialize_job(job)
    second = service.serialize_job(job)

    assert first == second
    assert set(first or {}) == {
        "enabled",
        "status",
        "stage",
        "progress",
        "message",
        "model",
        "version",
        "separatedAt",
        "canStart",
        "startUrl",
        "detailsUrl",
        "error",
        "runtime",
    }
    assert first["runtime"]["state"] == "ready"
    assert first["canStart"] is True
    assert runtime.runtime_probe_calls == 1
    assert runtime.model_probe_calls == 1


def test_download_required_without_consent_never_claims_or_schedules(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    scheduled = []

    with pytest.raises(SeparationStartConflict, match="consent"):
        service.request_start(
            JOB_ID,
            allow_model_download=False,
            schedule=lambda *args: scheduled.append(args),
        )

    record = db.get_job(settings.database_path, JOB_ID)
    assert record["separation_status"] == "not_started"
    assert runtime.prepare_calls == 0
    assert runtime.separate_calls == 0
    assert scheduled == []


def test_consented_first_use_claims_prepares_and_separates_once(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    scheduled = []

    claimed = service.request_start(
        JOB_ID,
        allow_model_download=True,
        schedule=lambda *args: scheduled.append(args),
    )
    assert claimed["separation_status"] == "processing"
    assert len(scheduled) == 1

    run_scheduled(scheduled)

    record = db.get_job(settings.database_path, JOB_ID)
    assert runtime.prepare_calls == 1
    assert runtime.separate_calls == 1
    assert record["separation_status"] == "completed"
    assert record["separation_progress"] == 100
    assert record["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH
    assert record["separated_at"]


def test_ready_request_does_not_prepare_even_with_redundant_true(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=True)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    scheduled = []

    service.request_start(
        JOB_ID,
        allow_model_download=True,
        schedule=lambda *args: scheduled.append(args),
    )
    run_scheduled(scheduled)

    assert runtime.prepare_calls == 0
    assert runtime.separate_calls == 1


def test_two_concurrent_requests_schedule_exactly_one_attempt(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    barrier = Barrier(2)
    scheduled = []

    def start() -> str:
        barrier.wait(timeout=5)
        try:
            service.request_start(
                JOB_ID,
                allow_model_download=False,
                schedule=lambda *args: scheduled.append(args),
            )
            return "claimed"
        except SeparationStartConflict:
            return "rejected"

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda _: start(), range(2)))

    assert sorted(results) == ["claimed", "rejected"]
    assert len(scheduled) == 1


def test_callback_progress_never_persists_100_before_success(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()

    def failing_processor(job_id, settings, options, *, stage_callback=None):
        stage_callback("separating_stems", "Separating stems.", 100)
        raise StemSeparationError("Synthetic failure")

    service = SeparationService(
        settings,
        runtime_client=runtime,
        processor=failing_processor,
    )
    create_ready_job(settings)
    scheduled = []
    service.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    run_scheduled(scheduled)

    record = db.get_job(settings.database_path, JOB_ID)
    assert record["separation_status"] == "failed"
    assert record["separation_progress"] == 99


def test_failed_retry_preserves_previous_manifest_stems_and_analysis(tmp_path: Path):
    settings = make_settings(tmp_path)
    first_runtime = FakeRuntimeClient()
    first = SeparationService(settings, runtime_client=first_runtime)
    original = create_ready_job(settings)
    scheduled = []
    first.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    run_scheduled(scheduled)
    success = db.get_job(settings.database_path, JOB_ID)
    job_dir = settings.exports_dir / JOB_ID
    manifest = job_dir / STEM_MANIFEST_RELATIVE_PATH
    manifest_bytes = manifest.read_bytes()
    stem_paths = sorted((job_dir / "stems" / "runs").glob("*/*.wav"))
    stem_bytes = {path: path.read_bytes() for path in stem_paths}

    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="failed",
        separation_stage="failed",
        separation_progress=50,
        separation_error="Retryable failure.",
    )
    failing_runtime = FakeRuntimeClient(fail_separation=True)
    retry = SeparationService(settings, runtime_client=failing_runtime)
    scheduled = []
    retry.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    run_scheduled(scheduled)

    failed = db.get_job(settings.database_path, JOB_ID)
    assert failed["separation_status"] == "failed"
    assert failed["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH
    assert failed["separated_at"] == success["separated_at"]
    assert failed["preparation_status"] == original["preparation_status"]
    assert failed["analysis_status"] == original["analysis_status"]
    assert failed["analysis_json_file_name"] == original["analysis_json_file_name"]
    assert manifest.read_bytes() == manifest_bytes
    assert {path: path.read_bytes() for path in stem_paths} == stem_bytes


def test_disabled_service_does_not_create_default_cache(tmp_path: Path):
    settings = replace(
        make_settings(tmp_path, enabled=False),
        stem_separation_cache_dir=None,
    )
    service = SeparationService(settings, runtime_client=FakeRuntimeClient())

    assert service.initialize() is None
    assert not (tmp_path / "runtime-cache" / "demucs").exists()


def test_fresh_default_cache_reaches_download_consent_and_prepares_once(tmp_path: Path):
    settings = replace(
        make_settings(tmp_path),
        stem_separation_cache_dir=None,
    )
    runtime = FakeRuntimeClient(ready=False)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    cache_root = tmp_path / "runtime-cache" / "demucs"

    capability = service.initialize()
    assert cache_root.is_dir()
    assert capability.state == "download_required"

    scheduled = []
    service.request_start(
        JOB_ID,
        allow_model_download=True,
        schedule=lambda *args: scheduled.append(args),
    )
    run_scheduled(scheduled)

    assert runtime.prepare_calls == 1
    assert runtime.separate_calls == 1
    assert db.get_job(settings.database_path, JOB_ID)["separation_status"] == "completed"


def test_unsafe_cache_path_becomes_unavailable_without_startup_failure(tmp_path: Path):
    settings = make_settings(tmp_path)
    unsafe_cache = tmp_path / "runtime-cache"
    unsafe_cache.write_text("not a directory", encoding="utf-8")
    service = SeparationService(settings, runtime_client=FakeRuntimeClient())

    capability = service.initialize()

    assert capability.state == "unavailable"


def test_worker_slot_serializes_two_different_ready_jobs(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(operation_delay=0.05)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings, JOB_ID)
    create_ready_job(settings, SECOND_JOB_ID)
    scheduled: list[tuple] = []

    for job_id in (JOB_ID, SECOND_JOB_ID):
        service.request_start(
            job_id,
            allow_model_download=False,
            schedule=lambda *args: scheduled.append(args),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda call: call[0](*call[1:]), scheduled))

    assert runtime.separate_calls == 2
    assert runtime.max_active_operations == 1
    assert db.get_job(settings.database_path, JOB_ID)["separation_status"] == "completed"
    assert db.get_job(settings.database_path, SECOND_JOB_ID)["separation_status"] == "completed"


def test_two_first_use_jobs_prepare_model_only_once(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False, operation_delay=0.05)
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings, JOB_ID)
    create_ready_job(settings, SECOND_JOB_ID)
    scheduled: list[tuple] = []

    for job_id in (JOB_ID, SECOND_JOB_ID):
        service.request_start(
            job_id,
            allow_model_download=True,
            schedule=lambda *args: scheduled.append(args),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda call: call[0](*call[1:]), scheduled))

    assert runtime.prepare_calls == 1
    assert runtime.separate_calls == 2
    assert runtime.max_active_operations == 1


def test_invalid_enabled_settings_are_safe_unavailable(tmp_path: Path):
    settings = replace(
        make_settings(tmp_path),
        stem_separation_device="invalid-device",
    )
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)

    capability = service.initialize()

    assert capability.state == "unavailable"
    assert runtime.runtime_probe_calls == 0
    assert runtime.model_probe_calls == 0


def test_unknown_persisted_status_is_non_actionable_and_not_claimed(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)
    create_ready_job(settings)
    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="paused_future_state",
        separation_stage="paused",
        separation_message="Future state",
    )
    record = db.get_job(settings.database_path, JOB_ID)

    summary = service.serialize_job(record)
    assert summary["status"] == "failed"
    assert summary["stage"] == "failed"
    assert summary["canStart"] is False
    assert summary["startUrl"] is None
    assert "state is unavailable" in summary["message"]

    scheduled = []
    with pytest.raises(SeparationStartConflict, match="state is unavailable"):
        service.request_start(
            JOB_ID,
            allow_model_download=False,
            schedule=lambda *args: scheduled.append(args),
        )
    assert scheduled == []
    assert db.get_job(settings.database_path, JOB_ID)["separation_status"] == "paused_future_state"


def test_one_shot_callback_persistence_failure_becomes_retryable(
    tmp_path: Path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    service = SeparationService(settings, runtime_client=runtime)
    original = create_ready_job(settings)
    scheduled = []
    service.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    real_update = db.update_job
    failed_once = False

    def flaky_update(database_path, job_id, **fields):
        nonlocal failed_once
        if not failed_once and fields.get("separation_stage") == "separating_stems":
            failed_once = True
            raise OSError("one-shot callback persistence failure")
        return real_update(database_path, job_id, **fields)

    monkeypatch.setattr(db, "update_job", flaky_update)
    run_scheduled(scheduled)

    record = db.get_job(settings.database_path, JOB_ID)
    assert failed_once
    assert record["separation_status"] == "failed"
    assert record["preparation_status"] == original["preparation_status"]
    assert record["analysis_status"] == original["analysis_status"]


def test_one_shot_failure_recording_is_retried(
    tmp_path: Path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(fail_separation=True)
    service = SeparationService(settings, runtime_client=runtime)
    original = create_ready_job(settings)
    scheduled = []
    service.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    real_update = db.update_job
    failed_once = False

    def flaky_update(database_path, job_id, **fields):
        nonlocal failed_once
        if not failed_once and fields.get("separation_status") == "failed":
            failed_once = True
            raise OSError("one-shot failure persistence error")
        return real_update(database_path, job_id, **fields)

    monkeypatch.setattr(db, "update_job", flaky_update)
    run_scheduled(scheduled)

    record = db.get_job(settings.database_path, JOB_ID)
    assert failed_once
    assert record["separation_status"] == "failed"
    assert record["preparation_status"] == original["preparation_status"]
    assert record["analysis_status"] == original["analysis_status"]


def test_completion_persistence_failure_restores_prior_manifest_and_retry_state(
    tmp_path: Path,
    monkeypatch,
):
    settings = make_settings(tmp_path)
    first_runtime = FakeRuntimeClient()
    first_service = SeparationService(settings, runtime_client=first_runtime)
    original = create_ready_job(settings)
    first_scheduled = []
    first_service.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: first_scheduled.append(args),
    )
    run_scheduled(first_scheduled)
    prior = db.get_job(settings.database_path, JOB_ID)
    job_dir = settings.exports_dir / JOB_ID
    manifest = job_dir / STEM_MANIFEST_RELATIVE_PATH
    prior_manifest = manifest.read_bytes()
    prior_stems = {
        path: path.read_bytes()
        for path in sorted((job_dir / "stems" / "runs").glob("*/*.wav"))
    }

    db.update_job(
        settings.database_path,
        JOB_ID,
        separation_status="failed",
        separation_stage="failed",
        separation_progress=40,
        separation_error="Retry requested.",
    )
    retry_service = SeparationService(
        settings,
        runtime_client=FakeRuntimeClient(),
    )
    scheduled = []
    retry_service.request_start(
        JOB_ID,
        allow_model_download=False,
        schedule=lambda *args: scheduled.append(args),
    )
    real_update = db.update_job
    failed_once = False

    def flaky_update(database_path, job_id, **fields):
        nonlocal failed_once
        if not failed_once and fields.get("separation_status") == "completed":
            failed_once = True
            raise OSError("one-shot completion persistence error")
        return real_update(database_path, job_id, **fields)

    monkeypatch.setattr(db, "update_job", flaky_update)
    run_scheduled(scheduled)

    record = db.get_job(settings.database_path, JOB_ID)
    assert failed_once
    assert record["separation_status"] == "failed"
    assert record["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH
    assert record["separated_at"] == prior["separated_at"]
    assert record["preparation_status"] == original["preparation_status"]
    assert record["analysis_status"] == original["analysis_status"]
    assert manifest.read_bytes() == prior_manifest
    assert {path: path.read_bytes() for path in prior_stems} == prior_stems
