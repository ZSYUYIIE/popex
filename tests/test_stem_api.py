from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from app import db
from app.config import Settings
from app.main import create_app
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    STEM_MANIFEST_RELATIVE_PATH,
)
from app.separation_runtime import (
    CHECKPOINT_SIZE_BYTES,
    ModelPreparationResult,
    ModelProbeResult,
    RuntimeProbeResult,
    WorkerCommandError,
    WorkerErrorDetail,
)

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
        stem_separation_worker_executable=(
            tmp_path / "bin" / "worker" if enabled else None
        ),
        stem_separation_runtime_lock=(
            tmp_path / "runtime-lock.json" if enabled else None
        ),
        stem_separation_cache_dir=(
            tmp_path / "runtime-cache" if enabled else None
        ),
        stem_separation_runtime_profile=("linux-cpu-v1" if enabled else None),
        stem_separation_device="cpu",
        stem_separation_timeout_seconds=30,
    )


def create_ready_job(settings: Settings, job_id: str) -> dict:
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
    (job_dir / "metadata.json").write_text("{}", encoding="utf-8")
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
    def __init__(self, *, ready: bool = True):
        self.ready = ready
        self.fail_separation = False
        self.runtime_probe_calls = 0
        self.model_probe_calls = 0
        self.prepare_calls = 0
        self.separate_calls = 0

    def runtime_probe(self) -> RuntimeProbeResult:
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
        self.model_probe_calls += 1
        if not self.ready:
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
        self.prepare_calls += 1
        self.ready = True
        return self._model_result(ModelPreparationResult)

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
        self.separate_calls += 1
        if self.fail_separation:
            raise RuntimeError(f"worker failure in {cache_root}")
        output = workspace_root / output_relative
        output.mkdir(parents=True, exist_ok=True)
        for index, kind in enumerate(("vocals", "bass", "drums", "other"), 1):
            sf.write(
                output / f"{kind}.wav",
                np.full((SAMPLE_RATE // 20, 2), index / 10, dtype=np.float32),
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


def test_disabled_app_preserves_existing_shape_and_never_probes(tmp_path: Path):
    settings = make_settings(tmp_path, enabled=False)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "1" * 32)
        payload = client.get(f"/api/jobs/{'1' * 32}").json()
        health = client.get("/api/health").json()

    assert "separation" not in payload
    assert runtime.runtime_probe_calls == 0
    assert runtime.model_probe_calls == 0
    assert "separation" not in health


def test_enabled_missing_runtime_is_optional_and_health_remains_available(tmp_path: Path):
    settings = make_settings(tmp_path)
    settings = Settings(
        **{
            **settings.__dict__,
            "stem_separation_worker_executable": None,
            "stem_separation_runtime_lock": None,
        }
    )
    app = create_app(settings=settings)

    with TestClient(app) as client:
        create_ready_job(settings, "2" * 32)
        payload = client.get(f"/api/jobs/{'2' * 32}").json()
        health = client.get("/api/health").json()

    assert payload["separation"]["runtime"]["state"] == "runtime_missing"
    assert payload["separation"]["canStart"] is False
    assert health["status"] in {"ok", "degraded"}


def test_enabled_job_summary_uses_cached_capability_and_exact_casing(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "3" * 32)
        create_ready_job(settings, "4" * 32)
        jobs = client.get("/api/jobs").json()
        client.get("/api/jobs")

    summary = jobs[0]["separation"]
    assert set(summary) == {
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
    assert summary["runtime"]["state"] == "ready"
    assert runtime.runtime_probe_calls == 1
    assert runtime.model_probe_calls == 1


@pytest.mark.parametrize("bad", ["true", 1, 0, None])
def test_start_consent_field_is_strict_boolean(tmp_path: Path, bad):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False)
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "5" * 32)
        response = client.post(
            f"/api/jobs/{'5' * 32}/separate",
            json={"allowModelDownload": bad},
        )

    assert response.status_code == 422
    assert runtime.prepare_calls == 0
    assert runtime.separate_calls == 0


def test_download_required_needs_consent_before_claim(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False)
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "6" * 32)
        response = client.post(
            f"/api/jobs/{'6' * 32}/separate",
            json={"allowModelDownload": False},
        )

    record = db.get_job(settings.database_path, "6" * 32)
    assert response.status_code == 409
    assert record["separation_status"] == "not_started"
    assert runtime.prepare_calls == 0


def test_consented_first_use_prepares_and_separates(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=False)
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "7" * 32)
        response = client.post(
            f"/api/jobs/{'7' * 32}/separate",
            json={"allowModelDownload": True},
        )
        completed = client.get(f"/api/jobs/{'7' * 32}").json()

    assert response.status_code == 202
    assert response.json()["separation"]["status"] == "processing"
    assert completed["separation"]["status"] == "completed"
    assert completed["separation"]["progress"] == 100
    assert runtime.prepare_calls == 1
    assert runtime.separate_calls == 1


def test_ready_request_never_prepares_model(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient(ready=True)
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "8" * 32)
        response = client.post(
            f"/api/jobs/{'8' * 32}/separate",
            json={"allowModelDownload": True},
        )

    assert response.status_code == 202
    assert runtime.prepare_calls == 0
    assert runtime.separate_calls == 1


@pytest.mark.parametrize(
    "state,expected",
    [
        ("processing", "already running"),
        ("completed", "already complete"),
    ],
)
def test_processing_and_completed_attempts_are_rejected(tmp_path: Path, state, expected):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "9" * 32)
        db.update_job(
            settings.database_path,
            "9" * 32,
            separation_status=state,
            separation_stage=state,
        )
        response = client.post(f"/api/jobs/{'9' * 32}/separate", json={})

    assert response.status_code == 409
    assert expected in response.json()["detail"]


def test_incomplete_and_missing_analysis_requests_are_rejected(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)

    with TestClient(app) as client:
        create_ready_job(settings, "a" * 32)
        db.update_job(
            settings.database_path,
            "a" * 32,
            preparation_status="pending",
        )
        incomplete = client.post(f"/api/jobs/{'a' * 32}/separate", json={})

        create_ready_job(settings, "b" * 32)
        (settings.exports_dir / ("b" * 32) / "analysis.wav").unlink()
        missing = client.post(f"/api/jobs/{'b' * 32}/separate", json={})

    assert incomplete.status_code == 409
    assert missing.status_code == 409
    assert "missing or unsafe" in missing.json()["detail"]


def test_missing_job_returns_404(tmp_path: Path):
    settings = make_settings(tmp_path)
    app = create_app(
        settings=settings,
        separation_runtime_client=FakeRuntimeClient(),
    )
    with TestClient(app) as client:
        response = client.post(f"/api/jobs/{'c' * 32}/separate", json={})
    assert response.status_code == 404


def test_details_preview_and_download_use_published_manifest(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)
    job_id = "d" * 32

    with TestClient(app) as client:
        create_ready_job(settings, job_id)
        assert client.post(f"/api/jobs/{job_id}/separate", json={}).status_code == 202
        details = client.get(f"/api/jobs/{job_id}/stems")
        preview = client.get(f"/api/jobs/{job_id}/stems/vocals/preview")
        download = client.get(f"/api/jobs/{job_id}/stems/vocals/download")

    payload = details.json()
    assert details.status_code == 200
    assert payload["available"] is True
    assert payload["model"] == "htdemucs"
    assert [stem["kind"] for stem in payload["stems"]] == [
        "vocals",
        "bass",
        "drums",
        "other",
    ]
    assert preview.status_code == 200
    assert preview.headers["content-type"].startswith("audio/wav")
    assert "content-disposition" not in preview.headers
    assert download.status_code == 200
    assert "vocals.wav" in download.headers["content-disposition"]


def test_previous_stems_remain_readable_during_failed_retry(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)
    job_id = "e" * 32

    with TestClient(app) as client:
        create_ready_job(settings, job_id)
        client.post(f"/api/jobs/{job_id}/separate", json={})
        before = client.get(f"/api/jobs/{job_id}/stems/vocals/preview").content
        db.update_job(
            settings.database_path,
            job_id,
            separation_status="failed",
            separation_stage="failed",
            separation_progress=50,
            separation_error="Retryable failure.",
        )
        runtime.fail_separation = True
        retry = client.post(f"/api/jobs/{job_id}/separate", json={})
        details = client.get(f"/api/jobs/{job_id}/stems")
        after = client.get(f"/api/jobs/{job_id}/stems/vocals/preview").content

    assert retry.status_code == 202
    assert details.status_code == 200
    assert before == after
    record = db.get_job(settings.database_path, job_id)
    assert record["separation_status"] == "failed"
    assert record["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH


def test_missing_corrupt_and_unknown_stems_map_safely(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)
    job_id = "f" * 32

    with TestClient(app) as client:
        create_ready_job(settings, job_id)
        missing = client.get(f"/api/jobs/{job_id}/stems")
        client.post(f"/api/jobs/{job_id}/separate", json={})
        unknown = client.get(f"/api/jobs/{job_id}/stems/unknown/preview")
        traversal = client.get(f"/api/jobs/{job_id}/stems/%2E%2E/preview")
        manifest = settings.exports_dir / job_id / STEM_MANIFEST_RELATIVE_PATH
        manifest.write_text("not-json", encoding="utf-8")
        corrupt = client.get(f"/api/jobs/{job_id}/stems")

    assert missing.status_code == 404
    assert unknown.status_code == 404
    assert traversal.status_code == 404
    assert corrupt.status_code == 500
    assert str(settings.data_dir) not in corrupt.text


def test_restart_marks_processing_retryable_and_preserves_manifest_pointer(tmp_path: Path):
    settings = make_settings(tmp_path)
    runtime = FakeRuntimeClient()
    app = create_app(settings=settings, separation_runtime_client=runtime)
    job_id = "0" * 32

    with TestClient(app) as client:
        create_ready_job(settings, job_id)
        client.post(f"/api/jobs/{job_id}/separate", json={})
        db.update_job(
            settings.database_path,
            job_id,
            separation_status="processing",
            separation_stage="separating_stems",
            separation_progress=40,
        )

    restarted = create_app(
        settings=settings,
        separation_runtime_client=runtime,
    )
    with TestClient(restarted) as client:
        payload = client.get(f"/api/jobs/{job_id}").json()
        details = client.get(f"/api/jobs/{job_id}/stems")

    assert payload["separation"]["status"] == "failed"
    assert payload["separation"]["detailsUrl"] == f"/api/jobs/{job_id}/stems"
    assert details.status_code == 200
