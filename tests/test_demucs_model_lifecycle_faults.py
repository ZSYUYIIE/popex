from __future__ import annotations

import hashlib
import json
import os
import sys
import types
from pathlib import Path
from typing import Callable

import pytest

from app import db
from app.config import Settings
from app.separation import STEM_MANIFEST_RELATIVE_PATH
from app.separation_runtime import (
    RuntimeProbeResult,
    WorkerCommandError,
    WorkerErrorDetail,
)
from app.separation_service import SeparationService


ROOT = Path(__file__).resolve().parents[1]
WORKER_ROOT = ROOT / "runtimes" / "demucs_worker"
WORKER_SRC = WORKER_ROOT / "src"
if str(WORKER_SRC) not in sys.path:
    sys.path.insert(0, str(WORKER_SRC))

from popex_demucs_worker import (  # noqa: E402
    cli,
    commands,
    constants,
    model_artifacts,
    paths,
    probes,
)
from popex_demucs_worker.protocol import WorkerError  # noqa: E402


PROFILE = "fault-validation-linux-cpu-v1"
LOCKED_PACKAGES = {
    "demucs": "4.1.0",
    "torch": "2.13.0+cpu",
    "huggingface_hub": "1.26.0",
    "safetensors": "0.8.0",
    "PyYAML": "6.0.3",
}
JOB_ID = "f" * 32
READINESS_NAME = Path(constants.READINESS_RELATIVE_PATH).name
WORKFLOW_PATH = ROOT / ".github" / "workflows" / "demucs-model-lifecycle-faults.yml"
VALIDATOR_PATH = ROOT / "scripts" / "validate_demucs_model_lifecycle_faults.py"
REVIEW_PATH = ROOT / "docs" / "reviews" / "demucs-model-lifecycle-faults.md"


@pytest.fixture(autouse=True)
def exact_runtime_lock(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    runtime_lock = tmp_path / "runtime-lock.json"
    runtime_lock.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "runtimeProfile": PROFILE,
                "workerVersion": constants.WORKER_VERSION,
                "packages": LOCKED_PACKAGES,
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv(constants.RUNTIME_LOCK_ENV, str(runtime_lock))

    def version(distribution: str) -> str:
        if distribution == "popex-demucs-worker":
            return constants.WORKER_VERSION
        if distribution in LOCKED_PACKAGES:
            return LOCKED_PACKAGES[distribution]
        raise probes.metadata.PackageNotFoundError(distribution)

    monkeypatch.setattr(probes.metadata, "version", version)


@pytest.fixture
def synthetic_checkpoint(monkeypatch: pytest.MonkeyPatch) -> bytes:
    data = b"synthetic-approved-checkpoint"
    digest = hashlib.sha256(data).hexdigest()
    for module in (model_artifacts, probes):
        monkeypatch.setattr(module, "CHECKPOINT_SIZE_BYTES", len(data))
        monkeypatch.setattr(module, "CHECKPOINT_SHA256", digest)
    return data


def readiness_path(cache_root: Path) -> Path:
    return cache_root.joinpath(*constants.READINESS_RELATIVE_PATH.split("/"))


def install_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    module = types.ModuleType("yaml")
    module.safe_load = json.loads
    monkeypatch.setitem(sys.modules, "yaml", module)


def install_hub(
    monkeypatch: pytest.MonkeyPatch,
    download: Callable[..., str],
) -> None:
    module = types.ModuleType("huggingface_hub")
    module.hf_hub_download = download
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)


def exact_asset_path(cache_dir: str, revision: str, filename: str) -> Path:
    path = Path(cache_dir) / "snapshots" / revision / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def successful_downloader(
    checkpoint: bytes,
    calls: list[dict[str, object]],
    *,
    after_checkpoint: Callable[[], None] | None = None,
):
    def download(**kwargs) -> str:
        calls.append(dict(kwargs))
        path = exact_asset_path(
            str(kwargs["cache_dir"]),
            str(kwargs["revision"]),
            str(kwargs["filename"]),
        )
        if kwargs["filename"] == constants.BAG_FILE:
            path.write_text(
                json.dumps({"models": [constants.BAG_SIGNATURE]}),
                encoding="utf-8",
            )
        else:
            path.write_bytes(checkpoint)
            if after_checkpoint is not None:
                after_checkpoint()
        return str(path)

    return download


def prepare_successfully(
    monkeypatch: pytest.MonkeyPatch,
    cache_root: Path,
    checkpoint: bytes,
) -> tuple[dict, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(checkpoint, calls))
    result = commands.prepare_model(str(cache_root))
    return result, calls


def run_worker(capsys: pytest.CaptureFixture[str], *args: str):
    exit_code = cli.main(list(args))
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if line.strip()]
    assert len(lines) == 1
    return exit_code, json.loads(lines[0]), captured.err


def model_probe(cache_root: Path) -> dict:
    return probes.model_probe(str(cache_root))


def assert_no_authoritative_readiness(cache_root: Path) -> None:
    assert not readiness_path(cache_root).exists()
    with pytest.raises(WorkerError) as caught:
        model_probe(cache_root)
    assert caught.value.code == "MODEL_DOWNLOAD_REQUIRED"


def test_exact_download_identity_and_privacy_controls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    calls: list[dict[str, object]] = []
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, calls))
    monkeypatch.setenv("HF_TOKEN", "hf_never_forward_this")
    monkeypatch.setenv("HUGGING_FACE_HUB_TOKEN", "hf_never_forward_this_either")

    commands.prepare_model(str(tmp_path))

    assert [call["filename"] for call in calls] == [
        constants.BAG_FILE,
        constants.CHECKPOINT_FILE,
    ]
    assert all(call["repo_id"] == constants.MODEL_REPOSITORY for call in calls)
    assert all(call["revision"] == constants.MODEL_REVISION for call in calls)
    assert all(call["token"] is False for call in calls)
    assert all(call["local_files_only"] is False for call in calls)
    assert all("url" not in call for call in calls)
    assert "HF_TOKEN" not in os.environ
    assert "HUGGING_FACE_HUB_TOKEN" not in os.environ
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"


@pytest.mark.parametrize(
    "fault_stage",
    ["before-any-bytes", "during-yaml", "during-checkpoint"],
)
def test_interrupted_download_never_publishes_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
    fault_stage: str,
) -> None:
    install_yaml(monkeypatch)
    calls: list[str] = []

    def download(**kwargs) -> str:
        filename = str(kwargs["filename"])
        calls.append(filename)
        path = exact_asset_path(
            str(kwargs["cache_dir"]),
            str(kwargs["revision"]),
            filename,
        )
        if fault_stage == "before-any-bytes":
            raise ConnectionError("network unavailable")
        if filename == constants.BAG_FILE:
            if fault_stage == "during-yaml":
                path.with_suffix(path.suffix + ".partial").write_bytes(b"{\"models\":")
                raise ConnectionError("YAML transfer interrupted")
            path.write_text(
                json.dumps({"models": [constants.BAG_SIGNATURE]}),
                encoding="utf-8",
            )
            return str(path)
        if fault_stage == "during-checkpoint":
            path.with_suffix(path.suffix + ".partial").write_bytes(
                synthetic_checkpoint[:5]
            )
            raise ConnectionError("checkpoint transfer interrupted")
        path.write_bytes(synthetic_checkpoint)
        return str(path)

    install_hub(monkeypatch, download)
    with pytest.raises(WorkerError) as caught:
        commands.prepare_model(str(tmp_path))
    assert caught.value.code == "MODEL_DOWNLOAD_FAILED"
    assert caught.value.retryable is True
    assert_no_authoritative_readiness(tmp_path)
    for partial in tmp_path.rglob("*.partial"):
        assert partial.resolve().is_relative_to(tmp_path.resolve())


def test_short_checkpoint_write_is_rejected_and_retry_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    install_yaml(monkeypatch)
    install_hub(
        monkeypatch,
        successful_downloader(synthetic_checkpoint[:-1], []),
    )
    with pytest.raises(WorkerError) as caught:
        commands.prepare_model(str(tmp_path))
    assert caught.value.code == "CHECKPOINT_SIZE_MISMATCH"
    assert_no_authoritative_readiness(tmp_path)

    result, _ = prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    assert result["offlineReady"] is True
    assert model_probe(tmp_path)["checkpointSha256"] == hashlib.sha256(
        synthetic_checkpoint
    ).hexdigest()
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.partial"))


@pytest.mark.parametrize("failure", ["disk-full", "write-failure"])
def test_storage_failure_has_stable_download_error_and_no_false_readiness(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
) -> None:
    install_yaml(monkeypatch)
    secret = "hf_fault_secret_123456"
    leaked_url = "https://user:password@example.invalid/model"
    leaked_path = str(tmp_path / "private" / "checkpoint")

    def download(**kwargs) -> str:
        raise OSError(
            f"{failure}: token={secret} url={leaked_url} path={leaked_path}"
        )

    install_hub(monkeypatch, download)
    code, envelope, stderr = run_worker(
        capsys,
        "--protocol-version",
        "1",
        "prepare-model",
        "--cache-root",
        str(tmp_path),
    )
    assert code == 22
    assert envelope["error"] == {
        "code": "MODEL_DOWNLOAD_FAILED",
        "message": "The authorized model download could not be completed.",
        "retryable": True,
    }
    public = json.dumps(envelope) + stderr
    assert secret not in public
    assert leaked_url not in public
    assert leaked_path not in public
    assert_no_authoritative_readiness(tmp_path)


def test_manifest_file_fsync_failure_removes_temporary_and_retry_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    calls: list[dict[str, object]] = []
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, calls))
    real_fsync = paths.os.fsync
    state = {"failed": False}

    def fail_first_fsync(fd: int) -> None:
        if not state["failed"]:
            state["failed"] = True
            raise OSError("fsync failed")
        real_fsync(fd)

    monkeypatch.setattr(paths.os, "fsync", fail_first_fsync)
    with pytest.raises(OSError):
        commands.prepare_model(str(tmp_path))
    assert_no_authoritative_readiness(tmp_path)
    assert not list(readiness_path(tmp_path).parent.glob("*.tmp"))

    monkeypatch.setattr(paths.os, "fsync", real_fsync)
    result, _ = prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    assert result["offlineReady"] is True
    assert model_probe(tmp_path)["offlineReady"] is True


def test_manifest_atomic_rename_failure_removes_temporary_and_retry_recovers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, []))
    real_replace = paths.os.replace

    def fail_replace(source, destination) -> None:
        raise OSError("atomic rename failed")

    monkeypatch.setattr(paths.os, "replace", fail_replace)
    with pytest.raises(OSError):
        commands.prepare_model(str(tmp_path))
    assert_no_authoritative_readiness(tmp_path)
    assert not list(readiness_path(tmp_path).parent.glob("*.tmp"))

    monkeypatch.setattr(paths.os, "replace", real_replace)
    result, _ = prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    assert result["offlineReady"] is True


def test_read_only_manifest_boundary_produces_no_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, []))
    real_open = Path.open

    def guarded_open(path: Path, *args, **kwargs):
        if path.name.endswith(".tmp") and path.parent.name == "readiness":
            raise PermissionError("read-only readiness directory")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    with pytest.raises(PermissionError):
        commands.prepare_model(str(tmp_path))
    assert_no_authoritative_readiness(tmp_path)


def test_non_directory_readiness_parent_produces_no_readiness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    (tmp_path / "readiness").write_bytes(b"not-a-directory")
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, []))
    with pytest.raises((FileExistsError, NotADirectoryError)):
        commands.prepare_model(str(tmp_path))
    assert not readiness_path(tmp_path).exists()


def test_prepare_model_rejects_symlinked_readiness_parent_without_external_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    """Characterize production defect: atomic readiness publication follows a symlink."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    install_yaml(monkeypatch)

    def create_symlink() -> None:
        readiness_parent = cache_root / "readiness"
        if not readiness_parent.exists():
            readiness_parent.symlink_to(outside, target_is_directory=True)

    install_hub(
        monkeypatch,
        successful_downloader(
            synthetic_checkpoint,
            [],
            after_checkpoint=create_symlink,
        ),
    )

    with pytest.raises(WorkerError):
        commands.prepare_model(str(cache_root))
    assert not (outside / READINESS_NAME).exists()
    assert not readiness_path(cache_root).exists()


@pytest.mark.parametrize(
    "exception,exit_code,error_code",
    [
        (KeyboardInterrupt(), 41, "CANCELLED"),
        (TimeoutError(), 42, "WORKER_TIMEOUT"),
    ],
)
def test_process_style_cancellation_and_timeout_envelopes(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    exception: BaseException,
    exit_code: int,
    error_code: str,
) -> None:
    def fail(cache_root: str):
        raise exception

    monkeypatch.setattr(commands, "prepare_model", fail)
    code, envelope, _ = run_worker(
        capsys,
        "--protocol-version",
        "1",
        "prepare-model",
        "--cache-root",
        str(tmp_path),
    )
    assert code == exit_code
    assert envelope["error"]["code"] == error_code
    assert envelope["error"]["retryable"] is True
    assert_no_authoritative_readiness(tmp_path)


def test_failed_repreparation_preserves_existing_verified_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    manifest_before = manifest.read_bytes()
    checkpoint_relative = json.loads(manifest_before)["cacheAssets"]["checkpoint"]
    checkpoint = tmp_path.joinpath(*checkpoint_relative.split("/"))
    checkpoint_before = checkpoint.read_bytes()

    def network_failure(**kwargs) -> str:
        raise ConnectionError("transient failure")

    install_hub(monkeypatch, network_failure)
    with pytest.raises(WorkerError) as caught:
        commands.prepare_model(str(tmp_path))
    assert caught.value.code == "MODEL_DOWNLOAD_FAILED"
    assert manifest.read_bytes() == manifest_before
    assert checkpoint.read_bytes() == checkpoint_before
    assert model_probe(tmp_path)["offlineReady"] is True


def test_changed_checkpoint_size_invalidates_ready_state_without_deleting_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = tmp_path.joinpath(*payload["cacheAssets"]["checkpoint"].split("/"))
    checkpoint.write_bytes(synthetic_checkpoint + b"x")

    with pytest.raises(WorkerError) as caught:
        model_probe(tmp_path)
    assert caught.value.code == "CHECKPOINT_SIZE_MISMATCH"
    assert manifest.is_file()


@pytest.mark.parametrize("payload", [b"{", b"[]", b"null", b"not-json"])
def test_corrupted_readiness_json_is_rejected(
    tmp_path: Path,
    payload: bytes,
) -> None:
    manifest = readiness_path(tmp_path)
    manifest.parent.mkdir(parents=True)
    manifest.write_bytes(payload)
    with pytest.raises(WorkerError) as caught:
        model_probe(tmp_path)
    assert caught.value.code == "READINESS_MANIFEST_INVALID"


def test_readiness_asset_traversal_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["cacheAssets"]["checkpoint"] = "../outside.safetensors"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(WorkerError) as caught:
        model_probe(tmp_path)
    assert caught.value.code in {"MODEL_ASSET_INVALID", "UNSAFE_PATH"}


def test_symlinked_readiness_file_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    payload = manifest.read_bytes()
    manifest.unlink()
    outside = tmp_path / "outside-readiness.json"
    outside.write_bytes(payload)
    manifest.symlink_to(outside)
    with pytest.raises(WorkerError) as caught:
        model_probe(tmp_path)
    assert caught.value.code == "READINESS_MANIFEST_INVALID"


def test_passive_model_probe_rehashes_same_size_checkpoint_replacement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    """Characterize production defect: model_probe currently checks size, not digest."""
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = tmp_path.joinpath(*payload["cacheAssets"]["checkpoint"].split("/"))
    replacement = bytes(byte ^ 0xFF for byte in synthetic_checkpoint)
    assert len(replacement) == len(synthetic_checkpoint)
    assert hashlib.sha256(replacement).hexdigest() != payload["checkpointSha256"]
    checkpoint.write_bytes(replacement)

    with pytest.raises(WorkerError) as caught:
        model_probe(tmp_path)
    assert caught.value.code == "CHECKPOINT_HASH_MISMATCH"


def test_replacement_during_verification_must_not_publish_ready_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    manifest = readiness_path(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    checkpoint = tmp_path.joinpath(*payload["cacheAssets"]["checkpoint"].split("/"))
    install_yaml(monkeypatch)
    install_hub(monkeypatch, successful_downloader(synthetic_checkpoint, []))
    real_sha256 = model_artifacts._sha256

    def replace_after_hash(path: Path) -> str:
        digest = real_sha256(path)
        path.write_bytes(bytes(byte ^ 0xA5 for byte in synthetic_checkpoint))
        return digest

    monkeypatch.setattr(model_artifacts, "_sha256", replace_after_hash)
    commands.verify_model(str(tmp_path))
    monkeypatch.setattr(model_artifacts, "_sha256", real_sha256)

    with pytest.raises(WorkerError):
        model_probe(tmp_path)
    assert checkpoint.read_bytes() != synthetic_checkpoint


def test_transient_failure_then_explicit_retry_has_no_stale_temporary_influence(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    synthetic_checkpoint: bytes,
) -> None:
    install_yaml(monkeypatch)

    def partial_failure(**kwargs) -> str:
        path = exact_asset_path(
            str(kwargs["cache_dir"]),
            str(kwargs["revision"]),
            str(kwargs["filename"]),
        )
        path.with_suffix(path.suffix + ".partial").write_bytes(b"stale")
        raise ConnectionError("interrupted")

    install_hub(monkeypatch, partial_failure)
    with pytest.raises(WorkerError):
        commands.prepare_model(str(tmp_path))
    for partial in tmp_path.rglob("*.partial"):
        partial.unlink()

    result, calls = prepare_successfully(monkeypatch, tmp_path, synthetic_checkpoint)
    assert result["offlineReady"] is True
    assert model_probe(tmp_path)["offlineReady"] is True
    assert len(calls) == 2
    assert not list(tmp_path.rglob("*.partial"))
    assert not list(tmp_path.rglob("*.tmp"))


def make_service_settings(tmp_path: Path) -> Settings:
    return Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com",),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=500,
        audio_quality="192",
        stem_separation_enabled=True,
        stem_separation_worker_executable=tmp_path / "bin" / "worker",
        stem_separation_runtime_lock=tmp_path / "runtime-lock-for-service.json",
        stem_separation_cache_dir=tmp_path / "runtime-cache",
        stem_separation_runtime_profile=PROFILE,
        stem_separation_device="cpu",
        stem_separation_timeout_seconds=30,
    )


def create_job_with_prior_artifacts(settings: Settings) -> dict[str, bytes]:
    settings.ensure_directories()
    db.init_database(settings.database_path)
    db.create_job(
        settings.database_path,
        JOB_ID,
        source_type="upload",
        original_filename="song.wav",
    )
    job_dir = settings.exports_dir / JOB_ID
    job_dir.mkdir(parents=True)
    files = {
        "source.wav": b"source-bytes",
        "analysis.wav": b"analysis-bytes",
        "metadata.json": b'{"title":"safe"}',
        "analysis/audio-analysis.json": b'{"tempo":120}',
        STEM_MANIFEST_RELATIVE_PATH: b'{"schemaVersion":3,"prior":true}',
        "stems/runs/11111111111111111111111111111111/vocals.wav": b"prior-vocals",
        "stems/runs/11111111111111111111111111111111/bass.wav": b"prior-bass",
        "stems/runs/11111111111111111111111111111111/drums.wav": b"prior-drums",
        "stems/runs/11111111111111111111111111111111/other.wav": b"prior-other",
    }
    for relative, content in files.items():
        path = job_dir.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    db.update_job(
        settings.database_path,
        JOB_ID,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        analysis_status="completed",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        analysis_json_file_name="analysis/audio-analysis.json",
        separation_status="failed",
        separation_stage="failed",
        separation_progress=17,
        separation_model="htdemucs",
        separation_version="demucs-worker-v3",
        stem_manifest_file_name=STEM_MANIFEST_RELATIVE_PATH,
        separated_at="2026-08-04T00:00:00+00:00",
        separation_error="Previous retry failed.",
    )
    return files


class ServiceFaultRuntime:
    def runtime_probe(self) -> RuntimeProbeResult:
        return RuntimeProbeResult(
            runtime_profile=PROFILE,
            worker_version="1.0.0",
            python_version="3.13.14",
            runtime_lock_source="profile",
            demucs_version="4.1.0",
            torch_version="2.13.0+cpu",
            huggingface_hub_version="1.26.0",
            safetensors_version="0.8.0",
            pyyaml_version="6.0.3",
        )

    def model_probe(self):
        raise WorkerCommandError(
            WorkerErrorDetail(
                code="MODEL_DOWNLOAD_REQUIRED",
                worker_code="MODEL_DOWNLOAD_REQUIRED",
                message="The verified model is not prepared.",
                retryable=True,
                exit_code=20,
            )
        )

    def prepare_model(self, *, allow_model_download: bool):
        assert allow_model_download is True
        raise WorkerCommandError(
            WorkerErrorDetail(
                code="MODEL_DOWNLOAD_FAILED",
                worker_code="MODEL_DOWNLOAD_FAILED",
                message=(
                    "token=hf_service_secret bearer abcdef123456 "
                    "https://example.invalid/private "
                    "C:\\private\\cache /private/cache traceback"
                ),
                retryable=True,
                exit_code=22,
            )
        )


def test_claimed_service_failure_is_retryable_and_preserves_all_prior_artifacts(
    tmp_path: Path,
) -> None:
    settings = make_service_settings(tmp_path)
    before = create_job_with_prior_artifacts(settings)
    runtime = ServiceFaultRuntime()
    service = SeparationService(settings, runtime_client=runtime)
    scheduled: list[tuple] = []

    claimed = service.request_start(
        JOB_ID,
        allow_model_download=True,
        schedule=lambda *args: scheduled.append(args),
    )
    assert claimed["separation_status"] == "processing"
    function, *arguments = scheduled[0]
    function(*arguments)

    record = db.get_job(settings.database_path, JOB_ID)
    assert record is not None
    assert record["separation_status"] == "failed"
    assert record["separation_stage"] == "failed"
    assert record["stem_manifest_file_name"] == STEM_MANIFEST_RELATIVE_PATH
    public_error = str(record["separation_error"])
    for secret in (
        "hf_service_secret",
        "abcdef123456",
        "example.invalid",
        "C:\\private\\cache",
        "/private/cache",
    ):
        assert secret not in public_error

    job_dir = settings.exports_dir / JOB_ID
    for relative, content in before.items():
        assert job_dir.joinpath(*relative.split("/")).read_bytes() == content

    serialized = service.serialize_job(record)
    assert serialized is not None
    serialized_text = json.dumps(serialized)
    assert "hf_service_secret" not in serialized_text
    assert "/private/cache" not in serialized_text


def test_no_claim_means_no_failure_state_or_artifact_mutation(tmp_path: Path) -> None:
    settings = make_service_settings(tmp_path)
    before = create_job_with_prior_artifacts(settings)
    runtime = ServiceFaultRuntime()
    service = SeparationService(settings, runtime_client=runtime)

    with pytest.raises(Exception, match="consent"):
        service.request_start(
            JOB_ID,
            allow_model_download=False,
            schedule=lambda *args: None,
        )

    record = db.get_job(settings.database_path, JOB_ID)
    assert record is not None
    assert record["separation_status"] == "failed"
    assert record["separation_error"] == "Previous retry failed."
    job_dir = settings.exports_dir / JOB_ID
    for relative, content in before.items():
        assert job_dir.joinpath(*relative.split("/")).read_bytes() == content


def test_fault_workflow_is_linux_only_offline_path_focused_and_cleans() -> None:
    text = WORKFLOW_PATH.read_text(encoding="utf-8")
    assert "runs-on: ubuntu-latest" in text
    assert "windows-latest" not in text
    assert "workflow_dispatch:" in text
    for path in (
        ".github/workflows/demucs-model-lifecycle-faults.yml",
        "scripts/validate_demucs_model_lifecycle_faults.py",
        "tests/test_demucs_model_lifecycle_faults.py",
        "docs/reviews/demucs-model-lifecycle-faults.md",
        "runtimes/demucs_worker/**",
        "app/separation_runtime.py",
        "app/separation_service.py",
    ):
        assert path in text
    for variable in (
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_UPDATE_CHECK",
    ):
        assert variable in text
    assert "permissions:\n  contents: read" in text
    assert "if: always()" in text
    assert "upload-artifact" not in text
    assert "huggingface.co" not in text
    assert "curl " not in text
    assert "wget " not in text


def test_validator_and_review_files_exist_and_forbid_real_model_access() -> None:
    validator = VALIDATOR_PATH.read_text(encoding="utf-8")
    review = REVIEW_PATH.read_text(encoding="utf-8")
    assert "tests/test_demucs_model_lifecycle_faults.py" in validator
    assert "HF_HUB_OFFLINE" in validator
    assert "allow_nan=False" in validator
    assert "955717e8.safetensors" not in validator
    assert "hf_hub_download" not in validator
    assert "BLOCKED" in review
    assert "symlink" in review.lower()
    assert "same-size" in review.lower()
