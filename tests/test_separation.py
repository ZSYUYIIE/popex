import inspect
import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

import numpy as np
import pytest
import soundfile as sf

from app import separation
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
    STEM_MANIFEST_SCHEMA_VERSION,
    SeparationOptions,
    StemSeparationError,
    load_stem_manifest,
    separate_stems,
    stem_manifest_path,
)


SAMPLE_RATE = 8000
JOB_ID = "a" * 32
RUNTIME_PROFILE = "linux-cpu-v1"
WORKER_VERSION = "1.0.0"
TORCH_VERSION = "2.13.0"
HF_HUB_VERSION = "1.16.1"
EXPECTED_WEIGHTS_IDENTIFIER = f"sha256:{AUDITED_CHECKPOINT_SHA256}"
REQUIRED_OUTPUT_NAMES = [f"{kind}.wav" for kind in REQUIRED_STEM_KINDS]


def settings(tmp_path: Path) -> Settings:
    config = Settings(
        data_dir=tmp_path,
        allowed_hosts=("youtube.com",),
        max_duration_seconds=60,
        max_filesize_mb=10,
        max_upload_mb=10,
        audio_quality="192",
    )
    config.ensure_directories()
    return config


def create_analysis_wav(config: Settings, job_id: str = JOB_ID) -> Path:
    job_dir = config.exports_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(SAMPLE_RATE // 10, dtype=np.float32) / SAMPLE_RATE
    audio = 0.1 * np.sin(2 * np.pi * 220 * t)
    path = job_dir / "analysis.wav"
    sf.write(path, audio, SAMPLE_RATE, subtype="PCM_16")
    return path


def write_stem(path: Path, *, valid: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not valid:
        path.write_bytes(b"not a wav")
        return
    samples = np.linspace(-0.1, 0.1, SAMPLE_RATE // 20, dtype=np.float32)
    sf.write(path, samples, SAMPLE_RATE, subtype="PCM_16")


def normalized_result(**overrides: Any) -> dict[str, Any]:
    result: dict[str, Any] = {
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "demucsVersion": AUDITED_DEMUCS_VERSION,
        "torchVersion": TORCH_VERSION,
        "huggingfaceHubVersion": HF_HUB_VERSION,
        "modelRepository": AUDITED_MODEL_REPOSITORY,
        "modelRevision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "device": "cpu",
        "outputs": list(REQUIRED_OUTPUT_NAMES),
    }
    result.update(overrides)
    return result


def output_directory(workspace_root: Path, output_relative: str) -> Path:
    relative = PurePosixPath(output_relative)
    return workspace_root.joinpath(*relative.parts)


class FakeWorkerRunner:
    def __init__(
        self,
        *,
        result: Mapping[str, Any] | None = None,
        invalid_kind: str | None = None,
        missing_kind: str | None = None,
        extra_file: str | None = None,
        symlink_output_outside: bool = False,
    ) -> None:
        self.result = dict(result or normalized_result())
        self.invalid_kind = invalid_kind
        self.missing_kind = missing_kind
        self.extra_file = extra_file
        self.symlink_output_outside = symlink_output_outside
        self.calls: list[dict[str, Any]] = []
        self.output_bytes: dict[str, bytes] = {}

    def __call__(
        self,
        *,
        workspace_root: Path,
        cache_root: Path,
        input_relative: str,
        output_relative: str,
        device: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]:
        self.calls.append(
            {
                "workspace_root": workspace_root,
                "cache_root": cache_root,
                "input_relative": input_relative,
                "output_relative": output_relative,
                "device": device,
                "timeout_seconds": timeout_seconds,
            }
        )
        root = output_directory(workspace_root, output_relative)
        if self.symlink_output_outside:
            outside = workspace_root.parent / "outside-worker-output"
            outside.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(root)
            root.symlink_to(outside, target_is_directory=True)
        for kind in REQUIRED_STEM_KINDS:
            if kind == self.missing_kind:
                continue
            path = root / f"{kind}.wav"
            write_stem(path, valid=kind != self.invalid_kind)
            self.output_bytes[kind] = path.read_bytes()
        if self.extra_file:
            (root / self.extra_file).write_bytes(b"unexpected")
        return dict(self.result)


def options(
    runner,
    tmp_path: Path,
    **overrides: Any,
) -> SeparationOptions:
    values: dict[str, Any] = {
        "separation_version": "demucs-worker-v3",
        "worker_runner": runner,
        "cache_root": tmp_path / "trusted-model-cache",
        "expected_model_repository": AUDITED_MODEL_REPOSITORY,
        "expected_model_revision": AUDITED_MODEL_REVISION,
        "expected_checkpoint_file": AUDITED_CHECKPOINT_FILE,
        "expected_checkpoint_sha256": AUDITED_CHECKPOINT_SHA256,
        "expected_demucs_version": AUDITED_DEMUCS_VERSION,
        "expected_runtime_profile": RUNTIME_PROFILE,
        "device": "cpu",
        "timeout_seconds": 15,
    }
    values.update(overrides)
    return SeparationOptions(**values)


def create_success(config: Settings, tmp_path: Path):
    runner = FakeWorkerRunner()
    result = separate_stems(JOB_ID, config, options(runner, tmp_path))
    return runner, result


def test_missing_analysis_wav(tmp_path: Path):
    config = settings(tmp_path)
    (config.exports_dir / JOB_ID).mkdir()

    with pytest.raises(StemSeparationError, match="analysis.wav|Analysis audio"):
        separate_stems(JOB_ID, config, options(FakeWorkerRunner(), tmp_path))


def test_worker_receives_exact_trusted_roots_and_relative_paths(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner()
    cache_root = tmp_path / "cache-root" / "profile"

    result = separate_stems(
        JOB_ID,
        config,
        options(runner, tmp_path, cache_root=cache_root),
    )

    assert len(runner.calls) == 1
    call = runner.calls[0]
    assert call["workspace_root"] == (config.exports_dir / JOB_ID).resolve()
    assert call["cache_root"] == cache_root.resolve()
    assert call["input_relative"] == "analysis.wav"
    assert call["output_relative"] == (
        f"stems/runs/{result.run_id}/worker-output"
    )
    assert call["device"] == "cpu"
    assert call["timeout_seconds"] == 15.0


def test_production_path_contains_no_direct_demucs_command():
    source = inspect.getsource(separation)

    assert "python_executable" not in source
    assert "command_runner" not in source
    assert "loader_reference" not in source
    assert "environment_overrides" not in source
    assert "subprocess.run(" not in source
    assert "python -m demucs" not in source
    assert '"-m",\n        "demucs"' not in source


def test_valid_normalized_worker_result_and_exact_wav_preservation(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner, result = create_success(config, tmp_path)

    assert [stem.kind for stem in result.stems] == list(REQUIRED_STEM_KINDS)
    assert result.model_name == AUDITED_MODEL_NAME
    assert result.package_version == AUDITED_DEMUCS_VERSION
    assert result.weights_identifier == EXPECTED_WEIGHTS_IDENTIFIER
    assert result.device == "cpu"
    for stem in result.stems:
        stored = config.exports_dir / JOB_ID / stem.file_name
        assert stored.is_file()
        assert stored.read_bytes() == runner.output_bytes[stem.kind]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("demucsVersion", "4.0.1", "Demucs 4.1.0|Demucs version"),
        ("modelRepository", "someone/HTDemucs", "model repository"),
        ("modelRevision", "0" * 40, "model revision"),
        ("checkpointFile", "other.safetensors", "checkpoint filename"),
        ("checkpointSha256", "0" * 64, "checkpoint SHA-256"),
        ("runtimeProfile", "windows-cpu-v1", "runtime profile"),
        ("device", "cuda", "device"),
    ],
)
def test_wrong_worker_trust_values_are_rejected(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner(result=normalized_result(**{field: value}))

    with pytest.raises(StemSeparationError, match=message):
        separate_stems(JOB_ID, config, options(runner, tmp_path))


@pytest.mark.parametrize("value", ["abc", "A" * 64, "g" * 64, "0" * 63])
def test_malformed_or_uppercase_worker_hash_is_rejected(tmp_path: Path, value: str):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner(result=normalized_result(checkpointSha256=value))

    with pytest.raises(StemSeparationError, match="64 lowercase hexadecimal"):
        separate_stems(JOB_ID, config, options(runner, tmp_path))


@pytest.mark.parametrize(
    ("outputs", "message"),
    [
        (["vocals.wav", "bass.wav", "drums.wav"], "exactly"),
        (REQUIRED_OUTPUT_NAMES + ["piano.wav"], "exactly"),
        (["bass.wav", "vocals.wav", "drums.wav", "other.wav"], "stable order"),
        (["vocals.wav", "bass.wav", "drums.wav", "drums.wav"], "duplicate"),
        (["nested/vocals.wav", "bass.wav", "drums.wav", "other.wav"], "unsafe"),
        (["/vocals.wav", "bass.wav", "drums.wav", "other.wav"], "unsafe"),
        (["..", "bass.wav", "drums.wav", "other.wav"], "unsafe"),
    ],
)
def test_invalid_worker_output_name_sets_are_rejected(
    tmp_path: Path,
    outputs: list[str],
    message: str,
):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner(result=normalized_result(outputs=outputs))

    with pytest.raises(StemSeparationError, match=message):
        separate_stems(JOB_ID, config, options(runner, tmp_path))


def test_worker_version_fields_cannot_persist_paths(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner(result=normalized_result(workerVersion="/private/worker"))

    with pytest.raises(StemSeparationError, match="malformed worker version"):
        separate_stems(JOB_ID, config, options(runner, tmp_path))


def test_separation_version_cannot_persist_paths(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner()

    with pytest.raises(StemSeparationError, match="separation version is malformed"):
        separate_stems(
            JOB_ID,
            config,
            options(runner, tmp_path, separation_version="/private/version"),
        )
    assert runner.calls == []


def test_unknown_worker_result_field_is_rejected(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    result = normalized_result()
    result["loaderReference"] = "hf://not-accepted-here"

    with pytest.raises(StemSeparationError, match="missing or unknown fields"):
        separate_stems(
            JOB_ID,
            config,
            options(FakeWorkerRunner(result=result), tmp_path),
        )


def test_worker_output_directory_requires_exactly_four_files(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    with pytest.raises(StemSeparationError, match="exactly the required"):
        separate_stems(
            JOB_ID,
            config,
            options(FakeWorkerRunner(extra_file="notes.txt"), tmp_path),
        )


def test_worker_output_outside_allocated_directory_is_rejected(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner(symlink_output_outside=True)

    try:
        with pytest.raises(StemSeparationError, match="unsafe|outside"):
            separate_stems(JOB_ID, config, options(runner, tmp_path))
    except OSError:
        pytest.skip("Directory symlinks are unavailable on this platform.")

    outside = config.exports_dir / "outside-worker-output"
    assert not stem_manifest_path(JOB_ID, config).exists()
    assert outside.exists()


def test_manifest_schema_3_and_complete_runtime_provenance(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    _, result = create_success(config, tmp_path)
    payload = json.loads(stem_manifest_path(JOB_ID, config).read_text(encoding="utf-8"))

    assert result.manifest_file_name == STEM_MANIFEST_RELATIVE_PATH
    assert payload["schemaVersion"] == STEM_MANIFEST_SCHEMA_VERSION == 3
    assert payload["separationVersion"] == "demucs-worker-v3"
    assert payload["sourceAsset"] == "analysis.wav"
    assert payload["model"] == {
        "name": AUDITED_MODEL_NAME,
        "packageVersion": AUDITED_DEMUCS_VERSION,
        "runtimeProfile": RUNTIME_PROFILE,
        "workerVersion": WORKER_VERSION,
        "torchVersion": TORCH_VERSION,
        "huggingfaceHubVersion": HF_HUB_VERSION,
        "repository": AUDITED_MODEL_REPOSITORY,
        "revision": AUDITED_MODEL_REVISION,
        "checkpointFile": AUDITED_CHECKPOINT_FILE,
        "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
        "weightsIdentifier": EXPECTED_WEIGHTS_IDENTIFIER,
        "device": "cpu",
    }
    assert [item["kind"] for item in payload["stems"]] == list(REQUIRED_STEM_KINDS)
    for item in payload["stems"]:
        assert item["fileName"] == (
            f"stems/runs/{payload['runId']}/{item['kind']}.wav"
        )
        assert not Path(item["fileName"]).is_absolute()
        assert "\\" not in item["fileName"]
        assert item["durationSeconds"] > 0
        assert item["sampleRate"] == SAMPLE_RATE
        assert item["channels"] == 1
        assert item["sizeBytes"] > 0
    assert payload["warnings"] == []

    loaded = load_stem_manifest(JOB_ID, config)
    assert loaded is not None
    assert loaded.payload == payload
    assert loaded.provenance == result.provenance


def test_cache_and_executable_paths_are_absent_from_manifest(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    cache_root = tmp_path / "private" / "worker-cache"
    runner = FakeWorkerRunner()

    result = separate_stems(
        JOB_ID,
        config,
        options(runner, tmp_path, cache_root=cache_root),
    )
    serialized = json.dumps(result.payload, sort_keys=True)

    assert str(cache_root) not in serialized
    assert "cacheRoot" not in serialized
    assert "readiness" not in serialized
    assert "executable" not in serialized.casefold()
    assert "worker-output" not in serialized


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda model: model.__setitem__("packageVersion", "4.0.1"), "Demucs version"),
        (lambda model: model.__setitem__("repository", "other/HTDemucs"), "repository"),
        (lambda model: model.__setitem__("revision", "0" * 40), "revision"),
        (
            lambda model: model.__setitem__("checkpointFile", "other.safetensors"),
            "checkpoint filename",
        ),
        (lambda model: model.__setitem__("checkpointSha256", "A" * 64), "SHA-256"),
        (lambda model: model.__setitem__("weightsIdentifier", "sha256:" + "0" * 64), "weight identity"),
        (lambda model: model.__setitem__("runtimeProfile", ""), "runtimeProfile"),
        (lambda model: model.__setitem__("workerVersion", ""), "workerVersion"),
        (lambda model: model.__setitem__("torchVersion", ""), "torchVersion"),
        (lambda model: model.__setitem__("huggingfaceHubVersion", ""), "huggingfaceHubVersion"),
        (lambda model: model.__setitem__("device", "tpu"), "device"),
        (lambda model: model.__setitem__("unknownTrustField", "value"), "unknown fields"),
    ],
)
def test_saved_provenance_is_revalidated(tmp_path: Path, mutation, message: str):
    config = settings(tmp_path)
    create_analysis_wav(config)
    _, result = create_success(config, tmp_path)
    payload = json.loads(json.dumps(result.payload))
    mutation(payload["model"])
    stem_manifest_path(JOB_ID, config).write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StemSeparationError, match=message):
        load_stem_manifest(JOB_ID, config)


@pytest.mark.parametrize("schema", [1, 2, 99])
def test_unreleased_old_and_unknown_manifest_schemas_are_rejected(
    tmp_path: Path,
    schema: int,
):
    config = settings(tmp_path)
    create_analysis_wav(config)
    path = stem_manifest_path(JOB_ID, config)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schemaVersion": schema}), encoding="utf-8")

    with pytest.raises(StemSeparationError, match="unsupported schema"):
        load_stem_manifest(JOB_ID, config)


def test_corrupt_nonfinite_manifest_is_rejected(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    path = stem_manifest_path(JOB_ID, config)
    path.parent.mkdir(parents=True)
    path.write_text('{"schemaVersion": 3, "value": NaN}', encoding="utf-8")

    with pytest.raises(StemSeparationError, match="unreadable|corrupted"):
        load_stem_manifest(JOB_ID, config)


def test_canonical_stage_callback_order_and_no_100(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    stages: list[tuple[str, str, float]] = []

    separate_stems(
        JOB_ID,
        config,
        options(FakeWorkerRunner(), tmp_path),
        stage_callback=lambda stage, message, progress: stages.append(
            (stage, message, progress)
        ),
    )

    assert [stage for stage, _, _ in stages] == [
        "preparing_separation",
        "separating_stems",
        "validating_stems",
        "saving_stems",
    ]
    assert [progress for _, _, progress in stages] == [3, 10, 85, 95]
    assert all(progress < 100 for _, _, progress in stages)
    assert all(message and len(message) < 80 for _, message, _ in stages)


def test_callback_failure_aborts_and_cleans_unpublished_run(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    def callback(stage: str, message: str, progress: float) -> None:
        if stage == "validating_stems":
            raise RuntimeError("callback failed")

    with pytest.raises(StemSeparationError, match="progress reporting failed"):
        separate_stems(
            JOB_ID,
            config,
            options(FakeWorkerRunner(), tmp_path),
            stage_callback=callback,
        )

    assert load_stem_manifest(JOB_ID, config) is None
    runs = config.exports_dir / JOB_ID / "stems" / "runs"
    assert not runs.exists() or list(runs.iterdir()) == []


def test_worker_exception_is_sanitized_and_does_not_expose_traceback(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    private_path = str(config.data_dir.resolve())

    def runner(**kwargs):
        raise RuntimeError(f"Traceback: internal failure at {private_path}/private")

    with pytest.raises(StemSeparationError) as raised:
        separate_stems(JOB_ID, config, options(runner, tmp_path))

    message = str(raised.value)
    assert message == "The trusted stem-separation worker failed."
    assert private_path not in message
    assert "Traceback" not in message


def test_timeout_is_mapped_to_concise_error(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    def runner(**kwargs):
        raise subprocess.TimeoutExpired("popex-demucs-worker", kwargs["timeout_seconds"])

    with pytest.raises(StemSeparationError, match="timed out after 15 seconds"):
        separate_stems(JOB_ID, config, options(runner, tmp_path))


def test_worker_missing_is_mapped_without_executable_path(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    def runner(**kwargs):
        raise FileNotFoundError("/private/runtime/bin/popex-demucs-worker")

    with pytest.raises(StemSeparationError) as raised:
        separate_stems(JOB_ID, config, options(runner, tmp_path))
    assert str(raised.value) == "The trusted stem-separation worker is unavailable."
    assert "/private/runtime" not in str(raised.value)


def test_invalid_or_unreadable_wav_output(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    with pytest.raises(StemSeparationError, match="unreadable|corrupted|metadata"):
        separate_stems(
            JOB_ID,
            config,
            options(FakeWorkerRunner(invalid_kind="bass"), tmp_path),
        )
    assert load_stem_manifest(JOB_ID, config) is None


def test_one_missing_required_stem_file(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)

    with pytest.raises(StemSeparationError, match="exactly the required|missing"):
        separate_stems(
            JOB_ID,
            config,
            options(FakeWorkerRunner(missing_kind="other"), tmp_path),
        )


def test_failed_retry_preserves_earlier_successful_manifest_and_files(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    _, first = create_success(config, tmp_path)
    manifest_path = stem_manifest_path(JOB_ID, config)
    original_manifest = manifest_path.read_bytes()
    original_files = [
        config.exports_dir / JOB_ID / stem.file_name for stem in first.stems
    ]
    original_bytes = {path: path.read_bytes() for path in original_files}

    def failed_runner(
        *,
        workspace_root: Path,
        cache_root: Path,
        input_relative: str,
        output_relative: str,
        device: str,
        timeout_seconds: float,
    ):
        root = output_directory(workspace_root, output_relative)
        write_stem(root / "vocals.wav")
        raise RuntimeError("worker inference failed")

    with pytest.raises(StemSeparationError, match="worker failed"):
        separate_stems(JOB_ID, config, options(failed_runner, tmp_path))

    assert manifest_path.read_bytes() == original_manifest
    assert all(path.is_file() for path in original_files)
    assert all(path.read_bytes() == original_bytes[path] for path in original_files)
    loaded = load_stem_manifest(JOB_ID, config)
    assert loaded is not None
    assert loaded.run_id == first.run_id
    run_dirs = list((config.exports_dir / JOB_ID / "stems" / "runs").iterdir())
    assert run_dirs == [config.exports_dir / JOB_ID / "stems" / "runs" / first.run_id]


def test_path_containment_and_no_absolute_path_leakage(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    _, result = create_success(config, tmp_path)
    manifest_path = stem_manifest_path(JOB_ID, config)
    payload = json.loads(json.dumps(result.payload))
    payload["stems"][0]["fileName"] = str((tmp_path / "outside.wav").resolve())
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(StemSeparationError) as raised:
        load_stem_manifest(JOB_ID, config)
    assert str(tmp_path.resolve()) not in str(raised.value)

    with pytest.raises(StemSeparationError, match="Invalid job identifier"):
        stem_manifest_path("../escape", config)


def test_saved_wav_metadata_is_revalidated(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    _, result = create_success(config, tmp_path)
    stem_path = config.exports_dir / JOB_ID / result.stems[0].file_name
    replacement = np.linspace(-0.2, 0.2, SAMPLE_RATE // 10, dtype=np.float32)
    sf.write(stem_path, replacement, SAMPLE_RATE, subtype="PCM_16")

    with pytest.raises(StemSeparationError, match="metadata does not match"):
        load_stem_manifest(JOB_ID, config)


def test_symlinked_stems_directory_cannot_escape_job(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    job_dir = config.exports_dir / JOB_ID
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (job_dir / "stems").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Directory symlinks are unavailable on this platform.")

    with pytest.raises(StemSeparationError, match="unsafe|escaped"):
        separate_stems(JOB_ID, config, options(FakeWorkerRunner(), tmp_path))
    assert not (outside / "runs").exists()


def test_options_reject_unapproved_expected_identity_before_worker_runs(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner()

    with pytest.raises(StemSeparationError, match="not approved"):
        separate_stems(
            JOB_ID,
            config,
            options(
                runner,
                tmp_path,
                expected_model_repository="other/HTDemucs",
            ),
        )
    assert runner.calls == []


def test_options_reject_malformed_expected_hash_before_worker_runs(tmp_path: Path):
    config = settings(tmp_path)
    create_analysis_wav(config)
    runner = FakeWorkerRunner()

    with pytest.raises(StemSeparationError, match="64 lowercase hexadecimal"):
        separate_stems(
            JOB_ID,
            config,
            options(
                runner,
                tmp_path,
                expected_checkpoint_sha256="A" * 64,
            ),
        )
    assert runner.calls == []
