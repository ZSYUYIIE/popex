from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import FrozenInstanceError, dataclass
from pathlib import Path

import pytest

from app.separation_runtime import (
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ModelDownloadConsentRequiredError,
    RuntimeMissingError,
    RuntimeProbeResult,
    SeparationRuntimeClient,
    WorkerCommandError,
    WorkerProtocolError,
)

RUN_ID = "0123456789abcdef0123456789abcdef"
OUTPUT_RELATIVE = f"stems/runs/{RUN_ID}/worker-output"
VERSIONS = {
    "demucs": "4.1.0",
    "torch": "2.13.0+cpu",
    "huggingface_hub": "1.16.1",
    "safetensors": "0.6.2",
    "PyYAML": "6.0.3",
}


@dataclass
class FakeCompleted:
    returncode: int
    stdout: bytes
    stderr: bytes = b""
    stdout_overflow: bool = False
    stderr_overflow: bool = False


class FakeRunner:
    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), kwargs))
        if not self.outcomes:
            raise AssertionError("unexpected process invocation")
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def ok(command: str, result: dict, *, warnings=None) -> FakeCompleted:
    payload = {
        "protocolVersion": 1,
        "command": command,
        "status": "ok",
        "result": result,
        "warnings": warnings or [],
    }
    return FakeCompleted(0, json.dumps(payload, allow_nan=False).encode())


def error(
    exit_code: int,
    command: str,
    worker_code: str,
    *,
    message="failed",
    retryable=True,
    stderr=b"",
):
    payload = {
        "protocolVersion": 1,
        "command": command,
        "status": "error",
        "error": {"code": worker_code, "message": message, "retryable": retryable},
        "warnings": [],
    }
    return FakeCompleted(exit_code, json.dumps(payload).encode(), stderr)


def runtime_result(**overrides):
    value = {
        "runtimeProfile": "linux-cpu-v1",
        "workerVersion": "1.0.0",
        "pythonVersion": "3.13.14",
        "runtimeLockSource": "profile",
        "installedVersions": dict(VERSIONS),
        "lockedVersions": dict(VERSIONS),
        "compatible": True,
    }
    value.update(overrides)
    return value


def model_result(**overrides):
    value = {
        "schemaVersion": 1,
        "state": "MODEL_READY",
        "runtimeProfile": "linux-cpu-v1",
        "workerVersion": "1.0.0",
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": VERSIONS["torch"],
        "huggingfaceHubVersion": VERSIONS["huggingface_hub"],
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "bagFile": "htdemucs.yaml",
        "bagModelSignatures": ["955717e8"],
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSizeBytes": CHECKPOINT_SIZE_BYTES,
        "checkpointSha256": CHECKPOINT_SHA256,
        "verifiedAt": "2026-08-03T00:00:00Z",
        "offlineReady": True,
        "readinessManifest": "readiness/htdemucs-bf35a81b-v1.json",
    }
    value.update(overrides)
    return value


def separation_result(**overrides):
    value = {
        "runtimeProfile": "linux-cpu-v1",
        "workerVersion": "1.0.0",
        "demucsVersion": DEMUCS_VERSION,
        "torchVersion": VERSIONS["torch"],
        "huggingfaceHubVersion": VERSIONS["huggingface_hub"],
        "modelRepository": MODEL_REPOSITORY,
        "modelRevision": MODEL_REVISION,
        "checkpointFile": CHECKPOINT_FILE,
        "checkpointSha256": CHECKPOINT_SHA256,
        "device": "cpu",
        "outputs": ["vocals.wav", "bass.wav", "drums.wav", "other.wav"],
    }
    value.update(overrides)
    return value


@pytest.fixture
def roots(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "analysis.wav").write_bytes(b"RIFFsynthetic")
    worker = tmp_path / "bin" / "popex-demucs-worker"
    return worker, cache, workspace


def make_client(roots, runner, **kwargs):
    worker, cache, _ = roots
    return SeparationRuntimeClient(worker, cache, process_runner=runner, **kwargs)


def test_import_has_no_optional_runtime_modules():
    forbidden = {"demucs", "torch", "huggingface_hub", "safetensors", "yaml"}
    assert forbidden.isdisjoint(sys.modules)


def test_result_types_are_immutable():
    result = RuntimeProbeResult(
        "p", "w", "py", "profile", "4.1.0", "t", "h", "s", "y"
    )
    with pytest.raises(FrozenInstanceError):
        result.worker_version = "other"


def test_missing_executable(roots):
    runner = FakeRunner(FileNotFoundError("missing"))
    with pytest.raises(RuntimeMissingError) as caught:
        make_client(roots, runner).runtime_probe()
    assert caught.value.code == "RUNTIME_MISSING"
    assert caught.value.detail.exit_code is None
    assert caught.value.detail.worker_code is None


def test_canonical_nested_runtime_probe(roots):
    runner = FakeRunner(ok("runtime-probe", runtime_result()))
    result = make_client(roots, runner).runtime_probe()
    assert result.runtime_profile == "linux-cpu-v1"
    assert result.runtime_lock_source == "profile"
    assert result.demucs_version == "4.1.0"
    assert result.torch_version == VERSIONS["torch"]
    assert result.huggingface_hub_version == VERSIONS["huggingface_hub"]
    assert result.safetensors_version == VERSIONS["safetensors"]
    assert result.pyyaml_version == VERSIONS["PyYAML"]


def test_bundled_runtime_lock_source_is_valid(roots):
    runner = FakeRunner(ok("runtime-probe", runtime_result(runtimeLockSource="bundled")))
    assert make_client(roots, runner).runtime_probe().runtime_lock_source == "bundled"


@pytest.mark.parametrize("source", ["/absolute/lock.json", "profile.json", "", "other"])
def test_invalid_runtime_lock_source(roots, source):
    runner = FakeRunner(ok("runtime-probe", runtime_result(runtimeLockSource=source)))
    with pytest.raises(WorkerProtocolError):
        make_client(roots, runner).runtime_probe()


def test_installed_locked_mismatch(roots):
    locked = dict(VERSIONS)
    locked["torch"] = "different"
    runner = FakeRunner(ok("runtime-probe", runtime_result(lockedVersions=locked)))
    with pytest.raises(WorkerProtocolError, match="do not match"):
        make_client(roots, runner).runtime_probe()


def test_compatible_false(roots):
    runner = FakeRunner(ok("runtime-probe", runtime_result(compatible=False)))
    with pytest.raises(WorkerProtocolError, match="not compatible"):
        make_client(roots, runner).runtime_probe()


@pytest.mark.parametrize("container", ["installedVersions", "lockedVersions"])
@pytest.mark.parametrize("mutation", ["missing", "extra", "non-string", "empty"])
def test_runtime_version_maps_require_exact_keys(roots, container, mutation):
    versions = dict(VERSIONS)
    if mutation == "missing":
        versions.pop("PyYAML")
    elif mutation == "extra":
        versions["numpy"] = "2.5.1"
    elif mutation == "non-string":
        versions["torch"] = 1
    else:
        versions["torch"] = ""
    runner = FakeRunner(ok("runtime-probe", runtime_result(**{container: versions})))
    with pytest.raises(WorkerProtocolError):
        make_client(roots, runner).runtime_probe()


def test_runtime_demucs_must_equal_410(roots):
    installed = dict(VERSIONS)
    locked = dict(VERSIONS)
    installed["demucs"] = locked["demucs"] = "4.0.1"
    runner = FakeRunner(
        ok(
            "runtime-probe",
            runtime_result(installedVersions=installed, lockedVersions=locked),
        )
    )
    with pytest.raises(WorkerProtocolError, match="Demucs"):
        make_client(roots, runner).runtime_probe()


def test_exact_runtime_probe_arguments(roots):
    runner = FakeRunner(ok("runtime-probe", runtime_result()))
    client = make_client(roots, runner)
    client.runtime_probe()
    worker, _, _ = roots
    assert runner.calls[0][0] == [
        str(worker),
        "--protocol-version",
        "1",
        "runtime-probe",
    ]


def test_exact_model_probe_arguments_and_required_versions(roots):
    runner = FakeRunner(ok("model-probe", model_result()))
    client = make_client(roots, runner)
    result = client.model_probe()
    worker, cache, _ = roots
    assert runner.calls[0][0] == [
        str(worker),
        "--protocol-version",
        "1",
        "model-probe",
        "--cache-root",
        str(cache),
    ]
    assert result.demucs_version == "4.1.0"
    assert result.torch_version == VERSIONS["torch"]
    assert result.huggingface_hub_version == VERSIONS["huggingface_hub"]
    assert not hasattr(result, "readiness_manifest")


def test_prepare_refuses_without_consent_before_spawn(roots):
    runner = FakeRunner()
    with pytest.raises(ModelDownloadConsentRequiredError):
        make_client(roots, runner).prepare_model(allow_model_download=False)
    assert runner.calls == []


def test_prepare_requires_exact_true(roots):
    runner = FakeRunner()
    with pytest.raises(ModelDownloadConsentRequiredError):
        make_client(roots, runner).prepare_model(allow_model_download=1)
    assert runner.calls == []


def test_authorized_prepare_arguments(roots):
    runner = FakeRunner(ok("prepare-model", model_result()))
    client = make_client(roots, runner)
    client.prepare_model(allow_model_download=True)
    worker, cache, _ = roots
    assert runner.calls[0][0] == [
        str(worker),
        "--protocol-version",
        "1",
        "prepare-model",
        "--cache-root",
        str(cache),
    ]
    assert "HF_HUB_OFFLINE" not in runner.calls[0][1]["env"]


def test_verify_model_is_offline(roots):
    runner = FakeRunner(ok("verify-model", model_result()))
    make_client(roots, runner).verify_model()
    assert runner.calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"


def test_separate_is_offline_and_exact_arguments(roots):
    runner = FakeRunner(ok("separate", separation_result()))
    client = make_client(roots, runner)
    _, cache, workspace = roots
    client.separate(
        workspace_root=workspace,
        input_relative="analysis.wav",
        output_relative=OUTPUT_RELATIVE,
        device="cpu",
        timeout_seconds=22,
    )
    worker, _, _ = roots
    assert runner.calls[0][0] == [
        str(worker),
        "--protocol-version",
        "1",
        "separate",
        "--cache-root",
        str(cache),
        "--workspace-root",
        str(workspace),
        "--input-relative",
        "analysis.wav",
        "--output-relative",
        OUTPUT_RELATIVE,
        "--device",
        "cpu",
    ]
    assert runner.calls[0][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert runner.calls[0][1]["timeout"] == 22


def test_minimal_environment_and_no_credentials(roots, monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "/secret.json")
    monkeypatch.setenv("PYTHONPATH", "/inject")
    runner = FakeRunner(ok("runtime-probe", runtime_result()))
    make_client(roots, runner).runtime_probe()
    env = runner.calls[0][1]["env"]
    assert env["PATH"] == "/bin"
    assert env["PYTHONNOUSERSITE"] == "1"
    for key in [
        "HF_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PYTHONPATH",
    ]:
        assert key not in env


def test_shell_false_and_binary_capture(roots):
    runner = FakeRunner(ok("runtime-probe", runtime_result()))
    make_client(roots, runner).runtime_probe()
    kwargs = runner.calls[0][1]
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is False
    assert kwargs["check"] is False


def test_strict_utf8(roots):
    runner = FakeRunner(FakeCompleted(0, b"\xff"))
    with pytest.raises(WorkerProtocolError, match="UTF-8"):
        make_client(roots, runner).runtime_probe()


@pytest.mark.parametrize(
    "stdout",
    [b"banner\n{}", b"{}\n{}", b"[]", b"1", b"null", b"", b"  "],
)
def test_one_object_stdout_enforcement(roots, stdout):
    runner = FakeRunner(FakeCompleted(0, stdout))
    with pytest.raises(WorkerProtocolError):
        make_client(roots, runner).runtime_probe()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_non_finite_json_rejected(roots, number):
    text = (
        '{"protocolVersion":1,"command":"runtime-probe","status":"ok",'
        '"result":{"x":%s},"warnings":[]}' % number
    )
    runner = FakeRunner(FakeCompleted(0, text.encode()))
    with pytest.raises(WorkerProtocolError):
        make_client(roots, runner).runtime_probe()


def test_duplicate_json_key_rejected(roots):
    text = (
        '{"protocolVersion":1,"protocolVersion":1,"command":"runtime-probe",'
        '"status":"ok","result":{},"warnings":[]}'
    )
    with pytest.raises(WorkerProtocolError):
        make_client(roots, FakeRunner(FakeCompleted(0, text.encode()))).runtime_probe()


def test_protocol_version_mismatch(roots):
    payload = json.loads(ok("runtime-probe", runtime_result()).stdout)
    payload["protocolVersion"] = 2
    with pytest.raises(WorkerProtocolError, match="protocol"):
        make_client(
            roots, FakeRunner(FakeCompleted(0, json.dumps(payload).encode()))
        ).runtime_probe()


def test_command_echo_mismatch(roots):
    with pytest.raises(WorkerProtocolError, match="command echo"):
        make_client(roots, FakeRunner(ok("model-probe", runtime_result()))).runtime_probe()


def test_zero_error_inconsistency(roots):
    completed = error(10, "runtime-probe", "RUNTIME_PROFILE_UNPROVISIONED")
    completed.returncode = 0
    with pytest.raises(WorkerProtocolError, match="exit code 0"):
        make_client(roots, FakeRunner(completed)).runtime_probe()


def test_nonzero_ok_inconsistency(roots):
    completed = ok("runtime-probe", runtime_result())
    completed.returncode = 10
    with pytest.raises(WorkerProtocolError, match="nonzero"):
        make_client(roots, FakeRunner(completed)).runtime_probe()


@pytest.mark.parametrize(
    "exit_code,broad",
    [
        (10, "RUNTIME_INCOMPATIBLE"),
        (20, "MODEL_DOWNLOAD_REQUIRED"),
        (21, "MODEL_VERIFICATION_FAILED"),
        (22, "MODEL_DOWNLOAD_FAILED"),
        (30, "INVALID_WORKER_REQUEST"),
        (40, "SEPARATION_FAILED"),
        (41, "CANCELLED"),
        (42, "TIMEOUT"),
        (50, "WORKER_INTERNAL_ERROR"),
    ],
)
def test_stable_exit_code_mapping_allows_detailed_worker_code(
    roots, exit_code, broad
):
    runner = FakeRunner(
        error(exit_code, "runtime-probe", "DETAILED_WORKER_FAILURE")
    )
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).runtime_probe()
    assert caught.value.detail.code == broad
    assert caught.value.detail.worker_code == "DETAILED_WORKER_FAILURE"
    assert caught.value.detail.exit_code == exit_code


def test_exact_detailed_exit_10_fixture(roots):
    runner = FakeRunner(
        error(10, "runtime-probe", "RUNTIME_PROFILE_UNPROVISIONED")
    )
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).runtime_probe()
    assert caught.value.code == "RUNTIME_INCOMPATIBLE"
    assert caught.value.detail.worker_code == "RUNTIME_PROFILE_UNPROVISIONED"


def test_exact_detailed_exit_21_fixture(roots):
    runner = FakeRunner(
        error(21, "model-probe", "READINESS_MANIFEST_INVALID")
    )
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).model_probe()
    assert caught.value.code == "MODEL_VERIFICATION_FAILED"
    assert caught.value.detail.worker_code == "READINESS_MANIFEST_INVALID"


@pytest.mark.parametrize(
    "code", ["", "lowercase", "BAD-DASH", "1START", "A" * 129, 7]
)
def test_worker_code_requires_uppercase_safe_structure(roots, code):
    payload = {
        "protocolVersion": 1,
        "command": "runtime-probe",
        "status": "error",
        "error": {"code": code, "message": "failed", "retryable": True},
        "warnings": [],
    }
    with pytest.raises(WorkerProtocolError, match="error code"):
        make_client(
            roots, FakeRunner(FakeCompleted(10, json.dumps(payload).encode()))
        ).runtime_probe()


def test_unsupported_nonzero_exit(roots):
    with pytest.raises(WorkerProtocolError, match="unsupported exit"):
        make_client(
            roots, FakeRunner(error(99, "runtime-probe", "SOMETHING_FAILED"))
        ).runtime_probe()


def test_safe_readiness_manifest_is_validated_then_discarded(roots):
    result = make_client(
        roots, FakeRunner(ok("model-probe", model_result()))
    ).model_probe()
    assert result.offline_ready is True
    assert "readiness" not in result.__slots__


@pytest.mark.parametrize(
    "manifest",
    [
        "/absolute.json",
        "../escape.json",
        "readiness/../escape.json",
        "readiness\\file.json",
        "readiness/./file.json",
        "readiness/\x00file.json",
        "",
    ],
)
def test_unsafe_readiness_manifest_rejected(roots, manifest):
    runner = FakeRunner(
        ok("model-probe", model_result(readinessManifest=manifest))
    )
    with pytest.raises(WorkerProtocolError, match="readinessManifest"):
        make_client(roots, runner).model_probe()


def test_readiness_manifest_is_optional(roots):
    value = model_result()
    value.pop("readinessManifest")
    assert make_client(
        roots, FakeRunner(ok("model-probe", value))
    ).model_probe().offline_ready


@pytest.mark.parametrize(
    "field", ["demucsVersion", "torchVersion", "huggingfaceHubVersion"]
)
def test_model_probe_requires_version_fields(roots, field):
    value = model_result()
    value.pop(field)
    with pytest.raises(WorkerProtocolError):
        make_client(roots, FakeRunner(ok("model-probe", value))).model_probe()


@pytest.mark.parametrize(
    "field,bad",
    [
        ("demucsVersion", "4.0.1"),
        ("modelRepository", "other/model"),
        ("modelRevision", "main"),
        ("checkpointFile", "other.safetensors"),
        ("checkpointSizeBytes", 1),
        ("checkpointSha256", "0" * 64),
        ("offlineReady", False),
    ],
)
def test_model_identity_and_readiness_locked(roots, field, bad):
    runner = FakeRunner(ok("model-probe", model_result(**{field: bad})))
    with pytest.raises(WorkerProtocolError):
        make_client(roots, runner).model_probe()


def test_path_traversal_rejected_before_spawn(roots):
    runner = FakeRunner()
    _, _, workspace = roots
    with pytest.raises(ValueError):
        make_client(roots, runner).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative="stems/runs/../worker-output",
            device="cpu",
            timeout_seconds=1,
        )
    assert runner.calls == []


def test_exact_analysis_wav_required(roots):
    runner = FakeRunner()
    _, _, workspace = roots
    with pytest.raises(ValueError, match="analysis.wav"):
        make_client(roots, runner).separate(
            workspace_root=workspace,
            input_relative="source.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


@pytest.mark.parametrize(
    "output",
    [
        f"stems/runs/{RUN_ID}/demucs-output",
        f"stems/runs/{RUN_ID.upper()}/worker-output",
        "stems/runs/short/worker-output",
        "/stems/runs/x/worker-output",
        f"stems\\runs\\{RUN_ID}\\worker-output",
        f"stems/runs/{RUN_ID}/worker-output/extra",
    ],
)
def test_exact_output_structure(roots, output):
    _, _, workspace = roots
    with pytest.raises(ValueError):
        make_client(roots, FakeRunner()).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=output,
            device="cpu",
            timeout_seconds=1,
        )


def test_symlinked_input_rejected(roots, tmp_path):
    _, _, workspace = roots
    (workspace / "analysis.wav").unlink()
    target = tmp_path / "outside.wav"
    target.write_bytes(b"RIFF")
    try:
        (workspace / "analysis.wav").symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        make_client(roots, FakeRunner()).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


def test_symlinked_output_component_rejected(roots, tmp_path):
    _, _, workspace = roots
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "stems").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        make_client(roots, FakeRunner()).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


@pytest.mark.parametrize("device", ["", "gpu", "CPU", "cuda:0", None])
def test_device_allowlist(roots, device):
    _, _, workspace = roots
    with pytest.raises(ValueError):
        make_client(roots, FakeRunner()).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device=device,
            timeout_seconds=1,
        )


def test_local_timeout_mapping(roots):
    runner = FakeRunner(subprocess.TimeoutExpired(["worker"], 1))
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).runtime_probe()
    assert caught.value.code == "TIMEOUT"
    assert caught.value.detail.worker_code is None


def test_keyboard_cancellation_mapping(roots):
    runner = FakeRunner(KeyboardInterrupt())
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).runtime_probe()
    assert caught.value.code == "CANCELLED"


def test_diagnostic_path_credential_url_and_traceback_redaction(roots):
    worker, cache, _ = roots
    stderr = (
        f"Traceback (most recent call last):\n File \"{worker}\"\n"
        f"authorization=secret HF_TOKEN=abc https://user:pass@example.test/x "
        f"{cache}/file\nRuntimeError: safe explanation"
    ).encode()
    runner = FakeRunner(
        error(
            21,
            "model-probe",
            "CHECKPOINT_HASH_MISMATCH",
            stderr=stderr,
        )
    )
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).model_probe()
    diagnostic = caught.value.detail.diagnostic
    assert diagnostic == "RuntimeError: safe explanation"
    assert "secret" not in diagnostic and str(cache) not in diagnostic


def test_message_sanitization(roots):
    _, cache, _ = roots
    runner = FakeRunner(
        error(
            21,
            "model-probe",
            "MODEL_FAMILY_MISMATCH",
            message=(
                f"token=secret failed at {cache}/manifest "
                "https://example.test/token"
            ),
        )
    )
    with pytest.raises(WorkerCommandError) as caught:
        make_client(roots, runner).model_probe()
    assert "secret" not in caught.value.detail.message
    assert str(cache) not in caught.value.detail.message
    assert "example.test" not in caught.value.detail.message


def test_output_size_limit(roots):
    runner = FakeRunner(FakeCompleted(0, b"x" * 101))
    with pytest.raises(WorkerProtocolError, match="size limit"):
        make_client(roots, runner, max_stdout_bytes=100).runtime_probe()


def test_runner_overflow_flag(roots):
    runner = FakeRunner(FakeCompleted(0, b"{}", stdout_overflow=True))
    with pytest.raises(WorkerProtocolError, match="size limit"):
        make_client(roots, runner).runtime_probe()


def test_normalized_separation_mapping_exact(roots):
    _, cache, workspace = roots
    runner = FakeRunner(ok("separate", separation_result()))
    mapping = make_client(roots, runner)(
        workspace_root=workspace,
        cache_root=cache,
        input_relative="analysis.wav",
        output_relative=OUTPUT_RELATIVE,
        device="cpu",
        timeout_seconds=1,
    )
    assert list(mapping) == [
        "runtimeProfile",
        "workerVersion",
        "demucsVersion",
        "torchVersion",
        "huggingfaceHubVersion",
        "modelRepository",
        "modelRevision",
        "checkpointFile",
        "checkpointSha256",
        "device",
        "outputs",
    ]
    assert mapping["outputs"] == [
        "vocals.wav",
        "bass.wav",
        "drums.wav",
        "other.wav",
    ]
    assert "readinessManifest" not in mapping


def test_callable_cache_must_match(roots, tmp_path):
    _, _, workspace = roots
    with pytest.raises(ValueError, match="does not match"):
        make_client(roots, FakeRunner())(
            workspace_root=workspace,
            cache_root=tmp_path / "other",
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


def test_callable_invokes_only_separate(roots):
    _, cache, workspace = roots
    runner = FakeRunner(ok("separate", separation_result()))
    make_client(roots, runner)(
        workspace_root=workspace,
        cache_root=cache,
        input_relative="analysis.wav",
        output_relative=OUTPUT_RELATIVE,
        device="cpu",
        timeout_seconds=1,
    )
    assert runner.calls[0][0][3] == "separate"


def test_passive_operations_never_prepare_model(roots):
    _, _, workspace = roots
    runner = FakeRunner(
        ok("runtime-probe", runtime_result()),
        ok("model-probe", model_result()),
        ok("verify-model", model_result()),
        ok("separate", separation_result()),
    )
    client = make_client(roots, runner)
    client.runtime_probe()
    client.model_probe()
    client.verify_model()
    client.separate(
        workspace_root=workspace,
        input_relative="analysis.wav",
        output_relative=OUTPUT_RELATIVE,
        device="cpu",
        timeout_seconds=1,
    )
    assert [call[0][3] for call in runner.calls] == [
        "runtime-probe",
        "model-probe",
        "verify-model",
        "separate",
    ]


def test_expected_runtime_profile_enforced(roots):
    runner = FakeRunner(
        ok("runtime-probe", runtime_result(runtimeProfile="other"))
    )
    with pytest.raises(WorkerProtocolError, match="profile"):
        make_client(
            roots,
            runner,
            expected_runtime_profile="linux-cpu-v1",
        ).runtime_probe()


def test_unknown_trust_relevant_model_field_rejected(roots):
    runner = FakeRunner(ok("model-probe", model_result(cachePath="private")))
    with pytest.raises(WorkerProtocolError, match="trust-relevant"):
        make_client(roots, runner).model_probe()


def test_missing_warnings_array_rejected(roots):
    payload = json.loads(ok("runtime-probe", runtime_result()).stdout)
    payload.pop("warnings")
    with pytest.raises(WorkerProtocolError, match="missing"):
        make_client(
            roots, FakeRunner(FakeCompleted(0, json.dumps(payload).encode()))
        ).runtime_probe()


def test_separation_outputs_locked(roots):
    _, _, workspace = roots
    runner = FakeRunner(
        ok("separate", separation_result(outputs=["vocals.wav"]))
    )
    with pytest.raises(WorkerProtocolError, match="outputs"):
        make_client(roots, runner).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


def test_separation_result_extra_field_rejected(roots):
    _, _, workspace = roots
    runner = FakeRunner(
        ok(
            "separate",
            separation_result(readinessManifest="readiness/x.json"),
        )
    )
    with pytest.raises(WorkerProtocolError, match="fields"):
        make_client(roots, runner).separate(
            workspace_root=workspace,
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )
