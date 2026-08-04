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
class Completed:
    returncode: int
    stdout: bytes
    stderr: bytes = b""
    stdout_overflow: bool = False
    stderr_overflow: bool = False


class Runner:
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


def ok(command: str, result: dict, warnings=()) -> Completed:
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
        ).encode(),
    )


def failure(exit_code: int, command: str, worker_code: str, **overrides) -> Completed:
    detail = {
        "code": worker_code,
        "message": "failed",
        "retryable": True,
    }
    detail.update(overrides.pop("error_overrides", {}))
    return Completed(
        exit_code,
        json.dumps(
            {
                "protocolVersion": 1,
                "command": command,
                "status": "error",
                "error": detail,
                "warnings": [],
            }
        ).encode(),
        overrides.pop("stderr", b""),
        **overrides,
    )


def runtime_result(**overrides):
    result = {
        "runtimeProfile": "linux-cpu-v1",
        "workerVersion": "1.0.0",
        "pythonVersion": "3.13.14",
        "runtimeLockSource": "profile",
        "installedVersions": dict(VERSIONS),
        "lockedVersions": dict(VERSIONS),
        "compatible": True,
    }
    result.update(overrides)
    return result


def model_result(**overrides):
    result = {
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
    result.update(overrides)
    return result


def separation_result(**overrides):
    result = {
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
    result.update(overrides)
    return result


@pytest.fixture
def roots(tmp_path: Path):
    cache = tmp_path / "cache"
    cache.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "analysis.wav").write_bytes(b"RIFFsynthetic")
    return tmp_path / "bin" / "popex-demucs-worker", cache, workspace


def client(roots, runner, **kwargs):
    worker, cache, _ = roots
    return SeparationRuntimeClient(worker, cache, process_runner=runner, **kwargs)


def invoke_separate(instance, workspace, **overrides):
    values = {
        "workspace_root": workspace,
        "input_relative": "analysis.wav",
        "output_relative": OUTPUT_RELATIVE,
        "device": "cpu",
        "timeout_seconds": 5,
    }
    values.update(overrides)
    return instance.separate(**values)


def test_import_has_no_optional_runtime_modules():
    script = r"""
import sys
forbidden = {"demucs", "torch", "huggingface_hub", "safetensors", "yaml"}
before = {name for name in sys.modules if name.split(".", 1)[0] in forbidden}
import app.separation_runtime  # noqa: F401
after = {name for name in sys.modules if name.split(".", 1)[0] in forbidden}
raise SystemExit(0 if after == before else 1)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_result_types_are_immutable():
    result = RuntimeProbeResult(
        "p", "w", "py", "profile", "4.1.0", "t", "h", "s", "y"
    )
    with pytest.raises(FrozenInstanceError):
        result.worker_version = "other"


def test_missing_executable(roots):
    with pytest.raises(RuntimeMissingError) as caught:
        client(roots, Runner(FileNotFoundError("missing"))).runtime_probe()
    assert caught.value.code == "RUNTIME_MISSING"
    assert caught.value.detail.exit_code is None
    assert caught.value.detail.worker_code is None


def test_canonical_nested_runtime_probe(roots):
    result = client(roots, Runner(ok("runtime-probe", runtime_result()))).runtime_probe()
    assert result.runtime_profile == "linux-cpu-v1"
    assert result.runtime_lock_source == "profile"
    assert (
        result.demucs_version,
        result.torch_version,
        result.huggingface_hub_version,
        result.safetensors_version,
        result.pyyaml_version,
    ) == tuple(VERSIONS.values())


@pytest.mark.parametrize("source", ["profile", "bundled"])
def test_valid_runtime_lock_sources(roots, source):
    result = client(
        roots, Runner(ok("runtime-probe", runtime_result(runtimeLockSource=source)))
    ).runtime_probe()
    assert result.runtime_lock_source == source


@pytest.mark.parametrize("source", ["/lock.json", "profile.json", "", "other", None])
def test_invalid_runtime_lock_sources(roots, source):
    with pytest.raises(WorkerProtocolError, match="lock source|runtimeLockSource"):
        client(
            roots, Runner(ok("runtime-probe", runtime_result(runtimeLockSource=source)))
        ).runtime_probe()


def test_installed_locked_mismatch(roots):
    locked = dict(VERSIONS)
    locked["torch"] = "different"
    with pytest.raises(WorkerProtocolError, match="do not match"):
        client(
            roots, Runner(ok("runtime-probe", runtime_result(lockedVersions=locked)))
        ).runtime_probe()


def test_compatible_false(roots):
    with pytest.raises(WorkerProtocolError, match="not compatible"):
        client(
            roots, Runner(ok("runtime-probe", runtime_result(compatible=False)))
        ).runtime_probe()


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
    with pytest.raises(WorkerProtocolError):
        client(
            roots,
            Runner(ok("runtime-probe", runtime_result(**{container: versions}))),
        ).runtime_probe()


def test_runtime_demucs_must_equal_410(roots):
    installed = dict(VERSIONS)
    installed["demucs"] = "4.0.1"
    with pytest.raises(WorkerProtocolError, match="Demucs"):
        client(
            roots,
            Runner(
                ok(
                    "runtime-probe",
                    runtime_result(
                        installedVersions=installed,
                        lockedVersions=dict(installed),
                    ),
                )
            ),
        ).runtime_probe()


def test_exact_probe_arguments(roots):
    runner = Runner(
        ok("runtime-probe", runtime_result()),
        ok("model-probe", model_result()),
    )
    instance = client(roots, runner)
    instance.runtime_probe()
    instance.model_probe()
    worker, cache, _ = roots
    assert runner.calls[0][0] == [str(worker), "--protocol-version", "1", "runtime-probe"]
    assert runner.calls[1][0] == [
        str(worker),
        "--protocol-version",
        "1",
        "model-probe",
        "--cache-root",
        str(cache),
    ]


def test_model_probe_requires_versions_and_discards_manifest(roots):
    result = client(roots, Runner(ok("model-probe", model_result()))).model_probe()
    assert result.demucs_version == "4.1.0"
    assert result.torch_version == VERSIONS["torch"]
    assert result.huggingface_hub_version == VERSIONS["huggingface_hub"]
    assert not hasattr(result, "readiness_manifest")


@pytest.mark.parametrize("field", ["demucsVersion", "torchVersion", "huggingfaceHubVersion"])
def test_model_probe_requires_version_fields(roots, field):
    value = model_result()
    value.pop(field)
    with pytest.raises(WorkerProtocolError):
        client(roots, Runner(ok("model-probe", value))).model_probe()


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
    with pytest.raises(WorkerProtocolError, match="readinessManifest"):
        client(
            roots,
            Runner(ok("model-probe", model_result(readinessManifest=manifest))),
        ).model_probe()


def test_readiness_manifest_optional(roots):
    value = model_result()
    value.pop("readinessManifest")
    assert client(roots, Runner(ok("model-probe", value))).model_probe().offline_ready


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
    with pytest.raises(WorkerProtocolError):
        client(
            roots, Runner(ok("model-probe", model_result(**{field: bad})))
        ).model_probe()


def test_prepare_refuses_before_spawn_without_exact_true(roots):
    for consent in (False, None, 1, "true"):
        runner = Runner()
        with pytest.raises(ModelDownloadConsentRequiredError):
            client(roots, runner).prepare_model(allow_model_download=consent)
        assert not runner.calls


def test_authorized_prepare_and_offline_commands(roots):
    runner = Runner(
        ok("prepare-model", model_result()),
        ok("verify-model", model_result()),
        ok("separate", separation_result()),
    )
    instance = client(roots, runner)
    _, cache, workspace = roots
    instance.prepare_model(allow_model_download=True)
    instance.verify_model()
    invoke_separate(instance, workspace)
    assert [call[0][3] for call in runner.calls] == [
        "prepare-model",
        "verify-model",
        "separate",
    ]
    assert "HF_HUB_OFFLINE" not in runner.calls[0][1]["env"]
    assert runner.calls[1][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert runner.calls[2][1]["env"]["HF_HUB_OFFLINE"] == "1"
    assert runner.calls[0][0][-2:] == ["--cache-root", str(cache)]


def test_minimal_environment_and_shell_false(roots, monkeypatch):
    monkeypatch.setenv("PATH", "/bin")
    for key in (
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "GOOGLE_APPLICATION_CREDENTIALS",
        "PYTHONPATH",
    ):
        monkeypatch.setenv(key, "secret")
    runner = Runner(ok("runtime-probe", runtime_result()))
    client(roots, runner).runtime_probe()
    kwargs = runner.calls[0][1]
    assert kwargs["shell"] is False
    assert kwargs["capture_output"] is True
    assert kwargs["text"] is False
    assert kwargs["check"] is False
    assert kwargs["env"]["PATH"] == "/bin"
    assert kwargs["env"]["PYTHONNOUSERSITE"] == "1"
    assert not set(kwargs["env"]).intersection(
        {
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "AWS_SECRET_ACCESS_KEY",
            "GOOGLE_APPLICATION_CREDENTIALS",
            "PYTHONPATH",
        }
    )


@pytest.mark.parametrize(
    "stdout",
    [b"\xff", b"banner\n{}", b"{}\n{}", b"[]", b"1", b"null", b"", b"  "],
)
def test_strict_utf8_and_one_object_stdout(roots, stdout):
    with pytest.raises(WorkerProtocolError):
        client(roots, Runner(Completed(0, stdout))).runtime_probe()


@pytest.mark.parametrize("number", ["NaN", "Infinity", "-Infinity", "1e9999"])
def test_non_finite_json_rejected(roots, number):
    stdout = (
        '{"protocolVersion":1,"command":"runtime-probe","status":"ok",'
        f'"result":{{"x":{number}}},"warnings":[]}}'
    ).encode()
    with pytest.raises(WorkerProtocolError):
        client(roots, Runner(Completed(0, stdout))).runtime_probe()


def test_duplicate_json_keys_rejected(roots):
    stdout = (
        b'{"protocolVersion":1,"protocolVersion":1,"command":"runtime-probe",'
        b'"status":"ok","result":{},"warnings":[]}'
    )
    with pytest.raises(WorkerProtocolError):
        client(roots, Runner(Completed(0, stdout))).runtime_probe()


@pytest.mark.parametrize("change,match", [("protocol", "protocol"), ("command", "command echo")])
def test_protocol_and_command_mismatch(roots, change, match):
    payload = json.loads(ok("runtime-probe", runtime_result()).stdout)
    if change == "protocol":
        payload["protocolVersion"] = 2
    else:
        payload["command"] = "model-probe"
    with pytest.raises(WorkerProtocolError, match=match):
        client(roots, Runner(Completed(0, json.dumps(payload).encode()))).runtime_probe()


def test_exit_envelope_consistency(roots):
    bad_zero = failure(10, "runtime-probe", "RUNTIME_PROFILE_UNPROVISIONED")
    bad_zero.returncode = 0
    with pytest.raises(WorkerProtocolError, match="exit code 0"):
        client(roots, Runner(bad_zero)).runtime_probe()
    bad_nonzero = ok("runtime-probe", runtime_result())
    bad_nonzero.returncode = 10
    with pytest.raises(WorkerProtocolError, match="nonzero"):
        client(roots, Runner(bad_nonzero)).runtime_probe()


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
def test_exit_code_broad_mapping_preserves_detailed_code(roots, exit_code, broad):
    with pytest.raises(WorkerCommandError) as caught:
        client(
            roots,
            Runner(failure(exit_code, "runtime-probe", "DETAILED_WORKER_FAILURE")),
        ).runtime_probe()
    assert caught.value.code == broad
    assert caught.value.detail.worker_code == "DETAILED_WORKER_FAILURE"
    assert caught.value.detail.exit_code == exit_code


@pytest.mark.parametrize(
    "exit_code,command,worker_code,broad",
    [
        (10, "runtime-probe", "RUNTIME_PROFILE_UNPROVISIONED", "RUNTIME_INCOMPATIBLE"),
        (21, "model-probe", "READINESS_MANIFEST_INVALID", "MODEL_VERIFICATION_FAILED"),
    ],
)
def test_exact_detailed_worker_error_fixtures(roots, exit_code, command, worker_code, broad):
    runner = Runner(failure(exit_code, command, worker_code))
    with pytest.raises(WorkerCommandError) as caught:
        getattr(client(roots, runner), command.replace("-", "_"))()
    assert caught.value.code == broad
    assert caught.value.detail.worker_code == worker_code


@pytest.mark.parametrize("code", ["", "lowercase", "BAD-DASH", "1START", "A" * 129, 7])
def test_worker_code_must_be_uppercase_safe(roots, code):
    completed = failure(10, "runtime-probe", "VALID")
    payload = json.loads(completed.stdout)
    payload["error"]["code"] = code
    completed.stdout = json.dumps(payload).encode()
    with pytest.raises(WorkerProtocolError, match="error code"):
        client(roots, Runner(completed)).runtime_probe()


def test_unsupported_exit_code(roots):
    with pytest.raises(WorkerProtocolError, match="unsupported exit"):
        client(roots, Runner(failure(99, "runtime-probe", "UNKNOWN_FAILURE"))).runtime_probe()


@pytest.mark.parametrize(
    "output",
    [
        f"stems/runs/{RUN_ID}/demucs-output",
        f"stems/runs/{RUN_ID.upper()}/worker-output",
        "stems/runs/short/worker-output",
        "/stems/runs/x/worker-output",
        f"stems\\runs\\{RUN_ID}\\worker-output",
        f"stems/runs/{RUN_ID}/worker-output/extra",
        "stems/runs/../worker-output",
    ],
)
def test_exact_output_structure_and_traversal_rejected_before_spawn(roots, output):
    runner = Runner()
    _, _, workspace = roots
    with pytest.raises(ValueError):
        invoke_separate(client(roots, runner), workspace, output_relative=output)
    assert not runner.calls


def test_exact_analysis_wav_required(roots):
    _, _, workspace = roots
    with pytest.raises(ValueError, match="analysis.wav"):
        invoke_separate(
            client(roots, Runner()), workspace, input_relative="source.wav"
        )


@pytest.mark.parametrize("device", ["", "gpu", "CPU", "cuda:0", None])
def test_device_allowlist(roots, device):
    _, _, workspace = roots
    with pytest.raises(ValueError):
        invoke_separate(client(roots, Runner()), workspace, device=device)


def test_symlink_containment(roots, tmp_path):
    _, _, workspace = roots
    (workspace / "analysis.wav").unlink()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"RIFF")
    try:
        (workspace / "analysis.wav").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError):
        invoke_separate(client(roots, Runner()), workspace)


def test_output_symlink_containment(roots, tmp_path):
    _, _, workspace = roots
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        (workspace / "stems").symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="symlink"):
        invoke_separate(client(roots, Runner()), workspace)


def test_timeout_and_cancellation_mapping(roots):
    for outcome, code in (
        (subprocess.TimeoutExpired(["worker"], 1), "TIMEOUT"),
        (KeyboardInterrupt(), "CANCELLED"),
    ):
        with pytest.raises(WorkerCommandError) as caught:
            client(roots, Runner(outcome)).runtime_probe()
        assert caught.value.code == code
        assert caught.value.detail.worker_code is None


def test_diagnostic_and_message_sanitization(roots):
    worker, cache, _ = roots
    stderr = (
        f'Traceback (most recent call last):\n File "{worker}"\n'
        f"authorization=secret https://user:pass@example.test/x {cache}/file\n"
        "RuntimeError: safe explanation"
    ).encode()
    completed = failure(
        21,
        "model-probe",
        "CHECKPOINT_HASH_MISMATCH",
        stderr=stderr,
        error_overrides={
            "message": f"token=secret at {cache}/manifest https://example.test/token"
        },
    )
    with pytest.raises(WorkerCommandError) as caught:
        client(roots, Runner(completed)).model_probe()
    assert caught.value.detail.diagnostic == "RuntimeError: safe explanation"
    assert "secret" not in caught.value.detail.message
    assert str(cache) not in caught.value.detail.message
    assert "example.test" not in caught.value.detail.message


@pytest.mark.parametrize("overflow", ["length", "flag", "stderr"])
def test_output_size_limits(roots, overflow):
    kwargs = {}
    if overflow == "length":
        completed = Completed(0, b"x" * 101)
    elif overflow == "flag":
        completed = Completed(0, b"{}", stdout_overflow=True)
    else:
        completed = Completed(0, b"{}", b"x" * 101)
        kwargs["max_stderr_bytes"] = 100
    with pytest.raises(WorkerProtocolError, match="size limit"):
        client(
            roots,
            Runner(completed),
            max_stdout_bytes=100,
            **kwargs,
        ).runtime_probe()


def test_normalized_callable_mapping_and_exact_arguments(roots):
    runner = Runner(ok("separate", separation_result()))
    instance = client(roots, runner)
    worker, cache, workspace = roots
    mapping = instance(
        workspace_root=workspace,
        cache_root=cache,
        input_relative="analysis.wav",
        output_relative=OUTPUT_RELATIVE,
        device="cpu",
        timeout_seconds=22,
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
    assert mapping["outputs"] == ["vocals.wav", "bass.wav", "drums.wav", "other.wav"]
    assert "readinessManifest" not in mapping
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
    assert runner.calls[0][1]["timeout"] == 22


def test_callable_cache_must_match(roots, tmp_path):
    _, _, workspace = roots
    with pytest.raises(ValueError, match="does not match"):
        client(roots, Runner())(
            workspace_root=workspace,
            cache_root=tmp_path / "other",
            input_relative="analysis.wav",
            output_relative=OUTPUT_RELATIVE,
            device="cpu",
            timeout_seconds=1,
        )


def test_passive_operations_never_prepare_model(roots):
    _, _, workspace = roots
    runner = Runner(
        ok("runtime-probe", runtime_result()),
        ok("model-probe", model_result()),
        ok("verify-model", model_result()),
        ok("separate", separation_result()),
    )
    instance = client(roots, runner)
    instance.runtime_probe()
    instance.model_probe()
    instance.verify_model()
    invoke_separate(instance, workspace)
    assert [call[0][3] for call in runner.calls] == [
        "runtime-probe",
        "model-probe",
        "verify-model",
        "separate",
    ]


def test_expected_runtime_profile_enforced(roots):
    with pytest.raises(WorkerProtocolError, match="profile"):
        client(
            roots,
            Runner(ok("runtime-probe", runtime_result(runtimeProfile="other"))),
            expected_runtime_profile="linux-cpu-v1",
        ).runtime_probe()


def test_unknown_trust_field_and_missing_warnings_rejected(roots):
    with pytest.raises(WorkerProtocolError, match="trust-relevant"):
        client(
            roots, Runner(ok("model-probe", model_result(cachePath="private")))
        ).model_probe()
    payload = json.loads(ok("runtime-probe", runtime_result()).stdout)
    payload.pop("warnings")
    with pytest.raises(WorkerProtocolError, match="missing"):
        client(roots, Runner(Completed(0, json.dumps(payload).encode()))).runtime_probe()


def test_separation_result_contract_locked(roots):
    _, _, workspace = roots
    for invalid in (
        separation_result(outputs=["vocals.wav"]),
        separation_result(readinessManifest="readiness/x.json"),
        separation_result(device="cuda"),
    ):
        with pytest.raises(WorkerProtocolError):
            invoke_separate(client(roots, Runner(ok("separate", invalid))), workspace)
