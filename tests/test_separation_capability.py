from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from app.separation_capability import (
    DEFAULT_CACHE_LABEL,
    MODEL_DISCLOSURE,
    STATE_DOWNLOAD_REQUIRED,
    STATE_READY,
    STATE_RUNTIME_MISSING,
    STATE_UNAVAILABLE,
    SeparationCapability,
    probe_separation_capability,
)
from app.separation_runtime import (
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ModelProbeResult,
    RuntimeMissingError,
    RuntimeProbeResult,
    WorkerCommandError,
    WorkerErrorDetail,
    WorkerProtocolError,
)


class Client:
    def __init__(self, runtime, model):
        self.runtime = runtime
        self.model = model
        self.calls: list[str] = []

    def runtime_probe(self):
        self.calls.append("runtime_probe")
        if isinstance(self.runtime, BaseException):
            raise self.runtime
        return self.runtime

    def model_probe(self):
        self.calls.append("model_probe")
        if isinstance(self.model, BaseException):
            raise self.model
        return self.model

    def prepare_model(self, *args, **kwargs):
        raise AssertionError("prepare_model must never be called")

    def verify_model(self, *args, **kwargs):
        raise AssertionError("verify_model must never be called")

    def separate(self, *args, **kwargs):
        raise AssertionError("separate must never be called")

    def __call__(self, *args, **kwargs):
        raise AssertionError("__call__ must never be called")


def runtime(**overrides):
    values = {
        "runtime_profile": "linux-cpu-v1",
        "worker_version": "1.0.0",
        "python_version": "3.13.14",
        "runtime_lock_source": "profile",
        "demucs_version": DEMUCS_VERSION,
        "torch_version": "2.13.0+cpu",
        "huggingface_hub_version": "1.16.1",
        "safetensors_version": "0.6.2",
        "pyyaml_version": "6.0.3",
        "warnings": (),
    }
    values.update(overrides)
    return RuntimeProbeResult(**values)


def model(**overrides):
    values = {
        "runtime_profile": "linux-cpu-v1",
        "worker_version": "1.0.0",
        "demucs_version": DEMUCS_VERSION,
        "torch_version": "2.13.0+cpu",
        "huggingface_hub_version": "1.16.1",
        "model_repository": MODEL_REPOSITORY,
        "model_revision": MODEL_REVISION,
        "checkpoint_file": CHECKPOINT_FILE,
        "checkpoint_size_bytes": CHECKPOINT_SIZE_BYTES,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "verified_at": "2026-08-03T00:00:00Z",
        "offline_ready": True,
        "warnings": (),
    }
    values.update(overrides)
    return ModelProbeResult(**values)


def error(error_type, code, message="safe backend message"):
    return error_type(WorkerErrorDetail(code=code, message=message, retryable=True))


def probe(client, **overrides):
    values = {"enabled": True, "device": "cpu"}
    values.update(overrides)
    return probe_separation_capability(client, **values)


def all_text(capability):
    return " ".join(
        [
            capability.message,
            capability.disclosure or "",
            *(capability.warnings or ()),
            *[str(value) for value in capability.runtime_payload().values()],
        ]
    )


def test_disabled_returns_none_without_client_calls():
    client = Client(AssertionError("runtime"), AssertionError("model"))
    assert probe_separation_capability(client, enabled=False, device="cpu") is None
    assert client.calls == []


def test_enabled_without_client_is_runtime_missing():
    result = probe(None)
    assert result.state == STATE_RUNTIME_MISSING
    assert result.actionable is False
    assert result.profile is None
    assert result.network_required is None


def test_runtime_and_model_ready():
    client = Client(runtime(), model())
    result = probe(client)
    assert client.calls == ["runtime_probe", "model_probe"]
    assert result.state == STATE_READY
    assert result.actionable is True
    assert result.network_required is False
    assert result.profile == "linux-cpu-v1"


def test_model_download_required_is_actionable_capability_state():
    client = Client(
        runtime(),
        error(WorkerCommandError, "MODEL_DOWNLOAD_REQUIRED", "model preparation required"),
    )
    result = probe(client)
    assert result.state == STATE_DOWNLOAD_REQUIRED
    assert result.actionable is True
    assert result.network_required is True
    assert client.calls == ["runtime_probe", "model_probe"]


def test_missing_executable_maps_to_runtime_missing():
    client = Client(error(RuntimeMissingError, "RUNTIME_MISSING"), model())
    result = probe(client)
    assert result.state == STATE_RUNTIME_MISSING
    assert client.calls == ["runtime_probe"]


@pytest.mark.parametrize(
    "runtime_error",
    [
        error(WorkerCommandError, "RUNTIME_INCOMPATIBLE"),
        error(WorkerProtocolError, "WORKER_PROTOCOL_ERROR"),
    ],
)
def test_runtime_failures_map_to_unavailable_without_model_probe(runtime_error):
    client = Client(runtime_error, model())
    result = probe(client)
    assert result.state == STATE_UNAVAILABLE
    assert result.actionable is False
    assert client.calls == ["runtime_probe"]


def test_model_verification_failure_maps_to_unavailable():
    client = Client(runtime(), error(WorkerCommandError, "MODEL_VERIFICATION_FAILED"))
    result = probe(client)
    assert result.state == STATE_UNAVAILABLE
    assert result.actionable is False
    assert client.calls == ["runtime_probe", "model_probe"]


def test_unexpected_runtime_exception_is_safe_and_does_not_call_model():
    client = Client(RuntimeError("failed at /home/user/cache/runtime-lock.json"), model())
    result = probe(client)
    assert result.state == STATE_UNAVAILABLE
    assert client.calls == ["runtime_probe"]
    assert "/home/user" not in all_text(result)


def test_no_download_or_separation_capable_methods_are_reached():
    client = Client(runtime(), model())
    result = probe(client)
    assert result.state == STATE_READY
    assert client.calls == ["runtime_probe", "model_probe"]


def test_runtime_payload_has_exact_frontend_keys():
    payload = probe(Client(runtime(), model())).runtime_payload()
    assert tuple(payload) == (
        "state",
        "profile",
        "device",
        "modelSource",
        "modelRevision",
        "checkpointSizeBytes",
        "cacheLabel",
        "networkRequired",
        "audioRemainsLocal",
        "disclosure",
    )
    assert "message" not in payload
    assert "warnings" not in payload


def test_exact_audited_public_model_identity():
    result = probe(None)
    assert result.model_source == MODEL_REPOSITORY
    assert result.model_revision == MODEL_REVISION
    assert result.checkpoint_size_bytes == CHECKPOINT_SIZE_BYTES
    assert result.audio_remains_local is True
    assert result.disclosure == MODEL_DISCLOSURE


@pytest.mark.parametrize(
    "state,expected_actionable,expected_network",
    [
        (STATE_READY, True, False),
        (STATE_DOWNLOAD_REQUIRED, True, True),
        (STATE_RUNTIME_MISSING, False, None),
        (STATE_UNAVAILABLE, False, None),
    ],
)
def test_actionable_and_network_semantics(state, expected_actionable, expected_network):
    result = SeparationCapability(
        state=state,
        profile=None,
        device=None,
        model_source=None,
        model_revision=None,
        checkpoint_size_bytes=None,
        cache_label=None,
        network_required=expected_network,
        audio_remains_local=None,
        disclosure=None,
        message="safe",
        warnings=(),
    )
    assert result.actionable is expected_actionable
    assert result.network_required is expected_network


def test_mismatched_or_not_offline_ready_model_is_unavailable():
    for invalid_model in (
        model(offline_ready=False),
        model(model_revision="0" * 40),
        model(checkpoint_size_bytes=1),
        model(runtime_profile="other"),
    ):
        result = probe(Client(runtime(), invalid_model))
        assert result.state == STATE_UNAVAILABLE
        assert result.actionable is False


def test_paths_tokens_and_diagnostics_are_not_exposed():
    warning = (
        "https://token@example.test /home/user/cache/readiness/model.json "
        r"C:\Users\name\runtime-lock.json token=secret"
    )
    client = Client(
        runtime(warnings=(warning,)),
        error(
            WorkerCommandError,
            "MODEL_VERIFICATION_FAILED",
            "Readiness failed at readiness/htdemucs.json in /tmp/cache",
        ),
    )
    result = probe(client, cache_label="/home/user/cache")
    text = all_text(result).lower()
    for secret in (
        "example.test",
        "/home/user",
        "/tmp/cache",
        "c:\\users",
        "runtime-lock.json",
        "readiness/htdemucs.json",
        "token=secret",
    ):
        assert secret not in text
    assert result.cache_label == DEFAULT_CACHE_LABEL


def test_warnings_are_sanitized_deduplicated_and_bounded():
    warnings = tuple(
        f"warning {index} /private/cache/{index} " + ("x" * 400) for index in range(20)
    )
    result = probe(Client(runtime(warnings=warnings), model(warnings=warnings)))
    assert len(result.warnings) == 8
    assert all(len(item) <= 192 for item in result.warnings)
    assert all("/private/cache" not in item for item in result.warnings)


def test_unknown_future_state_remains_serializable():
    result = SeparationCapability(
        state="future_runtime_state_v2",
        profile=None,
        device=None,
        model_source=None,
        model_revision=None,
        checkpoint_size_bytes=None,
        cache_label=None,
        network_required=None,
        audio_remains_local=None,
        disclosure=None,
        message="future",
        warnings=(),
    )
    assert result.runtime_payload()["state"] == "future_runtime_state_v2"
    assert result.actionable is False


def test_capability_is_immutable():
    result = probe(None)
    with pytest.raises(FrozenInstanceError):
        result.state = STATE_READY
