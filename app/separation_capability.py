"""Passive, UI-safe capability mapping for optional stem separation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol

from .separation_runtime import (
    CHECKPOINT_FILE,
    CHECKPOINT_SHA256,
    CHECKPOINT_SIZE_BYTES,
    DEMUCS_VERSION,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    ModelProbeResult,
    RuntimeMissingError,
    RuntimeProbeResult,
    SeparationRuntimeError,
    WorkerCommandError,
)

STATE_READY = "ready"
STATE_DOWNLOAD_REQUIRED = "download_required"
STATE_RUNTIME_MISSING = "runtime_missing"
STATE_UNAVAILABLE = "unavailable"

DEFAULT_CACHE_LABEL = "Local PopEx model cache"
MODEL_DISCLOSURE = (
    "The first stem-separation preparation may download the audited htdemucs "
    "checkpoint to a local PopEx model cache. Source audio remains on this device. "
    "Later runs may reuse the verified local cache. Separation is approximate and "
    "may need musical review."
)

_MAX_WARNINGS = 8
_MAX_WARNING_LENGTH = 192
_MAX_MESSAGE_LENGTH = 240
_SAFE_TOKEN_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_URL_RE = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s,;\"']+")
_POSIX_PATH_RE = re.compile(r"(?<![:\w])/(?:[^\s,;\"']+)")
_RELATIVE_PATH_RE = re.compile(
    r"(?i)(?<!\w)(?:[A-Za-z0-9._-]+[\\/])+(?:[A-Za-z0-9._-]+)"
)
_SENSITIVE_ASSIGNMENT_RE = re.compile(
    r"(?i)\b(?:token|password|passwd|secret|authorization|api[_-]?key|"
    r"access[_-]?key|runtime[_-]?lock|cache[_-]?root)\b\s*[:=]\s*[^\s,;]+"
)
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_SPACE_RE = re.compile(r"\s+")


class RuntimeProbeClient(Protocol):
    def runtime_probe(self) -> RuntimeProbeResult: ...

    def model_probe(self) -> ModelProbeResult: ...


@dataclass(frozen=True, slots=True)
class SeparationCapability:
    state: str
    profile: str | None
    device: str | None
    model_source: str | None
    model_revision: str | None
    checkpoint_size_bytes: int | None
    cache_label: str | None
    network_required: bool | None
    audio_remains_local: bool | None
    disclosure: str | None
    message: str
    warnings: tuple[str, ...]

    @property
    def actionable(self) -> bool:
        return self.state in {STATE_READY, STATE_DOWNLOAD_REQUIRED}

    def runtime_payload(self) -> dict[str, object | None]:
        return {
            "state": self.state,
            "profile": self.profile,
            "device": self.device,
            "modelSource": self.model_source,
            "modelRevision": self.model_revision,
            "checkpointSizeBytes": self.checkpoint_size_bytes,
            "cacheLabel": self.cache_label,
            "networkRequired": self.network_required,
            "audioRemainsLocal": self.audio_remains_local,
            "disclosure": self.disclosure,
        }


def probe_separation_capability(
    client: RuntimeProbeClient | None,
    *,
    enabled: bool,
    device: str,
    cache_label: str = DEFAULT_CACHE_LABEL,
) -> SeparationCapability | None:
    """Passively probe runtime/model readiness and return a UI-safe state."""

    if enabled is not True:
        return None

    safe_device = _safe_token(device)
    safe_cache_label = _safe_cache_label(cache_label)
    if client is None:
        return _capability(
            STATE_RUNTIME_MISSING,
            profile=None,
            device=safe_device,
            cache_label=safe_cache_label,
            message="The optional stem-separation runtime is not installed or could not be started.",
        )

    try:
        runtime = client.runtime_probe()
    except RuntimeMissingError:
        return _capability(
            STATE_RUNTIME_MISSING,
            profile=None,
            device=safe_device,
            cache_label=safe_cache_label,
            message="The optional stem-separation runtime is not installed or could not be started.",
        )
    except SeparationRuntimeError as exc:
        return _unavailable(exc, None, safe_device, safe_cache_label)
    except Exception:
        return _unavailable(None, None, safe_device, safe_cache_label)

    if not isinstance(runtime, RuntimeProbeResult):
        return _unavailable(None, None, safe_device, safe_cache_label)

    profile = _safe_token(runtime.runtime_profile)
    runtime_warnings = _sanitize_warnings(runtime.warnings)

    try:
        model = client.model_probe()
    except RuntimeMissingError:
        return _capability(
            STATE_RUNTIME_MISSING,
            profile=profile,
            device=safe_device,
            cache_label=safe_cache_label,
            message="The optional stem-separation runtime is not installed or could not be started.",
            warnings=runtime_warnings,
        )
    except WorkerCommandError as exc:
        if exc.code == "MODEL_DOWNLOAD_REQUIRED":
            return _capability(
                STATE_DOWNLOAD_REQUIRED,
                profile=profile,
                device=safe_device,
                cache_label=safe_cache_label,
                message=(
                    "The audited htdemucs model is not prepared yet. Explicit consent "
                    "is required before the first local-cache download."
                ),
                warnings=runtime_warnings,
            )
        return _unavailable(exc, profile, safe_device, safe_cache_label, runtime_warnings)
    except SeparationRuntimeError as exc:
        return _unavailable(exc, profile, safe_device, safe_cache_label, runtime_warnings)
    except Exception:
        return _unavailable(None, profile, safe_device, safe_cache_label, runtime_warnings)

    if not isinstance(model, ModelProbeResult) or not _is_audited_ready_model(runtime, model):
        return _capability(
            STATE_UNAVAILABLE,
            profile=profile,
            device=safe_device,
            cache_label=safe_cache_label,
            message=(
                "The separation runtime reported model readiness that does not match "
                "the audited local model."
            ),
            warnings=_sanitize_warnings(runtime_warnings, getattr(model, "warnings", ())),
        )

    return _capability(
        STATE_READY,
        profile=_safe_token(model.runtime_profile) or profile,
        device=safe_device,
        cache_label=safe_cache_label,
        message="Stem separation is ready with the verified local htdemucs model.",
        warnings=_sanitize_warnings(runtime_warnings, model.warnings),
    )


def _is_audited_ready_model(runtime: RuntimeProbeResult, model: ModelProbeResult) -> bool:
    return (
        model.offline_ready is True
        and model.runtime_profile == runtime.runtime_profile
        and model.worker_version == runtime.worker_version
        and model.demucs_version == DEMUCS_VERSION
        and model.demucs_version == runtime.demucs_version
        and model.model_repository == MODEL_REPOSITORY
        and model.model_revision == MODEL_REVISION
        and model.checkpoint_file == CHECKPOINT_FILE
        and model.checkpoint_size_bytes == CHECKPOINT_SIZE_BYTES
        and model.checkpoint_sha256 == CHECKPOINT_SHA256
    )


def _capability(
    state: str,
    *,
    profile: str | None,
    device: str | None,
    cache_label: str,
    message: str,
    warnings: tuple[str, ...] = (),
) -> SeparationCapability:
    return SeparationCapability(
        state=state,
        profile=profile,
        device=device,
        model_source=MODEL_REPOSITORY,
        model_revision=MODEL_REVISION,
        checkpoint_size_bytes=CHECKPOINT_SIZE_BYTES,
        cache_label=cache_label,
        network_required=(
            True
            if state == STATE_DOWNLOAD_REQUIRED
            else False if state == STATE_READY else None
        ),
        audio_remains_local=True,
        disclosure=MODEL_DISCLOSURE,
        message=_safe_text(message, "Stem separation is unavailable.", _MAX_MESSAGE_LENGTH),
        warnings=_sanitize_warnings(warnings),
    )


def _unavailable(
    error: SeparationRuntimeError | None,
    profile: str | None,
    device: str | None,
    cache_label: str,
    warnings: tuple[str, ...] = (),
) -> SeparationCapability:
    message = "Stem separation is temporarily unavailable."
    if error is not None:
        message = _safe_text(str(error), message, _MAX_MESSAGE_LENGTH)
    return _capability(
        STATE_UNAVAILABLE,
        profile=profile,
        device=device,
        cache_label=cache_label,
        message=message,
        warnings=warnings,
    )


def _safe_token(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    candidate = value.strip()
    return candidate if _SAFE_TOKEN_RE.fullmatch(candidate) else None


def _safe_cache_label(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        return DEFAULT_CACHE_LABEL
    candidate = _safe_text(value, DEFAULT_CACHE_LABEL, 80)
    if candidate == DEFAULT_CACHE_LABEL or "[redacted]" in candidate:
        return DEFAULT_CACHE_LABEL
    return candidate


def _sanitize_warnings(*groups: object) -> tuple[str, ...]:
    sanitized: list[str] = []
    for group in groups:
        if isinstance(group, str):
            values = (group,)
        elif isinstance(group, (tuple, list)):
            values = group
        else:
            continue
        for value in values:
            warning = _safe_text(value, "", _MAX_WARNING_LENGTH)
            if warning and warning not in sanitized:
                sanitized.append(warning)
            if len(sanitized) >= _MAX_WARNINGS:
                return tuple(sanitized)
    return tuple(sanitized)


def _safe_text(value: object, fallback: str, limit: int) -> str:
    if not isinstance(value, str):
        return fallback
    text = _CONTROL_RE.sub(" ", value)
    text = _URL_RE.sub("[redacted]", text)
    text = _SENSITIVE_ASSIGNMENT_RE.sub("[redacted]", text)
    text = _WINDOWS_PATH_RE.sub("[redacted]", text)
    text = _POSIX_PATH_RE.sub("[redacted]", text)
    text = _RELATIVE_PATH_RE.sub("[redacted]", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if not text:
        return fallback
    if len(text) > limit:
        text = text[: limit - 1].rstrip() + "…"
    return text
