"""Strict subprocess client for the optional PopEx Demucs worker.

Only Python's standard library is imported. The base application can therefore
import and test this module without Demucs, PyTorch, Hugging Face Hub, model
weights, or a provisioned worker executable.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Sequence, TypeVar

PROTOCOL_VERSION = 1
MODEL_REPOSITORY = "adefossez/HTDemucs"
MODEL_REVISION = "bf35a81b663819a8255c8fefee17f9d812b786b5"
CHECKPOINT_FILE = "955717e8.safetensors"
CHECKPOINT_SIZE_BYTES = 84_025_440
CHECKPOINT_SHA256 = "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
DEMUCS_VERSION = "4.1.0"
SEPARATION_OUTPUTS = ("vocals.wav", "bass.wav", "drums.wav", "other.wav")

_EXIT_CODE_ERRORS: Mapping[int, str] = {
    10: "RUNTIME_INCOMPATIBLE",
    20: "MODEL_DOWNLOAD_REQUIRED",
    21: "MODEL_VERIFICATION_FAILED",
    22: "MODEL_DOWNLOAD_FAILED",
    30: "INVALID_WORKER_REQUEST",
    40: "SEPARATION_FAILED",
    41: "CANCELLED",
    42: "TIMEOUT",
    50: "WORKER_INTERNAL_ERROR",
}
_COMMANDS = frozenset(
    {"runtime-probe", "model-probe", "prepare-model", "verify-model", "separate"}
)
_DEFAULT_TIMEOUTS: Mapping[str, float] = {
    "runtime-probe": 10.0,
    "model-probe": 10.0,
    "prepare-model": 900.0,
    "verify-model": 180.0,
    "separate": 3_600.0,
}
_ENV_ALLOWLIST = (
    "PATH",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
    "TEMP",
    "TMP",
    "TMPDIR",
    "LD_LIBRARY_PATH",
    "DYLD_LIBRARY_PATH",
    "LANG",
    "LC_ALL",
)
_RUNTIME_VERSION_KEYS = (
    "demucs",
    "torch",
    "huggingface_hub",
    "safetensors",
    "PyYAML",
)
_WORKER_CODE_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")
_OUTPUT_PATTERN = re.compile(r"^stems/runs/(?P<run_id>[0-9a-f]{32})/worker-output$")
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(token|password|passwd|secret|authorization|api[_-]?key|access[_-]?key)\b"
    r"\s*[:=]\s*([^\s,;]+)"
)
_BEARER_RE = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+\-/=]+")
_URL_RE = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?i)\b[A-Z]:[\\/](?:[^\s\"']+)")
_POSIX_PATH_RE = re.compile(r"(?<![:\w])/(?:[^\s\"']+)")
_TRUST_RELEVANT_FIELD = re.compile(
    r"(?i)(protocol|command|status|error|version|profile|repository|revision|"
    r"checkpoint|manifest|model|device|output|path|cache|token|credential|ready|offline|lock)"
)


@dataclass(frozen=True, slots=True)
class WorkerErrorDetail:
    code: str
    message: str
    retryable: bool
    exit_code: int | None = None
    worker_code: str | None = None
    diagnostic: str | None = None


class SeparationRuntimeError(RuntimeError):
    """Base class for all client-visible runtime failures."""

    def __init__(self, detail: WorkerErrorDetail):
        self.detail = detail
        super().__init__(detail.message)

    @property
    def code(self) -> str:
        return self.detail.code


class RuntimeMissingError(SeparationRuntimeError):
    """The configured optional-runtime executable could not be started."""


class WorkerProtocolError(SeparationRuntimeError):
    """The worker response violated the versioned JSON protocol."""


class WorkerCommandError(SeparationRuntimeError):
    """The worker returned a valid structured command failure."""


class ModelDownloadConsentRequiredError(SeparationRuntimeError):
    """Model preparation was requested without explicit local authorization."""


@dataclass(frozen=True, slots=True)
class RuntimeProbeResult:
    runtime_profile: str
    worker_version: str
    python_version: str
    runtime_lock_source: str
    demucs_version: str
    torch_version: str
    huggingface_hub_version: str
    safetensors_version: str
    pyyaml_version: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelProbeResult:
    runtime_profile: str
    worker_version: str
    demucs_version: str
    torch_version: str
    huggingface_hub_version: str
    model_repository: str
    model_revision: str
    checkpoint_file: str
    checkpoint_size_bytes: int
    checkpoint_sha256: str
    verified_at: str
    offline_ready: bool
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelPreparationResult(ModelProbeResult):
    pass


@dataclass(frozen=True, slots=True)
class ModelVerificationResult(ModelProbeResult):
    pass


@dataclass(frozen=True, slots=True)
class WorkerSeparationResult:
    runtime_profile: str
    worker_version: str
    demucs_version: str
    torch_version: str
    huggingface_hub_version: str
    model_repository: str
    model_revision: str
    checkpoint_file: str
    checkpoint_sha256: str
    device: str
    outputs: tuple[str, ...]
    warnings: tuple[str, ...] = ()

    def normalized_mapping(self) -> dict[str, Any]:
        return {
            "runtimeProfile": self.runtime_profile,
            "workerVersion": self.worker_version,
            "demucsVersion": self.demucs_version,
            "torchVersion": self.torch_version,
            "huggingfaceHubVersion": self.huggingface_hub_version,
            "modelRepository": self.model_repository,
            "modelRevision": self.model_revision,
            "checkpointFile": self.checkpoint_file,
            "checkpointSha256": self.checkpoint_sha256,
            "device": self.device,
            "outputs": list(self.outputs),
        }


@dataclass(frozen=True, slots=True)
class _ProcessResult:
    returncode: int
    stdout: bytes
    stderr: bytes
    stdout_overflow: bool = False
    stderr_overflow: bool = False


class _BoundedCollector(threading.Thread):
    def __init__(self, stream: Any, limit: int):
        super().__init__(daemon=True)
        self._stream = stream
        self._limit = limit
        self._data = bytearray()
        self.overflow = False

    def run(self) -> None:
        try:
            while True:
                chunk = self._stream.read(8192)
                if not chunk:
                    return
                remaining = self._limit + 1 - len(self._data)
                if remaining > 0:
                    self._data.extend(chunk[:remaining])
                if len(chunk) > remaining or len(self._data) > self._limit:
                    self.overflow = True
        finally:
            self._stream.close()

    @property
    def data(self) -> bytes:
        return bytes(self._data[: self._limit + 1])


class SeparationRuntimeClient:
    """Launch and strictly validate ``popex-demucs-worker`` protocol v1."""

    def __init__(
        self,
        worker_executable: Path | str,
        cache_root: Path | str,
        *,
        expected_protocol_version: int = PROTOCOL_VERSION,
        command_timeouts: Mapping[str, float] | None = None,
        expected_runtime_profile: str | None = None,
        process_runner: Callable[..., Any] | None = None,
        max_stdout_bytes: int = 1_048_576,
        max_stderr_bytes: int = 65_536,
        diagnostic_limit: int = 2_048,
        termination_grace_seconds: float = 1.0,
    ) -> None:
        if type(expected_protocol_version) is not int or expected_protocol_version <= 0:
            raise ValueError("expected_protocol_version must be a positive integer")
        self._worker_executable = _normalize_executable(worker_executable)
        self._cache_root = _normalize_trusted_root(cache_root, "cache_root")
        self._expected_protocol_version = expected_protocol_version
        self._expected_runtime_profile = _optional_nonempty_string(
            expected_runtime_profile, "expected_runtime_profile"
        )
        self._timeouts = dict(_DEFAULT_TIMEOUTS)
        if command_timeouts is not None:
            unknown = set(command_timeouts) - _COMMANDS
            if unknown:
                raise ValueError(f"unknown command timeout keys: {sorted(unknown)!r}")
            for command, value in command_timeouts.items():
                self._timeouts[command] = _positive_finite(value, f"timeout for {command}")
        for name, value in (
            ("max_stdout_bytes", max_stdout_bytes),
            ("max_stderr_bytes", max_stderr_bytes),
            ("diagnostic_limit", diagnostic_limit),
        ):
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self._max_stdout_bytes = max_stdout_bytes
        self._max_stderr_bytes = max_stderr_bytes
        self._diagnostic_limit = diagnostic_limit
        self._termination_grace_seconds = _positive_finite(
            termination_grace_seconds, "termination_grace_seconds"
        )
        self._process_runner = process_runner or self._run_subprocess

    @property
    def worker_executable(self) -> Path:
        return self._worker_executable

    @property
    def cache_root(self) -> Path:
        return self._cache_root

    def runtime_probe(self) -> RuntimeProbeResult:
        envelope = self._invoke("runtime-probe", (), self._timeouts["runtime-probe"])
        result = _parse_runtime_probe(envelope["result"], envelope["warnings"])
        self._check_runtime_profile(result.runtime_profile)
        return result

    def model_probe(self) -> ModelProbeResult:
        envelope = self._invoke(
            "model-probe",
            ("--cache-root", str(self._cache_root)),
            self._timeouts["model-probe"],
        )
        result = _parse_model_result(envelope["result"], envelope["warnings"], ModelProbeResult)
        self._check_runtime_profile(result.runtime_profile)
        return result

    def prepare_model(self, *, allow_model_download: bool) -> ModelPreparationResult:
        if allow_model_download is not True:
            raise ModelDownloadConsentRequiredError(
                WorkerErrorDetail(
                    code="MODEL_DOWNLOAD_CONSENT_REQUIRED",
                    message="Explicit authorization is required before model preparation.",
                    retryable=True,
                )
            )
        envelope = self._invoke(
            "prepare-model",
            ("--cache-root", str(self._cache_root)),
            self._timeouts["prepare-model"],
        )
        result = _parse_model_result(
            envelope["result"], envelope["warnings"], ModelPreparationResult
        )
        self._check_runtime_profile(result.runtime_profile)
        return result

    def verify_model(self) -> ModelVerificationResult:
        envelope = self._invoke(
            "verify-model",
            ("--cache-root", str(self._cache_root)),
            self._timeouts["verify-model"],
        )
        result = _parse_model_result(
            envelope["result"], envelope["warnings"], ModelVerificationResult
        )
        self._check_runtime_profile(result.runtime_profile)
        return result

    def separate(
        self,
        *,
        workspace_root: Path | str,
        input_relative: str,
        output_relative: str,
        device: str,
        timeout_seconds: float,
    ) -> WorkerSeparationResult:
        workspace = _validate_separation_paths(workspace_root, input_relative, output_relative)
        selected_device = _validate_device(device)
        timeout = _positive_finite(timeout_seconds, "timeout_seconds")
        envelope = self._invoke(
            "separate",
            (
                "--cache-root",
                str(self._cache_root),
                "--workspace-root",
                str(workspace),
                "--input-relative",
                input_relative,
                "--output-relative",
                output_relative,
                "--device",
                selected_device,
            ),
            timeout,
        )
        result = _parse_separation_result(envelope["result"], envelope["warnings"])
        if result.device != selected_device:
            raise _protocol_error("Worker returned a device different from the request.")
        self._check_runtime_profile(result.runtime_profile)
        return result

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
        supplied_cache = _normalize_trusted_root(cache_root, "cache_root")
        if supplied_cache != self._cache_root:
            raise ValueError("cache_root does not match the configured trusted cache root")
        return self.separate(
            workspace_root=workspace_root,
            input_relative=input_relative,
            output_relative=output_relative,
            device=device,
            timeout_seconds=timeout_seconds,
        ).normalized_mapping()

    def _check_runtime_profile(self, actual: str) -> None:
        if self._expected_runtime_profile is not None and actual != self._expected_runtime_profile:
            raise _protocol_error("Worker runtime profile does not match configuration.")

    def _invoke(self, command: str, command_args: Sequence[str], timeout: float) -> dict[str, Any]:
        if command not in _COMMANDS:
            raise ValueError(f"unsupported command: {command}")
        argv = [
            str(self._worker_executable),
            "--protocol-version",
            str(self._expected_protocol_version),
            command,
            *command_args,
        ]
        try:
            completed = self._process_runner(
                argv,
                shell=False,
                capture_output=True,
                text=False,
                timeout=timeout,
                env=self._build_environment(command),
                check=False,
            )
        except (FileNotFoundError, PermissionError, NotADirectoryError, OSError) as exc:
            diagnostic = _sanitize_diagnostic(
                str(exc), (self._worker_executable, self._cache_root), self._diagnostic_limit
            )
            raise RuntimeMissingError(
                WorkerErrorDetail(
                    code="RUNTIME_MISSING",
                    message="The configured separation runtime could not be started.",
                    retryable=True,
                    diagnostic=diagnostic or None,
                )
            ) from None
        except subprocess.TimeoutExpired:
            raise WorkerCommandError(
                WorkerErrorDetail(
                    code="TIMEOUT",
                    message="The separation runtime command timed out.",
                    retryable=True,
                )
            ) from None
        except KeyboardInterrupt:
            raise WorkerCommandError(
                WorkerErrorDetail(
                    code="CANCELLED",
                    message="The separation runtime command was cancelled.",
                    retryable=True,
                )
            ) from None

        returncode = getattr(completed, "returncode", None)
        stdout = getattr(completed, "stdout", None)
        stderr = getattr(completed, "stderr", None)
        if type(returncode) is not int or not isinstance(stdout, bytes) or not isinstance(stderr, bytes):
            raise _protocol_error("Process runner returned an invalid result object.")
        if bool(getattr(completed, "stdout_overflow", False)) or len(stdout) > self._max_stdout_bytes:
            raise _protocol_error("Worker output exceeded the configured size limit.")
        if bool(getattr(completed, "stderr_overflow", False)) or len(stderr) > self._max_stderr_bytes:
            raise _protocol_error("Worker output exceeded the configured size limit.")
        try:
            stdout_text = stdout.decode("utf-8", errors="strict")
            stderr_text = stderr.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            raise _protocol_error("Worker output was not valid UTF-8.") from None

        known_paths = (self._worker_executable, self._cache_root)
        diagnostic = _sanitize_diagnostic(stderr_text, known_paths, self._diagnostic_limit)
        envelope = _parse_envelope(stdout_text, command, self._expected_protocol_version)
        envelope["warnings"] = tuple(
            _sanitize_diagnostic(warning, known_paths, self._diagnostic_limit)
            for warning in envelope["warnings"]
        )
        if returncode == 0:
            if envelope["status"] != "ok":
                raise _protocol_error("Worker returned an error envelope with exit code 0.")
            return envelope
        if envelope["status"] != "error":
            raise _protocol_error("Worker returned an ok envelope with a nonzero exit code.")
        broad_code = _EXIT_CODE_ERRORS.get(returncode)
        if broad_code is None:
            raise _protocol_error("Worker returned an unsupported exit code.")
        error = envelope["error"]
        safe_message = _sanitize_diagnostic(
            error["message"], known_paths, self._diagnostic_limit
        ) or "The separation runtime command failed."
        raise WorkerCommandError(
            WorkerErrorDetail(
                code=broad_code,
                worker_code=error["code"],
                message=safe_message,
                retryable=error["retryable"],
                exit_code=returncode,
                diagnostic=diagnostic or None,
            )
        )

    def _build_environment(self, command: str) -> dict[str, str]:
        environment = {
            name: value
            for name in _ENV_ALLOWLIST
            if (value := os.environ.get(name)) is not None
        }
        environment.update(
            {
                "HF_HOME": str(self._cache_root),
                "HF_HUB_CACHE": str(self._cache_root / "hub"),
                "HF_XET_CACHE": str(self._cache_root / "xet"),
                "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
                "HF_HUB_DISABLE_TELEMETRY": "1",
                "HF_HUB_DISABLE_UPDATE_CHECK": "1",
                "HF_HUB_DISABLE_PROGRESS_BARS": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONIOENCODING": "utf-8",
                "PYTHONUTF8": "1",
            }
        )
        if command in {"verify-model", "separate"}:
            environment["HF_HUB_OFFLINE"] = "1"
        return environment

    def _run_subprocess(self, argv: Sequence[str], **kwargs: Any) -> _ProcessResult:
        if kwargs.get("shell") is not False:
            raise ValueError("worker subprocess must use shell=False")
        process = subprocess.Popen(
            list(argv),
            shell=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=kwargs["env"],
            text=False,
        )
        assert process.stdout is not None and process.stderr is not None
        stdout_collector = _BoundedCollector(process.stdout, self._max_stdout_bytes)
        stderr_collector = _BoundedCollector(process.stderr, self._max_stderr_bytes)
        stdout_collector.start()
        stderr_collector.start()
        try:
            returncode = process.wait(timeout=kwargs["timeout"])
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            process.terminate()
            try:
                process.wait(timeout=self._termination_grace_seconds)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
        finally:
            stdout_collector.join(timeout=self._termination_grace_seconds)
            stderr_collector.join(timeout=self._termination_grace_seconds)
        return _ProcessResult(
            returncode=returncode,
            stdout=stdout_collector.data,
            stderr=stderr_collector.data,
            stdout_overflow=stdout_collector.overflow,
            stderr_overflow=stderr_collector.overflow,
        )


def _normalize_executable(value: Path | str) -> Path:
    text = os.fspath(value)
    if not text or "\x00" in text:
        raise ValueError("worker_executable must be a non-empty local path")
    path = Path(text)
    if not path.is_absolute():
        raise ValueError("worker_executable must be an absolute path")
    return Path(os.path.normpath(str(path)))


def _normalize_trusted_root(value: Path | str, name: str) -> Path:
    text = os.fspath(value)
    if not text or "\x00" in text:
        raise ValueError(f"{name} must be a non-empty local path")
    path = Path(text)
    if not path.is_absolute():
        raise ValueError(f"{name} must be absolute")
    normalized = Path(os.path.normpath(str(path)))
    if path != normalized:
        raise ValueError(f"{name} must be normalized")
    if path.exists() and path.is_symlink():
        raise ValueError(f"{name} must not be a symlink")
    return path.resolve(strict=False)


def _validate_separation_paths(
    workspace_root: Path | str, input_relative: str, output_relative: str
) -> Path:
    workspace = _normalize_trusted_root(workspace_root, "workspace_root")
    if not workspace.is_dir() or workspace.is_symlink():
        raise ValueError("workspace_root must be an existing safe directory")
    if input_relative != "analysis.wav":
        raise ValueError("input_relative must be exactly analysis.wav")
    if not isinstance(output_relative, str) or not output_relative or "\x00" in output_relative or "\\" in output_relative:
        raise ValueError("output_relative must be a safe relative POSIX path")
    pure_output = PurePosixPath(output_relative)
    if pure_output.is_absolute() or any(part in {"", ".", ".."} for part in pure_output.parts) or str(pure_output) != output_relative:
        raise ValueError("output_relative must be a safe relative POSIX path")
    if _OUTPUT_PATTERN.fullmatch(output_relative) is None:
        raise ValueError(
            "output_relative must match stems/runs/{32-lowercase-hex}/worker-output"
        )
    input_path = workspace / "analysis.wav"
    if input_path.is_symlink() or not input_path.is_file():
        raise ValueError("analysis.wav must be an existing regular file")
    if input_path.resolve(strict=True).parent != workspace:
        raise ValueError("analysis.wav escapes the trusted workspace")
    current = workspace
    output_parts = PurePosixPath(output_relative).parts
    for index, component in enumerate(output_parts):
        current = current / component
        if current.exists():
            if current.is_symlink():
                raise ValueError("output path contains a symlink")
            if index < len(output_parts) - 1 and not current.is_dir():
                raise ValueError("output path contains a non-directory component")
    output_path = workspace / output_relative
    if not _is_relative_to(output_path.resolve(strict=False), workspace):
        raise ValueError("output path escapes the trusted workspace")
    if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
        raise ValueError("output directory must be new or empty")
    return workspace


def _safe_relative_posix(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\\" in value:
        raise _protocol_error(f"Worker result field {name} must be a safe relative POSIX path.")
    pure = PurePosixPath(value)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise _protocol_error(f"Worker result field {name} must be a safe relative POSIX path.")
    if str(pure) != value:
        raise _protocol_error(f"Worker result field {name} must be a safe relative POSIX path.")
    return value


def _validate_device(device: str) -> str:
    if device not in {"cpu", "cuda", "mps"}:
        raise ValueError("device must be cpu, cuda, or mps")
    return device


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a positive finite number") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be a positive finite number")
    return number


def _optional_nonempty_string(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string when supplied")
    return value


def _parse_envelope(
    stdout: str, expected_command: str, expected_protocol_version: int
) -> dict[str, Any]:
    if not stdout.strip():
        raise _protocol_error("Worker stdout did not contain a JSON object.")
    try:
        value = json.loads(
            stdout,
            parse_constant=_reject_json_constant,
            parse_float=_finite_json_float,
            object_pairs_hook=_strict_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _protocol_error("Worker stdout was not exactly one valid JSON object.") from exc
    if not isinstance(value, dict):
        raise _protocol_error("Worker JSON must have an object at the top level.")
    common = {"protocolVersion", "command", "status", "warnings"}
    if not common.issubset(value):
        raise _protocol_error("Worker envelope is missing required fields.")
    if type(value["protocolVersion"]) is not int or value["protocolVersion"] != expected_protocol_version:
        raise _protocol_error("Worker protocol version is unsupported.")
    if value["command"] != expected_command:
        raise _protocol_error("Worker command echo does not match the request.")
    if value["status"] not in {"ok", "error"}:
        raise _protocol_error("Worker returned an unknown status.")
    value["warnings"] = _parse_warnings(value["warnings"])
    if value["status"] == "ok":
        if set(value) != common | {"result"} or not isinstance(value["result"], dict):
            raise _protocol_error("Worker success envelope is malformed.")
    else:
        if set(value) != common | {"error"}:
            raise _protocol_error("Worker error envelope contains unknown fields.")
        value["error"] = _parse_worker_error(value["error"])
    return value


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON constant: {value}")


def _finite_json_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("non-finite JSON number")
    return number


def _parse_warnings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise _protocol_error("Worker warnings must be an array of strings.")
    return tuple(value)


def _parse_worker_error(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {"code", "message", "retryable"}:
        raise _protocol_error("Worker error structure is malformed.")
    code = value["code"]
    if not isinstance(code, str) or _WORKER_CODE_RE.fullmatch(code) is None:
        raise _protocol_error("Worker error code is malformed.")
    if not isinstance(value["message"], str) or not value["message"]:
        raise _protocol_error("Worker error message is malformed.")
    if type(value["retryable"]) is not bool:
        raise _protocol_error("Worker error retryable flag is malformed.")
    return value


def _parse_runtime_probe(value: Mapping[str, Any], warnings: tuple[str, ...]) -> RuntimeProbeResult:
    allowed = {
        "runtimeProfile",
        "workerVersion",
        "pythonVersion",
        "runtimeLockSource",
        "installedVersions",
        "lockedVersions",
        "compatible",
    }
    _reject_unknown_trust_fields(value, allowed)
    if set(value) != allowed:
        raise _protocol_error("Worker runtime probe result is missing required fields.")
    lock_source = _required_string(value, "runtimeLockSource")
    if lock_source not in {"profile", "bundled"}:
        raise _protocol_error("Worker runtime lock source is invalid.")
    if value["compatible"] is not True:
        raise _protocol_error("Worker runtime is not compatible with its lock.")
    installed = _version_mapping(value.get("installedVersions"), "installedVersions")
    locked = _version_mapping(value.get("lockedVersions"), "lockedVersions")
    if installed != locked:
        raise _protocol_error("Worker installed versions do not match the runtime lock.")
    if installed["demucs"] != DEMUCS_VERSION:
        raise _protocol_error("Worker reported an unapproved Demucs version.")
    return RuntimeProbeResult(
        runtime_profile=_required_string(value, "runtimeProfile"),
        worker_version=_required_string(value, "workerVersion"),
        python_version=_required_string(value, "pythonVersion"),
        runtime_lock_source=lock_source,
        demucs_version=installed["demucs"],
        torch_version=installed["torch"],
        huggingface_hub_version=installed["huggingface_hub"],
        safetensors_version=installed["safetensors"],
        pyyaml_version=installed["PyYAML"],
        warnings=warnings,
    )


def _version_mapping(value: Any, name: str) -> dict[str, str]:
    if not isinstance(value, dict) or set(value) != set(_RUNTIME_VERSION_KEYS):
        raise _protocol_error(f"Worker result field {name} must contain the exact locked package keys.")
    result: dict[str, str] = {}
    for key in _RUNTIME_VERSION_KEYS:
        item = value[key]
        if not isinstance(item, str) or not item:
            raise _protocol_error(f"Worker result field {name}.{key} must be a non-empty string.")
        result[key] = item
    return result


_ModelResultT = TypeVar(
    "_ModelResultT", ModelProbeResult, ModelPreparationResult, ModelVerificationResult
)


def _parse_model_result(
    value: Mapping[str, Any], warnings: tuple[str, ...], result_type: type[_ModelResultT]
) -> _ModelResultT:
    allowed = {
        "schemaVersion",
        "state",
        "runtimeProfile",
        "workerVersion",
        "demucsVersion",
        "torchVersion",
        "huggingfaceHubVersion",
        "modelRepository",
        "modelRevision",
        "bagFile",
        "bagModelSignatures",
        "checkpointFile",
        "checkpointSizeBytes",
        "checkpointSha256",
        "verifiedAt",
        "offlineReady",
        "readinessManifest",
    }
    _reject_unknown_trust_fields(value, allowed)
    if "schemaVersion" in value and value["schemaVersion"] != 1:
        raise _protocol_error("Worker model result schema is unsupported.")
    if "state" in value and value["state"] != "MODEL_READY":
        raise _protocol_error("Worker model result state is unknown.")
    if "bagFile" in value and value["bagFile"] != "htdemucs.yaml":
        raise _protocol_error("Worker model bag does not match the audited model.")
    if "bagModelSignatures" in value and value["bagModelSignatures"] != ["955717e8"]:
        raise _protocol_error("Worker model signatures do not match the audited bag.")
    if "readinessManifest" in value:
        _safe_relative_posix(value["readinessManifest"], "readinessManifest")
    result = result_type(
        runtime_profile=_required_string(value, "runtimeProfile"),
        worker_version=_required_string(value, "workerVersion"),
        demucs_version=_required_string(value, "demucsVersion"),
        torch_version=_required_string(value, "torchVersion"),
        huggingface_hub_version=_required_string(value, "huggingfaceHubVersion"),
        model_repository=_required_string(value, "modelRepository"),
        model_revision=_required_string(value, "modelRevision"),
        checkpoint_file=_required_string(value, "checkpointFile"),
        checkpoint_size_bytes=_required_int(value, "checkpointSizeBytes"),
        checkpoint_sha256=_required_string(value, "checkpointSha256"),
        verified_at=_required_string(value, "verifiedAt"),
        offline_ready=_required_bool(value, "offlineReady"),
        warnings=warnings,
    )
    _validate_model_identity(result)
    if result.offline_ready is not True:
        raise _protocol_error("Worker reported a model result that is not offline ready.")
    return result


def _parse_separation_result(
    value: Mapping[str, Any], warnings: tuple[str, ...]
) -> WorkerSeparationResult:
    expected = {
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
    }
    if set(value) != expected:
        raise _protocol_error("Worker separation result fields do not match the locked contract.")
    outputs = value["outputs"]
    if not isinstance(outputs, list) or tuple(outputs) != SEPARATION_OUTPUTS:
        raise _protocol_error("Worker separation outputs do not match the locked contract.")
    result = WorkerSeparationResult(
        runtime_profile=_required_string(value, "runtimeProfile"),
        worker_version=_required_string(value, "workerVersion"),
        demucs_version=_required_string(value, "demucsVersion"),
        torch_version=_required_string(value, "torchVersion"),
        huggingface_hub_version=_required_string(value, "huggingfaceHubVersion"),
        model_repository=_required_string(value, "modelRepository"),
        model_revision=_required_string(value, "modelRevision"),
        checkpoint_file=_required_string(value, "checkpointFile"),
        checkpoint_sha256=_required_string(value, "checkpointSha256"),
        device=_required_string(value, "device"),
        outputs=tuple(outputs),
        warnings=warnings,
    )
    _validate_model_identity(result)
    _validate_device(result.device)
    return result


def _validate_model_identity(value: Any) -> None:
    checks = (
        (value.demucs_version, DEMUCS_VERSION, "Demucs version"),
        (value.model_repository, MODEL_REPOSITORY, "model repository"),
        (value.model_revision, MODEL_REVISION, "model revision"),
        (value.checkpoint_file, CHECKPOINT_FILE, "checkpoint file"),
        (value.checkpoint_sha256, CHECKPOINT_SHA256, "checkpoint digest"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise _protocol_error(f"Worker reported an unapproved {label}.")
    if hasattr(value, "checkpoint_size_bytes") and value.checkpoint_size_bytes != CHECKPOINT_SIZE_BYTES:
        raise _protocol_error("Worker reported an unapproved checkpoint size.")


def _reject_unknown_trust_fields(value: Mapping[str, Any], allowed: set[str]) -> None:
    unsafe = sorted(
        field
        for field in set(value) - allowed
        if _TRUST_RELEVANT_FIELD.search(field)
    )
    if unsafe:
        raise _protocol_error(f"Worker result contains unknown trust-relevant fields: {unsafe!r}.")


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise _protocol_error(f"Worker result field {key} must be a non-empty string.")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if type(item) is not int or item < 0:
        raise _protocol_error(f"Worker result field {key} must be a non-negative integer.")
    return item


def _required_bool(value: Mapping[str, Any], key: str) -> bool:
    item = value.get(key)
    if type(item) is not bool:
        raise _protocol_error(f"Worker result field {key} must be a Boolean.")
    return item


def _sanitize_diagnostic(value: str, known_paths: Sequence[Path], limit: int) -> str:
    if not value:
        return ""
    text = value.replace("\x00", "")
    if "Traceback (most recent call last)" in text:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        safe = [
            line
            for line in lines
            if not line.startswith("Traceback")
            and not line.startswith("File ")
            and not line.startswith("at ")
        ]
        text = safe[-1] if safe else "Worker traceback omitted."
    text = _URL_RE.sub("<url>", text)
    text = _BEARER_RE.sub("Bearer [REDACTED]", text)
    text = _CREDENTIAL_RE.sub(lambda match: f"{match.group(1)}=[REDACTED]", text)
    for path in sorted({str(path) for path in known_paths}, key=len, reverse=True):
        if path:
            text = text.replace(path, "<path>")
    text = _WINDOWS_PATH_RE.sub("<path>", text)
    text = _POSIX_PATH_RE.sub("<path>", text)
    text = " ".join(text.split())
    if len(text) > limit:
        text = text[: max(0, limit - 1)] + "…"
    return text


def _protocol_error(message: str) -> WorkerProtocolError:
    return WorkerProtocolError(
        WorkerErrorDetail(code="WORKER_PROTOCOL_ERROR", message=message, retryable=False)
    )


__all__ = [
    "CHECKPOINT_FILE",
    "CHECKPOINT_SHA256",
    "CHECKPOINT_SIZE_BYTES",
    "DEMUCS_VERSION",
    "MODEL_REPOSITORY",
    "MODEL_REVISION",
    "ModelDownloadConsentRequiredError",
    "ModelPreparationResult",
    "ModelProbeResult",
    "ModelVerificationResult",
    "PROTOCOL_VERSION",
    "RuntimeMissingError",
    "RuntimeProbeResult",
    "SEPARATION_OUTPUTS",
    "SeparationRuntimeClient",
    "SeparationRuntimeError",
    "WorkerCommandError",
    "WorkerErrorDetail",
    "WorkerProtocolError",
    "WorkerSeparationResult",
]
