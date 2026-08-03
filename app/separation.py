from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Mapping, Protocol, TypeAlias
from uuid import uuid4

import soundfile as sf

from app.config import Settings
from app.media import MediaProcessingError, secure_job_dir


STEM_MANIFEST_RELATIVE_PATH = "stems/stem-separation.json"
STEM_MANIFEST_SCHEMA_VERSION = 3
StemKind: TypeAlias = str
REQUIRED_STEM_KINDS: tuple[StemKind, ...] = (
    "vocals",
    "bass",
    "drums",
    "other",
)

AUDITED_MODEL_NAME = "htdemucs"
AUDITED_DEMUCS_VERSION = "4.1.0"
AUDITED_MODEL_REPOSITORY = "adefossez/HTDemucs"
AUDITED_MODEL_REVISION = "bf35a81b663819a8255c8fefee17f9d812b786b5"
AUDITED_CHECKPOINT_FILE = "955717e8.safetensors"
AUDITED_CHECKPOINT_SHA256 = (
    "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
)

_STEM_LABELS = {
    "vocals": "Vocals",
    "bass": "Bass",
    "drums": "Drums",
    "other": "Other",
}
_REQUIRED_OUTPUT_NAMES = tuple(f"{kind}.wav" for kind in REQUIRED_STEM_KINDS)
_ALLOWED_DEVICES = frozenset({"cpu", "cuda", "mps"})
_WORKER_RESULT_KEYS = frozenset(
    {
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
)
_MANIFEST_KEYS = frozenset(
    {
        "schemaVersion",
        "separationVersion",
        "createdAt",
        "sourceAsset",
        "runId",
        "model",
        "stems",
        "warnings",
    }
)
_MODEL_KEYS = frozenset(
    {
        "name",
        "packageVersion",
        "runtimeProfile",
        "workerVersion",
        "torchVersion",
        "huggingfaceHubVersion",
        "repository",
        "revision",
        "checkpointFile",
        "checkpointSha256",
        "weightsIdentifier",
        "device",
    }
)
_STEM_KEYS = frozenset(
    {
        "kind",
        "label",
        "fileName",
        "durationSeconds",
        "sampleRate",
        "channels",
        "sizeBytes",
    }
)
_REPOSITORY_PATTERN = re.compile(
    r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*"
)
_REVISION_PATTERN = re.compile(r"[0-9a-f]{40}")
_CHECKPOINT_FILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_RUNTIME_PROFILE_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_VERSION_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+_-]{0,127}")
_RUN_ID_PATTERN = re.compile(r"[a-f0-9]{32}")


class StemSeparationError(RuntimeError):
    """Raised when worker-backed stem output cannot be produced or validated."""


class WorkerRunner(Protocol):
    def __call__(
        self,
        *,
        workspace_root: Path,
        cache_root: Path,
        input_relative: str,
        output_relative: str,
        device: str,
        timeout_seconds: float,
    ) -> Mapping[str, Any]: ...


StageCallback = Callable[[str, str, float], None]


@dataclass(frozen=True, kw_only=True)
class SeparationOptions:
    separation_version: str
    worker_runner: WorkerRunner
    cache_root: Path
    expected_model_repository: str
    expected_model_revision: str
    expected_checkpoint_file: str
    expected_checkpoint_sha256: str
    expected_demucs_version: str
    expected_runtime_profile: str | None = None
    device: str = "cpu"
    timeout_seconds: float = 1800.0


@dataclass(frozen=True)
class WorkerProvenance:
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

    @property
    def weights_identifier(self) -> str:
        return f"sha256:{self.checkpoint_sha256}"


@dataclass(frozen=True)
class StemArtifact:
    kind: StemKind
    label: str
    file_name: str
    duration_seconds: float
    sample_rate: int
    channels: int
    size_bytes: int


@dataclass(frozen=True)
class StemSeparationResult:
    separation_version: str
    created_at: str
    run_id: str
    provenance: WorkerProvenance
    stems: tuple[StemArtifact, ...]
    warnings: tuple[str, ...]
    manifest_file_name: str
    payload: dict[str, Any]

    @property
    def model_name(self) -> str:
        return AUDITED_MODEL_NAME

    @property
    def package_version(self) -> str:
        return self.provenance.demucs_version

    @property
    def weights_identifier(self) -> str:
        return self.provenance.weights_identifier

    @property
    def device(self) -> str:
        return self.provenance.device


def separate_stems(
    job_id: str,
    settings: Settings,
    options: SeparationOptions,
    *,
    stage_callback: StageCallback | None = None,
) -> StemSeparationResult:
    """Run one trusted worker attempt and atomically publish its validated result."""
    _validate_options(options)
    job_dir = _job_dir(settings, job_id)
    source = job_dir / "analysis.wav"
    if not _regular_file_in(source, job_dir):
        raise StemSeparationError(
            "Analysis audio is missing. Prepare analysis.wav before separating stems."
        )

    cache_root = options.cache_root.expanduser().resolve(strict=False)
    run_id = uuid4().hex
    run_dir = job_dir / "stems" / "runs" / run_id
    worker_output = run_dir / "worker-output"
    output_relative = worker_output.relative_to(job_dir).as_posix()
    published = False

    try:
        _report_stage(
            stage_callback,
            "preparing_separation",
            "Preparing stem separation.",
            3,
        )
        _ensure_directory(worker_output, job_dir)

        _report_stage(
            stage_callback,
            "separating_stems",
            "Separating vocals, bass, drums, and accompaniment.",
            10,
        )
        worker_result = _run_worker(
            options,
            workspace_root=job_dir,
            cache_root=cache_root,
            output_relative=output_relative,
        )

        _report_stage(
            stage_callback,
            "validating_stems",
            "Checking separated audio.",
            85,
        )
        provenance = _parse_worker_result(worker_result, options)
        outputs = _worker_outputs(worker_output, run_dir)
        artifacts: list[StemArtifact] = []
        for kind in REQUIRED_STEM_KINDS:
            destination = run_dir / f"{kind}.wav"
            _move_raw_output(outputs[kind], destination, run_dir)
            artifacts.append(_inspect_stem(destination, kind, job_dir, run_dir))
        shutil.rmtree(worker_output, ignore_errors=True)

        created_at = datetime.now(timezone.utc).isoformat()
        warnings: tuple[str, ...] = ()
        payload = _manifest_payload(
            options=options,
            created_at=created_at,
            run_id=run_id,
            provenance=provenance,
            stems=tuple(artifacts),
            warnings=warnings,
        )

        _report_stage(
            stage_callback,
            "saving_stems",
            "Saving separated stems.",
            95,
        )
        _write_manifest(job_dir, payload)
        published = True
        return StemSeparationResult(
            separation_version=options.separation_version,
            created_at=created_at,
            run_id=run_id,
            provenance=provenance,
            stems=tuple(artifacts),
            warnings=warnings,
            manifest_file_name=STEM_MANIFEST_RELATIVE_PATH,
            payload=payload,
        )
    except StemSeparationError:
        raise
    except OSError as exc:
        raise StemSeparationError(
            "Stem separation could not safely create or publish its output files."
        ) from exc
    finally:
        if not published:
            _remove_failed_run(run_dir, job_dir)


def load_stem_manifest(
    job_id: str,
    settings: Settings,
) -> StemSeparationResult | None:
    """Load and strictly validate the published schema-3 worker result."""
    job_dir = _job_dir(settings, job_id)
    path = job_dir / STEM_MANIFEST_RELATIVE_PATH
    if not path.exists():
        return None
    if not _regular_file_in(path, job_dir):
        raise StemSeparationError("Saved stem manifest is stored at an invalid path.")
    try:
        payload = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise StemSeparationError(
            "Saved stem manifest is unreadable or corrupted."
        ) from exc
    return _parse_manifest(payload, job_dir)


def stem_manifest_path(job_id: str, settings: Settings) -> Path:
    return _job_dir(settings, job_id) / STEM_MANIFEST_RELATIVE_PATH


def _validate_options(options: SeparationOptions) -> None:
    if not isinstance(options, SeparationOptions):
        raise StemSeparationError("Stem-separation options are invalid.")
    if (
        not isinstance(options.separation_version, str)
        or not _VERSION_PATTERN.fullmatch(options.separation_version)
    ):
        raise StemSeparationError("The separation version is malformed.")
    if not callable(options.worker_runner):
        raise StemSeparationError("A trusted stem worker runner is required.")
    if not isinstance(options.cache_root, Path):
        raise StemSeparationError("The model cache root must be a pathlib.Path.")
    if "\x00" in os.fspath(options.cache_root):
        raise StemSeparationError("The model cache root is invalid.")
    if options.device not in _ALLOWED_DEVICES:
        raise StemSeparationError("The separation device must be cpu, cuda, or mps.")
    timeout = options.timeout_seconds
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(float(timeout))
        or timeout <= 0
    ):
        raise StemSeparationError("The stem-separation timeout must be positive.")

    _validate_expected_identity(options)
    if options.expected_runtime_profile is not None:
        _validate_runtime_profile(
            options.expected_runtime_profile,
            context="expected runtime profile",
        )


def _validate_expected_identity(options: SeparationOptions) -> None:
    expected_values = (
        ("model repository", options.expected_model_repository),
        ("model revision", options.expected_model_revision),
        ("checkpoint filename", options.expected_checkpoint_file),
        ("checkpoint SHA-256", options.expected_checkpoint_sha256),
        ("Demucs version", options.expected_demucs_version),
    )
    for label, value in expected_values:
        if not isinstance(value, str) or not value.strip():
            raise StemSeparationError(f"The expected {label} must be a non-empty string.")

    if not _REPOSITORY_PATTERN.fullmatch(options.expected_model_repository):
        raise StemSeparationError("The expected model repository is malformed.")
    if not _REVISION_PATTERN.fullmatch(options.expected_model_revision):
        raise StemSeparationError(
            "The expected model revision must be a full lowercase commit identifier."
        )
    _validate_checkpoint_file(options.expected_checkpoint_file, context="expected")
    _validate_sha256(options.expected_checkpoint_sha256, context="expected")

    if options.expected_demucs_version != AUDITED_DEMUCS_VERSION:
        raise StemSeparationError(
            f"The approved Demucs version is {AUDITED_DEMUCS_VERSION}."
        )
    if options.expected_model_repository != AUDITED_MODEL_REPOSITORY:
        raise StemSeparationError("The configured model repository is not approved.")
    if options.expected_model_revision != AUDITED_MODEL_REVISION:
        raise StemSeparationError("The configured model revision is not approved.")
    if options.expected_checkpoint_file != AUDITED_CHECKPOINT_FILE:
        raise StemSeparationError("The configured model checkpoint is not approved.")
    if options.expected_checkpoint_sha256 != AUDITED_CHECKPOINT_SHA256:
        raise StemSeparationError("The configured checkpoint SHA-256 is not approved.")


def _run_worker(
    options: SeparationOptions,
    *,
    workspace_root: Path,
    cache_root: Path,
    output_relative: str,
) -> Mapping[str, Any]:
    try:
        result = options.worker_runner(
            workspace_root=workspace_root,
            cache_root=cache_root,
            input_relative="analysis.wav",
            output_relative=output_relative,
            device=options.device,
            timeout_seconds=float(options.timeout_seconds),
        )
    except (subprocess.TimeoutExpired, TimeoutError) as exc:
        raise StemSeparationError(
            f"Stem separation timed out after {options.timeout_seconds:g} seconds."
        ) from exc
    except FileNotFoundError as exc:
        raise StemSeparationError(
            "The trusted stem-separation worker is unavailable."
        ) from exc
    except Exception as exc:
        raise StemSeparationError(
            "The trusted stem-separation worker failed."
        ) from exc
    if not isinstance(result, Mapping):
        raise StemSeparationError("The stem worker returned an invalid result.")
    return result


def _parse_worker_result(
    value: Mapping[str, Any],
    options: SeparationOptions,
) -> WorkerProvenance:
    _require_exact_keys(value, _WORKER_RESULT_KEYS, "worker result")

    runtime_profile = _worker_string(value, "runtimeProfile")
    worker_version = _worker_string(value, "workerVersion")
    demucs_version = _worker_string(value, "demucsVersion")
    torch_version = _worker_string(value, "torchVersion")
    huggingface_hub_version = _worker_string(value, "huggingfaceHubVersion")
    for label, version in (
        ("worker version", worker_version),
        ("Demucs version", demucs_version),
        ("PyTorch version", torch_version),
        ("Hugging Face Hub version", huggingface_hub_version),
    ):
        if not _VERSION_PATTERN.fullmatch(version):
            raise StemSeparationError(f"The stem worker returned a malformed {label}.")
    model_repository = _worker_string(value, "modelRepository")
    model_revision = _worker_string(value, "modelRevision")
    checkpoint_file = _worker_string(value, "checkpointFile")
    checkpoint_sha256 = _worker_string(value, "checkpointSha256")
    device = _worker_string(value, "device")

    _validate_runtime_profile(runtime_profile, context="worker runtime profile")
    if demucs_version != AUDITED_DEMUCS_VERSION:
        raise StemSeparationError(
            f"The stem worker must use Demucs {AUDITED_DEMUCS_VERSION}."
        )
    if demucs_version != options.expected_demucs_version:
        raise StemSeparationError("The stem worker returned an unexpected Demucs version.")
    if model_repository != options.expected_model_repository:
        raise StemSeparationError("The stem worker returned an unexpected model repository.")
    if model_revision != options.expected_model_revision:
        raise StemSeparationError("The stem worker returned an unexpected model revision.")
    if checkpoint_file != options.expected_checkpoint_file:
        raise StemSeparationError("The stem worker returned an unexpected checkpoint filename.")
    _validate_checkpoint_file(checkpoint_file, context="worker")
    _validate_sha256(checkpoint_sha256, context="worker")
    if checkpoint_sha256 != options.expected_checkpoint_sha256:
        raise StemSeparationError("The stem worker returned an unexpected checkpoint SHA-256.")
    if device != options.device:
        raise StemSeparationError("The stem worker returned an unexpected device.")
    if device not in _ALLOWED_DEVICES:
        raise StemSeparationError("The stem worker returned an unsupported device.")
    if (
        options.expected_runtime_profile is not None
        and runtime_profile != options.expected_runtime_profile
    ):
        raise StemSeparationError("The stem worker returned an unexpected runtime profile.")

    _validate_worker_outputs_metadata(value.get("outputs"))
    return WorkerProvenance(
        runtime_profile=runtime_profile,
        worker_version=worker_version,
        demucs_version=demucs_version,
        torch_version=torch_version,
        huggingface_hub_version=huggingface_hub_version,
        model_repository=model_repository,
        model_revision=model_revision,
        checkpoint_file=checkpoint_file,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )


def _validate_worker_outputs_metadata(value: Any) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise StemSeparationError("The stem worker returned invalid output names.")
    for name in value:
        relative = PurePosixPath(name)
        if (
            not name
            or relative.is_absolute()
            or relative.name != name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or "\x00" in name
        ):
            raise StemSeparationError("The stem worker returned an unsafe output name.")
    if len(value) != len(set(value)):
        raise StemSeparationError("The stem worker returned duplicate output names.")
    if tuple(value) != _REQUIRED_OUTPUT_NAMES:
        raise StemSeparationError(
            "The stem worker must return exactly vocals, bass, drums, and other WAV outputs in stable order."
        )


def _worker_outputs(output_dir: Path, run_dir: Path) -> dict[StemKind, Path]:
    if not _safe_directory_in(output_dir, run_dir):
        raise StemSeparationError(
            "The stem worker output directory is unsafe or outside the allocated run."
        )
    try:
        children = list(output_dir.iterdir())
    except OSError as exc:
        raise StemSeparationError("The stem worker output directory is unreadable.") from exc

    names = {child.name for child in children}
    expected = set(_REQUIRED_OUTPUT_NAMES)
    if names != expected or len(children) != len(expected):
        raise StemSeparationError(
            "The stem worker output directory does not contain exactly the required WAV files."
        )

    found: dict[StemKind, Path] = {}
    for kind in REQUIRED_STEM_KINDS:
        path = output_dir / f"{kind}.wav"
        if not _regular_file_in(path, output_dir):
            raise StemSeparationError(
                f"The {kind}.wav worker output is missing or stored unsafely."
            )
        found[kind] = path
    return found


def _report_stage(
    callback: StageCallback | None,
    stage: str,
    message: str,
    progress: float,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, message, progress)
    except Exception as exc:
        raise StemSeparationError(
            "Stem-separation progress reporting failed."
        ) from exc


def _job_dir(settings: Settings, job_id: str) -> Path:
    try:
        return secure_job_dir(settings, job_id)
    except MediaProcessingError as exc:
        raise StemSeparationError(str(exc)) from exc


def _move_raw_output(source: Path, destination: Path, run_dir: Path) -> None:
    if not _regular_file_in(source, run_dir):
        raise StemSeparationError(
            "The stem worker produced a file outside the allocated run directory."
        )
    if destination.exists() or destination.is_symlink():
        raise StemSeparationError("The allocated stem destination is not empty.")
    os.replace(source, destination)
    if not _regular_file_in(destination, run_dir):
        raise StemSeparationError("A separated stem was not stored safely.")


def _audio_metadata(
    path: Path,
    kind: StemKind,
    run_dir: Path,
) -> tuple[float, int, int, int]:
    if not _regular_file_in(path, run_dir):
        raise StemSeparationError(f"The {kind}.wav stem is not a safe regular file.")
    try:
        info = sf.info(path)
        size = path.stat(follow_symlinks=False).st_size
    except (OSError, RuntimeError, ValueError) as exc:
        raise StemSeparationError(
            f"The {kind}.wav stem is unreadable or corrupted."
        ) from exc
    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    duration = frames / sample_rate if sample_rate > 0 else 0.0
    numbers = (sample_rate, channels, frames, duration, size)
    if (
        str(info.format).upper() != "WAV"
        or any(value <= 0 for value in numbers)
        or not all(math.isfinite(float(value)) for value in numbers)
    ):
        raise StemSeparationError(f"The {kind}.wav stem has invalid audio metadata.")
    return float(duration), sample_rate, channels, int(size)


def _inspect_stem(
    path: Path,
    kind: StemKind,
    job_dir: Path,
    run_dir: Path,
) -> StemArtifact:
    duration, sample_rate, channels, size = _audio_metadata(path, kind, run_dir)
    return StemArtifact(
        kind=kind,
        label=_STEM_LABELS[kind],
        file_name=path.resolve().relative_to(job_dir.resolve()).as_posix(),
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        size_bytes=size,
    )


def _manifest_payload(
    *,
    options: SeparationOptions,
    created_at: str,
    run_id: str,
    provenance: WorkerProvenance,
    stems: tuple[StemArtifact, ...],
    warnings: tuple[str, ...],
) -> dict[str, Any]:
    return {
        "schemaVersion": STEM_MANIFEST_SCHEMA_VERSION,
        "separationVersion": options.separation_version,
        "createdAt": created_at,
        "sourceAsset": "analysis.wav",
        "runId": run_id,
        "model": {
            "name": AUDITED_MODEL_NAME,
            "packageVersion": provenance.demucs_version,
            "runtimeProfile": provenance.runtime_profile,
            "workerVersion": provenance.worker_version,
            "torchVersion": provenance.torch_version,
            "huggingfaceHubVersion": provenance.huggingface_hub_version,
            "repository": provenance.model_repository,
            "revision": provenance.model_revision,
            "checkpointFile": provenance.checkpoint_file,
            "checkpointSha256": provenance.checkpoint_sha256,
            "weightsIdentifier": provenance.weights_identifier,
            "device": provenance.device,
        },
        "stems": [
            {
                "kind": stem.kind,
                "label": stem.label,
                "fileName": stem.file_name,
                "durationSeconds": stem.duration_seconds,
                "sampleRate": stem.sample_rate,
                "channels": stem.channels,
                "sizeBytes": stem.size_bytes,
            }
            for stem in stems
        ],
        "warnings": list(warnings),
    }


def _write_manifest(job_dir: Path, payload: dict[str, Any]) -> None:
    destination = job_dir / STEM_MANIFEST_RELATIVE_PATH
    _ensure_directory(destination.parent, job_dir)
    temporary = destination.parent / f".stem-separation-{uuid4().hex}.tmp"
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except (OSError, TypeError, ValueError) as exc:
        temporary.unlink(missing_ok=True)
        raise StemSeparationError(
            "Stem separation completed but its manifest could not be saved."
        ) from exc


def _parse_manifest(payload: Any, job_dir: Path) -> StemSeparationResult:
    if not isinstance(payload, dict):
        raise StemSeparationError("Saved stem manifest has an invalid structure.")
    if payload.get("schemaVersion") != STEM_MANIFEST_SCHEMA_VERSION:
        raise StemSeparationError(
            "Saved stem manifest uses an unsupported schema version."
        )
    _require_exact_keys(payload, _MANIFEST_KEYS, "saved stem manifest")

    separation_version = _string(payload, "separationVersion")
    created_at = _string(payload, "createdAt")
    _validate_timestamp(created_at)
    if payload.get("sourceAsset") != "analysis.wav":
        raise StemSeparationError(
            "Saved stem manifest references an unsupported source asset."
        )
    run_id = _string(payload, "runId")
    if not _RUN_ID_PATTERN.fullmatch(run_id):
        raise StemSeparationError("Saved stem manifest has an invalid run identifier.")

    model = payload.get("model")
    if not isinstance(model, dict):
        raise StemSeparationError("Saved stem manifest has invalid model metadata.")
    _require_exact_keys(model, _MODEL_KEYS, "saved model metadata")
    provenance = _parse_manifest_provenance(model)

    warnings = payload.get("warnings")
    if not isinstance(warnings, list) or not all(
        isinstance(item, str) for item in warnings
    ):
        raise StemSeparationError("Saved stem manifest has invalid warnings.")
    stem_values = payload.get("stems")
    if not isinstance(stem_values, list) or len(stem_values) != len(REQUIRED_STEM_KINDS):
        raise StemSeparationError("Saved stem manifest has invalid stem metadata.")

    stems = tuple(_parse_stem(item, run_id, job_dir) for item in stem_values)
    if tuple(stem.kind for stem in stems) != REQUIRED_STEM_KINDS:
        raise StemSeparationError(
            "Saved stem manifest has an invalid required-stem order."
        )
    return StemSeparationResult(
        separation_version=separation_version,
        created_at=created_at,
        run_id=run_id,
        provenance=provenance,
        stems=stems,
        warnings=tuple(warnings),
        manifest_file_name=STEM_MANIFEST_RELATIVE_PATH,
        payload=payload,
    )


def _parse_manifest_provenance(model: dict[str, Any]) -> WorkerProvenance:
    name = _string(model, "name")
    package_version = _string(model, "packageVersion")
    runtime_profile = _string(model, "runtimeProfile")
    worker_version = _string(model, "workerVersion")
    torch_version = _string(model, "torchVersion")
    huggingface_hub_version = _string(model, "huggingfaceHubVersion")
    for label, version in (
        ("workerVersion", worker_version),
        ("packageVersion", package_version),
        ("torchVersion", torch_version),
        ("huggingfaceHubVersion", huggingface_hub_version),
    ):
        if not _VERSION_PATTERN.fullmatch(version):
            raise StemSeparationError(
                f"Saved stem manifest has malformed {label} metadata."
            )
    repository = _string(model, "repository")
    revision = _string(model, "revision")
    checkpoint_file = _string(model, "checkpointFile")
    checkpoint_sha256 = _string(model, "checkpointSha256")
    weights_identifier = _string(model, "weightsIdentifier")
    device = _string(model, "device")

    if name != AUDITED_MODEL_NAME:
        raise StemSeparationError("Saved stem manifest has an unexpected model name.")
    if package_version != AUDITED_DEMUCS_VERSION:
        raise StemSeparationError("Saved stem manifest has an unsupported Demucs version.")
    _validate_runtime_profile(runtime_profile, context="saved runtime profile")
    if repository != AUDITED_MODEL_REPOSITORY:
        raise StemSeparationError("Saved stem manifest has an unexpected model repository.")
    if revision != AUDITED_MODEL_REVISION or not _REVISION_PATTERN.fullmatch(revision):
        raise StemSeparationError("Saved stem manifest has an unexpected model revision.")
    _validate_checkpoint_file(checkpoint_file, context="saved")
    if checkpoint_file != AUDITED_CHECKPOINT_FILE:
        raise StemSeparationError("Saved stem manifest has an unexpected checkpoint filename.")
    _validate_sha256(checkpoint_sha256, context="saved")
    if checkpoint_sha256 != AUDITED_CHECKPOINT_SHA256:
        raise StemSeparationError("Saved stem manifest has an unexpected checkpoint SHA-256.")
    if weights_identifier != f"sha256:{checkpoint_sha256}":
        raise StemSeparationError(
            "Saved stem manifest has inconsistent model weight identity."
        )
    if device not in _ALLOWED_DEVICES:
        raise StemSeparationError("Saved stem manifest has an unsupported device.")

    return WorkerProvenance(
        runtime_profile=runtime_profile,
        worker_version=worker_version,
        demucs_version=package_version,
        torch_version=torch_version,
        huggingface_hub_version=huggingface_hub_version,
        model_repository=repository,
        model_revision=revision,
        checkpoint_file=checkpoint_file,
        checkpoint_sha256=checkpoint_sha256,
        device=device,
    )


def _parse_stem(item: Any, run_id: str, job_dir: Path) -> StemArtifact:
    if not isinstance(item, dict):
        raise StemSeparationError(
            "Saved stem manifest contains invalid stem metadata."
        )
    _require_exact_keys(item, _STEM_KEYS, "saved stem metadata")
    kind = _string(item, "kind")
    label = _string(item, "label")
    file_name = _string(item, "fileName")
    duration = _positive_number(item, "durationSeconds")
    sample_rate = _positive_int(item, "sampleRate")
    channels = _positive_int(item, "channels")
    size = _positive_int(item, "sizeBytes")

    relative = PurePosixPath(file_name)
    expected_parent = PurePosixPath("stems", "runs", run_id)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or "\\" in file_name
        or relative.suffix.lower() != ".wav"
        or relative.parent != expected_parent
        or relative.name != f"{kind}.wav"
    ):
        raise StemSeparationError(
            "Saved stem manifest contains an unsafe or mismatched stem path."
        )
    path = job_dir.joinpath(*relative.parts)
    run_dir = job_dir / "stems" / "runs" / run_id
    actual_duration, actual_rate, actual_channels, actual_size = _audio_metadata(
        path, kind, run_dir
    )
    if (
        actual_rate != sample_rate
        or actual_channels != channels
        or actual_size != size
        or not math.isclose(actual_duration, duration, rel_tol=1e-9, abs_tol=1e-9)
    ):
        raise StemSeparationError(
            "Saved stem manifest metadata does not match the stored stem file."
        )
    return StemArtifact(
        kind=kind,
        label=label,
        file_name=file_name,
        duration_seconds=duration,
        sample_rate=sample_rate,
        channels=channels,
        size_bytes=size,
    )


def _ensure_directory(path: Path, container: Path) -> None:
    """Create a contained directory one component at a time without symlinks."""
    try:
        root = container.resolve(strict=True)
        target = Path(os.path.abspath(path))
        relative = target.relative_to(Path(os.path.abspath(root)))
        if container.is_symlink() or not root.is_dir():
            raise StemSeparationError("Stem output container is unsafe.")
    except (OSError, RuntimeError, ValueError) as exc:
        if isinstance(exc, StemSeparationError):
            raise
        raise StemSeparationError(
            "Stem output directory escaped the job directory."
        ) from exc

    current = root
    for part in relative.parts:
        current = current / part
        try:
            info = current.lstat()
        except FileNotFoundError:
            try:
                current.mkdir()
            except FileExistsError:
                pass
            info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StemSeparationError("Stem output directory is unsafe.")
        resolved = current.resolve(strict=True)
        if resolved != root and root not in resolved.parents:
            raise StemSeparationError(
                "Stem output directory escaped the job directory."
            )


def _safe_directory_in(path: Path, container: Path) -> bool:
    try:
        if path.is_symlink():
            return False
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISDIR(info.st_mode):
            return False
        resolved = path.resolve(strict=True)
        root = container.resolve(strict=True)
        return resolved == root or root in resolved.parents
    except (OSError, RuntimeError):
        return False


def _regular_file_in(path: Path, container: Path) -> bool:
    try:
        if path.is_symlink():
            return False
        info = path.stat(follow_symlinks=False)
        if not stat.S_ISREG(info.st_mode):
            return False
        resolved = path.resolve(strict=True)
        root = container.resolve(strict=True)
        return root in resolved.parents
    except (OSError, RuntimeError):
        return False


def _remove_failed_run(run_dir: Path, job_dir: Path) -> None:
    try:
        root = job_dir.resolve(strict=True)
        lexical_run = Path(os.path.abspath(run_dir))
        expected_parent = root / "stems" / "runs"
        if (
            lexical_run.parent != expected_parent
            or not _RUN_ID_PATTERN.fullmatch(lexical_run.name)
            or not lexical_run.exists()
            or lexical_run.is_symlink()
        ):
            return
        resolved = lexical_run.resolve(strict=True)
        if root in resolved.parents:
            shutil.rmtree(resolved, ignore_errors=True)
    except (OSError, RuntimeError):
        return


def _require_exact_keys(
    mapping: Mapping[str, Any],
    expected: frozenset[str],
    context: str,
) -> None:
    keys = set(mapping.keys())
    if keys != expected:
        raise StemSeparationError(f"The {context} has missing or unknown fields.")


def _worker_string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StemSeparationError(f"The stem worker returned invalid {key} metadata.")
    return value


def _string(mapping: Mapping[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value.strip():
        raise StemSeparationError(f"Saved stem manifest has invalid {key} metadata.")
    return value


def _validate_runtime_profile(value: str, *, context: str) -> None:
    if not _RUNTIME_PROFILE_PATTERN.fullmatch(value):
        raise StemSeparationError(f"The {context} is malformed.")


def _validate_checkpoint_file(value: str, *, context: str) -> None:
    if (
        not _CHECKPOINT_FILE_PATTERN.fullmatch(value)
        or PurePosixPath(value).name != value
        or "/" in value
        or "\\" in value
        or "\x00" in value
    ):
        raise StemSeparationError(f"The {context} checkpoint filename is malformed.")


def _validate_sha256(value: str, *, context: str) -> None:
    if not _SHA256_PATTERN.fullmatch(value):
        raise StemSeparationError(
            f"The {context} checkpoint SHA-256 must be 64 lowercase hexadecimal characters."
        )


def _positive_number(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise StemSeparationError(f"Saved stem manifest has invalid {key} metadata.")
    result = float(value)
    if result <= 0 or not math.isfinite(result):
        raise StemSeparationError(f"Saved stem manifest has invalid {key} metadata.")
    return result


def _positive_int(mapping: Mapping[str, Any], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StemSeparationError(f"Saved stem manifest has invalid {key} metadata.")
    return value


def _validate_timestamp(value: str) -> None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise StemSeparationError(
            "Saved stem manifest has an invalid creation time."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise StemSeparationError("Saved stem manifest creation time is not UTC.")


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")
