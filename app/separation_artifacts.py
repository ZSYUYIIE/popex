from __future__ import annotations

import math
import os
import re
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Mapping
from urllib.parse import unquote

import soundfile as sf

from app.config import Settings
from app.media import MediaProcessingError, friendly_error, secure_job_dir
from app.separation import (
    STEM_MANIFEST_RELATIVE_PATH,
    StemArtifact,
    StemSeparationError,
    StemSeparationResult,
    load_stem_manifest,
)


_KIND_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_JOB_ID_PATTERN = re.compile(r"[a-f0-9]{32}")


class StemArtifactError(RuntimeError):
    """Base error for published stem detail and artifact resolution failures."""


class StemManifestUnavailableError(StemArtifactError):
    """Raised when no valid published stem manifest is available."""


class StemKindNotFoundError(StemArtifactError):
    """Raised when a requested stem kind is unsafe or absent from the manifest."""


@dataclass(frozen=True, slots=True)
class ResolvedStemArtifact:
    kind: str
    label: str
    path: Path
    file_name: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    media_type: str = "audio/wav"

    @property
    def download_name(self) -> str:
        return f"{self.kind}.wav"


@dataclass(frozen=True, slots=True)
class StemDetails:
    available: bool
    status: str
    model: str | None
    version: str | None
    separated_at: str | None
    warnings: tuple[str, ...]
    stems: tuple[dict[str, object | None], ...]
    error: str | None

    def payload(self) -> dict[str, object | None]:
        return {
            "available": self.available,
            "status": self.status,
            "model": self.model,
            "version": self.version,
            "separatedAt": self.separated_at,
            "warnings": list(self.warnings),
            "stems": [dict(stem) for stem in self.stems],
            "error": self.error,
        }


def load_stem_details(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> StemDetails:
    """Return the frontend details contract for the published stem manifest."""
    _validate_job_id(job_id)
    status = _job_status(job)
    error = _job_error(job, settings)

    if not _has_canonical_pointer(job):
        return _unavailable_details(status=status, error=error)

    result = _load_published_manifest(job_id, settings, required=False)
    if result is None:
        return _unavailable_details(status=status, error=error)

    stems = tuple(_stem_payload(job_id, stem) for stem in result.stems)
    return StemDetails(
        available=True,
        status=status,
        model=result.model_name,
        version=result.separation_version,
        separated_at=_successful_timestamp(job, result.created_at),
        warnings=tuple(_safe_message(item, settings) for item in result.warnings),
        stems=stems,
        error=error,
    )


def resolve_stem_artifact(
    job_id: str,
    kind: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> ResolvedStemArtifact:
    """Resolve one validated manifest stem without accepting a caller path."""
    _validate_job_id(job_id)
    safe_kind = _validate_kind(kind)
    if not _has_canonical_pointer(job):
        raise StemManifestUnavailableError(
            "Published stem artifacts are unavailable."
        )

    result = _load_published_manifest(job_id, settings, required=True)
    assert result is not None
    stem = next(
        (
            item
            for item in result.stems
            if _manifest_kind(item.kind) == safe_kind
        ),
        None,
    )
    if stem is None:
        raise StemKindNotFoundError("The requested stem kind is unavailable.")

    job_dir = _secure_job_root(settings, job_id)
    path = _resolve_and_recheck(job_dir, result, stem)
    return ResolvedStemArtifact(
        kind=stem.kind,
        label=stem.label,
        path=path,
        file_name=stem.file_name,
        size_bytes=stem.size_bytes,
        duration_seconds=stem.duration_seconds,
        sample_rate=stem.sample_rate,
        channels=stem.channels,
    )


def _stem_payload(
    job_id: str,
    stem: StemArtifact,
) -> dict[str, object | None]:
    kind = _manifest_kind(stem.kind)
    return {
        "kind": kind,
        "label": stem.label,
        "fileName": stem.file_name,
        "sizeBytes": stem.size_bytes,
        "durationSeconds": stem.duration_seconds,
        "sampleRate": stem.sample_rate,
        "channels": stem.channels,
        "previewUrl": f"/api/jobs/{job_id}/stems/{kind}/preview",
        "downloadUrl": f"/api/jobs/{job_id}/stems/{kind}/download",
    }


def _manifest_kind(kind: str) -> str:
    try:
        return _validate_kind(kind)
    except StemKindNotFoundError as exc:
        raise StemArtifactError(
            "Published stem artifacts failed validation."
        ) from exc


def _unavailable_details(*, status: str, error: str | None) -> StemDetails:
    return StemDetails(
        available=False,
        status=status,
        model=None,
        version=None,
        separated_at=None,
        warnings=(),
        stems=(),
        error=error,
    )


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise StemArtifactError("The stem artifact request is invalid.")


def _validate_kind(kind: str) -> str:
    if (
        not isinstance(kind, str)
        or not kind
        or kind != kind.strip()
        or unquote(kind) != kind
        or "\x00" in kind
        or "/" in kind
        or "\\" in kind
        or "." in kind
        or not _KIND_PATTERN.fullmatch(kind)
    ):
        raise StemKindNotFoundError("The requested stem kind is unavailable.")
    return kind


def _has_canonical_pointer(job: Mapping[str, Any]) -> bool:
    return job.get("stem_manifest_file_name") == STEM_MANIFEST_RELATIVE_PATH


def _job_status(job: Mapping[str, Any]) -> str:
    value = job.get("separation_status")
    if isinstance(value, str) and value.strip():
        return value
    return "not_started"


def _job_error(job: Mapping[str, Any], settings: Settings) -> str | None:
    value = job.get("separation_error")
    if not isinstance(value, str) or not value.strip():
        return None
    return _safe_message(value, settings)


def _safe_message(value: str, settings: Settings) -> str:
    if "traceback (most recent call last)" in value.lower():
        return "Stem separation failed."
    return friendly_error(value, settings=settings)


def _successful_timestamp(
    job: Mapping[str, Any],
    manifest_created_at: str,
) -> str:
    value = job.get("separated_at")
    if isinstance(value, str) and _is_utc_timestamp(value):
        return value
    return manifest_created_at


def _is_utc_timestamp(value: str) -> bool:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return (
        parsed.tzinfo is not None
        and parsed.utcoffset() == timezone.utc.utcoffset(parsed)
    )


def _load_published_manifest(
    job_id: str,
    settings: Settings,
    *,
    required: bool,
) -> StemSeparationResult | None:
    try:
        result = load_stem_manifest(job_id, settings)
    except (MediaProcessingError, StemSeparationError, OSError) as exc:
        raise StemArtifactError(
            "Published stem artifacts failed validation."
        ) from exc
    if result is None and required:
        raise StemManifestUnavailableError(
            "Published stem artifacts are unavailable."
        )
    return result


def _secure_job_root(settings: Settings, job_id: str) -> Path:
    try:
        resolved_job = secure_job_dir(settings, job_id)
        exports_root = settings.exports_dir.resolve(strict=True)
        lexical_job = exports_root / job_id
        info = lexical_job.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StemArtifactError("Published stem artifacts failed validation.")
        resolved = lexical_job.resolve(strict=True)
        if resolved != resolved_job.resolve(strict=True):
            raise StemArtifactError("Published stem artifacts failed validation.")
        if exports_root not in resolved.parents:
            raise StemArtifactError("Published stem artifacts failed validation.")
        return resolved
    except StemArtifactError:
        raise
    except (MediaProcessingError, OSError, RuntimeError) as exc:
        raise StemArtifactError(
            "Published stem artifacts failed validation."
        ) from exc


def _resolve_and_recheck(
    job_dir: Path,
    result: StemSeparationResult,
    stem: StemArtifact,
) -> Path:
    relative = PurePosixPath(stem.file_name)
    expected = PurePosixPath(
        "stems",
        "runs",
        result.run_id,
        f"{stem.kind}.wav",
    )
    if (
        relative != expected
        or relative.is_absolute()
        or ".." in relative.parts
        or "\\" in stem.file_name
        or "\x00" in stem.file_name
        or relative.name != f"{stem.kind}.wav"
    ):
        raise StemArtifactError("Published stem artifacts failed validation.")

    current = job_dir
    for component in relative.parts[:-1]:
        current = current / component
        _require_contained_directory(current, job_dir)

    path = current / relative.name
    before = _require_contained_regular_file(path, job_dir)
    metadata = _read_wav_metadata(path)
    after = _require_contained_regular_file(path, job_dir)
    if _snapshot(before) != _snapshot(after):
        raise StemArtifactError("Published stem artifacts changed during validation.")

    duration, sample_rate, channels, size_bytes = metadata
    if (
        sample_rate != stem.sample_rate
        or channels != stem.channels
        or size_bytes != stem.size_bytes
        or not math.isclose(
            duration,
            stem.duration_seconds,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
    ):
        raise StemArtifactError(
            "Published stem artifact metadata no longer matches its manifest."
        )
    return path.resolve(strict=True)


def _require_contained_directory(path: Path, job_dir: Path) -> os.stat_result:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise StemArtifactError("Published stem artifacts failed validation.")
        resolved = path.resolve(strict=True)
        root = job_dir.resolve(strict=True)
        if resolved != root and root not in resolved.parents:
            raise StemArtifactError("Published stem artifacts failed validation.")
        return info
    except StemArtifactError:
        raise
    except (OSError, RuntimeError) as exc:
        raise StemArtifactError(
            "Published stem artifacts failed validation."
        ) from exc


def _require_contained_regular_file(
    path: Path,
    job_dir: Path,
) -> os.stat_result:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise StemArtifactError("Published stem artifacts failed validation.")
        resolved = path.resolve(strict=True)
        root = job_dir.resolve(strict=True)
        if root not in resolved.parents:
            raise StemArtifactError("Published stem artifacts failed validation.")
        return info
    except StemArtifactError:
        raise
    except (OSError, RuntimeError) as exc:
        raise StemArtifactError(
            "Published stem artifacts failed validation."
        ) from exc


def _read_wav_metadata(path: Path) -> tuple[float, int, int, int]:
    try:
        info = sf.info(str(path))
        size_bytes = path.stat(follow_symlinks=False).st_size
    except (OSError, RuntimeError, ValueError) as exc:
        raise StemArtifactError(
            "Published stem audio could not be validated."
        ) from exc

    sample_rate = int(info.samplerate)
    channels = int(info.channels)
    frames = int(info.frames)
    duration = frames / sample_rate if sample_rate > 0 else 0.0
    values = (sample_rate, channels, frames, duration, size_bytes)
    if (
        str(info.format).upper() != "WAV"
        or any(value <= 0 for value in values)
        or not all(math.isfinite(float(value)) for value in values)
    ):
        raise StemArtifactError("Published stem audio has invalid metadata.")
    return float(duration), sample_rate, channels, int(size_bytes)


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )
