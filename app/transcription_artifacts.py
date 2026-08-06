from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from app.config import Settings
from app.media import MediaProcessingError, friendly_error, secure_job_dir
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    RawTranscriptionValidationError,
    load_raw_transcription,
)


_POINTER_FIELD = "transcription_artifact_file_name"
_DOWNLOAD_FILE_NAME = "raw-transcription.json"
_JOB_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
_STATE_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
    r"\s*(?:=|:)?\s*[^\s,;]+"
)
_URL_PATTERN = re.compile(r"(?i)\b(?:https?|file)://[^\s]+")
_ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{6,}")
_MAX_JOB_TEXT_LENGTH = 500


class TranscriptionArtifactError(RuntimeError):
    """A transcription artifact could not be exposed safely."""


class TranscriptionArtifactUnavailableError(TranscriptionArtifactError):
    """A canonical transcription artifact is not currently available."""


@dataclass(frozen=True, slots=True)
class TranscriptionDetails:
    available: bool
    status: str
    stage: str | None
    progress: float
    message: str | None
    transcription_version: str | None
    transcribed_at: str | None
    pitched_event_count: int
    percussion_event_count: int
    aligned_event_count: int
    source_kinds: tuple[str, ...]
    algorithms: dict[str, Any]
    warnings: tuple[str, ...]
    pitched_note_events: tuple[dict[str, Any], ...]
    percussion_events: tuple[dict[str, Any], ...]
    alignment_candidates: tuple[dict[str, Any], ...]
    error: str | None
    download_file_name: str | None

    def payload(self, *, include_events: bool = True) -> dict[str, Any]:
        """Return a fresh API-ready JSON mapping without machine paths."""
        result: dict[str, Any] = {
            "available": self.available,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "version": self.transcription_version,
            "createdAt": self.transcribed_at,
            "counts": {
                "pitched": self.pitched_event_count,
                "percussion": self.percussion_event_count,
                "aligned": self.aligned_event_count,
            },
            "sourceKinds": list(self.source_kinds),
            "algorithms": copy.deepcopy(self.algorithms),
            "warnings": list(self.warnings),
            "error": self.error,
            "downloadFileName": self.download_file_name,
        }
        if include_events:
            result["pitchedNoteEvents"] = copy.deepcopy(
                list(self.pitched_note_events)
            )
            result["percussionEvents"] = copy.deepcopy(
                list(self.percussion_events)
            )
            result["alignmentCandidates"] = copy.deepcopy(
                list(self.alignment_candidates)
            )
        _require_safe_json(result)
        return result


def load_transcription_details(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> TranscriptionDetails:
    """Load the canonical validated raw artifact as a safe musician read model."""
    _validate_job_id(job_id)
    _validate_job_mapping(job)
    state = _job_state(job, settings)

    if not _has_canonical_pointer(job):
        return _unavailable_details(**state)

    artifact = _load_published_artifact(job_id, settings)
    pitched = tuple(
        copy.deepcopy(item) for item in artifact["pitchedNoteEvents"]
    )
    percussion = tuple(
        copy.deepcopy(item) for item in artifact["percussionEvents"]
    )
    alignment = tuple(
        copy.deepcopy(item) for item in artifact["alignmentCandidates"]
    )
    source_kinds = tuple(
        sorted(
            {
                item["sourceKind"]
                for item in (*pitched, *percussion)
            }
        )
    )
    return TranscriptionDetails(
        available=True,
        status=state["status"],
        stage=state["stage"],
        progress=state["progress"],
        message=state["message"],
        transcription_version=artifact["transcriptionVersion"],
        transcribed_at=artifact["createdAt"],
        pitched_event_count=len(pitched),
        percussion_event_count=len(percussion),
        aligned_event_count=sum(
            1 for item in alignment if "alignedTimeSeconds" in item
        ),
        source_kinds=source_kinds,
        algorithms=copy.deepcopy(artifact["algorithms"]),
        warnings=tuple(artifact["warnings"]),
        pitched_note_events=pitched,
        percussion_events=percussion,
        alignment_candidates=alignment,
        error=state["error"],
        download_file_name=_DOWNLOAD_FILE_NAME,
    )


def transcription_json_path(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> Path:
    """Resolve the validated canonical JSON for a trusted download response."""
    _validate_job_id(job_id)
    _validate_job_mapping(job)
    if not _has_canonical_pointer(job):
        raise TranscriptionArtifactUnavailableError(
            "Published raw transcription is unavailable."
        )

    job_dir = _secure_job_root(job_id, settings)
    path, before = _canonical_file_snapshot(job_dir)
    _load_published_artifact(job_id, settings)
    final_path, after = _canonical_file_snapshot(job_dir)
    if path != final_path or before != after:
        raise TranscriptionArtifactError(
            "Published raw transcription changed during validation."
        )
    return final_path.resolve(strict=True)


def _unavailable_details(
    *,
    status: str,
    stage: str | None,
    progress: float,
    message: str | None,
    error: str | None,
) -> TranscriptionDetails:
    return TranscriptionDetails(
        available=False,
        status=status,
        stage=stage,
        progress=progress,
        message=message,
        transcription_version=None,
        transcribed_at=None,
        pitched_event_count=0,
        percussion_event_count=0,
        aligned_event_count=0,
        source_kinds=(),
        algorithms={},
        warnings=(),
        pitched_note_events=(),
        percussion_events=(),
        alignment_candidates=(),
        error=error,
        download_file_name=None,
    )


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise TranscriptionArtifactError(
            "The transcription artifact request is invalid."
        )


def _validate_job_mapping(job: Mapping[str, Any]) -> None:
    if not isinstance(job, Mapping):
        raise TranscriptionArtifactError(
            "The transcription artifact request is invalid."
        )


def _has_canonical_pointer(job: Mapping[str, Any]) -> bool:
    return job.get(_POINTER_FIELD) == RAW_TRANSCRIPTION_RELATIVE_PATH


def _job_state(
    job: Mapping[str, Any],
    settings: Settings,
) -> dict[str, Any]:
    return {
        "status": _safe_state(job.get("transcription_status"), "not_started"),
        "stage": _safe_optional_state(job.get("transcription_stage")),
        "progress": _safe_progress(job.get("transcription_progress")),
        "message": _safe_job_text(
            job.get("transcription_message"),
            settings,
            fallback="Raw transcription status is unavailable.",
        ),
        "error": _safe_job_text(
            job.get("transcription_error"),
            settings,
            fallback="Raw transcription failed.",
        ),
    }


def _safe_state(value: object, fallback: str) -> str:
    if (
        isinstance(value, str)
        and value == value.strip()
        and _STATE_PATTERN.fullmatch(value)
    ):
        return value
    return fallback


def _safe_optional_state(value: object) -> str | None:
    if value is None or value == "":
        return None
    if (
        isinstance(value, str)
        and value == value.strip()
        and _STATE_PATTERN.fullmatch(value)
    ):
        return value
    return None


def _safe_progress(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return 0.0
    number = float(value)
    if not math.isfinite(number):
        return 0.0
    return max(0.0, min(100.0, number))


def _safe_job_text(
    value: object,
    settings: Settings,
    *,
    fallback: str,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    lowered = value.lower()
    if (
        "traceback (most recent call last)" in lowered
        or "stack trace" in lowered
    ):
        return fallback
    try:
        cleaned = friendly_error(value, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return fallback
    cleaned = _SECRET_PATTERN.sub("<redacted>", cleaned)
    cleaned = _URL_PATTERN.sub("<external location>", cleaned)
    cleaned = _ADDRESS_PATTERN.sub("<address>", cleaned)
    cleaned = cleaned.replace("\x00", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return fallback
    if len(cleaned) > _MAX_JOB_TEXT_LENGTH:
        cleaned = cleaned[-_MAX_JOB_TEXT_LENGTH:]
    return cleaned


def _load_published_artifact(
    job_id: str,
    settings: Settings,
) -> dict[str, Any]:
    try:
        artifact = load_raw_transcription(job_id, settings)
    except (
        MediaProcessingError,
        RawTranscriptionError,
        RawTranscriptionValidationError,
        OSError,
        RuntimeError,
    ) as exc:
        raise TranscriptionArtifactError(
            "Published raw transcription failed validation."
        ) from exc
    if artifact is None:
        raise TranscriptionArtifactUnavailableError(
            "Published raw transcription is unavailable."
        )
    return artifact


def _secure_job_root(job_id: str, settings: Settings) -> Path:
    try:
        resolved_job = secure_job_dir(settings, job_id)
        exports_root = settings.exports_dir.resolve(strict=True)
        lexical_job = exports_root / job_id
        info = lexical_job.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        resolved = lexical_job.resolve(strict=True)
        if resolved != resolved_job.resolve(strict=True):
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        if exports_root not in resolved.parents:
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        return resolved
    except TranscriptionArtifactError:
        raise
    except (MediaProcessingError, OSError, RuntimeError) as exc:
        raise TranscriptionArtifactError(
            "Published raw transcription failed validation."
        ) from exc


def _canonical_file_snapshot(
    job_dir: Path,
) -> tuple[Path, tuple[int, int, int, int]]:
    relative = PurePosixPath(RAW_TRANSCRIPTION_RELATIVE_PATH)
    if relative != PurePosixPath("transcription", "raw-events.json"):
        raise TranscriptionArtifactError(
            "Published raw transcription failed validation."
        )
    directory = job_dir / relative.parts[0]
    path = directory / relative.name
    try:
        directory_info = directory.lstat()
        if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(
            directory_info.st_mode
        ):
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        root = job_dir.resolve(strict=True)
        resolved_directory = directory.resolve(strict=True)
        if root not in resolved_directory.parents:
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        resolved_path = path.resolve(strict=True)
        if root not in resolved_path.parents:
            raise TranscriptionArtifactError(
                "Published raw transcription failed validation."
            )
        return path, _snapshot(info)
    except FileNotFoundError as exc:
        raise TranscriptionArtifactUnavailableError(
            "Published raw transcription is unavailable."
        ) from exc
    except TranscriptionArtifactError:
        raise
    except (OSError, RuntimeError) as exc:
        raise TranscriptionArtifactError(
            "Published raw transcription failed validation."
        ) from exc


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_size,
        info.st_mtime_ns,
    )


def _require_safe_json(value: Mapping[str, Any]) -> None:
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise TranscriptionArtifactError(
            "Transcription details could not be serialized safely."
        ) from exc
