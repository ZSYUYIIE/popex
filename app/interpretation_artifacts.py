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
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    TranscriptionDraftError,
    load_transcription_draft,
)


_POINTER_FIELD = "interpretation_artifact_file_name"
_DOWNLOAD_FILE_NAME = "editable-interpretation.json"
_JOB_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
_STATE_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_SECRET_PATTERN = re.compile(
    r"(?i)\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
    r"\s*(?:=|:)?\s*[^\s,;]+"
)
_URL_PATTERN = re.compile(r"(?i)\b(?:https?|file)://[^\s]+")
_ADDRESS_PATTERN = re.compile(r"(?i)0x[0-9a-f]{6,}")
_WINDOWS_PATH_PATTERN = re.compile(r"(?i)\b[A-Z]:[\\/][^\s,;]+")
_UNIX_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z0-9])/(?:home|users|tmp|var|etc|mnt|volumes|private|opt|usr)"
    r"(?:/[^\s,;]+)+"
)
_MAX_JOB_TEXT_LENGTH = 500


class InterpretationArtifactError(RuntimeError):
    """An editable interpretation artifact could not be exposed safely."""


class InterpretationArtifactUnavailableError(InterpretationArtifactError):
    """A canonical editable interpretation artifact is not currently available."""


@dataclass(frozen=True, slots=True)
class InterpretationDetails:
    available: bool
    status: str
    stage: str | None
    progress: float
    message: str | None
    version: str | None
    created_at: str | None
    part_count: int
    voice_count: int
    measure_count: int
    phrase_count: int
    pitched_item_count: int
    percussion_item_count: int
    warning_count: int
    unassigned_pitched_count: int
    unplaced_percussion_count: int
    source_kinds: tuple[str, ...]
    algorithms: dict[str, Any]
    warnings: tuple[str, ...]
    parts: tuple[dict[str, Any], ...]
    voices: tuple[dict[str, Any], ...]
    measures: tuple[dict[str, Any], ...]
    phrases: tuple[dict[str, Any], ...]
    pitched_items: tuple[dict[str, Any], ...]
    percussion_items: tuple[dict[str, Any], ...]
    interpretation_evidence: dict[str, Any]
    error: str | None
    download_file_name: str | None

    def payload(self, *, include_items: bool = True) -> dict[str, Any]:
        """Return a fresh API-ready mapping derived from validated artifact truth."""
        result: dict[str, Any] = {
            "available": self.available,
            "status": self.status,
            "stage": self.stage,
            "progress": self.progress,
            "message": self.message,
            "version": self.version,
            "createdAt": self.created_at,
            "counts": {
                "parts": self.part_count,
                "voices": self.voice_count,
                "measures": self.measure_count,
                "phrases": self.phrase_count,
                "pitched": self.pitched_item_count,
                "percussion": self.percussion_item_count,
                "warnings": self.warning_count,
                "unassignedPitched": self.unassigned_pitched_count,
                "unplacedPercussion": self.unplaced_percussion_count,
            },
            "sourceKinds": list(self.source_kinds),
            "algorithms": copy.deepcopy(self.algorithms),
            "warnings": list(self.warnings),
            "error": self.error,
            "downloadFileName": self.download_file_name,
        }
        if include_items:
            result.update(
                parts=copy.deepcopy(list(self.parts)),
                voices=copy.deepcopy(list(self.voices)),
                measures=copy.deepcopy(list(self.measures)),
                phrases=copy.deepcopy(list(self.phrases)),
                pitchedItems=copy.deepcopy(list(self.pitched_items)),
                percussionItems=copy.deepcopy(list(self.percussion_items)),
                interpretationEvidence=copy.deepcopy(self.interpretation_evidence),
            )
        _require_safe_json(result)
        return result


def load_interpretation_details(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> InterpretationDetails:
    """Load a safe read model whose musical facts come from the validated draft."""
    _validate_job_id(job_id)
    _validate_job_mapping(job)
    state = _job_state(job, settings)

    if not _has_canonical_pointer(job):
        return _unavailable_details(**state)

    artifact = _load_published_artifact(job_id, settings)
    parts = tuple(copy.deepcopy(item) for item in artifact["parts"])
    voices = tuple(copy.deepcopy(item) for item in artifact["voices"])
    measures = tuple(copy.deepcopy(item) for item in artifact["measures"])
    phrases = tuple(copy.deepcopy(item) for item in artifact["phrases"])
    pitched = tuple(copy.deepcopy(item) for item in artifact["pitchedItems"])
    percussion = tuple(copy.deepcopy(item) for item in artifact["percussionItems"])
    warnings = tuple(artifact["warnings"])
    source_event_index = artifact["sourceTranscription"]["sourceEventIndex"]
    source_kinds = tuple(sorted({item["sourceKind"] for item in source_event_index}))

    return InterpretationDetails(
        available=True,
        status=state["status"],
        stage=state["stage"],
        progress=state["progress"],
        message=state["message"],
        version=artifact["draftVersion"],
        created_at=artifact["createdAt"],
        part_count=len(parts),
        voice_count=len(voices),
        measure_count=len(measures),
        phrase_count=len(phrases),
        pitched_item_count=len(pitched),
        percussion_item_count=len(percussion),
        warning_count=len(warnings),
        unassigned_pitched_count=sum(
            1
            for item in pitched
            if item.get("interpretationType") == "unassigned"
            or item.get("placementStatus") != "placed"
        ),
        unplaced_percussion_count=sum(
            1 for item in percussion if item.get("placementStatus") != "placed"
        ),
        source_kinds=source_kinds,
        algorithms=copy.deepcopy(artifact["algorithms"]),
        warnings=warnings,
        parts=parts,
        voices=voices,
        measures=measures,
        phrases=phrases,
        pitched_items=pitched,
        percussion_items=percussion,
        interpretation_evidence=copy.deepcopy(artifact["interpretationEvidence"]),
        error=state["error"],
        download_file_name=_DOWNLOAD_FILE_NAME,
    )


def interpretation_json_path(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> Path:
    """Resolve the canonical draft only if it remains unchanged through validation."""
    _validate_job_id(job_id)
    _validate_job_mapping(job)
    if not _has_canonical_pointer(job):
        raise InterpretationArtifactUnavailableError(
            "Published editable interpretation is unavailable."
        )

    job_dir = _secure_job_root(job_id, settings)
    path, before = _canonical_file_snapshot(job_dir)
    _load_published_artifact(job_id, settings)
    final_path, after = _canonical_file_snapshot(job_dir)
    if path != final_path or before != after:
        raise InterpretationArtifactError(
            "Published editable interpretation changed during validation."
        )
    return final_path.resolve(strict=True)


def _unavailable_details(
    *,
    status: str,
    stage: str | None,
    progress: float,
    message: str | None,
    error: str | None,
) -> InterpretationDetails:
    return InterpretationDetails(
        available=False,
        status=status,
        stage=stage,
        progress=progress,
        message=message,
        version=None,
        created_at=None,
        part_count=0,
        voice_count=0,
        measure_count=0,
        phrase_count=0,
        pitched_item_count=0,
        percussion_item_count=0,
        warning_count=0,
        unassigned_pitched_count=0,
        unplaced_percussion_count=0,
        source_kinds=(),
        algorithms={},
        warnings=(),
        parts=(),
        voices=(),
        measures=(),
        phrases=(),
        pitched_items=(),
        percussion_items=(),
        interpretation_evidence={},
        error=error,
        download_file_name=None,
    )


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise InterpretationArtifactError(
            "The editable interpretation artifact request is invalid."
        )


def _validate_job_mapping(job: Mapping[str, Any]) -> None:
    if not isinstance(job, Mapping):
        raise InterpretationArtifactError(
            "The editable interpretation artifact request is invalid."
        )


def _has_canonical_pointer(job: Mapping[str, Any]) -> bool:
    return job.get(_POINTER_FIELD) == INTERPRETATION_DRAFT_RELATIVE_PATH


def _job_state(job: Mapping[str, Any], settings: Settings) -> dict[str, Any]:
    return {
        "status": _safe_state(job.get("interpretation_status"), "not_started"),
        "stage": _safe_optional_state(job.get("interpretation_stage")),
        "progress": _safe_progress(job.get("interpretation_progress")),
        "message": _safe_job_text(
            job.get("interpretation_message"),
            settings,
            fallback="Editable interpretation status is unavailable.",
        ),
        "error": _safe_job_text(
            job.get("interpretation_error"),
            settings,
            fallback="Editable interpretation failed.",
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
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered:
        return fallback
    try:
        cleaned = friendly_error(value, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return fallback
    cleaned = _SECRET_PATTERN.sub("<redacted>", cleaned)
    cleaned = _URL_PATTERN.sub("<external location>", cleaned)
    cleaned = _ADDRESS_PATTERN.sub("<address>", cleaned)
    cleaned = _WINDOWS_PATH_PATTERN.sub("<local path>", cleaned)
    cleaned = _UNIX_PATH_PATTERN.sub("<local path>", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned.replace("\x00", "")).strip()
    if not cleaned:
        return fallback
    if len(cleaned) > _MAX_JOB_TEXT_LENGTH:
        cleaned = cleaned[-_MAX_JOB_TEXT_LENGTH:]
    return cleaned


def _load_published_artifact(job_id: str, settings: Settings) -> dict[str, Any]:
    try:
        artifact = load_transcription_draft(job_id, settings)
    except (MediaProcessingError, TranscriptionDraftError, OSError, RuntimeError) as exc:
        raise InterpretationArtifactError(
            "Published editable interpretation failed validation."
        ) from exc
    if artifact is None:
        raise InterpretationArtifactUnavailableError(
            "Published editable interpretation is unavailable."
        )
    return artifact


def _secure_job_root(job_id: str, settings: Settings) -> Path:
    try:
        resolved_job = secure_job_dir(settings, job_id)
        exports_root = settings.exports_dir.resolve(strict=True)
        lexical_job = exports_root / job_id
        info = lexical_job.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        resolved = lexical_job.resolve(strict=True)
        if resolved != resolved_job.resolve(strict=True):
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        if exports_root not in resolved.parents:
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        return resolved
    except InterpretationArtifactError:
        raise
    except (MediaProcessingError, OSError, RuntimeError) as exc:
        raise InterpretationArtifactError(
            "Published editable interpretation failed validation."
        ) from exc


def _canonical_file_snapshot(
    job_dir: Path,
) -> tuple[Path, tuple[int, int, int, int]]:
    relative = PurePosixPath(INTERPRETATION_DRAFT_RELATIVE_PATH)
    if relative != PurePosixPath("interpretation", "draft.json"):
        raise InterpretationArtifactError(
            "Published editable interpretation failed validation."
        )
    directory = job_dir / relative.parts[0]
    path = directory / relative.name
    try:
        directory_info = directory.lstat()
        if stat.S_ISLNK(directory_info.st_mode) or not stat.S_ISDIR(
            directory_info.st_mode
        ):
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        root = job_dir.resolve(strict=True)
        resolved_directory = directory.resolve(strict=True)
        if root not in resolved_directory.parents:
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        resolved_path = path.resolve(strict=True)
        if root not in resolved_path.parents:
            raise InterpretationArtifactError(
                "Published editable interpretation failed validation."
            )
        return path, _snapshot(info)
    except FileNotFoundError as exc:
        raise InterpretationArtifactUnavailableError(
            "Published editable interpretation is unavailable."
        ) from exc
    except InterpretationArtifactError:
        raise
    except (OSError, RuntimeError) as exc:
        raise InterpretationArtifactError(
            "Published editable interpretation failed validation."
        ) from exc


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


def _require_safe_json(value: Mapping[str, Any]) -> None:
    try:
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise InterpretationArtifactError(
            "Editable interpretation details could not be serialized safely."
        ) from exc


__all__ = [
    "InterpretationArtifactError",
    "InterpretationArtifactUnavailableError",
    "InterpretationDetails",
    "interpretation_json_path",
    "load_interpretation_details",
]
