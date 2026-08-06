from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.config import Settings


class TranscriptionArtifactError(RuntimeError):
    """A transcription artifact could not be exposed safely."""


class TranscriptionArtifactUnavailableError(TranscriptionArtifactError):
    """A canonical transcription artifact is not currently available."""


@dataclass(frozen=True)
class TranscriptionDetails:
    available: bool
    status: str
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

    def payload(self, *, include_events: bool = True) -> dict[str, Any]:
        raise NotImplementedError


def load_transcription_details(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> TranscriptionDetails:
    raise NotImplementedError


def transcription_json_path(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> Path:
    raise NotImplementedError
