"""Local raw-transcription pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TRANSCRIPTION_VERSION = "baseline-pyin-onset-v1"


class TranscriptionPipelineError(RuntimeError):
    """Raised when a job cannot publish a safe raw transcription artifact."""


@dataclass(frozen=True)
class TranscriptionPipelineResult:
    transcription_version: str
    artifact_file_name: str
    transcribed_at: str
    pitched_event_count: int
    percussion_event_count: int
    aligned_event_count: int
    input_mode: str
    warnings: tuple[str, ...]
    payload: dict[str, Any]
