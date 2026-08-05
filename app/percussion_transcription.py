from __future__ import annotations

from pathlib import Path
from typing import Any


DEFAULT_ALGORITHM_VERSION = "baseline-onset-bands-v1"


class PercussionTranscriptionError(RuntimeError):
    """A percussion transcription request could not be completed safely."""


def transcribe_percussion_audio(
    audio_path: Path,
    *,
    source_kind: str,
    algorithm_version: str = DEFAULT_ALGORITHM_VERSION,
) -> dict[str, Any]:
    """Return raw percussion-event candidates for one trusted local WAV file."""
    raise PercussionTranscriptionError(
        "Percussion transcription is not implemented at this checkpoint."
    )
