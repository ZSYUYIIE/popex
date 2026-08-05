"""Unquantized local pitched-event transcription baseline."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class PitchedTranscriptionError(RuntimeError):
    """Raised when pitched transcription cannot safely produce evidence."""


def transcribe_pitched_audio(
    audio_path: Path,
    *,
    source_kind: str,
    algorithm_version: str = "baseline-pyin-v1",
    fmin_hz: float | None = None,
    fmax_hz: float | None = None,
) -> dict[str, Any]:
    """Return raw pitched-event evidence for one local WAV input."""

    raise PitchedTranscriptionError("Pitched transcription is not implemented yet.")
