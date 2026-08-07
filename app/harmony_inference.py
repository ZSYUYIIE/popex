"""Conservative harmonic-context inference over raw pitched-note evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


HARMONY_INFERENCE_VERSION = "pitch-class-window-v1"


class HarmonyInferenceError(RuntimeError):
    """Raised when harmonic evidence cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class HarmonyInferenceResult:
    version: str
    segments: tuple[dict[str, Any], ...]
    tonal_context: dict[str, Any] | None
    unresolved_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]
