from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence


PITCHED_PART_INFERENCE_VERSION = "source-phrase-v1"


class PitchedPartInferenceError(RuntimeError):
    """Pitched part inference could not be completed safely."""


@dataclass(frozen=True, slots=True)
class PitchedPartInferenceResult:
    version: str
    parts: tuple[dict[str, Any], ...]
    voices: tuple[dict[str, Any], ...]
    phrases: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    unassigned_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def infer_pitched_parts(
    pitched_events: Sequence[Mapping[str, Any]],
    alignment_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    version: str = PITCHED_PART_INFERENCE_VERSION,
) -> PitchedPartInferenceResult:
    """Infer conservative source-aware parts, voices, and phrases."""
    raise NotImplementedError
