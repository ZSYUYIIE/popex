"""Compose validated raw transcription into an editable interpretation draft."""

from __future__ import annotations

from dataclasses import dataclass


INTERPRETATION_PIPELINE_VERSION = "editable-interpretation-v1"


class InterpretationPipelineError(RuntimeError):
    """A raw transcription could not be converted into an editable draft safely."""


@dataclass(frozen=True, slots=True)
class InterpretationPipelineResult:
    version: str
    draft_file_name: str
    created_at: str
    part_count: int
    phrase_count: int
    pitched_item_count: int
    percussion_item_count: int
    warning_count: int
