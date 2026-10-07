"""Private grouping of recordings into songs, arrangements, and versions.

The hierarchy follows the Product source of truth::

    Composition (song) → Arrangement → Recording version (job) → revisions

Grouping is metadata only. Every recording version keeps its own source,
analysis, stems, events, score, and corrections; nothing here reads one
version's data into another. The comparison places each version's own
estimates side by side and never merges parts.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

VERSION_KINDS: dict[str, str] = {
    "studio": "Studio",
    "live": "Live",
    "acoustic": "Acoustic",
    "concert": "Concert",
    "cover": "Cover",
    "remix": "Remix",
    "radio_edit": "Radio edit",
    "other": "Other",
}
_ID = re.compile(r"[a-f0-9]{32}")
_MAX_TITLE = 160
_MAX_CREDITS = 300
_MAX_NAME = 120
_MAX_LABEL = 80
_MAX_CHORDS_LISTED = 24


class LibraryError(ValueError):
    """Library input is invalid."""


def is_library_id(value: object) -> bool:
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def clean_text(value: Any, label: str, maximum: int, *, required: bool) -> str | None:
    """Return collapsed single-line text, or ``None`` when optional and empty."""
    if value is None:
        if required:
            raise LibraryError(f"{label} is required.")
        return None
    if not isinstance(value, str):
        raise LibraryError(f"{label} must be text.")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise LibraryError(f"{label} must be a single line of text.")
    text = " ".join(value.split())
    if not text:
        if required:
            raise LibraryError(f"{label} is required.")
        return None
    if len(text) > maximum:
        raise LibraryError(f"{label} must be at most {maximum} characters.")
    if "<" in text or ">" in text:
        raise LibraryError(f"{label} must not contain angle brackets.")
    return text


def clean_title(value: Any) -> str:
    return clean_text(value, "Song title", _MAX_TITLE, required=True)  # type: ignore[return-value]


def clean_credits(value: Any) -> str | None:
    return clean_text(value, "Credits", _MAX_CREDITS, required=False)


def clean_arrangement_name(value: Any) -> str:
    return clean_text(value, "Arrangement name", _MAX_NAME, required=True)  # type: ignore[return-value]


def clean_version(label: Any, kind: Any) -> tuple[str | None, str | None]:
    clean_label = clean_text(label, "Version label", _MAX_LABEL, required=False)
    if kind is not None and kind not in VERSION_KINDS:
        raise LibraryError("Version kind is not supported.")
    return clean_label, kind


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _tonal_summary(tonality: Any) -> dict[str, Any] | None:
    if not isinstance(tonality, Mapping) or not isinstance(tonality.get("primaryCandidate"), Mapping):
        return None
    primary = tonality["primaryCandidate"]
    return {
        "displayName": primary.get("displayName"),
        "confidence": primary.get("confidence"),
        "regionsDiffering": sum(
            1 for region in tonality.get("localRegions") or () if region.get("differsFromWhole")
        ),
    }


def version_summary(
    record: Mapping[str, Any],
    document: Mapping[str, Any] | None,
    *,
    stale: bool,
    corrections_active: int,
) -> dict[str, Any]:
    """Summarize one recording version from its own record and saved score."""
    timing = (document or {}).get("timing") or {}
    counts = (document or {}).get("counts") or {}
    chords: list[str] = []
    if document is not None:
        for measure in document.get("measures", ()):
            symbol = measure.get("chordSymbol")
            if isinstance(symbol, str) and symbol not in chords:
                chords.append(symbol)
    return {
        "jobId": record["id"],
        "title": record.get("title") or record.get("original_filename") or "Untitled recording",
        "versionLabel": record.get("version_label"),
        "versionKind": record.get("version_kind"),
        "durationSeconds": _number(record.get("duration_seconds")),
        "tempoBpm": _number(record.get("tempo_bpm")),
        "tempoConfidence": _number(record.get("tempo_confidence")),
        "keySymbol": record.get("key_symbol") if isinstance(record.get("key_symbol"), str) else None,
        "keyConfidence": _number(record.get("key_confidence")),
        "analysisStatus": record.get("analysis_status") or "not_started",
        "score": None
        if document is None
        else {
            "stale": stale,
            "beatsPerMeasure": timing.get("beatsPerMeasure"),
            "meterSource": timing.get("meterSource"),
            "measures": counts.get("measures"),
            "notes": counts.get("notes"),
            "chordSymbols": counts.get("chordSymbols"),
            "percussionHits": counts.get("percussionHits"),
            "fingeredTabNotes": counts.get("fingeredTabNotes"),
            "correctionsActive": corrections_active,
            "tonalContext": _tonal_summary(document.get("tonality")),
            "chordsUsed": chords[:_MAX_CHORDS_LISTED],
            "chordsTruncated": len(chords) > _MAX_CHORDS_LISTED,
        },
    }


def compare_versions(summaries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add each version's chords that appear in no other compared version."""
    used = [set((item["score"] or {}).get("chordsUsed") or ()) for item in summaries]
    for index, item in enumerate(summaries):
        if item["score"] is None:
            continue
        others = set().union(*(chords for position, chords in enumerate(used) if position != index))
        item["score"]["chordsOnlyHere"] = [
            chord for chord in item["score"]["chordsUsed"] if chord not in others
        ]
    return list(summaries)


__all__ = [
    "LibraryError",
    "VERSION_KINDS",
    "clean_arrangement_name",
    "clean_credits",
    "clean_title",
    "clean_version",
    "compare_versions",
    "is_library_id",
    "version_summary",
]
