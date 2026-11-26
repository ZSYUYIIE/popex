"""Private grouping of recordings into songs, arrangements, and versions.

The hierarchy follows the Product source of truth::

    Composition (song) → Arrangement → Recording version (job) → revisions

Grouping is metadata only. Every recording version keeps its own source,
analysis, stems, events, score, and corrections; nothing here reads one
version's data into another. The comparison places each version's own
estimates side by side and never merges parts.
"""

from __future__ import annotations

import difflib
import re
import unicodedata
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
            symbols = [measure.get("chordSymbol")]
            if symbols[0] is None:
                # Half-bar chords (schema 9) count when the bar has no whole-bar chord.
                symbols = [change.get("symbol") for change in measure.get("chordChanges") or () if isinstance(change, dict)]
            for symbol in symbols:
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
            "sections": _section_summary(document),
        },
    }


def _section_summary(document: Mapping[str, Any]) -> dict[str, Any] | None:
    """This version's own song sections, compact, for side-by-side form."""
    from app.song_sections import public_sections

    try:
        value = public_sections(document)
    except (KeyError, IndexError, TypeError):
        return None
    if value is None:
        return None
    return {
        "form": " ".join(item["label"] for item in value["sections"]) or None,
        "note": value["note"],
        "sections": [
            {key: item[key] for key in ("label", "startMeasure", "endMeasure", "bars", "startSeconds", "heardText")}
            for item in value["sections"]
        ],
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


# Filing suggestions: read only the recording's own name and existing song titles.

_EXTENSION = re.compile(r"\.(?:mp3|m4a|aac|wav|flac|ogg|opus|webm|mp4|mkv|mov|wma|aiff?)$", re.IGNORECASE)
_BRACKETS = re.compile(r"[\(\[\{]([^\)\]\}]*)[\)\]\}]")
_NOISE = re.compile(
    r"^(?:official(?: music| lyric)?(?: video| audio| visuali[sz]er)?|music video|lyric video|lyrics?|audio|video|"
    r"visuali[sz]er|hd|hq|4k|1080p|720p|explicit|clean|full song|with lyrics|mp3\.\w+|[\w.-]+\.(?:com|net|pm|org|io))$",
    re.IGNORECASE,
)
_SEPARATORS = (" - ", " – ", " — ", " | ", ": ")
_FEATURING = re.compile(r"\s+(?:feat\.?|ft\.?|featuring)\s+.*$", re.IGNORECASE)
# (pattern, kind) in priority order; the first match wins.
_KIND_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bradio edit\b", re.IGNORECASE), "radio_edit"),
    (re.compile(r"\b(?:remix|rmx|re-?mix|bootleg|edit\s+mix|club mix|extended mix)\b", re.IGNORECASE), "remix"),
    (re.compile(r"\b(?:acoustic|unplugged|stripped)\b", re.IGNORECASE), "acoustic"),
    (re.compile(r"\b(?:in concert|concert|tour)\b", re.IGNORECASE), "concert"),
    (re.compile(r"\blive\b", re.IGNORECASE), "live"),
    (re.compile(r"\b(?:cover|covered by|tribute)\b", re.IGNORECASE), "cover"),
)
_MATCH_RATIO = 0.84
_VERSION_TAIL = re.compile(r"(?:live|acoustic|unplugged|stripped|remix|cover|radio edit|in concert)\b", re.IGNORECASE)


def _plain(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).replace("_", " ").split())


def song_key(title: str) -> str:
    """A loose key for matching song titles: lower case letters and digits only."""
    text = unicodedata.normalize("NFKD", title.casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = re.sub(r"^(?:the|a|an)\s+", "", text)
    return re.sub(r"[^0-9a-z\u0080-￿]+", "", text)


def suggest_filing(name: str | None, songs: Sequence[Mapping[str, Any]] = ()) -> dict[str, Any]:
    """Guess song title, credits, version kind and label, and a matching song.

    Returns ``title``, ``credits``, ``kind``, ``label`` (each may be ``None``)
    and ``matchingSongId`` when an existing song's title matches closely.
    """
    text = _plain(_EXTENSION.sub("", (name or "").strip()))
    if text and " " not in text and re.search(r"[a-z]", text) and not re.search(r"[A-Z]", text):
        # A file-style name such as "shake-it-off": hyphens were spaces.
        text = " ".join(word.capitalize() for word in re.split(r"[-.]+", text) if word)
    kind = None
    label = None
    kept_brackets = []
    for match in _BRACKETS.finditer(text):
        inner = " ".join(match.group(1).split())
        if not inner or _NOISE.match(inner):
            continue
        found = _kind_of(inner)
        if found and kind is None:
            kind, label = found, inner[:80]
        elif not found:
            kept_brackets.append(inner)
    text = " ".join(_BRACKETS.sub(" ", text).split())

    credits = None
    for separator in _SEPARATORS:
        if separator in text:
            left, right = (part.strip() for part in text.split(separator, 1))
            if left and right:
                found = _kind_of(right)
                if found and kind is None and _VERSION_TAIL.match(right):
                    # "Song - Live at X": the right side names the version.
                    kind, label, text = found, right[:80], left
                else:
                    credits, text = left, right
            break
    text = _FEATURING.sub("", text).strip(" -–—|:")
    if kind is None:
        found = _kind_of(text)
        if found:
            kind = found
            # "Song - Live at X" style tails are a label, not part of the title.
            tail = re.search(r"\b(live|acoustic|unplugged|remix|cover|radio edit)\b.*$", text, re.IGNORECASE)
            if tail and tail.start() > 0:
                label = tail.group(0)[:80]
                text = text[: tail.start()].strip(" -–—|:,")
    title = text[:160] or None

    matching = None
    if title:
        key = song_key(title)
        best = 0.0
        for song in songs:
            other = song_key(str(song.get("title") or ""))
            if not other or not key:
                continue
            ratio = 1.0 if other == key else difflib.SequenceMatcher(None, key, other).ratio()
            if ratio >= _MATCH_RATIO and ratio > best:
                best, matching = ratio, song.get("id")
    return {
        "title": title,
        "credits": credits[:300] if credits else None,
        "kind": kind,
        "label": label,
        "matchingSongId": matching,
    }


def _kind_of(text: str) -> str | None:
    for pattern, kind in _KIND_PATTERNS:
        if pattern.search(text):
            return kind
    return None


__all__ = [
    "LibraryError",
    "VERSION_KINDS",
    "clean_arrangement_name",
    "clean_credits",
    "clean_title",
    "clean_version",
    "compare_versions",
    "is_library_id",
    "song_key",
    "suggest_filing",
    "version_summary",
]
