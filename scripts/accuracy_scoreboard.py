"""Score a saved PopEx draft score against a hand-written reference.

The scoreboard answers one question with the same numbers every time: how close
is a saved draft to what is known about the recording? It reads artifacts only.
It never changes predictions, corrections, or saved scores, needs no audio, and
uses the standard library alone.

Reference-based accuracy (timing, key, chords, notes) and reference-free
indicators are reported in separate groups and are never mixed.

See docs/accuracy-scoreboard.md for the reference format and for what each
number does and does not mean.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Iterable, Sequence

SCOREBOARD_VERSION = "accuracy-scoreboard-v1"
REFERENCE_ARTIFACT = "popex-accuracy-reference"
RESULT_ARTIFACT = "popex-accuracy-result"
RESULTS_ARTIFACT = "popex-accuracy-results"
SCHEMA_VERSION = 1

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_REFERENCE_BARS = 4000
MAX_REFERENCE_ITEMS = 200_000

# A note counts as found when its start is within this many seconds of a
# reference note with the same pitch. 50 ms is the usual tolerance in the
# transcription literature.
NOTE_ONSET_TOLERANCE_SECONDS = 0.05
# A bar counts as aligned when it starts within this many beats of a reference
# downbeat.
BAR_ALIGNMENT_TOLERANCE_BEATS = 0.25
LOW_CONFIDENCE_THRESHOLD = 0.5

_PITCH_CLASSES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_PITCH_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_ACCIDENTALS = {"#": 1, "♯": 1, "b": -1, "♭": -1}

# Interval sets match app/tonal_context.py so a reference can name any
# collection the score's tonal context can report.
COLLECTION_INTERVALS: dict[str, tuple[int, ...]] = {
    "ionian": (0, 2, 4, 5, 7, 9, 11),
    "dorian": (0, 2, 3, 5, 7, 9, 10),
    "phrygian": (0, 1, 3, 5, 7, 8, 10),
    "lydian": (0, 2, 4, 6, 7, 9, 11),
    "mixolydian": (0, 2, 4, 5, 7, 9, 10),
    "aeolian": (0, 2, 3, 5, 7, 8, 10),
    "locrian": (0, 1, 3, 5, 6, 8, 10),
    "harmonic_minor": (0, 2, 3, 5, 7, 8, 11),
    "melodic_minor": (0, 2, 3, 5, 7, 9, 11),
    "major_pentatonic": (0, 2, 4, 7, 9),
    "minor_pentatonic": (0, 3, 5, 7, 10),
    "blues": (0, 3, 5, 6, 7, 10),
}
_COLLECTION_ALIASES = {"major": "ionian", "minor": "aeolian", "natural_minor": "aeolian"}

_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._()+-]{0,79}$")
_CHORD_PATTERN = re.compile(r"^([A-G])([#b♯♭]?)(.*)$")
_NO_CHORD = {"N", "N.C.", "NC", "-"}


class ScoreboardError(ValueError):
    """An input could not be used. The message is safe to show as-is."""


# ---------------------------------------------------------------------------
# Small helpers


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _require_number(value: Any, name: str, *, minimum: float | None = None, maximum: float | None = None) -> float:
    if not _is_number(value):
        raise ScoreboardError(f"{name} must be a finite number.")
    number = float(value)
    if minimum is not None and number < minimum:
        raise ScoreboardError(f"{name} must be at least {minimum:g}.")
    if maximum is not None and number > maximum:
        raise ScoreboardError(f"{name} must be at most {maximum:g}.")
    return number


def _percent(part: float, whole: float) -> float | None:
    if whole <= 0:
        return None
    return round(100.0 * part / whole, 1)


def pitch_class(name: Any) -> int:
    """Pitch class 0-11 for a note name such as ``G``, ``Bb`` or ``F#``."""
    if not isinstance(name, str):
        raise ScoreboardError("A note name must be text such as G, Bb or F#.")
    text = name.strip()
    if not text or text[0].upper() not in _PITCH_CLASSES or len(text) > 2:
        raise ScoreboardError(f"'{name}' is not a note name such as G, Bb or F#.")
    value = _PITCH_CLASSES[text[0].upper()]
    if len(text) == 2:
        if text[1] not in _ACCIDENTALS:
            raise ScoreboardError(f"'{name}' is not a note name such as G, Bb or F#.")
        value += _ACCIDENTALS[text[1]]
    return value % 12


def normalize_collection(name: Any) -> str | None:
    """Canonical collection id, or None when the name is not one we know."""
    if not isinstance(name, str):
        return None
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    key = _COLLECTION_ALIASES.get(key, key)
    return key if key in COLLECTION_INTERVALS else None


def scale_pitch_classes(root: int, collection: str) -> frozenset[int]:
    return frozenset((root + step) % 12 for step in COLLECTION_INTERVALS[collection])


def parse_chord(symbol: Any) -> tuple[int, str] | None:
    """Reduce a chord symbol to ``(root pitch class, quality)``.

    Quality is ``maj``, ``min`` or ``other`` (suspended, diminished, augmented,
    power chords). Sevenths, extensions and slash basses are dropped, so ``Am7``
    and ``Am/G`` both become A minor. ``None`` and no-chord markers return None.
    """
    if symbol is None:
        return None
    if not isinstance(symbol, str):
        raise ScoreboardError("A chord symbol must be text such as Am, C or G7.")
    text = symbol.strip()
    if not text or text.upper() in _NO_CHORD:
        return None
    match = _CHORD_PATTERN.match(text)
    if not match:
        raise ScoreboardError(f"'{symbol}' is not a chord symbol such as Am, C or G7.")
    root = (_PITCH_CLASSES[match.group(1)] + _ACCIDENTALS.get(match.group(2), 0)) % 12
    raw_rest = match.group(3).split("/")[0].replace(":", "").strip()
    if raw_rest.startswith("M"):
        # A capital M means major (CM7), unlike the lowercase m of a minor chord.
        return root, "maj"
    rest = raw_rest.lower()
    if rest.startswith(("sus", "dim", "aug", "5", "o", "+")) or "sus" in rest or "dim" in rest or "aug" in rest:
        return root, "other"
    if rest.startswith("min") or rest.startswith("-") or (rest.startswith("m") and not rest.startswith("maj")):
        return root, "min"
    return root, "maj"


# ---------------------------------------------------------------------------
# Interval arithmetic: every chord figure is a duration, never a sample count.


def _union(intervals: Iterable[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for start, end in sorted(i for i in intervals if i[1] > i[0]):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _overlap_seconds(span: tuple[float, float], merged: Sequence[tuple[float, float]]) -> float:
    total = 0.0
    for start, end in merged:
        if end <= span[0]:
            continue
        if start >= span[1]:
            break
        total += min(end, span[1]) - max(start, span[0])
    return total


# ---------------------------------------------------------------------------
# Loading


def read_json_file(path: str | os.PathLike[str], what: str) -> dict[str, Any]:
    """Read one strict JSON object, refusing oversized or non-object files."""
    file_path = Path(path)
    try:
        size = file_path.stat().st_size
    except OSError:
        raise ScoreboardError(f"The {what} file could not be found.") from None
    if not file_path.is_file():
        raise ScoreboardError(f"The {what} path is not a file.")
    if size > MAX_INPUT_BYTES:
        raise ScoreboardError(f"The {what} file is larger than {MAX_INPUT_BYTES // (1024 * 1024)} MiB.")

    def reject_constant(token: str) -> Any:
        raise ScoreboardError(f"The {what} file contains {token}, which is not valid JSON.")

    try:
        with file_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle, parse_constant=reject_constant)
    except ScoreboardError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError):
        raise ScoreboardError(f"The {what} file is not readable JSON.") from None
    if not isinstance(payload, dict):
        raise ScoreboardError(f"The {what} file must contain a JSON object.")
    return payload


def load_reference(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate a reference and return it in a normalized, time-based form."""
    if not isinstance(payload, dict):
        raise ScoreboardError("The reference must be a JSON object.")
    if payload.get("artifactType") != REFERENCE_ARTIFACT:
        raise ScoreboardError(f"The reference must have artifactType '{REFERENCE_ARTIFACT}'.")
    if payload.get("schemaVersion") != SCHEMA_VERSION:
        raise ScoreboardError(f"Only reference schemaVersion {SCHEMA_VERSION} is supported.")
    reference_id = payload.get("id")
    if not isinstance(reference_id, str) or not _ID_PATTERN.match(reference_id):
        raise ScoreboardError("The reference id must be lowercase letters, digits and hyphens (at most 64).")
    title = payload.get("title")
    if title is not None and (not isinstance(title, str) or len(title) > 200):
        raise ScoreboardError("The reference title must be text of at most 200 characters.")

    timing_in = payload.get("timing")
    if not isinstance(timing_in, dict):
        raise ScoreboardError("The reference needs a timing object with tempoBpm and beatsPerMeasure.")
    tempo = _require_number(timing_in.get("tempoBpm"), "timing.tempoBpm", minimum=20, maximum=400)
    beats = timing_in.get("beatsPerMeasure")
    if not isinstance(beats, int) or isinstance(beats, bool) or not 1 <= beats <= 16:
        raise ScoreboardError("timing.beatsPerMeasure must be a whole number from 1 to 16.")
    first_downbeat = timing_in.get("firstDownbeatSeconds")
    if first_downbeat is not None:
        first_downbeat = _require_number(first_downbeat, "timing.firstDownbeatSeconds", minimum=0, maximum=36000)
    timing = {"tempoBpm": tempo, "beatsPerMeasure": beats, "firstDownbeatSeconds": first_downbeat}

    key = None
    key_in = payload.get("key")
    if key_in is not None:
        if not isinstance(key_in, dict):
            raise ScoreboardError("key must be an object with tonalCenter and collections.")
        root = pitch_class(key_in.get("tonalCenter"))
        names = key_in.get("collections")
        if not isinstance(names, list) or not names or len(names) > len(COLLECTION_INTERVALS):
            raise ScoreboardError("key.collections must list at least one accepted collection.")
        collections: list[str] = []
        for name in names:
            canonical = normalize_collection(name)
            if canonical is None:
                known = ", ".join(sorted(COLLECTION_INTERVALS))
                raise ScoreboardError(f"'{name}' is not a known collection. Known collections: {known}.")
            if canonical not in collections:
                collections.append(canonical)
        key = {"tonalCenter": _PITCH_NAMES[root], "rootPitchClass": root, "collections": collections}

    chords = _load_reference_chords(payload.get("chords"), timing)
    notes = _load_reference_notes(payload.get("notes"))
    return {
        "id": reference_id,
        "title": title,
        "timing": timing,
        "key": key,
        "chords": chords,
        "notes": notes,
    }


def _load_reference_chords(chords_in: Any, timing: dict[str, Any]) -> list[dict[str, Any]] | None:
    if chords_in is None:
        return None
    if not isinstance(chords_in, dict) or ("bars" in chords_in) == ("segments" in chords_in):
        raise ScoreboardError("chords must contain exactly one of 'bars' or 'segments'.")
    segments: list[dict[str, Any]] = []
    if "segments" in chords_in:
        items = chords_in["segments"]
        if not isinstance(items, list) or not items or len(items) > MAX_REFERENCE_ITEMS:
            raise ScoreboardError("chords.segments must be a non-empty list.")
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ScoreboardError(f"chords.segments[{index}] must be an object.")
            start = _require_number(item.get("startSeconds"), f"chords.segments[{index}].startSeconds", minimum=0)
            end = _require_number(item.get("endSeconds"), f"chords.segments[{index}].endSeconds", minimum=0)
            if end <= start:
                raise ScoreboardError(f"chords.segments[{index}] must end after it starts.")
            segments.append({"start": start, "end": end, "symbol": item.get("symbol"), "chord": parse_chord(item.get("symbol"))})
    else:
        first = timing["firstDownbeatSeconds"]
        if first is None:
            raise ScoreboardError("chords.bars needs timing.firstDownbeatSeconds to place bars in time.")
        bar_seconds = timing["beatsPerMeasure"] * 60.0 / timing["tempoBpm"]
        ranges = chords_in["bars"]
        if not isinstance(ranges, list) or not ranges or len(ranges) > MAX_REFERENCE_BARS:
            raise ScoreboardError("chords.bars must be a non-empty list of bar ranges.")
        for index, item in enumerate(ranges):
            if not isinstance(item, dict):
                raise ScoreboardError(f"chords.bars[{index}] must be an object.")
            start_bar, end_bar = item.get("startBar"), item.get("endBar")
            for name, value in (("startBar", start_bar), ("endBar", end_bar)):
                if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= MAX_REFERENCE_BARS:
                    raise ScoreboardError(f"chords.bars[{index}].{name} must be a bar number from 1 to {MAX_REFERENCE_BARS}.")
            if end_bar < start_bar:
                raise ScoreboardError(f"chords.bars[{index}] must not end before it starts.")
            pattern = item.get("pattern")
            if not isinstance(pattern, list) or not pattern or len(pattern) > 64:
                raise ScoreboardError(f"chords.bars[{index}].pattern must list 1 to 64 bars of chords.")
            for bar in range(start_bar, end_bar + 1):
                entry = pattern[(bar - start_bar) % len(pattern)]
                parts = entry if isinstance(entry, list) else [entry]
                if not parts or len(parts) > 8:
                    raise ScoreboardError(f"chords.bars[{index}] has a bar split into more than 8 chords.")
                bar_start = first + (bar - 1) * bar_seconds
                for slot, symbol in enumerate(parts):
                    segments.append(
                        {
                            "start": bar_start + slot * bar_seconds / len(parts),
                            "end": bar_start + (slot + 1) * bar_seconds / len(parts),
                            "symbol": symbol,
                            "chord": parse_chord(symbol),
                        }
                    )
    segments.sort(key=lambda segment: segment["start"])
    for previous, current in zip(segments, segments[1:]):
        if current["start"] < previous["end"] - 1e-9:
            raise ScoreboardError("Reference chords overlap in time; each moment may have only one chord.")
    return segments


def _load_reference_notes(notes_in: Any) -> dict[str, list[tuple[float, int]]] | None:
    if notes_in is None:
        return None
    if not isinstance(notes_in, dict) or not notes_in or len(notes_in) > 16:
        raise ScoreboardError("notes must map a source line (for example vocals or bass) to a list of notes.")
    loaded: dict[str, list[tuple[float, int]]] = {}
    for source, items in notes_in.items():
        if not isinstance(source, str) or not re.match(r"^[a-z][a-z0-9_]{0,31}$", source):
            raise ScoreboardError("A source line name must be lowercase letters, digits and underscores.")
        if not isinstance(items, list) or not items or len(items) > MAX_REFERENCE_ITEMS:
            raise ScoreboardError(f"notes.{source} must be a non-empty list of notes.")
        notes: list[tuple[float, int]] = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                raise ScoreboardError(f"notes.{source}[{index}] must be an object.")
            start = _require_number(item.get("startSeconds"), f"notes.{source}[{index}].startSeconds", minimum=0)
            midi = item.get("midiNote")
            if not isinstance(midi, int) or isinstance(midi, bool) or not 0 <= midi <= 127:
                raise ScoreboardError(f"notes.{source}[{index}].midiNote must be a MIDI note number from 0 to 127.")
            notes.append((start, midi))
        loaded[source] = sorted(notes)
    return loaded


def load_score(payload: dict[str, Any]) -> dict[str, Any]:
    """Pull out the fields the scoreboard reads from a saved score document.

    Accepts the saved score artifact, the JSON download, and the
    ``score/saved?includeMeasures=true`` payload: anything with ``measures`` and
    ``timing``.
    """
    if not isinstance(payload, dict):
        raise ScoreboardError("The score must be a JSON object.")
    measures_in = payload.get("measures")
    if not isinstance(measures_in, list) or not measures_in:
        raise ScoreboardError("The score has no measures. Use a saved score that includes its measures.")
    timing_in = payload.get("timing")
    if not isinstance(timing_in, dict):
        raise ScoreboardError("The score has no timing information.")
    tempo = _require_number(timing_in.get("tempoBpm"), "The score tempo", minimum=1, maximum=1000)
    beats = timing_in.get("beatsPerMeasure")
    if not isinstance(beats, int) or isinstance(beats, bool) or not 1 <= beats <= 32:
        raise ScoreboardError("The score beatsPerMeasure must be a whole number.")

    measures: list[dict[str, Any]] = []
    previous_start = -math.inf
    for index, item in enumerate(measures_in):
        if not isinstance(item, dict):
            raise ScoreboardError(f"Score measure {index + 1} is not an object.")
        start = _require_number(item.get("startSeconds"), f"Score measure {index + 1} start")
        end = _require_number(item.get("endSeconds"), f"Score measure {index + 1} end")
        if end <= start or start < previous_start:
            raise ScoreboardError(f"Score measure {index + 1} has an invalid time span.")
        previous_start = start
        measures.append(
            {
                "index": index,
                "start": start,
                "end": end,
                "chordSymbol": item.get("chordSymbol"),
                "chordChanges": item.get("chordChanges") if isinstance(item.get("chordChanges"), list) else [],
                "harmony": item.get("harmony") if isinstance(item.get("harmony"), list) else [],
                "notes": item.get("notes") if isinstance(item.get("notes"), list) else [],
                "hits": item.get("percussionHits") if isinstance(item.get("percussionHits"), list) else [],
            }
        )
    tonality = payload.get("tonality") if isinstance(payload.get("tonality"), dict) else None
    return {
        "identity": {
            "jobId": payload.get("jobId") if isinstance(payload.get("jobId"), str) else None,
            "builderVersion": payload.get("builderVersion") if isinstance(payload.get("builderVersion"), str) else None,
            "schemaVersion": payload.get("schemaVersion") if isinstance(payload.get("schemaVersion"), int) else None,
            "createdAt": payload.get("createdAt") if isinstance(payload.get("createdAt"), str) else None,
            "view": payload.get("view") if isinstance(payload.get("view"), str) else None,
        },
        "tempoBpm": tempo,
        "beatsPerMeasure": beats,
        "measures": measures,
        "tonality": tonality,
    }


# ---------------------------------------------------------------------------
# Timing


def evaluate_timing(score: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    timing = reference["timing"]
    reference_beat = 60.0 / timing["tempoBpm"]
    reference_bar = reference_beat * timing["beatsPerMeasure"]
    measures = score["measures"]
    origin = measures[0]["start"]
    beats_per_measure = score["beatsPerMeasure"]

    # Drift compares the beats the draft notated with the beats that really
    # passed at the reference tempo. It does not depend on where bar 1 starts.
    drifts = [(measure["start"] - origin) / reference_beat - measure["index"] * beats_per_measure for measure in measures]
    final_drift = (measures[-1]["end"] - origin) / reference_beat - len(measures) * beats_per_measure
    drifts.append(final_drift)

    ratio = score["tempoBpm"] / timing["tempoBpm"]
    if abs(ratio - 2.0) <= 0.12:
        octave = "double"
    elif abs(ratio - 0.5) <= 0.03:
        octave = "half"
    else:
        octave = None

    result: dict[str, Any] = {
        "tempoBpm": round(score["tempoBpm"], 3),
        "referenceTempoBpm": round(timing["tempoBpm"], 3),
        "tempoErrorPercent": round(100.0 * (ratio - 1.0), 2),
        "tempoOctaveError": octave,
        "beatsPerMeasure": beats_per_measure,
        "referenceBeatsPerMeasure": timing["beatsPerMeasure"],
        "meterMatches": beats_per_measure == timing["beatsPerMeasure"],
        "bars": len(measures),
        "expectedBars": max(1, round((measures[-1]["end"] - origin) / reference_bar)),
        "finalGridDriftBeats": round(final_drift, 2),
        "maxGridDriftBeats": round(max(abs(value) for value in drifts), 2),
        "alignedBarsPercent": None,
        "medianBarOffsetBeats": None,
    }
    first = timing["firstDownbeatSeconds"]
    if first is not None:
        offsets = []
        for measure in measures:
            position = (measure["start"] - first) / reference_bar
            offsets.append(abs(position - round(position)) * timing["beatsPerMeasure"])
        aligned = sum(1 for offset in offsets if offset <= BAR_ALIGNMENT_TOLERANCE_BEATS)
        ordered = sorted(offsets)
        middle = len(ordered) // 2
        median = ordered[middle] if len(ordered) % 2 else (ordered[middle - 1] + ordered[middle]) / 2
        result["alignedBarsPercent"] = _percent(aligned, len(offsets))
        result["medianBarOffsetBeats"] = round(median, 2)
    return result


# ---------------------------------------------------------------------------
# Key


def _judge_key(root: int | None, collection: str | None, reference_key: dict[str, Any]) -> dict[str, Any]:
    reference_root = reference_key["rootPitchClass"]
    centre_matches = root is not None and root == reference_root
    collection_matches = bool(centre_matches and collection in reference_key["collections"])
    same_notes = None
    if root is not None and collection is not None:
        predicted = scale_pitch_classes(root, collection)
        same_notes = any(predicted == scale_pitch_classes(reference_root, name) for name in reference_key["collections"])
    return {"tonalCenterMatches": centre_matches, "collectionMatches": collection_matches, "sameNotes": same_notes}


def _candidate_key(candidate: Any) -> tuple[int | None, str | None, str | None, float | None]:
    if not isinstance(candidate, dict):
        return None, None, None, None
    root: int | None = None
    if isinstance(candidate.get("rootPitchClass"), int) and not isinstance(candidate.get("rootPitchClass"), bool):
        root = candidate["rootPitchClass"] % 12
    elif isinstance(candidate.get("tonalCenter"), str):
        try:
            root = pitch_class(candidate["tonalCenter"])
        except ScoreboardError:
            root = None
    collection = normalize_collection(candidate.get("collection"))
    display = candidate.get("displayName") if isinstance(candidate.get("displayName"), str) else None
    confidence = float(candidate["confidence"]) if _is_number(candidate.get("confidence")) else None
    return root, collection, display, confidence


def evaluate_key(score: dict[str, Any], reference: dict[str, Any], analysis: dict[str, Any] | None) -> dict[str, Any] | None:
    reference_key = reference["key"]
    if reference_key is None:
        return None
    result: dict[str, Any] = {
        "reference": {"tonalCenter": reference_key["tonalCenter"], "collections": list(reference_key["collections"])},
        "score": None,
        "analysis": None,
    }
    tonality = score["tonality"]
    if tonality is not None:
        root, collection, display, confidence = _candidate_key(tonality.get("primaryCandidate"))
        result["score"] = {
            "displayName": display,
            "confidence": None if confidence is None else round(confidence, 3),
            "keySignatureWritten": tonality.get("keySignature") is not None,
            **_judge_key(root, collection, reference_key),
        }
    if analysis is not None:
        analysis_tonality = analysis.get("tonality") if isinstance(analysis.get("tonality"), dict) else None
        if analysis_tonality is not None:
            root, collection, display, confidence = _candidate_key(analysis_tonality.get("primaryCandidate"))
            result["analysis"] = {
                "displayName": display,
                "confidence": None if confidence is None else round(confidence, 3),
                **_judge_key(root, collection, reference_key),
            }
    return result


# ---------------------------------------------------------------------------
# Chords


def _chord_figures(estimated: list[tuple[float, float, tuple[int, str]]], reference_segments: list[dict[str, Any]]) -> dict[str, Any]:
    # Overlapping harmony windows may disagree. A moment counts as right when at
    # least one window covering it is right, so candidate figures are an upper
    # bound on what a perfect choice between the windows could show.
    any_union = _union((start, end) for start, end, _ in estimated)
    by_root: dict[int, list[tuple[float, float]]] = {}
    by_chord: dict[tuple[int, str], list[tuple[float, float]]] = {}
    for start, end, chord in estimated:
        by_root.setdefault(chord[0], []).append((start, end))
        by_chord.setdefault(chord, []).append((start, end))
    root_unions = {root: _union(spans) for root, spans in by_root.items()}
    chord_unions = {chord: _union(spans) for chord, spans in by_chord.items()}
    total = covered = root_ok = quality_ok = 0.0
    for segment in reference_segments:
        chord = segment["chord"]
        if chord is None:
            continue
        span = (segment["start"], segment["end"])
        total += span[1] - span[0]
        covered += _overlap_seconds(span, any_union)
        root_ok += _overlap_seconds(span, root_unions.get(chord[0], ()))
        quality_ok += _overlap_seconds(span, chord_unions.get(chord, ()))
    return {
        "referenceSeconds": round(total, 2),
        "coveragePercent": _percent(covered, total),
        "rootAccuracyPercent": _percent(root_ok, total),
        "majMinAccuracyPercent": _percent(quality_ok, total),
        "rootAccuracyWhereShownPercent": _percent(root_ok, covered),
    }


def evaluate_chords(score: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any] | None:
    reference_segments = reference["chords"]
    if reference_segments is None:
        return None
    shown: list[tuple[float, float, tuple[int, str]]] = []
    candidates: list[tuple[float, float, tuple[int, str]]] = []
    unreadable = 0
    bars_with_symbol = 0
    for measure in score["measures"]:
        beat_seconds = (measure["end"] - measure["start"]) / score["beatsPerMeasure"]
        symbol = measure["chordSymbol"]
        # A whole-bar symbol spans the bar; otherwise half-bar changes (schema 9)
        # each span from their beat to the next change or the bar's end.
        if isinstance(symbol, str) and symbol.strip():
            spans = [(measure["start"], measure["end"], symbol)]
        else:
            changes = [
                change for change in measure.get("chordChanges") or ()
                if isinstance(change, dict) and isinstance(change.get("symbol"), str) and _is_number(change.get("beat"))
            ]
            starts = [measure["start"] + float(change["beat"]) * beat_seconds for change in changes]
            spans = [
                (start, starts[index + 1] if index + 1 < len(starts) else measure["end"], change["symbol"])
                for index, (start, change) in enumerate(zip(starts, changes))
            ]
        if spans:
            bars_with_symbol += 1
        for start, end, text in spans:
            try:
                chord = parse_chord(text)
            except ScoreboardError:
                chord = None
                unreadable += 1
            if chord is not None and end > start:
                shown.append((start, end, chord))
        for window in measure["harmony"]:
            if not isinstance(window, dict) or window.get("unresolved") or not isinstance(window.get("symbol"), str):
                continue
            if not _is_number(window.get("startBeat")) or not _is_number(window.get("endBeat")):
                continue
            try:
                chord = parse_chord(window["symbol"])
            except ScoreboardError:
                chord = None
                unreadable += 1
            if chord is None:
                continue
            start = measure["start"] + float(window["startBeat"]) * beat_seconds
            end = measure["start"] + float(window["endBeat"]) * beat_seconds
            if end > start:
                candidates.append((start, end, chord))
    return {
        "barsWithSymbol": bars_with_symbol,
        "bars": len(score["measures"]),
        "unreadableSymbols": unreadable,
        "shown": _chord_figures(shown, reference_segments),
        "candidates": _chord_figures(candidates, reference_segments),
    }


# ---------------------------------------------------------------------------
# Notes


def _match_count(estimated: Sequence[float], reference: Sequence[float], tolerance: float) -> int:
    """Largest one-to-one matching of sorted times within ``tolerance``."""
    matched = 0
    i = j = 0
    while i < len(estimated) and j < len(reference):
        difference = estimated[i] - reference[j]
        if abs(difference) <= tolerance:
            matched += 1
            i += 1
            j += 1
        elif difference < 0:
            i += 1
        else:
            j += 1
    return matched


def note_f1(estimated: Sequence[tuple[float, int]], reference: Sequence[tuple[float, int]], *, octave_tolerant: bool = False) -> dict[str, Any]:
    """Precision, recall and F1 for notes matched on pitch and start time."""

    def group(notes: Sequence[tuple[float, int]]) -> dict[int, list[float]]:
        groups: dict[int, list[float]] = {}
        for start, midi in notes:
            groups.setdefault(midi % 12 if octave_tolerant else midi, []).append(start)
        for times in groups.values():
            times.sort()
        return groups

    estimated_groups, reference_groups = group(estimated), group(reference)
    matched = sum(
        _match_count(estimated_groups.get(pitch, ()), times, NOTE_ONSET_TOLERANCE_SECONDS)
        for pitch, times in reference_groups.items()
    )
    precision = matched / len(estimated) if estimated else 0.0
    recall = matched / len(reference) if reference else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "estimatedNotes": len(estimated),
        "referenceNotes": len(reference),
        "matched": matched,
        "precisionPercent": round(100 * precision, 1),
        "recallPercent": round(100 * recall, 1),
        "f1Percent": round(100 * f1, 1),
    }


def _score_notes(score: dict[str, Any]) -> list[dict[str, Any]]:
    notes: list[dict[str, Any]] = []
    beats = score["beatsPerMeasure"]
    for measure in score["measures"]:
        beat_seconds = (measure["end"] - measure["start"]) / beats
        for item in measure["notes"]:
            if not isinstance(item, dict):
                continue
            midi = item.get("midiNote")
            if not isinstance(midi, int) or isinstance(midi, bool) or not _is_number(item.get("startSeconds")):
                continue
            notated = None
            if _is_number(item.get("quantizedBeat")):
                notated = measure["start"] + (float(item["quantizedBeat"]) - measure["index"] * beats) * beat_seconds
            notes.append(
                {
                    "source": item.get("sourceKind") if isinstance(item.get("sourceKind"), str) else "unknown",
                    "midi": midi,
                    "detected": float(item["startSeconds"]),
                    "notated": notated,
                    "confidence": float(item["confidence"]) if _is_number(item.get("confidence")) else None,
                    "measure": measure["index"],
                }
            )
    return notes


def evaluate_notes(notes: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any] | None:
    reference_notes = reference["notes"]
    if reference_notes is None:
        return None
    result: dict[str, Any] = {}
    for source, expected in reference_notes.items():
        mine = [note for note in notes if source == "all" or note["source"] == source]
        detected = [(note["detected"], note["midi"]) for note in mine]
        notated = [(note["notated"], note["midi"]) for note in mine if note["notated"] is not None]
        result[source] = {
            "toleranceSeconds": NOTE_ONSET_TOLERANCE_SECONDS,
            "detected": note_f1(detected, expected),
            "detectedOctaveTolerant": note_f1(detected, expected, octave_tolerant=True),
            "notated": note_f1(notated, expected),
        }
    return result


# ---------------------------------------------------------------------------
# Indicators: no reference notes needed. These describe the draft; they are
# not accuracy.


def evaluate_indicators(score: dict[str, Any], notes: list[dict[str, Any]], reference: dict[str, Any]) -> dict[str, Any]:
    in_key: frozenset[int] | None = None
    if reference["key"] is not None:
        root = reference["key"]["rootPitchClass"]
        in_key = frozenset().union(*(scale_pitch_classes(root, name) for name in reference["key"]["collections"]))
    sources: dict[str, dict[str, Any]] = {}
    for source in sorted({note["source"] for note in notes}):
        mine = [note for note in notes if note["source"] == source]
        confidences = [note["confidence"] for note in mine if note["confidence"] is not None]
        sources[source] = {
            "notes": len(mine),
            "notesPerBar": round(len(mine) / len(score["measures"]), 2),
            "lowConfidencePercent": _percent(sum(1 for value in confidences if value < LOW_CONFIDENCE_THRESHOLD), len(confidences)),
            "inReferenceKeyPercent": None if in_key is None else _percent(sum(1 for note in mine if note["midi"] % 12 in in_key), len(mine)),
        }

    bass = None
    segments = [segment for segment in (reference["chords"] or []) if segment["chord"] is not None]
    bass_notes = [note for note in notes if note["source"] == "bass"]
    if segments and bass_notes:
        judged = agree = 0
        for note in bass_notes:
            for segment in segments:
                if segment["start"] <= note["detected"] < segment["end"]:
                    judged += 1
                    agree += note["midi"] % 12 == segment["chord"][0]
                    break
        chord_union = _union((segment["start"], segment["end"]) for segment in segments)
        chord_bars = [m for m in score["measures"] if _overlap_seconds((m["start"], m["end"]), chord_union) > 0]
        bars_with_bass = {note["measure"] for note in bass_notes}
        bass = {
            "notesDuringReferenceChords": judged,
            "onChordRootPercent": _percent(agree, judged),
            "barsWithBassPercent": _percent(sum(1 for m in chord_bars if m["index"] in bars_with_bass), len(chord_bars)),
        }

    voices: dict[str, int] = {}
    hits = 0
    for measure in score["measures"]:
        for hit in measure["hits"]:
            if isinstance(hit, dict):
                hits += 1
                voice = hit.get("broadVoice") if isinstance(hit.get("broadVoice"), str) else "unknown"
                voices[voice] = voices.get(voice, 0) + 1
    return {
        "note": "Indicators describe the draft. They are not accuracy and must not be read as such.",
        "sources": sources,
        "bass": bass,
        "percussion": {"hits": hits, "hitsPerBar": round(hits / len(score["measures"]), 2), "byVoice": dict(sorted(voices.items()))},
    }


# ---------------------------------------------------------------------------
# Whole evaluation, saved runs, and text output


def evaluate(score_payload: dict[str, Any], reference_payload: dict[str, Any], *, analysis: dict[str, Any] | None = None, label: str | None = None) -> dict[str, Any]:
    """Score one saved draft against one reference."""
    reference = load_reference(reference_payload)
    score = load_score(score_payload)
    if analysis is not None and not isinstance(analysis, dict):
        raise ScoreboardError("The analysis must be a JSON object.")
    if label is not None and (not isinstance(label, str) or not _LABEL_PATTERN.match(label)):
        raise ScoreboardError("The label must be 1 to 80 letters, digits, spaces or . _ ( ) + - characters.")
    notes = _score_notes(score)
    return {
        "artifactType": RESULT_ARTIFACT,
        "schemaVersion": SCHEMA_VERSION,
        "scoreboardVersion": SCOREBOARD_VERSION,
        "label": label,
        "reference": {"id": reference["id"], "title": reference["title"]},
        "score": {**score["identity"], "measures": len(score["measures"]), "notes": len(notes)},
        "timing": evaluate_timing(score, reference),
        "key": evaluate_key(score, reference, analysis),
        "chords": evaluate_chords(score, reference),
        "notes": evaluate_notes(notes, reference),
        "indicators": evaluate_indicators(score, notes, reference),
    }


def save_run(path: str | os.PathLike[str], result: dict[str, Any]) -> int:
    """Add a run to a results file, replacing one with the same reference and label."""
    file_path = Path(path)
    runs: list[dict[str, Any]] = []
    if file_path.exists():
        existing = read_json_file(file_path, "results")
        if existing.get("artifactType") != RESULTS_ARTIFACT or existing.get("schemaVersion") != SCHEMA_VERSION:
            raise ScoreboardError("The results file was not written by this scoreboard version.")
        if not isinstance(existing.get("runs"), list):
            raise ScoreboardError("The results file has no list of runs.")
        runs = [run for run in existing["runs"] if isinstance(run, dict)]
    identity = (result["reference"]["id"], result["label"])
    runs = [run for run in runs if (run.get("reference", {}).get("id"), run.get("label")) != identity]
    runs.append(result)
    payload = {"artifactType": RESULTS_ARTIFACT, "schemaVersion": SCHEMA_VERSION, "runs": runs}
    file_path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=file_path.parent, prefix=".scoreboard-", suffix=".tmp", delete=False)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(handle.name, file_path)
    except BaseException:
        try:
            os.unlink(handle.name)
        except OSError:
            pass
        raise
    return len(runs)


def _show(value: Any, suffix: str = "") -> str:
    if value is None:
        return "not available"
    if isinstance(value, bool):
        return "yes" if value else "no"
    return f"{value}{suffix}"


def format_report(result: dict[str, Any]) -> str:
    """Plain-text report for one run."""
    timing = result["timing"]
    lines = [
        f"Reference: {result['reference']['id']}" + (f" ({result['reference']['title']})" if result["reference"]["title"] else ""),
        f"Run: {result['label'] or 'unlabelled'} | {result['score']['measures']} bars, {result['score']['notes']} notes",
        "",
        "ACCURACY AGAINST THE REFERENCE",
        "Timing",
        f"  Tempo: {timing['tempoBpm']} BPM (reference {timing['referenceTempoBpm']}), error {timing['tempoErrorPercent']}%"
        + (f", {timing['tempoOctaveError']}-tempo error" if timing["tempoOctaveError"] else ""),
        f"  Meter: {timing['beatsPerMeasure']} beats per bar (reference {timing['referenceBeatsPerMeasure']}), matches: {_show(timing['meterMatches'])}",
        f"  Bars: {timing['bars']} written, about {timing['expectedBars']} expected",
        f"  Bar-grid drift: {timing['finalGridDriftBeats']} beats by the end, {timing['maxGridDriftBeats']} beats at worst",
        f"  Bars starting on a reference downbeat: {_show(timing['alignedBarsPercent'], '%')}",
    ]
    key = result["key"]
    lines.append("Key")
    if key is None:
        lines.append("  No key in the reference.")
    else:
        lines.append(f"  Reference: {key['reference']['tonalCenter']} ({', '.join(key['reference']['collections'])})")
        for name, title in (("score", "Score tonal context"), ("analysis", "Analysis key")):
            entry = key[name]
            if entry is None:
                lines.append(f"  {title}: not available")
                continue
            written = f", key signature written: {_show(entry['keySignatureWritten'])}" if "keySignatureWritten" in entry else ""
            lines.append(
                f"  {title}: {_show(entry['displayName'])}, confidence {_show(entry['confidence'])}, "
                f"tonal centre right: {_show(entry['tonalCenterMatches'])}, collection accepted: {_show(entry['collectionMatches'])}, "
                f"same notes as the reference: {_show(entry['sameNotes'])}{written}"
            )
    chords = result["chords"]
    lines.append("Chords (share of the reference chord time)")
    if chords is None:
        lines.append("  No chords in the reference.")
    else:
        lines.append(f"  Bars showing a chord symbol: {chords['barsWithSymbol']} of {chords['bars']}")
        for name, title in (("shown", "Symbols shown on bars"), ("candidates", "Harmony windows (not all shown)")):
            entry = chords[name]
            lines.append(
                f"  {title}: covered {_show(entry['coveragePercent'], '%')}, root right {_show(entry['rootAccuracyPercent'], '%')}, "
                f"root and major/minor right {_show(entry['majMinAccuracyPercent'], '%')}, "
                f"root right where shown {_show(entry['rootAccuracyWhereShownPercent'], '%')}"
            )
    notes = result["notes"]
    lines.append("Notes (same pitch, start within 50 ms)")
    if notes is None:
        lines.append("  No notes in the reference, so note accuracy is not measured.")
    else:
        for source, entry in notes.items():
            lines.append(
                f"  {source}: F1 {entry['detected']['f1Percent']}% as detected "
                f"(precision {entry['detected']['precisionPercent']}%, recall {entry['detected']['recallPercent']}%), "
                f"{entry['detectedOctaveTolerant']['f1Percent']}% ignoring octave, {entry['notated']['f1Percent']}% as notated on the grid"
            )
    indicators = result["indicators"]
    lines += ["", "INDICATORS (describe the draft; not accuracy)"]
    for source, entry in indicators["sources"].items():
        lines.append(
            f"  {source}: {entry['notes']} notes, {entry['notesPerBar']} per bar, low confidence {_show(entry['lowConfidencePercent'], '%')}, "
            f"in the reference key {_show(entry['inReferenceKeyPercent'], '%')}"
        )
    if indicators["bass"] is not None:
        bass = indicators["bass"]
        lines.append(
            f"  Bass on the reference chord root: {_show(bass['onChordRootPercent'], '%')} of {bass['notesDuringReferenceChords']} notes; "
            f"bars with any bass note: {_show(bass['barsWithBassPercent'], '%')}"
        )
    percussion = indicators["percussion"]
    lines.append(f"  Percussion: {percussion['hits']} hits, {percussion['hitsPerBar']} per bar")
    return "\n".join(lines)


def format_table(runs: Sequence[dict[str, Any]]) -> str:
    """Side-by-side table of saved runs."""
    header = ("reference", "run", "tempo err %", "max drift (beats)", "aligned bars %", "key centre", "chord root % shown", "chord root % windows", "best note F1 %")
    rows = [header]
    for run in runs:
        timing, key, chords, notes = run.get("timing", {}), run.get("key"), run.get("chords"), run.get("notes")
        best = None
        if isinstance(notes, dict) and notes:
            best = max(entry["detected"]["f1Percent"] for entry in notes.values())
        score_key = key.get("score") if isinstance(key, dict) else None
        rows.append(
            (
                str(run.get("reference", {}).get("id", "?")),
                str(run.get("label") or "unlabelled"),
                _show(timing.get("tempoErrorPercent")),
                _show(timing.get("maxGridDriftBeats")),
                _show(timing.get("alignedBarsPercent")),
                _show(score_key.get("tonalCenterMatches")) if isinstance(score_key, dict) else "not available",
                _show(chords["shown"]["rootAccuracyPercent"]) if isinstance(chords, dict) else "not available",
                _show(chords["candidates"]["rootAccuracyPercent"]) if isinstance(chords, dict) else "not available",
                _show(best),
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    lines = ["  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]
    lines.insert(1, "  ".join("-" * width for width in widths))
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a saved PopEx draft score against a hand-written reference. Reads files only; changes nothing in PopEx.",
    )
    parser.add_argument("--score", help="Saved score JSON (the artifact, the JSON download, or the saved-score API payload with measures).")
    parser.add_argument("--reference", help="Reference JSON (see docs/accuracy-scoreboard.md).")
    parser.add_argument("--analysis", help="Optional audio-analysis.json, to judge the analysis key as well.")
    parser.add_argument("--label", help="Name for this run, for example 'main' or 'beat-grid fix'.")
    parser.add_argument("--save", help="Results JSON to add this run to; a run with the same reference and label is replaced.")
    parser.add_argument("--json", action="store_true", help="Print the result as JSON instead of text.")
    parser.add_argument("--table", help="Print the runs saved in a results JSON side by side and exit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.table:
            if args.score or args.reference:
                raise ScoreboardError("--table prints saved runs; do not combine it with --score or --reference.")
            saved = read_json_file(args.table, "results")
            if saved.get("artifactType") != RESULTS_ARTIFACT or not isinstance(saved.get("runs"), list):
                raise ScoreboardError("The results file was not written by this scoreboard.")
            print(format_table([run for run in saved["runs"] if isinstance(run, dict)]))
            return 0
        if not args.score or not args.reference:
            raise ScoreboardError("Give both --score and --reference, or use --table.")
        result = evaluate(
            read_json_file(args.score, "score"),
            read_json_file(args.reference, "reference"),
            analysis=read_json_file(args.analysis, "analysis") if args.analysis else None,
            label=args.label,
        )
        if args.save:
            save_run(args.save, result)
        print(json.dumps(result, indent=2, allow_nan=False) if args.json else format_report(result))
        return 0
    except ScoreboardError as error:
        print(f"Scoreboard error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
