"""Minimal stdlib-only score construction from existing transcription evidence.

This is the first narrow vertical slice of canonical step 4 (measure, rhythm,
MIDI, and MusicXML score construction). It builds a versioned, honest draft
score document plus minimal MIDI (SMF type 0) and MusicXML (partwise) bytes
from already-validated pitched-note evidence, tempo, and meter.

Scope limits (deliberate):
- read-only pure functions; no database, filesystem, or network access;
- stdlib only (``struct``, ``xml.etree``); no music21/pretty_midi dependency;
- pitched notes plus an optional, separate percussion layer of broad drum
  voices; tabs and instrument-specific parts follow later;
- quantization is explicit and warnings are preserved, never fabricated.

Percussion notation tables (documented, fixed):

=======================  ==============  ========  ===============  ==========
Broad voice              Display step    Notehead  GM drum note     MIDI name
=======================  ==============  ========  ===============  ==========
low_drum                 F4              normal    36               Bass Drum 1
mid_drum                 C5              normal    38               Acoustic Snare
tom_like                 D5              normal    45               Low Tom
closed_high_frequency    G5              x         42               Closed Hi-Hat
open_high_frequency      G5              circle-x  46               Open Hi-Hat
cymbal_like              A5              x         49               Crash Cymbal 1
unresolved_percussion    B4              triangle  76               Hi Wood Block
=======================  ==============  ========  ===============  ==========

Broad voices stay broad: a ``tom_like`` hit is not claimed to be a particular
tom, and ``unresolved_percussion`` hits use a deliberately neutral, non-kit
lane (middle line, triangle notehead, wood-block sound) so they are visible as
unresolved rather than silently assigned to a drum. Percussion is written to
General MIDI channel 10 only; pitched notes never use that channel.
"""

from __future__ import annotations

import copy
import math
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

from app.tablature import (
    TAB_INSTRUMENT_ORDER,
    TAB_INSTRUMENTS,
    TAB_STATUSES,
    tab_position_is_consistent,
)

SCORE_SCHEMA_VERSION = 1
SCORE_BUILDER_VERSION = "score-construction-v3"
# Fixed MusicXML part identities: combined pitched part, drums, then one part
# per fretted instrument (standard staff plus a synchronized TAB staff).
_TAB_PART_IDS = {"bass": "P3", "guitar": "P4"}
_TAB_PART_NAMES = {"bass": "Bass (draft, with TAB)", "guitar": "Guitar (fingering suggestion, with TAB)"}
_TAB_CLEFS = {"bass": ("F", "4"), "guitar": ("G", "2")}
_TAB_GM_PROGRAMS = {"bass": 34, "guitar": 26}  # MusicXML 1-based: fingered bass, steel guitar

# broad voice -> (display step, display octave, notehead, GM drum note, label)
PERCUSSION_NOTATION: dict[str, tuple[str, int, str, int, str]] = {
    "low_drum": ("F", 4, "normal", 36, "Low drum"),
    "mid_drum": ("C", 5, "normal", 38, "Mid drum"),
    "tom_like": ("D", 5, "normal", 45, "Tom-like voice"),
    "closed_high_frequency": ("G", 5, "x", 42, "Closed high-frequency voice"),
    "open_high_frequency": ("G", 5, "circle-x", 46, "Open high-frequency voice"),
    "cymbal_like": ("A", 5, "x", 49, "Cymbal-like voice"),
    "unresolved_percussion": ("B", 4, "triangle", 76, "Unresolved percussion"),
}
UNRESOLVED_PERCUSSION_VOICE = "unresolved_percussion"
PERCUSSION_MIDI_CHANNEL = 9  # zero-based; General MIDI channel 10
_PERCUSSION_FOOT_VOICES = frozenset({"low_drum"})
_PERCUSSION_MIDI_TICKS = 120  # a sixteenth: hits are points, not sustains
_PERCUSSION_PLACEMENTS = frozenset({"placed", "unassigned"})
_PERCUSSION_NOTATION_STATES = frozenset({"notated", "collapsed"})
_OFF_GRID_SECONDS = 0.050

_MAX_NOTES = 5000
_MAX_PERCUSSION_HITS = 8192
_MAX_MEASURES = 2048
_MAX_MUSICXML_FRAGMENTS = 10000
_MAX_TEXT = 500
_MAX_EVENT_WARNINGS = 128
_DIVISIONS = 480  # ticks per quarter note for MIDI + MusicXML divisions

_STEP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_STEP_BASE = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
_STEP_ALTER = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)
_SAFE_SOURCE_KIND = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}\Z")


class ScoreConstructionError(RuntimeError):
    """A score document could not be built from the supplied evidence."""


def _number(value: Any, label: str, *, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ScoreConstructionError(f"{label} must be a number.")
    result = float(value)
    if not math.isfinite(result) or not minimum <= result <= maximum:
        raise ScoreConstructionError(f"{label} is out of range.")
    return result


def _integer(value: Any, label: str, *, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ScoreConstructionError(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise ScoreConstructionError(f"{label} is out of range.")
    return value


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ScoreConstructionError(f"{label} must be text.")
    if len(value) > _MAX_TEXT or any(ord(c) < 0x20 and c not in ("\t",) for c in value):
        raise ScoreConstructionError(f"{label} is unsafe or too long.")
    if "<" in value or ">" in value:
        raise ScoreConstructionError(f"{label} must not contain markup.")
    return value


def _event_warning_text(value: Any, label: str) -> str:
    """Validate bounded warning text while retaining canonical line breaks."""
    if not isinstance(value, str) or len(value) > _MAX_TEXT or value != value.strip():
        raise ScoreConstructionError(f"{label} must be bounded warning text.")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise ScoreConstructionError(f"{label} contains control characters.")
    return value


def _parse_note(value: Any, index: int) -> dict[str, Any]:
    label = f"pitchedNoteEvents[{index}]"
    if not isinstance(value, Mapping):
        raise ScoreConstructionError(f"{label} must be a mapping.")
    required = {"id", "startSeconds", "endSeconds", "midiNote", "confidence"}
    allowed = required | {"midiPitch", "sourceKind", "warnings"}
    unknown = set(value.keys()) - allowed
    if unknown:
        raise ScoreConstructionError(f"{label} has unsupported fields.")
    for key in required:
        if key not in value:
            raise ScoreConstructionError(f"{label} is missing {key}.")
    midi_note = _integer(value["midiNote"], f"{label}.midiNote", minimum=0, maximum=127)
    result = {
        "id": _text(value["id"], f"{label}.id"),
        "startSeconds": _number(value["startSeconds"], f"{label}.startSeconds", minimum=0.0, maximum=36000.0),
        "endSeconds": _number(value["endSeconds"], f"{label}.endSeconds", minimum=0.0, maximum=36000.0),
        "midiNote": midi_note,
        "confidence": _number(value["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0),
    }
    if "midiPitch" in value:
        midi_pitch = _number(value["midiPitch"], f"{label}.midiPitch", minimum=0.0, maximum=127.0)
        if abs(midi_pitch - midi_note) > 0.75:
            raise ScoreConstructionError(f"{label}.midiPitch is inconsistent with midiNote.")
        result["midiPitch"] = midi_pitch
    if "sourceKind" in value:
        source_kind = _text(value["sourceKind"], f"{label}.sourceKind")
        if not _SAFE_SOURCE_KIND.fullmatch(source_kind):
            raise ScoreConstructionError(f"{label}.sourceKind is invalid.")
        result["sourceKind"] = source_kind
    if "warnings" in value:
        source_warnings = value["warnings"]
        if (
            not isinstance(source_warnings, Sequence)
            or isinstance(source_warnings, (str, bytes))
            or len(source_warnings) > _MAX_EVENT_WARNINGS
        ):
            raise ScoreConstructionError(f"{label}.warnings must be a bounded sequence.")
        result["warnings"] = [
            _event_warning_text(item, f"{label}.warnings[{warning_index}]")
            for warning_index, item in enumerate(source_warnings)
        ]
    return result


def _parse_percussion_hit(value: Any, index: int) -> dict[str, Any]:
    label = f"percussionHits[{index}]"
    if not isinstance(value, Mapping):
        raise ScoreConstructionError(f"{label} must be a mapping.")
    required = {
        "eventId",
        "hitIndex",
        "sourceKind",
        "rawKind",
        "broadVoice",
        "resolved",
        "timeSeconds",
        "strength",
        "confidence",
    }
    allowed = required | {"interpretationPlacement"}
    if set(value.keys()) - allowed:
        raise ScoreConstructionError(f"{label} has unsupported fields.")
    for key in required:
        if key not in value:
            raise ScoreConstructionError(f"{label} is missing {key}.")
    broad_voice = value["broadVoice"]
    if broad_voice not in PERCUSSION_NOTATION:
        raise ScoreConstructionError(f"{label}.broadVoice is not a supported broad voice.")
    resolved = value["resolved"]
    if type(resolved) is not bool:
        raise ScoreConstructionError(f"{label}.resolved must be a boolean.")
    if resolved == (broad_voice == UNRESOLVED_PERCUSSION_VOICE):
        raise ScoreConstructionError(
            f"{label} must be unresolved exactly when it uses the unresolved lane."
        )
    placement = value.get("interpretationPlacement")
    if placement is not None and placement not in _PERCUSSION_PLACEMENTS:
        raise ScoreConstructionError(f"{label}.interpretationPlacement is invalid.")
    source_kind = _text(value["sourceKind"], f"{label}.sourceKind")
    raw_kind = _text(value["rawKind"], f"{label}.rawKind")
    for name, token in (("sourceKind", source_kind), ("rawKind", raw_kind)):
        if not _SAFE_SOURCE_KIND.fullmatch(token):
            raise ScoreConstructionError(f"{label}.{name} is invalid.")
    return {
        "eventId": _text(value["eventId"], f"{label}.eventId"),
        "hitIndex": _integer(value["hitIndex"], f"{label}.hitIndex", minimum=0, maximum=63),
        "sourceKind": source_kind,
        "rawKind": raw_kind,
        "broadVoice": broad_voice,
        "resolved": resolved,
        "timeSeconds": _number(
            value["timeSeconds"], f"{label}.timeSeconds", minimum=0.0, maximum=36000.0
        ),
        "strength": _number(value["strength"], f"{label}.strength", minimum=0.0, maximum=1.0),
        "confidence": _number(
            value["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0
        ),
        "interpretationPlacement": placement,
    }


def _place_percussion_hits(
    percussion_hits: Sequence[Mapping[str, Any]],
    *,
    seconds_per_beat: float,
) -> list[dict[str, Any]]:
    """Quantize hits to the eighth grid and collapse same-voice duplicates.

    Every hit is kept with its raw provenance. When two hits of one broad
    voice land in the same grid slot, the most confident one is notated and
    the others are kept as ``collapsed`` evidence instead of being dropped.
    """
    if not isinstance(percussion_hits, Sequence) or isinstance(
        percussion_hits, (str, bytes)
    ):
        raise ScoreConstructionError("percussionHits must be a sequence.")
    if len(percussion_hits) > _MAX_PERCUSSION_HITS:
        raise ScoreConstructionError("Too many percussion hits.")
    hits = [_parse_percussion_hit(item, index) for index, item in enumerate(percussion_hits)]
    keys = [(hit["eventId"], hit["hitIndex"]) for hit in hits]
    if len(keys) != len(set(keys)):
        raise ScoreConstructionError("Duplicate percussion hit identity.")
    placed: list[dict[str, Any]] = []
    for hit in hits:
        raw_beat = hit["timeSeconds"] / seconds_per_beat
        quantized = round(raw_beat * 2.0) / 2.0
        placed.append(
            {
                "eventId": hit["eventId"],
                "hitIndex": hit["hitIndex"],
                "sourceKind": hit["sourceKind"],
                "rawKind": hit["rawKind"],
                "broadVoice": hit["broadVoice"],
                "resolved": hit["resolved"],
                "rawTimeSeconds": hit["timeSeconds"],
                "rawBeat": round(raw_beat, 4),
                "quantizedBeat": quantized,
                "quantizationShiftSeconds": round(
                    abs(quantized - raw_beat) * seconds_per_beat, 4
                ),
                "strength": hit["strength"],
                "confidence": hit["confidence"],
                "interpretationPlacement": hit["interpretationPlacement"],
                "notation": "notated",
            }
        )
    by_slot: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for hit in placed:
        by_slot.setdefault((hit["broadVoice"], hit["quantizedBeat"]), []).append(hit)
    for slot_hits in by_slot.values():
        slot_hits.sort(
            key=lambda item: (
                -item["confidence"],
                item["quantizationShiftSeconds"],
                item["eventId"],
                item["hitIndex"],
            )
        )
        for duplicate in slot_hits[1:]:
            duplicate["notation"] = "collapsed"
    placed.sort(
        key=lambda item: (
            item["quantizedBeat"],
            item["rawTimeSeconds"],
            item["eventId"],
            item["hitIndex"],
        )
    )
    return placed


def build_score_document(
    pitched_events: Sequence[Mapping[str, Any]],
    *,
    tempo_bpm: float,
    beats_per_measure: int = 4,
    chord_symbols: Sequence[str] | None = None,
    percussion_hits: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a versioned draft score document with explicit quantization.

    When ``percussion_hits`` is supplied, every measure gains a separate
    ``percussionHits`` list on the same eighth-note grid; pitched notes and
    percussion hits never share a representation.
    """
    tempo = _number(tempo_bpm, "tempoBpm", minimum=20.0, maximum=300.0)
    meter = _integer(beats_per_measure, "beatsPerMeasure", minimum=1, maximum=12)
    if not isinstance(pitched_events, Sequence) or isinstance(pitched_events, (str, bytes)):
        raise ScoreConstructionError("pitchedNoteEvents must be a sequence.")
    if len(pitched_events) > _MAX_NOTES:
        raise ScoreConstructionError("Too many pitched-note events.")
    notes = [_parse_note(item, index) for index, item in enumerate(pitched_events)]
    for note in notes:
        if not note["startSeconds"] < note["endSeconds"]:
            raise ScoreConstructionError("Note must satisfy startSeconds < endSeconds.")
    ids = [note["id"] for note in notes]
    if len(ids) != len(set(ids)):
        raise ScoreConstructionError("Duplicate pitched-note event ID.")

    seconds_per_beat = 60.0 / tempo
    seconds_per_measure = seconds_per_beat * meter
    symbols: list[str] = []
    if chord_symbols is not None:
        if not isinstance(chord_symbols, Sequence) or isinstance(chord_symbols, (str, bytes)):
            raise ScoreConstructionError("chordSymbols must be a sequence.")
        for index, symbol in enumerate(chord_symbols):
            symbols.append(_text(symbol, f"chordSymbols[{index}]"))
        if len(symbols) > _MAX_MEASURES:
            raise ScoreConstructionError("Too many chord symbols.")

    sorted_notes = sorted(notes, key=lambda item: (item["startSeconds"], item["endSeconds"], item["id"]))
    for note in sorted_notes:
        note["rawStartBeat"] = note["startSeconds"] / seconds_per_beat
        note["rawEndBeat"] = note["endSeconds"] / seconds_per_beat
        note["quantizedStartBeat"] = round(note["rawStartBeat"] * 2.0) / 2.0
        note["quantizedEndBeat"] = round(note["rawEndBeat"] * 2.0) / 2.0
        if note["quantizedEndBeat"] <= note["quantizedStartBeat"]:
            note["quantizedEndBeat"] = note["quantizedStartBeat"] + 0.5
        note["startShiftSeconds"] = abs(
            note["quantizedStartBeat"] - note["rawStartBeat"]
        ) * seconds_per_beat
        note["endShiftSeconds"] = abs(
            note["quantizedEndBeat"] - note["rawEndBeat"]
        ) * seconds_per_beat

    placed_hits = (
        None
        if percussion_hits is None
        else _place_percussion_hits(percussion_hits, seconds_per_beat=seconds_per_beat)
    )
    latest_end_beat = max(
        (note["quantizedEndBeat"] for note in sorted_notes), default=0.0
    )
    if placed_hits:
        # A hit occupies one grid slot, so its bar must contain that slot.
        latest_end_beat = max(latest_end_beat, placed_hits[-1]["quantizedBeat"] + 0.5)
    required_measure_count = max(1, int(math.ceil(latest_end_beat / meter)))
    if required_measure_count > _MAX_MEASURES:
        raise ScoreConstructionError(
            "Score exceeds the supported measure limit; events were not collapsed."
        )
    measure_count = required_measure_count
    measures: list[dict[str, Any]] = [
        {"measureIndex": index, "startSeconds": index * seconds_per_measure,
         "endSeconds": (index + 1) * seconds_per_measure, "notes": [], "chordSymbol": None}
        for index in range(measure_count)
    ]
    if symbols:
        if len(symbols) > measure_count:
            raise ScoreConstructionError(
                "Chord symbols extend beyond the generated measure range."
            )
        for index, symbol in enumerate(symbols[:measure_count]):
            measures[index]["chordSymbol"] = symbol

    warnings: list[str] = []
    low_confidence = 0
    large_shift = 0
    fractional_pitch = 0
    for note in sorted_notes:
        if note["confidence"] < 0.50:
            low_confidence += 1
        shift = max(note["startShiftSeconds"], note["endShiftSeconds"])
        if shift > 0.050:
            large_shift += 1
        midi_pitch = note.get("midiPitch")
        if midi_pitch is not None and abs(midi_pitch - note["midiNote"]) > 0.25:
            fractional_pitch += 1
        measure_index = int(note["quantizedStartBeat"] // meter)
        if not 0 <= measure_index < measure_count:
            raise ScoreConstructionError("Quantized note start is outside the measure range.")
        entry = {
            "id": note["id"],
            "sourceKind": note.get("sourceKind", "unassigned"),
            "sourceWarnings": note.get("warnings", []),
            "midiNote": note["midiNote"],
            "rawMidiPitch": midi_pitch,
            "noteName": _STEP_NAMES[note["midiNote"] % 12] + str(note["midiNote"] // 12 - 1),
            "startSeconds": note["startSeconds"],
            "endSeconds": note["endSeconds"],
            "startBeat": round(note["rawStartBeat"], 4),
            "endBeat": round(note["rawEndBeat"], 4),
            "quantizedBeat": note["quantizedStartBeat"],
            "quantizedEndBeat": note["quantizedEndBeat"],
            "quantizedDurationBeats": note["quantizedEndBeat"] - note["quantizedStartBeat"],
            "quantizationShiftSeconds": round(shift, 4),
            "confidence": note["confidence"],
        }
        measures[measure_index]["notes"].append(copy.deepcopy(entry))
    if low_confidence:
        warnings.append(
            f"{low_confidence} note(s) have confidence below 0.50; review pitches and rhythms."
        )
    if large_shift:
        warnings.append(
            f"{large_shift} note(s) start or end shifted by more than 50ms during 8th-note quantization."
        )
    if fractional_pitch:
        warnings.append(
            f"{fractional_pitch} note(s) differ from the nominal semitone pitch by more than 0.25; "
            "raw pitch is preserved, while MIDI and MusicXML use the nominal semitone pitch."
        )
    if not sorted_notes:
        warnings.append("No pitched-note evidence was available; the draft contains empty measures.")
    if placed_hits is not None:
        for measure in measures:
            measure["percussionHits"] = []
        for hit in placed_hits:
            measure_index = int(hit["quantizedBeat"] // meter)
            if not 0 <= measure_index < measure_count:
                raise ScoreConstructionError("Quantized hit is outside the measure range.")
            measures[measure_index]["percussionHits"].append(dict(hit))
        warnings.extend(percussion_warnings(placed_hits))
    warnings = warnings[:16]

    document = {
        "schemaVersion": SCORE_SCHEMA_VERSION,
        "builderVersion": SCORE_BUILDER_VERSION,
        "tempoBpm": tempo,
        "beatsPerMeasure": meter,
        "divisions": _DIVISIONS,
        "measureCount": measure_count,
        "noteCount": len(sorted_notes),
        "measures": measures,
        "warnings": warnings,
    }
    if placed_hits is not None:
        document["percussionHitCount"] = len(placed_hits)
    return copy.deepcopy(document)


def percussion_hit_counts(hits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize placed percussion hits for review: per voice and honesty counts."""
    by_voice = {voice: 0 for voice in PERCUSSION_NOTATION}
    counts = {
        "hits": 0,
        "notated": 0,
        "collapsed": 0,
        "unresolved": 0,
        "offGrid": 0,
        "unplaced": 0,
    }
    for hit in hits:
        counts["hits"] += 1
        by_voice[hit["broadVoice"]] += 1
        counts["notated" if hit["notation"] == "notated" else "collapsed"] += 1
        if not hit["resolved"]:
            counts["unresolved"] += 1
        if hit["quantizationShiftSeconds"] > _OFF_GRID_SECONDS:
            counts["offGrid"] += 1
        if hit.get("interpretationPlacement") == "unassigned":
            counts["unplaced"] += 1
    counts["byVoice"] = {voice: count for voice, count in by_voice.items() if count}
    return counts


def percussion_warnings(hits: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return honest review warnings for one placed percussion layer."""
    counts = percussion_hit_counts(hits)
    warnings: list[str] = []
    if counts["unresolved"]:
        warnings.append(
            f"{counts['unresolved']} percussion hit(s) are unresolved; they are written "
            "as triangle noteheads on the middle line, not assigned to a drum."
        )
    if counts["offGrid"]:
        warnings.append(
            f"{counts['offGrid']} percussion hit(s) shifted by more than 50ms during "
            "8th-note quantization; fills and 16th-note figures need review."
        )
    if counts["collapsed"]:
        warnings.append(
            f"{counts['collapsed']} percussion hit(s) share a grid slot with a hit of "
            "the same voice and are kept as evidence but not written twice."
        )
    if counts["unplaced"]:
        warnings.append(
            f"{counts['unplaced']} percussion hit(s) had no confident rhythm-grid "
            "placement in the interpretation; they are notated at their quantized raw time."
        )
    return warnings


def _midi_varlen(value: int) -> bytes:
    if not 0 <= value <= 0x0FFFFFFF:
        raise ScoreConstructionError("MIDI tick value is out of range.")
    encoded = bytes([value & 0x7F])
    value >>= 7
    while value:
        encoded = bytes([(value & 0x7F) | 0x80]) + encoded
        value >>= 7
    return encoded


def _notated_percussion_hits(
    measure: Mapping[str, Any],
    position: int,
    *,
    meter: int,
    total_beats: int,
    seen: set[tuple[str, int]],
    slots: set[tuple[str, float]],
) -> list[dict[str, Any]]:
    """Validate one measure's optional percussion hits; return notated ones."""
    hits = measure.get("percussionHits")
    if hits is None:
        return []
    if not isinstance(hits, Sequence) or isinstance(hits, (str, bytes)):
        raise ScoreConstructionError("Measure percussion hits must be a sequence.")
    notated: list[dict[str, Any]] = []
    for hit in hits:
        if not isinstance(hit, Mapping):
            raise ScoreConstructionError("Score percussion hit must be a mapping.")
        identity = (
            _text(hit.get("eventId"), "percussion eventId"),
            _integer(hit.get("hitIndex"), "percussion hitIndex", minimum=0, maximum=63),
        )
        if identity in seen:
            raise ScoreConstructionError("Duplicate score percussion hit.")
        seen.add(identity)
        voice = hit.get("broadVoice")
        if voice not in PERCUSSION_NOTATION:
            raise ScoreConstructionError("Score percussion hit has an unsupported voice.")
        beat = _number(
            hit.get("quantizedBeat"), "percussion quantizedBeat", minimum=0.0, maximum=total_beats
        )
        if int(beat // meter) != position or beat * 2 != int(beat * 2):
            raise ScoreConstructionError("Score percussion hit timing is inconsistent.")
        strength = _number(hit.get("strength"), "percussion strength", minimum=0.0, maximum=1.0)
        if hit.get("notation") not in _PERCUSSION_NOTATION_STATES:
            raise ScoreConstructionError("Score percussion hit notation state is invalid.")
        if hit["notation"] != "notated":
            continue
        if (voice, beat) in slots:
            raise ScoreConstructionError("Two notated percussion hits share one slot.")
        slots.add((voice, beat))
        notated.append({"broadVoice": voice, "quantizedBeat": beat, "strength": strength})
    return notated


def score_to_midi_bytes(document: Mapping[str, Any]) -> bytes:
    """Render a minimal SMF type-0 MIDI file from a score document."""
    if not isinstance(document, Mapping):
        raise ScoreConstructionError("Score document must be a mapping.")
    if document.get("schemaVersion") != SCORE_SCHEMA_VERSION:
        raise ScoreConstructionError("Unsupported score schema version.")
    tempo = _number(document.get("tempoBpm"), "tempoBpm", minimum=20.0, maximum=300.0)
    measures = document.get("measures")
    if not isinstance(measures, Sequence) or isinstance(measures, (str, bytes)):
        raise ScoreConstructionError("Score measures must be a sequence.")
    meter = _integer(document.get("beatsPerMeasure"), "beatsPerMeasure", minimum=1, maximum=12)
    measure_count = _integer(
        document.get("measureCount"), "measureCount", minimum=1, maximum=_MAX_MEASURES
    )
    if document.get("divisions") != _DIVISIONS:
        raise ScoreConstructionError("Unsupported score divisions.")
    if len(measures) != measure_count:
        raise ScoreConstructionError("Score measure count is inconsistent.")
    total_beats = measure_count * meter
    notes: list[tuple[int, int, int, str]] = []
    drum_hits: list[dict[str, Any]] = []
    hit_ids: set[tuple[str, int]] = set()
    hit_slots: set[tuple[str, float]] = set()
    event_ids: set[str] = set()
    for position, measure in enumerate(measures):
        if not isinstance(measure, Mapping):
            raise ScoreConstructionError("Score measure must be a mapping.")
        measure_index = _integer(
            measure.get("measureIndex"),
            "measureIndex",
            minimum=0,
            maximum=_MAX_MEASURES - 1,
        )
        if measure_index != position:
            raise ScoreConstructionError("Score measure index is inconsistent.")
        drum_hits.extend(
            _notated_percussion_hits(
                measure,
                position,
                meter=meter,
                total_beats=total_beats,
                seen=hit_ids,
                slots=hit_slots,
            )
        )
        measure_notes = measure.get("notes", [])
        if not isinstance(measure_notes, Sequence) or isinstance(measure_notes, (str, bytes)):
            raise ScoreConstructionError("Measure notes must be a sequence.")
        for note in measure_notes:
            if not isinstance(note, Mapping):
                raise ScoreConstructionError("Score note must be a mapping.")
            midi_note = _integer(note.get("midiNote"), "midiNote", minimum=0, maximum=127)
            event_id = _text(note.get("id"), "score note id")
            if event_id in event_ids:
                raise ScoreConstructionError("Duplicate score event ID.")
            event_ids.add(event_id)
            start_beat = _number(
                note.get("quantizedBeat"), "quantizedBeat", minimum=0.0, maximum=total_beats
            )
            end_beat = _number(
                note.get("quantizedEndBeat"),
                "quantizedEndBeat",
                minimum=0.0,
                maximum=total_beats,
            )
            if end_beat <= start_beat or int(start_beat // meter) != position:
                raise ScoreConstructionError("Score note timing is inconsistent.")
            start_tick = int(round(start_beat * _DIVISIONS))
            end_tick = int(round(end_beat * _DIVISIONS))
            duration_tick = end_tick - start_tick
            if duration_tick <= 0:
                raise ScoreConstructionError("Score note duration is invalid.")
            notes.append((start_tick, end_tick, midi_note, event_id))
    if len(notes) > _MAX_NOTES:
        raise ScoreConstructionError("Too many MIDI events.")

    # A note-off must not silence another still-held copy of the same pitch.
    # Different pitches can share a channel; overlapping unisons cannot.
    # General MIDI channel 10 is reserved for percussion, not this pitched draft.
    melodic_channels = tuple(channel for channel in range(16) if channel != 9)
    channel_ends: dict[int, list[int]] = {}
    events: list[tuple[int, bytes]] = []
    for start_tick, end_tick, midi_note, _ in sorted(
        notes, key=lambda note: (note[0], note[1], note[3])
    ):
        ends = channel_ends.setdefault(midi_note, [0] * len(melodic_channels))
        available = next((index for index, end in enumerate(ends) if end <= start_tick), None)
        if available is None:
            raise ScoreConstructionError("MIDI overlapping unison channel limit exceeded.")
        channel = melodic_channels[available]
        ends[available] = end_tick
        events.append((start_tick, bytes((0x90 | channel, midi_note, 80))))
        events.append((end_tick, bytes((0x80 | channel, midi_note, 0x40))))
    # Percussion uses only General MIDI channel 10 and the documented table.
    # Velocity follows measured onset strength; nothing else is inferred.
    for hit in drum_hits:
        gm_note = PERCUSSION_NOTATION[hit["broadVoice"]][3]
        start_tick = int(round(hit["quantizedBeat"] * _DIVISIONS))
        velocity = max(1, min(127, int(round(40 + 80 * hit["strength"]))))
        events.append(
            (start_tick, bytes((0x90 | PERCUSSION_MIDI_CHANNEL, gm_note, velocity)))
        )
        events.append(
            (
                start_tick + _PERCUSSION_MIDI_TICKS,
                bytes((0x80 | PERCUSSION_MIDI_CHANNEL, gm_note, 0x40)),
            )
        )
    events.sort(key=lambda item: (item[0], item[1]))

    track = bytearray()
    microseconds = int(round(60_000_000 / tempo))
    track += b"\x00\xff\x51\x03" + struct.pack(">I", microseconds)[1:]
    track += b"\x00\xff\x58\x04" + bytes((meter, 2, 24, 8))
    last_tick = 0
    for tick, payload in events:
        track += _midi_varlen(tick - last_tick) + payload
        last_tick = tick
    track += _midi_varlen(total_beats * _DIVISIONS - last_tick) + b"\xff\x2f\x00"
    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, _DIVISIONS)
    return header + struct.pack(">4sI", b"MTrk", len(track)) + bytes(track)


def _append_forward(parent: ET.Element, duration_ticks: int) -> None:
    if duration_ticks <= 0:
        return
    forward = ET.SubElement(parent, "forward")
    ET.SubElement(forward, "duration").text = str(duration_ticks)


def _musicxml_note(
    measure: ET.Element,
    *,
    midi_note: int,
    duration_ticks: int,
    voice: int,
    tie_stop: bool,
    tie_start: bool,
    staff: int = 1,
    technical: tuple[int, int] | None = None,
) -> None:
    pitch_class = midi_note % 12
    note_el = ET.SubElement(measure, "note")
    pitch_el = ET.SubElement(note_el, "pitch")
    ET.SubElement(pitch_el, "step").text = _STEP_BASE[pitch_class]
    if _STEP_ALTER[pitch_class]:
        ET.SubElement(pitch_el, "alter").text = "1"
    ET.SubElement(pitch_el, "octave").text = str(midi_note // 12 - 1)
    ET.SubElement(note_el, "duration").text = str(duration_ticks)
    tie_types = (["stop"] if tie_stop else []) + (["start"] if tie_start else [])
    for tie_type in tie_types:
        ET.SubElement(note_el, "tie", type=tie_type)
    ET.SubElement(note_el, "voice").text = str(voice)
    if _STEP_ALTER[pitch_class] and technical is None:
        ET.SubElement(note_el, "accidental").text = "sharp"
    ET.SubElement(note_el, "staff").text = str(staff)
    if tie_types or technical is not None:
        notations = ET.SubElement(note_el, "notations")
        for tie_type in tie_types:
            ET.SubElement(notations, "tied", type=tie_type)
        if technical is not None:
            technical_el = ET.SubElement(notations, "technical")
            ET.SubElement(technical_el, "string").text = str(technical[0])
            ET.SubElement(technical_el, "fret").text = str(technical[1])


def _parse_note_tab(value: Any, midi_note: int) -> dict[str, Any] | None:
    """Validate one note's optional tablature suggestion."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"instrument", "status", "string", "fret"}:
        raise ScoreConstructionError("Score note tablature is malformed.")
    instrument = value["instrument"]
    status = value["status"]
    if instrument not in TAB_INSTRUMENTS or status not in TAB_STATUSES:
        raise ScoreConstructionError("Score note tablature is unsupported.")
    string, fret = value["string"], value["fret"]
    if status == "assigned":
        if (
            isinstance(string, bool)
            or isinstance(fret, bool)
            or not isinstance(string, int)
            or not isinstance(fret, int)
            or not tab_position_is_consistent(midi_note, instrument, string, fret)
        ):
            raise ScoreConstructionError("Score note tablature does not sound the note's pitch.")
    elif string is not None or fret is not None:
        raise ScoreConstructionError("An unassigned tablature note cannot name a position.")
    return {"instrument": instrument, "status": status, "string": string, "fret": fret}


def _voice_lanes(notes: Sequence[Mapping[str, Any]], first_voice: int) -> tuple[dict[str, int], int]:
    """Keep source kinds in separate voices and split overlaps into more voices."""
    voice_by_id: dict[str, int] = {}
    next_voice = first_voice
    for source_kind in sorted({note["sourceKind"] for note in notes}):
        source_notes = sorted(
            (note for note in notes if note["sourceKind"] == source_kind),
            key=lambda item: (item["quantizedBeat"], item["quantizedEndBeat"], item["id"]),
        )
        lane_ends: list[float] = []
        for note in source_notes:
            lane = next(
                (index for index, lane_end in enumerate(lane_ends) if lane_end <= note["quantizedBeat"]),
                None,
            )
            if lane is None:
                lane = len(lane_ends)
                lane_ends.append(note["quantizedEndBeat"])
            else:
                lane_ends[lane] = note["quantizedEndBeat"]
            voice_by_id[note["id"]] = next_voice + lane
        next_voice += len(lane_ends)
    return voice_by_id, next_voice


def _measure_fragments(
    notes: Sequence[Mapping[str, Any]],
    voice_by_id: Mapping[str, int],
    *,
    meter: int,
    budget: list[int],
) -> dict[int, dict[int, list[dict[str, Any]]]]:
    """Split sustained notes at bar lines; bound the total fragment count."""
    fragments_by_measure: dict[int, dict[int, list[dict[str, Any]]]] = {}
    for note in notes:
        first_measure = int(note["quantizedBeat"] // meter)
        end_measure = int(math.ceil(note["quantizedEndBeat"] / meter))
        budget[0] += end_measure - first_measure
        if budget[0] > _MAX_MUSICXML_FRAGMENTS:
            raise ScoreConstructionError(
                "MusicXML note splitting exceeds the supported fragment limit."
            )
        for position in range(first_measure, end_measure):
            bar_start = position * meter
            bar_end = bar_start + meter
            fragment_start = max(note["quantizedBeat"], bar_start)
            fragment_end = min(note["quantizedEndBeat"], bar_end)
            if fragment_start >= fragment_end:
                continue
            fragment = {
                **note,
                "localStartBeat": fragment_start - bar_start,
                "localEndBeat": fragment_end - bar_start,
                "tieStop": note["quantizedBeat"] < bar_start,
                "tieStart": note["quantizedEndBeat"] > bar_end,
                "voice": voice_by_id[note["id"]],
            }
            fragments_by_measure.setdefault(position, {}).setdefault(
                fragment["voice"], []
            ).append(fragment)
    return fragments_by_measure


def _write_measure_voices(
    measure_el: ET.Element,
    voices: Sequence[tuple[int, Sequence[Mapping[str, Any]], int, bool]],
    bar_ticks: int,
) -> None:
    """Write ``(voice, fragments, staff, with_tab)`` lanes separated by backups.

    A forward is only a timing spacer. It does not claim that missing
    transcription evidence proves a musical rest.
    """
    if not voices:
        _append_forward(measure_el, bar_ticks)
        return
    for index, (voice, fragments, staff, with_tab) in enumerate(voices):
        if index:
            ET.SubElement(ET.SubElement(measure_el, "backup"), "duration").text = str(bar_ticks)
        cursor_ticks = 0
        for fragment in sorted(
            fragments, key=lambda item: (item["localStartBeat"], item["localEndBeat"], item["id"])
        ):
            start_ticks = int(round(fragment["localStartBeat"] * _DIVISIONS))
            end_ticks = int(round(fragment["localEndBeat"] * _DIVISIONS))
            if start_ticks < cursor_ticks or end_ticks <= start_ticks:
                raise ScoreConstructionError("Overlapping score notes share a MusicXML voice.")
            _append_forward(measure_el, start_ticks - cursor_ticks)
            tab = fragment.get("tab")
            _musicxml_note(
                measure_el,
                midi_note=fragment["midiNote"],
                duration_ticks=end_ticks - start_ticks,
                voice=voice,
                tie_stop=fragment["tieStop"],
                tie_start=fragment["tieStart"],
                staff=staff,
                technical=(tab["string"], tab["fret"]) if with_tab else None,
            )
            cursor_ticks = end_ticks
        _append_forward(measure_el, bar_ticks - cursor_ticks)


def _musicxml_tab_score_part(part_list: ET.Element, instrument_id: str) -> None:
    part_id = _TAB_PART_IDS[instrument_id]
    score_part = ET.SubElement(part_list, "score-part", id=part_id)
    ET.SubElement(score_part, "part-name").text = _TAB_PART_NAMES[instrument_id]
    instrument = ET.SubElement(score_part, "score-instrument", id=f"{part_id}-I1")
    ET.SubElement(instrument, "instrument-name").text = TAB_INSTRUMENTS[instrument_id].label
    midi = ET.SubElement(score_part, "midi-instrument", id=f"{part_id}-I1")
    ET.SubElement(midi, "midi-program").text = str(_TAB_GM_PROGRAMS[instrument_id])


def _musicxml_tab_part(
    root: ET.Element,
    instrument_id: str,
    notes: Sequence[Mapping[str, Any]],
    *,
    measure_count: int,
    meter: int,
    budget: list[int],
) -> None:
    """Write one fretted instrument: standard staff 1 and TAB staff 2.

    Both staves carry the same notes and durations so they stay synchronized;
    notes without a playable position appear only on the standard staff.
    """
    instrument = TAB_INSTRUMENTS[instrument_id]
    voice_by_id, next_voice = _voice_lanes(notes, 1)
    standard = _measure_fragments(notes, voice_by_id, meter=meter, budget=budget)
    fingered = [note for note in notes if note["tab"]["status"] == "assigned"]
    tab_voice_by_id = {
        note_id: voice + max(4, next_voice - 1)
        for note_id, voice in voice_by_id.items()
    }
    tab = _measure_fragments(fingered, tab_voice_by_id, meter=meter, budget=budget)
    bar_ticks = meter * _DIVISIONS
    part = ET.SubElement(root, "part", id=_TAB_PART_IDS[instrument_id])
    for position in range(measure_count):
        measure_el = ET.SubElement(part, "measure", number=str(position + 1))
        if position == 0:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(_DIVISIONS)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(meter)
            ET.SubElement(time_el, "beat-type").text = "4"
            ET.SubElement(attrs, "staves").text = "2"
            sign, line = _TAB_CLEFS[instrument_id]
            clef = ET.SubElement(attrs, "clef", number="1")
            ET.SubElement(clef, "sign").text = sign
            ET.SubElement(clef, "line").text = line
            # Fretted instruments sound an octave below written pitch; pitches
            # stay at sounding pitch and the clef carries the octave.
            ET.SubElement(clef, "clef-octave-change").text = "-1"
            tab_clef = ET.SubElement(attrs, "clef", number="2")
            ET.SubElement(tab_clef, "sign").text = "TAB"
            ET.SubElement(tab_clef, "line").text = "5"
            details = ET.SubElement(attrs, "staff-details", number="2")
            ET.SubElement(details, "staff-lines").text = str(len(instrument.strings))
            for line_number, open_note in enumerate(reversed(instrument.strings), start=1):
                tuning = ET.SubElement(details, "staff-tuning", line=str(line_number))
                pitch_class = open_note % 12
                ET.SubElement(tuning, "tuning-step").text = _STEP_BASE[pitch_class]
                if _STEP_ALTER[pitch_class]:
                    ET.SubElement(tuning, "tuning-alter").text = "1"
                ET.SubElement(tuning, "tuning-octave").text = str(open_note // 12 - 1)
        lanes = [
            (voice, fragments, 1, False)
            for voice, fragments in sorted(standard.get(position, {}).items())
        ] + [
            (voice, fragments, 2, True)
            for voice, fragments in sorted(tab.get(position, {}).items())
        ]
        _write_measure_voices(measure_el, lanes, bar_ticks)


def _percussion_instrument_id(voice: str) -> str:
    return f"P2-I{PERCUSSION_NOTATION[voice][3]}"


def _musicxml_percussion_score_part(part_list: ET.Element, voices: Sequence[str]) -> None:
    score_part = ET.SubElement(part_list, "score-part", id="P2")
    ET.SubElement(score_part, "part-name").text = "Drum Kit (draft, broad voices)"
    for voice in voices:
        instrument = ET.SubElement(
            score_part, "score-instrument", id=_percussion_instrument_id(voice)
        )
        label = PERCUSSION_NOTATION[voice][4]
        if voice == UNRESOLVED_PERCUSSION_VOICE:
            label += " (review)"
        ET.SubElement(instrument, "instrument-name").text = label
    for voice in voices:
        midi = ET.SubElement(
            score_part, "midi-instrument", id=_percussion_instrument_id(voice)
        )
        ET.SubElement(midi, "midi-channel").text = str(PERCUSSION_MIDI_CHANNEL + 1)
        # MusicXML numbers unpitched MIDI keys from 1.
        ET.SubElement(midi, "midi-unpitched").text = str(PERCUSSION_NOTATION[voice][3] + 1)


def _musicxml_percussion_part(
    root: ET.Element,
    hits_by_measure: Sequence[Sequence[Mapping[str, Any]]],
    *,
    meter: int,
    unresolved_present: bool,
) -> None:
    """Write broad drum voices as unpitched notes on a percussion staff.

    Hands and the low drum use two conventional voices (stems up and down).
    Every hit fills one eighth-note grid slot; empty time uses ``forward``
    spacers rather than claiming that the recording proves a rest.
    """
    slot_ticks = _DIVISIONS // 2
    bar_ticks = meter * _DIVISIONS
    part = ET.SubElement(root, "part", id="P2")
    order = list(PERCUSSION_NOTATION)
    for position, hits in enumerate(hits_by_measure):
        measure_el = ET.SubElement(part, "measure", number=str(position + 1))
        if position == 0:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(_DIVISIONS)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(meter)
            ET.SubElement(time_el, "beat-type").text = "4"
            ET.SubElement(ET.SubElement(attrs, "clef"), "sign").text = "percussion"
            if unresolved_present:
                direction = ET.SubElement(measure_el, "direction", placement="above")
                ET.SubElement(
                    ET.SubElement(direction, "direction-type"), "words"
                ).text = "Triangle noteheads on the middle line are unresolved percussion; review them."
        groups = (
            (1, "up", [hit for hit in hits if hit["broadVoice"] not in _PERCUSSION_FOOT_VOICES]),
            (2, "down", [hit for hit in hits if hit["broadVoice"] in _PERCUSSION_FOOT_VOICES]),
        )
        written = False
        for voice_number, stem, group in groups:
            if not group:
                continue
            if written:
                ET.SubElement(ET.SubElement(measure_el, "backup"), "duration").text = str(
                    bar_ticks
                )
            written = True
            by_tick: dict[int, list[Mapping[str, Any]]] = {}
            for hit in group:
                tick = int(round((hit["quantizedBeat"] - position * meter) * _DIVISIONS))
                by_tick.setdefault(tick, []).append(hit)
            cursor = 0
            for tick in sorted(by_tick):
                _append_forward(measure_el, tick - cursor)
                chord_hits = sorted(by_tick[tick], key=lambda item: order.index(item["broadVoice"]))
                for chord_index, hit in enumerate(chord_hits):
                    step, octave, notehead, _gm, _label = PERCUSSION_NOTATION[hit["broadVoice"]]
                    note_el = ET.SubElement(measure_el, "note")
                    if chord_index:
                        ET.SubElement(note_el, "chord")
                    unpitched = ET.SubElement(note_el, "unpitched")
                    ET.SubElement(unpitched, "display-step").text = step
                    ET.SubElement(unpitched, "display-octave").text = str(octave)
                    ET.SubElement(note_el, "duration").text = str(slot_ticks)
                    ET.SubElement(
                        note_el, "instrument", id=_percussion_instrument_id(hit["broadVoice"])
                    )
                    ET.SubElement(note_el, "voice").text = str(voice_number)
                    ET.SubElement(note_el, "type").text = "eighth"
                    ET.SubElement(note_el, "stem").text = stem
                    if notehead != "normal":
                        ET.SubElement(note_el, "notehead").text = notehead
                cursor = tick + slot_ticks
            _append_forward(measure_el, bar_ticks - cursor)
        if not written:
            _append_forward(measure_el, bar_ticks)


def score_to_musicxml_text(document: Mapping[str, Any]) -> str:
    """Render standards-shaped MusicXML with measured timing and tied bars."""
    if not isinstance(document, Mapping):
        raise ScoreConstructionError("Score document must be a mapping.")
    if document.get("schemaVersion") != SCORE_SCHEMA_VERSION:
        raise ScoreConstructionError("Unsupported score schema version.")
    tempo = _number(document.get("tempoBpm"), "tempoBpm", minimum=20.0, maximum=300.0)
    meter = _integer(document.get("beatsPerMeasure"), "beatsPerMeasure", minimum=1, maximum=12)
    measures = document.get("measures")
    if not isinstance(measures, Sequence) or isinstance(measures, (str, bytes)):
        raise ScoreConstructionError("Score measures must be a sequence.")
    measure_count = _integer(
        document.get("measureCount"), "measureCount", minimum=1, maximum=_MAX_MEASURES
    )
    if len(measures) != measure_count:
        raise ScoreConstructionError("Score measure count is inconsistent.")
    if document.get("divisions") != _DIVISIONS:
        raise ScoreConstructionError("Unsupported score divisions.")

    all_notes: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    validated_measures: list[Mapping[str, Any]] = []
    drum_hits_by_measure: list[list[dict[str, Any]]] = []
    hit_ids: set[tuple[str, int]] = set()
    hit_slots: set[tuple[str, float]] = set()
    total_beats = measure_count * meter
    for position, measure in enumerate(measures):
        if not isinstance(measure, Mapping):
            raise ScoreConstructionError("Score measure must be a mapping.")
        measure_index = _integer(
            measure.get("measureIndex"),
            "measureIndex",
            minimum=0,
            maximum=_MAX_MEASURES - 1,
        )
        if measure_index != position:
            raise ScoreConstructionError("Score measure index is inconsistent.")
        validated_measures.append(measure)
        drum_hits_by_measure.append(
            _notated_percussion_hits(
                measure,
                position,
                meter=meter,
                total_beats=total_beats,
                seen=hit_ids,
                slots=hit_slots,
            )
        )
        notes = measure.get("notes", [])
        if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
            raise ScoreConstructionError("Measure notes must be a sequence.")
        chord = measure.get("chordSymbol")
        if chord is not None:
            _text(chord, "chordSymbol")
        for note in notes:
            if not isinstance(note, Mapping):
                raise ScoreConstructionError("Score note must be a mapping.")
            event_id = _text(note.get("id"), "score note id")
            if event_id in event_ids:
                raise ScoreConstructionError("Duplicate score event ID.")
            event_ids.add(event_id)
            start_beat = _number(
                note.get("quantizedBeat"), "quantizedBeat", minimum=0.0, maximum=total_beats
            )
            end_beat = _number(
                note.get("quantizedEndBeat"),
                "quantizedEndBeat",
                minimum=0.0,
                maximum=total_beats,
            )
            if end_beat <= start_beat or int(start_beat // meter) != position:
                raise ScoreConstructionError("Score note timing is inconsistent.")
            midi_note = _integer(note.get("midiNote"), "midiNote", minimum=0, maximum=127)
            source_kind = note.get("sourceKind", "unassigned")
            if midi_note < 12:
                raise ScoreConstructionError(
                    "Note is below the supported MusicXML pitch range (C0 and above)."
                )
            source_kind = _text(source_kind, "sourceKind")
            if not _SAFE_SOURCE_KIND.fullmatch(source_kind):
                raise ScoreConstructionError("Score sourceKind is invalid.")
            all_notes.append(
                {
                    "id": event_id,
                    "sourceKind": source_kind,
                    "midiNote": midi_note,
                    "quantizedBeat": start_beat,
                    "quantizedEndBeat": end_beat,
                    "tab": _parse_note_tab(note.get("tab"), midi_note),
                }
            )
    if len(all_notes) > _MAX_NOTES:
        raise ScoreConstructionError("Too many score notes.")

    tab_notes = {
        instrument: [note for note in all_notes if note["tab"] and note["tab"]["instrument"] == instrument]
        for instrument in TAB_INSTRUMENT_ORDER
    }
    tab_instruments = [instrument for instrument in TAB_INSTRUMENT_ORDER if tab_notes[instrument]]
    combined_notes = [note for note in all_notes if note["tab"] is None]
    budget = [0]
    voice_by_id, _ = _voice_lanes(combined_notes, 1)
    fragments_by_measure = _measure_fragments(
        combined_notes, voice_by_id, meter=meter, budget=budget
    )

    root = ET.Element("score-partwise", version="3.1")
    encoding = ET.SubElement(ET.SubElement(root, "identification"), "encoding")
    ET.SubElement(encoding, "software").text = f"PopEx {SCORE_BUILDER_VERSION}"
    ET.SubElement(encoding, "encoding-description").text = "Draft; review required"
    part_list = ET.SubElement(root, "part-list")
    ET.SubElement(ET.SubElement(part_list, "score-part", id="P1"), "part-name").text = (
        "Other Pitched Lines" if tab_instruments else "Draft Pitched Events"
    )
    for instrument in tab_instruments:
        _musicxml_tab_score_part(part_list, instrument)
    drum_voices = sorted(
        {hit["broadVoice"] for hits in drum_hits_by_measure for hit in hits},
        key=list(PERCUSSION_NOTATION).index,
    )
    if drum_voices:
        _musicxml_percussion_score_part(part_list, drum_voices)
    part = ET.SubElement(root, "part", id="P1")
    bar_ticks = meter * _DIVISIONS
    for position, measure in enumerate(validated_measures):
        measure_el = ET.SubElement(part, "measure", number=str(position + 1))
        if position == 0:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(_DIVISIONS)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(meter)
            ET.SubElement(time_el, "beat-type").text = "4"
            clef = ET.SubElement(attrs, "clef")
            ET.SubElement(clef, "sign").text = "G"
            ET.SubElement(clef, "line").text = "2"
            direction = ET.SubElement(measure_el, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            metronome = ET.SubElement(direction_type, "metronome")
            ET.SubElement(metronome, "beat-unit").text = "quarter"
            ET.SubElement(metronome, "per-minute").text = f"{tempo:.6g}"
            ET.SubElement(direction, "sound", tempo=f"{tempo:.6g}")
        chord = measure.get("chordSymbol")
        if chord is not None:
            # Preserve exact text without fabricating a parsed root/kind claim.
            direction = ET.SubElement(measure_el, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "words").text = chord
        _write_measure_voices(
            measure_el,
            [
                (voice, fragments, 1, False)
                for voice, fragments in sorted(fragments_by_measure.get(position, {}).items())
            ],
            bar_ticks,
        )

    for instrument in tab_instruments:
        _musicxml_tab_part(
            root,
            instrument,
            tab_notes[instrument],
            measure_count=measure_count,
            meter=meter,
            budget=budget,
        )

    if drum_voices:
        _musicxml_percussion_part(
            root,
            drum_hits_by_measure,
            meter=meter,
            unresolved_present=UNRESOLVED_PERCUSSION_VOICE in drum_voices,
        )

    text = ET.tostring(root, encoding="unicode")
    if len(text) > 2_000_000:
        raise ScoreConstructionError("Generated MusicXML is too large.")
    return text


__all__ = [
    "PERCUSSION_MIDI_CHANNEL",
    "PERCUSSION_NOTATION",
    "SCORE_BUILDER_VERSION",
    "SCORE_SCHEMA_VERSION",
    "ScoreConstructionError",
    "UNRESOLVED_PERCUSSION_VOICE",
    "build_score_document",
    "percussion_hit_counts",
    "percussion_warnings",
    "score_to_midi_bytes",
    "score_to_musicxml_text",
]
