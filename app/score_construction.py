"""Minimal stdlib-only score construction from existing transcription evidence.

This is the first narrow vertical slice of canonical step 4 (measure, rhythm,
MIDI, and MusicXML score construction). It builds a versioned, honest draft
score document plus minimal MIDI (SMF type 0) and MusicXML (partwise) bytes
from already-validated pitched-note evidence, tempo, and meter.

Scope limits (deliberate):
- read-only pure functions; no database, filesystem, or network access;
- stdlib only (``struct``, ``xml.etree``); no music21/pretty_midi dependency;
- pitched notes only in this slice; percussion, tabs, and parts follow later;
- quantization is explicit and warnings are preserved, never fabricated.
"""

from __future__ import annotations

import copy
import math
import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

SCORE_SCHEMA_VERSION = 1
SCORE_BUILDER_VERSION = "score-construction-v1"

_MAX_NOTES = 5000
_MAX_MEASURES = 2048
_MAX_TEXT = 500
_DIVISIONS = 480  # ticks per quarter note for MIDI + MusicXML divisions

_STEP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_STEP_BASE = ("C", "C", "D", "D", "E", "F", "F", "G", "G", "A", "A", "B")
_STEP_ALTER = (0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0)


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


def _parse_note(value: Any, index: int) -> dict[str, Any]:
    label = f"pitchedNoteEvents[{index}]"
    if not isinstance(value, Mapping):
        raise ScoreConstructionError(f"{label} must be a mapping.")
    allowed = {"id", "startSeconds", "endSeconds", "midiNote", "confidence"}
    unknown = set(value.keys()) - allowed
    if unknown:
        raise ScoreConstructionError(f"{label} has unsupported fields.")
    for key in ("id", "startSeconds", "endSeconds", "midiNote", "confidence"):
        if key not in value:
            raise ScoreConstructionError(f"{label} is missing {key}.")
    return {
        "id": _text(value["id"], f"{label}.id"),
        "startSeconds": _number(value["startSeconds"], f"{label}.startSeconds", minimum=0.0, maximum=36000.0),
        "endSeconds": _number(value["endSeconds"], f"{label}.endSeconds", minimum=0.0, maximum=36000.0),
        "midiNote": _integer(value["midiNote"], f"{label}.midiNote", minimum=0, maximum=127),
        "confidence": _number(value["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0),
    }


def build_score_document(
    pitched_events: Sequence[Mapping[str, Any]],
    *,
    tempo_bpm: float,
    beats_per_measure: int = 4,
    chord_symbols: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Build a versioned draft score document with explicit quantization."""
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
    total_seconds = max((note["endSeconds"] for note in sorted_notes), default=0.0)
    measure_count = min(
        _MAX_MEASURES,
        max(1, int(math.ceil(total_seconds / seconds_per_measure)) if total_seconds > 0 else 1),
    )
    measures: list[dict[str, Any]] = [
        {"measureIndex": index, "startSeconds": index * seconds_per_measure,
         "endSeconds": (index + 1) * seconds_per_measure, "notes": [], "chordSymbol": None}
        for index in range(measure_count)
    ]
    if symbols:
        for index, symbol in enumerate(symbols[:measure_count]):
            measures[index]["chordSymbol"] = symbol

    warnings: list[str] = []
    low_confidence = 0
    large_shift = 0
    eighth = seconds_per_beat / 2.0
    for note in sorted_notes:
        raw_beat = note["startSeconds"] / seconds_per_beat
        quantized_beat = round(raw_beat * 2.0) / 2.0  # 8th-note grid
        shift = abs(quantized_beat - raw_beat) * seconds_per_beat
        if note["confidence"] < 0.50:
            low_confidence += 1
        if shift > 0.050:
            large_shift += 1
        measure_index = min(int(note["startSeconds"] // seconds_per_measure), measure_count - 1)
        entry = {
            "id": note["id"],
            "midiNote": note["midiNote"],
            "noteName": _STEP_NAMES[note["midiNote"] % 12] + str(note["midiNote"] // 12 - 1),
            "startSeconds": note["startSeconds"],
            "endSeconds": note["endSeconds"],
            "startBeat": round(raw_beat, 4),
            "quantizedBeat": quantized_beat,
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
            f"{large_shift} note(s) shifted by more than 50ms during 8th-note quantization."
        )
    if not sorted_notes:
        warnings.append("No pitched-note evidence was available; the draft contains empty measures.")
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
    return copy.deepcopy(document)


def _midi_varlen(value: int) -> bytes:
    if value < 0:
        raise ScoreConstructionError("MIDI tick value is out of range.")
    encoded = bytes([value & 0x7F])
    value >>= 7
    while value:
        encoded = bytes([(value & 0x7F) | 0x80]) + encoded
        value >>= 7
    return encoded


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
    events: list[tuple[int, bytes]] = []
    for measure in measures:
        if not isinstance(measure, Mapping):
            raise ScoreConstructionError("Score measure must be a mapping.")
        for note in measure.get("notes", []):
            if not isinstance(note, Mapping):
                raise ScoreConstructionError("Score note must be a mapping.")
            midi_note = _integer(note.get("midiNote"), "midiNote", minimum=0, maximum=127)
            start_beat = _number(note.get("quantizedBeat"), "quantizedBeat", minimum=0.0, maximum=1_000_000.0)
            raw_end = _number(note.get("endSeconds"), "endSeconds", minimum=0.0, maximum=36000.0)
            raw_start = _number(note.get("startSeconds"), "startSeconds", minimum=0.0, maximum=36000.0)
            duration_beats = max(0.25, (raw_end - raw_start) / (60.0 / tempo))
            duration_beats = min(duration_beats, 16.0)
            start_tick = int(round(start_beat * _DIVISIONS))
            duration_tick = max(60, int(round(duration_beats * _DIVISIONS)))
            velocity = 80
            events.append((start_tick, bytes((0x90, midi_note, velocity))))
            events.append((start_tick + duration_tick, bytes((0x80, midi_note, 0x40))))
    if len(events) > _MAX_NOTES * 2:
        raise ScoreConstructionError("Too many MIDI events.")
    events.sort(key=lambda item: (item[0], item[1]))

    track = bytearray()
    microseconds = int(round(60_000_000 / tempo))
    track += b"\x00\xff\x51\x03" + struct.pack(">I", microseconds)[1:]
    last_tick = 0
    for tick, payload in events:
        track += _midi_varlen(tick - last_tick) + payload
        last_tick = tick
    track += b"\x00\xff\x2f\x00"
    header = struct.pack(">4sIHHH", b"MThd", 6, 0, 1, _DIVISIONS)
    return header + struct.pack(">4sI", b"MTrk", len(track)) + bytes(track)


def score_to_musicxml_text(document: Mapping[str, Any]) -> str:
    """Render minimal MusicXML partwise text from a score document."""
    if not isinstance(document, Mapping):
        raise ScoreConstructionError("Score document must be a mapping.")
    if document.get("schemaVersion") != SCORE_SCHEMA_VERSION:
        raise ScoreConstructionError("Unsupported score schema version.")
    tempo = _number(document.get("tempoBpm"), "tempoBpm", minimum=20.0, maximum=300.0)
    meter = _integer(document.get("beatsPerMeasure"), "beatsPerMeasure", minimum=1, maximum=12)
    measures = document.get("measures")
    if not isinstance(measures, Sequence) or isinstance(measures, (str, bytes)):
        raise ScoreConstructionError("Score measures must be a sequence.")
    if len(measures) > _MAX_MEASURES:
        raise ScoreConstructionError("Too many measures.")

    root = ET.Element("score-partwise", version="3.1")
    ET.SubElement(ET.SubElement(root, "identification"), "encoding").text = (
        f"PopEx {SCORE_BUILDER_VERSION} draft; review required"
    )
    part_list = ET.SubElement(root, "part-list")
    ET.SubElement(ET.SubElement(part_list, "score-part", id="P1"), "part-name").text = "Draft Melody"
    part = ET.SubElement(root, "part", id="P1")
    for measure in measures:
        if not isinstance(measure, Mapping):
            raise ScoreConstructionError("Score measure must be a mapping.")
        measure_el = ET.SubElement(part, "measure", number=str(int(measure.get("measureIndex", 0)) + 1))
        if int(measure_el.get("number", "1")) == 1:
            attrs = ET.SubElement(measure_el, "attributes")
            ET.SubElement(attrs, "divisions").text = str(_DIVISIONS)
            time_el = ET.SubElement(attrs, "time")
            ET.SubElement(time_el, "beats").text = str(meter)
            ET.SubElement(time_el, "beat-type").text = "4"
            direction = ET.SubElement(measure_el, "direction", placement="above")
            direction_type = ET.SubElement(direction, "direction-type")
            ET.SubElement(direction_type, "metronome").text = str(int(round(tempo)))
        chord = measure.get("chordSymbol")
        if chord is not None:
            harmony = ET.SubElement(measure_el, "harmony")
            ET.SubElement(harmony, "root-step").text = "C"
            ET.SubElement(harmony, "kind", text="major").text = "major"
        notes = measure.get("notes", [])
        if not isinstance(notes, Sequence) or isinstance(notes, (str, bytes)):
            raise ScoreConstructionError("Measure notes must be a sequence.")
        if not notes:
            note_el = ET.SubElement(measure_el, "note")
            ET.SubElement(note_el, "rest")
            ET.SubElement(note_el, "duration").text = str(_DIVISIONS * meter)
            continue
        for note in notes:
            if not isinstance(note, Mapping):
                raise ScoreConstructionError("Score note must be a mapping.")
            midi_note = _integer(note.get("midiNote"), "midiNote", minimum=0, maximum=127)
            pitch_class = midi_note % 12
            note_el = ET.SubElement(measure_el, "note")
            pitch_el = ET.SubElement(note_el, "pitch")
            ET.SubElement(pitch_el, "step").text = _STEP_BASE[pitch_class]
            if _STEP_ALTER[pitch_class]:
                ET.SubElement(pitch_el, "alter").text = "1"
            ET.SubElement(pitch_el, "octave").text = str(midi_note // 12 - 1)
            ET.SubElement(note_el, "duration").text = str(_DIVISIONS // 2)
    text = ET.tostring(root, encoding="unicode")
    if len(text) > 2_000_000:
        raise ScoreConstructionError("Generated MusicXML is too large.")
    return text


__all__ = [
    "SCORE_BUILDER_VERSION",
    "SCORE_SCHEMA_VERSION",
    "ScoreConstructionError",
    "build_score_document",
    "score_to_midi_bytes",
    "score_to_musicxml_text",
]
