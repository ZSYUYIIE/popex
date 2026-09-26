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
import re
import struct
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from typing import Any

SCORE_SCHEMA_VERSION = 1
SCORE_BUILDER_VERSION = "score-construction-v1"

_MAX_NOTES = 5000
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

    latest_end_beat = max(
        (note["quantizedEndBeat"] for note in sorted_notes), default=0.0
    )
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
            f"{fractional_pitch} note(s) differ from the nearest semitone by more than 0.25; "
            "raw pitch is preserved, while MIDI and MusicXML use the nearest semitone."
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
    if not 0 <= value <= 0x0FFFFFFF:
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
    meter = _integer(document.get("beatsPerMeasure"), "beatsPerMeasure", minimum=1, maximum=12)
    measure_count = _integer(
        document.get("measureCount"), "measureCount", minimum=1, maximum=_MAX_MEASURES
    )
    if document.get("divisions") != _DIVISIONS:
        raise ScoreConstructionError("Unsupported score divisions.")
    if len(measures) != measure_count:
        raise ScoreConstructionError("Score measure count is inconsistent.")
    total_beats = measure_count * meter
    events: list[tuple[int, bytes]] = []
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
            velocity = 80
            events.append((start_tick, bytes((0x90, midi_note, velocity))))
            events.append((end_tick, bytes((0x80, midi_note, 0x40))))
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
    if _STEP_ALTER[pitch_class]:
        ET.SubElement(note_el, "accidental").text = "sharp"
    ET.SubElement(note_el, "staff").text = "1"
    if tie_types:
        notations = ET.SubElement(note_el, "notations")
        for tie_type in tie_types:
            ET.SubElement(notations, "tied", type=tie_type)


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
                }
            )
    if len(all_notes) > _MAX_NOTES:
        raise ScoreConstructionError("Too many score notes.")

    # Keep source kinds in independent MusicXML voices and split overlapping
    # events within a source into additional voices without inventing parts.
    voice_by_id: dict[str, int] = {}
    source_kinds = sorted({note["sourceKind"] for note in all_notes})
    next_voice = 1
    for source_kind in source_kinds:
        source_notes = sorted(
            (note for note in all_notes if note["sourceKind"] == source_kind),
            key=lambda item: (
                item["quantizedBeat"],
                item["quantizedEndBeat"],
                item["id"],
            ),
        )
        lane_ends: list[float] = []
        for note in source_notes:
            lane = next(
                (
                    index
                    for index, lane_end in enumerate(lane_ends)
                    if lane_end <= note["quantizedBeat"]
                ),
                None,
            )
            if lane is None:
                lane = len(lane_ends)
                lane_ends.append(note["quantizedEndBeat"])
            else:
                lane_ends[lane] = note["quantizedEndBeat"]
            voice_by_id[note["id"]] = next_voice + lane
        next_voice += len(lane_ends)

    # A note sustained across bar lines creates one MusicXML note per bar.
    # Bound that expansion before allocating ElementTree nodes, then index
    # fragments by measure so sparse scores do not scan every event in every bar.
    fragments_by_measure: dict[int, dict[int, list[dict[str, Any]]]] = {}
    fragment_count = 0
    for note in all_notes:
        first_measure = int(note["quantizedBeat"] // meter)
        end_measure = int(math.ceil(note["quantizedEndBeat"] / meter))
        fragment_count += end_measure - first_measure
        if fragment_count > _MAX_MUSICXML_FRAGMENTS:
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

    root = ET.Element("score-partwise", version="3.1")
    ET.SubElement(ET.SubElement(root, "identification"), "encoding").text = (
        f"PopEx {SCORE_BUILDER_VERSION} draft; review required"
    )
    part_list = ET.SubElement(root, "part-list")
    ET.SubElement(ET.SubElement(part_list, "score-part", id="P1"), "part-name").text = "Draft Pitched Events"
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

        fragments = fragments_by_measure.get(position, {})

        if not fragments:
            # A forward is only a timing spacer. It does not claim that missing
            # transcription evidence proves a musical rest.
            _append_forward(measure_el, bar_ticks)
            continue
        for voice_index, voice in enumerate(sorted(fragments)):
            if voice_index:
                backup = ET.SubElement(measure_el, "backup")
                ET.SubElement(backup, "duration").text = str(bar_ticks)
            cursor_ticks = 0
            voice_fragments = sorted(
                fragments[voice],
                key=lambda item: (
                    item["localStartBeat"],
                    item["localEndBeat"],
                    item["id"],
                ),
            )
            for fragment in voice_fragments:
                start_ticks = int(round(fragment["localStartBeat"] * _DIVISIONS))
                end_ticks = int(round(fragment["localEndBeat"] * _DIVISIONS))
                if start_ticks < cursor_ticks or end_ticks <= start_ticks:
                    raise ScoreConstructionError("Overlapping score notes share a MusicXML voice.")
                _append_forward(measure_el, start_ticks - cursor_ticks)
                _musicxml_note(
                    measure_el,
                    midi_note=fragment["midiNote"],
                    duration_ticks=end_ticks - start_ticks,
                    voice=voice,
                    tie_stop=fragment["tieStop"],
                    tie_start=fragment["tieStart"],
                )
                cursor_ticks = end_ticks
            _append_forward(measure_el, bar_ticks - cursor_ticks)

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
