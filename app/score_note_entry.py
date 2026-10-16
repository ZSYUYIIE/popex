"""Explicit musician-note projections; these are not detected raw events."""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from app.score_construction import SCORE_PARTS
from app.tablature import TAB_INSTRUMENTS, positions_for

MANUAL_NOTE_VERSION = 1
MANUAL_NOTE_WARNING = "Musician-entered note; no detector evidence."
NOTE_ENTRY_PART_IDS = frozenset(SCORE_PARTS) | {"combined", "guitar-tab", "bass-tab"}
_RAW_FIELDS = ("confidence", "rawMidiPitch", "startSeconds", "endSeconds",
               "startBeat", "endBeat", "quantizationShiftSeconds")
_NOTE_KEYS = frozenset({"id", "origin", "entryVersion", "correctionId", "targetPartId",
                        "sourceKind", "sourceWarnings", "midiNote", "noteName", "partId",
                        "quantizedBeat", "quantizedEndBeat", "quantizedDurationBeats", *_RAW_FIELDS})
_ID = re.compile(r"[a-f0-9]{32}")
_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_entry_parts(document: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Targets from the immutable prediction, not the possibly emptied view."""
    if document["schemaVersion"] >= 5:
        parts = [{"id": part["id"], "name": part["name"],
                  "sourceKind": SCORE_PARTS[part["id"]][1][0]
                  if SCORE_PARTS[part["id"]][1] else "unassigned", "instrument": None}
                 for part in document["scoreParts"] if part["id"] != "musician-notes"]
        parts.append({"id": "musician-notes", "name": "Musician-entered notes",
                      "sourceKind": "unassigned", "instrument": None})
    else:
        parts = [{"id": "combined", "name": "Combined pitched part",
                  "sourceKind": "unassigned", "instrument": None}]
    for item in (document.get("tablature") or {}).get("instruments", ()):
        if item["noteCount"]:
            parts.append({"id": item["instrument"] + "-tab", "name": item["label"] + " with TAB",
                          "sourceKind": item["sourceKind"], "instrument": item["instrument"]})
    return parts


def _manual_tab(note: Mapping[str, Any], instrument: str,
                notes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    positions = positions_for(note["midiNote"], TAB_INSTRUMENTS[instrument])
    free = [(string, fret) for string, fret in positions if not any(
        other.get("tab") and other["tab"]["status"] == "assigned"
        and other["tab"]["instrument"] == instrument and other["tab"]["string"] == string
        and note["quantizedBeat"] < other["quantizedEndBeat"]
        and note["quantizedEndBeat"] > other["quantizedBeat"] for other in notes)]
    if not free:
        return {"instrument": instrument, "status": "unplayable" if positions else "out_of_range",
                "string": None, "fret": None}
    string, fret = min(free, key=lambda pair: (pair[1], pair[0]))
    return {"instrument": instrument, "status": "assigned", "string": string, "fret": fret}


def make_manual_note(operation: Mapping[str, Any], part: Mapping[str, Any],
                     document: Mapping[str, Any], notes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    meter = document["timing"]["beatsPerMeasure"]
    start = operation["measureIndex"] * meter + operation["beat"]
    midi = operation["midiNote"]
    note = {"id": "user:" + operation["id"], "origin": "musician", "entryVersion": MANUAL_NOTE_VERSION,
            "correctionId": operation["id"], "targetPartId": part["id"],
            "sourceKind": part["sourceKind"], "sourceWarnings": [MANUAL_NOTE_WARNING],
            "midiNote": midi, "noteName": _NAMES[midi % 12] + str(midi // 12 - 1), "partId": None,
            "quantizedBeat": start, "quantizedEndBeat": start + operation["durationBeats"],
            "quantizedDurationBeats": operation["durationBeats"], **{key: None for key in _RAW_FIELDS}}
    if document["schemaVersion"] >= 3:
        note["tab"] = _manual_tab(note, part["instrument"], notes) if part["instrument"] else None
    return note


def validate_manual_note(note: Mapping[str, Any], *, schema: int) -> None:
    """Validate a versioned projection only after its ID was authorized by replay."""
    keys = _NOTE_KEYS | ({"tab"} if schema >= 3 else set())
    if set(note) != keys:
        raise ValueError("The musician note has unsupported fields.")
    identity = note["correctionId"]
    if (not isinstance(identity, str) or not _ID.fullmatch(identity)
            or note["id"] != "user:" + identity or note["origin"] != "musician"
            or type(note["entryVersion"]) is not int or note["entryVersion"] != MANUAL_NOTE_VERSION):
        raise ValueError("The musician note identity is invalid.")
    if any(note[key] is not None for key in _RAW_FIELDS) or note["sourceWarnings"] != [MANUAL_NOTE_WARNING]:
        raise ValueError("A musician note must not fabricate detector evidence.")
    if note["partId"] is not None:
        raise ValueError("A musician note must not claim an inferred part label.")
    midi = note["midiNote"]
    if type(midi) is not int or not 12 <= midi <= 127 or note["noteName"] != _NAMES[midi % 12] + str(midi // 12 - 1):
        raise ValueError("The musician note pitch is invalid.")
    target = note["targetPartId"]
    if not isinstance(target, str) or target not in NOTE_ENTRY_PART_IDS:
        raise ValueError("The musician note part is invalid.")
    tab = note.get("tab")
    if target.endswith("-tab"):
        if not isinstance(tab, Mapping) or tab.get("instrument") != target[:-4]:
            raise ValueError("The musician note tablature target is inconsistent.")
    else:
        source = SCORE_PARTS.get(target, ("", (), 1))[1]
        if tab is not None or note["sourceKind"] != (source[0] if source else "unassigned"):
            raise ValueError("The musician note source is not its selected part.")
    for key in ("quantizedBeat", "quantizedEndBeat", "quantizedDurationBeats"):
        value = note[key]
        maximum = 4096 if key == "quantizedDurationBeats" else 2048 * 12
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not 0 <= value <= maximum or not math.isfinite(value)
                or value * 4 != round(value * 4)):
            raise ValueError("The musician note timing is invalid.")
    if (note["quantizedDurationBeats"] <= 0
            or note["quantizedEndBeat"] - note["quantizedBeat"] != note["quantizedDurationBeats"]):
        raise ValueError("The musician note duration is inconsistent.")
