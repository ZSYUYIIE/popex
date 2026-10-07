"""Musician corrections applied on top of a saved draft score.

Predictions and corrections stay separate. A saved score document is never
modified; corrections are an ordered operation log (stored in SQLite by
:mod:`app.db`) that is replayed onto a detached copy whenever the corrected
score is read or exported. Targets use stable identities (raw event IDs for
notes and hits, bar indexes for chord symbols), so a log survives a rebuild of
the score. An operation whose target no longer exists, or no longer makes
sense, is reported as not applicable; it is never dropped or redirected.

Operations (schema 1):

- ``set_pitch``: ``{noteId}`` → ``midiNote`` (12–127). A tabbed note is
  re-fingered at the playable position closest to its previous fret;
- ``delete_note``: ``{noteId}``;
- ``set_tab``: ``{noteId}`` → ``string`` and ``fret`` that must sound the
  note's current pitch on its tablature instrument;
- ``set_chord``: ``{measureIndex}`` → ``symbol`` (text) or ``None``;
- ``set_drum_voice``: ``{eventId, hitIndex}`` → ``broadVoice``;
- ``delete_hit``: ``{eventId, hitIndex}``;
- ``reset_all``: ignore every earlier operation (itself undoable).
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.score_construction import (
    PERCUSSION_NOTATION,
    UNRESOLVED_PERCUSSION_VOICE,
    percussion_hit_counts,
    plan_score_parts,
)
from app.tablature import TAB_INSTRUMENTS, positions_for, tab_position_is_consistent

CORRECTIONS_SCHEMA_VERSION = 1
MAX_OPERATIONS = 2000
_MAX_WARNINGS = 32
_OPERATION_ID = re.compile(r"[a-f0-9]{32}")
_EVENT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,255}")
_CHORD_TEXT = re.compile(r"[A-Za-z0-9#♯♭b/+()°ø△ ,.-]{1,32}")
_STEP_NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
OPERATION_KINDS = frozenset(
    {"set_pitch", "delete_note", "set_tab", "set_chord", "set_drum_voice", "delete_hit", "reset_all"}
)


class CorrectionError(ValueError):
    """A correction or correction log is invalid."""


def note_name(midi_note: int) -> str:
    return _STEP_NAMES[midi_note % 12] + str(midi_note // 12 - 1)


def _int(value: Any, label: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise CorrectionError(f"{label} is out of range.")
    return value


def _event_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _EVENT_ID.fullmatch(value):
        raise CorrectionError(f"{label} is invalid.")
    return value


def _target(value: Any, keys: frozenset[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        raise CorrectionError("The correction target is invalid.")
    return value


def validate_operation(value: Any) -> dict[str, Any]:
    """Return the normalized content of one operation (without id/createdAt)."""
    if not isinstance(value, Mapping):
        raise CorrectionError("A correction must be an object.")
    kind = value.get("op")
    if kind not in OPERATION_KINDS:
        raise CorrectionError("The correction type is not supported.")
    allowed = {
        "set_pitch": {"op", "target", "midiNote"},
        "delete_note": {"op", "target"},
        "set_tab": {"op", "target", "string", "fret"},
        "set_chord": {"op", "target", "symbol"},
        "set_drum_voice": {"op", "target", "broadVoice"},
        "delete_hit": {"op", "target"},
        "reset_all": {"op"},
    }[kind]
    if set(value) != allowed:
        raise CorrectionError("The correction has missing or unsupported fields.")
    operation: dict[str, Any] = {"op": kind}
    if kind in {"set_pitch", "delete_note", "set_tab"}:
        target = _target(value["target"], frozenset({"noteId"}))
        operation["target"] = {"noteId": _event_id(target["noteId"], "noteId")}
    elif kind in {"set_drum_voice", "delete_hit"}:
        target = _target(value["target"], frozenset({"eventId", "hitIndex"}))
        operation["target"] = {
            "eventId": _event_id(target["eventId"], "eventId"),
            "hitIndex": _int(target["hitIndex"], "hitIndex", 0, 63),
        }
    elif kind == "set_chord":
        target = _target(value["target"], frozenset({"measureIndex"}))
        operation["target"] = {"measureIndex": _int(target["measureIndex"], "measureIndex", 0, 2047)}
    if kind == "set_pitch":
        operation["midiNote"] = _int(value["midiNote"], "midiNote", 12, 127)
    elif kind == "set_tab":
        operation["string"] = _int(value["string"], "string", 1, 6)
        operation["fret"] = _int(value["fret"], "fret", 0, 24)
    elif kind == "set_chord":
        symbol = value["symbol"]
        if symbol is not None:
            if not isinstance(symbol, str) or not _CHORD_TEXT.fullmatch(symbol) or symbol != symbol.strip():
                raise CorrectionError("Chord symbols use letters, numbers and common chord marks only.")
        operation["symbol"] = symbol
    elif kind == "set_drum_voice":
        if value["broadVoice"] not in PERCUSSION_NOTATION:
            raise CorrectionError("The drum voice is not supported.")
        operation["broadVoice"] = value["broadVoice"]
    return operation


def new_operation(value: Any, *, now: str | None = None) -> dict[str, Any]:
    operation = validate_operation(value)
    operation["id"] = uuid4().hex
    operation["createdAt"] = now or datetime.now(timezone.utc).isoformat()
    return operation


def _stored_operation(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise CorrectionError("A stored correction is invalid.")
    content = {key: item for key, item in value.items() if key not in {"id", "createdAt"}}
    operation = validate_operation(content)
    if not isinstance(value.get("id"), str) or not _OPERATION_ID.fullmatch(value["id"]):
        raise CorrectionError("A stored correction ID is invalid.")
    created = value.get("createdAt")
    if not isinstance(created, str) or not 0 < len(created) <= 64:
        raise CorrectionError("A stored correction time is invalid.")
    operation["id"] = value["id"]
    operation["createdAt"] = created
    return operation


def empty_log() -> dict[str, Any]:
    return {"schemaVersion": CORRECTIONS_SCHEMA_VERSION, "revision": 0, "operations": [], "redo": []}


def validate_log(value: Any) -> dict[str, Any]:
    """Validate a stored correction log and return a detached copy."""
    if not isinstance(value, Mapping) or set(value) != {"schemaVersion", "revision", "operations", "redo"}:
        raise CorrectionError("The correction log is malformed.")
    if value["schemaVersion"] != CORRECTIONS_SCHEMA_VERSION or type(value["schemaVersion"]) is not int:
        raise CorrectionError("The correction log schema is unsupported.")
    revision = _int(value["revision"], "revision", 0, 10**9)
    for key in ("operations", "redo"):
        if not isinstance(value[key], list) or len(value[key]) > MAX_OPERATIONS:
            raise CorrectionError("The correction log is too long.")
    operations = [_stored_operation(item) for item in value["operations"]]
    redo = [_stored_operation(item) for item in value["redo"]]
    ids = [item["id"] for item in operations + redo]
    if len(ids) != len(set(ids)):
        raise CorrectionError("The correction log repeats an operation.")
    return {"schemaVersion": CORRECTIONS_SCHEMA_VERSION, "revision": revision, "operations": operations, "redo": redo}


def active_operations(operations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return operations after the most recent ``reset_all``."""
    start = 0
    for index, operation in enumerate(operations):
        if operation["op"] == "reset_all":
            start = index + 1
    return list(operations[start:])


def _tab_counts_template() -> dict[str, int]:
    return {"noteCount": 0, "assignedCount": 0, "outOfRangeCount": 0, "unplayableCount": 0}


def _refinger(note: dict[str, Any], measure_notes: Sequence[Mapping[str, Any]]) -> None:
    tab = note.get("tab")
    if not tab:
        return
    instrument = TAB_INSTRUMENTS[tab["instrument"]]
    busy = {
        other["tab"]["string"]
        for other in measure_notes
        if other is not note
        and other.get("tab")
        and other["tab"]["instrument"] == tab["instrument"]
        and other["tab"]["status"] == "assigned"
        and other["quantizedBeat"] == note["quantizedBeat"]
    }
    options = [option for option in positions_for(note["midiNote"], instrument) if option[0] not in busy]
    if not options:
        note["tab"] = {"instrument": tab["instrument"], "status": "out_of_range"
                       if not positions_for(note["midiNote"], instrument) else "unplayable",
                       "string": None, "fret": None}
        return
    previous = tab["fret"] if tab["status"] == "assigned" else 0
    string, fret = min(options, key=lambda item: (abs(item[1] - previous), item[1], item[0]))
    note["tab"] = {"instrument": tab["instrument"], "status": "assigned", "string": string, "fret": fret}


def apply_corrections(
    document: Mapping[str, Any],
    operations: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Replay ``operations`` onto a detached copy of a saved score document.

    Returns ``(corrected document, review report)``. The input is not changed.
    """
    corrected = copy.deepcopy(dict(document))
    measures = corrected["measures"]
    schema = corrected["schemaVersion"]
    notes: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {}
    hits: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    for measure in measures:
        for note in measure["notes"]:
            notes[note["id"]] = (measure, note)
        for hit in measure.get("percussionHits", ()):
            hits[(hit["eventId"], hit["hitIndex"])] = (measure, hit)

    report: dict[str, Any] = {
        "applied": [],
        "notApplicable": [],
        "notes": {},
        "chords": {},
        "hits": {},
        "deletedNotes": [],
        "deletedHits": [],
    }
    edited_hits: set[tuple[str, int]] = set()

    def skip(operation: Mapping[str, Any], reason: str) -> None:
        report["notApplicable"].append({"id": operation["id"], "op": operation["op"], "reason": reason})

    def remember_note(note: Mapping[str, Any]) -> None:
        report["notes"].setdefault(
            note["id"],
            {"midiNote": note["midiNote"], "noteName": note["noteName"], "tab": copy.deepcopy(note.get("tab"))},
        )

    for operation in active_operations(operations):
        kind = operation["op"]
        target = operation.get("target", {})
        if kind in {"set_pitch", "delete_note", "set_tab"}:
            located = notes.get(target["noteId"])
            if located is None:
                skip(operation, "The note is not in this score.")
                continue
            measure, note = located
            if kind == "delete_note":
                remember_note(note)
                measure["notes"].remove(note)
                del notes[note["id"]]
                report["deletedNotes"].append(
                    {
                        "id": note["id"],
                        "measureIndex": measure["measureIndex"],
                        "noteName": report["notes"][note["id"]]["noteName"],
                        "quantizedBeat": note["quantizedBeat"],
                    }
                )
            elif kind == "set_pitch":
                remember_note(note)
                note["midiNote"] = operation["midiNote"]
                note["noteName"] = note_name(operation["midiNote"])
                _refinger(note, measure["notes"])
            else:
                tab = note.get("tab")
                if not tab:
                    skip(operation, "The note is not on a tablature line.")
                    continue
                if not tab_position_is_consistent(
                    note["midiNote"], tab["instrument"], operation["string"], operation["fret"]
                ):
                    skip(operation, "That string and fret do not sound the note's pitch.")
                    continue
                remember_note(note)
                note["tab"] = {
                    "instrument": tab["instrument"],
                    "status": "assigned",
                    "string": operation["string"],
                    "fret": operation["fret"],
                }
        elif kind == "set_chord":
            index = target["measureIndex"]
            if index >= len(measures):
                skip(operation, "The bar is not in this score.")
                continue
            measure = measures[index]
            report["chords"].setdefault(str(index), measure["chordSymbol"])
            measure["chordSymbol"] = operation["symbol"]
        elif kind in {"set_drum_voice", "delete_hit"}:
            key = (target["eventId"], target["hitIndex"])
            located = hits.get(key)
            if located is None:
                skip(operation, "The drum hit is not in this score.")
                continue
            measure, hit = located
            report_key = f"{key[0]}#{key[1]}"
            report["hits"].setdefault(report_key, hit["broadVoice"])
            if kind == "delete_hit":
                measure["percussionHits"].remove(hit)
                del hits[key]
                report["deletedHits"].append(
                    {
                        "eventId": key[0],
                        "hitIndex": key[1],
                        "measureIndex": measure["measureIndex"],
                        "broadVoice": report["hits"][report_key],
                        "quantizedBeat": hit["quantizedBeat"],
                    }
                )
            else:
                hit["broadVoice"] = operation["broadVoice"]
                hit["resolved"] = operation["broadVoice"] != UNRESOLVED_PERCUSSION_VOICE
                edited_hits.add(key)
        report["applied"].append(operation["id"])

    _recompute(corrected, edited_hits, schema, bool(report["applied"]))
    for note_id in list(report["notes"]):
        if note_id in notes:
            _measure, note = notes[note_id]
            original = report["notes"][note_id]
            if original["midiNote"] == note["midiNote"] and original["tab"] == note.get("tab"):
                del report["notes"][note_id]  # edited back to the prediction
    for index, original in list(report["chords"].items()):
        if measures[int(index)]["chordSymbol"] == original:
            del report["chords"][index]
    for key, original in list(report["hits"].items()):
        event_id, hit_index = key.rsplit("#", 1)
        located = hits.get((event_id, int(hit_index)))
        if located is not None and located[1]["broadVoice"] == original:
            del report["hits"][key]
    return corrected, report


def _recompute(document: dict[str, Any], edited_hits: set[tuple[str, int]], schema: int, changed: bool) -> None:
    measures = document["measures"]
    counts = document["counts"]
    all_notes = [note for measure in measures for note in measure["notes"]]
    counts["notes"] = len(all_notes)
    counts["chordSymbols"] = sum(1 for measure in measures if measure["chordSymbol"] is not None)
    counts["notesWithPart"] = sum(1 for note in all_notes if note.get("partId") is not None)
    part_counts: dict[str, int] = {}
    for note in all_notes:
        if note.get("partId") is not None:
            part_counts[note["partId"]] = part_counts.get(note["partId"], 0) + 1
    for part in document["parts"]:
        part["noteCount"] = part_counts.get(part["id"], 0)
    if counts["chordSymbols"] and document["layers"]["chordSymbols"]["status"] == "omitted":
        document["layers"]["chordSymbols"] = {
            "status": "included",
            "note": "Chord symbols were entered by the musician.",
        }

    if schema >= 2:
        hits = [hit for measure in measures for hit in measure["percussionHits"]]
        slots: dict[tuple[str, float], list[dict[str, Any]]] = {}
        for hit in hits:
            slots.setdefault((hit["broadVoice"], hit["quantizedBeat"]), []).append(hit)
        for slot in slots.values():
            slot.sort(
                key=lambda item: (
                    (item["eventId"], item["hitIndex"]) not in edited_hits,
                    -item["confidence"],
                    item["quantizationShiftSeconds"],
                    item["eventId"],
                    item["hitIndex"],
                )
            )
            for position, hit in enumerate(slot):
                hit["notation"] = "notated" if position == 0 else "collapsed"
        summary = percussion_hit_counts(hits)
        counts.update(
            percussionHits=summary["hits"],
            notatedPercussionHits=summary["notated"],
            collapsedPercussionHits=summary["collapsed"],
            unresolvedPercussionHits=summary["unresolved"],
            offGridPercussionHits=summary["offGrid"],
            unplacedPercussionHits=summary["unplaced"],
        )
        percussion = document["percussion"]
        percussion["voices"] = [
            {
                "broadVoice": voice,
                "label": PERCUSSION_NOTATION[voice][4],
                "displayStep": PERCUSSION_NOTATION[voice][0],
                "displayOctave": PERCUSSION_NOTATION[voice][1],
                "notehead": PERCUSSION_NOTATION[voice][2],
                "gmNote": PERCUSSION_NOTATION[voice][3],
                "hitCount": summary["byVoice"][voice],
            }
            for voice in PERCUSSION_NOTATION
            if voice in summary["byVoice"]
        ]
        if not hits:
            percussion["voiceSource"] = "none"
            if document["layers"]["percussion"]["status"] == "included":
                document["layers"]["percussion"] = {
                    "status": "omitted",
                    "note": "Every drum hit was removed by corrections.",
                }

    if schema >= 3:
        per_instrument: dict[str, dict[str, int]] = {}
        for note in all_notes:
            tab = note.get("tab")
            if not tab:
                continue
            item = per_instrument.setdefault(tab["instrument"], _tab_counts_template())
            item["noteCount"] += 1
            item[{"assigned": "assignedCount", "out_of_range": "outOfRangeCount",
                  "unplayable": "unplayableCount"}[tab["status"]]] += 1
        for instrument in document["tablature"]["instruments"]:
            instrument.update(per_instrument.get(instrument["instrument"], _tab_counts_template()))
        counts["tabNotes"] = sum(item["noteCount"] for item in per_instrument.values())
        counts["fingeredTabNotes"] = sum(item["assignedCount"] for item in per_instrument.values())
        layer = document["layers"]["tablature"]
        if counts["fingeredTabNotes"] and layer["status"] == "omitted":
            document["layers"]["tablature"] = {"status": "included", "note": "Positions follow musician corrections."}
        elif not counts["fingeredTabNotes"] and layer["status"] == "included":
            document["layers"]["tablature"] = {
                "status": "omitted",
                "note": "No tablature position remains after corrections.",
            }

    if schema >= 5:
        document["scoreParts"] = plan_score_parts(
            [note for note in all_notes if note.get("tab") is None]
        )

    if changed:
        warnings = ["Musician corrections are applied; the original prediction is kept and can be downloaded."]
        warnings += [item for item in document["warnings"] if item != warnings[0]]
        document["warnings"] = warnings[:_MAX_WARNINGS]


__all__ = [
    "CORRECTIONS_SCHEMA_VERSION",
    "CorrectionError",
    "MAX_OPERATIONS",
    "OPERATION_KINDS",
    "active_operations",
    "apply_corrections",
    "empty_log",
    "new_operation",
    "note_name",
    "validate_log",
    "validate_operation",
]
