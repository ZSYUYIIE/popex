"""Confined, attempt-scoped storage for persisted draft-score documents.

Each score attempt publishes exactly one immutable file,
``score/score-document.<attempt>.json``, inside the job's export directory.
The database pointer switches to a new file only after a successful,
fingerprint-checked completion, so the previous successful score stays
readable while a rebuild runs or fails. Files that are neither the durable
score nor the active attempt are removed only while the caller holds the
job's database write lease.

Safety rules:
- job and attempt identifiers are validated before any path is formed;
- the job and ``score`` directories must be real directories (no symlinks)
  that stay inside the exports root and keep their identity during I/O;
- reads are bounded, regular-file-only, ``O_NOFOLLOW`` and stable across the
  read; JSON is strict (no NaN/Infinity);
- publication writes a private temporary file then hard-links it into place,
  so an existing attempt file is never overwritten with different content.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.config import Settings
from app.score_construction import PERCUSSION_NOTATION, UNRESOLVED_PERCUSSION_VOICE
from app.score_sources import (
    is_score_fingerprint,
    score_source_fingerprint,
    validate_tablature_request,
)
from app.tablature import TAB_INSTRUMENTS, TAB_STATUSES, tab_position_is_consistent

# Schema 2 adds a separate percussion part and schema 3 a tablature layer;
# schema-1 and schema-2 documents remain readable and downloadable.
SCORE_ARTIFACT_SCHEMA_VERSION = 3
SUPPORTED_SCORE_ARTIFACT_SCHEMA_VERSIONS = frozenset({1, 2, 3})
SCORE_ARTIFACT_TYPE = "popex-draft-score"
SCORE_DIRECTORY_NAME = "score"

_JOB_ID = re.compile(r"[a-f0-9]{32}")
_ATTEMPT_ID = re.compile(r"[a-f0-9]{32}")
_ARTIFACT_FILE = re.compile(r"score/score-document\.([a-f0-9]{32})\.json")
_ARTIFACT_LEAF = re.compile(r"score-document\.([a-f0-9]{32})\.json")
_TEMPORARY_LEAF = re.compile(r"\.score-document\.([a-f0-9]{32})\.[a-f0-9]{32}\.tmp")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_TOKEN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}")
_MAX_ARTIFACT_BYTES = 16 * 1024 * 1024
_MAX_MEASURES = 2048
_MAX_NOTES = 5000
_MAX_WARNINGS = 32
_MAX_TEXT = 500
_MAX_PARTS = 64
_MAX_MEASURE_HARMONY = 16
_MAX_PERCUSSION_HITS = 8192
_TOP_LEVEL_KEYS_V1 = frozenset(
    {
        "schemaVersion",
        "artifactType",
        "jobId",
        "pipelineVersion",
        "builderVersion",
        "createdAt",
        "sourceFingerprint",
        "sources",
        "layers",
        "timing",
        "parts",
        "measures",
        "counts",
        "warnings",
        "exports",
    }
)
_TOP_LEVEL_KEYS_V2 = _TOP_LEVEL_KEYS_V1 | {"percussion"}
_TOP_LEVEL_KEYS_V3 = _TOP_LEVEL_KEYS_V2 | {"tablature"}
_TAB_COUNT_KEYS = frozenset({"tabNotes", "fingeredTabNotes"})
_TAB_INSTRUMENT_KEYS = frozenset(
    {
        "instrument",
        "label",
        "sourceKind",
        "tuningName",
        "strings",
        "frets",
        "noteCount",
        "assignedCount",
        "outOfRangeCount",
        "unplayableCount",
    }
)
_COUNT_KEYS_V1 = frozenset(
    {
        "measures",
        "notes",
        "chordSymbols",
        "harmonyWindows",
        "unresolvedHarmonyWindows",
        "ambiguousHarmonyMeasures",
        "percussionEvents",
        "notesWithPart",
    }
)
_PERCUSSION_COUNT_KEYS = frozenset(
    {
        "percussionHits",
        "notatedPercussionHits",
        "collapsedPercussionHits",
        "unresolvedPercussionHits",
        "offGridPercussionHits",
        "unplacedPercussionHits",
    }
)
_COUNT_KEYS_V2 = _COUNT_KEYS_V1 | _PERCUSSION_COUNT_KEYS
_COUNT_KEYS_V3 = _COUNT_KEYS_V2 | _TAB_COUNT_KEYS
_MEASURE_KEYS_V1 = frozenset(
    {"measureIndex", "startSeconds", "endSeconds", "notes", "chordSymbol", "harmony"}
)
_MEASURE_KEYS_V2 = _MEASURE_KEYS_V1 | {"percussionHits"}
_HIT_KEYS = frozenset(
    {
        "eventId",
        "hitIndex",
        "sourceKind",
        "rawKind",
        "broadVoice",
        "resolved",
        "rawTimeSeconds",
        "rawBeat",
        "quantizedBeat",
        "quantizationShiftSeconds",
        "strength",
        "confidence",
        "interpretationPlacement",
        "notation",
    }
)
_VOICE_SOURCES = frozenset({"interpretation", "raw-hit-kinds", "none"})
_LAYER_KEYS = frozenset(
    {"pitchedNotes", "chordSymbols", "partLabels", "percussion", "tablature"}
)
_LAYER_STATUSES = frozenset({"included", "omitted"})


class ScoreArtifactError(RuntimeError):
    """A persisted score artifact could not be stored or read safely."""


class ScoreArtifactValidationError(ScoreArtifactError, ValueError):
    """A persisted score document failed structural validation."""


def score_attempt_artifact_file_name(attempt_id: str) -> str:
    if not isinstance(attempt_id, str) or not _ATTEMPT_ID.fullmatch(attempt_id):
        raise ScoreArtifactError("The score attempt identity is invalid.")
    return f"score/score-document.{attempt_id}.json"


def is_score_artifact_file_name(value: object) -> bool:
    return isinstance(value, str) and _ARTIFACT_FILE.fullmatch(value) is not None


# ---------------------------------------------------------------------------
# Validation and encoding


def _fail(message: str) -> ScoreArtifactValidationError:
    return ScoreArtifactValidationError(message)


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(f"{label} must be an object.")
    return value


def _sequence(value: Any, label: str, maximum: int) -> Sequence[Any]:
    if not isinstance(value, list):
        raise _fail(f"{label} must be an array.")
    if len(value) > maximum:
        raise _fail(f"{label} is too long.")
    return value


def _integer(value: Any, label: str, minimum: int = 0, maximum: int = 2**31) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise _fail(f"{label} must be an integer.")
    if not minimum <= value <= maximum:
        raise _fail(f"{label} is out of range.")
    return value


def _number(value: Any, label: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _fail(f"{label} must be a number.")
    if not math.isfinite(float(value)) or not minimum <= value <= maximum:
        raise _fail(f"{label} is out of range.")
    return float(value)


def _optional_number(value: Any, label: str, minimum: float, maximum: float) -> None:
    if value is not None:
        _number(value, label, minimum, maximum)


def _text(value: Any, label: str, maximum: int = _MAX_TEXT) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise _fail(f"{label} must be bounded text.")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise _fail(f"{label} contains control characters.")
    return value


def _token(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _TOKEN.fullmatch(value):
        raise _fail(f"{label} must be a safe identifier.")
    return value


def _exact_keys(value: Mapping[str, Any], expected: frozenset[str], label: str) -> None:
    if set(value) != set(expected):
        raise _fail(f"{label} has missing or unsupported fields.")


def _validate_sources(value: Any) -> None:
    sources = _mapping(value, "sources")
    base_keys = frozenset(
        {"identityVersion", "analysis", "transcription", "interpretation", "harmony"}
    )
    _exact_keys(
        sources,
        base_keys | ({"tablature"} if "tablature" in sources else set()),
        "sources",
    )
    if "tablature" in sources:
        try:
            validate_tablature_request(sources["tablature"])
        except ValueError as exc:
            raise _fail("sources.tablature is invalid.") from exc
    _integer(sources["identityVersion"], "sources.identityVersion", 1, 1)
    for name in ("analysis", "transcription"):
        layer = _mapping(sources[name], f"sources.{name}")
        for key in ("fileName", "version", "createdAt"):
            _text(layer.get(key), f"sources.{name}.{key}", 256)
    for name in ("interpretation", "harmony"):
        layer = sources[name]
        if layer is None:
            continue
        layer = _mapping(layer, f"sources.{name}")
        _exact_keys(layer, frozenset({"fileName", "version", "createdAt"}), f"sources.{name}")
        for key in ("fileName", "version", "createdAt"):
            _text(layer[key], f"sources.{name}.{key}", 256)


def _validate_layers(value: Any, schema_version: int) -> dict[str, str]:
    layers = _mapping(value, "layers")
    _exact_keys(layers, _LAYER_KEYS, "layers")
    statuses = {}
    for name in sorted(_LAYER_KEYS):
        layer = _mapping(layers[name], f"layers.{name}")
        _exact_keys(layer, frozenset({"status", "note"}), f"layers.{name}")
        if layer["status"] not in _LAYER_STATUSES:
            raise _fail(f"layers.{name}.status is invalid.")
        _text(layer["note"], f"layers.{name}.note")
        statuses[name] = layer["status"]
    if statuses["pitchedNotes"] != "included" or (
        schema_version < 3 and statuses["tablature"] != "omitted"
    ):
        raise _fail("layers do not match the supported score package.")
    return statuses


def _validate_timing(value: Any) -> tuple[float, int]:
    timing = _mapping(value, "timing")
    _exact_keys(
        timing,
        frozenset(
            {
                "tempoBpm",
                "beatsPerMeasure",
                "divisions",
                "meterSource",
                "tempoConfidence",
                "tempoStable",
                "meterConfidence",
            }
        ),
        "timing",
    )
    tempo = _number(timing["tempoBpm"], "timing.tempoBpm", 20.0, 300.0)
    meter = _integer(timing["beatsPerMeasure"], "timing.beatsPerMeasure", 1, 12)
    _integer(timing["divisions"], "timing.divisions", 1, 9600)
    if timing["meterSource"] not in {"analysis", "fallback-4/4"}:
        raise _fail("timing.meterSource is invalid.")
    _optional_number(timing["tempoConfidence"], "timing.tempoConfidence", 0.0, 1.0)
    _optional_number(timing["meterConfidence"], "timing.meterConfidence", 0.0, 1.0)
    if timing["tempoStable"] is not None and type(timing["tempoStable"]) is not bool:
        raise _fail("timing.tempoStable is invalid.")
    return tempo, meter


def _validate_parts(value: Any) -> set[str]:
    parts = _sequence(value, "parts", _MAX_PARTS)
    ids: set[str] = set()
    for index, raw in enumerate(parts):
        part = _mapping(raw, f"parts[{index}]")
        _exact_keys(
            part,
            frozenset({"id", "role", "instrumentKind", "sourceKind", "noteCount"}),
            f"parts[{index}]",
        )
        part_id = _token(part["id"], f"parts[{index}].id")
        if part_id in ids:
            raise _fail("parts contains a duplicate ID.")
        ids.add(part_id)
        for key in ("role", "instrumentKind", "sourceKind"):
            _token(part[key], f"parts[{index}].{key}")
        _integer(part["noteCount"], f"parts[{index}].noteCount", 0, _MAX_NOTES)
    return ids


def _validate_percussion_hit(
    value: Any,
    label: str,
    position: int,
    meter: int,
    seen: set[tuple[str, int]],
    slots: set[tuple[str, float]],
    totals: dict[str, Any],
) -> None:
    hit = _mapping(value, label)
    _exact_keys(hit, _HIT_KEYS, label)
    identity = (
        _text(hit["eventId"], f"{label}.eventId", 256),
        _integer(hit["hitIndex"], f"{label}.hitIndex", 0, 63),
    )
    if identity in seen:
        raise _fail("measures contain a duplicate percussion hit.")
    seen.add(identity)
    for key in ("sourceKind", "rawKind"):
        _token(hit[key], f"{label}.{key}")
    voice = hit["broadVoice"]
    if voice not in PERCUSSION_NOTATION:
        raise _fail(f"{label}.broadVoice is not a supported broad voice.")
    if type(hit["resolved"]) is not bool or hit["resolved"] == (
        voice == UNRESOLVED_PERCUSSION_VOICE
    ):
        raise _fail(f"{label} must be unresolved exactly in the unresolved lane.")
    _number(hit["rawTimeSeconds"], f"{label}.rawTimeSeconds", 0.0, 1e6)
    _number(hit["rawBeat"], f"{label}.rawBeat", 0.0, 1e7)
    beat = _number(hit["quantizedBeat"], f"{label}.quantizedBeat", 0.0, 1e7)
    if int(beat // meter) != position or beat * 2 != int(beat * 2):
        raise _fail(f"{label}.quantizedBeat is outside the measure grid.")
    _number(hit["quantizationShiftSeconds"], f"{label}.quantizationShiftSeconds", 0.0, 60.0)
    _number(hit["strength"], f"{label}.strength", 0.0, 1.0)
    _number(hit["confidence"], f"{label}.confidence", 0.0, 1.0)
    if hit["interpretationPlacement"] not in {None, "placed", "unassigned"}:
        raise _fail(f"{label}.interpretationPlacement is invalid.")
    if hit["notation"] not in {"notated", "collapsed"}:
        raise _fail(f"{label}.notation is invalid.")
    if hit["notation"] == "notated":
        if (voice, beat) in slots:
            raise _fail("Two notated percussion hits share one grid slot.")
        slots.add((voice, beat))
        totals["notatedPercussionHits"] += 1
    else:
        totals["collapsedPercussionHits"] += 1
    totals["percussionHits"] += 1
    if not hit["resolved"]:
        totals["unresolvedPercussionHits"] += 1
    if hit["quantizationShiftSeconds"] > 0.050:
        totals["offGridPercussionHits"] += 1
    if hit["interpretationPlacement"] == "unassigned":
        totals["unplacedPercussionHits"] += 1
    totals["byVoice"][voice] = totals["byVoice"].get(voice, 0) + 1


def _validate_note_tab(note: Mapping[str, Any], label: str, totals: dict[str, Any]) -> None:
    if "tab" not in note:
        raise _fail(f"{label}.tab is required.")
    tab = note["tab"]
    if tab is None:
        return
    tab = _mapping(tab, f"{label}.tab")
    _exact_keys(tab, frozenset({"instrument", "status", "string", "fret"}), f"{label}.tab")
    instrument = tab["instrument"]
    if instrument not in TAB_INSTRUMENTS or tab["status"] not in TAB_STATUSES:
        raise _fail(f"{label}.tab is unsupported.")
    if tab["status"] == "assigned":
        string, fret = tab["string"], tab["fret"]
        if (
            type(string) is not int
            or type(fret) is not int
            or not tab_position_is_consistent(note["midiNote"], instrument, string, fret)
        ):
            raise _fail(f"{label}.tab does not sound the note's pitch.")
        totals["fingeredTabNotes"] += 1
    elif tab["string"] is not None or tab["fret"] is not None:
        raise _fail(f"{label}.tab cannot name a position without assignment.")
    totals["tabNotes"] += 1
    counts = totals["tabByInstrument"].setdefault(
        instrument,
        {"noteCount": 0, "assignedCount": 0, "outOfRangeCount": 0, "unplayableCount": 0, "sources": set()},
    )
    counts["noteCount"] += 1
    key = {"assigned": "assignedCount", "out_of_range": "outOfRangeCount", "unplayable": "unplayableCount"}[
        tab["status"]
    ]
    counts[key] += 1
    counts["sources"].add(note.get("sourceKind"))


def _validate_tablature_summary(
    value: Any, sources: Mapping[str, Any], by_instrument: Mapping[str, Any]
) -> None:
    summary = _mapping(value, "tablature")
    _exact_keys(summary, frozenset({"version", "origin", "request", "instruments"}), "tablature")
    if not isinstance(summary["version"], str) or not _VERSION.fullmatch(summary["version"]):
        raise _fail("tablature.version is invalid.")
    try:
        request = validate_tablature_request(summary["request"])
    except ValueError as exc:
        raise _fail("tablature.request is invalid.") from exc
    if summary["origin"] == "musician":
        if sources.get("tablature") != request:
            raise _fail("tablature.request does not match the recorded score inputs.")
    elif summary["origin"] != "default" or "tablature" in sources:
        raise _fail("tablature.origin does not match the recorded score inputs.")
    instruments = _sequence(summary["instruments"], "tablature.instruments", len(TAB_INSTRUMENTS))
    listed: set[str] = set()
    for index, raw in enumerate(instruments):
        label = f"tablature.instruments[{index}]"
        item = _mapping(raw, label)
        _exact_keys(item, _TAB_INSTRUMENT_KEYS, label)
        instrument_id = item["instrument"]
        instrument = TAB_INSTRUMENTS.get(instrument_id)
        if instrument is None or instrument_id in listed:
            raise _fail(f"{label}.instrument is invalid.")
        listed.add(instrument_id)
        if (
            item["sourceKind"] != request[instrument_id]
            or item["label"] != instrument.label
            or item["tuningName"] != instrument.tuning_name
            or item["strings"] != list(instrument.strings)
            or item["frets"] != instrument.frets
        ):
            raise _fail(f"{label} does not match the tablature request or tuning.")
        expected = by_instrument.get(
            instrument_id,
            {"noteCount": 0, "assignedCount": 0, "outOfRangeCount": 0, "unplayableCount": 0, "sources": set()},
        )
        for key in ("noteCount", "assignedCount", "outOfRangeCount", "unplayableCount"):
            if _integer(item[key], f"{label}.{key}", 0, _MAX_NOTES) != expected[key]:
                raise _fail(f"{label}.{key} does not match the measures.")
        if expected["sources"] - {item["sourceKind"]}:
            raise _fail(f"{label} covers notes from another line.")
    requested = {name for name, source in request.items() if source is not None}
    if listed != requested or set(by_instrument) - listed:
        raise _fail("tablature.instruments does not match the request.")


def _validate_measures(
    value: Any,
    part_ids: set[str],
    meter: int,
    schema_version: int,
) -> dict[str, Any]:
    measures = _sequence(value, "measures", _MAX_MEASURES)
    if not measures:
        raise _fail("measures must not be empty.")
    note_ids: set[str] = set()
    hit_ids: set[tuple[str, int]] = set()
    hit_slots: set[tuple[str, float]] = set()
    totals: dict[str, Any] = {
        "notes": 0,
        "chordSymbols": 0,
        "notesWithPart": 0,
        "harmonyEntries": 0,
        **{key: 0 for key in _PERCUSSION_COUNT_KEYS},
        "byVoice": {},
        "tabNotes": 0,
        "fingeredTabNotes": 0,
        "tabByInstrument": {},
    }
    for position, raw in enumerate(measures):
        label = f"measures[{position}]"
        measure = _mapping(raw, label)
        _exact_keys(
            measure,
            _MEASURE_KEYS_V2 if schema_version >= 2 else _MEASURE_KEYS_V1,
            label,
        )
        if _integer(measure["measureIndex"], f"{label}.measureIndex") != position:
            raise _fail(f"{label}.measureIndex is inconsistent.")
        start = _number(measure["startSeconds"], f"{label}.startSeconds", 0.0, 1e6)
        end = _number(measure["endSeconds"], f"{label}.endSeconds", 0.0, 1e6)
        if end <= start:
            raise _fail(f"{label} has an invalid time range.")
        if measure["chordSymbol"] is not None:
            _text(measure["chordSymbol"], f"{label}.chordSymbol", 64)
            totals["chordSymbols"] += 1
        harmony = _sequence(measure["harmony"], f"{label}.harmony", _MAX_MEASURE_HARMONY)
        for index, raw_entry in enumerate(harmony):
            entry_label = f"{label}.harmony[{index}]"
            entry = _mapping(raw_entry, entry_label)
            _exact_keys(
                entry,
                frozenset(
                    {"segmentId", "startBeat", "endBeat", "symbol", "confidence", "unresolved"}
                ),
                entry_label,
            )
            _token(entry["segmentId"], f"{entry_label}.segmentId")
            entry_start = _number(entry["startBeat"], f"{entry_label}.startBeat", 0.0, meter)
            entry_end = _number(entry["endBeat"], f"{entry_label}.endBeat", 0.0, meter)
            if entry_end <= entry_start:
                raise _fail(f"{entry_label} has an invalid beat range.")
            if type(entry["unresolved"]) is not bool:
                raise _fail(f"{entry_label}.unresolved is invalid.")
            if entry["unresolved"]:
                if entry["symbol"] is not None or entry["confidence"] is not None:
                    raise _fail(f"{entry_label} must not name a chord when unresolved.")
            else:
                _text(entry["symbol"], f"{entry_label}.symbol", 64)
                _number(entry["confidence"], f"{entry_label}.confidence", 0.0, 1.0)
        totals["harmonyEntries"] += len(harmony)
        notes = _sequence(measure["notes"], f"{label}.notes", _MAX_NOTES)
        for index, raw_note in enumerate(notes):
            note_label = f"{label}.notes[{index}]"
            note = _mapping(raw_note, note_label)
            note_id = _text(note.get("id"), f"{note_label}.id", 256)
            if note_id in note_ids:
                raise _fail("measures contain a duplicate note ID.")
            note_ids.add(note_id)
            _integer(note.get("midiNote"), f"{note_label}.midiNote", 0, 127)
            _number(note.get("confidence"), f"{note_label}.confidence", 0.0, 1.0)
            if "partId" not in note:
                raise _fail(f"{note_label}.partId is required.")
            if schema_version >= 3:
                _validate_note_tab(note, note_label, totals)
            if note["partId"] is not None:
                if note["partId"] not in part_ids:
                    raise _fail(f"{note_label}.partId references an unknown part.")
                totals["notesWithPart"] += 1
        totals["notes"] += len(notes)
        if totals["notes"] > _MAX_NOTES:
            raise _fail("measures contain too many notes.")
        if schema_version >= 2:
            hits = _sequence(
                measure["percussionHits"], f"{label}.percussionHits", _MAX_PERCUSSION_HITS
            )
            for index, raw_hit in enumerate(hits):
                _validate_percussion_hit(
                    raw_hit,
                    f"{label}.percussionHits[{index}]",
                    position,
                    meter,
                    hit_ids,
                    hit_slots,
                    totals,
                )
            if totals["percussionHits"] > _MAX_PERCUSSION_HITS:
                raise _fail("measures contain too many percussion hits.")
    totals["measures"] = len(measures)
    return totals


def _validate_percussion_summary(value: Any, by_voice: Mapping[str, int]) -> None:
    summary = _mapping(value, "percussion")
    _exact_keys(summary, frozenset({"voiceSource", "voices"}), "percussion")
    if summary["voiceSource"] not in _VOICE_SOURCES:
        raise _fail("percussion.voiceSource is invalid.")
    if (summary["voiceSource"] == "none") != (not by_voice):
        raise _fail("percussion.voiceSource does not match the hits.")
    voices = _sequence(summary["voices"], "percussion.voices", len(PERCUSSION_NOTATION))
    listed: dict[str, int] = {}
    for index, raw in enumerate(voices):
        label = f"percussion.voices[{index}]"
        voice = _mapping(raw, label)
        _exact_keys(
            voice,
            frozenset(
                {
                    "broadVoice",
                    "label",
                    "displayStep",
                    "displayOctave",
                    "notehead",
                    "gmNote",
                    "hitCount",
                }
            ),
            label,
        )
        name = voice["broadVoice"]
        if name not in PERCUSSION_NOTATION or name in listed:
            raise _fail(f"{label}.broadVoice is invalid.")
        step, octave, notehead, gm_note, voice_label = PERCUSSION_NOTATION[name]
        if (
            voice["label"],
            voice["displayStep"],
            voice["displayOctave"],
            voice["notehead"],
            voice["gmNote"],
        ) != (voice_label, step, octave, notehead, gm_note):
            raise _fail(f"{label} does not match the documented notation table.")
        listed[name] = _integer(voice["hitCount"], f"{label}.hitCount", 1, _MAX_PERCUSSION_HITS)
    if listed != dict(by_voice):
        raise _fail("percussion.voices does not match the measures.")


def validate_score_artifact(payload: Any) -> dict[str, Any]:
    """Validate one persisted score document and return a detached copy."""
    document = _mapping(payload, "score document")
    schema_version = document.get("schemaVersion")
    if (
        type(schema_version) is not int
        or schema_version not in SUPPORTED_SCORE_ARTIFACT_SCHEMA_VERSIONS
    ):
        raise _fail("Unsupported score document schema version.")
    _exact_keys(
        document,
        {1: _TOP_LEVEL_KEYS_V1, 2: _TOP_LEVEL_KEYS_V2, 3: _TOP_LEVEL_KEYS_V3}[schema_version],
        "score document",
    )
    if document["artifactType"] != SCORE_ARTIFACT_TYPE:
        raise _fail("Unsupported score document type.")
    if not isinstance(document["jobId"], str) or not _JOB_ID.fullmatch(document["jobId"]):
        raise _fail("jobId is invalid.")
    for key in ("pipelineVersion", "builderVersion"):
        if not isinstance(document[key], str) or not _VERSION.fullmatch(document[key]):
            raise _fail(f"{key} is invalid.")
    _text(document["createdAt"], "createdAt", 128)
    fingerprint = document["sourceFingerprint"]
    if not is_score_fingerprint(fingerprint):
        raise _fail("sourceFingerprint is invalid.")
    _validate_sources(document["sources"])
    if score_source_fingerprint(document["sources"]) != fingerprint:
        raise _fail("sourceFingerprint does not match the recorded sources.")
    layers = _validate_layers(document["layers"], schema_version)
    _tempo, meter = _validate_timing(document["timing"])
    part_ids = _validate_parts(document["parts"])
    totals = _validate_measures(document["measures"], part_ids, meter, schema_version)
    counts = _mapping(document["counts"], "counts")
    count_keys = {1: _COUNT_KEYS_V1, 2: _COUNT_KEYS_V2, 3: _COUNT_KEYS_V3}[schema_version]
    _exact_keys(counts, count_keys, "counts")
    for key in sorted(count_keys):
        _integer(counts[key], f"counts.{key}", 0, 1_000_000)
    checked = ["measures", "notes", "chordSymbols", "notesWithPart"]
    if schema_version >= 2:
        checked += sorted(_PERCUSSION_COUNT_KEYS)
    if schema_version >= 3:
        checked += sorted(_TAB_COUNT_KEYS)
    for key in checked:
        if counts[key] != totals[key]:
            raise _fail(f"counts.{key} does not match the measures.")
    if schema_version >= 2:
        _validate_percussion_summary(document["percussion"], totals["byVoice"])
        if (layers["percussion"] == "included") != bool(totals["percussionHits"]):
            raise _fail("The percussion layer status does not match its hits.")
    elif layers["percussion"] != "omitted":
        raise _fail("Schema-1 score documents cannot carry percussion.")
    if schema_version >= 3:
        _validate_tablature_summary(
            document["tablature"], document["sources"], totals["tabByInstrument"]
        )
        if (layers["tablature"] == "included") != bool(totals["fingeredTabNotes"]):
            raise _fail("The tablature layer status does not match its notes.")
    if counts["unresolvedHarmonyWindows"] > counts["harmonyWindows"]:
        raise _fail("counts.unresolvedHarmonyWindows is inconsistent.")
    if layers["chordSymbols"] == "omitted" and (
        counts["chordSymbols"] or totals["harmonyEntries"]
    ):
        raise _fail("An omitted chord-symbol layer cannot carry harmony.")
    if layers["partLabels"] == "omitted" and (counts["notesWithPart"] or part_ids):
        raise _fail("An omitted part-label layer cannot carry parts.")
    warnings = _sequence(document["warnings"], "warnings", _MAX_WARNINGS)
    for index, warning in enumerate(warnings):
        _text(warning, f"warnings[{index}]")
    exports = _mapping(document["exports"], "exports")
    _exact_keys(exports, frozenset({"midi", "musicxml"}), "exports")
    for key in ("midi", "musicxml"):
        if not isinstance(exports[key], str) or not _VERSION.fullmatch(exports[key]):
            raise _fail(f"exports.{key} is invalid.")
    return copy.deepcopy(dict(document))


def encode_score_artifact(payload: Mapping[str, Any]) -> bytes:
    """Return canonical UTF-8 JSON bytes for a validated score document."""
    validated = validate_score_artifact(payload)
    try:
        data = json.dumps(
            validated,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise ScoreArtifactValidationError(
            "Score document could not be serialized safely."
        ) from exc
    if len(data) > _MAX_ARTIFACT_BYTES:
        raise ScoreArtifactValidationError("Score document is too large.")
    return data


# ---------------------------------------------------------------------------
# Filesystem confinement


def _directory_identity(path: Path) -> tuple[int, int, int]:
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise ScoreArtifactError("Score storage directory is unsafe.")
    return info.st_dev, info.st_ino, info.st_mode


def _job_directory(job_id: str, settings: Settings) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise ScoreArtifactError("The score request is invalid.")
    try:
        root = settings.exports_dir.resolve(strict=True)
        job_dir = root / job_id
        _directory_identity(job_dir)
        if job_dir.resolve(strict=True).parent != root:
            raise ScoreArtifactError("Score job directory is unsafe.")
    except ScoreArtifactError:
        raise
    except OSError as exc:
        raise ScoreArtifactError("Score job directory is unavailable.") from exc
    return job_dir


def _score_directory(job_dir: Path, *, create: bool) -> Path | None:
    directory = job_dir / SCORE_DIRECTORY_NAME
    try:
        directory.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise ScoreArtifactError("Score directory could not be created safely.") from exc
    except OSError as exc:
        raise ScoreArtifactError("Score directory could not be inspected safely.") from exc
    try:
        _directory_identity(directory)
        if directory.resolve(strict=True).parent != job_dir.resolve(strict=True):
            raise ScoreArtifactError("Score directory is unsafe.")
    except ScoreArtifactError:
        raise
    except OSError as exc:
        raise ScoreArtifactError("Score directory is unsafe.") from exc
    return directory


def _snapshot(paths: Sequence[Path]) -> tuple[tuple[int, int, int], ...]:
    try:
        return tuple(_directory_identity(path) for path in paths)
    except OSError as exc:
        raise ScoreArtifactError("Score storage directory changed.") from exc


def _file_state(info: os.stat_result) -> tuple:
    state = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
    # Windows lstat and fstat may disagree on creation time for one file.
    return state if os.name == "nt" else (*state, info.st_ctime_ns)


_READ_FLAGS = (
    os.O_RDONLY
    | getattr(os, "O_NOFOLLOW", 0)
    | getattr(os, "O_NONBLOCK", 0)
    | getattr(os, "O_BINARY", 0)
)


def _read_stable_file(path: Path, directories: Sequence[Path]) -> bytes | None:
    """Read one bounded regular file whose identity is stable during the read."""
    before_directories = _snapshot(directories)
    try:
        before = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ScoreArtifactError("Saved score is unavailable.") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_ARTIFACT_BYTES:
        raise ScoreArtifactError("Saved score is unsafe or too large.")
    try:
        with os.fdopen(os.open(path, _READ_FLAGS), "rb") as reader:
            if _file_state(os.fstat(reader.fileno())) != _file_state(before):
                raise ScoreArtifactError("Saved score changed during validation.")
            data = reader.read(_MAX_ARTIFACT_BYTES + 1)
            after_read = os.fstat(reader.fileno())
        if (
            len(data) > _MAX_ARTIFACT_BYTES
            or _file_state(after_read) != _file_state(before)
            or _file_state(path.lstat()) != _file_state(before)
            or _snapshot(directories) != before_directories
        ):
            raise ScoreArtifactError("Saved score changed during validation.")
    except ScoreArtifactError:
        raise
    except OSError as exc:
        raise ScoreArtifactError("Saved score could not be read safely.") from exc
    return data


def _reject_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _decode(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ScoreArtifactValidationError("Saved score is unreadable.") from exc
    return validate_score_artifact(payload)


def load_score_artifact(
    job_id: str,
    settings: Settings,
    *,
    artifact_file_name: str,
) -> dict[str, Any] | None:
    """Load and revalidate one persisted score document; ``None`` if absent."""
    match = _ARTIFACT_FILE.fullmatch(artifact_file_name or "")
    if match is None:
        raise ScoreArtifactError("The saved score pointer is invalid.")
    job_dir = _job_directory(job_id, settings)
    directory = _score_directory(job_dir, create=False)
    if directory is None:
        return None
    data = _read_stable_file(
        directory / f"score-document.{match.group(1)}.json",
        (settings.exports_dir, job_dir, directory),
    )
    if data is None:
        return None
    document = _decode(data)
    if document["jobId"] != job_id:
        raise ScoreArtifactValidationError("Saved score belongs to a different job.")
    return document


def write_score_artifact(
    job_id: str,
    settings: Settings,
    payload: Mapping[str, Any],
    *,
    artifact_file_name: str,
) -> None:
    """Publish one immutable attempt-scoped score document atomically."""
    match = _ARTIFACT_FILE.fullmatch(artifact_file_name or "")
    if match is None:
        raise ScoreArtifactError("The score attempt pointer is invalid.")
    if payload.get("jobId") != job_id:
        raise ScoreArtifactValidationError("Score document belongs to a different job.")
    encoded = encode_score_artifact(payload)
    job_dir = _job_directory(job_id, settings)
    directory = _score_directory(job_dir, create=True)
    assert directory is not None
    directories = (settings.exports_dir, job_dir, directory)
    before = _snapshot(directories)
    attempt_id = match.group(1)
    destination = directory / f"score-document.{attempt_id}.json"
    temporary = directory / f".score-document.{attempt_id}.{uuid4().hex}.tmp"
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    try:
        descriptor = os.open(temporary, flags, 0o600)
        with os.fdopen(descriptor, "wb") as writer:
            writer.write(encoded)
            writer.flush()
            os.fsync(writer.fileno())
        if _snapshot(directories) != before:
            raise ScoreArtifactError("Score storage changed during publication.")
        try:
            os.link(temporary, destination)
        except FileExistsError:
            existing = _read_stable_file(destination, directories)
            if existing != encoded:
                raise ScoreArtifactError(
                    "A different score is already saved for this attempt."
                ) from None
        except (NotImplementedError, PermissionError):
            # Filesystems without hard links: the attempt-scoped name is unique
            # per claim, so only a duplicate worker of this attempt can race.
            if os.path.lexists(destination):
                raise ScoreArtifactError(
                    "A score is already saved for this attempt."
                ) from None
            os.replace(temporary, destination)
        if _snapshot(directories) != before:
            raise ScoreArtifactError("Score storage changed during publication.")
        if _read_stable_file(destination, directories) != encoded:
            raise ScoreArtifactError("Saved score could not be verified.")
        _fsync_directory(directory)
    except ScoreArtifactError:
        raise
    except OSError as exc:
        raise ScoreArtifactError("Score could not be saved safely.") from exc
    finally:
        try:
            info = temporary.lstat()
        except OSError:
            pass
        else:
            if stat.S_ISREG(info.st_mode):
                try:
                    temporary.unlink()
                except OSError:
                    pass


def _fsync_directory(directory: Path) -> None:
    if os.name != "posix":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def remove_unowned_score_artifacts(
    job_id: str,
    settings: Settings,
    lease: Callable[[], AbstractContextManager[tuple[str | None, str | None]]],
) -> int:
    """Remove score files that are neither durable nor the active attempt.

    ``lease`` must hold the job's database write lock and yield the durable
    pointer and active attempt ID, so ownership cannot change mid-cleanup.
    Returns the number of entries removed. Directories and unexpected names
    are left untouched.
    """
    job_dir = _job_directory(job_id, settings)
    directory = _score_directory(job_dir, create=False)
    if directory is None:
        return 0
    removed = 0
    with lease() as (durable, active):
        durable_leaf = durable.split("/", 1)[1] if is_score_artifact_file_name(durable) else None
        identity = _directory_identity(directory)
        use_descriptor = (
            os.name == "posix"
            and os.unlink in os.supports_dir_fd
            and os.scandir in os.supports_fd
        )
        descriptor = None
        if use_descriptor:
            flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(directory, flags)
        try:
            if descriptor is not None:
                opened = os.fstat(descriptor)
                if (opened.st_dev, opened.st_ino, opened.st_mode) != identity:
                    raise ScoreArtifactError("Score directory changed during cleanup.")
                names = [entry.name for entry in os.scandir(descriptor)]
            else:
                names = [entry.name for entry in os.scandir(directory)]
            for name in sorted(names):
                artifact = _ARTIFACT_LEAF.fullmatch(name)
                temporary = _TEMPORARY_LEAF.fullmatch(name)
                owner = artifact or temporary
                if owner is None:
                    continue
                if name == durable_leaf or owner.group(1) == active:
                    continue
                try:
                    if descriptor is not None:
                        info = os.stat(name, dir_fd=descriptor, follow_symlinks=False)
                        if stat.S_ISDIR(info.st_mode):
                            continue
                        os.unlink(name, dir_fd=descriptor)
                    else:
                        if _directory_identity(directory) != identity:
                            raise ScoreArtifactError(
                                "Score directory changed during cleanup."
                            )
                        path = directory / name
                        if stat.S_ISDIR(path.lstat().st_mode):
                            continue
                        path.unlink()
                except FileNotFoundError:
                    continue
                removed += 1
        except OSError as exc:
            raise ScoreArtifactError("Score cleanup could not complete safely.") from exc
        finally:
            if descriptor is not None:
                os.close(descriptor)
    return removed


__all__ = [
    "SCORE_ARTIFACT_SCHEMA_VERSION",
    "SCORE_ARTIFACT_TYPE",
    "ScoreArtifactError",
    "ScoreArtifactValidationError",
    "encode_score_artifact",
    "is_score_artifact_file_name",
    "load_score_artifact",
    "remove_unowned_score_artifacts",
    "score_attempt_artifact_file_name",
    "validate_score_artifact",
    "write_score_artifact",
]
