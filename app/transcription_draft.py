from __future__ import annotations

import json
import math
import os
import re
import stat
from datetime import datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import unquote
from uuid import uuid4

from app.config import Settings
from app.media import MediaProcessingError, secure_job_dir

INTERPRETATION_DRAFT_RELATIVE_PATH = "interpretation/draft.json"
INTERPRETATION_DRAFT_SCHEMA_VERSION = 1
_SOURCE_PATH = "transcription/raw-events.json"
_ID = re.compile(r"[a-z][a-z0-9_-]{0,95}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")
_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_KEY = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_JOB = re.compile(r"[a-f0-9]{32}")
_SUBDIVISION = re.compile(r"[1-9][0-9]*(?:/[1-9][0-9]*)?(?:[TDtd])?")
_DRIVE = re.compile(r"^[A-Za-z]:[\\/]")
_URI = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_MACHINE = re.compile(r"(?:^|/)(?:home|users|tmp|var|etc|mnt|volumes|private|opt|usr)(?:/|$)", re.I)
_MAX_ARTIFACT_BYTES = 12 * 1024 * 1024
_MAX_COLLECTION = 200_000
_MAX_ALTERNATIVES = 16
_MAX_WARNINGS = 128
_MAX_TEXT = 1024
_MAX_WARNING = 500
_MAX_DEPTH = 6
_MAX_LIST = 256
_MAX_KEYS = 128
_MAX_INDEX = 2_147_483_647
_FORBIDDEN = {
    "path", "filepath", "absolutepath", "localpath", "machinepath", "filename",
    "filesystem", "uri", "url", "tensor", "tensors", "waveform", "waveforms",
    "audiosamples", "rawaudio", "rawsamples", "pcm", "usercorrection",
    "usercorrections", "correctionhistory", "edithistory", "musicxml", "engraving",
    "sheetmusic", "tablature", "notationglyph", "glyph",
}


class TranscriptionDraftError(RuntimeError):
    """Base error for editable transcription-draft validation and storage."""


class TranscriptionDraftValidationError(TranscriptionDraftError, ValueError):
    """Raised when a schema-1 editable transcription draft is invalid."""


def validate_transcription_draft(payload: Mapping[str, Any]) -> dict[str, Any]:
    value = _mapping(payload, "transcription draft")
    required = {
        "schemaVersion", "draftVersion", "createdAt", "sourceTranscription",
        "algorithms", "parts", "voices", "measures", "phrases", "pitchedItems",
        "percussionItems", "alternatives", "warnings",
    }
    _keys(value, required, set(), "transcription draft")
    if _integer(value["schemaVersion"], "schemaVersion") != 1:
        raise TranscriptionDraftValidationError("Unsupported draft schema version.")

    source, source_index = _source(value["sourceTranscription"])
    algorithms = _algorithms(value["algorithms"])
    parts = _parts(value["parts"], source_index)
    part_index = {item["id"]: item for item in parts}
    voices = _voices(value["voices"], source_index, part_index)
    voice_index = {item["id"]: item for item in voices}
    measures = _measures(value["measures"])
    measure_index = {item["id"]: item for item in measures}
    phrases = _phrases(value["phrases"], source_index, part_index, voice_index, measure_index)
    phrase_index = {item["id"]: item for item in phrases}
    pitched = _pitched_items(
        value["pitchedItems"], source_index, part_index, voice_index, measure_index, phrase_index
    )
    percussion = _percussion_items(
        value["percussionItems"], source_index, part_index, voice_index, measure_index, phrase_index
    )
    _part_voice_lists(parts, voice_index)
    _primary_assignments(pitched, percussion)
    pitched_index = {item["id"]: item for item in pitched}
    percussion_index = {item["id"]: item for item in percussion}
    _tie_targets(pitched, pitched_index)
    alternatives = _top_alternatives(
        value["alternatives"], source_index, part_index, voice_index, measure_index,
        phrase_index, pitched_index, percussion_index,
    )
    nested_ids = _nested_alternative_ids([*parts, *voices, *measures, *phrases, *pitched, *percussion])
    _global_ids(parts, voices, measures, phrases, pitched, percussion, alternatives, nested_ids)

    result = {
        "schemaVersion": 1,
        "draftVersion": _version(value["draftVersion"], "draftVersion"),
        "createdAt": _utc(value["createdAt"], "createdAt"),
        "sourceTranscription": source,
        "algorithms": algorithms,
        "parts": parts,
        "voices": voices,
        "measures": measures,
        "phrases": phrases,
        "pitchedItems": pitched,
        "percussionItems": percussion,
        "alternatives": alternatives,
        "warnings": _warnings(value["warnings"], "warnings", _MAX_WARNINGS),
    }
    _encoded(result)
    return result


def write_transcription_draft(job_id: str, settings: Settings, payload: Mapping[str, Any]) -> Path:
    data = _encoded(validate_transcription_draft(payload))
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=True)
    assert directory is not None
    destination = directory / "draft.json"
    _existing_destination(destination, directory)
    temporary = directory / f".draft.json.{uuid4().hex}.tmp"
    snapshot = _directory_snapshot(directory, job_dir)
    try:
        _write_file(temporary, data, directory)
        if _directory_snapshot(directory, job_dir) != snapshot:
            raise TranscriptionDraftError("Interpretation draft directory changed during publication.")
        _replace_atomic(temporary, destination)
        if _directory_snapshot(directory, job_dir) != snapshot:
            raise TranscriptionDraftError("Interpretation draft directory changed during publication.")
        _regular_file(destination, directory)
        _fsync_directory(directory)
    except TranscriptionDraftError:
        raise
    except OSError as exc:
        raise TranscriptionDraftError("Interpretation draft could not be published safely.") from exc
    finally:
        _remove_temporary(temporary, directory)
    return destination.resolve(strict=True)


def load_transcription_draft(job_id: str, settings: Settings) -> dict[str, Any] | None:
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=False)
    if directory is None:
        return None
    destination = directory / "draft.json"
    try:
        destination.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise TranscriptionDraftError("Saved interpretation draft is unavailable.") from exc
    data = _read_file(destination, directory)
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_reject_constant)
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise TranscriptionDraftError("Saved interpretation draft is unreadable or corrupted.") from exc
    try:
        return validate_transcription_draft(payload)
    except TranscriptionDraftValidationError as exc:
        raise TranscriptionDraftError("Saved interpretation draft failed schema validation.") from exc


def _source(value: Any) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    source = _mapping(value, "sourceTranscription")
    required = {"fileName", "schemaVersion", "transcriptionVersion", "provenance", "sourceEventIndex"}
    _keys(source, required, set(), "sourceTranscription")
    if _integer(source["schemaVersion"], "sourceTranscription.schemaVersion") != 1:
        raise TranscriptionDraftValidationError("sourceTranscription.schemaVersion must be 1.")
    provenance = _mapping(source["provenance"], "sourceTranscription.provenance")
    if not provenance:
        raise TranscriptionDraftValidationError("sourceTranscription.provenance must not be empty.")
    events = _sequence(source["sourceEventIndex"], "sourceTranscription.sourceEventIndex")
    if len(events) > _MAX_COLLECTION:
        raise TranscriptionDraftValidationError("Too many source events.")
    normalized = [_source_event(item, index) for index, item in enumerate(events)]
    normalized.sort(key=lambda item: (item["rawStartSeconds"], item["id"]))
    index: dict[str, dict[str, Any]] = {}
    for event in normalized:
        if event["id"] in index:
            raise TranscriptionDraftValidationError("Duplicate source raw event ID.")
        index[event["id"]] = event
    return {
        "fileName": _relative_path(source["fileName"], _SOURCE_PATH, "sourceTranscription.fileName"),
        "schemaVersion": 1,
        "transcriptionVersion": _version(source["transcriptionVersion"], "sourceTranscription.transcriptionVersion"),
        "provenance": _safe_mapping(provenance, "sourceTranscription.provenance"),
        "sourceEventIndex": normalized,
    }, index


def _source_event(value: Any, index: int) -> dict[str, Any]:
    label = f"sourceTranscription.sourceEventIndex[{index}]"
    event = _mapping(value, label)
    required = {"id", "eventType", "sourceKind", "rawStartSeconds", "rawEndSeconds", "confidence"}
    _keys_open(event, required, label)
    event_type = _choice(event["eventType"], {"pitched", "percussion"}, f"{label}.eventType")
    start = _number(event["rawStartSeconds"], f"{label}.rawStartSeconds", minimum=0)
    end = _number(event["rawEndSeconds"], f"{label}.rawEndSeconds", minimum=0)
    if end < start or (event_type == "pitched" and end == start):
        raise TranscriptionDraftValidationError(f"{label} has an invalid raw range.")
    output = {
        "id": _id(event["id"], f"{label}.id"),
        "eventType": event_type,
        "sourceKind": _slug(event["sourceKind"], f"{label}.sourceKind"),
        "rawStartSeconds": start,
        "rawEndSeconds": end,
        "confidence": _confidence(event["confidence"], f"{label}.confidence"),
    }
    _extras(event, output, required, label)
    return output


def _algorithms(value: Any) -> dict[str, Any]:
    records = _mapping(value, "algorithms")
    if not records or len(records) > _MAX_KEYS:
        raise TranscriptionDraftValidationError("Invalid algorithm records.")
    output: dict[str, Any] = {}
    for name in sorted(records):
        if not isinstance(name, str) or not _NAME.fullmatch(name):
            raise TranscriptionDraftValidationError("Unsafe algorithm name.")
        record = _mapping(records[name], f"algorithms.{name}")
        if "version" not in record:
            raise TranscriptionDraftValidationError(f"algorithms.{name}.version is required.")
        clean = {"version": _version(record["version"], f"algorithms.{name}.version")}
        _extras(record, clean, {"version"}, f"algorithms.{name}")
        output[name] = clean
    return output


def _parts(value: Any, source: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "parts", _MAX_COLLECTION)
    output = []
    for i, raw in enumerate(items):
        label = f"parts[{i}]"; item = _mapping(raw, label)
        required = {"id", "sourceKind", "role", "instrumentKind", "voiceIds", "sourceEventIds", "confidence"}
        _keys_open(item, required, label)
        clean = {
            "id": _id(item["id"], f"{label}.id"),
            "sourceKind": _slug(item["sourceKind"], f"{label}.sourceKind"),
            "role": _slug(item["role"], f"{label}.role"),
            "instrumentKind": _slug(item["instrumentKind"], f"{label}.instrumentKind"),
            "voiceIds": _ids(item["voiceIds"], f"{label}.voiceIds"),
            "sourceEventIds": _source_refs(item["sourceEventIds"], source, f"{label}.sourceEventIds"),
            "confidence": _confidence(item["confidence"], f"{label}.confidence"),
        }
        _extras(item, clean, required, label)
        output.append(clean)
    _unique(output, "part")
    return sorted(output, key=lambda item: item["id"])


def _voices(value: Any, source: Mapping[str, Any], parts: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "voices", _MAX_COLLECTION); output = []
    for i, raw in enumerate(items):
        label = f"voices[{i}]"; item = _mapping(raw, label)
        required = {"id", "partId", "voiceKind", "sourceEventIds", "confidence"}
        _keys_open(item, required, label)
        part_id = _ref(item["partId"], parts, f"{label}.partId")
        clean = {
            "id": _id(item["id"], f"{label}.id"), "partId": part_id,
            "voiceKind": _slug(item["voiceKind"], f"{label}.voiceKind"),
            "sourceEventIds": _source_refs(item["sourceEventIds"], source, f"{label}.sourceEventIds"),
            "confidence": _confidence(item["confidence"], f"{label}.confidence"),
        }
        _extras(item, clean, required, label); output.append(clean)
    _unique(output, "voice")
    return sorted(output, key=lambda item: item["id"])


def _measures(value: Any) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "measures", _MAX_COLLECTION); output = []
    for i, raw in enumerate(items):
        label = f"measures[{i}]"; item = _mapping(raw, label)
        required = {"id", "index", "rawStartSeconds", "rawEndSeconds", "interpretedStartSeconds", "interpretedDurationSeconds", "confidence"}
        _keys_open(item, required, label)
        raw_start, raw_end = _range(item, "rawStartSeconds", "rawEndSeconds", label, allow_point=False)
        interpreted_start = _number(item["interpretedStartSeconds"], f"{label}.interpretedStartSeconds", minimum=0)
        duration = _number(item["interpretedDurationSeconds"], f"{label}.interpretedDurationSeconds", minimum=0, positive=True)
        clean = {
            "id": _id(item["id"], f"{label}.id"),
            "index": _integer_range(item["index"], f"{label}.index", 0, _MAX_INDEX),
            "rawStartSeconds": raw_start, "rawEndSeconds": raw_end,
            "interpretedStartSeconds": interpreted_start,
            "interpretedDurationSeconds": duration,
            "confidence": _confidence(item["confidence"], f"{label}.confidence"),
        }
        _paired_ints(item, clean, "meterNumerator", "meterDenominator", label, minimum=1)
        _paired_ints(item, clean, "startBeatIndex", "endBeatIndex", label, minimum=0)
        if "startBeatIndex" in clean and clean["endBeatIndex"] <= clean["startBeatIndex"]:
            raise TranscriptionDraftValidationError(f"{label} has invalid beat bounds.")
        _extras(item, clean, required | {"meterNumerator", "meterDenominator", "startBeatIndex", "endBeatIndex"}, label)
        output.append(clean)
    _unique(output, "measure"); output.sort(key=lambda item: (item["index"], item["id"]))
    for expected, item in enumerate(output):
        if item["index"] != expected:
            raise TranscriptionDraftValidationError("Measure indices must be contiguous from zero.")
        if expected:
            previous = output[expected - 1]
            if item["interpretedStartSeconds"] < previous["interpretedStartSeconds"] + previous["interpretedDurationSeconds"]:
                raise TranscriptionDraftValidationError("Measures must not overlap.")
    return output


def _phrases(value: Any, source: Mapping[str, Any], parts: Mapping[str, Any], voices: Mapping[str, Any], measures: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "phrases", _MAX_COLLECTION); output = []
    for i, raw in enumerate(items):
        label = f"phrases[{i}]"; item = _mapping(raw, label)
        required = {"id", "partId", "voiceId", "sourceEventIds", "rawStartSeconds", "rawEndSeconds", "confidence"}
        _keys_open(item, required, label)
        part_id = _ref(item["partId"], parts, f"{label}.partId")
        voice_id = _ref(item["voiceId"], voices, f"{label}.voiceId")
        if voices[voice_id]["partId"] != part_id:
            raise TranscriptionDraftValidationError(f"{label}.voiceId does not belong to partId.")
        refs = _source_refs(item["sourceEventIds"], source, f"{label}.sourceEventIds", nonempty=True)
        start, end = _range(item, "rawStartSeconds", "rawEndSeconds", label, allow_point=True)
        _source_envelope(refs, source, start, end, label)
        clean = {"id": _id(item["id"], f"{label}.id"), "partId": part_id, "voiceId": voice_id,
                 "sourceEventIds": refs, "rawStartSeconds": start, "rawEndSeconds": end,
                 "confidence": _confidence(item["confidence"], f"{label}.confidence")}
        _interpreted_pair(item, clean, label, required=False)
        if "measureIds" in item:
            clean["measureIds"] = _refs(item["measureIds"], measures, f"{label}.measureIds")
        _extras(item, clean, required | {"interpretedStartSeconds", "interpretedDurationSeconds", "measureIds"}, label)
        output.append(clean)
    _unique(output, "phrase")
    return sorted(output, key=lambda item: (item["rawStartSeconds"], item["id"]))


def _pitched_items(value: Any, source: Mapping[str, Any], parts: Mapping[str, Any], voices: Mapping[str, Any], measures: Mapping[str, Any], phrases: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "pitchedItems", _MAX_COLLECTION); output = []
    for i, raw in enumerate(items):
        label = f"pitchedItems[{i}]"; item = _mapping(raw, label)
        required = {"id", "interpretationType", "placementStatus", "partId", "voiceId", "sourceEventIds", "rawStartSeconds", "rawEndSeconds", "sourceKind", "confidence"}
        _keys_open(item, required, label)
        kind = _choice(item["interpretationType"], {"note", "rest", "unassigned"}, f"{label}.interpretationType")
        clean = _item_common(item, label, source, parts, voices, measures, phrases, "pitched")
        clean["interpretationType"] = kind
        if kind == "note" and "pitch" not in item:
            raise TranscriptionDraftValidationError(f"{label}.pitch is required for note hypotheses.")
        if kind == "rest" and ("pitch" in item or "tieCandidate" in item):
            raise TranscriptionDraftValidationError(f"{label} rest hypotheses cannot contain pitch or tie data.")
        if "pitch" in item:
            clean["pitch"] = _pitch(item["pitch"], f"{label}.pitch")
        if "tieCandidate" in item:
            clean["tieCandidate"] = _tie(item["tieCandidate"], f"{label}.tieCandidate")
        _extras(item, clean, required | {"measureId", "phraseId", "interpretedStartSeconds", "interpretedDurationSeconds", "gridPosition", "pitch", "tieCandidate", "sharedEvidence"}, label)
        output.append(clean)
    _unique(output, "pitched item")
    return sorted(output, key=lambda item: (item["rawStartSeconds"], item["id"]))


def _percussion_items(value: Any, source: Mapping[str, Any], parts: Mapping[str, Any], voices: Mapping[str, Any], measures: Mapping[str, Any], phrases: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "percussionItems", _MAX_COLLECTION); output = []
    for i, raw in enumerate(items):
        label = f"percussionItems[{i}]"; item = _mapping(raw, label)
        required = {"id", "placementStatus", "partId", "voiceIds", "sourceEventIds", "rawStartSeconds", "rawEndSeconds", "sourceKind", "hits", "confidence"}
        _keys_open(item, required, label)
        clean = _item_common(item, label, source, parts, voices, measures, phrases, "percussion", voice_field=False)
        voice_ids = _refs(item["voiceIds"], voices, f"{label}.voiceIds", nonempty=True)
        if any(voices[voice_id]["partId"] != clean["partId"] for voice_id in voice_ids):
            raise TranscriptionDraftValidationError(f"{label}.voiceIds must belong to partId.")
        hits = _bounded_sequence(item["hits"], f"{label}.hits", 64)
        if not hits:
            raise TranscriptionDraftValidationError(f"{label}.hits must not be empty.")
        normalized_hits = [_hit(hit, j, label, voices, clean["partId"]) for j, hit in enumerate(hits)]
        normalized_hits.sort(key=lambda hit: hit["sourceHitIndex"])
        if len({hit["sourceHitIndex"] for hit in normalized_hits}) != len(normalized_hits):
            raise TranscriptionDraftValidationError(f"{label}.hits have duplicate sourceHitIndex values.")
        hit_voice_ids = sorted({hit["voiceId"] for hit in normalized_hits if "voiceId" in hit})
        if sorted(voice_ids) != hit_voice_ids:
            raise TranscriptionDraftValidationError(f"{label}.voiceIds must match hit voice assignments.")
        clean["voiceIds"] = voice_ids; clean["hits"] = normalized_hits
        _extras(item, clean, required | {"measureId", "phraseId", "interpretedStartSeconds", "interpretedDurationSeconds", "gridPosition", "sharedEvidence"}, label)
        output.append(clean)
    _unique(output, "percussion item")
    return sorted(output, key=lambda item: (item["rawStartSeconds"], item["id"]))


def _item_common(item: Mapping[str, Any], label: str, source: Mapping[str, Any], parts: Mapping[str, Any], voices: Mapping[str, Any], measures: Mapping[str, Any], phrases: Mapping[str, Any], expected_type: str, *, voice_field: bool = True) -> dict[str, Any]:
    part_id = _ref(item["partId"], parts, f"{label}.partId")
    refs = _source_refs(item["sourceEventIds"], source, f"{label}.sourceEventIds", nonempty=True, expected_type=expected_type)
    start, end = _range(item, "rawStartSeconds", "rawEndSeconds", label, allow_point=True)
    _source_envelope(refs, source, start, end, label)
    status_value = _choice(item["placementStatus"], {"placed", "unassigned"}, f"{label}.placementStatus")
    clean: dict[str, Any] = {
        "id": _id(item["id"], f"{label}.id"), "placementStatus": status_value,
        "partId": part_id, "sourceEventIds": refs, "rawStartSeconds": start,
        "rawEndSeconds": end, "sourceKind": _slug(item["sourceKind"], f"{label}.sourceKind"),
        "confidence": _confidence(item["confidence"], f"{label}.confidence"),
    }
    if voice_field:
        voice_id = _ref(item["voiceId"], voices, f"{label}.voiceId")
        if voices[voice_id]["partId"] != part_id:
            raise TranscriptionDraftValidationError(f"{label}.voiceId does not belong to partId.")
        clean["voiceId"] = voice_id
    if "measureId" in item:
        clean["measureId"] = _ref(item["measureId"], measures, f"{label}.measureId")
    if "phraseId" in item:
        clean["phraseId"] = _ref(item["phraseId"], phrases, f"{label}.phraseId")
    _interpreted_pair(item, clean, label, required=status_value == "placed")
    if status_value == "placed" and "gridPosition" not in item:
        raise TranscriptionDraftValidationError(f"{label}.gridPosition is required when placed.")
    if "gridPosition" in item:
        clean["gridPosition"] = _grid(item["gridPosition"], f"{label}.gridPosition", measures, start)
        grid_measure = clean["gridPosition"].get("measureId")
        if "measureId" in clean and grid_measure != clean["measureId"]:
            raise TranscriptionDraftValidationError(f"{label}.gridPosition.measureId must match measureId.")
    if "sharedEvidence" in item:
        clean["sharedEvidence"] = _boolean(item["sharedEvidence"], f"{label}.sharedEvidence")
    return clean


def _grid(value: Any, label: str, measures: Mapping[str, Any], raw_start: float) -> dict[str, Any]:
    grid = _mapping(value, label); output: dict[str, Any] = {}
    allowed = {"measureId", "measureIndex", "beatIndex", "beatInMeasure", "subdivision", "subdivisionIndex", "alignedTimeSeconds", "offsetSeconds"}
    if set(grid) - allowed:
        raise TranscriptionDraftValidationError(f"{label} has invalid fields.")
    if "beatIndex" in grid:
        output["beatIndex"] = _integer_range(grid["beatIndex"], f"{label}.beatIndex", 0, _MAX_INDEX)
    has_measure = "measureId" in grid or "measureIndex" in grid or "beatInMeasure" in grid
    if has_measure:
        if not {"measureId", "measureIndex", "beatInMeasure"} <= set(grid):
            raise TranscriptionDraftValidationError(f"{label} has incomplete measure placement.")
        measure_id = _ref(grid["measureId"], measures, f"{label}.measureId")
        measure_index = _integer_range(grid["measureIndex"], f"{label}.measureIndex", 0, _MAX_INDEX)
        if measures[measure_id]["index"] != measure_index:
            raise TranscriptionDraftValidationError(f"{label}.measureIndex does not match measureId.")
        output.update(measureId=measure_id, measureIndex=measure_index,
                      beatInMeasure=_integer_range(grid["beatInMeasure"], f"{label}.beatInMeasure", 1, _MAX_INDEX))
    has_subdivision = "subdivision" in grid or "subdivisionIndex" in grid
    if has_subdivision:
        if not {"subdivision", "subdivisionIndex"} <= set(grid):
            raise TranscriptionDraftValidationError(f"{label} has incomplete subdivision placement.")
        subdivision = _subdivision(grid["subdivision"], f"{label}.subdivision")
        subdivision_index = _integer_range(grid["subdivisionIndex"], f"{label}.subdivisionIndex", 0, _MAX_INDEX)
        if isinstance(subdivision, int) and subdivision_index >= subdivision:
            raise TranscriptionDraftValidationError(f"{label}.subdivisionIndex must be less than subdivision.")
        output.update(subdivision=subdivision, subdivisionIndex=subdivision_index)
    has_time = "alignedTimeSeconds" in grid or "offsetSeconds" in grid
    if has_time:
        if not {"alignedTimeSeconds", "offsetSeconds"} <= set(grid):
            raise TranscriptionDraftValidationError(f"{label} has incomplete aligned timing.")
        aligned = _number(grid["alignedTimeSeconds"], f"{label}.alignedTimeSeconds", minimum=0)
        offset = _number(grid["offsetSeconds"], f"{label}.offsetSeconds")
        if not math.isclose(raw_start - aligned, offset, rel_tol=1e-9, abs_tol=1e-9):
            raise TranscriptionDraftValidationError(f"{label}.offsetSeconds must equal raw minus aligned time.")
        output.update(alignedTimeSeconds=aligned, offsetSeconds=offset)
    if not output:
        raise TranscriptionDraftValidationError(f"{label} must contain placement evidence.")
    return output


def _pitch(value: Any, label: str) -> dict[str, Any]:
    pitch = _mapping(value, label); allowed = {"midiNote", "midiPitch", "frequencyHz", "noteName"}
    if not set(pitch) or set(pitch) - allowed:
        raise TranscriptionDraftValidationError(f"{label} has invalid fields.")
    output: dict[str, Any] = {}
    if "midiNote" in pitch: output["midiNote"] = _integer_range(pitch["midiNote"], f"{label}.midiNote", 0, 127)
    if "midiPitch" in pitch: output["midiPitch"] = _number(pitch["midiPitch"], f"{label}.midiPitch")
    if "frequencyHz" in pitch: output["frequencyHz"] = _number(pitch["frequencyHz"], f"{label}.frequencyHz", minimum=0, positive=True)
    if "noteName" in pitch: output["noteName"] = _text(pitch["noteName"], f"{label}.noteName", 64)
    return output


def _tie(value: Any, label: str) -> dict[str, Any]:
    tie = _mapping(value, label); required = {"role", "confidence"}; _keys_open(tie, required, label)
    output = {"role": _slug(tie["role"], f"{label}.role"), "confidence": _confidence(tie["confidence"], f"{label}.confidence")}
    if "targetItemId" in tie: output["targetItemId"] = _id(tie["targetItemId"], f"{label}.targetItemId")
    _extras(tie, output, required | {"targetItemId"}, label)
    return output


def _hit(value: Any, index: int, parent: str, voices: Mapping[str, Any], part_id: str) -> dict[str, Any]:
    label = f"{parent}.hits[{index}]"; hit = _mapping(value, label)
    required = {"sourceHitIndex", "rawKind", "broadVoice", "confidence"}; _keys_open(hit, required, label)
    output = {
        "sourceHitIndex": _integer_range(hit["sourceHitIndex"], f"{label}.sourceHitIndex", 0, _MAX_INDEX),
        "rawKind": _slug(hit["rawKind"], f"{label}.rawKind"),
        "broadVoice": _slug(hit["broadVoice"], f"{label}.broadVoice"),
        "confidence": _confidence(hit["confidence"], f"{label}.confidence"),
    }
    if "voiceId" in hit:
        voice_id = _ref(hit["voiceId"], voices, f"{label}.voiceId")
        if voices[voice_id]["partId"] != part_id:
            raise TranscriptionDraftValidationError(f"{label}.voiceId does not belong to partId.")
        output["voiceId"] = voice_id
    _extras(hit, output, required | {"voiceId"}, label)
    return output


def _top_alternatives(value: Any, source: Mapping[str, Any], parts: Mapping[str, Any], voices: Mapping[str, Any], measures: Mapping[str, Any], phrases: Mapping[str, Any], pitched: Mapping[str, Any], percussion: Mapping[str, Any]) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, "alternatives", _MAX_COLLECTION); output = []
    subjects = {"part": parts, "voice": voices, "measure": measures, "phrase": phrases, "pitched_item": pitched, "percussion_item": percussion}
    for i, raw in enumerate(items):
        label = f"alternatives[{i}]"; item = _mapping(raw, label)
        required = {"id", "subjectType", "subjectId", "kind", "confidence"}; _keys_open(item, required, label)
        subject_type = _choice(item["subjectType"], set(subjects), f"{label}.subjectType")
        clean = {
            "id": _id(item["id"], f"{label}.id"), "subjectType": subject_type,
            "subjectId": _ref(item["subjectId"], subjects[subject_type], f"{label}.subjectId"),
            "kind": _slug(item["kind"], f"{label}.kind"),
            "confidence": _confidence(item["confidence"], f"{label}.confidence"),
        }
        if "sourceEventIds" in item:
            clean["sourceEventIds"] = _source_refs(item["sourceEventIds"], source, f"{label}.sourceEventIds")
        _extras(item, clean, required | {"sourceEventIds"}, label); output.append(clean)
    _unique(output, "top-level alternative")
    return sorted(output, key=lambda item: item["id"])


def _extras(source: Mapping[str, Any], output: dict[str, Any], known: set[str], label: str) -> None:
    for key in sorted(source):
        if key in known: continue
        if key == "warnings": output[key] = _warnings(source[key], f"{label}.warnings", _MAX_WARNINGS)
        elif key == "alternatives": output[key] = _nested_alternatives(source[key], f"{label}.alternatives")
        else:
            _metadata_key(key, label); output[key] = _safe_value(source[key], f"{label}.{key}", 1)


def _nested_alternatives(value: Any, label: str) -> list[dict[str, Any]]:
    items = _bounded_sequence(value, label, _MAX_ALTERNATIVES); output = []
    for i, raw in enumerate(items):
        item_label = f"{label}[{i}]"; item = _mapping(raw, item_label)
        required = {"id", "kind", "confidence"}; _keys_open(item, required, item_label)
        clean = {"id": _id(item["id"], f"{item_label}.id"), "kind": _slug(item["kind"], f"{item_label}.kind"), "confidence": _confidence(item["confidence"], f"{item_label}.confidence")}
        _extras(item, clean, required, item_label); output.append(clean)
    _unique(output, "alternative")
    return sorted(output, key=lambda item: item["id"])


def _safe_mapping(value: Mapping[str, Any], label: str) -> dict[str, Any]:
    if len(value) > _MAX_KEYS: raise TranscriptionDraftValidationError(f"{label} contains too many fields.")
    output = {}
    for key in sorted(value):
        _metadata_key(key, label); output[key] = _safe_value(value[key], f"{label}.{key}", 1)
    return output


def _safe_value(value: Any, label: str, depth: int) -> Any:
    if depth > _MAX_DEPTH: raise TranscriptionDraftValidationError(f"{label} is nested too deeply.")
    if value is None or isinstance(value, bool): return value
    if isinstance(value, str): return _text(value, label, _MAX_TEXT, allow_empty=True)
    if isinstance(value, int): return value
    if isinstance(value, float):
        if not math.isfinite(value): raise TranscriptionDraftValidationError(f"{label} must be finite.")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_KEYS: raise TranscriptionDraftValidationError(f"{label} contains too many fields.")
        output = {}
        for key in sorted(value):
            _metadata_key(key, label); output[key] = _safe_value(value[key], f"{label}.{key}", depth + 1)
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_LIST: raise TranscriptionDraftValidationError(f"{label} contains too many values.")
        return [_safe_value(item, f"{label}[{i}]", depth + 1) for i, item in enumerate(value)]
    raise TranscriptionDraftValidationError(f"{label} is not safe JSON metadata.")


def _part_voice_lists(parts: Sequence[Mapping[str, Any]], voices: Mapping[str, Any]) -> None:
    for part in parts:
        for voice_id in part["voiceIds"]:
            if voice_id not in voices or voices[voice_id]["partId"] != part["id"]:
                raise TranscriptionDraftValidationError("Part voiceIds must reference voices belonging to the part.")


def _primary_assignments(pitched: Sequence[Mapping[str, Any]], percussion: Sequence[Mapping[str, Any]]) -> None:
    owners: dict[str, list[Mapping[str, Any]]] = {}
    for item in [*pitched, *percussion]:
        for event_id in item["sourceEventIds"]: owners.setdefault(event_id, []).append(item)
    for event_id, items in owners.items():
        if len(items) > 1 and not all(item.get("sharedEvidence") is True for item in items):
            raise TranscriptionDraftValidationError(f"Source event {event_id!r} is assigned more than once.")


def _tie_targets(items: Sequence[Mapping[str, Any]], index: Mapping[str, Any]) -> None:
    for item in items:
        tie = item.get("tieCandidate")
        if not tie or "targetItemId" not in tie: continue
        target = tie["targetItemId"]
        if target not in index or target == item["id"] or index[target]["interpretationType"] == "rest":
            raise TranscriptionDraftValidationError("Invalid tie target.")


def _nested_alternative_ids(items: Sequence[Mapping[str, Any]]) -> list[str]:
    return [alt["id"] for item in items for alt in item.get("alternatives", [])]


def _global_ids(*collections: Any) -> None:
    ids: list[str] = []
    for collection in collections:
        if collection and isinstance(collection[0] if isinstance(collection, list) else None, str): ids.extend(collection)
        else: ids.extend(item["id"] for item in collection)
    if len(ids) != len(set(ids)): raise TranscriptionDraftValidationError("Draft IDs must be globally unique.")


def _unique(items: Sequence[Mapping[str, Any]], label: str) -> None:
    ids = [item["id"] for item in items]
    if len(ids) != len(set(ids)): raise TranscriptionDraftValidationError(f"Duplicate {label} ID.")


def _source_envelope(refs: Sequence[str], source: Mapping[str, Any], start: float, end: float, label: str) -> None:
    expected_start = min(source[event_id]["rawStartSeconds"] for event_id in refs)
    expected_end = max(source[event_id]["rawEndSeconds"] for event_id in refs)
    if start != expected_start or end != expected_end:
        raise TranscriptionDraftValidationError(f"{label} raw range must retain its source evidence envelope.")


def _interpreted_pair(item: Mapping[str, Any], output: dict[str, Any], label: str, *, required: bool) -> None:
    present = {"interpretedStartSeconds", "interpretedDurationSeconds"}.intersection(item)
    if present and len(present) != 2: raise TranscriptionDraftValidationError(f"{label} has incomplete interpreted timing.")
    if required and len(present) != 2: raise TranscriptionDraftValidationError(f"{label} requires interpreted timing when placed.")
    if present:
        output["interpretedStartSeconds"] = _number(item["interpretedStartSeconds"], f"{label}.interpretedStartSeconds", minimum=0)
        output["interpretedDurationSeconds"] = _number(item["interpretedDurationSeconds"], f"{label}.interpretedDurationSeconds", minimum=0)


def _paired_ints(item: Mapping[str, Any], output: dict[str, Any], first: str, second: str, label: str, *, minimum: int) -> None:
    present = {first, second}.intersection(item)
    if present and len(present) != 2: raise TranscriptionDraftValidationError(f"{label}.{first} and {second} must appear together.")
    if present:
        output[first] = _integer_range(item[first], f"{label}.{first}", minimum, _MAX_INDEX)
        output[second] = _integer_range(item[second], f"{label}.{second}", minimum, _MAX_INDEX)


def _range(item: Mapping[str, Any], first: str, second: str, label: str, *, allow_point: bool) -> tuple[float, float]:
    start = _number(item[first], f"{label}.{first}", minimum=0); end = _number(item[second], f"{label}.{second}", minimum=0)
    if end < start or (not allow_point and end == start): raise TranscriptionDraftValidationError(f"{label} has an invalid range.")
    return start, end


def _source_refs(value: Any, source: Mapping[str, Any], label: str, *, nonempty: bool = False, expected_type: str | None = None) -> list[str]:
    refs = _refs(value, source, label, nonempty=nonempty)
    if expected_type and any(source[event_id]["eventType"] != expected_type for event_id in refs):
        raise TranscriptionDraftValidationError(f"{label} references the wrong source event type.")
    return sorted(refs, key=lambda event_id: (source[event_id]["rawStartSeconds"], event_id))


def _refs(value: Any, index: Mapping[str, Any], label: str, *, nonempty: bool = False) -> list[str]:
    refs = _ids(value, label)
    if nonempty and not refs: raise TranscriptionDraftValidationError(f"{label} must not be empty.")
    if any(ref not in index for ref in refs): raise TranscriptionDraftValidationError(f"{label} contains an unknown reference.")
    return refs


def _ref(value: Any, index: Mapping[str, Any], label: str) -> str:
    ref = _id(value, label)
    if ref not in index: raise TranscriptionDraftValidationError(f"{label} references a missing object.")
    return ref


def _ids(value: Any, label: str) -> list[str]:
    items = _sequence(value, label)
    if len(items) > _MAX_LIST: raise TranscriptionDraftValidationError(f"{label} contains too many IDs.")
    output = [_id(item, f"{label}[{i}]") for i, item in enumerate(items)]
    if len(output) != len(set(output)): raise TranscriptionDraftValidationError(f"{label} contains duplicate IDs.")
    return sorted(output)


def _warnings(value: Any, label: str, maximum: int) -> list[str]:
    items = _sequence(value, label)
    if len(items) > maximum: raise TranscriptionDraftValidationError(f"{label} contains too many warnings.")
    return [_text(item, f"{label}[{i}]", _MAX_WARNING) for i, item in enumerate(items)]


def _metadata_key(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _KEY.fullmatch(value): raise TranscriptionDraftValidationError(f"{label} has an unsafe field name.")
    if re.sub(r"[^a-z0-9]", "", value.lower()) in _FORBIDDEN: raise TranscriptionDraftValidationError(f"{label} contains prohibited data.")
    return value


def _relative_path(value: Any, expected: str, label: str) -> str:
    text = _text(value, label, 256)
    if unquote(text) != text or "\\" in text: raise TranscriptionDraftValidationError(f"{label} is not a safe relative path.")
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != expected: raise TranscriptionDraftValidationError(f"{label} is not the canonical artifact path.")
    return expected


def _text(value: Any, label: str, maximum: int, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str): raise TranscriptionDraftValidationError(f"{label} must be a string.")
    if value != value.strip() or (not value and not allow_empty) or len(value) > maximum: raise TranscriptionDraftValidationError(f"{label} has invalid text.")
    if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in value): raise TranscriptionDraftValidationError(f"{label} contains control characters.")
    _unsafe_path_text(value, label); return value


def _unsafe_path_text(value: str, label: str) -> None:
    decoded = value
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded: break
        decoded = next_value
    for candidate in (value, decoded):
        if "\x00" in candidate: raise TranscriptionDraftValidationError(f"{label} contains NUL.")
        normalized = candidate.replace("\\", "/")
        parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
        if normalized.startswith("/") or _DRIVE.match(candidate) or _URI.match(candidate) or candidate.lower().startswith("file:") or ".." in parts or _MACHINE.search(normalized):
            raise TranscriptionDraftValidationError(f"{label} contains an unsafe path or URI.")


def _utc(value: Any, label: str) -> str:
    text = _text(value, label, 128)
    try: parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc: raise TranscriptionDraftValidationError(f"{label} must be an ISO timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0): raise TranscriptionDraftValidationError(f"{label} must be UTC.")
    return text


def _number(value: Any, label: str, *, minimum: float | None = None, maximum: float | None = None, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)): raise TranscriptionDraftValidationError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number): raise TranscriptionDraftValidationError(f"{label} must be finite.")
    if minimum is not None and number < minimum: raise TranscriptionDraftValidationError(f"{label} is below its minimum.")
    if positive and number <= 0: raise TranscriptionDraftValidationError(f"{label} must be positive.")
    if maximum is not None and number > maximum: raise TranscriptionDraftValidationError(f"{label} is above its maximum.")
    return number


def _confidence(value: Any, label: str) -> float: return _number(value, label, minimum=0, maximum=1)
def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int): raise TranscriptionDraftValidationError(f"{label} must be an integer.")
    return value

def _integer_range(value: Any, label: str, minimum: int, maximum: int) -> int:
    number = _integer(value, label)
    if not minimum <= number <= maximum: raise TranscriptionDraftValidationError(f"{label} is out of range.")
    return number

def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool): raise TranscriptionDraftValidationError(f"{label} must be a Boolean.")
    return value

def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value): raise TranscriptionDraftValidationError(f"{label} is not a safe ID.")
    return value

def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value): raise TranscriptionDraftValidationError(f"{label} is not a safe slug.")
    return value

def _version(value: Any, label: str) -> str: return _text(value, label, 256)
def _choice(value: Any, allowed: set[str], label: str) -> str:
    if not isinstance(value, str) or value not in allowed: raise TranscriptionDraftValidationError(f"{label} has an unsupported value.")
    return value

def _subdivision(value: Any, label: str) -> int | str:
    if isinstance(value, bool): raise TranscriptionDraftValidationError(f"{label} is invalid.")
    if isinstance(value, int) and value > 0: return value
    if isinstance(value, str) and _SUBDIVISION.fullmatch(value): return value
    raise TranscriptionDraftValidationError(f"{label} is invalid.")
def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value): raise TranscriptionDraftValidationError(f"{label} must be an object with string keys.")
    return value
def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)): raise TranscriptionDraftValidationError(f"{label} must be an array.")
    return list(value)
def _bounded_sequence(value: Any, label: str, maximum: int) -> list[Any]:
    items = _sequence(value, label)
    if len(items) > maximum: raise TranscriptionDraftValidationError(f"{label} is too large.")
    return items
def _keys(value: Mapping[str, Any], required: set[str], optional: set[str], label: str) -> None:
    if required - set(value) or set(value) - required - optional: raise TranscriptionDraftValidationError(f"{label} has invalid fields.")
def _keys_open(value: Mapping[str, Any], required: set[str], label: str) -> None:
    if required - set(value): raise TranscriptionDraftValidationError(f"{label} is missing required fields.")

def _encoded(payload: Mapping[str, Any]) -> bytes:
    try: data = (json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError) as exc: raise TranscriptionDraftValidationError("Draft is not valid JSON data.") from exc
    if len(data) > _MAX_ARTIFACT_BYTES: raise TranscriptionDraftValidationError("Interpretation draft is too large.")
    return data

def _reject_constant(value: str) -> None: raise ValueError(f"Invalid JSON constant: {value}")


def _secure_job_root(job_id: str, settings: Settings) -> Path:
    if not isinstance(job_id, str) or not _JOB.fullmatch(job_id): raise TranscriptionDraftValidationError("Invalid job identifier.")
    try:
        exports = settings.exports_dir; info = exports.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise TranscriptionDraftError("Interpretation draft job directory is unsafe.")
        root = exports.resolve(strict=True); lexical = exports / job_id; job_info = lexical.lstat()
        if stat.S_ISLNK(job_info.st_mode) or not stat.S_ISDIR(job_info.st_mode): raise TranscriptionDraftError("Interpretation draft job directory is unsafe.")
        job_root = lexical.resolve(strict=True)
        if job_root.parent != root or secure_job_dir(settings, job_id).resolve(strict=True) != job_root: raise TranscriptionDraftError("Interpretation draft job directory is unsafe.")
        return job_root
    except (TranscriptionDraftError, TranscriptionDraftValidationError): raise
    except (AttributeError, MediaProcessingError, OSError, RuntimeError) as exc: raise TranscriptionDraftError("Interpretation draft job directory is unavailable.") from exc


def _artifact_directory(job_dir: Path, *, create: bool) -> Path | None:
    directory = job_dir / "interpretation"
    if create:
        try: directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc: raise TranscriptionDraftError("Interpretation draft directory could not be created safely.") from exc
    try: directory.lstat()
    except FileNotFoundError: return None
    except OSError as exc: raise TranscriptionDraftError("Interpretation draft directory is unavailable.") from exc
    _directory_snapshot(directory, job_dir); return directory


def _directory_snapshot(directory: Path, job_dir: Path) -> tuple[int, int, int]:
    try:
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode): raise TranscriptionDraftError("Interpretation draft directory is unsafe.")
        if directory.resolve(strict=True).parent != job_dir.resolve(strict=True): raise TranscriptionDraftError("Interpretation draft directory is unsafe.")
        return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)
    except TranscriptionDraftError: raise
    except OSError as exc: raise TranscriptionDraftError("Interpretation draft directory is unsafe.") from exc


def _existing_destination(path: Path, directory: Path) -> None:
    try: path.lstat()
    except FileNotFoundError: return
    except OSError as exc: raise TranscriptionDraftError("Existing interpretation draft is unavailable.") from exc
    _regular_file(path, directory)


def _write_file(path: Path, data: bytes, directory: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0); fd = None
    try:
        fd = os.open(path, flags, 0o600); offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0: raise OSError("short write")
            offset += written
        os.fsync(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode): raise TranscriptionDraftError("Temporary interpretation draft is unsafe.")
    except TranscriptionDraftError: raise
    except OSError as exc: raise TranscriptionDraftError("Temporary interpretation draft could not be written.") from exc
    finally:
        if fd is not None: os.close(fd)
    _regular_file(path, directory)


def _replace_atomic(source: Path, destination: Path) -> None:
    try: os.replace(source, destination)
    except OSError as exc: raise TranscriptionDraftError("Interpretation draft could not replace the published artifact.") from exc


def _regular_file(path: Path, directory: Path) -> os.stat_result:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or path.resolve(strict=True).parent != directory.resolve(strict=True): raise TranscriptionDraftError("Interpretation draft path is unsafe.")
        return info
    except TranscriptionDraftError: raise
    except OSError as exc: raise TranscriptionDraftError("Interpretation draft path is unsafe.") from exc


def _read_file(path: Path, directory: Path) -> bytes:
    before = _regular_file(path, directory); flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0); fd = None; after_open = None
    try:
        fd = os.open(path, flags); opened = os.fstat(fd)
        if _snapshot(opened) != _snapshot(before): raise TranscriptionDraftError("Saved interpretation draft changed during validation.")
        chunks = []; total = 0
        while True:
            chunk = os.read(fd, min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total))
            if not chunk: break
            chunks.append(chunk); total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES: raise TranscriptionDraftError("Saved interpretation draft is too large.")
        after_open = os.fstat(fd)
    except TranscriptionDraftError: raise
    except OSError as exc: raise TranscriptionDraftError("Saved interpretation draft could not be read safely.") from exc
    finally:
        if fd is not None: os.close(fd)
    after = _regular_file(path, directory)
    if after_open is None or _snapshot(before) != _snapshot(after_open) or _snapshot(before) != _snapshot(after): raise TranscriptionDraftError("Saved interpretation draft changed during validation.")
    return b"".join(chunks)


def _remove_temporary(path: Path, directory: Path) -> None:
    try: info = path.lstat()
    except (FileNotFoundError, OSError): return
    if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        try:
            if path.resolve(strict=True).parent == directory.resolve(strict=True): path.unlink()
        except OSError: pass


def _fsync_directory(path: Path) -> None:
    try: fd = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError: return
    try: os.fsync(fd)
    except OSError: pass
    finally: os.close(fd)


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]: return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns

__all__ = [
    "INTERPRETATION_DRAFT_RELATIVE_PATH", "INTERPRETATION_DRAFT_SCHEMA_VERSION",
    "TranscriptionDraftError", "TranscriptionDraftValidationError",
    "load_transcription_draft", "validate_transcription_draft", "write_transcription_draft",
]
