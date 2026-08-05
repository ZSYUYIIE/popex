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


RAW_TRANSCRIPTION_RELATIVE_PATH = "transcription/raw-events.json"
RAW_TRANSCRIPTION_SCHEMA_VERSION = 1

_ANALYSIS_RELATIVE_PATH = "analysis/audio-analysis.json"
_SEPARATION_RELATIVE_PATH = "stems/stem-separation.json"
_JOB_ID_PATTERN = re.compile(r"[a-f0-9]{32}")
_SAFE_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_SAFE_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_OPEN_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_METADATA_KEY_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,127}")
_SUBDIVISION_PATTERN = re.compile(r"(?:[1-9][0-9]*)(?:/[1-9][0-9]*)?(?:[TDtd])?")
_WINDOWS_DRIVE_PATTERN = re.compile(r"^[A-Za-z]:[\\/]")
_URI_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*://")
_MACHINE_COMPONENT_PATTERN = re.compile(
    r"(?:^|/)(?:home|users|tmp|var|etc|mnt|volumes|private|opt|usr)(?:/|$)",
    re.IGNORECASE,
)

_MAX_ARTIFACT_BYTES = 8 * 1024 * 1024
_MAX_EVENTS_PER_COLLECTION = 100_000
_MAX_ALIGNMENT_CANDIDATES = 200_000
_MAX_WARNINGS = 128
_MAX_CANDIDATE_WARNINGS = 8
_MAX_WARNING_LENGTH = 500
_MAX_NOTE_NAME_LENGTH = 64
_MAX_VERSION_LENGTH = 256
_MAX_ALGORITHMS = 128
_MAX_HITS_PER_EVENT = 64
_MAX_FEATURE_KEYS = 64
_MAX_METADATA_KEYS = 128
_MAX_METADATA_LIST = 128
_MAX_METADATA_STRING = 1024
_MAX_METADATA_DEPTH = 6
_MAX_INDEX = 2_147_483_647

_FORBIDDEN_METADATA_KEYS = frozenset(
    {
        "path",
        "filepath",
        "absolutepath",
        "localpath",
        "machinepath",
        "tensor",
        "tensors",
        "waveform",
        "waveforms",
        "audiosamples",
        "rawaudio",
        "rawsamples",
        "usercorrection",
        "usercorrections",
        "correction",
        "corrections",
    }
)


class RawTranscriptionError(RuntimeError):
    """Base error for raw-transcription validation and artifact operations."""


class RawTranscriptionValidationError(RawTranscriptionError, ValueError):
    """Raised when a schema-1 raw-transcription payload is invalid."""


def validate_raw_transcription(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a validated deterministic copy of a schema-1 transcription artifact."""
    value = _mapping(payload, "raw transcription")
    _require_keys(
        value,
        required={
            "schemaVersion",
            "transcriptionVersion",
            "createdAt",
            "sourceAnalysis",
            "algorithms",
            "pitchedNoteEvents",
            "percussionEvents",
            "alignmentCandidates",
            "warnings",
        },
        optional={"sourceSeparation"},
        label="raw transcription",
    )
    schema_version = _integer(value["schemaVersion"], "schemaVersion")
    if schema_version != RAW_TRANSCRIPTION_SCHEMA_VERSION:
        raise RawTranscriptionValidationError(
            "Unsupported raw-transcription schema version."
        )

    result: dict[str, Any] = {
        "schemaVersion": RAW_TRANSCRIPTION_SCHEMA_VERSION,
        "transcriptionVersion": _version(
            value["transcriptionVersion"], "transcriptionVersion"
        ),
        "createdAt": _utc_timestamp(value["createdAt"], "createdAt"),
        "sourceAnalysis": _source_analysis(value["sourceAnalysis"]),
    }
    if "sourceSeparation" in value:
        result["sourceSeparation"] = _source_separation(value["sourceSeparation"])
    result["algorithms"] = _algorithms(value["algorithms"])

    pitched = _pitched_events(value["pitchedNoteEvents"])
    percussion = _percussion_events(value["percussionEvents"])
    event_index: dict[str, tuple[str, float]] = {}
    for event in pitched:
        event_id = event["id"]
        if event_id in event_index:
            raise RawTranscriptionValidationError("Duplicate transcription event ID.")
        event_index[event_id] = ("pitched", event["startSeconds"])
    for event in percussion:
        event_id = event["id"]
        if event_id in event_index:
            raise RawTranscriptionValidationError("Duplicate transcription event ID.")
        event_index[event_id] = ("percussion", event["timeSeconds"])

    result["pitchedNoteEvents"] = pitched
    result["percussionEvents"] = percussion
    result["alignmentCandidates"] = _alignment_candidates(
        value["alignmentCandidates"], event_index=event_index
    )
    result["warnings"] = _warnings(value["warnings"], "warnings")
    _encoded_payload(result)
    return result


def write_raw_transcription(
    job_id: str,
    settings: Settings,
    payload: Mapping[str, Any],
) -> Path:
    """Validate and atomically publish the raw-transcription artifact."""
    validated = validate_raw_transcription(payload)
    encoded = _encoded_payload(validated)
    job_dir = _secure_job_root(job_id, settings)
    artifact_dir = _artifact_directory(job_dir, create=True)
    assert artifact_dir is not None
    destination = artifact_dir / "raw-events.json"
    _validate_existing_destination(destination, artifact_dir)
    temporary = artifact_dir / f".raw-events.json.{uuid4().hex}.tmp"
    directory_snapshot = _directory_snapshot(artifact_dir, job_dir)
    try:
        _write_exclusive_regular_file(temporary, encoded, artifact_dir)
        if _directory_snapshot(artifact_dir, job_dir) != directory_snapshot:
            raise RawTranscriptionError(
                "Raw transcription artifact directory changed during publication."
            )
        _replace_atomic(temporary, destination)
        _fsync_directory(artifact_dir)
    except RawTranscriptionError:
        raise
    except OSError as exc:
        raise RawTranscriptionError(
            "Raw transcription could not be published safely."
        ) from exc
    finally:
        _remove_temporary(temporary, artifact_dir)
    return destination.resolve(strict=True)


def load_raw_transcription(
    job_id: str,
    settings: Settings,
) -> dict[str, Any] | None:
    """Load and validate the currently published raw-transcription artifact."""
    job_dir = _secure_job_root(job_id, settings)
    artifact_dir = _artifact_directory(job_dir, create=False)
    if artifact_dir is None:
        return None
    destination = artifact_dir / "raw-events.json"
    try:
        destination.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RawTranscriptionError("Saved raw transcription is unavailable.") from exc
    data = _read_stable_regular_file(destination, artifact_dir)
    try:
        payload = json.loads(
            data.decode("utf-8"), parse_constant=_reject_json_constant
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RawTranscriptionError(
            "Saved raw transcription is unreadable or corrupted."
        ) from exc
    try:
        return validate_raw_transcription(payload)
    except RawTranscriptionValidationError as exc:
        raise RawTranscriptionError(
            "Saved raw transcription failed schema validation."
        ) from exc


def _source_analysis(value: Any) -> dict[str, Any]:
    source = _mapping(value, "sourceAnalysis")
    _require_keys(
        source,
        required={"fileName", "analysisVersion"},
        optional=set(),
        label="sourceAnalysis",
    )
    return {
        "fileName": _relative_path(
            source["fileName"],
            expected=_ANALYSIS_RELATIVE_PATH,
            label="sourceAnalysis.fileName",
        ),
        "analysisVersion": _version(
            source["analysisVersion"], "sourceAnalysis.analysisVersion"
        ),
    }


def _source_separation(value: Any) -> dict[str, Any]:
    source = _mapping(value, "sourceSeparation")
    _require_keys(
        source,
        required={"fileName", "separationVersion", "model"},
        optional=set(),
        label="sourceSeparation",
    )
    model = _mapping(source["model"], "sourceSeparation.model")
    if not model:
        raise RawTranscriptionValidationError(
            "sourceSeparation.model must contain provenance."
        )
    return {
        "fileName": _relative_path(
            source["fileName"],
            expected=_SEPARATION_RELATIVE_PATH,
            label="sourceSeparation.fileName",
        ),
        "separationVersion": _version(
            source["separationVersion"], "sourceSeparation.separationVersion"
        ),
        "model": _metadata_mapping(
            model, label="sourceSeparation.model", max_depth=_MAX_METADATA_DEPTH
        ),
    }


def _algorithms(value: Any) -> dict[str, Any]:
    algorithms = _mapping(value, "algorithms")
    if not algorithms:
        raise RawTranscriptionValidationError("At least one algorithm record is required.")
    if len(algorithms) > _MAX_ALGORITHMS:
        raise RawTranscriptionValidationError("Too many algorithm records.")
    normalized: dict[str, Any] = {}
    for component in sorted(algorithms):
        if not isinstance(component, str) or not _OPEN_NAME_PATTERN.fullmatch(component):
            raise RawTranscriptionValidationError("Algorithm component name is unsafe.")
        record = _mapping(algorithms[component], f"algorithms.{component}")
        if "version" not in record:
            raise RawTranscriptionValidationError(
                f"Algorithm {component!r} must declare a version."
            )
        output: dict[str, Any] = {
            "version": _version(record["version"], f"algorithms.{component}.version")
        }
        for key in sorted(record):
            if key == "version":
                continue
            _metadata_key(key, f"algorithms.{component}")
            output[key] = _metadata_value(
                record[key],
                label=f"algorithms.{component}.{key}",
                depth=1,
                max_depth=_MAX_METADATA_DEPTH,
            )
        normalized[component] = output
    return normalized


def _pitched_events(value: Any) -> list[dict[str, Any]]:
    events = _sequence(value, "pitchedNoteEvents")
    if len(events) > _MAX_EVENTS_PER_COLLECTION:
        raise RawTranscriptionValidationError("Too many pitched-note events.")
    normalized = [_pitched_event(item, index) for index, item in enumerate(events)]
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise RawTranscriptionValidationError("Duplicate pitched-note event ID.")
    normalized.sort(
        key=lambda item: (item["startSeconds"], item["endSeconds"], item["id"])
    )
    return normalized


def _pitched_event(value: Any, index: int) -> dict[str, Any]:
    label = f"pitchedNoteEvents[{index}]"
    event = _mapping(value, label)
    _require_keys(
        event,
        required={
            "id",
            "sourceKind",
            "startSeconds",
            "endSeconds",
            "midiNote",
            "midiPitch",
            "frequencyHz",
            "noteName",
            "confidence",
        },
        optional={
            "velocity",
            "warnings",
            "collection",
            "rawFeatureSummary",
            "rawFeatures",
        },
        label=label,
    )
    start = _number(event["startSeconds"], f"{label}.startSeconds", minimum=0.0)
    end = _number(event["endSeconds"], f"{label}.endSeconds", minimum=0.0)
    if not start < end:
        raise RawTranscriptionValidationError(
            f"{label} must satisfy startSeconds < endSeconds."
        )
    output: dict[str, Any] = {
        "id": _event_id(event["id"], f"{label}.id"),
        "sourceKind": _slug(event["sourceKind"], f"{label}.sourceKind"),
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": _integer_range(event["midiNote"], f"{label}.midiNote", 0, 127),
        "midiPitch": _number(event["midiPitch"], f"{label}.midiPitch"),
        "frequencyHz": _number(
            event["frequencyHz"],
            f"{label}.frequencyHz",
            minimum=0.0,
            exclusive_minimum=True,
        ),
        "noteName": _text(event["noteName"], f"{label}.noteName", _MAX_NOTE_NAME_LENGTH),
        "confidence": _number(
            event["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0
        ),
    }
    if "velocity" in event:
        output["velocity"] = _integer_range(
            event["velocity"], f"{label}.velocity", 1, 127
        )
    _optional_common_event_fields(event, output, label)
    return output


def _percussion_events(value: Any) -> list[dict[str, Any]]:
    events = _sequence(value, "percussionEvents")
    if len(events) > _MAX_EVENTS_PER_COLLECTION:
        raise RawTranscriptionValidationError("Too many percussion events.")
    normalized = [_percussion_event(item, index) for index, item in enumerate(events)]
    ids = [item["id"] for item in normalized]
    if len(ids) != len(set(ids)):
        raise RawTranscriptionValidationError("Duplicate percussion event ID.")
    normalized.sort(key=lambda item: (item["timeSeconds"], item["id"]))
    return normalized


def _percussion_event(value: Any, index: int) -> dict[str, Any]:
    label = f"percussionEvents[{index}]"
    event = _mapping(value, label)
    _require_keys(
        event,
        required={"id", "sourceKind", "timeSeconds", "strength", "hits"},
        optional={"warnings", "collection", "rawFeatureSummary", "rawFeatures"},
        label=label,
    )
    hits = _sequence(event["hits"], f"{label}.hits")
    if not hits or len(hits) > _MAX_HITS_PER_EVENT:
        raise RawTranscriptionValidationError(
            f"{label}.hits must contain between 1 and {_MAX_HITS_PER_EVENT} hits."
        )
    normalized_hits = [_hit(item, label, i) for i, item in enumerate(hits)]
    output: dict[str, Any] = {
        "id": _event_id(event["id"], f"{label}.id"),
        "sourceKind": _slug(event["sourceKind"], f"{label}.sourceKind"),
        "timeSeconds": _number(event["timeSeconds"], f"{label}.timeSeconds", minimum=0.0),
        "strength": _number(
            event["strength"], f"{label}.strength", minimum=0.0, maximum=1.0
        ),
        "hits": normalized_hits,
    }
    _optional_common_event_fields(event, output, label)
    return output


def _hit(value: Any, event_label: str, index: int) -> dict[str, Any]:
    label = f"{event_label}.hits[{index}]"
    hit = _mapping(value, label)
    _require_keys(
        hit,
        required={"kind", "confidence"},
        optional={
            "strength",
            "label",
            "warnings",
            "collection",
            "rawFeatureSummary",
            "rawFeatures",
        },
        label=label,
    )
    output: dict[str, Any] = {
        "kind": _slug(hit["kind"], f"{label}.kind"),
        "confidence": _number(
            hit["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0
        ),
    }
    if "strength" in hit:
        output["strength"] = _number(
            hit["strength"], f"{label}.strength", minimum=0.0, maximum=1.0
        )
    if "label" in hit:
        output["label"] = _text(hit["label"], f"{label}.label", 128)
    _optional_common_event_fields(hit, output, label)
    return output


def _optional_common_event_fields(
    source: Mapping[str, Any], output: dict[str, Any], label: str
) -> None:
    if "collection" in source:
        output["collection"] = _slug(source["collection"], f"{label}.collection")
    if "warnings" in source:
        output["warnings"] = _warnings(source["warnings"], f"{label}.warnings")
    if "rawFeatureSummary" in source and "rawFeatures" in source:
        raise RawTranscriptionValidationError(
            f"{label} may define only one raw feature summary."
        )
    for key in ("rawFeatureSummary", "rawFeatures"):
        if key in source:
            output[key] = _feature_summary(source[key], f"{label}.{key}")


def _alignment_candidates(
    value: Any,
    *,
    event_index: Mapping[str, tuple[str, float]],
) -> list[dict[str, Any]]:
    candidates = _sequence(value, "alignmentCandidates")
    if len(candidates) > _MAX_ALIGNMENT_CANDIDATES:
        raise RawTranscriptionValidationError("Too many alignment candidates.")
    normalized = [
        _alignment_candidate(item, index, event_index)
        for index, item in enumerate(candidates)
    ]
    normalized.sort(
        key=lambda item: (
            item["rawTimeSeconds"],
            item["eventType"],
            item["eventId"],
            item.get("beatIndex", -1),
            (
                0,
                item.get("subdivision", -1),
            )
            if isinstance(item.get("subdivision"), int)
            else (1, str(item.get("subdivision", ""))),
            item.get("subdivisionIndex", -1),
            json.dumps(item, ensure_ascii=False, sort_keys=True, allow_nan=False),
        )
    )
    return normalized


def _alignment_candidate(
    value: Any,
    index: int,
    event_index: Mapping[str, tuple[str, float]],
) -> dict[str, Any]:
    label = f"alignmentCandidates[{index}]"
    candidate = _mapping(value, label)
    _require_keys(
        candidate,
        required={"eventId", "eventType", "rawTimeSeconds"},
        optional={
            "beatIndex",
            "subdivision",
            "subdivisionIndex",
            "alignedTimeSeconds",
            "offsetSeconds",
            "confidence",
            "measureIndex",
            "beatInMeasure",
            "warnings",
            "collection",
            "eventCollection",
        },
        label=label,
    )
    event_id = _event_id(candidate["eventId"], f"{label}.eventId")
    if event_id not in event_index:
        raise RawTranscriptionValidationError(
            f"{label}.eventId does not reference a raw event."
        )
    expected_type, expected_time = event_index[event_id]
    event_type = _alignment_event_type(candidate["eventType"], f"{label}.eventType")
    if event_type != expected_type:
        raise RawTranscriptionValidationError(
            f"{label}.eventType does not match the referenced raw event collection."
        )
    raw_time = _number(
        candidate["rawTimeSeconds"], f"{label}.rawTimeSeconds", minimum=0.0
    )
    if raw_time != expected_time:
        raise RawTranscriptionValidationError(
            f"{label}.rawTimeSeconds must retain the referenced raw event time."
        )
    output: dict[str, Any] = {
        "eventId": event_id,
        "eventType": event_type,
        "rawTimeSeconds": raw_time,
    }

    if "confidence" in candidate:
        output["confidence"] = _number(
            candidate["confidence"], f"{label}.confidence", minimum=0.0, maximum=1.0
        )
    if "warnings" in candidate:
        output["warnings"] = _candidate_warnings(
            candidate["warnings"], f"{label}.warnings"
        )

    if "beatIndex" in candidate:
        output["beatIndex"] = _integer_range(
            candidate["beatIndex"], f"{label}.beatIndex", 0, _MAX_INDEX
        )
    if "subdivision" in candidate:
        output["subdivision"] = _subdivision(
            candidate["subdivision"], f"{label}.subdivision"
        )
    if "subdivisionIndex" in candidate:
        subdivision_index = _integer_range(
            candidate["subdivisionIndex"],
            f"{label}.subdivisionIndex",
            0,
            _MAX_INDEX,
        )
        subdivision = output.get("subdivision")
        if not isinstance(subdivision, int):
            raise RawTranscriptionValidationError(
                f"{label}.subdivisionIndex requires an integer subdivision."
            )
        if subdivision_index >= subdivision:
            raise RawTranscriptionValidationError(
                f"{label}.subdivisionIndex must be less than subdivision."
            )
        output["subdivisionIndex"] = subdivision_index

    has_aligned = "alignedTimeSeconds" in candidate
    has_offset = "offsetSeconds" in candidate
    if has_aligned != has_offset:
        raise RawTranscriptionValidationError(
            f"{label}.alignedTimeSeconds and offsetSeconds must appear together."
        )
    if has_aligned:
        aligned = _number(
            candidate["alignedTimeSeconds"],
            f"{label}.alignedTimeSeconds",
            minimum=0.0,
        )
        offset = _number(candidate["offsetSeconds"], f"{label}.offsetSeconds")
        if not math.isclose(raw_time - aligned, offset, rel_tol=1e-9, abs_tol=1e-9):
            raise RawTranscriptionValidationError(
                f"{label}.offsetSeconds does not match rawTimeSeconds - alignedTimeSeconds."
            )
        output["alignedTimeSeconds"] = aligned
        output["offsetSeconds"] = offset

    has_measure = "measureIndex" in candidate
    has_beat_in_measure = "beatInMeasure" in candidate
    if has_measure != has_beat_in_measure:
        raise RawTranscriptionValidationError(
            f"{label}.measureIndex and beatInMeasure must appear together."
        )
    if has_measure:
        if "beatIndex" not in output:
            raise RawTranscriptionValidationError(
                f"{label}.measure fields require beatIndex."
            )
        output["measureIndex"] = _integer_range(
            candidate["measureIndex"], f"{label}.measureIndex", 0, _MAX_INDEX
        )
        output["beatInMeasure"] = _integer_range(
            candidate["beatInMeasure"], f"{label}.beatInMeasure", 1, _MAX_INDEX
        )

    grid_fields = {"beatIndex", "subdivision", "subdivisionIndex"}
    present_grid = grid_fields.intersection(output)
    if present_grid and present_grid != grid_fields:
        raise RawTranscriptionValidationError(
            f"{label} has incomplete subdivision alignment metadata."
        )
    if has_aligned and present_grid != grid_fields:
        raise RawTranscriptionValidationError(
            f"{label}.alignedTimeSeconds requires complete grid metadata."
        )
    if present_grid == grid_fields and not has_aligned:
        raise RawTranscriptionValidationError(
            f"{label}.grid metadata requires an aligned time and offset."
        )

    for key in ("collection", "eventCollection"):
        if key in candidate:
            output[key] = _slug(candidate[key], f"{label}.{key}")
    return output


def _alignment_event_type(value: Any, label: str) -> str:
    if value not in {"pitched", "percussion"}:
        raise RawTranscriptionValidationError(
            f"{label} must be pitched or percussion."
        )
    return value


def _candidate_warnings(value: Any, label: str) -> list[str]:
    warnings = _sequence(value, label)
    if len(warnings) > _MAX_CANDIDATE_WARNINGS:
        raise RawTranscriptionValidationError(f"{label} contains too many warnings.")
    return [
        _text(item, f"{label}[{index}]", _MAX_WARNING_LENGTH)
        for index, item in enumerate(warnings)
    ]


def _feature_summary(value: Any, label: str) -> dict[str, Any]:
    summary = _mapping(value, label)
    if len(summary) > _MAX_FEATURE_KEYS:
        raise RawTranscriptionValidationError(f"{label} contains too many features.")
    return _metadata_mapping(summary, label=label, max_depth=3)


def _metadata_mapping(
    value: Mapping[str, Any],
    *,
    label: str,
    max_depth: int,
) -> dict[str, Any]:
    if len(value) > _MAX_METADATA_KEYS:
        raise RawTranscriptionValidationError(f"{label} contains too many fields.")
    output: dict[str, Any] = {}
    for key in sorted(value):
        _metadata_key(key, label)
        output[key] = _metadata_value(
            value[key], label=f"{label}.{key}", depth=1, max_depth=max_depth
        )
    return output


def _metadata_value(
    value: Any,
    *,
    label: str,
    depth: int,
    max_depth: int,
) -> Any:
    if depth > max_depth:
        raise RawTranscriptionValidationError(f"{label} is nested too deeply.")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _text(value, label, _MAX_METADATA_STRING, allow_empty=True)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise RawTranscriptionValidationError(f"{label} must be finite.")
        return value
    if isinstance(value, Mapping):
        if len(value) > _MAX_METADATA_KEYS:
            raise RawTranscriptionValidationError(f"{label} contains too many fields.")
        output: dict[str, Any] = {}
        for key in sorted(value):
            _metadata_key(key, label)
            output[key] = _metadata_value(
                value[key],
                label=f"{label}.{key}",
                depth=depth + 1,
                max_depth=max_depth,
            )
        return output
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if len(value) > _MAX_METADATA_LIST:
            raise RawTranscriptionValidationError(f"{label} contains too many values.")
        return [
            _metadata_value(
                item,
                label=f"{label}[{index}]",
                depth=depth + 1,
                max_depth=max_depth,
            )
            for index, item in enumerate(value)
        ]
    raise RawTranscriptionValidationError(f"{label} is not safe JSON metadata.")


def _metadata_key(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _METADATA_KEY_PATTERN.fullmatch(value):
        raise RawTranscriptionValidationError(f"{label} contains an unsafe field name.")
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if normalized in _FORBIDDEN_METADATA_KEYS:
        raise RawTranscriptionValidationError(
            f"{label} contains prohibited raw or machine-local data."
        )
    return value


def _warnings(value: Any, label: str) -> list[str]:
    warnings = _sequence(value, label)
    if len(warnings) > _MAX_WARNINGS:
        raise RawTranscriptionValidationError(f"{label} contains too many warnings.")
    return [
        _text(item, f"{label}[{index}]", _MAX_WARNING_LENGTH)
        for index, item in enumerate(warnings)
    ]


def _event_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID_PATTERN.fullmatch(value):
        raise RawTranscriptionValidationError(f"{label} is not a stable safe ID.")
    return value


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_SLUG_PATTERN.fullmatch(value):
        raise RawTranscriptionValidationError(f"{label} is not a safe open slug.")
    return value


def _version(value: Any, label: str) -> str:
    return _text(value, label, _MAX_VERSION_LENGTH)


def _subdivision(value: Any, label: str) -> int | str:
    if isinstance(value, bool):
        raise RawTranscriptionValidationError(f"{label} is invalid.")
    if isinstance(value, int):
        if value <= 0:
            raise RawTranscriptionValidationError(f"{label} must be positive.")
        return value
    if isinstance(value, str) and _SUBDIVISION_PATTERN.fullmatch(value):
        return value
    raise RawTranscriptionValidationError(f"{label} is invalid.")


def _text(
    value: Any,
    label: str,
    maximum_length: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise RawTranscriptionValidationError(f"{label} must be a string.")
    if value != value.strip():
        raise RawTranscriptionValidationError(
            f"{label} must not have leading or trailing whitespace."
        )
    if (not value and not allow_empty) or len(value) > maximum_length:
        raise RawTranscriptionValidationError(f"{label} has an invalid length.")
    if any(ord(character) < 32 and character not in "\t\n\r" for character in value):
        raise RawTranscriptionValidationError(f"{label} contains control characters.")
    _reject_unsafe_path_text(value, label)
    return value


def _relative_path(value: Any, *, expected: str, label: str) -> str:
    text = _text(value, label, 256)
    if unquote(text) != text or "\\" in text:
        raise RawTranscriptionValidationError(
            f"{label} is not a safe job-relative path."
        )
    path = PurePosixPath(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != expected:
        raise RawTranscriptionValidationError(
            f"{label} is not the canonical artifact path."
        )
    return expected


def _reject_unsafe_path_text(value: str, label: str) -> None:
    decoded = value
    for _ in range(4):
        next_value = unquote(decoded)
        if next_value == decoded:
            break
        decoded = next_value
    for candidate in (value, decoded):
        if "\x00" in candidate:
            raise RawTranscriptionValidationError(f"{label} contains NUL.")
        normalized = candidate.replace("\\", "/")
        if (
            normalized.startswith("/")
            or normalized.startswith("//")
            or _WINDOWS_DRIVE_PATTERN.match(candidate)
            or _URI_PATTERN.match(candidate)
            or candidate.lower().startswith("file:")
        ):
            raise RawTranscriptionValidationError(f"{label} contains a machine path.")
        parts = tuple(part for part in normalized.split("/") if part not in {"", "."})
        if ".." in parts or _MACHINE_COMPONENT_PATTERN.search(normalized):
            raise RawTranscriptionValidationError(
                f"{label} contains unsafe traversal or a machine path."
            )


def _utc_timestamp(value: Any, label: str) -> str:
    text = _text(value, label, 128)
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError as exc:
        raise RawTranscriptionValidationError(
            f"{label} must be an ISO timestamp."
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
        raise RawTranscriptionValidationError(f"{label} must be a UTC timestamp.")
    return text


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RawTranscriptionValidationError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise RawTranscriptionValidationError(f"{label} must be finite.")
    if minimum is not None:
        if exclusive_minimum and not number > minimum:
            raise RawTranscriptionValidationError(
                f"{label} must be greater than {minimum}."
            )
        if not exclusive_minimum and number < minimum:
            raise RawTranscriptionValidationError(
                f"{label} must be at least {minimum}."
            )
    if maximum is not None and number > maximum:
        raise RawTranscriptionValidationError(
            f"{label} must be at most {maximum}."
        )
    return number


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise RawTranscriptionValidationError(f"{label} must be an integer.")
    return value


def _integer_range(value: Any, label: str, minimum: int, maximum: int) -> int:
    number = _integer(value, label)
    if not minimum <= number <= maximum:
        raise RawTranscriptionValidationError(
            f"{label} must be between {minimum} and {maximum}."
        )
    return number


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RawTranscriptionValidationError(f"{label} must be an object.")
    if not all(isinstance(key, str) for key in value):
        raise RawTranscriptionValidationError(
            f"{label} must use string field names."
        )
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise RawTranscriptionValidationError(f"{label} must be an array.")
    return list(value)


def _require_keys(
    value: Mapping[str, Any],
    *,
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    keys = set(value)
    if required - keys or keys - required - optional:
        raise RawTranscriptionValidationError(f"{label} has invalid fields.")


def _encoded_payload(payload: Mapping[str, Any]) -> bytes:
    try:
        data = (
            json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RawTranscriptionValidationError(
            "Raw transcription is not valid JSON data."
        ) from exc
    if len(data) > _MAX_ARTIFACT_BYTES:
        raise RawTranscriptionValidationError(
            "Raw transcription artifact is too large."
        )
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _secure_job_root(job_id: str, settings: Settings) -> Path:
    if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
        raise RawTranscriptionValidationError("Invalid job identifier.")
    try:
        lexical_exports = settings.exports_dir
        exports_info = lexical_exports.lstat()
        if stat.S_ISLNK(exports_info.st_mode) or not stat.S_ISDIR(exports_info.st_mode):
            raise RawTranscriptionError("Raw transcription job directory is unsafe.")
        exports_root = lexical_exports.resolve(strict=True)
        lexical_job = lexical_exports / job_id
        job_info = lexical_job.lstat()
        if stat.S_ISLNK(job_info.st_mode) or not stat.S_ISDIR(job_info.st_mode):
            raise RawTranscriptionError("Raw transcription job directory is unsafe.")
        job_root = lexical_job.resolve(strict=True)
        if job_root.parent != exports_root:
            raise RawTranscriptionError("Raw transcription job directory is unsafe.")
        expected = secure_job_dir(settings, job_id).resolve(strict=True)
        if expected != job_root:
            raise RawTranscriptionError("Raw transcription job directory is unsafe.")
        return job_root
    except (RawTranscriptionError, RawTranscriptionValidationError):
        raise
    except (AttributeError, MediaProcessingError, OSError, RuntimeError) as exc:
        raise RawTranscriptionError(
            "Raw transcription job directory is unavailable."
        ) from exc


def _artifact_directory(job_dir: Path, *, create: bool) -> Path | None:
    directory = job_dir / "transcription"
    if create:
        try:
            directory.mkdir(mode=0o700, exist_ok=True)
        except OSError as exc:
            raise RawTranscriptionError(
                "Raw transcription directory could not be created safely."
            ) from exc
    try:
        directory.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise RawTranscriptionError(
            "Raw transcription directory is unavailable."
        ) from exc
    _directory_snapshot(directory, job_dir)
    return directory


def _directory_snapshot(directory: Path, job_dir: Path) -> tuple[int, int, int]:
    try:
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RawTranscriptionError("Raw transcription directory is unsafe.")
        resolved = directory.resolve(strict=True)
        root = job_dir.resolve(strict=True)
        if resolved.parent != root:
            raise RawTranscriptionError("Raw transcription directory is unsafe.")
        return (info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode))
    except RawTranscriptionError:
        raise
    except OSError as exc:
        raise RawTranscriptionError("Raw transcription directory is unsafe.") from exc


def _validate_existing_destination(destination: Path, artifact_dir: Path) -> None:
    try:
        destination.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise RawTranscriptionError(
            "Existing raw transcription artifact is unavailable."
        ) from exc
    _require_regular_file(destination, artifact_dir)


def _write_exclusive_regular_file(path: Path, data: bytes, artifact_dir: Path) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    try:
        fd = os.open(path, flags, 0o600)
        offset = 0
        while offset < len(data):
            written = os.write(fd, data[offset:])
            if written <= 0:
                raise OSError("Short write")
            offset += written
        os.fsync(fd)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RawTranscriptionError(
                "Temporary raw transcription file is unsafe."
            )
    except RawTranscriptionError:
        raise
    except OSError as exc:
        raise RawTranscriptionError(
            "Raw transcription temporary file could not be written."
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
    _require_regular_file(path, artifact_dir)


def _replace_atomic(temporary: Path, destination: Path) -> None:
    try:
        os.replace(temporary, destination)
    except OSError as exc:
        raise RawTranscriptionError(
            "Raw transcription could not replace the published artifact."
        ) from exc


def _read_stable_regular_file(path: Path, artifact_dir: Path) -> bytes:
    before = _require_regular_file(path, artifact_dir)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    fd: int | None = None
    after_open: os.stat_result | None = None
    try:
        fd = os.open(path, flags)
        opened = os.fstat(fd)
        if _snapshot(opened) != _snapshot(before):
            raise RawTranscriptionError(
                "Saved raw transcription changed during validation."
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            remaining = _MAX_ARTIFACT_BYTES + 1 - total
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES:
                raise RawTranscriptionError("Saved raw transcription is too large.")
        after_open = os.fstat(fd)
    except RawTranscriptionError:
        raise
    except OSError as exc:
        raise RawTranscriptionError(
            "Saved raw transcription could not be read safely."
        ) from exc
    finally:
        if fd is not None:
            os.close(fd)
    after = _require_regular_file(path, artifact_dir)
    assert after_open is not None
    if _snapshot(before) != _snapshot(after_open) or _snapshot(before) != _snapshot(after):
        raise RawTranscriptionError(
            "Saved raw transcription changed during validation."
        )
    return b"".join(chunks)


def _require_regular_file(path: Path, artifact_dir: Path) -> os.stat_result:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise RawTranscriptionError("Raw transcription artifact path is unsafe.")
        resolved = path.resolve(strict=True)
        root = artifact_dir.resolve(strict=True)
        if resolved.parent != root:
            raise RawTranscriptionError("Raw transcription artifact path is unsafe.")
        return info
    except RawTranscriptionError:
        raise
    except OSError as exc:
        raise RawTranscriptionError(
            "Raw transcription artifact path is unsafe."
        ) from exc


def _remove_temporary(path: Path, artifact_dir: Path) -> None:
    try:
        info = path.lstat()
    except (FileNotFoundError, OSError):
        return
    if stat.S_ISREG(info.st_mode) and not stat.S_ISLNK(info.st_mode):
        try:
            if path.resolve(strict=True).parent == artifact_dir.resolve(strict=True):
                path.unlink()
        except OSError:
            pass


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        fd = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]:
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns)


__all__ = [
    "RAW_TRANSCRIPTION_RELATIVE_PATH",
    "RAW_TRANSCRIPTION_SCHEMA_VERSION",
    "RawTranscriptionError",
    "RawTranscriptionValidationError",
    "load_raw_transcription",
    "validate_raw_transcription",
    "write_raw_transcription",
]
