"""Conservative percussion interpretation for editable draft structure.

Raw onset evidence remains authoritative. This module maps detector hit labels to
broad, editable drum voices and advisory rhythmic groups; it does not construct
final drum notation or infer kit orchestration, sticking, fills, or articulation.
"""

from __future__ import annotations

import copy
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

PERCUSSION_INTERPRETATION_VERSION = "broad-drum-structure-v1"

_MAX_EVENTS = 2_048
_MAX_HITS_PER_EVENT = 16
_MAX_ASSIGNMENTS = 8_192
_MAX_EVENT_WARNINGS = 16
_MAX_WARNING_LENGTH = 240
_MAX_TOP_LEVEL_WARNINGS = 16
_MAX_RAW_FEATURE_KEYS = 48
_MAX_JSON_NODES = 256
_MAX_ALIGNMENT_WARNINGS = 8
_MAX_ALTERNATIVES = 4

_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SLUG_RE = re.compile(r"^[a-z][a-z0-9]*(?:[_-][a-z0-9]+)*$")
_VERSION_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,127}$")
_JSON_KEY_RE = re.compile(r"^[A-Za-z][A-Za-z0-9._-]{0,63}$")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]")
_URL_RE = re.compile(r"(?i)https?://")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:^|\s)(?:[A-Z]:[\\/]|\\\\)")
_POSIX_PATH_RE = re.compile(r"(?:^|\s)/(?:[^\s/]+/)+")
_CREDENTIAL_RE = re.compile(
    r"(?i)\b(?:token|password|secret|authorization|api[_-]?key|access[_-]?key)\b\s*[:=]"
)

_PART_ID = "drum-part-001"

_VOICE_DEFINITIONS: dict[str, tuple[str, str]] = {
    "low_drum": ("drum-voice-low-drum", "Low drum"),
    "mid_drum": ("drum-voice-mid-drum", "Mid drum"),
    "tom_like": ("drum-voice-tom-like", "Tom-like voice"),
    "closed_high_frequency": (
        "drum-voice-closed-high-frequency",
        "Closed high-frequency voice",
    ),
    "open_high_frequency": (
        "drum-voice-open-high-frequency",
        "Open high-frequency voice",
    ),
    "cymbal_like": ("drum-voice-cymbal-like", "Cymbal-like voice"),
    "unresolved_percussion": (
        "drum-voice-unresolved-percussion",
        "Unresolved percussion",
    ),
}
_VOICE_ORDER = tuple(_VOICE_DEFINITIONS)

_HIT_KIND_TO_VOICE = {
    "kick": "low_drum",
    "low_drum": "low_drum",
    "snare": "mid_drum",
    "mid_drum": "mid_drum",
    "tom": "tom_like",
    "tom_like": "tom_like",
    "closed_hihat": "closed_high_frequency",
    "closed_hat": "closed_high_frequency",
    "hihat_closed": "closed_high_frequency",
    "open_hihat": "open_high_frequency",
    "open_hat": "open_high_frequency",
    "hihat_open": "open_high_frequency",
    "cymbal": "cymbal_like",
    "cymbal_like": "cymbal_like",
    "unknown_percussion": "unresolved_percussion",
}
_MIN_RESOLVED_CONFIDENCE = 0.35

_ALIGNMENT_KEYS = {
    "eventId",
    "eventType",
    "rawTimeSeconds",
    "confidence",
    "beatIndex",
    "subdivision",
    "subdivisionIndex",
    "alignedTimeSeconds",
    "offsetSeconds",
    "measureIndex",
    "beatInMeasure",
    "warnings",
}
_GRID_KEYS = {
    "beatIndex",
    "subdivision",
    "subdivisionIndex",
    "alignedTimeSeconds",
    "offsetSeconds",
}


class PercussionInterpretationError(RuntimeError):
    """Raw percussion evidence could not be interpreted safely."""


@dataclass(frozen=True, slots=True)
class PercussionInterpretationResult:
    version: str
    parts: tuple[dict[str, Any], ...]
    voices: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    unresolved_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def interpret_percussion(
    percussion_events: Sequence[Mapping[str, Any]],
    alignment_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    version: str = PERCUSSION_INTERPRETATION_VERSION,
) -> PercussionInterpretationResult:
    """Return deterministic broad drum structure without replacing raw timing."""

    safe_version = _version(version)
    events = _events(percussion_events)
    alignments = _alignments(alignment_candidates, events)

    assignments: list[dict[str, Any]] = []
    event_resolution: dict[str, bool] = {}
    used_voice_kinds: dict[str, set[str]] = {}
    simultaneous_count = 0
    aligned_event_count = 0

    for event_order, event in enumerate(events, start=1):
        event_id = event["id"]
        alignment = alignments.get(event_id)
        if alignment is not None and alignment["aligned"]:
            aligned_event_count += 1
        if len(event["hits"]) > 1:
            simultaneous_count += 1

        has_resolved = False
        for hit_index, hit in enumerate(event["hits"]):
            if len(assignments) >= _MAX_ASSIGNMENTS:
                raise PercussionInterpretationError(
                    "Percussion interpretation exceeded the assignment limit."
                )
            raw_kind = hit["kind"]
            confidence = hit["confidence"]
            mapped_kind = _HIT_KIND_TO_VOICE.get(raw_kind)
            resolved = (
                mapped_kind is not None
                and mapped_kind != "unresolved_percussion"
                and confidence >= _MIN_RESOLVED_CONFIDENCE
            )
            voice_kind = mapped_kind if resolved else "unresolved_percussion"
            voice_id, _ = _VOICE_DEFINITIONS[voice_kind]
            used_voice_kinds.setdefault(voice_kind, set()).add(raw_kind)
            has_resolved = has_resolved or resolved

            assignment: dict[str, Any] = {
                "id": f"drum-assignment-{len(assignments) + 1:06d}",
                "partId": _PART_ID,
                "voiceId": voice_id,
                "eventId": event_id,
                "eventOrder": event_order,
                "hitIndex": hit_index,
                "rawHitKind": raw_kind,
                "rawHit": copy.deepcopy(hit["raw"]),
                "rawTimeSeconds": event["timeSeconds"],
                "sourceKind": event["sourceKind"],
                "strength": event["strength"],
                "confidence": confidence,
                "resolution": "resolved" if resolved else "unresolved",
                "alternatives": [],
                "eventWarnings": list(event["warnings"]),
                "rawFeatureSummary": copy.deepcopy(event["rawFeatureSummary"]),
                "alignment": _assignment_alignment(alignment, event["timeSeconds"]),
            }
            assignments.append(assignment)
        event_resolution[event_id] = has_resolved

    voices = _voices(used_voice_kinds, assignments)
    groups = _groups(events, alignments, assignments)
    unresolved_event_ids = tuple(
        event["id"] for event in events if not event_resolution[event["id"]]
    )

    warnings: list[str] = []
    unaligned_count = len(events) - aligned_event_count
    unresolved_assignment_count = sum(
        item["resolution"] == "unresolved" for item in assignments
    )
    if unresolved_assignment_count:
        _append_warning(
            warnings,
            "Some percussion hits remain unresolved; their raw kinds, timing, and confidence are preserved.",
        )
    if unaligned_count:
        _append_warning(
            warnings,
            "Some percussion events remain time-relative because no valid advisory grid placement was supplied.",
        )
    if simultaneous_count:
        _append_warning(
            warnings,
            "Simultaneous broad hit evidence is preserved as separate editable assignments.",
        )

    parts: tuple[dict[str, Any], ...]
    if events:
        parts = (
            {
                "id": _PART_ID,
                "kind": "percussion",
                "label": "Percussion",
                "editable": True,
                "voiceIds": [voice["id"] for voice in voices],
                "groupIds": [group["id"] for group in groups],
                "assignmentIds": [item["id"] for item in assignments],
                "rawEventIds": [event["id"] for event in events],
            },
        )
    else:
        parts = ()
        _append_warning(warnings, "No raw percussion events were supplied.")

    diagnostics = {
        "eventCount": len(events),
        "hitCount": len(assignments),
        "assignmentCount": len(assignments),
        "resolvedAssignmentCount": len(assignments) - unresolved_assignment_count,
        "unresolvedAssignmentCount": unresolved_assignment_count,
        "unresolvedEventCount": len(unresolved_event_ids),
        "simultaneousEventCount": simultaneous_count,
        "alignedEventCount": aligned_event_count,
        "unalignedEventCount": unaligned_count,
        "voiceCount": len(voices),
        "groupCount": len(groups),
        "rawTimesPreserved": True,
        "simultaneousHitsPreserved": True,
        "broadVoicesOnly": True,
        "finalNotationConstructed": False,
    }
    _validate_json(diagnostics)
    return PercussionInterpretationResult(
        version=safe_version,
        parts=parts,
        voices=voices,
        groups=groups,
        assignments=tuple(assignments),
        unresolved_event_ids=unresolved_event_ids,
        warnings=tuple(warnings),
        diagnostics=diagnostics,
    )


def _version(value: object) -> str:
    if not isinstance(value, str) or _VERSION_RE.fullmatch(value) is None:
        raise PercussionInterpretationError(
            "The percussion interpretation version is invalid."
        )
    return value


def _sequence(value: object, label: str, *, limit: int) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise PercussionInterpretationError(f"{label} must be a sequence.")
    if len(value) > limit:
        raise PercussionInterpretationError(f"{label} exceeds the supported limit.")
    return value


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PercussionInterpretationError(f"{label} must be a mapping.")
    return value


def _events(value: object) -> tuple[dict[str, Any], ...]:
    sequence = _sequence(value, "Percussion events", limit=_MAX_EVENTS)
    seen: set[str] = set()
    parsed: list[dict[str, Any]] = []
    total_hits = 0
    for index, raw_event in enumerate(sequence):
        event = _mapping(raw_event, f"Percussion event {index}")
        event_id = _event_id(event.get("id"), seen)
        source_kind = _slug(event.get("sourceKind"), "source kind")
        time_seconds = _nonnegative(event.get("timeSeconds"), "timeSeconds")
        strength = _confidence(event.get("strength"), "strength")
        hits_raw = _sequence(
            event.get("hits"),
            f"Percussion event {event_id} hits",
            limit=_MAX_HITS_PER_EVENT,
        )
        if not hits_raw:
            raise PercussionInterpretationError(
                "Percussion events must contain at least one hit candidate."
            )
        hits: list[dict[str, Any]] = []
        for hit_index, raw_hit in enumerate(hits_raw):
            hit = _mapping(raw_hit, f"Percussion hit {event_id}:{hit_index}")
            kind = _slug(hit.get("kind"), "hit kind")
            confidence = _confidence(hit.get("confidence"), "hit confidence")
            raw_copy = _json_copy(hit, label="Raw hit")
            hits.append(
                {
                    "kind": kind,
                    "confidence": confidence,
                    "raw": raw_copy,
                }
            )
        total_hits += len(hits)
        if total_hits > _MAX_ASSIGNMENTS:
            raise PercussionInterpretationError(
                "Percussion interpretation exceeded the assignment limit."
            )

        warnings = _warnings(
            event.get("warnings", ()),
            label=f"Percussion event {event_id} warnings",
            limit=_MAX_EVENT_WARNINGS,
        )
        feature_summary_raw = event.get("rawFeatureSummary", {})
        feature_summary_mapping = _mapping(
            feature_summary_raw, f"Percussion event {event_id} rawFeatureSummary"
        )
        if len(feature_summary_mapping) > _MAX_RAW_FEATURE_KEYS:
            raise PercussionInterpretationError(
                "A raw feature summary exceeds the supported limit."
            )
        feature_summary = _json_copy(
            feature_summary_mapping,
            label="Raw feature summary",
        )
        parsed.append(
            {
                "id": event_id,
                "sourceKind": source_kind,
                "timeSeconds": time_seconds,
                "strength": strength,
                "hits": hits,
                "rawFeatureSummary": feature_summary,
                "warnings": warnings,
            }
        )
    parsed.sort(key=lambda item: (item["timeSeconds"], item["id"]))
    return tuple(parsed)


def _event_id(value: object, seen: set[str]) -> str:
    if not isinstance(value, str) or _EVENT_ID_RE.fullmatch(value) is None:
        raise PercussionInterpretationError(
            "Percussion event IDs must be safe non-empty identifiers."
        )
    if value in seen:
        raise PercussionInterpretationError("Percussion event IDs must be unique.")
    seen.add(value)
    return value


def _slug(value: object, label: str) -> str:
    if not isinstance(value, str) or len(value) > 64 or _SLUG_RE.fullmatch(value) is None:
        raise PercussionInterpretationError(f"The {label} is invalid.")
    return value


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PercussionInterpretationError(f"{label} must be a finite number.")
    number = float(value)
    if not math.isfinite(number):
        raise PercussionInterpretationError(f"{label} must be a finite number.")
    return number


def _nonnegative(value: object, label: str) -> float:
    number = _number(value, label)
    if number < 0:
        raise PercussionInterpretationError(f"{label} must be non-negative.")
    return number


def _confidence(value: object, label: str) -> float:
    number = _number(value, label)
    if not 0 <= number <= 1:
        raise PercussionInterpretationError(f"{label} must be between 0 and 1.")
    return number


def _warnings(value: object, *, label: str, limit: int) -> list[str]:
    sequence = _sequence(value, label, limit=limit)
    result: list[str] = []
    for item in sequence:
        if not isinstance(item, str) or not item or len(item) > _MAX_WARNING_LENGTH:
            raise PercussionInterpretationError(f"{label} contains an invalid warning.")
        _safe_text(item, label)
        result.append(item)
    return result


def _safe_text(value: str, label: str) -> None:
    if (
        _CONTROL_RE.search(value)
        or _URL_RE.search(value)
        or _WINDOWS_PATH_RE.search(value)
        or _POSIX_PATH_RE.search(value)
        or _CREDENTIAL_RE.search(value)
    ):
        raise PercussionInterpretationError(f"{label} contains unsafe text.")


def _json_copy(value: object, *, label: str) -> Any:
    nodes = [0]

    def visit(item: object, depth: int) -> Any:
        nodes[0] += 1
        if nodes[0] > _MAX_JSON_NODES or depth > 5:
            raise PercussionInterpretationError(f"{label} exceeds the supported limit.")
        if item is None or type(item) is bool or type(item) is int:
            return item
        if type(item) is float:
            if not math.isfinite(item):
                raise PercussionInterpretationError(f"{label} contains an invalid number.")
            return item
        if isinstance(item, str):
            if len(item) > _MAX_WARNING_LENGTH:
                raise PercussionInterpretationError(f"{label} contains an oversized string.")
            _safe_text(item, label)
            return item
        if isinstance(item, Mapping):
            result: dict[str, Any] = {}
            for key, nested in item.items():
                if not isinstance(key, str) or _JSON_KEY_RE.fullmatch(key) is None:
                    raise PercussionInterpretationError(f"{label} contains an invalid key.")
                result[key] = visit(nested, depth + 1)
            return result
        if isinstance(item, Sequence) and not isinstance(item, (str, bytes, bytearray)):
            if len(item) > 64:
                raise PercussionInterpretationError(f"{label} contains an oversized array.")
            return [visit(nested, depth + 1) for nested in item]
        raise PercussionInterpretationError(f"{label} contains unsupported data.")

    return visit(value, 0)


def _alignments(
    value: object,
    events: tuple[dict[str, Any], ...],
) -> dict[str, dict[str, Any]]:
    sequence = _sequence(value, "Alignment candidates", limit=_MAX_EVENTS)
    event_by_id = {event["id"]: event for event in events}
    result: dict[str, dict[str, Any]] = {}
    for index, raw_candidate in enumerate(sequence):
        candidate = _mapping(raw_candidate, f"Alignment candidate {index}")
        unknown = set(candidate) - _ALIGNMENT_KEYS
        if unknown:
            raise PercussionInterpretationError(
                "Alignment candidates contain unsupported fields."
            )
        event_id = candidate.get("eventId")
        if not isinstance(event_id, str) or _EVENT_ID_RE.fullmatch(event_id) is None:
            raise PercussionInterpretationError(
                "Alignment event references must be safe identifiers."
            )
        if event_id in result:
            raise PercussionInterpretationError(
                "Alignment candidates must reference each event at most once."
            )
        event = event_by_id.get(event_id)
        if event is None:
            raise PercussionInterpretationError(
                "Alignment candidates must reference supplied percussion events."
            )
        event_type = candidate.get("eventType")
        if event_type is not None and event_type != "percussion":
            raise PercussionInterpretationError(
                "Percussion alignment candidates must use eventType percussion."
            )
        raw_time = _nonnegative(candidate.get("rawTimeSeconds"), "rawTimeSeconds")
        if raw_time != event["timeSeconds"]:
            raise PercussionInterpretationError(
                "Alignment raw times must match the supplied percussion event."
            )
        confidence = _confidence(candidate.get("confidence"), "alignment confidence")
        warnings = _warnings(
            candidate.get("warnings", ()),
            label="Alignment warnings",
            limit=_MAX_ALIGNMENT_WARNINGS,
        )
        has_grid = "alignedTimeSeconds" in candidate
        present_grid = _GRID_KEYS.intersection(candidate)
        if has_grid and present_grid != _GRID_KEYS:
            raise PercussionInterpretationError(
                "Aligned candidates must contain the complete grid placement."
            )
        if not has_grid and present_grid:
            raise PercussionInterpretationError(
                "Unaligned candidates may not contain partial grid placement."
            )

        parsed: dict[str, Any] = {
            "aligned": has_grid,
            "confidence": confidence,
            "warnings": warnings,
        }
        if has_grid:
            beat_index = _nonnegative_int(candidate.get("beatIndex"), "beatIndex")
            subdivision = _positive_int(candidate.get("subdivision"), "subdivision")
            if subdivision > 16:
                raise PercussionInterpretationError(
                    "Alignment subdivision exceeds the supported limit."
                )
            subdivision_index = _nonnegative_int(
                candidate.get("subdivisionIndex"), "subdivisionIndex"
            )
            if subdivision_index >= subdivision:
                raise PercussionInterpretationError(
                    "subdivisionIndex must be smaller than subdivision."
                )
            aligned_time = _nonnegative(
                candidate.get("alignedTimeSeconds"), "alignedTimeSeconds"
            )
            offset = _number(candidate.get("offsetSeconds"), "offsetSeconds")
            if not math.isclose(
                raw_time - aligned_time,
                offset,
                rel_tol=0.0,
                abs_tol=1e-9,
            ):
                raise PercussionInterpretationError(
                    "Alignment offset is inconsistent with raw and aligned time."
                )
            parsed.update(
                {
                    "beatIndex": beat_index,
                    "subdivision": subdivision,
                    "subdivisionIndex": subdivision_index,
                    "alignedTimeSeconds": aligned_time,
                    "offsetSeconds": offset,
                }
            )
            has_measure = "measureIndex" in candidate or "beatInMeasure" in candidate
            if has_measure:
                if "measureIndex" not in candidate or "beatInMeasure" not in candidate:
                    raise PercussionInterpretationError(
                        "Measure placement must include measureIndex and beatInMeasure."
                    )
                measure_index = _nonnegative_int(
                    candidate.get("measureIndex"), "measureIndex"
                )
                beat_in_measure = _positive_int(
                    candidate.get("beatInMeasure"), "beatInMeasure"
                )
                if beat_in_measure > 12:
                    raise PercussionInterpretationError(
                        "beatInMeasure exceeds the supported meter bound."
                    )
                parsed["measureIndex"] = measure_index
                parsed["beatInMeasure"] = beat_in_measure
        result[event_id] = parsed
    return result


def _nonnegative_int(value: object, label: str) -> int:
    if type(value) is not int or value < 0:
        raise PercussionInterpretationError(f"{label} must be a non-negative integer.")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PercussionInterpretationError(f"{label} must be a positive integer.")
    return value


def _assignment_alignment(
    alignment: dict[str, Any] | None,
    raw_time: float,
) -> dict[str, Any]:
    if alignment is None:
        return {
            "aligned": False,
            "rawTimeSeconds": raw_time,
            "confidence": 0.0,
            "warnings": [],
        }
    result: dict[str, Any] = {
        "aligned": alignment["aligned"],
        "rawTimeSeconds": raw_time,
        "confidence": alignment["confidence"],
        "warnings": list(alignment["warnings"]),
    }
    if alignment["aligned"]:
        for key in (
            "beatIndex",
            "subdivision",
            "subdivisionIndex",
            "alignedTimeSeconds",
            "offsetSeconds",
            "measureIndex",
            "beatInMeasure",
        ):
            if key in alignment:
                result[key] = alignment[key]
    return result


def _voices(
    used_voice_kinds: dict[str, set[str]],
    assignments: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    assignment_ids_by_voice: dict[str, list[str]] = {}
    for assignment in assignments:
        assignment_ids_by_voice.setdefault(assignment["voiceId"], []).append(
            assignment["id"]
        )
    voices: list[dict[str, Any]] = []
    for kind in _VOICE_ORDER:
        if kind not in used_voice_kinds:
            continue
        voice_id, label = _VOICE_DEFINITIONS[kind]
        voices.append(
            {
                "id": voice_id,
                "partId": _PART_ID,
                "kind": kind,
                "label": label,
                "broad": True,
                "sourceHitKinds": sorted(used_voice_kinds[kind]),
                "assignmentIds": assignment_ids_by_voice.get(voice_id, []),
            }
        )
    return tuple(voices)


def _groups(
    events: tuple[dict[str, Any], ...],
    alignments: dict[str, dict[str, Any]],
    assignments: list[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    event_by_id = {event["id"]: event for event in events}
    assignments_by_event: dict[str, list[str]] = {}
    for assignment in assignments:
        assignments_by_event.setdefault(assignment["eventId"], []).append(assignment["id"])

    buckets: dict[tuple[str, int], list[str]] = {}
    for event in events:
        alignment = alignments.get(event["id"])
        if not alignment or not alignment["aligned"]:
            continue
        if "measureIndex" in alignment:
            key = ("measure_candidate", alignment["measureIndex"])
        else:
            key = ("beat_candidate", alignment["beatIndex"])
        buckets.setdefault(key, []).append(event["id"])

    ordered = sorted(
        buckets.items(),
        key=lambda item: (
            min(event_by_id[event_id]["timeSeconds"] for event_id in item[1]),
            item[0][0],
            item[0][1],
        ),
    )
    groups: list[dict[str, Any]] = []
    for index, ((kind, position), event_ids) in enumerate(ordered, start=1):
        event_ids.sort(
            key=lambda event_id: (
                event_by_id[event_id]["timeSeconds"],
                event_id,
            )
        )
        group: dict[str, Any] = {
            "id": f"drum-group-{index:06d}",
            "partId": _PART_ID,
            "kind": kind,
            "advisory": True,
            "eventIds": event_ids,
            "assignmentIds": [
                assignment_id
                for event_id in event_ids
                for assignment_id in assignments_by_event[event_id]
            ],
            "rawStartSeconds": min(
                event_by_id[event_id]["timeSeconds"] for event_id in event_ids
            ),
            "rawEndSeconds": max(
                event_by_id[event_id]["timeSeconds"] for event_id in event_ids
            ),
        }
        if kind == "measure_candidate":
            group["measureIndex"] = position
            group["beatIndices"] = sorted(
                {
                    alignments[event_id]["beatIndex"]
                    for event_id in event_ids
                }
            )
        else:
            group["beatIndex"] = position
        groups.append(group)
    return tuple(groups)


def _append_warning(warnings: list[str], message: str) -> None:
    if message not in warnings and len(warnings) < _MAX_TOP_LEVEL_WARNINGS:
        warnings.append(message)


def _validate_json(value: object) -> None:
    try:
        _json_copy(value, label="Interpretation diagnostics")
    except PercussionInterpretationError:
        raise


__all__ = [
    "PERCUSSION_INTERPRETATION_VERSION",
    "PercussionInterpretationError",
    "PercussionInterpretationResult",
    "interpret_percussion",
]
