"""Advisory alignment of raw transcription events to analysis timing.

Raw event timestamps remain authoritative. This module derives reproducible
beat/subdivision candidates only; it does not alter note durations, construct
measures as notation, or mutate caller-owned mappings.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

ALIGNMENT_VERSION = "advisory-beat-grid-v1"
_MAX_WARNINGS = 8
_MAX_CANDIDATE_WARNINGS = 3
_MAX_WARNING_LENGTH = 160
_EVENT_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_MIN_DEFAULT_WINDOW_SECONDS = 0.02
_MAX_DEFAULT_WINDOW_SECONDS = 0.12
_MEASURE_CONFIDENCE_THRESHOLD = 0.5
_TIME_MATCH_EPSILON = 1e-7


class EventAlignmentError(RuntimeError):
    """Raised when raw events or timing evidence are malformed."""


@dataclass(frozen=True, slots=True)
class _GridPoint:
    time_seconds: float
    beat_index: int
    subdivision: int
    subdivision_index: int
    local_beat_seconds: float


@dataclass(frozen=True, slots=True)
class _MeasureEvidence:
    meter: int
    first_downbeat_index: int
    confidence: float


def align_raw_events_to_timing(
    pitched_events: Sequence[Mapping[str, Any]],
    percussion_events: Sequence[Mapping[str, Any]],
    timing: Mapping[str, Any],
    *,
    max_subdivision: int = 4,
    max_offset_seconds: float | None = None,
) -> dict[str, Any]:
    """Return advisory alignment candidates while preserving raw event onsets.

    ``offsetSeconds`` is signed as ``rawTimeSeconds - alignedTimeSeconds``:
    positive values are late relative to the candidate grid point.
    ``beatIndex`` and ``measureIndex`` are zero-based; ``beatInMeasure`` is
    one-based for musician-facing readability.
    """

    subdivisions = _validate_max_subdivision(max_subdivision)
    explicit_window = _validate_max_offset(max_offset_seconds)
    if not isinstance(timing, Mapping):
        raise EventAlignmentError("Timing evidence must be a mapping.")

    events = _validate_events(pitched_events, percussion_events)
    timing_data = _validate_timing(timing)
    warnings: list[str] = []

    beats = timing_data["beats"]
    beat_confidence = timing_data["beat_confidence"]
    tempo_confidence = timing_data["tempo_confidence"]
    tempo_stable = timing_data["tempo_stable"]

    if beat_confidence is None:
        _append_warning(
            warnings,
            "Beat confidence is unavailable; advisory confidence is intentionally limited.",
        )
    elif beat_confidence < 0.35:
        _append_warning(
            warnings,
            "Beat confidence is weak; treat all alignment candidates cautiously.",
        )
    if tempo_stable is False:
        _append_warning(
            warnings,
            "Tempo is not stable; local beat spacing is used and acceptance windows are reduced.",
        )

    grid: tuple[_GridPoint, ...] = ()
    measure_evidence: _MeasureEvidence | None = None
    if len(beats) < 2:
        _append_warning(
            warnings,
            "At least two observed beats are required; no beat-grid alignment was fabricated.",
        )
    else:
        grid = _build_grid(beats, subdivisions)
        measure_evidence = _measure_evidence(
            beats,
            timing_data["downbeats"],
            timing_data["meter"],
            timing_data["meter_confidence"],
            warnings,
        )

    candidates: list[dict[str, Any]] = []
    aligned_count = 0
    for event_id, event_type, raw_time in events:
        candidate = _candidate_for_event(
            event_id=event_id,
            event_type=event_type,
            raw_time=raw_time,
            grid=grid,
            explicit_window=explicit_window,
            beat_confidence=beat_confidence,
            tempo_confidence=tempo_confidence,
            tempo_stable=tempo_stable,
            measure_evidence=measure_evidence,
        )
        if "alignedTimeSeconds" in candidate:
            aligned_count += 1
        candidates.append(candidate)

    return {
        "alignmentVersion": ALIGNMENT_VERSION,
        "candidates": candidates,
        "warnings": warnings,
        "diagnostics": {
            "eventCount": len(events),
            "pitchedEventCount": len(pitched_events),
            "percussionEventCount": len(percussion_events),
            "beatCount": len(beats),
            "downbeatCount": len(timing_data["downbeats"]),
            "gridPointCount": len(grid),
            "alignedCount": aligned_count,
            "unalignedCount": len(events) - aligned_count,
            "maxSubdivision": subdivisions,
            "explicitMaxOffsetSeconds": explicit_window,
            "measureEvidenceUsed": measure_evidence is not None,
            "rawTimesPreserved": True,
        },
    }


def _validate_events(
    pitched_events: Sequence[Mapping[str, Any]],
    percussion_events: Sequence[Mapping[str, Any]],
) -> tuple[tuple[str, str, float], ...]:
    pitched = _require_sequence(pitched_events, "Pitched events")
    percussion = _require_sequence(percussion_events, "Percussion events")
    seen: set[str] = set()
    result: list[tuple[str, str, float]] = []

    for index, event in enumerate(pitched):
        mapping = _require_mapping(event, f"Pitched event {index}")
        event_id = _event_id(mapping, seen)
        start = _finite_nonnegative(mapping.get("startSeconds"), "Pitched startSeconds")
        if "endSeconds" in mapping:
            end = _finite_nonnegative(mapping.get("endSeconds"), "Pitched endSeconds")
            if end <= start:
                raise EventAlignmentError(
                    "Pitched endSeconds must be greater than startSeconds."
                )
        result.append((event_id, "pitched", start))

    for index, event in enumerate(percussion):
        mapping = _require_mapping(event, f"Percussion event {index}")
        event_id = _event_id(mapping, seen)
        onset = _finite_nonnegative(
            mapping.get("timeSeconds"), "Percussion timeSeconds"
        )
        result.append((event_id, "percussion", onset))

    return tuple(result)


def _require_sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EventAlignmentError(f"{label} must be a sequence.")
    return value


def _require_mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise EventAlignmentError(f"{label} must be a mapping.")
    return value


def _event_id(mapping: Mapping[str, Any], seen: set[str]) -> str:
    value = mapping.get("id")
    if not isinstance(value, str) or not _EVENT_ID_RE.fullmatch(value):
        raise EventAlignmentError("Event IDs must be safe non-empty identifiers.")
    if value in seen:
        raise EventAlignmentError("Event IDs must be unique across all event types.")
    seen.add(value)
    return value


def _validate_timing(timing: Mapping[str, Any]) -> dict[str, Any]:
    beats = _time_array(timing.get("beatsSeconds"), "Beat times", missing_ok=True)
    downbeats = _time_array(
        timing.get("downbeatsSeconds"), "Downbeat times", missing_ok=True
    )
    tempo_bpm = _optional_positive(timing.get("tempoBpm"), "tempoBpm")
    tempo_confidence = _optional_confidence(
        timing.get("tempoConfidence"), "tempoConfidence"
    )
    beat_confidence = _optional_confidence(
        timing.get("beatConfidence"), "beatConfidence"
    )
    meter_confidence = _optional_confidence(
        timing.get("meterConfidence"), "meterConfidence"
    )
    tempo_stable = timing.get("tempoStable")
    if tempo_stable is not None and type(tempo_stable) is not bool:
        raise EventAlignmentError("tempoStable must be a Boolean or null.")

    meter = timing.get("meter")
    if meter is not None:
        if type(meter) is not int or not 2 <= meter <= 12:
            raise EventAlignmentError("meter must be an integer from 2 through 12.")

    return {
        "beats": beats,
        "downbeats": downbeats,
        "tempo_bpm": tempo_bpm,
        "tempo_confidence": tempo_confidence,
        "beat_confidence": beat_confidence,
        "tempo_stable": tempo_stable,
        "meter": meter,
        "meter_confidence": meter_confidence,
    }


def _time_array(value: object, label: str, *, missing_ok: bool) -> tuple[float, ...]:
    if value is None and missing_ok:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise EventAlignmentError(f"{label} must be an array.")
    result = tuple(_finite_nonnegative(item, label) for item in value)
    if any(later <= earlier for earlier, later in zip(result, result[1:])):
        raise EventAlignmentError(f"{label} must be strictly increasing.")
    return result


def _validate_max_subdivision(value: object) -> int:
    if type(value) is not int or not 1 <= value <= 4:
        raise EventAlignmentError("max_subdivision must be an integer from 1 through 4.")
    return value


def _validate_max_offset(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventAlignmentError(
            "max_offset_seconds must be a finite non-negative number."
        )
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EventAlignmentError(
            "max_offset_seconds must be a finite non-negative number."
        )
    return number


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventAlignmentError(f"{label} must contain finite non-negative numbers.")
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise EventAlignmentError(f"{label} must contain finite non-negative numbers.")
    return number


def _optional_positive(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventAlignmentError(f"{label} must be a positive finite number or null.")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise EventAlignmentError(f"{label} must be a positive finite number or null.")
    return number


def _optional_confidence(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EventAlignmentError(f"{label} must be between 0 and 1 or null.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise EventAlignmentError(f"{label} must be between 0 and 1 or null.")
    return number


def _build_grid(
    beats: tuple[float, ...], max_subdivision: int
) -> tuple[_GridPoint, ...]:
    # A rounded key removes arithmetic duplicates such as 1/2 and 2/4 while
    # retaining the simpler canonical subdivision in deterministic order.
    points: dict[float, _GridPoint] = {}
    for beat_index, (start, end) in enumerate(zip(beats, beats[1:])):
        duration = end - start
        for subdivision in range(1, max_subdivision + 1):
            for subdivision_index in range(subdivision):
                point_time = start + duration * subdivision_index / subdivision
                key = round(point_time, 12)
                candidate = _GridPoint(
                    time_seconds=point_time,
                    beat_index=beat_index,
                    subdivision=subdivision,
                    subdivision_index=subdivision_index,
                    local_beat_seconds=duration,
                )
                current = points.get(key)
                if current is None or _grid_identity(candidate) < _grid_identity(current):
                    points[key] = candidate

    final_duration = beats[-1] - beats[-2]
    final = _GridPoint(
        time_seconds=beats[-1],
        beat_index=len(beats) - 1,
        subdivision=1,
        subdivision_index=0,
        local_beat_seconds=final_duration,
    )
    points[round(beats[-1], 12)] = final
    return tuple(sorted(points.values(), key=_grid_sort_key))


def _grid_identity(point: _GridPoint) -> tuple[int, int, int]:
    return (point.subdivision, point.subdivision_index, point.beat_index)


def _grid_sort_key(point: _GridPoint) -> tuple[float, int, int, int]:
    return (
        point.time_seconds,
        point.subdivision,
        point.subdivision_index,
        point.beat_index,
    )


def _candidate_for_event(
    *,
    event_id: str,
    event_type: str,
    raw_time: float,
    grid: tuple[_GridPoint, ...],
    explicit_window: float | None,
    beat_confidence: float | None,
    tempo_confidence: float | None,
    tempo_stable: bool | None,
    measure_evidence: _MeasureEvidence | None,
) -> dict[str, Any]:
    candidate: dict[str, Any] = {
        "eventId": event_id,
        "eventType": event_type,
        "rawTimeSeconds": raw_time,
        "confidence": 0.0,
    }
    if not grid:
        candidate["warnings"] = [
            "No observed beat grid is available; the raw event time is unchanged."
        ]
        return candidate

    admissible: list[tuple[float, int, float, int, int, _GridPoint, float]] = []
    for point in grid:
        window = _acceptance_window(
            point.local_beat_seconds,
            explicit_window,
            beat_confidence,
            tempo_stable,
        )
        distance = abs(raw_time - point.time_seconds)
        if distance <= window + _TIME_MATCH_EPSILON:
            admissible.append(
                (
                    distance,
                    point.subdivision,
                    point.time_seconds,
                    point.beat_index,
                    point.subdivision_index,
                    point,
                    window,
                )
            )

    if not admissible:
        candidate["warnings"] = [
            "No beat-grid point is within the local acceptance window; the raw time is unchanged."
        ]
        return candidate

    _, _, _, _, _, point, window = min(admissible)
    offset = raw_time - point.time_seconds
    confidence = _alignment_confidence(
        abs(offset),
        window,
        beat_confidence,
        tempo_confidence,
        tempo_stable,
        measure_evidence.confidence if measure_evidence is not None else None,
    )
    candidate.update(
        {
            "beatIndex": point.beat_index,
            "subdivision": point.subdivision,
            "subdivisionIndex": point.subdivision_index,
            "alignedTimeSeconds": _clean_number(point.time_seconds),
            "offsetSeconds": _clean_number(offset),
            "confidence": confidence,
        }
    )

    candidate_warnings: list[str] = []
    evidence = beat_confidence if beat_confidence is not None else 0.35
    if evidence < 0.35 or tempo_stable is False:
        _append_candidate_warning(
            candidate_warnings,
            "Timing evidence is weak; this candidate is advisory and should be reviewed.",
        )
    if (
        measure_evidence is not None
        and point.beat_index >= measure_evidence.first_downbeat_index
    ):
        relative = point.beat_index - measure_evidence.first_downbeat_index
        candidate["measureIndex"] = relative // measure_evidence.meter
        candidate["beatInMeasure"] = relative % measure_evidence.meter + 1
    if candidate_warnings:
        candidate["warnings"] = candidate_warnings
    return candidate


def _acceptance_window(
    local_beat_seconds: float,
    explicit_window: float | None,
    beat_confidence: float | None,
    tempo_stable: bool | None,
) -> float:
    stability_factor = (
        0.18 if tempo_stable is True else 0.12 if tempo_stable is False else 0.15
    )
    window = local_beat_seconds * stability_factor
    window = max(
        _MIN_DEFAULT_WINDOW_SECONDS,
        min(_MAX_DEFAULT_WINDOW_SECONDS, window),
    )
    confidence = beat_confidence if beat_confidence is not None else 0.35
    if confidence < 0.4:
        window = max(_MIN_DEFAULT_WINDOW_SECONDS, window * 0.75)
    if explicit_window is not None:
        window = min(window, explicit_window)
    return window


def _alignment_confidence(
    distance: float,
    window: float,
    beat_confidence: float | None,
    tempo_confidence: float | None,
    tempo_stable: bool | None,
    meter_confidence: float | None,
) -> float:
    if window <= 0:
        distance_score = 1.0 if distance <= _TIME_MATCH_EPSILON else 0.0
    else:
        distance_score = max(0.0, 1.0 - distance / window)
    beat_score = beat_confidence if beat_confidence is not None else 0.35
    tempo_score = tempo_confidence if tempo_confidence is not None else beat_score
    stability_score = (
        0.95 if tempo_stable is True else 0.55 if tempo_stable is False else 0.7
    )
    meter_score = meter_confidence if meter_confidence is not None else 0.0
    evidence = (
        0.5 * beat_score
        + 0.2 * tempo_score
        + 0.2 * stability_score
        + 0.1 * meter_score
    )
    return round(max(0.0, min(0.98, distance_score * evidence)), 6)


def _measure_evidence(
    beats: tuple[float, ...],
    downbeats: tuple[float, ...],
    meter: int | None,
    meter_confidence: float | None,
    warnings: list[str],
) -> _MeasureEvidence | None:
    if meter is None or meter_confidence is None or not downbeats:
        return None
    if meter_confidence < _MEASURE_CONFIDENCE_THRESHOLD:
        _append_warning(
            warnings,
            "Meter confidence is insufficient; measure and beat-in-measure indices are omitted.",
        )
        return None

    indices: list[int] = []
    for downbeat in downbeats:
        matches = [
            index
            for index, beat in enumerate(beats)
            if abs(beat - downbeat) <= _TIME_MATCH_EPSILON
        ]
        if len(matches) != 1:
            _append_warning(
                warnings,
                "Downbeat evidence does not match observed beats; measure indices are omitted.",
            )
            return None
        indices.append(matches[0])
    if any(
        later - earlier != meter for earlier, later in zip(indices, indices[1:])
    ):
        _append_warning(
            warnings,
            "Downbeat spacing is inconsistent with the reported meter; measure indices are omitted.",
        )
        return None
    return _MeasureEvidence(
        meter=meter,
        first_downbeat_index=indices[0],
        confidence=meter_confidence,
    )


def _clean_number(value: float) -> float:
    rounded = round(value, 12)
    return 0.0 if rounded == 0 else rounded


def _append_warning(warnings: list[str], message: str) -> None:
    bounded = message[:_MAX_WARNING_LENGTH]
    if bounded not in warnings and len(warnings) < _MAX_WARNINGS:
        warnings.append(bounded)


def _append_candidate_warning(warnings: list[str], message: str) -> None:
    bounded = message[:_MAX_WARNING_LENGTH]
    if bounded not in warnings and len(warnings) < _MAX_CANDIDATE_WARNINGS:
        warnings.append(bounded)


__all__ = [
    "ALIGNMENT_VERSION",
    "EventAlignmentError",
    "align_raw_events_to_timing",
]
