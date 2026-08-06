"""Conservative rhythm hypotheses with authoritative raw timing."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any, TypeAlias


RHYTHM_INTERPRETATION_VERSION = "conservative-grid-v1"

_EVENT_ID_PATTERN = re.compile(r"[\w.-]{1,128}")
_SOURCE_SLUG_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,63}")
_VERSION_PATTERN = re.compile(r"[\w.+-]{1,128}")
_UNSAFE_TEXT_PATTERN = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|users|tmp|var|etc|mnt|private|opt|usr)/|\w+://)",
    re.IGNORECASE,
)

_MAX_EVENTS = 100_000
_MAX_ALIGNMENT_CANDIDATES = 200_000
_MAX_TIMING_POINTS = 200_000
_MAX_WARNINGS_PER_ALIGNMENT = 6
_MAX_WARNING_LENGTH = 200
_MAX_DURATION_ALTERNATIVES = 3
_MAX_CONTINUATION_HYPOTHESES = 4
_MAX_TOP_LEVEL_WARNINGS = 32

_PLACEMENT_CONFIDENCE_THRESHOLD = 0.55
_DURATION_CONFIDENCE_THRESHOLD = 0.60
_DURATION_AMBIGUITY_MARGIN = 0.12
_DURATION_SELECTION_WINDOW = 0.20
_UNSTABLE_TEMPO_CONFIDENCE_CAP = 0.58
_UNCERTAIN_SOURCE_REST_CONFIDENCE_CAP = 0.45

_DEFAULT_EVENT_CONFIDENCE = 0.50
_DEFAULT_ALIGNMENT_CONFIDENCE = 0.0
_DEFAULT_METER_CONFIDENCE = 0.25
_DEFAULT_BEAT_CONFIDENCE_FOR_DURATION = 0.35

_DURATION_CANDIDATES: tuple[tuple[float, str, int | str], ...] = (
    (0.25, "sixteenth", 4),
    (1 / 3, "eighth_triplet", "3T"),
    (0.5, "eighth", 2),
    (2 / 3, "quarter_triplet", "3T"),
    (0.75, "dotted_eighth", 4),
    (1.0, "quarter", 1),
    (1.5, "dotted_quarter", 2),
    (2.0, "half", 1),
    (3.0, "dotted_half", 1),
    (4.0, "whole", 1),
)

Event: TypeAlias = dict[str, Any]
Alignment: TypeAlias = dict[str, Any]
TimingData: TypeAlias = dict[str, Any]


class RhythmInterpretationError(RuntimeError):
    """Rhythm evidence could not be interpreted safely."""


@dataclass(frozen=True)
class RhythmInterpretationResult:
    version: str
    meter_candidates: tuple[dict[str, Any], ...]
    measures: tuple[dict[str, Any], ...]
    event_interpretations: tuple[dict[str, Any], ...]
    rest_candidates: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]


def interpret_rhythm(
    pitched_events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    percussion_events: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    alignment_candidates: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    timing: dict[str, Any],
    *,
    version: str = RHYTHM_INTERPRETATION_VERSION,
) -> RhythmInterpretationResult:
    """Return deterministic, conservative rhythm hypotheses for raw events."""
    _validate_version(version)

    pitched, percussion, event_index = _parse_events(
        pitched_events,
        percussion_events,
    )
    alignments = _parse_alignments(alignment_candidates, event_index)
    timing_data = _parse_timing(timing)
    _validate_alignment_grid(alignments, timing_data)

    meter_candidates = _build_meter_candidates(timing_data)
    measures = _build_measure_containers(timing_data, meter_candidates)
    next_onset_by_event = _next_pitched_onsets(pitched)
    measure_boundaries = {
        measure["endSeconds"] for measure in measures[:-1]
    }

    ordered_events = sorted(
        [*pitched, *percussion],
        key=lambda event: (
            event["onset"],
            event["event_type"] != "pitched",
            event["id"],
        ),
    )

    interpretations: list[dict[str, Any]] = []
    resolved_placement_count = 0
    ambiguous_duration_count = 0

    for position, event in enumerate(ordered_events, start=1):
        alignment = alignments.get(event["id"])
        placement_hypotheses, placement_resolved = _placement_hypotheses(
            event,
            alignment,
        )
        resolved_placement_count += int(placement_resolved)

        duration_hypotheses: list[dict[str, Any]] = []
        duration_resolved = event["event_type"] == "percussion"
        if event["event_type"] == "pitched":
            next_onset = next_onset_by_event.get((event["source_kind"], event["id"]))
            duration_hypotheses, duration_resolved = _duration_hypotheses(
                event,
                next_onset,
                timing_data,
            )
            ambiguous_duration_count += int(len(duration_hypotheses) > 1)

        continuation_hypotheses = _continuation_hypotheses(
            event,
            duration_hypotheses,
            alignment,
            timing_data,
            measure_boundaries,
        )
        raw_timing = _raw_timing(event)

        confidence_inputs = [
            hypothesis["confidence"]
            for hypothesis in placement_hypotheses
            if hypothesis["kind"] != "unresolved"
        ]
        if duration_hypotheses:
            confidence_inputs.append(duration_hypotheses[0]["confidence"])

        interpretations.append(
            {
                "id": f"rh{position:06d}",
                "eventId": event["id"],
                "sourceEventIds": [event["id"]],
                "eventType": event["event_type"],
                "sourceKind": event["source_kind"],
                "rawTiming": raw_timing,
                "placementHypotheses": placement_hypotheses,
                "durationHypotheses": duration_hypotheses,
                "continuationHypotheses": continuation_hypotheses,
                "unresolved": not placement_resolved or not duration_resolved,
                "confidence": _bounded_confidence(
                    min(confidence_inputs) if confidence_inputs else 0.0
                ),
                "warnings": [],
            }
        )

    rest_candidates = _rest_candidates(pitched, timing_data)
    unresolved_placement_count = len(interpretations) - resolved_placement_count
    warnings = _top_level_warnings(
        timing_data=timing_data,
        measures=measures,
        unresolved_placement_count=unresolved_placement_count,
        ambiguous_duration_count=ambiguous_duration_count,
    )
    diagnostics = {
        "pitchedEventCount": len(pitched),
        "percussionEventCount": len(percussion),
        "alignmentCandidateCount": len(alignments),
        "resolvedPlacementCount": resolved_placement_count,
        "unresolvedPlacementCount": unresolved_placement_count,
        "ambiguousDurationCount": ambiguous_duration_count,
        "meterCandidateCount": len(meter_candidates),
        "measureCount": len(measures),
        "restCandidateCount": len(rest_candidates),
        "placementConfidenceThreshold": _PLACEMENT_CONFIDENCE_THRESHOLD,
        "durationConfidenceThreshold": _DURATION_CONFIDENCE_THRESHOLD,
        "rawTimingAuthoritative": True,
        "timingMode": (
            "measured"
            if measures
            else "beat_relative"
            if timing_data["beats"]
            else "absolute_time"
        ),
    }

    result = RhythmInterpretationResult(
        version=version,
        meter_candidates=tuple(meter_candidates),
        measures=tuple(measures),
        event_interpretations=tuple(interpretations),
        rest_candidates=tuple(rest_candidates),
        warnings=tuple(warnings),
        diagnostics=diagnostics,
    )
    try:
        json.dumps(result.__dict__, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RhythmInterpretationError("output") from exc
    return result


def _validate_version(version: object) -> None:
    if not isinstance(version, str) or _VERSION_PATTERN.fullmatch(version) is None:
        raise RhythmInterpretationError("version")


def _parse_events(
    pitched_values: object,
    percussion_values: object,
) -> tuple[list[Event], list[Event], dict[str, Event]]:
    if not isinstance(pitched_values, (list, tuple)) or not isinstance(
        percussion_values, (list, tuple)
    ):
        raise RhythmInterpretationError("events")
    if len(pitched_values) + len(percussion_values) > _MAX_EVENTS:
        raise RhythmInterpretationError("events")

    event_index: dict[str, Event] = {}
    pitched: list[Event] = []
    percussion: list[Event] = []

    for value in pitched_values:
        mapping = _event_mapping(value)
        event_id = _event_id(mapping, event_index)
        source_kind = _source_kind(mapping)
        start_seconds = _finite_number(mapping.get("startSeconds"), minimum=0.0)
        end_seconds = _finite_number(mapping.get("endSeconds"), minimum=0.0)
        if end_seconds <= start_seconds:
            raise RhythmInterpretationError("range")
        event = {
            "id": event_id,
            "event_type": "pitched",
            "source_kind": source_kind,
            "start": start_seconds,
            "end": end_seconds,
            "onset": start_seconds,
            "confidence": _input_confidence(
                mapping.get("confidence"),
                default=_DEFAULT_EVENT_CONFIDENCE,
                label="event confidence",
            ),
        }
        pitched.append(event)
        event_index[event_id] = event

    for value in percussion_values:
        mapping = _event_mapping(value)
        event_id = _event_id(mapping, event_index)
        source_kind = _source_kind(mapping)
        onset_seconds = _finite_number(mapping.get("timeSeconds"), minimum=0.0)

        if "strength" in mapping:
            _input_confidence(
                mapping.get("strength"),
                default=_DEFAULT_EVENT_CONFIDENCE,
                label="event strength",
            )
        confidence_value = mapping.get("confidence", mapping.get("strength"))
        event = {
            "id": event_id,
            "event_type": "percussion",
            "source_kind": source_kind,
            "onset": onset_seconds,
            "confidence": _input_confidence(
                confidence_value,
                default=_DEFAULT_EVENT_CONFIDENCE,
                label="event confidence",
            ),
        }
        percussion.append(event)
        event_index[event_id] = event

    return pitched, percussion, event_index


def _event_mapping(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RhythmInterpretationError("mapping")
    return value


def _event_id(mapping: dict[str, Any], event_index: dict[str, Event]) -> str:
    value = mapping.get("id")
    if (
        not isinstance(value, str)
        or _EVENT_ID_PATTERN.fullmatch(value) is None
        or "/" in value
        or "\\" in value
        or value.startswith(".")
    ):
        raise RhythmInterpretationError("id")
    if value in event_index:
        raise RhythmInterpretationError("unique")
    return value


def _source_kind(mapping: dict[str, Any]) -> str:
    value = mapping.get("sourceKind")
    if not isinstance(value, str) or _SOURCE_SLUG_PATTERN.fullmatch(value) is None:
        raise RhythmInterpretationError("slug")
    return value


def _parse_alignments(
    values: object,
    event_index: dict[str, Event],
) -> dict[str, Alignment]:
    if not isinstance(values, (list, tuple)) or len(values) > _MAX_ALIGNMENT_CANDIDATES:
        raise RhythmInterpretationError("alignment")

    alignments: dict[str, Alignment] = {}
    for value in values:
        if not isinstance(value, dict):
            raise RhythmInterpretationError("mapping")

        event_id = value.get("eventId")
        if not isinstance(event_id, str) or event_id not in event_index:
            raise RhythmInterpretationError("existing")
        if event_id in alignments:
            raise RhythmInterpretationError("at most one")

        event = event_index[event_id]
        if value.get("eventType") != event["event_type"]:
            raise RhythmInterpretationError("type")

        raw_time = _finite_number(value.get("rawTimeSeconds"), minimum=0.0)
        if raw_time != event["onset"]:
            raise RhythmInterpretationError("preserve")

        alignment: Alignment = {
            "confidence": _input_confidence(
                value.get("confidence"),
                default=_DEFAULT_ALIGNMENT_CONFIDENCE,
                label="alignment confidence",
            )
        }
        _validate_warnings(value.get("warnings", []))

        has_aligned_time = "alignedTimeSeconds" in value
        present_grid_fields = {
            "beatIndex",
            "subdivision",
            "subdivisionIndex",
        }.intersection(value)
        if has_aligned_time:
            if "offsetSeconds" not in value or len(present_grid_fields) != 3:
                raise RhythmInterpretationError("partial")
            aligned_time = _finite_number(
                value["alignedTimeSeconds"],
                minimum=0.0,
            )
            offset = _finite_number(value["offsetSeconds"])
            if not math.isclose(raw_time - aligned_time, offset, abs_tol=1e-9):
                raise RhythmInterpretationError("offset")
            alignment.update(
                alignedTimeSeconds=aligned_time,
                offsetSeconds=offset,
                beatIndex=_integer(value["beatIndex"], minimum=0),
                subdivision=value["subdivision"],
                subdivisionIndex=_integer(value["subdivisionIndex"], minimum=0),
            )
        elif present_grid_fields or "offsetSeconds" in value:
            raise RhythmInterpretationError("partial")

        has_measure_index = "measureIndex" in value
        has_beat_in_measure = "beatInMeasure" in value
        if has_measure_index != has_beat_in_measure:
            raise RhythmInterpretationError("measure")
        if has_measure_index:
            alignment.update(
                measureIndex=_integer(value["measureIndex"], minimum=0),
                beatInMeasure=_integer(value["beatInMeasure"], minimum=1),
            )

        alignments[event_id] = alignment

    return alignments


def _parse_timing(value: object) -> TimingData:
    if not isinstance(value, dict):
        raise RhythmInterpretationError("timing")

    meter = value.get("meter")
    if meter is not None:
        meter = _integer(meter, minimum=2, maximum=12)

    tempo_stable = value.get("tempoStable")
    if tempo_stable is not None and type(tempo_stable) is not bool:
        raise RhythmInterpretationError("tempo")

    return {
        "beats": _timing_points(value.get("beatsSeconds", [])),
        "downbeats": _timing_points(value.get("downbeatsSeconds", [])),
        "meter": meter,
        "meter_confidence": _optional_input_confidence(
            value.get("meterConfidence"),
            label="meter confidence",
        ),
        "beat_confidence": _optional_input_confidence(
            value.get("beatConfidence"),
            label="beat confidence",
        ),
        "tempo_confidence": _optional_input_confidence(
            value.get("tempoConfidence"),
            label="tempo confidence",
        ),
        "tempo_stable": tempo_stable,
    }


def _validate_alignment_grid(
    alignments: dict[str, Alignment],
    timing: TimingData,
) -> None:
    beats: tuple[float, ...] = timing["beats"]
    for alignment in alignments.values():
        if "alignedTimeSeconds" not in alignment:
            continue

        beat_index = alignment["beatIndex"]
        if beat_index >= len(beats):
            raise RhythmInterpretationError("beatIndex")

        subdivision = alignment["subdivision"]
        subdivision_index = alignment["subdivisionIndex"]
        if isinstance(subdivision, int):
            if subdivision < 1 or subdivision_index >= subdivision:
                raise RhythmInterpretationError("subdivision")
            if beat_index == len(beats) - 1:
                expected_time = beats[beat_index]
            else:
                expected_time = beats[beat_index] + (
                    beats[beat_index + 1] - beats[beat_index]
                ) * subdivision_index / subdivision
            if not math.isclose(
                alignment["alignedTimeSeconds"],
                expected_time,
                abs_tol=1e-7,
            ):
                raise RhythmInterpretationError("grid placement")

        if "measureIndex" in alignment:
            if not timing["downbeats"]:
                raise RhythmInterpretationError("downbeat")
            if not timing["meter"]:
                raise RhythmInterpretationError("measure")


def _build_meter_candidates(timing: TimingData) -> list[dict[str, Any]]:
    meter = timing["meter"]
    if meter is None:
        return []

    confidence = timing["meter_confidence"]
    if confidence is None:
        confidence = _DEFAULT_METER_CONFIDENCE
    return [
        {
            "meter": meter,
            "confidence": confidence,
            "evidence": ["timing_meter"],
            "resolved": confidence >= _PLACEMENT_CONFIDENCE_THRESHOLD,
            "warnings": (
                []
                if confidence >= _PLACEMENT_CONFIDENCE_THRESHOLD
                else ["Meter evidence is weak."]
            ),
        }
    ]


def _build_measure_containers(
    timing: TimingData,
    meter_candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if (
        not meter_candidates
        or not meter_candidates[0]["resolved"]
        or len(timing["downbeats"]) < 2
    ):
        return []

    meter = meter_candidates[0]["meter"]
    beat_lookup = {
        round(beat, 9): index for index, beat in enumerate(timing["beats"])
    }
    measures: list[dict[str, Any]] = []
    for start, end in zip(timing["downbeats"], timing["downbeats"][1:]):
        start_index = beat_lookup.get(round(start, 9))
        end_index = beat_lookup.get(round(end, 9))
        if (
            start_index is not None
            and end_index is not None
            and end_index - start_index == meter
        ):
            measures.append(
                {
                    "id": f"m{len(measures) + 1:06d}",
                    "index": len(measures),
                    "startSeconds": start,
                    "endSeconds": end,
                    "meter": meter,
                    "beatIndices": list(range(start_index, end_index)),
                    "confidence": meter_candidates[0]["confidence"],
                    "evidence": ["observed_downbeats", "observed_beats"],
                    "warnings": [],
                }
            )
    return measures


def _placement_hypotheses(
    event: Event,
    alignment: Alignment | None,
) -> tuple[list[dict[str, Any]], bool]:
    raw_time = event["onset"]
    if alignment is None or "alignedTimeSeconds" not in alignment:
        confidence = alignment["confidence"] if alignment is not None else 0.0
        return [
            {
                "kind": "unresolved",
                "rawTimeSeconds": raw_time,
                "confidence": confidence,
            }
        ], False

    resolved = alignment["confidence"] >= _PLACEMENT_CONFIDENCE_THRESHOLD
    grid = {
        "kind": "grid",
        "status": "resolved" if resolved else "uncertain",
        "rawTimeSeconds": raw_time,
        "alignedTimeSeconds": alignment["alignedTimeSeconds"],
        "offsetSeconds": alignment["offsetSeconds"],
        "beatIndex": alignment["beatIndex"],
        "subdivision": alignment["subdivision"],
        "subdivisionIndex": alignment["subdivisionIndex"],
        "confidence": alignment["confidence"],
    }
    if resolved:
        return [grid], True
    return [
        grid,
        {
            "kind": "unresolved",
            "rawTimeSeconds": raw_time,
            "confidence": _bounded_confidence(1.0 - alignment["confidence"]),
        },
    ], False


def _duration_hypotheses(
    event: Event,
    next_onset: float | None,
    timing: TimingData,
) -> tuple[list[dict[str, Any]], bool]:
    raw_duration = event["end"] - event["start"]
    beats: tuple[float, ...] = timing["beats"]
    if len(beats) < 2:
        return [
            {
                "kind": "absolute_duration",
                "durationSeconds": _rounded(raw_duration),
                "confidence": min(event["confidence"], 0.35),
                "unresolved": True,
            }
        ], False

    local_beat_seconds = min(
        (
            abs((beats[index] + beats[index + 1]) / 2 - event["start"]),
            beats[index + 1] - beats[index],
        )
        for index in range(len(beats) - 1)
    )[1]
    evidence_duration = raw_duration
    if next_onset is not None and next_onset > event["start"]:
        evidence_duration = min(raw_duration, next_onset - event["start"])
    duration_beats = evidence_duration / local_beat_seconds

    beat_confidence = timing["beat_confidence"] or _DEFAULT_BEAT_CONFIDENCE_FOR_DURATION
    candidates: list[dict[str, Any]] = []
    for candidate_beats, label, subdivision in _DURATION_CANDIDATES:
        confidence = (
            0.65
            * max(
                0.0,
                1.0
                - abs(candidate_beats - duration_beats)
                / max(0.25, candidate_beats),
            )
            + 0.20 * event["confidence"]
            + 0.15 * beat_confidence
        )
        if timing["tempo_stable"] is False:
            confidence = min(confidence, _UNSTABLE_TEMPO_CONFIDENCE_CAP)
        candidates.append(
            {
                "kind": "grid_duration",
                "label": label,
                "durationBeats": candidate_beats,
                "durationSeconds": candidate_beats * local_beat_seconds,
                "subdivision": subdivision,
                "confidence": _bounded_confidence(confidence),
                "rawDurationSeconds": _rounded(raw_duration),
                "warnings": [],
            }
        )

    candidates.sort(
        key=lambda candidate: (
            -candidate["confidence"],
            abs(candidate["durationBeats"] - duration_beats),
        )
    )
    best_confidence = candidates[0]["confidence"]
    selected = [
        candidate
        for candidate in candidates
        if candidate["confidence"] >= best_confidence - _DURATION_SELECTION_WINDOW
    ][:_MAX_DURATION_ALTERNATIVES]
    if len(selected) == 1 and best_confidence < 0.80:
        selected.append(candidates[1])

    resolved = best_confidence >= _DURATION_CONFIDENCE_THRESHOLD and not (
        len(selected) > 1
        and best_confidence - selected[1]["confidence"] < _DURATION_AMBIGUITY_MARGIN
    )
    return selected, resolved


def _continuation_hypotheses(
    event: Event,
    duration_hypotheses: list[dict[str, Any]],
    alignment: Alignment | None,
    timing: TimingData,
    measure_boundaries: set[float],
) -> list[dict[str, Any]]:
    if event["event_type"] != "pitched":
        return []

    duration_confidence = (
        duration_hypotheses[0]["confidence"] if duration_hypotheses else 0.0
    )
    alignment_confidence = alignment["confidence"] if alignment is not None else 0.25
    confidence = _bounded_confidence(
        min(duration_confidence, alignment_confidence)
    )
    boundaries = sorted(set(timing["beats"]) | measure_boundaries)
    return [
        {
            "kind": "tie_or_continuation",
            "boundaryType": (
                "measure" if boundary in measure_boundaries else "beat"
            ),
            "boundaryTimeSeconds": boundary,
            "confidence": confidence,
            "resolved": False,
            "warnings": ["Continuation is provisional."],
        }
        for boundary in boundaries
        if event["start"] < boundary < event["end"]
    ][:_MAX_CONTINUATION_HYPOTHESES]


def _raw_timing(event: Event) -> dict[str, float]:
    if event["event_type"] == "pitched":
        return {
            "startSeconds": event["start"],
            "endSeconds": event["end"],
            "durationSeconds": _rounded(event["end"] - event["start"]),
        }
    return {"timeSeconds": event["onset"]}


def _next_pitched_onsets(
    pitched_events: list[Event],
) -> dict[tuple[str, str], float | None]:
    by_source: dict[str, list[Event]] = {}
    for event in pitched_events:
        by_source.setdefault(event["source_kind"], []).append(event)

    result: dict[tuple[str, str], float | None] = {}
    for source_kind, events in by_source.items():
        events.sort(key=lambda event: event["start"])
        for index, event in enumerate(events):
            result[(source_kind, event["id"])] = (
                events[index + 1]["start"] if index + 1 < len(events) else None
            )
    return result


def _rest_candidates(
    pitched_events: list[Event],
    timing: TimingData,
) -> list[dict[str, Any]]:
    by_source: dict[str, list[Event]] = {}
    for event in pitched_events:
        by_source.setdefault(event["source_kind"], []).append(event)

    rests: list[dict[str, Any]] = []
    for source_kind, events in by_source.items():
        events.sort(key=lambda event: event["start"])
        for previous, following in zip(events, events[1:]):
            gap_start = previous["end"]
            gap_end = following["start"]
            gap_seconds = gap_end - gap_start
            if gap_seconds <= 0:
                continue

            beat_seconds = (
                timing["beats"][1] - timing["beats"][0]
                if len(timing["beats"]) > 1
                else None
            )
            if (
                beat_seconds is not None
                and gap_seconds / beat_seconds < _PLACEMENT_CONFIDENCE_THRESHOLD
            ) or (beat_seconds is None and gap_seconds < 0.5):
                continue

            confidence = 0.65 * min(
                previous["confidence"],
                following["confidence"],
            )
            if beat_seconds is None:
                duration_hypotheses = [
                    {
                        "kind": "absolute_duration",
                        "durationSeconds": _rounded(gap_seconds),
                        "confidence": min(confidence, 0.35),
                        "unresolved": True,
                    }
                ]
            else:
                duration_beats = gap_seconds / beat_seconds
                nearest = sorted(
                    _DURATION_CANDIDATES,
                    key=lambda candidate: abs(candidate[0] - duration_beats),
                )[:2]
                duration_hypotheses = [
                    {
                        "kind": "grid_duration",
                        "label": label,
                        "durationBeats": candidate_beats,
                        "durationSeconds": candidate_beats * beat_seconds,
                        "subdivision": subdivision,
                        "confidence": _bounded_confidence(
                            confidence
                            * max(
                                0.0,
                                1.0
                                - abs(candidate_beats - duration_beats)
                                / max(0.25, candidate_beats),
                            )
                        ),
                    }
                    for candidate_beats, label, subdivision in nearest
                ]

            rest = {
                "id": f"rest{len(rests) + 1:06d}",
                "sourceKind": source_kind,
                "afterEventId": previous["id"],
                "beforeEventId": following["id"],
                "sourceEventIds": [previous["id"], following["id"]],
                "rawGap": {
                    "startSeconds": gap_start,
                    "endSeconds": gap_end,
                    "durationSeconds": _rounded(gap_seconds),
                },
                "durationHypotheses": duration_hypotheses,
                "confidence": _bounded_confidence(confidence),
                "resolved": confidence >= _DURATION_CONFIDENCE_THRESHOLD,
                "warnings": [],
            }
            if source_kind in {"full_mix", "other"}:
                rest.update(
                    confidence=min(
                        rest["confidence"],
                        _UNCERTAIN_SOURCE_REST_CONFIDENCE_CAP,
                    ),
                    resolved=False,
                    warnings=["Source coverage is uncertain."],
                )
            rests.append(rest)
    return rests


def _top_level_warnings(
    *,
    timing_data: TimingData,
    measures: list[dict[str, Any]],
    unresolved_placement_count: int,
    ambiguous_duration_count: int,
) -> list[str]:
    warnings: list[str] = []
    if not timing_data["beats"]:
        warnings.append("Beat evidence is unavailable; raw time is preserved.")
    if not measures:
        warnings.append("Measures remain unresolved.")
    if unresolved_placement_count:
        warnings.append(
            f"{unresolved_placement_count} event placement(s) remain unresolved."
        )
    if ambiguous_duration_count:
        warnings.append(
            f"{ambiguous_duration_count} pitched event(s) retain duration alternatives."
        )
    return warnings[:_MAX_TOP_LEVEL_WARNINGS]


def _finite_number(value: object, minimum: float | None = None) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or (minimum is not None and value < minimum)
    ):
        raise RhythmInterpretationError("number")
    return float(value)


def _integer(
    value: object,
    minimum: int = 0,
    maximum: int | None = None,
) -> int:
    if (
        type(value) is not int
        or value < minimum
        or (maximum is not None and value > maximum)
    ):
        raise RhythmInterpretationError("integer")
    return value


def _input_confidence(
    value: object,
    *,
    default: float,
    label: str,
) -> float:
    if value is None:
        return default
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RhythmInterpretationError(label)
    number = float(value)
    if not 0.0 <= number <= 1.0:
        raise RhythmInterpretationError(label)
    return number


def _optional_input_confidence(value: object, *, label: str) -> float | None:
    if value is None:
        return None
    return _input_confidence(value, default=0.0, label=label)


def _timing_points(value: object) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_TIMING_POINTS:
        raise RhythmInterpretationError("timing")
    points = tuple(_finite_number(point, minimum=0.0) for point in value)
    if any(following <= previous for previous, following in zip(points, points[1:])):
        raise RhythmInterpretationError("timing")
    return points


def _validate_warnings(value: object) -> None:
    if not isinstance(value, (list, tuple)) or len(value) > _MAX_WARNINGS_PER_ALIGNMENT:
        raise RhythmInterpretationError("warnings")
    for warning in value:
        if (
            not isinstance(warning, str)
            or not warning
            or len(warning) > _MAX_WARNING_LENGTH
            or _UNSAFE_TEXT_PATTERN.search(warning)
        ):
            raise RhythmInterpretationError("paths")


def _bounded_confidence(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _rounded(value: float) -> float:
    return round(float(value), 12)
