from __future__ import annotations

import copy
import json
import math
from typing import Any

import pytest

from app.rhythm_interpretation import (
    RHYTHM_INTERPRETATION_VERSION,
    RhythmInterpretationError,
    RhythmInterpretationResult,
    interpret_rhythm,
)


INVALID_CONFIDENCES = (
    -0.01,
    1.01,
    True,
    False,
    math.nan,
    math.inf,
    -math.inf,
)


def timing(**overrides: Any) -> dict[str, Any]:
    value: dict[str, Any] = {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "beatConfidence": 0.9,
        "downbeatsSeconds": [0.0, 2.0, 4.0],
        "meter": 4,
        "meterConfidence": 0.85,
    }
    value.update(overrides)
    return value


def pitched(
    event_id: str,
    start: float,
    end: float,
    source: str = "vocals",
    confidence: object = 0.9,
) -> dict[str, Any]:
    return {
        "id": event_id,
        "sourceKind": source,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": 69,
        "midiPitch": 69.0,
        "frequencyHz": 440.0,
        "noteName": "A4",
        "confidence": confidence,
        "warnings": [],
    }


def percussion(
    event_id: str,
    onset: float,
    *,
    strength: object = 0.9,
    confidence: object | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "id": event_id,
        "sourceKind": "drums",
        "timeSeconds": onset,
        "strength": strength,
        "hits": [{"kind": "kick", "confidence": 0.8}],
    }
    if confidence is not None:
        value["confidence"] = confidence
    return value


def alignment(
    event_id: str,
    event_type: str,
    raw_time: float,
    aligned_time: float | None = None,
    confidence: object = 0.9,
    **overrides: Any,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "eventId": event_id,
        "eventType": event_type,
        "rawTimeSeconds": raw_time,
        "confidence": confidence,
        "warnings": [],
    }
    if aligned_time is not None:
        value.update(
            alignedTimeSeconds=aligned_time,
            offsetSeconds=raw_time - aligned_time,
            beatIndex=round(aligned_time / 0.5),
            subdivision=1,
            subdivisionIndex=0,
        )
    value.update(overrides)
    return value


def event_item(
    result: RhythmInterpretationResult,
    event_id: str,
) -> dict[str, Any]:
    return next(
        item
        for item in result.event_interpretations
        if item["eventId"] == event_id
    )


def test_quarter_duration_and_raw_time() -> None:
    events = [
        pitched("p1", 0.0, 0.5),
        pitched("p2", 0.5, 1.0),
        pitched("p3", 1.0, 1.5),
    ]
    result = interpret_rhythm(
        events,
        [],
        [
            alignment(
                event["id"],
                "pitched",
                event["startSeconds"],
                event["startSeconds"],
            )
            for event in events
        ],
        timing(),
    )

    assert result.version == RHYTHM_INTERPRETATION_VERSION
    assert all(
        item["durationHypotheses"][0]["label"] == "quarter"
        and item["durationHypotheses"][0]["confidence"] >= 0.75
        for item in result.event_interpretations
    )
    assert event_item(result, "p1")["rawTiming"]["durationSeconds"] == 0.5


def test_uncertain_grid_retains_unresolved_alternative() -> None:
    result = interpret_rhythm(
        [pitched("p1", 0.137, 0.48)],
        [],
        [
            alignment(
                "p1",
                "pitched",
                0.137,
                0.125,
                0.42,
                subdivision=4,
                subdivisionIndex=1,
            )
        ],
        timing(),
    )
    item = event_item(result, "p1")

    assert item["rawTiming"]["startSeconds"] == 0.137
    assert item["placementHypotheses"][0]["status"] == "uncertain"
    assert item["placementHypotheses"][1]["kind"] == "unresolved"
    assert item["unresolved"] is True


def test_aligned_and_unaligned_remain_distinct() -> None:
    result = interpret_rhythm(
        [pitched("p1", 0.0, 0.5), pitched("p2", 0.73, 1.1)],
        [],
        [
            alignment("p1", "pitched", 0.0, 0.0),
            alignment("p2", "pitched", 0.73, None, 0.0),
        ],
        timing(),
    )

    assert event_item(result, "p1")["placementHypotheses"][0]["kind"] == "grid"
    assert (
        event_item(result, "p2")["placementHypotheses"][0]["kind"]
        == "unresolved"
    )


def test_measure_crossing_continuation() -> None:
    result = interpret_rhythm(
        [pitched("p1", 1.75, 2.25)],
        [],
        [
            alignment(
                "p1",
                "pitched",
                1.75,
                1.75,
                beatIndex=3,
                subdivision=2,
                subdivisionIndex=1,
            )
        ],
        timing(),
    )

    assert any(
        item["boundaryType"] == "measure"
        and item["boundaryTimeSeconds"] == 2.0
        for item in event_item(result, "p1")["continuationHypotheses"]
    )


def test_gap_creates_rest_and_full_mix_is_provisional() -> None:
    result = interpret_rhythm(
        [pitched("p1", 0.0, 0.5), pitched("p2", 1.5, 2.0)],
        [],
        [],
        timing(),
    )
    rest = result.rest_candidates[0]
    assert rest["rawGap"]["durationSeconds"] == 1.0
    assert 1 <= len(rest["durationHypotheses"]) <= 2

    full_mix_result = interpret_rhythm(
        [
            pitched("p1", 0.0, 0.5, "full_mix"),
            pitched("p2", 1.5, 2.0, "full_mix"),
        ],
        [],
        [],
        timing(),
    )
    full_mix_rest = full_mix_result.rest_candidates[0]
    assert full_mix_rest["resolved"] is False
    assert full_mix_rest["confidence"] <= 0.45


def test_weak_meter_and_no_beats_fallbacks() -> None:
    weak_meter = interpret_rhythm(
        [pitched("p1", 0.5, 1.0)],
        [],
        [alignment("p1", "pitched", 0.5, 0.5)],
        timing(meterConfidence=0.2),
    )
    assert not weak_meter.measures
    assert weak_meter.diagnostics["timingMode"] == "beat_relative"

    no_beats = interpret_rhythm(
        [pitched("p1", 0.2, 0.7)],
        [],
        [],
        timing(
            beatsSeconds=[],
            downbeatsSeconds=[],
            meter=None,
            meterConfidence=None,
        ),
    )
    assert (
        event_item(no_beats, "p1")["durationHypotheses"][0]["kind"]
        == "absolute_duration"
    )
    assert no_beats.diagnostics["timingMode"] == "absolute_time"


def test_simultaneous_percussion_not_collapsed() -> None:
    result = interpret_rhythm(
        [],
        [percussion("r1", 0.5), percussion("r2", 0.5)],
        [
            alignment("r1", "percussion", 0.5, 0.5),
            alignment("r2", "percussion", 0.5, 0.5),
        ],
        timing(),
    )

    assert [item["eventId"] for item in result.event_interpretations] == [
        "r1",
        "r2",
    ]
    assert all(
        item["rawTiming"]["timeSeconds"] == 0.5
        for item in result.event_interpretations
    )


def test_irregular_timing_stays_ambiguous() -> None:
    timing_value = timing(
        beatsSeconds=[0.0, 0.41, 1.02, 1.43, 2.08],
        downbeatsSeconds=[],
        meter=None,
        meterConfidence=None,
        tempoStable=False,
        beatConfidence=0.55,
    )
    result = interpret_rhythm(
        [pitched("p1", 0.41, 0.92)],
        [],
        [alignment("p1", "pitched", 0.41, 0.41, 0.6)],
        timing_value,
    )
    item = event_item(result, "p1")

    assert len(item["durationHypotheses"]) >= 2
    assert item["durationHypotheses"][0]["confidence"] <= 0.58
    assert item["unresolved"] is True


def test_determinism_nonmutation_and_safe_output() -> None:
    args = (
        [pitched("p1", 0.0, 0.5)],
        [percussion("r1", 0.5)],
        [
            alignment("p1", "pitched", 0.0, 0.0),
            alignment("r1", "percussion", 0.5, 0.5),
        ],
        timing(),
    )
    before = copy.deepcopy(args)
    first = interpret_rhythm(*args)
    second = interpret_rhythm(*copy.deepcopy(args))
    assert args == before
    assert first == second
    json.dumps(first.__dict__, allow_nan=False)

    event = pitched("p1", 0.0, 0.5)
    event["privatePath"] = "/home/user/x.wav"
    encoded = json.dumps(
        interpret_rhythm([event], [], [], timing()).__dict__,
        allow_nan=False,
    )
    assert "privatePath" not in encoded


def test_shuffled_input_stable_ids() -> None:
    events = [pitched("p2", 0.5, 1.0), pitched("p1", 0.0, 0.5)]
    candidates = [
        alignment("p2", "pitched", 0.5, 0.5),
        alignment("p1", "pitched", 0.0, 0.0),
    ]
    first = interpret_rhythm(events, [], candidates, timing())
    second = interpret_rhythm(events[::-1], [], candidates[::-1], timing())

    assert first == second
    assert [item["id"] for item in first.event_interpretations] == [
        "rh000001",
        "rh000002",
    ]


@pytest.mark.parametrize("bad", ["", "bad id", "../p", "p/1", "p\\1", "x" * 129])
def test_bad_ids(bad: str) -> None:
    with pytest.raises(RhythmInterpretationError):
        interpret_rhythm([pitched(bad, 0.0, 0.5)], [], [], timing())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, ".5", -0.1])
def test_bad_times(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError):
        interpret_rhythm([pitched("p", bad, 0.5)], [], [], timing())


@pytest.mark.parametrize("bad", ["Vocals", "bad source", "../x", "x/y"])
def test_bad_slugs(bad: str) -> None:
    with pytest.raises(RhythmInterpretationError):
        interpret_rhythm([pitched("p", 0.0, 0.5, bad)], [], [], timing())


@pytest.mark.parametrize(
    "beats",
    [[0.0, 0.5, 0.5], [0.0, 1.0, 0.5], [0.0, math.nan], [0.0, True], "bad"],
)
def test_bad_timing(beats: object) -> None:
    with pytest.raises(RhythmInterpretationError):
        interpret_rhythm([], [], [], timing(beatsSeconds=beats))


def test_duplicate_and_invalid_references() -> None:
    with pytest.raises(RhythmInterpretationError, match="unique"):
        interpret_rhythm(
            [pitched("x", 0.0, 0.5)],
            [percussion("x", 0.5)],
            [],
            timing(),
        )
    with pytest.raises(RhythmInterpretationError, match="existing"):
        interpret_rhythm(
            [],
            [],
            [alignment("x", "pitched", 0.0, 0.0)],
            timing(),
        )
    candidate = alignment("p", "pitched", 0.0, 0.0)
    with pytest.raises(RhythmInterpretationError, match="at most one"):
        interpret_rhythm(
            [pitched("p", 0.0, 0.5)],
            [],
            [candidate, candidate],
            timing(),
        )


def test_alignment_integrity_and_path_safety() -> None:
    with pytest.raises(RhythmInterpretationError, match="preserve"):
        interpret_rhythm(
            [pitched("p", 0.0, 0.5)],
            [],
            [alignment("p", "pitched", 0.1, 0.0)],
            timing(),
        )

    out_of_range = alignment("p", "pitched", 0.0, 0.0)
    out_of_range["beatIndex"] = 999
    with pytest.raises(RhythmInterpretationError, match="beatIndex"):
        interpret_rhythm(
            [pitched("p", 0.0, 0.5)],
            [],
            [out_of_range],
            timing(),
        )

    bad_grid = alignment(
        "p",
        "pitched",
        0.13,
        0.13,
        subdivision=4,
        subdivisionIndex=1,
    )
    with pytest.raises(RhythmInterpretationError, match="grid placement"):
        interpret_rhythm(
            [pitched("p", 0.13, 0.5)],
            [],
            [bad_grid],
            timing(),
        )

    path_warning = alignment("p", "pitched", 0.0, 0.0)
    path_warning["warnings"] = ["at /home/user/x.wav"]
    with pytest.raises(RhythmInterpretationError, match="paths"):
        interpret_rhythm(
            [pitched("p", 0.0, 0.5)],
            [],
            [path_warning],
            timing(),
        )


def test_measure_needs_downbeats_and_bounds() -> None:
    candidate = alignment(
        "p",
        "pitched",
        0.5,
        0.5,
        measureIndex=0,
        beatInMeasure=2,
    )
    with pytest.raises(RhythmInterpretationError, match="downbeat"):
        interpret_rhythm(
            [pitched("p", 0.5, 1.0)],
            [],
            [candidate],
            timing(downbeatsSeconds=[]),
        )

    events = [
        pitched(f"p{index}", index * 0.51, index * 0.51 + 0.31)
        for index in range(20)
    ]
    result = interpret_rhythm(events, [], [], timing(tempoStable=False))
    assert len(result.warnings) <= 32
    assert all(
        len(item["durationHypotheses"]) <= 3
        for item in result.event_interpretations
    )


def test_pitched_overlap_and_drum_assignment_shapes_remain_composable() -> None:
    from app.percussion_interpretation import interpret_percussion
    from app.pitched_part_inference import infer_pitched_parts

    pitched_events = [
        pitched("overlap-a", 0.0, 0.75),
        pitched("overlap-b", 0.25, 1.0),
    ]
    percussion_events = [
        {
            "id": "drum-event",
            "sourceKind": "drums",
            "timeSeconds": 0.5,
            "strength": 0.9,
            "hits": [
                {"kind": "kick", "confidence": 0.9},
                {"kind": "closed_hihat", "confidence": 0.8},
            ],
            "warnings": ["Two broad hit families remain simultaneous."],
            "rawFeatureSummary": {},
        }
    ]
    alignments = [
        alignment("overlap-a", "pitched", 0.0, 0.0),
        alignment(
            "overlap-b",
            "pitched",
            0.25,
            0.25,
            subdivision=2,
            subdivisionIndex=1,
        ),
        alignment("drum-event", "percussion", 0.5, 0.5),
    ]

    rhythm = interpret_rhythm(
        pitched_events,
        percussion_events,
        alignments,
        timing(),
    )
    parts = infer_pitched_parts(pitched_events, alignments[:2])
    drums = interpret_percussion(percussion_events, alignments[2:])

    assert {item["eventId"] for item in rhythm.event_interpretations} == {
        "overlap-a",
        "overlap-b",
        "drum-event",
    }
    assert len(parts.assignments) == 2
    assert len({item["eventId"] for item in parts.assignments}) == 2
    assert len(drums.assignments) == 2
    assert {item["eventId"] for item in drums.assignments} == {"drum-event"}
    placement = event_item(rhythm, "drum-event")["placementHypotheses"]
    assert all(item["rawTimeSeconds"] == 0.5 for item in drums.assignments)
    assert placement[0]["alignedTimeSeconds"] == 0.5


def test_result_is_directly_retainable_without_discarding_uncertainty() -> None:
    result = interpret_rhythm(
        [pitched("p1", 0.137, 0.48)],
        [],
        [
            alignment(
                "p1",
                "pitched",
                0.137,
                0.125,
                0.42,
                subdivision=4,
                subdivisionIndex=1,
                warnings=["Timing evidence remains uncertain."],
            )
        ],
        timing(),
    )
    item = event_item(result, "p1")
    retained = {
        "sourceEventIds": item["sourceEventIds"],
        "rawTiming": item["rawTiming"],
        "placementHypotheses": item["placementHypotheses"],
        "durationHypotheses": item["durationHypotheses"],
        "continuationHypotheses": item["continuationHypotheses"],
        "warnings": item["warnings"],
        "unresolved": item["unresolved"],
    }

    assert retained["sourceEventIds"] == ["p1"]
    assert retained["rawTiming"]["startSeconds"] == 0.137
    assert len(retained["placementHypotheses"]) == 2
    assert retained["unresolved"] is True
    json.dumps(retained, allow_nan=False)


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_pitched_event_confidence_must_be_in_closed_unit_interval(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError, match="event confidence"):
        interpret_rhythm(
            [pitched("p1", 0.0, 0.5, confidence=bad)],
            [],
            [],
            timing(),
        )


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_percussion_event_confidence_must_be_in_closed_unit_interval(
    bad: object,
) -> None:
    with pytest.raises(RhythmInterpretationError, match="event confidence"):
        interpret_rhythm(
            [],
            [percussion("r1", 0.0, confidence=bad)],
            [],
            timing(),
        )


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_percussion_strength_used_as_confidence_must_be_in_closed_unit_interval(
    bad: object,
) -> None:
    with pytest.raises(RhythmInterpretationError, match="event strength"):
        interpret_rhythm(
            [],
            [percussion("r1", 0.0, strength=bad)],
            [],
            timing(),
        )


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_alignment_confidence_must_be_in_closed_unit_interval(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError, match="alignment confidence"):
        interpret_rhythm(
            [pitched("p1", 0.0, 0.5)],
            [],
            [alignment("p1", "pitched", 0.0, 0.0, bad)],
            timing(),
        )


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_beat_confidence_must_be_in_closed_unit_interval(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError, match="beat confidence"):
        interpret_rhythm([], [], [], timing(beatConfidence=bad))


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_tempo_confidence_must_be_in_closed_unit_interval(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError, match="tempo confidence"):
        interpret_rhythm([], [], [], timing(tempoConfidence=bad))


@pytest.mark.parametrize("bad", INVALID_CONFIDENCES)
def test_meter_confidence_must_be_in_closed_unit_interval(bad: object) -> None:
    with pytest.raises(RhythmInterpretationError, match="meter confidence"):
        interpret_rhythm([], [], [], timing(meterConfidence=bad))


def test_confidence_endpoints_are_accepted_at_every_input_location() -> None:
    result = interpret_rhythm(
        [pitched("p1", 0.0, 0.5, confidence=0.0)],
        [percussion("r1", 0.5, strength=1.0, confidence=1.0)],
        [
            alignment("p1", "pitched", 0.0, 0.0, 0.0),
            alignment("r1", "percussion", 0.5, 0.5, 1.0),
        ],
        timing(
            beatConfidence=0.0,
            tempoConfidence=1.0,
            meterConfidence=1.0,
        ),
    )

    assert event_item(result, "p1")["placementHypotheses"][0]["confidence"] == 0.0
    assert event_item(result, "r1")["placementHypotheses"][0]["confidence"] == 1.0
    assert result.meter_candidates[0]["confidence"] == 1.0
