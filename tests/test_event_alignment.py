from __future__ import annotations

import copy
import json
import math

import pytest

from app.event_alignment import (
    ALIGNMENT_VERSION,
    EventAlignmentError,
    align_raw_events_to_timing,
)


def timing(**overrides):
    value = {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "beatConfidence": 0.9,
        "downbeatsSeconds": [0.0, 2.0],
        "meter": 4,
        "meterConfidence": 0.8,
    }
    value.update(overrides)
    return value


def pitched(event_id="p000001", start=0.0, end=None):
    if end is None and isinstance(start, (int, float)) and not isinstance(start, bool):
        end = float(start) + 0.2
    return {
        "id": event_id,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": 69,
    }


def percussion(event_id="r000001", onset=0.0):
    return {
        "id": event_id,
        "timeSeconds": onset,
        "hits": [{"kind": "kick"}],
    }


def candidate(result, event_id):
    return next(
        item for item in result["candidates"] if item["eventId"] == event_id
    )


def test_exact_beat_alignment_preserves_raw_time():
    result = align_raw_events_to_timing([pitched(start=1.0)], [], timing())
    item = result["candidates"][0]
    assert result["alignmentVersion"] == ALIGNMENT_VERSION
    assert item["rawTimeSeconds"] == 1.0
    assert item["alignedTimeSeconds"] == 1.0
    assert item["offsetSeconds"] == 0.0
    assert item["beatIndex"] == 2
    assert item["subdivision"] == 1
    assert item["subdivisionIndex"] == 0
    assert 0 <= item["confidence"] < 1


def test_eighth_triplet_and_sixteenth_positions_align_canonically():
    events = [
        percussion("r-eighth", 0.25),
        percussion("r-triplet", 0.5 / 3),
        percussion("r-sixteenth", 0.125),
    ]
    result = align_raw_events_to_timing([], events, timing())
    eighth = candidate(result, "r-eighth")
    triplet = candidate(result, "r-triplet")
    sixteenth = candidate(result, "r-sixteenth")
    assert (eighth["subdivision"], eighth["subdivisionIndex"]) == (2, 1)
    assert (triplet["subdivision"], triplet["subdivisionIndex"]) == (3, 1)
    assert (sixteenth["subdivision"], sixteenth["subdivisionIndex"]) == (4, 1)


def test_max_subdivision_limits_grid():
    result = align_raw_events_to_timing(
        [],
        [percussion(onset=0.125)],
        timing(),
        max_subdivision=2,
    )
    assert "alignedTimeSeconds" not in result["candidates"][0]


def test_event_outside_window_stays_unaligned_and_raw():
    result = align_raw_events_to_timing(
        [pitched(start=2.4, end=2.6)],
        [],
        timing(),
        max_offset_seconds=0.05,
    )
    item = result["candidates"][0]
    assert item["rawTimeSeconds"] == 2.4
    assert item["confidence"] == 0.0
    assert "alignedTimeSeconds" not in item
    assert "beatIndex" not in item
    assert item["warnings"]


def test_simultaneous_percussion_hits_remain_separate_candidates():
    result = align_raw_events_to_timing(
        [],
        [percussion("r000001", 0.5), percussion("r000002", 0.5)],
        timing(),
    )
    assert [item["eventId"] for item in result["candidates"]] == [
        "r000001",
        "r000002",
    ]
    assert all(item["rawTimeSeconds"] == 0.5 for item in result["candidates"])


def test_weak_beat_confidence_lowers_alignment_confidence():
    strong = align_raw_events_to_timing(
        [pitched(start=0.5)],
        [],
        timing(beatConfidence=0.95, tempoConfidence=0.95),
    )
    weak = align_raw_events_to_timing(
        [pitched(start=0.5)],
        [],
        timing(beatConfidence=0.1, tempoConfidence=0.1),
    )
    assert weak["candidates"][0]["confidence"] < strong["candidates"][0][
        "confidence"
    ]
    assert weak["warnings"]
    assert weak["candidates"][0]["warnings"]


def test_valid_downbeats_and_meter_add_measure_candidates():
    result = align_raw_events_to_timing([pitched(start=1.5)], [], timing())
    item = result["candidates"][0]
    assert item["measureIndex"] == 0
    assert item["beatInMeasure"] == 4
    assert result["diagnostics"]["measureEvidenceUsed"] is True


def test_low_meter_confidence_omits_measure_candidates():
    result = align_raw_events_to_timing(
        [pitched(start=1.5)],
        [],
        timing(meterConfidence=0.2),
    )
    item = result["candidates"][0]
    assert "measureIndex" not in item
    assert "beatInMeasure" not in item
    assert result["diagnostics"]["measureEvidenceUsed"] is False
    assert any("Meter confidence" in warning for warning in result["warnings"])


def test_missing_beats_returns_no_fabricated_alignment():
    result = align_raw_events_to_timing(
        [pitched(start=0.4)],
        [percussion(onset=0.8)],
        timing(beatsSeconds=[], downbeatsSeconds=[]),
    )
    assert len(result["candidates"]) == 2
    assert all(
        "alignedTimeSeconds" not in item for item in result["candidates"]
    )
    assert result["diagnostics"]["gridPointCount"] == 0
    assert result["warnings"]


def test_empty_events_are_allowed():
    result = align_raw_events_to_timing([], [], timing())
    assert result["candidates"] == []
    assert result["diagnostics"]["eventCount"] == 0


def test_irregular_intervals_use_local_spacing():
    evidence = timing(
        beatsSeconds=[0.0, 0.4, 1.0, 1.45],
        downbeatsSeconds=[],
        meter=None,
        meterConfidence=None,
    )
    result = align_raw_events_to_timing(
        [],
        [
            percussion("r-a", 0.2),
            percussion("r-b", 0.7),
            percussion("r-c", 1.225),
        ],
        evidence,
    )
    assert candidate(result, "r-a")["alignedTimeSeconds"] == 0.2
    assert candidate(result, "r-b")["alignedTimeSeconds"] == 0.7
    assert candidate(result, "r-c")["alignedTimeSeconds"] == 1.225
    assert all(
        candidate(result, event_id)["subdivision"] == 2
        for event_id in ("r-a", "r-b", "r-c")
    )


def test_deterministic_tie_breaking_prefers_simpler_subdivision():
    result = align_raw_events_to_timing([], [percussion(onset=0.25)], timing())
    item = result["candidates"][0]
    assert (item["subdivision"], item["subdivisionIndex"]) == (2, 1)


def test_raw_event_inputs_are_never_mutated():
    pitched_events = [pitched(start=0.125)]
    percussion_events = [percussion(onset=0.25)]
    timing_value = timing()
    before = copy.deepcopy((pitched_events, percussion_events, timing_value))
    align_raw_events_to_timing(pitched_events, percussion_events, timing_value)
    assert (pitched_events, percussion_events, timing_value) == before


def test_pitched_end_time_is_not_returned_as_quantized_data():
    event = pitched(start=0.125, end=0.39123)
    result = align_raw_events_to_timing([event], [], timing())
    assert event["endSeconds"] == 0.39123
    assert "alignedEndSeconds" not in result["candidates"][0]
    assert "duration" not in result["candidates"][0]


@pytest.mark.parametrize(
    "bad_id",
    ["", "../p1", "p/1", "p\\1", "p 1", "p\x00x", "x" * 129],
)
def test_unsafe_event_ids_are_rejected(bad_id):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([pitched(bad_id)], [], timing())


def test_duplicate_ids_across_types_are_rejected():
    with pytest.raises(EventAlignmentError, match="unique"):
        align_raw_events_to_timing(
            [pitched("same")],
            [percussion("same")],
            timing(),
        )


@pytest.mark.parametrize(
    "bad",
    [math.nan, math.inf, -math.inf, True, "0.5", -0.1],
)
def test_malformed_event_times_are_rejected(bad):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([pitched(start=bad)], [], timing())


def test_invalid_pitched_end_time_is_rejected():
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing(
            [pitched(start=0.5, end=0.5)],
            [],
            timing(),
        )


@pytest.mark.parametrize(
    "beats",
    [
        [0.0, 0.5, 0.5],
        [0.0, 1.0, 0.5],
        [0.0, math.nan, 1.0],
        [0.0, True, 1.0],
        [-0.1, 0.5],
        "0,0.5,1",
    ],
)
def test_invalid_beat_arrays_are_rejected(beats):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([], [], timing(beatsSeconds=beats))


@pytest.mark.parametrize("meter", [True, 0, 1, 13, 4.0, "4"])
def test_invalid_meter_is_rejected(meter):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([], [], timing(meter=meter))


@pytest.mark.parametrize("value", [0, 5, True, 2.5, "4"])
def test_invalid_subdivision_bounds_are_rejected(value):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([], [], timing(), max_subdivision=value)


@pytest.mark.parametrize(
    "value",
    [-0.1, math.nan, math.inf, True, "0.1"],
)
def test_invalid_max_offset_is_rejected(value):
    with pytest.raises(EventAlignmentError):
        align_raw_events_to_timing([], [], timing(), max_offset_seconds=value)


def test_zero_max_offset_allows_exact_alignment_only():
    result = align_raw_events_to_timing(
        [pitched("exact", 0.5), pitched("late", 0.5001, 0.7)],
        [],
        timing(),
        max_offset_seconds=0,
    )
    assert "alignedTimeSeconds" in candidate(result, "exact")
    assert "alignedTimeSeconds" not in candidate(result, "late")


def test_inconsistent_downbeats_are_omitted_with_warning():
    result = align_raw_events_to_timing(
        [pitched(start=1.0)],
        [],
        timing(
            beatsSeconds=[0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            downbeatsSeconds=[0.0, 1.5],
            meter=4,
        ),
    )
    assert "measureIndex" not in result["candidates"][0]
    assert any("inconsistent" in warning for warning in result["warnings"])


def test_downbeat_not_on_beat_is_omitted_with_warning():
    result = align_raw_events_to_timing(
        [pitched(start=1.0)],
        [],
        timing(downbeatsSeconds=[0.01]),
    )
    assert "measureIndex" not in result["candidates"][0]
    assert any("does not match" in warning for warning in result["warnings"])


def test_output_is_json_safe_and_contains_no_input_extras():
    event = pitched(start=0.125)
    event["privatePath"] = "/private/machine/path"
    result = align_raw_events_to_timing([event], [], timing())
    encoded = json.dumps(result, allow_nan=False)
    assert "/private/machine/path" not in encoded
    assert "privatePath" not in encoded
    assert result["diagnostics"]["rawTimesPreserved"] is True


def test_warnings_are_bounded():
    result = align_raw_events_to_timing(
        [pitched(start=9.0)],
        [],
        timing(
            beatConfidence=0.1,
            tempoStable=False,
            meterConfidence=0.1,
        ),
    )
    assert len(result["warnings"]) <= 8
    assert all(len(item) <= 160 for item in result["warnings"])
    assert len(result["candidates"][0].get("warnings", [])) <= 3


def test_identical_input_is_deterministic():
    args = (
        [pitched("p1", 0.13, 0.4)],
        [percussion("r1", 0.24), percussion("r2", 0.24)],
        timing(),
    )
    first = align_raw_events_to_timing(*copy.deepcopy(args))
    second = align_raw_events_to_timing(*copy.deepcopy(args))
    assert first == second
