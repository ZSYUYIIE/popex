from __future__ import annotations

import copy
import json
import math

import pytest

from app.percussion_interpretation import (
    PERCUSSION_INTERPRETATION_VERSION,
    PercussionInterpretationError,
    PercussionInterpretationResult,
    interpret_percussion,
)


def event(
    event_id: str,
    time: float,
    hits: list[dict],
    *,
    strength: float = 0.8,
    source: str = "drums",
    warnings: list[str] | None = None,
) -> dict:
    return {
        "id": event_id,
        "sourceKind": source,
        "timeSeconds": time,
        "strength": strength,
        "hits": hits,
        "rawFeatureSummary": {
            "lowBandRatio": 0.4,
            "midBandRatio": 0.3,
            "highBandRatio": 0.3,
            "transientStrength": strength,
        },
        "warnings": warnings or [],
    }


def hit(kind: str, confidence: float, **extra) -> dict:
    return {"kind": kind, "confidence": confidence, **extra}


def aligned(
    event_id: str,
    raw: float,
    aligned_time: float,
    *,
    beat: int = 0,
    subdivision: int = 1,
    subdivision_index: int = 0,
    confidence: float = 0.8,
    measure: int | None = None,
    beat_in_measure: int | None = None,
) -> dict:
    value = {
        "eventId": event_id,
        "eventType": "percussion",
        "rawTimeSeconds": raw,
        "beatIndex": beat,
        "subdivision": subdivision,
        "subdivisionIndex": subdivision_index,
        "alignedTimeSeconds": aligned_time,
        "offsetSeconds": raw - aligned_time,
        "confidence": confidence,
    }
    if measure is not None:
        value["measureIndex"] = measure
        value["beatInMeasure"] = beat_in_measure
    return value


def test_public_contract_and_empty_result() -> None:
    result = interpret_percussion([])
    assert isinstance(result, PercussionInterpretationResult)
    assert result.version == PERCUSSION_INTERPRETATION_VERSION
    assert result.parts == ()
    assert result.assignments == ()
    assert result.diagnostics["finalNotationConstructed"] is False
    assert "No raw percussion events" in result.warnings[0]


def test_kick_maps_conservatively_to_low_drum() -> None:
    result = interpret_percussion([event("r1", 0.25, [hit("kick", 0.9)])])
    assignment = result.assignments[0]
    assert assignment["voiceId"] == "drum-voice-low-drum"
    assert assignment["rawHitKind"] == "kick"
    assert assignment["rawTimeSeconds"] == 0.25
    assert assignment["confidence"] == 0.9
    assert assignment["resolution"] == "resolved"
    assert result.voices[0]["kind"] == "low_drum"


def test_snare_and_tom_remain_broad() -> None:
    result = interpret_percussion(
        [
            event("r1", 0.1, [hit("snare", 0.8)]),
            event("r2", 0.2, [hit("tom", 0.7)]),
        ]
    )
    assert [item["voiceId"] for item in result.assignments] == [
        "drum-voice-mid-drum",
        "drum-voice-tom-like",
    ]
    serialized = json.dumps(result.assignments)
    for forbidden in ("left_hand", "right_hand", "sticking", "fill", "flam"):
        assert forbidden not in serialized


def test_closed_hat_and_cymbal_are_distinguishable_but_not_exact() -> None:
    result = interpret_percussion(
        [
            event("r1", 0.1, [hit("closed_hihat", 0.8)]),
            event("r2", 0.2, [hit("cymbal", 0.8)]),
        ]
    )
    assert [item["voiceId"] for item in result.assignments] == [
        "drum-voice-closed-high-frequency",
        "drum-voice-cymbal-like",
    ]
    assert {voice["label"] for voice in result.voices} == {
        "Closed high-frequency voice",
        "Cymbal-like voice",
    }


def test_open_hat_kind_is_preserved_with_broad_open_voice() -> None:
    result = interpret_percussion([event("r1", 0.1, [hit("open_hihat", 0.8)])])
    assignment = result.assignments[0]
    assert assignment["rawHitKind"] == "open_hihat"
    assert assignment["rawHit"]["kind"] == "open_hihat"
    assert assignment["voiceId"] == "drum-voice-open-high-frequency"


def test_simultaneous_hits_remain_separate_assignments() -> None:
    raw = event(
        "r1",
        1.0,
        [hit("kick", 0.9), hit("closed_hihat", 0.75)],
        warnings=["Independent bands support two candidates."],
    )
    result = interpret_percussion([raw])
    assert len(result.assignments) == 2
    assert [item["hitIndex"] for item in result.assignments] == [0, 1]
    assert [item["rawTimeSeconds"] for item in result.assignments] == [1.0, 1.0]
    assert [item["eventId"] for item in result.assignments] == ["r1", "r1"]
    assert result.diagnostics["simultaneousEventCount"] == 1
    assert result.diagnostics["simultaneousHitsPreserved"] is True
    assert all(
        item["eventWarnings"] == ["Independent bands support two candidates."]
        for item in result.assignments
    )


def test_low_confidence_known_hit_remains_unresolved() -> None:
    result = interpret_percussion([event("r1", 0.1, [hit("kick", 0.2)])])
    assignment = result.assignments[0]
    assert assignment["rawHitKind"] == "kick"
    assert assignment["resolution"] == "unresolved"
    assert assignment["voiceId"] == "drum-voice-unresolved-percussion"
    assert result.unresolved_event_ids == ("r1",)


def test_unknown_future_hit_kind_remains_representable() -> None:
    result = interpret_percussion(
        [event("future", 0.4, [hit("electronic_click", 0.92, detectorRank=1)])]
    )
    assignment = result.assignments[0]
    assert assignment["rawHitKind"] == "electronic_click"
    assert assignment["rawHit"]["detectorRank"] == 1
    assert assignment["resolution"] == "unresolved"
    assert result.voices[0]["kind"] == "unresolved_percussion"


def test_every_event_is_accounted_for() -> None:
    result = interpret_percussion(
        [
            event("resolved", 0.1, [hit("kick", 0.9)]),
            event("unresolved", 0.2, [hit("future_hit", 0.9)]),
        ]
    )
    assigned = {item["eventId"] for item in result.assignments}
    assert assigned == {"resolved", "unresolved"}
    assert result.unresolved_event_ids == ("unresolved",)
    assert result.parts[0]["rawEventIds"] == ["resolved", "unresolved"]


def test_unaligned_event_keeps_raw_time_and_time_relative_assignment() -> None:
    result = interpret_percussion([event("r1", 1.234, [hit("kick", 0.9)])])
    alignment = result.assignments[0]["alignment"]
    assert alignment == {
        "aligned": False,
        "rawTimeSeconds": 1.234,
        "confidence": 0.0,
        "warnings": [],
    }
    assert result.groups == ()
    assert result.diagnostics["unalignedEventCount"] == 1


def test_aligned_event_retains_exact_grid_metadata_separately() -> None:
    candidate = aligned(
        "r1",
        1.03,
        1.0,
        beat=2,
        subdivision=4,
        subdivision_index=1,
        confidence=0.654321,
        measure=0,
        beat_in_measure=3,
    )
    result = interpret_percussion(
        [event("r1", 1.03, [hit("snare", 0.8)])], [candidate]
    )
    assignment = result.assignments[0]
    assert assignment["rawTimeSeconds"] == 1.03
    assert assignment["alignment"] == {
        "aligned": True,
        "rawTimeSeconds": 1.03,
        "confidence": 0.654321,
        "warnings": [],
        "beatIndex": 2,
        "subdivision": 4,
        "subdivisionIndex": 1,
        "alignedTimeSeconds": 1.0,
        "offsetSeconds": 0.030000000000000027,
        "measureIndex": 0,
        "beatInMeasure": 3,
    }


def test_measure_evidence_creates_advisory_group_without_merging_events() -> None:
    events = [
        event("r1", 1.0, [hit("kick", 0.9)]),
        event("r2", 1.5, [hit("snare", 0.9)]),
    ]
    alignments = [
        aligned("r1", 1.0, 1.0, beat=2, measure=0, beat_in_measure=3),
        aligned("r2", 1.5, 1.5, beat=3, measure=0, beat_in_measure=4),
    ]
    result = interpret_percussion(events, alignments)
    assert len(result.assignments) == 2
    assert len(result.groups) == 1
    group = result.groups[0]
    assert group["kind"] == "measure_candidate"
    assert group["measureIndex"] == 0
    assert group["eventIds"] == ["r1", "r2"]
    assert len(group["assignmentIds"]) == 2


def test_beat_evidence_creates_beat_group_when_measure_is_absent() -> None:
    result = interpret_percussion(
        [event("r1", 1.0, [hit("kick", 0.9)])],
        [aligned("r1", 1.0, 1.0, beat=4)],
    )
    assert result.groups[0]["kind"] == "beat_candidate"
    assert result.groups[0]["beatIndex"] == 4


def test_shuffled_input_is_deterministic() -> None:
    events = [
        event("r2", 0.5, [hit("snare", 0.8)]),
        event("r1", 0.1, [hit("kick", 0.9), hit("closed_hihat", 0.8)]),
    ]
    alignments = [
        aligned("r2", 0.5, 0.5, beat=1),
        aligned("r1", 0.1, 0.0, beat=0),
    ]
    first = interpret_percussion(events, alignments)
    second = interpret_percussion(list(reversed(events)), list(reversed(alignments)))
    assert first == second
    assert [item["eventId"] for item in first.assignments] == ["r1", "r1", "r2"]


def test_inputs_are_not_mutated_and_nested_evidence_is_copied() -> None:
    events = [event("r1", 0.1, [hit("kick", 0.8, evidence={"rank": 1})])]
    alignments = [aligned("r1", 0.1, 0.0)]
    original_events = copy.deepcopy(events)
    original_alignments = copy.deepcopy(alignments)
    result = interpret_percussion(events, alignments)
    assert events == original_events
    assert alignments == original_alignments
    events[0]["rawFeatureSummary"]["lowBandRatio"] = 0.0
    events[0]["hits"][0]["evidence"]["rank"] = 99
    assert result.assignments[0]["rawFeatureSummary"]["lowBandRatio"] == 0.4
    assert result.assignments[0]["rawHit"]["evidence"]["rank"] == 1


@pytest.mark.parametrize(
    "events",
    [
        [event("dup", 0.1, [hit("kick", 0.8)]), event("dup", 0.2, [hit("snare", 0.8)])],
        [{**event("bad/id", 0.1, [hit("kick", 0.8)])}],
        [{**event("r1", 0.1, [hit("kick", 0.8)]), "sourceKind": "Bad Source"}],
        [event("r1", 0.1, [hit("Bad Hit", 0.8)])],
        [{**event("r1", 0.1, [hit("kick", 0.8)]), "hits": []}],
        [{**event("r1", 0.1, [hit("kick", 0.8)]), "timeSeconds": math.nan}],
        [{**event("r1", 0.1, [hit("kick", 0.8)]), "strength": 1.1}],
        [event("r1", 0.1, [hit("kick", math.inf)])],
    ],
)
def test_malformed_events_fail_safely(events: list[dict]) -> None:
    with pytest.raises(PercussionInterpretationError):
        interpret_percussion(events)


@pytest.mark.parametrize(
    "candidate",
    [
        aligned("missing", 0.1, 0.0),
        {**aligned("r1", 0.2, 0.0), "rawTimeSeconds": 0.3},
        {**aligned("r1", 0.2, 0.0), "eventType": "pitched"},
        {**aligned("r1", 0.2, 0.0), "offsetSeconds": 9.0},
        {**aligned("r1", 0.2, 0.0), "subdivisionIndex": 4, "subdivision": 4},
        {**aligned("r1", 0.2, 0.0), "unexpected": "value"},
        {
            "eventId": "r1",
            "eventType": "percussion",
            "rawTimeSeconds": 0.2,
            "confidence": 0.0,
            "beatIndex": 0,
        },
    ],
)
def test_bad_alignment_references_and_partial_grids_fail(candidate: dict) -> None:
    events = [event("r1", 0.2, [hit("kick", 0.8)])]
    with pytest.raises(PercussionInterpretationError):
        interpret_percussion(events, [candidate])


def test_duplicate_alignment_reference_fails() -> None:
    events = [event("r1", 0.2, [hit("kick", 0.8)])]
    candidate = aligned("r1", 0.2, 0.0)
    with pytest.raises(PercussionInterpretationError):
        interpret_percussion(events, [candidate, candidate])


def test_unsafe_warning_and_raw_feature_data_fail() -> None:
    with pytest.raises(PercussionInterpretationError):
        interpret_percussion(
            [event("r1", 0.1, [hit("kick", 0.8)], warnings=["see https://bad.example"])]
        )
    bad = event("r1", 0.1, [hit("kick", 0.8)])
    bad["rawFeatureSummary"] = {"value": float("nan")}
    with pytest.raises(PercussionInterpretationError):
        interpret_percussion([bad])


def test_output_is_safe_json_compatible() -> None:
    result = interpret_percussion(
        [event("r1", 0.1, [hit("kick", 0.8), hit("future_hit", 0.5)])]
    )
    payload = {
        "version": result.version,
        "parts": result.parts,
        "voices": result.voices,
        "groups": result.groups,
        "assignments": result.assignments,
        "unresolvedEventIds": result.unresolved_event_ids,
        "warnings": result.warnings,
        "diagnostics": result.diagnostics,
    }
    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded


def test_result_has_no_final_notation_or_overprecision_fields() -> None:
    result = interpret_percussion([event("r1", 0.1, [hit("snare", 0.9)])])
    encoded = json.dumps(result.assignments).lower()
    for forbidden in (
        "notation",
        "sticking",
        "left hand",
        "right hand",
        "foot",
        "fill",
        "roll",
        "flam",
        "ghost",
        "accent",
    ):
        assert forbidden not in encoded
