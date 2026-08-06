from __future__ import annotations

import copy
import json
import math
import random

import pytest

from app.pitched_part_inference import (
    PITCHED_PART_INFERENCE_VERSION,
    PitchedPartInferenceError,
    infer_pitched_parts,
)


def event(
    event_id: str,
    source_kind: str,
    start: float,
    end: float,
    midi_note: int,
    *,
    midi_pitch: float | None = None,
    confidence: float = 0.9,
    warnings: list[str] | None = None,
) -> dict:
    pitch = float(midi_note) if midi_pitch is None else midi_pitch
    return {
        "id": event_id,
        "sourceKind": source_kind,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": midi_note,
        "midiPitch": pitch,
        "frequencyHz": 440.0 * (2.0 ** ((pitch - 69.0) / 12.0)),
        "noteName": f"N{midi_note}",
        "confidence": confidence,
        **({"warnings": warnings} if warnings is not None else {}),
    }


def aligned(event_id: str, raw: float, beat: int, measure: int) -> dict:
    aligned_time = beat * 0.5
    return {
        "eventId": event_id,
        "eventType": "pitched",
        "rawTimeSeconds": raw,
        "beatIndex": beat,
        "subdivision": 4,
        "subdivisionIndex": 0,
        "alignedTimeSeconds": aligned_time,
        "offsetSeconds": raw - aligned_time,
        "confidence": 0.8,
        "measureIndex": measure,
        "beatInMeasure": beat % 4 + 1,
    }


def assignment_by_id(result) -> dict[str, dict]:
    return {item["eventId"]: item for item in result.assignments}


def test_vocals_and_bass_create_separate_source_aware_parts() -> None:
    result = infer_pitched_parts(
        [
            event("p000001", "vocals", 0.0, 0.4, 69),
            event("p000002", "bass", 0.0, 0.7, 40),
        ]
    )
    assert [part["sourceKind"] for part in result.parts] == ["vocals", "bass"]
    assert {part["label"] for part in result.parts} == {"Vocals", "Bass"}
    assert result.diagnostics["accountedEventCount"] == 2
    assert result.unassigned_event_ids == ()


def test_short_gap_vocal_notes_form_one_phrase_without_quantizing() -> None:
    events = [
        event("p000001", "vocals", 0.113, 0.417, 67, midi_pitch=67.21),
        event("p000002", "vocals", 0.51, 0.91, 69, midi_pitch=69.37),
        event("p000003", "vocals", 1.02, 1.42, 72, midi_pitch=71.88),
    ]
    result = infer_pitched_parts(events)
    assert len(result.phrases) == 1
    phrase = result.phrases[0]
    assert phrase["sourceEventIds"] == ["p000001", "p000002", "p000003"]
    assignments = assignment_by_id(result)
    assert assignments["p000001"]["rawStartSeconds"] == 0.113
    assert assignments["p000001"]["midiPitch"] == 67.21
    assert result.diagnostics["timingQuantized"] is False
    assert result.diagnostics["pitchQuantized"] is False


def test_large_gap_creates_new_phrase() -> None:
    result = infer_pitched_parts(
        [
            event("p000001", "vocals", 0.0, 0.4, 69),
            event("p000002", "vocals", 2.2, 2.6, 71),
        ]
    )
    assert len(result.parts) == 1
    assert len(result.voices) == 1
    assert len(result.phrases) == 2
    assert [phrase["sourceEventIds"] for phrase in result.phrases] == [
        ["p000001"],
        ["p000002"],
    ]


def test_measure_evidence_can_create_conservative_boundary() -> None:
    events = [
        event("p000001", "vocals", 0.0, 0.4, 69),
        event("p000002", "vocals", 0.6, 1.0, 71),
    ]
    result = infer_pitched_parts(
        events,
        [aligned("p000001", 0.0, 0, 0), aligned("p000002", 0.6, 8, 2)],
    )
    assert len(result.phrases) == 2
    assert assignment_by_id(result)["p000002"]["alignmentCandidates"]


def test_overlapping_same_source_events_are_preserved_in_separate_voices() -> None:
    events = [
        event("p000001", "vocals", 0.0, 1.0, 69),
        event("p000002", "vocals", 0.4, 0.8, 72),
        event("p000003", "vocals", 1.0, 1.4, 71),
    ]
    result = infer_pitched_parts(events)
    assert len(result.parts) == 1
    assert len(result.voices) == 2
    assignments = assignment_by_id(result)
    assert assignments["p000001"]["voiceId"] != assignments["p000002"]["voiceId"]
    assert assignments["p000001"]["eventId"] == "p000001"
    assert assignments["p000002"]["alternatives"]
    assert result.diagnostics["overlapEventCount"] == 2
    assert {item["eventId"] for item in result.assignments} == {
        "p000001",
        "p000002",
        "p000003",
    }


def test_other_and_full_mix_remain_broad_without_precise_instrument() -> None:
    result = infer_pitched_parts(
        [
            event("p000001", "other", 0.0, 0.5, 60),
            event("p000002", "full_mix", 0.7, 1.2, 64),
        ]
    )
    parts = {part["sourceKind"]: part for part in result.parts}
    assert parts["other"]["label"] == "Other pitched source"
    assert parts["full_mix"]["label"] == "Full mix"
    assert parts["other"]["role"] == "unresolved_pitched_source"
    for part in result.parts:
        assert "instrument" not in part
        assert "tablature" not in part
        assert "staff" not in part
    for assignment in result.assignments:
        assert "chord" not in assignment
        assert "instrument" not in assignment


def test_future_source_kind_stays_open_and_source_named() -> None:
    result = infer_pitched_parts(
        [event("p000001", "future_stem", 0.0, 0.5, 60)]
    )
    assert result.parts[0]["sourceKind"] == "future_stem"
    assert result.parts[0]["label"] == "future stem"
    assert result.parts[0]["role"] == "source_named_pitched_line"


def test_low_confidence_event_is_unassigned_but_full_evidence_remains() -> None:
    raw = event(
        "p000001",
        "vocals",
        0.123456,
        0.654321,
        69,
        midi_pitch=69.321,
        confidence=0.2,
        warnings=["Pitch evidence is weak."],
    )
    result = infer_pitched_parts([raw])
    assert result.parts == ()
    assert result.unassigned_event_ids == ("p000001",)
    assignment = result.assignments[0]
    assert assignment["status"] == "unassigned"
    assert assignment["partId"] is None
    assert assignment["rawStartSeconds"] == 0.123456
    assert assignment["rawEndSeconds"] == 0.654321
    assert assignment["midiPitch"] == 69.321
    assert "Pitch evidence is weak." in assignment["warnings"]
    assert result.diagnostics["accountedEventCount"] == 1


def test_large_pitch_jump_lowers_continuity_without_forcing_boundary() -> None:
    result = infer_pitched_parts(
        [
            event("p000001", "vocals", 0.0, 0.4, 48),
            event("p000002", "vocals", 0.5, 0.9, 84),
        ]
    )
    assert len(result.phrases) == 1
    assert result.phrases[0]["warnings"]
    assert result.phrases[0]["continuityConfidence"] < 0.9


def test_shuffled_input_and_mapping_order_produce_identical_output() -> None:
    base = [
        event("p000003", "bass", 1.0, 1.4, 43),
        event("p000001", "vocals", 0.0, 0.4, 69),
        event("p000002", "vocals", 0.5, 0.9, 71),
    ]
    shuffled = list(base)
    random.Random(17).shuffle(shuffled)
    reordered = [dict(reversed(list(item.items()))) for item in shuffled]
    first = infer_pitched_parts(base)
    second = infer_pitched_parts(reordered)
    assert first == second
    assert first.payload() == second.payload()


def test_alignment_candidates_preserve_aligned_and_unaligned_evidence() -> None:
    raw = event("p000001", "vocals", 0.13, 0.5, 69)
    candidates = [
        {
            "eventId": "p000001",
            "eventType": "pitched",
            "rawTimeSeconds": 0.13,
            "confidence": 0.0,
            "warnings": ["Raw timing remains authoritative."],
        },
        {
            "eventId": "p000001",
            "eventType": "pitched",
            "rawTimeSeconds": 0.13,
            "beatIndex": 0,
            "subdivision": 4,
            "subdivisionIndex": 1,
            "alignedTimeSeconds": 0.125,
            "offsetSeconds": 0.005,
            "confidence": 0.8,
            "measureIndex": 0,
            "beatInMeasure": 1,
        },
    ]
    result = infer_pitched_parts([raw], list(reversed(candidates)))
    evidence = result.assignments[0]["alignmentCandidates"]
    assert len(evidence) == 2
    assert "alignedTimeSeconds" in evidence[0]
    assert "alignedTimeSeconds" not in evidence[1]
    assert result.assignments[0]["rawStartSeconds"] == 0.13


def test_non_pitched_alignment_is_safely_ignored() -> None:
    result = infer_pitched_parts(
        [event("p000001", "vocals", 0.0, 0.4, 69)],
        [
            {
                "eventId": "r000001",
                "eventType": "percussion",
                "rawTimeSeconds": 0.2,
            }
        ],
    )
    assert result.diagnostics["ignoredNonPitchedAlignmentCount"] == 1
    assert result.warnings


def test_payload_returns_fresh_copies_and_input_is_not_mutated() -> None:
    events = [
        {
            **event("p000001", "vocals", 0.0, 0.4, 69),
            "rawFeatureSummary": {"voicedProbability": 0.91},
        }
    ]
    before = copy.deepcopy(events)
    result = infer_pitched_parts(events)
    first = result.payload()
    first["assignments"][0]["rawFeatureSummary"]["voicedProbability"] = 0
    first["parts"][0]["label"] = "mutated"
    second = result.payload()
    assert events == before
    assert second["assignments"][0]["rawFeatureSummary"]["voicedProbability"] == 0.91
    assert second["parts"][0]["label"] == "Vocals"


@pytest.mark.parametrize(
    "events",
    [
        [
            event("p000001", "vocals", 0.0, 0.4, 69),
            event("p000001", "bass", 1.0, 1.4, 40),
        ],
        [event("bad id", "vocals", 0.0, 0.4, 69)],
        [event("p000001", "../vocals", 0.0, 0.4, 69)],
        [event("p000001", "vocals", 0.4, 0.4, 69)],
        [event("p000001", "vocals", 0.5, 0.4, 69)],
    ],
)
def test_duplicate_unsafe_and_invalid_ranges_fail_safely(events: list[dict]) -> None:
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts(events)


@pytest.mark.parametrize(
    "field",
    ["startSeconds", "endSeconds", "midiPitch", "frequencyHz", "confidence"],
)
def test_non_finite_values_fail_safely(field: str) -> None:
    raw = event("p000001", "vocals", 0.0, 0.4, 69)
    raw[field] = math.nan
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts([raw])


def test_bad_alignment_reference_and_raw_time_fail_safely() -> None:
    raw = event("p000001", "vocals", 0.0, 0.4, 69)
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts(
            [raw],
            [
                {
                    "eventId": "p999999",
                    "eventType": "pitched",
                    "rawTimeSeconds": 0.0,
                }
            ],
        )
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts(
            [raw],
            [
                {
                    "eventId": "p000001",
                    "eventType": "pitched",
                    "rawTimeSeconds": 0.1,
                }
            ],
        )


def test_machine_paths_samples_tensors_and_unknown_fields_fail_safely() -> None:
    raw = event("p000001", "vocals", 0.0, 0.4, 69)
    raw["warnings"] = ["Debug data at /home/user/private.wav"]
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts([raw])

    raw = event("p000001", "vocals", 0.0, 0.4, 69)
    raw["rawFeatureSummary"] = {"audioSamples": [0.1]}
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts([raw])

    raw = event("p000001", "vocals", 0.0, 0.4, 69)
    raw["chord"] = "Cmaj7"
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts([raw])


def test_empty_input_is_valid_and_json_safe() -> None:
    result = infer_pitched_parts([])
    assert result.version == PITCHED_PART_INFERENCE_VERSION
    assert result.parts == ()
    assert result.assignments == ()
    assert result.diagnostics["accountedEventCount"] == 0
    json.dumps(result.payload(), allow_nan=False)


def test_version_is_open_but_path_safe() -> None:
    assert (
        infer_pitched_parts([], version="source-phrase-v1.1").version
        == "source-phrase-v1.1"
    )
    with pytest.raises(PitchedPartInferenceError):
        infer_pitched_parts([], version="../private")
