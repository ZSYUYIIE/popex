from __future__ import annotations

import copy
import json
import math

import pytest

from app.harmony_inference import (
    HARMONY_INFERENCE_VERSION,
    HarmonyInferenceError,
    infer_harmony,
)


def event(
    event_id: str,
    midi_note: int,
    *,
    start: float = 0.0,
    end: float = 1.0,
    source: str = "other",
    confidence: object = 0.9,
    midi_pitch: float | None = None,
    warnings: list[str] | None = None,
) -> dict:
    return {
        "id": event_id,
        "sourceKind": source,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": midi_note,
        "midiPitch": float(midi_note) if midi_pitch is None else midi_pitch,
        "frequencyHz": 440.0,
        "noteName": "candidate",
        "confidence": confidence,
        "warnings": [] if warnings is None else warnings,
    }


def timing(*, confidence: object = 0.9, beats: object | None = None) -> dict:
    return {
        "beatsSeconds": [0.0, 1.0, 2.0] if beats is None else beats,
        "beatConfidence": confidence,
    }


def tonality(
    center: str = "C",
    collection: str = "ionian",
    confidence: object = 0.8,
) -> dict:
    return {
        "primaryCandidate": {
            "tonalCenter": center,
            "collection": collection,
            "displayName": f"{center} {collection}",
            "confidence": confidence,
        }
    }


def segment(result, index: int = 0) -> dict:
    return result.segments[index]


def test_clear_c_major_is_resolved_with_raw_evidence() -> None:
    events = [
        event("c", 60, source="vocals", midi_pitch=60.12),
        event("e", 64, source="other", midi_pitch=63.91),
        event("g", 67, source="bass", midi_pitch=67.08),
    ]
    before = copy.deepcopy(events)
    result = infer_harmony(events, timing(), tonality())
    first = segment(result)

    assert events == before
    assert result.version == HARMONY_INFERENCE_VERSION
    assert first["beatIndex"] == 0
    assert first["rawStartSeconds"] == 0.0
    assert first["rawEndSeconds"] == 1.0
    assert first["supportingEventIds"] == ["c", "e", "g"]
    assert first["sourceKinds"] == ["bass", "other", "vocals"]
    assert first["unresolved"] is False
    assert first["primaryCandidate"]["root"] == "C"
    assert first["primaryCandidate"]["quality"] == "major"
    assert first["primaryCandidate"]["symbol"] == "C"
    assert first["primaryCandidate"]["pitchClasses"] == [0, 4, 7]
    assert first["primaryCandidate"]["confidence"] >= 0.7
    assert {item["pitchName"] for item in first["observedPitchClasses"]} == {
        "C",
        "E",
        "G",
    }
    assert result.unresolved_event_ids == ()
    assert result.diagnostics["rawTimingAuthoritative"] is True
    assert result.diagnostics["fractionalPitchPreserved"] is True
    assert result.diagnostics["rawEvidenceIncluded"] is True
    assert result.raw_evidence[0]["id"] == "c"
    assert result.raw_evidence[0]["rawStartSeconds"] == 0.0
    assert result.raw_evidence[0]["rawEndSeconds"] == 1.0
    assert result.raw_evidence[0]["midiPitch"] == 60.12
    assert result.raw_evidence[0]["sourceKind"] == "vocals"


def test_clear_a_minor_is_resolved() -> None:
    result = infer_harmony(
        [event("a", 69), event("c", 72), event("e", 76)],
        timing(),
        tonality("A", "aeolian", 0.9),
    )
    first = segment(result)
    assert first["unresolved"] is False
    assert first["primaryCandidate"]["root"] == "A"
    assert first["primaryCandidate"]["quality"] == "minor"
    assert first["primaryCandidate"]["symbol"] == "Am"


def test_full_mix_arpeggio_is_usable_without_claiming_polyphonic_certainty() -> None:
    events = [
        event("c", 60, start=0.0, end=0.3, source="full_mix"),
        event("e", 64, start=0.3, end=0.6, source="full_mix"),
        event("g", 67, start=0.6, end=0.9, source="full_mix"),
    ]
    result = infer_harmony(events, timing())
    first = segment(result)

    assert first["primaryCandidate"]["quality"] == "major"
    assert first["primaryCandidate"]["root"] == "C"
    assert first["sourceKinds"] == ["full_mix"]
    assert any("Full-mix" in warning for warning in result.warnings)
    assert "inversionCandidate" not in first["primaryCandidate"]


def test_power_interval_keeps_major_minor_ambiguity_explicit() -> None:
    result = infer_harmony(
        [event("c", 60, source="full_mix"), event("g", 67, source="full_mix")],
        timing(),
    )
    first = segment(result)

    assert first["primaryCandidate"]["quality"] == "power"
    assert first["primaryCandidate"]["symbol"] == "C5"
    assert first["primaryCandidate"]["confidence"] <= 0.6
    qualities = {candidate["quality"] for candidate in first["alternatives"]}
    assert "major" in qualities or "minor" in qualities
    assert any("lacks a third" in warning for warning in first["warnings"])


def test_non_chord_tones_remain_visible_and_do_not_disappear() -> None:
    result = infer_harmony(
        [
            event("c", 60),
            event("e", 64),
            event("g", 67),
            event("fsharp", 66, confidence=0.35),
        ],
        timing(),
    )
    first = segment(result)

    assert set(first["supportingEventIds"]) == {"c", "e", "g", "fsharp"}
    assert any(item["pitchName"] == "F#" for item in first["observedPitchClasses"])
    candidates = [first["primaryCandidate"], *first["alternatives"]]
    assert any(candidate and candidate["nonChordToneRatio"] > 0 for candidate in candidates)


def test_bass_source_can_support_first_inversion() -> None:
    events = [
        event("bass_e", 52, source="bass", confidence=0.95),
        event("c", 60, source="other"),
        event("e", 64, source="vocals"),
        event("g", 67, source="other"),
    ]
    result = infer_harmony(events, timing(), tonality())
    primary = segment(result)["primaryCandidate"]

    assert primary["root"] == "C"
    assert primary["quality"] == "major"
    inversion = primary["inversionCandidate"]
    assert inversion["bassPitchName"] == "E"
    assert inversion["position"] == "first_inversion"
    assert inversion["sourceEventIds"] == ["bass_e"]
    assert inversion["confidence"] >= 0.6


def test_low_note_from_full_mix_never_creates_inversion_claim() -> None:
    result = infer_harmony(
        [
            event("low_e", 52, source="full_mix"),
            event("c", 60, source="full_mix"),
            event("g", 67, source="full_mix"),
        ],
        timing(),
    )
    candidates = [segment(result)["primaryCandidate"], *segment(result)["alternatives"]]
    assert all(
        candidate is None or "inversionCandidate" not in candidate
        for candidate in candidates
    )


def test_tonal_context_is_advisory_and_does_not_force_tonic_chord() -> None:
    result = infer_harmony(
        [event("d", 62), event("f", 65), event("a", 69)],
        timing(),
        tonality("C", "ionian", 1.0),
    )
    first = segment(result)

    assert result.tonal_context == {
        "tonalCenter": "C",
        "collection": "ionian",
        "displayName": "C ionian",
        "confidence": 1.0,
        "advisoryOnly": True,
    }
    assert first["primaryCandidate"]["root"] == "D"
    assert first["primaryCandidate"]["quality"] == "minor"
    assert first["primaryCandidate"]["tonalContextSupport"] > 0
    assert result.diagnostics["tonalContextAdvisoryOnly"] is True


def test_weak_or_missing_beats_use_explicit_absolute_fallback() -> None:
    events = [
        event("c", 60, start=0.1, end=0.4, source="full_mix"),
        event("e", 64, start=0.4, end=0.7, source="full_mix"),
        event("g", 67, start=0.7, end=0.95, source="full_mix"),
    ]
    weak = infer_harmony(events, timing(confidence=0.1))
    missing = infer_harmony(events, None)

    for result in (weak, missing):
        assert result.diagnostics["windowingMode"] == "absolute_time"
        assert result.diagnostics["fallbackWindowSeconds"] == 1.0
        assert result.segments[0]["windowMode"] == "absolute"
        assert "beatIndex" not in result.segments[0]
        assert any("absolute-time fallback" in warning for warning in result.warnings)


def test_part_assignment_context_is_provenance_not_a_raw_evidence_replacement() -> None:
    events = [event("c", 60), event("e", 64), event("g", 67)]
    evidence = {
        "version": "source-phrase-v1",
        "assignments": [
            {
                "eventId": "c",
                "status": "assigned",
                "partId": "part_main",
                "voiceId": "voice_main",
            },
            {"eventId": "e", "status": "unassigned"},
            {
                "eventId": "g",
                "status": "assigned",
                "partId": "part_main",
                "voiceId": "voice_main",
            },
        ],
        "unassignedEventIds": ["e"],
    }
    result = infer_harmony(events, timing(), pitched_part_evidence=evidence)
    first = segment(result)

    assert first["supportingEventIds"] == ["c", "e", "g"]
    assert first["partIds"] == ["part_main"]
    assert first["voiceIds"] == ["voice_main"]
    assert first["unassignedContextEventIds"] == ["e"]
    assert first["primaryCandidate"]["root"] == "C"
    assert any("unassigned" in warning.lower() for warning in first["warnings"])
    assert any("1 pitched event" in warning for warning in result.warnings)


def test_single_pitch_remains_unresolved_instead_of_fabricating_chord() -> None:
    result = infer_harmony([event("c", 60)], timing())
    first = segment(result)

    assert first["unresolved"] is True
    assert first["primaryCandidate"] is None
    assert first["alternatives"] == []
    assert result.unresolved_event_ids == ("c",)
    assert result.raw_evidence == (
        {
            "id": "c",
            "sourceKind": "other",
            "rawStartSeconds": 0.0,
            "rawEndSeconds": 1.0,
            "midiNote": 60,
            "midiPitch": 60.0,
            "pitchClass": 0,
            "pitchName": "C",
            "confidence": 0.9,
        },
    )
    assert any("does not support" in warning for warning in first["warnings"])


def test_ambiguity_preserves_bounded_alternatives() -> None:
    result = infer_harmony(
        [event("a", 69), event("c", 72), event("e", 76), event("g", 79)],
        timing(),
    )
    first = segment(result)

    assert len(first["alternatives"]) <= 3
    assert first["primaryCandidate"] is not None
    assert first["primaryCandidate"]["quality"] in {
        "minor7",
        "major",
        "minor",
    }
    assert all(0 <= candidate["confidence"] <= 1 for candidate in first["alternatives"])


def test_shuffle_and_mapping_insertion_order_are_deterministic() -> None:
    events = [
        event("g", 67, source="other"),
        event("c", 60, source="vocals"),
        event("e", 64, source="bass"),
    ]
    evidence_a = {
        "version": "source-phrase-v1",
        "assignments": [
            {
                "eventId": "c",
                "status": "assigned",
                "partId": "part_main",
                "voiceId": "voice_main",
            }
        ],
        "unassignedEventIds": [],
    }
    evidence_b = {
        "unassignedEventIds": [],
        "assignments": list(reversed(evidence_a["assignments"])),
        "version": "source-phrase-v1",
    }
    first = infer_harmony(events, timing(), tonality(), evidence_a)
    second = infer_harmony(list(reversed(events)), timing(), tonality(), evidence_b)

    assert first == second
    assert first.payload() == second.payload()
    encoded = json.dumps(first.payload(), allow_nan=False)
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_payload_returns_detached_copy() -> None:
    result = infer_harmony(
        [event("c", 60), event("e", 64), event("g", 67)], timing()
    )
    first = result.payload()
    second = result.payload()
    first["segments"][0]["supportingEventIds"].clear()
    first["raw_evidence"][0]["midiPitch"] = 0
    assert second["segments"][0]["supportingEventIds"] == ["c", "e", "g"]
    assert second["raw_evidence"][0]["midiPitch"] == 60.0


@pytest.mark.parametrize("bad", ["", "Bad ID", "../c", "c/1", "c\\1", "x" * 97])
def test_unsafe_event_ids_fail(bad: str) -> None:
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([event(bad, 60)], timing())


@pytest.mark.parametrize("bad", ["Vocals", "bad source", "../bass", "bass/one"])
def test_unsafe_source_slugs_fail(bad: str) -> None:
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([event("c", 60, source=bad)], timing())


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.8", -0.01, 1.01])
def test_event_confidence_must_be_finite_unit_interval(bad: object) -> None:
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([event("c", 60, confidence=bad)], timing())


def test_duplicate_ids_and_invalid_ranges_fail() -> None:
    with pytest.raises(HarmonyInferenceError, match="unique"):
        infer_harmony([event("c", 60), event("c", 64)], timing())
    with pytest.raises(HarmonyInferenceError, match="time range"):
        infer_harmony([event("c", 60, start=1.0, end=1.0)], timing())


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, "60", -1])
def test_invalid_fractional_pitch_or_note_values_fail(bad: object) -> None:
    value = event("c", 60)
    value["midiPitch"] = bad
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([value], timing())

    value = event("c", 60)
    value["midiNote"] = bad
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([value], timing())


def test_fractional_pitch_must_remain_near_nominal_note() -> None:
    with pytest.raises(HarmonyInferenceError, match="fractional pitch"):
        infer_harmony([event("c", 60, midi_pitch=61.2)], timing())


@pytest.mark.parametrize(
    "beats",
    ["bad", [0.0, 0.5, 0.5], [0.0, 1.0, 0.5], [0.0, math.nan], [0.0, True]],
)
def test_malformed_beats_fail(beats: object) -> None:
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([], timing(beats=beats))


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, math.nan])
def test_beat_confidence_must_be_unit_interval(bad: object) -> None:
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([], timing(confidence=bad))


def test_invalid_tonality_is_rejected_but_missing_tonality_is_valid() -> None:
    result = infer_harmony([], timing(), None)
    assert result.tonal_context is None

    with pytest.raises(HarmonyInferenceError):
        infer_harmony([], timing(), tonality("H", "ionian", 0.8))
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([], timing(), tonality("C", "bad collection", 0.8))
    with pytest.raises(HarmonyInferenceError):
        infer_harmony([], timing(), tonality("C", "ionian", 1.2))


@pytest.mark.parametrize(
    "warning",
    [
        "debug at /home/user/private.wav",
        "<script>alert(1)</script>",
        "api_key=private-value",
        "line one\nline two",
    ],
)
def test_unsafe_event_warning_text_is_rejected(warning: str) -> None:
    with pytest.raises(HarmonyInferenceError, match="unsafe text"):
        infer_harmony([event("c", 60, warnings=[warning])], timing())


def test_bad_part_evidence_references_and_assignments_fail() -> None:
    events = [event("c", 60)]
    with pytest.raises(HarmonyInferenceError, match="unknown raw event"):
        infer_harmony(
            events,
            timing(),
            pitched_part_evidence={
                "assignments": [{"eventId": "missing", "status": "unassigned"}],
                "unassignedEventIds": [],
            },
        )
    with pytest.raises(HarmonyInferenceError, match="requires partId"):
        infer_harmony(
            events,
            timing(),
            pitched_part_evidence={
                "assignments": [{"eventId": "c", "status": "assigned"}],
                "unassignedEventIds": [],
            },
        )
    with pytest.raises(HarmonyInferenceError, match="disagrees"):
        infer_harmony(
            events,
            timing(),
            pitched_part_evidence={
                "assignments": [
                    {
                        "eventId": "c",
                        "status": "assigned",
                        "partId": "part_main",
                        "voiceId": "voice_main",
                    }
                ],
                "unassignedEventIds": ["c"],
            },
        )


def test_unassigned_event_ids_are_bounded() -> None:
    events = [event("c", 60)]
    evidence = {
        "assignments": [],
        "unassignedEventIds": ["c"] * 100_001,
    }
    with pytest.raises(HarmonyInferenceError, match="Too many unassigned event IDs"):
        infer_harmony(events, timing(), pitched_part_evidence=evidence)


def test_no_final_notation_roman_numeral_or_tab_claims() -> None:
    result = infer_harmony(
        [event("c", 60), event("e", 64), event("g", 67)], timing()
    )
    payload = result.payload()
    encoded = json.dumps(payload).lower()

    assert payload["diagnostics"]["romanNumeralsGenerated"] is False
    assert payload["diagnostics"]["guitarVoicingsGenerated"] is False
    assert payload["diagnostics"]["notationGenerated"] is False
    for forbidden in (
        "romannumeral",
        "tablature",
        "musicxml",
        "fret",
        "stringnumber",
        "engraving",
    ):
        assert forbidden not in encoded.replace("romannumeralsgenerated", "")
