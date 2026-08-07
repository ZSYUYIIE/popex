from __future__ import annotations

import copy
import json
import math
from dataclasses import asdict
from pathlib import Path

import pytest

from app.config import Settings
from app.percussion_interpretation import (
    PERCUSSION_INTERPRETATION_VERSION,
    interpret_percussion,
)
from app.pitched_part_inference import (
    PITCHED_PART_INFERENCE_VERSION,
    infer_pitched_parts,
)
from app.rhythm_interpretation import (
    RHYTHM_INTERPRETATION_VERSION,
    interpret_rhythm,
)
from app.transcription_events import RAW_TRANSCRIPTION_RELATIVE_PATH
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    INTERPRETATION_DRAFT_SCHEMA_VERSION,
    TranscriptionDraftError,
    TranscriptionDraftValidationError,
    load_transcription_draft,
    validate_transcription_draft,
    write_transcription_draft,
)

JOB_ID = "a" * 32


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(
        data_dir=tmp_path,
        allowed_hosts=("example.invalid",),
        max_duration_seconds=60,
        max_filesize_mb=16,
        max_upload_mb=16,
        audio_quality="192",
    )
    value.exports_dir.mkdir(parents=True)
    (value.exports_dir / JOB_ID).mkdir()
    return value


def _interpretation_evidence() -> dict:
    return {
        "pitchedPartInference": {
            "version": "source-phrase-v1",
            "parts": [
                {
                    "id": "part_vocals_evidence",
                    "sourceEventIds": ["p000001", "p000003"],
                    "rawStartSeconds": 0.1,
                    "rawEndSeconds": 0.66,
                },
                {
                    "id": "part_other_evidence",
                    "sourceEventIds": ["p000002"],
                    "rawStartSeconds": 0.9,
                    "rawEndSeconds": 1.4,
                },
            ],
            "voices": [
                {
                    "id": "voice_vocals_evidence",
                    "sourceEventIds": ["p000001", "p000003"],
                }
            ],
            "phrases": [
                {
                    "id": "phrase_vocals_evidence",
                    "sourceEventIds": ["p000001", "p000003"],
                    "warnings": ["Boundary remains editable."],
                }
            ],
            "assignments": [
                {
                    "eventId": "p000001",
                    "sourceKind": "vocals",
                    "rawStartSeconds": 0.1,
                    "rawEndSeconds": 0.6,
                    "midiNote": 69,
                    "midiPitch": 69.2,
                    "frequencyHz": 445.1,
                    "noteName": "A4",
                    "alignmentCandidates": [
                        {
                            "eventId": "p000001",
                            "eventType": "pitched",
                            "rawTimeSeconds": 0.1,
                            "alignedTimeSeconds": 0.0,
                            "offsetSeconds": 0.1,
                            "beatIndex": 0,
                            "subdivision": 4,
                            "subdivisionIndex": 0,
                            "confidence": 0.9,
                        }
                    ],
                    "alternatives": [
                        {
                            "kind": "voice_assignment",
                            "confidence": 0.3,
                            "reasonCode": "editable_alternative",
                        }
                    ],
                },
                {
                    "eventId": "p000002",
                    "sourceKind": "other",
                    "rawStartSeconds": 0.9,
                    "rawEndSeconds": 1.4,
                    "midiNote": 72,
                    "midiPitch": 72.17,
                    "frequencyHz": 527.4,
                    "noteName": "C5",
                    "alignmentCandidates": [],
                    "alternatives": [],
                },
                {
                    "eventId": "p000003",
                    "sourceKind": "vocals",
                    "rawStartSeconds": 0.65,
                    "rawEndSeconds": 0.66,
                    "midiNote": 70,
                    "midiPitch": 70.25,
                    "frequencyHz": 470.0,
                    "noteName": "A#4",
                    "alignmentCandidates": [],
                    "alternatives": [],
                },
            ],
            "unassignedEventIds": ["p000003"],
            "warnings": ["Low-confidence event remains unassigned."],
            "diagnostics": {
                "inputEventCount": 3,
                "pitchQuantized": False,
                "notationGenerated": False,
            },
        },
        "percussionInterpretation": {
            "version": "broad-drum-structure-v1",
            "parts": [
                {
                    "id": "drum-part-001",
                    "rawEventIds": ["r000001"],
                }
            ],
            "voices": [
                {"id": "drum-voice-low-drum", "rawEventIds": ["r000001"]},
                {
                    "id": "drum-voice-closed-high-frequency",
                    "rawEventIds": ["r000001"],
                },
            ],
            "groups": [
                {
                    "id": "drum-group-001",
                    "eventIds": ["r000001"],
                    "rawStartSeconds": 0.5,
                }
            ],
            "assignments": [
                {
                    "eventId": "r000001",
                    "hitIndex": 0,
                    "rawHitKind": "kick",
                    "rawHit": {"kind": "kick", "confidence": 0.9},
                    "rawTimeSeconds": 0.5,
                    "rawFeatureSummary": {"lowBandRatio": 0.7},
                    "alignment": {
                        "eventId": "r000001",
                        "aligned": True,
                        "rawTimeSeconds": 0.5,
                        "alignedTimeSeconds": 0.5,
                        "offsetSeconds": 0.0,
                    },
                    "alternatives": [],
                },
                {
                    "eventId": "r000001",
                    "hitIndex": 1,
                    "rawHitKind": "closed_hihat",
                    "rawHit": {"kind": "closed_hihat", "confidence": 0.76},
                    "rawTimeSeconds": 0.5,
                    "rawFeatureSummary": {"highBandRatio": 0.8},
                    "alignment": {
                        "eventId": "r000001",
                        "aligned": True,
                        "rawTimeSeconds": 0.5,
                        "alignedTimeSeconds": 0.5,
                        "offsetSeconds": 0.0,
                    },
                    "alternatives": [],
                },
            ],
            "unresolved_event_ids": [],
            "warnings": ["Simultaneous hits remain separate."],
            "diagnostics": {
                "simultaneousHitsPreserved": True,
                "finalNotationConstructed": False,
            },
        },
        "rhythmInterpretation": {
            "version": "conservative-grid-v1",
            "meter_candidates": [
                {
                    "meter": 4,
                    "confidence": 0.85,
                    "reasonCodes": ["timing_meter_evidence"],
                }
            ],
            "measures": [
                {
                    "id": "rhythm_measure_0001",
                    "index": 0,
                    "startSeconds": 0.0,
                    "endSeconds": 2.0,
                }
            ],
            "event_interpretations": [
                {
                    "eventId": "p000001",
                    "sourceEventIds": ["p000001"],
                    "eventType": "pitched",
                    "rawTiming": {
                        "startSeconds": 0.1,
                        "endSeconds": 0.6,
                        "durationSeconds": 0.5,
                    },
                    "placementHypotheses": [
                        {
                            "kind": "grid",
                            "alignedTimeSeconds": 0.0,
                            "confidence": 0.9,
                        }
                    ],
                    "durationHypotheses": [
                        {
                            "label": "quarter",
                            "durationSeconds": 0.5,
                            "confidence": 0.8,
                        }
                    ],
                    "continuationHypotheses": [],
                    "alternatives": [],
                },
                {
                    "eventId": "p000002",
                    "sourceEventIds": ["p000002"],
                    "eventType": "pitched",
                    "rawTiming": {
                        "startSeconds": 0.9,
                        "endSeconds": 1.4,
                        "durationSeconds": 0.5,
                    },
                    "placementHypotheses": [{"kind": "unresolved"}],
                    "durationHypotheses": [],
                    "continuationHypotheses": [],
                    "alternatives": [],
                },
                {
                    "eventId": "p000003",
                    "sourceEventIds": ["p000003"],
                    "eventType": "pitched",
                    "rawTiming": {
                        "startSeconds": 0.65,
                        "endSeconds": 0.66,
                        "durationSeconds": 0.01,
                    },
                    "placementHypotheses": [{"kind": "unresolved"}],
                    "durationHypotheses": [],
                    "continuationHypotheses": [],
                    "alternatives": [],
                },
                {
                    "eventId": "r000001",
                    "sourceEventIds": ["r000001"],
                    "eventType": "percussion",
                    "rawTiming": {"startSeconds": 0.5, "endSeconds": 0.5},
                    "placementHypotheses": [
                        {
                            "kind": "grid",
                            "alignedTimeSeconds": 0.5,
                            "confidence": 0.9,
                        }
                    ],
                    "durationHypotheses": [],
                    "continuationHypotheses": [],
                    "alternatives": [],
                },
            ],
            "rest_candidates": [
                {
                    "id": "rest_candidate_001",
                    "sourceEventIds": ["p000001", "p000003"],
                    "rawStartSeconds": 0.6,
                    "rawEndSeconds": 0.65,
                    "alternatives": [
                        {"kind": "no_rest", "confidence": 0.4}
                    ],
                }
            ],
            "warnings": ["Raw timing remains authoritative."],
            "diagnostics": {
                "rawTimingAuthoritative": True,
                "timingMode": "measured",
            },
        },
    }


def payload() -> dict:
    return {
        "schemaVersion": 1,
        "draftVersion": "editable-interpretation-v1",
        "createdAt": "2026-08-06T05:00:00+00:00",
        "sourceTranscription": {
            "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
            "schemaVersion": 1,
            "transcriptionVersion": "raw-events-v1",
            "provenance": {
                "pipelineVersion": "raw-transcription-v1",
                "rawEventCount": 4,
            },
            "sourceEventIndex": [
                {
                    "id": "p000002",
                    "eventType": "pitched",
                    "sourceKind": "other",
                    "rawStartSeconds": 0.9,
                    "rawEndSeconds": 1.4,
                    "confidence": 0.45,
                    "midiPitch": 72.17,
                },
                {
                    "id": "r000001",
                    "eventType": "percussion",
                    "sourceKind": "drums",
                    "rawStartSeconds": 0.5,
                    "rawEndSeconds": 0.5,
                    "confidence": 0.88,
                    "hitKinds": ["kick", "closed_hihat"],
                },
                {
                    "id": "p000001",
                    "eventType": "pitched",
                    "sourceKind": "vocals",
                    "rawStartSeconds": 0.1,
                    "rawEndSeconds": 0.6,
                    "confidence": 0.91,
                    "midiPitch": 69.2,
                },
                {
                    "id": "p000003",
                    "eventType": "pitched",
                    "sourceKind": "vocals",
                    "rawStartSeconds": 0.65,
                    "rawEndSeconds": 0.66,
                    "confidence": 0.25,
                },
            ],
        },
        "algorithms": {
            "rhythmInterpretation": {
                "version": "conservative-grid-v1",
                "rawTimesPreserved": True,
            },
            "pitchedPartInference": {"version": "source-phrase-v1"},
            "percussionInterpretation": {
                "version": "broad-drum-structure-v1"
            },
        },
        "interpretationEvidence": _interpretation_evidence(),
        "parts": [
            {
                "id": "part_drums",
                "sourceKind": "drums",
                "role": "percussion",
                "instrumentKind": "broad_drum_part",
                "voiceIds": ["voice_hat", "voice_kick"],
                "sourceEventIds": ["r000001"],
                "confidence": 0.86,
                "warnings": [],
            },
            {
                "id": "part_other",
                "sourceKind": "other",
                "role": "pitched",
                "instrumentKind": "unresolved_instrument",
                "voiceIds": ["voice_other"],
                "sourceEventIds": ["p000002"],
                "confidence": 0.4,
                "alternatives": [
                    {
                        "id": "alt_part_full_mix",
                        "kind": "part_assignment",
                        "confidence": 0.28,
                        "proposedSourceKind": "full_mix",
                    }
                ],
                "warnings": ["Instrument identity remains unresolved."],
            },
            {
                "id": "part_vocals",
                "sourceKind": "vocals",
                "role": "pitched",
                "instrumentKind": "voice",
                "voiceIds": ["voice_vocals"],
                "sourceEventIds": ["p000001", "p000003"],
                "confidence": 0.9,
            },
        ],
        "voices": [
            {
                "id": "voice_vocals",
                "partId": "part_vocals",
                "voiceKind": "monophonic",
                "sourceEventIds": ["p000001", "p000003"],
                "confidence": 0.88,
            },
            {
                "id": "voice_other",
                "partId": "part_other",
                "voiceKind": "unresolved_pitched",
                "sourceEventIds": ["p000002"],
                "confidence": 0.42,
            },
            {
                "id": "voice_kick",
                "partId": "part_drums",
                "voiceKind": "low_drum",
                "sourceEventIds": ["r000001"],
                "confidence": 0.9,
            },
            {
                "id": "voice_hat",
                "partId": "part_drums",
                "voiceKind": "closed_high_frequency",
                "sourceEventIds": ["r000001"],
                "confidence": 0.76,
            },
        ],
        "measures": [
            {
                "id": "measure_0001",
                "index": 0,
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 2.0,
                "interpretedStartSeconds": 0.0,
                "interpretedDurationSeconds": 2.0,
                "meterNumerator": 4,
                "meterDenominator": 4,
                "startBeatIndex": 0,
                "endBeatIndex": 4,
                "confidence": 0.82,
            }
        ],
        "phrases": [
            {
                "id": "phrase_vocals",
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
                "sourceEventIds": ["p000001", "p000003"],
                "rawStartSeconds": 0.1,
                "rawEndSeconds": 0.66,
                "interpretedStartSeconds": 0.0,
                "interpretedDurationSeconds": 0.75,
                "measureIds": ["measure_0001"],
                "confidence": 0.78,
            }
        ],
        "pitchedItems": [
            {
                "id": "pitched_note",
                "interpretationType": "note",
                "placementStatus": "placed",
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
                "measureId": "measure_0001",
                "phraseId": "phrase_vocals",
                "sourceEventIds": ["p000001"],
                "rawStartSeconds": 0.1,
                "rawEndSeconds": 0.6,
                "interpretedStartSeconds": 0.0,
                "interpretedDurationSeconds": 0.5,
                "gridPosition": {
                    "measureId": "measure_0001",
                    "measureIndex": 0,
                    "beatIndex": 0,
                    "beatInMeasure": 1,
                    "subdivision": 4,
                    "subdivisionIndex": 0,
                    "alignedTimeSeconds": 0.0,
                    "offsetSeconds": 0.1,
                },
                "sourceKind": "vocals",
                "pitch": {
                    "midiNote": 69,
                    "midiPitch": 69.2,
                    "frequencyHz": 445.1,
                    "noteName": "A4",
                },
                "tieCandidate": {
                    "role": "start_candidate",
                    "targetItemId": "pitched_unassigned",
                    "confidence": 0.35,
                },
                "confidence": 0.87,
                "alternatives": [
                    {
                        "id": "alt_note_short",
                        "kind": "duration",
                        "confidence": 0.44,
                        "interpretedStartSeconds": 0.0,
                        "interpretedDurationSeconds": 0.25,
                    },
                    {
                        "id": "alt_note_late",
                        "kind": "grid_placement",
                        "confidence": 0.31,
                        "interpretedStartSeconds": 0.125,
                        "interpretedDurationSeconds": 0.5,
                    },
                ],
            },
            {
                "id": "pitched_rest",
                "interpretationType": "rest",
                "placementStatus": "placed",
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
                "measureId": "measure_0001",
                "phraseId": "phrase_vocals",
                "sourceEventIds": ["p000003"],
                "rawStartSeconds": 0.65,
                "rawEndSeconds": 0.66,
                "interpretedStartSeconds": 0.5,
                "interpretedDurationSeconds": 0.25,
                "gridPosition": {
                    "measureId": "measure_0001",
                    "measureIndex": 0,
                    "beatIndex": 1,
                    "beatInMeasure": 2,
                    "subdivision": 4,
                    "subdivisionIndex": 0,
                },
                "sourceKind": "vocals",
                "confidence": 0.41,
                "alternatives": [
                    {
                        "id": "alt_rest_none",
                        "kind": "no_rest",
                        "confidence": 0.39,
                    }
                ],
                "warnings": ["Rest is an editable gap hypothesis."],
            },
            {
                "id": "pitched_unassigned",
                "interpretationType": "unassigned",
                "placementStatus": "unassigned",
                "partId": "part_other",
                "voiceId": "voice_other",
                "sourceEventIds": ["p000002"],
                "rawStartSeconds": 0.9,
                "rawEndSeconds": 1.4,
                "sourceKind": "other",
                "pitch": {"midiNote": 72, "midiPitch": 72.17},
                "confidence": 0.32,
                "alternatives": [
                    {
                        "id": "alt_unassigned_vocal",
                        "kind": "voice_assignment",
                        "confidence": 0.26,
                        "partId": "part_vocals",
                        "voiceId": "voice_vocals",
                    }
                ],
                "warnings": ["Musical placement is unresolved."],
            },
        ],
        "percussionItems": [
            {
                "id": "percussion_item",
                "placementStatus": "placed",
                "partId": "part_drums",
                "voiceIds": ["voice_kick", "voice_hat"],
                "measureId": "measure_0001",
                "sourceEventIds": ["r000001"],
                "rawStartSeconds": 0.5,
                "rawEndSeconds": 0.5,
                "interpretedStartSeconds": 0.5,
                "interpretedDurationSeconds": 0.0,
                "gridPosition": {
                    "measureId": "measure_0001",
                    "measureIndex": 0,
                    "beatIndex": 1,
                    "beatInMeasure": 2,
                    "subdivision": 4,
                    "subdivisionIndex": 0,
                    "alignedTimeSeconds": 0.5,
                    "offsetSeconds": 0.0,
                },
                "sourceKind": "drums",
                "hits": [
                    {
                        "sourceHitIndex": 1,
                        "rawKind": "closed_hihat",
                        "broadVoice": "closed_high_frequency",
                        "voiceId": "voice_hat",
                        "confidence": 0.76,
                    },
                    {
                        "sourceHitIndex": 0,
                        "rawKind": "kick",
                        "broadVoice": "low_drum",
                        "voiceId": "voice_kick",
                        "confidence": 0.9,
                        "alternatives": [
                            {
                                "id": "alt_kick_unresolved",
                                "kind": "broad_voice",
                                "confidence": 0.22,
                                "broadVoice": "unresolved_percussion",
                            }
                        ],
                    },
                ],
                "confidence": 0.84,
                "warnings": ["Kit-piece precision is intentionally broad."],
            }
        ],
        "alternatives": [
            {
                "id": "alt_top_note_voice",
                "subjectType": "pitched_item",
                "subjectId": "pitched_unassigned",
                "kind": "part_assignment",
                "confidence": 0.24,
                "sourceEventIds": ["p000002"],
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
            }
        ],
        "warnings": [
            "This is an editable interpretation draft, not final notation."
        ],
    }


def _json_payload(value: object) -> dict:
    return json.loads(json.dumps(asdict(value), allow_nan=False))


def _production_inputs() -> tuple[list[dict], list[dict], list[dict], dict]:
    pitched_events = [
        {
            "id": "p000001",
            "sourceKind": "vocals",
            "startSeconds": 0.1,
            "endSeconds": 0.6,
            "midiNote": 69,
            "midiPitch": 69.2,
            "frequencyHz": 445.1,
            "noteName": "A4",
            "confidence": 0.91,
            "warnings": [],
        },
        {
            "id": "p000003",
            "sourceKind": "vocals",
            "startSeconds": 0.65,
            "endSeconds": 0.66,
            "midiNote": 70,
            "midiPitch": 70.25,
            "frequencyHz": 470.0,
            "noteName": "A#4",
            "confidence": 0.25,
            "warnings": [],
        },
        {
            "id": "p000002",
            "sourceKind": "other",
            "startSeconds": 0.9,
            "endSeconds": 1.4,
            "midiNote": 72,
            "midiPitch": 72.17,
            "frequencyHz": 527.4,
            "noteName": "C5",
            "confidence": 0.45,
            "warnings": ["Source identity remains broad."],
        },
    ]
    percussion_events = [
        {
            "id": "r000001",
            "sourceKind": "drums",
            "timeSeconds": 0.5,
            "strength": 0.88,
            "hits": [
                {"kind": "kick", "confidence": 0.9},
                {"kind": "closed_hihat", "confidence": 0.76},
            ],
            "rawFeatureSummary": {
                "lowBandRatio": 0.6,
                "midBandRatio": 0.1,
                "highBandRatio": 0.3,
                "transientStrength": 0.88,
            },
            "warnings": ["Two simultaneous broad hits are plausible."],
        }
    ]
    alignments = [
        {
            "eventId": "p000001",
            "eventType": "pitched",
            "rawTimeSeconds": 0.1,
            "beatIndex": 0,
            "subdivision": 4,
            "subdivisionIndex": 0,
            "alignedTimeSeconds": 0.0,
            "offsetSeconds": 0.1,
            "confidence": 0.9,
            "measureIndex": 0,
            "beatInMeasure": 1,
        },
        {
            "eventId": "p000002",
            "eventType": "pitched",
            "rawTimeSeconds": 0.9,
            "confidence": 0.2,
            "warnings": ["No accepted grid point; raw time is unchanged."],
        },
        {
            "eventId": "r000001",
            "eventType": "percussion",
            "rawTimeSeconds": 0.5,
            "beatIndex": 1,
            "subdivision": 4,
            "subdivisionIndex": 0,
            "alignedTimeSeconds": 0.5,
            "offsetSeconds": 0.0,
            "confidence": 0.92,
            "measureIndex": 0,
            "beatInMeasure": 2,
        },
    ]
    timing = {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "beatConfidence": 0.9,
        "downbeatsSeconds": [0.0, 2.0],
        "meter": 4,
        "meterConfidence": 0.85,
    }
    return pitched_events, percussion_events, alignments, timing


def test_mixed_draft_round_trips_deterministically(settings: Settings) -> None:
    assert INTERPRETATION_DRAFT_SCHEMA_VERSION == 1
    assert INTERPRETATION_DRAFT_RELATIVE_PATH == "interpretation/draft.json"
    original = payload()
    snapshot = copy.deepcopy(original)
    expected = validate_transcription_draft(original)
    path = write_transcription_draft(JOB_ID, settings, original)
    assert original == snapshot
    assert path == (
        settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    ).resolve()
    assert load_transcription_draft(JOB_ID, settings) == expected
    first = path.read_bytes()
    write_transcription_draft(JOB_ID, settings, original)
    assert path.read_bytes() == first
    assert expected["interpretationEvidence"] == original["interpretationEvidence"]


def test_production_interpretation_evidence_round_trip_without_adapters(
    settings: Settings,
) -> None:
    pitched_events, percussion_events, alignments, timing = _production_inputs()
    pitched_result = infer_pitched_parts(
        pitched_events,
        [item for item in alignments if item["eventType"] == "pitched"],
    )
    percussion_result = interpret_percussion(
        percussion_events,
        [item for item in alignments if item["eventType"] == "percussion"],
    )
    rhythm_result = interpret_rhythm(
        pitched_events,
        percussion_events,
        alignments,
        timing,
    )
    evidence = {
        "pitchedPartInference": pitched_result.payload(),
        "percussionInterpretation": _json_payload(percussion_result),
        "rhythmInterpretation": _json_payload(rhythm_result),
    }
    value = payload()
    value["algorithms"]["pitchedPartInference"]["version"] = (
        PITCHED_PART_INFERENCE_VERSION
    )
    value["algorithms"]["percussionInterpretation"]["version"] = (
        PERCUSSION_INTERPRETATION_VERSION
    )
    value["algorithms"]["rhythmInterpretation"]["version"] = (
        RHYTHM_INTERPRETATION_VERSION
    )
    value["interpretationEvidence"] = evidence
    snapshot = copy.deepcopy(value)

    validated = validate_transcription_draft(value)
    assert value == snapshot
    assert validated["interpretationEvidence"] == evidence
    assert validated["interpretationEvidence"]["pitchedPartInference"] == (
        pitched_result.payload()
    )
    assert validated["interpretationEvidence"]["percussionInterpretation"] == (
        _json_payload(percussion_result)
    )
    assert validated["interpretationEvidence"]["rhythmInterpretation"] == (
        _json_payload(rhythm_result)
    )

    path = write_transcription_draft(JOB_ID, settings, value)
    reloaded = load_transcription_draft(JOB_ID, settings)
    assert reloaded == validated
    assert reloaded is not None
    assert reloaded["interpretationEvidence"] == evidence
    assert json.loads(path.read_text(encoding="utf-8"))["interpretationEvidence"] == evidence


def test_interpretation_evidence_is_required_and_version_bound() -> None:
    value = payload()
    del value["interpretationEvidence"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)

    value = payload()
    value["interpretationEvidence"]["rhythmInterpretation"]["version"] = (
        "different-version"
    )
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_all_evidence_raw_references_are_cross_validated() -> None:
    value = payload()
    value["interpretationEvidence"]["pitchedPartInference"]["assignments"][0][
        "eventId"
    ] = "missing"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)

    value = payload()
    value["interpretationEvidence"]["percussionInterpretation"]["groups"][0][
        "eventIds"
    ] = ["p000001"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)

    value = payload()
    value["interpretationEvidence"]["rhythmInterpretation"][
        "rest_candidates"
    ][0]["sourceEventIds"] = ["missing"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_raw_timing_uncertainty_and_simultaneous_hits_survive() -> None:
    result = validate_transcription_draft(payload())
    note = next(
        item for item in result["pitchedItems"] if item["id"] == "pitched_note"
    )
    unresolved = next(
        item
        for item in result["pitchedItems"]
        if item["id"] == "pitched_unassigned"
    )
    percussion = result["percussionItems"][0]
    assert note["rawStartSeconds"] == 0.1
    assert note["interpretedStartSeconds"] == 0.0
    assert len(note["alternatives"]) == 2
    assert unresolved["placementStatus"] == "unassigned"
    assert "interpretedStartSeconds" not in unresolved
    assert [hit["rawKind"] for hit in percussion["hits"]] == [
        "kick",
        "closed_hihat",
    ]
    assert percussion["voiceIds"] == ["voice_hat", "voice_kick"]
    evidence = result["interpretationEvidence"]
    assert evidence["pitchedPartInference"]["assignments"][0]["midiPitch"] == 69.2
    assert len(evidence["percussionInterpretation"]["assignments"]) == 2
    assert evidence["rhythmInterpretation"]["meter_candidates"]


def test_note_rest_tie_are_hypotheses_not_notation() -> None:
    result = validate_transcription_draft(payload())
    items = {item["id"]: item for item in result["pitchedItems"]}
    assert items["pitched_note"]["pitch"]["midiPitch"] == 69.2
    assert (
        items["pitched_note"]["tieCandidate"]["targetItemId"]
        == "pitched_unassigned"
    )
    assert items["pitched_rest"]["interpretationType"] == "rest"
    assert "pitch" not in items["pitched_rest"]
    assert "musicXml" not in json.dumps(result)


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True])
def test_non_finite_and_boolean_numbers_fail_safely(bad: object) -> None:
    value = payload()
    value["pitchedItems"][0]["confidence"] = bad
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_cross_references_and_global_ids_are_validated() -> None:
    value = payload()
    value["pitchedItems"][0]["voiceId"] = "missing_voice"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["voices"][0]["partId"] = "part_drums"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["voices"][0]["id"] = "part_vocals"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["alternatives"][0]["subjectId"] = "missing_item"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_source_ranges_and_primary_assignment_are_authoritative() -> None:
    value = payload()
    value["pitchedItems"][0]["rawStartSeconds"] = 0.2
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    duplicate = copy.deepcopy(value["pitchedItems"][0])
    duplicate["id"] = "pitched_note_duplicate"
    duplicate["alternatives"][0]["id"] = "alt_duplicate_short"
    duplicate["alternatives"][1]["id"] = "alt_duplicate_late"
    value["pitchedItems"].append(duplicate)
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value["pitchedItems"][0]["sharedEvidence"] = True
    value["pitchedItems"][3]["sharedEvidence"] = True
    assert validate_transcription_draft(value)["pitchedItems"]


def test_grid_interpreted_timing_and_measure_consistency() -> None:
    value = payload()
    del value["pitchedItems"][0]["gridPosition"]["subdivisionIndex"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["pitchedItems"][0]["gridPosition"]["offsetSeconds"] = -0.1
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    del value["pitchedItems"][0]["interpretedDurationSeconds"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    second = copy.deepcopy(value["measures"][0])
    second.update(
        id="measure_0002",
        index=2,
        rawStartSeconds=2.0,
        rawEndSeconds=4.0,
        interpretedStartSeconds=2.0,
    )
    value["measures"].append(second)
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_note_rest_tie_and_percussion_contracts_fail_safely() -> None:
    value = payload()
    del value["pitchedItems"][0]["pitch"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["pitchedItems"][1]["pitch"] = {"midiNote": 60}
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["pitchedItems"][0]["tieCandidate"]["targetItemId"] = "pitched_rest"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["percussionItems"][0]["hits"] = []
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["percussionItems"][0]["sourceEventIds"] = ["p000001"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


@pytest.mark.parametrize(
    "bad",
    [
        "/transcription/raw-events.json",
        "../transcription/raw-events.json",
        "%2e%2e/transcription/raw-events.json",
        "%252e%252e/transcription/raw-events.json",
        "C:\\transcription\\raw-events.json",
        "transcription\\raw-events.json",
        "transcription/raw-events.json\x00",
    ],
)
def test_source_path_is_canonical_and_safe(bad: str) -> None:
    value = payload()
    value["sourceTranscription"]["fileName"] = bad
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_private_raw_and_final_notation_metadata_are_rejected() -> None:
    for key, bad in (
        ("debug", "/home/runner/private/model.bin"),
        ("audioSamples", [0.1]),
        ("tensor", [1, 2]),
        ("userCorrections", []),
        ("musicXml", "<score-partwise/>"),
    ):
        value = payload()
        value["algorithms"]["rhythmInterpretation"][key] = bad
        with pytest.raises(TranscriptionDraftValidationError):
            validate_transcription_draft(value)

    value = payload()
    value["interpretationEvidence"]["rhythmInterpretation"]["diagnostics"][
        "tensor"
    ] = [1, 2]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)


def test_schema_versions_utc_and_bounds_are_enforced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = payload()
    value["schemaVersion"] = 2
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["createdAt"] = "2026-08-06T13:00:00+08:00"
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    del value["algorithms"]["rhythmInterpretation"]["version"]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    value = payload()
    value["pitchedItems"][0]["alternatives"] = [
        {"id": f"alt_{index}", "kind": "duration", "confidence": 0.1}
        for index in range(17)
    ]
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(value)
    import app.transcription_draft as module

    monkeypatch.setattr(module, "_MAX_ARTIFACT_BYTES", 1024)
    with pytest.raises(TranscriptionDraftValidationError):
        validate_transcription_draft(payload())


def test_load_missing_and_corruption(settings: Settings) -> None:
    assert load_transcription_draft(JOB_ID, settings) is None
    path = write_transcription_draft(JOB_ID, settings, payload())
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["pitchedItems"][0]["voiceId"] = "missing_voice"
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(TranscriptionDraftError):
        load_transcription_draft(JOB_ID, settings)
    write_transcription_draft(JOB_ID, settings, payload())
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"confidence": 0.91', '"confidence": NaN', 1
        ),
        encoding="utf-8",
    )
    with pytest.raises(TranscriptionDraftError):
        load_transcription_draft(JOB_ID, settings)


def test_failed_replacement_preserves_previous_and_cleans_temp(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = write_transcription_draft(JOB_ID, settings, payload())
    before = path.read_bytes()
    import app.transcription_draft as module

    monkeypatch.setattr(
        module.os,
        "replace",
        lambda source, destination: (_ for _ in ()).throw(OSError("injected")),
    )
    replacement = payload()
    replacement["draftVersion"] = "editable-interpretation-v2"
    with pytest.raises(TranscriptionDraftError):
        write_transcription_draft(JOB_ID, settings, replacement)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".draft.json.*.tmp"))


def test_symlink_and_replacement_boundary_attacks_are_rejected(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transcription_draft as module

    outside = tmp_path / "outside"
    outside.mkdir()
    directory = settings.exports_dir / JOB_ID / "interpretation"
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(TranscriptionDraftError):
        write_transcription_draft(JOB_ID, settings, payload())
    directory.unlink()
    path = write_transcription_draft(JOB_ID, settings, payload())
    artifact_dir = path.parent
    job_dir = artifact_dir.parent
    original_replace = module._replace_atomic

    def swap_parent(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        artifact_dir.rename(job_dir / "interpretation-replaced")
        artifact_dir.mkdir(mode=0o700)

    monkeypatch.setattr(module, "_replace_atomic", swap_parent)
    with pytest.raises(TranscriptionDraftError):
        write_transcription_draft(JOB_ID, settings, payload())


def test_final_object_substitution_is_rejected(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.transcription_draft as module

    path = write_transcription_draft(JOB_ID, settings, payload())
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    original_replace = module._replace_atomic

    def replace_with_symlink(source: Path, destination: Path) -> None:
        original_replace(source, destination)
        destination.unlink()
        destination.symlink_to(outside)

    monkeypatch.setattr(module, "_replace_atomic", replace_with_symlink)
    with pytest.raises(TranscriptionDraftError):
        write_transcription_draft(JOB_ID, settings, payload())
    assert outside.read_text(encoding="utf-8") == "outside"
    assert not list(path.parent.glob(".draft.json.*.tmp"))


def test_invalid_job_id_fails_without_write(settings: Settings) -> None:
    with pytest.raises(TranscriptionDraftValidationError):
        write_transcription_draft("../bad", settings, payload())
    assert not (settings.exports_dir / "bad").exists()
