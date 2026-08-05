from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.event_alignment import align_raw_events_to_timing
from app.percussion_transcription import transcribe_percussion_audio
from app.pitch_transcription import transcribe_pitched_audio
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    RawTranscriptionValidationError,
    load_raw_transcription,
    validate_raw_transcription,
    write_raw_transcription,
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


def payload() -> dict:
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "raw-events-v1",
        "createdAt": "2026-08-05T08:00:00+00:00",
        "sourceAnalysis": {
            "fileName": "analysis/audio-analysis.json",
            "analysisVersion": "baseline-librosa-v1",
        },
        "sourceSeparation": {
            "fileName": "stems/stem-separation.json",
            "separationVersion": "demucs-worker-v3",
            "model": {
                "name": "htdemucs",
                "repository": "adefossez/HTDemucs",
                "revision": "b" * 40,
                "checkpointFile": "955717e8.safetensors",
                "checkpointSha256": "c" * 64,
                "device": "cpu",
            },
        },
        "algorithms": {
            "pitchTracking": {
                "version": "pyin-v1",
                "frameLength": 2048,
                "voicing": {"threshold": 0.42},
            },
            "percussionDetection": {
                "version": "onset-v2",
                "multiHit": True,
            },
            "eventAlignment": {
                "version": "advisory-beat-grid-v1",
                "rawTimesPreserved": True,
            },
        },
        "pitchedNoteEvents": [
            {
                "id": "p000002",
                "sourceKind": "other",
                "collection": "lead_notes",
                "startSeconds": 1.23456789,
                "endSeconds": 1.7654321,
                "midiNote": 64,
                "midiPitch": 63.842731,
                "frequencyHz": 326.8142,
                "noteName": "E4",
                "confidence": 0.83,
                "velocity": 92,
                "warnings": ["Pitch bends during this event."],
            },
            {
                "id": "p000001",
                "sourceKind": "bass",
                "startSeconds": 0.125001,
                "endSeconds": 0.749999,
                "midiNote": 40,
                "midiPitch": 40.187654,
                "frequencyHz": 83.31,
                "noteName": "E2",
                "confidence": 0.91,
            },
        ],
        "percussionEvents": [
            {
                "id": "r000002",
                "sourceKind": "drums",
                "timeSeconds": 1.0,
                "strength": 0.8,
                "hits": [
                    {"kind": "snare", "confidence": 0.75},
                    {"kind": "closed_hat", "confidence": 0.92},
                ],
                "rawFeatureSummary": {
                    "spectralFlux": 0.72,
                    "band": "mid",
                },
                "warnings": ["Two broad hit families are plausible."],
            },
            {
                "id": "r000001",
                "sourceKind": "full_mix",
                "collection": "detected_hits",
                "timeSeconds": 0.5,
                "strength": 0.93,
                "hits": [
                    {"kind": "future_kick_variant", "confidence": 0.88}
                ],
            },
        ],
        "alignmentCandidates": [
            {
                "eventId": "r000002",
                "eventType": "percussion",
                "rawTimeSeconds": 1.0,
                "beatIndex": 2,
                "subdivision": 4,
                "subdivisionIndex": 0,
                "alignedTimeSeconds": 1.01,
                "offsetSeconds": -0.01,
                "confidence": 0.7,
                "measureIndex": 0,
                "beatInMeasure": 3,
                "warnings": ["Timing evidence is weak; review this candidate."],
            },
            {
                "eventId": "p000001",
                "eventType": "pitched",
                "rawTimeSeconds": 0.125001,
                "beatIndex": 0,
                "subdivision": 4,
                "subdivisionIndex": 1,
                "alignedTimeSeconds": 0.125,
                "offsetSeconds": 0.000001,
                "confidence": 0.97,
            },
        ],
        "warnings": ["Percussion labels are preliminary."],
    }


def test_valid_mixed_payload_round_trips_deterministically(settings: Settings) -> None:
    original = payload()
    expected = validate_raw_transcription(original)
    path = write_raw_transcription(JOB_ID, settings, original)
    assert path == (settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH).resolve()
    assert load_raw_transcription(JOB_ID, settings) == expected
    first = path.read_bytes()
    write_raw_transcription(JOB_ID, settings, original)
    assert path.read_bytes() == first
    assert json.loads(first) == expected


def test_validation_sorts_copies_without_mutating_caller() -> None:
    original = payload()
    snapshot = copy.deepcopy(original)
    result = validate_raw_transcription(original)
    assert original == snapshot
    assert [item["id"] for item in result["pitchedNoteEvents"]] == ["p000001", "p000002"]
    assert [item["id"] for item in result["percussionEvents"]] == ["r000001", "r000002"]
    assert list(result["algorithms"]) == [
        "eventAlignment",
        "percussionDetection",
        "pitchTracking",
    ]


def test_raw_times_pitch_features_and_alignment_metadata_remain_unmodified() -> None:
    result = validate_raw_transcription(payload())
    pitched = {item["id"]: item for item in result["pitchedNoteEvents"]}
    percussion = {item["id"]: item for item in result["percussionEvents"]}
    alignment = {item["eventId"]: item for item in result["alignmentCandidates"]}
    assert pitched["p000001"]["startSeconds"] == 0.125001
    assert pitched["p000001"]["endSeconds"] == 0.749999
    assert pitched["p000001"]["midiPitch"] == 40.187654
    assert percussion["r000002"]["rawFeatureSummary"] == {
        "band": "mid",
        "spectralFlux": 0.72,
    }
    assert alignment["p000001"]["rawTimeSeconds"] == pitched["p000001"]["startSeconds"]
    assert alignment["r000002"]["subdivisionIndex"] == 0
    assert alignment["r000002"]["measureIndex"] == 0
    assert alignment["r000002"]["beatInMeasure"] == 3
    assert alignment["r000002"]["warnings"]


def test_simultaneous_percussion_hits_are_distinct() -> None:
    result = validate_raw_transcription(payload())
    event = next(item for item in result["percussionEvents"] if item["id"] == "r000002")
    assert {item["kind"] for item in event["hits"]} == {"snare", "closed_hat"}
    assert len(event["hits"]) == 2


def test_source_separation_is_optional() -> None:
    value = payload()
    del value["sourceSeparation"]
    assert "sourceSeparation" not in validate_raw_transcription(value)


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("startSeconds", math.nan),
        ("endSeconds", math.inf),
        ("midiPitch", -math.inf),
        ("frequencyHz", True),
        ("confidence", False),
        ("midiNote", True),
    ],
)
def test_pitched_invalid_numbers_fail(field: str, bad: object) -> None:
    value = payload()
    value["pitchedNoteEvents"][0][field] = bad
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize("bad", [math.nan, math.inf, True, -0.1, 1.1])
def test_percussion_strength_fails_safely(bad: object) -> None:
    value = payload()
    value["percussionEvents"][0]["strength"] = bad
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_invalid_ranges_fail() -> None:
    value = payload()
    value["pitchedNoteEvents"][0]["endSeconds"] = value["pitchedNoteEvents"][0]["startSeconds"]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["pitchedNoteEvents"][0]["midiNote"] = 128
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["pitchedNoteEvents"][0]["velocity"] = 0
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize("bad", ["bad id", "../p1", "P000001", "/tmp/p1", "p%2f1"])
def test_unsafe_event_ids_fail(bad: str) -> None:
    value = payload()
    value["pitchedNoteEvents"][0]["id"] = bad
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize(
    "bad", ["bad kind", "Vocals", "../vocals", "vocals/type", "vocals%2ftype"]
)
def test_unsafe_open_slugs_fail(bad: str) -> None:
    value = payload()
    value["pitchedNoteEvents"][0]["sourceKind"] = bad
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_future_open_source_collection_algorithm_and_hit_kind_are_allowed() -> None:
    value = payload()
    value["pitchedNoteEvents"][0]["sourceKind"] = "future_stem"
    value["pitchedNoteEvents"][0]["collection"] = "microtonal_lead"
    value["percussionEvents"][0]["hits"][0]["kind"] = "new_percussion_family"
    value["algorithms"]["futureComponent"] = {"version": "experimental-2026.1"}
    result = validate_raw_transcription(value)
    assert result["pitchedNoteEvents"][1]["sourceKind"] == "future_stem"
    assert "futureComponent" in result["algorithms"]


def test_duplicate_ids_fail_within_and_across_collections() -> None:
    value = payload()
    value["pitchedNoteEvents"][1]["id"] = value["pitchedNoteEvents"][0]["id"]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["percussionEvents"][0]["id"] = value["pitchedNoteEvents"][0]["id"]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize(
    "bad",
    [
        "/analysis/audio-analysis.json",
        "../analysis/audio-analysis.json",
        "%2e%2e/analysis/audio-analysis.json",
        "%252e%252e/analysis/audio-analysis.json",
        "C:\\analysis\\audio-analysis.json",
        "analysis\\audio-analysis.json",
        "analysis/audio-analysis.json\x00",
    ],
)
def test_analysis_paths_reject_absolute_traversal_encoded_and_nul(bad: str) -> None:
    value = payload()
    value["sourceAnalysis"]["fileName"] = bad
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_separation_path_must_be_canonical() -> None:
    value = payload()
    value["sourceSeparation"]["fileName"] = "stems/runs/private/stem-separation.json"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_machine_paths_and_raw_private_data_do_not_fit_open_metadata() -> None:
    value = payload()
    value["algorithms"]["pitchTracking"]["debug"] = "/home/runner/private/model.bin"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["sourceSeparation"]["model"]["tensor"] = [1, 2, 3]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["percussionEvents"][0]["rawFeatureSummary"]["audioSamples"] = [0.1]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_unknown_top_level_fields_and_schema_fail() -> None:
    value = payload()
    value["schemaVersion"] = 2
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["rawTensor"] = []
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_created_at_must_be_utc_iso() -> None:
    value = payload()
    value["createdAt"] = "2026-08-05T16:00:00+08:00"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value["createdAt"] = "not-a-date"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_algorithm_records_require_versions() -> None:
    value = payload()
    del value["algorithms"]["pitchTracking"]["version"]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_alignment_reference_raw_time_and_event_type_must_match() -> None:
    value = payload()
    value["alignmentCandidates"][0]["eventId"] = "r999999"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["alignmentCandidates"][0]["rawTimeSeconds"] = 1.001
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["alignmentCandidates"][0]["eventType"] = "pitched"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_alignment_offset_uses_raw_minus_aligned_convention() -> None:
    result = validate_raw_transcription(payload())
    item = next(x for x in result["alignmentCandidates"] if x["eventId"] == "r000002")
    assert item["offsetSeconds"] == pytest.approx(
        item["rawTimeSeconds"] - item["alignedTimeSeconds"]
    )
    value = payload()
    value["alignmentCandidates"][0]["offsetSeconds"] = 0.01
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("subdivisionIndex"),
        lambda c: c.update(subdivisionIndex=-1),
        lambda c: c.update(subdivisionIndex=4),
        lambda c: c.pop("offsetSeconds"),
        lambda c: c.pop("alignedTimeSeconds"),
    ],
)
def test_alignment_grid_and_time_fields_must_be_complete_and_consistent(mutate) -> None:
    value = payload()
    mutate(value["alignmentCandidates"][0])
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c.pop("beatInMeasure"),
        lambda c: c.pop("measureIndex"),
        lambda c: c.update(measureIndex=-1),
        lambda c: c.update(beatInMeasure=0),
    ],
)
def test_measure_fields_are_paired_and_use_expected_indices(mutate) -> None:
    value = payload()
    mutate(value["alignmentCandidates"][0])
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_unaligned_merged_candidate_with_warning_is_accepted() -> None:
    value = payload()
    value["alignmentCandidates"] = [
        {
            "eventId": "p000002",
            "eventType": "pitched",
            "rawTimeSeconds": 1.23456789,
            "confidence": 0.0,
            "warnings": [
                "No beat-grid point is within the local acceptance window; the raw time is unchanged."
            ],
        }
    ]
    result = validate_raw_transcription(value)["alignmentCandidates"][0]
    assert result["eventType"] == "pitched"
    assert "alignedTimeSeconds" not in result
    assert result["warnings"]


def test_candidate_warning_bounds_are_enforced() -> None:
    value = payload()
    value["alignmentCandidates"][0]["warnings"] = ["x"] * 9
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)
    value = payload()
    value["alignmentCandidates"][0]["warnings"] = ["x" * 501]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_load_missing_returns_none(settings: Settings) -> None:
    assert load_raw_transcription(JOB_ID, settings) is None


def test_load_revalidates_stored_artifact(settings: Settings) -> None:
    path = write_raw_transcription(JOB_ID, settings, payload())
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["pitchedNoteEvents"][0]["confidence"] = 2
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(RawTranscriptionError):
        load_raw_transcription(JOB_ID, settings)


def test_load_rejects_nan_json_constant(settings: Settings) -> None:
    path = write_raw_transcription(JOB_ID, settings, payload())
    path.write_text(
        path.read_text(encoding="utf-8").replace('"confidence": 0.91', '"confidence": NaN'),
        encoding="utf-8",
    )
    with pytest.raises(RawTranscriptionError):
        load_raw_transcription(JOB_ID, settings)


def test_failed_replacement_preserves_previous_and_removes_temp(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = payload()
    path = write_raw_transcription(JOB_ID, settings, first)
    before = path.read_bytes()
    replacement = payload()
    replacement["transcriptionVersion"] = "raw-events-v2"

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected replacement failure")

    import app.transcription_events as module

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(RawTranscriptionError):
        write_raw_transcription(JOB_ID, settings, replacement)
    assert path.read_bytes() == before
    assert load_raw_transcription(JOB_ID, settings) == validate_raw_transcription(first)
    assert not list(path.parent.glob(".raw-events.json.*.tmp"))


def test_symlinked_transcription_directory_is_rejected(settings: Settings, tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = settings.exports_dir / JOB_ID / "transcription"
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(RawTranscriptionError):
        write_raw_transcription(JOB_ID, settings, payload())
    assert not (outside / "raw-events.json").exists()


def test_symlinked_artifact_is_rejected(settings: Settings, tmp_path: Path) -> None:
    directory = settings.exports_dir / JOB_ID / "transcription"
    directory.mkdir()
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    target = directory / "raw-events.json"
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(RawTranscriptionError):
        write_raw_transcription(JOB_ID, settings, payload())
    assert outside.read_text(encoding="utf-8") == "{}"


def test_invalid_job_ids_fail_without_writes(settings: Settings) -> None:
    with pytest.raises(RawTranscriptionValidationError):
        write_raw_transcription("../bad", settings, payload())
    assert not (settings.exports_dir / "bad").exists()


def test_warning_bounds_are_enforced() -> None:
    value = payload()
    value["warnings"] = ["x" * 501]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_empty_alignment_candidates_are_allowed() -> None:
    value = payload()
    value["alignmentCandidates"] = []
    assert validate_raw_transcription(value)["alignmentCandidates"] == []


def _tone(frequency: float, duration: float, sample_rate: int = 22_050) -> np.ndarray:
    count = int(round(duration * sample_rate))
    times = np.arange(count, dtype=np.float64) / sample_rate
    audio = 0.35 * np.sin(2.0 * np.pi * frequency * times)
    fade = min(int(0.02 * sample_rate), count // 4)
    if fade:
        envelope = np.ones(count)
        envelope[:fade] = np.linspace(0.0, 1.0, fade, endpoint=False)
        envelope[-fade:] = np.linspace(1.0, 0.0, fade, endpoint=True)
        audio *= envelope
    return audio.astype(np.float32)


def _percussion_signal(sample_rate: int = 44_100) -> np.ndarray:
    duration = 2.0
    audio = np.zeros(int(duration * sample_rate), dtype=np.float32)
    start = int(0.5 * sample_rate)
    low_length = int(0.18 * sample_rate)
    low_time = np.arange(low_length, dtype=np.float64) / sample_rate
    low = 0.8 * np.sin(2.0 * np.pi * 70.0 * low_time) * np.exp(-20.0 * low_time)
    high_length = int(0.04 * sample_rate)
    generator = np.random.default_rng(29)
    noise = generator.standard_normal(high_length)
    spectrum = np.fft.rfft(noise)
    frequencies = np.fft.rfftfreq(high_length, d=1.0 / sample_rate)
    spectrum[(frequencies < 7_000.0) | (frequencies > 16_000.0)] = 0
    high = np.fft.irfft(spectrum, n=high_length)
    high /= max(float(np.max(np.abs(high))), np.finfo(float).eps)
    high *= 0.65 * np.exp(-30.0 * np.arange(high_length) / sample_rate)
    audio[start : start + low_length] += low.astype(np.float32)
    audio[start : start + high_length] += high.astype(np.float32)
    return audio


def test_real_detectors_alignment_and_artifact_round_trip_without_adapters(
    settings: Settings, tmp_path: Path
) -> None:
    pitched_path = tmp_path / "pitched.wav"
    percussion_path = tmp_path / "percussion.wav"
    sf.write(pitched_path, _tone(445.0, 1.2), 22_050, format="WAV", subtype="PCM_16")
    sf.write(
        percussion_path,
        _percussion_signal(),
        44_100,
        format="WAV",
        subtype="FLOAT",
    )

    pitched = transcribe_pitched_audio(pitched_path, source_kind="vocals")
    percussion = transcribe_percussion_audio(percussion_path, source_kind="drums")
    assert pitched["events"]
    assert percussion["events"]
    assert all("sourceKind" in event for event in percussion["events"])
    assert all("features" not in event for event in percussion["events"])
    assert all("rawFeatureSummary" in event for event in percussion["events"])

    timing = {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.92,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0],
        "beatConfidence": 0.94,
        "downbeatsSeconds": [0.0, 2.0],
        "meter": 4,
        "meterConfidence": 0.9,
    }
    alignment = align_raw_events_to_timing(
        pitched["events"], percussion["events"], timing
    )
    complete = {
        "schemaVersion": 1,
        "transcriptionVersion": "cycle-4a-integrated-v1",
        "createdAt": "2026-08-05T09:45:00+00:00",
        "sourceAnalysis": {
            "fileName": "analysis/audio-analysis.json",
            "analysisVersion": "baseline-librosa-v1",
        },
        "algorithms": {
            "pitchTracking": {
                "version": pitched["algorithmVersion"],
                "sourceKind": pitched["sourceKind"],
                "warnings": pitched["warnings"],
            },
            "percussionDetection": {
                "version": percussion["algorithmVersion"],
                "sourceKind": percussion["sourceKind"],
                "warnings": percussion["warnings"],
            },
            "eventAlignment": {
                "version": alignment["alignmentVersion"],
                "warnings": alignment["warnings"],
                "diagnostics": alignment["diagnostics"],
            },
        },
        "pitchedNoteEvents": pitched["events"],
        "percussionEvents": percussion["events"],
        "alignmentCandidates": alignment["candidates"],
        "warnings": [
            *pitched["warnings"],
            *percussion["warnings"],
            *alignment["warnings"],
        ],
    }
    before = copy.deepcopy(complete)
    validated = validate_raw_transcription(complete)
    assert complete == before
    assert validated["pitchedNoteEvents"] == pitched["events"]
    assert validated["percussionEvents"] == percussion["events"]
    assert validated["alignmentCandidates"] == alignment["candidates"]
    assert validated["algorithms"]["pitchTracking"]["version"] == pitched["algorithmVersion"]
    assert validated["algorithms"]["percussionDetection"]["version"] == percussion["algorithmVersion"]
    assert validated["algorithms"]["eventAlignment"]["version"] == alignment["alignmentVersion"]
    assert all(
        candidate["rawTimeSeconds"]
        == (
            next(event for event in pitched["events"] if event["id"] == candidate["eventId"])["startSeconds"]
            if candidate["eventType"] == "pitched"
            else next(event for event in percussion["events"] if event["id"] == candidate["eventId"])["timeSeconds"]
        )
        for candidate in validated["alignmentCandidates"]
    )
    for candidate in validated["alignmentCandidates"]:
        if "alignedTimeSeconds" in candidate:
            assert candidate["offsetSeconds"] == pytest.approx(
                candidate["rawTimeSeconds"] - candidate["alignedTimeSeconds"]
            )
            assert "subdivisionIndex" in candidate
    path = write_raw_transcription(JOB_ID, settings, complete)
    assert load_raw_transcription(JOB_ID, settings) == validated
    assert json.loads(path.read_text(encoding="utf-8")) == validated
