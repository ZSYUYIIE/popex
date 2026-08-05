from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from app.config import Settings
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
            },
            {
                "id": "r000001",
                "sourceKind": "full_mix",
                "collection": "detected_hits",
                "timeSeconds": 0.5,
                "strength": 0.93,
                "hits": [
                    {
                        "kind": "future_kick_variant",
                        "confidence": 0.88,
                    }
                ],
            },
        ],
        "alignmentCandidates": [
            {
                "eventId": "r000002",
                "rawTimeSeconds": 1.0,
                "beatIndex": 2,
                "subdivision": "1/16",
                "alignedTimeSeconds": 1.01,
                "offsetSeconds": 0.01,
                "confidence": 0.7,
            },
            {
                "eventId": "p000001",
                "rawTimeSeconds": 0.125001,
                "beatIndex": 0,
                "subdivision": 16,
                "alignedTimeSeconds": 0.125,
                "offsetSeconds": -0.000001,
                "confidence": 0.97,
            },
        ],
        "warnings": ["Percussion labels are preliminary."],
    }


def test_valid_mixed_payload_round_trips_deterministically(
    settings: Settings,
) -> None:
    original = payload()
    expected = validate_raw_transcription(original)

    path = write_raw_transcription(JOB_ID, settings, original)

    assert path == (
        settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    ).resolve()
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
    assert [item["id"] for item in result["pitchedNoteEvents"]] == [
        "p000001",
        "p000002",
    ]
    assert [item["id"] for item in result["percussionEvents"]] == [
        "r000001",
        "r000002",
    ]
    assert list(result["algorithms"]) == [
        "percussionDetection",
        "pitchTracking",
    ]


def test_raw_times_and_pitch_remain_unquantized() -> None:
    result = validate_raw_transcription(payload())
    pitched = {item["id"]: item for item in result["pitchedNoteEvents"]}
    alignment = {
        item["eventId"]: item for item in result["alignmentCandidates"]
    }

    assert pitched["p000001"]["startSeconds"] == 0.125001
    assert pitched["p000001"]["endSeconds"] == 0.749999
    assert pitched["p000001"]["midiPitch"] == 40.187654
    assert alignment["p000001"]["rawTimeSeconds"] == pitched["p000001"][
        "startSeconds"
    ]
    assert (
        alignment["p000001"]["alignedTimeSeconds"]
        != alignment["p000001"]["rawTimeSeconds"]
    )


def test_simultaneous_percussion_hits_are_distinct() -> None:
    result = validate_raw_transcription(payload())
    event = next(
        item for item in result["percussionEvents"] if item["id"] == "r000002"
    )

    assert {item["kind"] for item in event["hits"]} == {
        "snare",
        "closed_hat",
    }
    assert len(event["hits"]) == 2


def test_source_separation_is_optional() -> None:
    value = payload()
    del value["sourceSeparation"]

    result = validate_raw_transcription(value)

    assert "sourceSeparation" not in result


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
    value["pitchedNoteEvents"][0]["endSeconds"] = value[
        "pitchedNoteEvents"
    ][0]["startSeconds"]
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


@pytest.mark.parametrize(
    "bad",
    ["bad id", "../p1", "P000001", "/tmp/p1", "p%2f1"],
)
def test_unsafe_event_ids_fail(bad: str) -> None:
    value = payload()
    value["pitchedNoteEvents"][0]["id"] = bad

    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


@pytest.mark.parametrize(
    "bad",
    ["bad kind", "Vocals", "../vocals", "vocals/type", "vocals%2ftype"],
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
    value["percussionEvents"][0]["hits"][0][
        "kind"
    ] = "new_percussion_family"
    value["algorithms"]["futureComponent"] = {
        "version": "experimental-2026.1"
    }

    result = validate_raw_transcription(value)

    assert result["pitchedNoteEvents"][1]["sourceKind"] == "future_stem"
    assert "futureComponent" in result["algorithms"]


def test_duplicate_ids_fail_within_and_across_collections() -> None:
    value = payload()
    value["pitchedNoteEvents"][1]["id"] = value["pitchedNoteEvents"][0][
        "id"
    ]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)

    value = payload()
    value["percussionEvents"][0]["id"] = value["pitchedNoteEvents"][0][
        "id"
    ]
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
def test_analysis_paths_reject_absolute_traversal_encoded_and_nul(
    bad: str,
) -> None:
    value = payload()
    value["sourceAnalysis"]["fileName"] = bad

    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_separation_path_must_be_canonical() -> None:
    value = payload()
    value["sourceSeparation"][
        "fileName"
    ] = "stems/runs/private/stem-separation.json"

    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_machine_paths_and_raw_data_do_not_fit_open_metadata() -> None:
    value = payload()
    value["algorithms"]["pitchTracking"][
        "debug"
    ] = "/home/runner/private/model.bin"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)

    value = payload()
    value["sourceSeparation"]["model"]["tensor"] = [1, 2, 3]
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)

    value = payload()
    value["percussionEvents"][0]["rawFeatureSummary"]["audioSamples"] = [
        0.1
    ]
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


def test_alignment_reference_and_raw_time_must_match() -> None:
    value = payload()
    value["alignmentCandidates"][0]["eventId"] = "r999999"
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)

    value = payload()
    value["alignmentCandidates"][0]["rawTimeSeconds"] = 1.001
    with pytest.raises(RawTranscriptionValidationError):
        validate_raw_transcription(value)


def test_alignment_offset_must_match_advisory_time() -> None:
    value = payload()
    value["alignmentCandidates"][0]["offsetSeconds"] = 0.2

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
        path.read_text(encoding="utf-8").replace(
            '"confidence": 0.91',
            '"confidence": NaN',
        ),
        encoding="utf-8",
    )

    with pytest.raises(RawTranscriptionError):
        load_raw_transcription(JOB_ID, settings)


def test_failed_replacement_preserves_previous_and_removes_temp(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
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
    assert load_raw_transcription(JOB_ID, settings) == validate_raw_transcription(
        first
    )
    assert not list(path.parent.glob(".raw-events.json.*.tmp"))


def test_symlinked_transcription_directory_is_rejected(
    settings: Settings,
    tmp_path: Path,
) -> None:
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


def test_symlinked_artifact_is_rejected(
    settings: Settings,
    tmp_path: Path,
) -> None:
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
