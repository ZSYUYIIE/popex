from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from app.config import Settings
from app.transcription_artifacts import (
    TranscriptionArtifactError,
    TranscriptionArtifactUnavailableError,
    load_transcription_details,
    transcription_json_path,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    validate_raw_transcription,
    write_raw_transcription,
)


JOB_ID = "d" * 32
CREATED_AT = "2026-08-06T01:30:00+00:00"


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


def artifact_payload() -> dict:
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "baseline-pyin-onset-v1",
        "createdAt": CREATED_AT,
        "sourceAnalysis": {
            "fileName": "analysis/audio-analysis.json",
            "analysisVersion": "baseline-librosa-v1",
        },
        "algorithms": {
            "eventAlignment": {
                "version": "advisory-beat-grid-v1",
                "rawTimesPreserved": True,
            },
            "percussionDetection": {
                "version": "baseline-onset-bands-v1",
                "multiHit": True,
            },
            "pitchTracking": {
                "version": "baseline-pyin-v1",
                "fractionalPitch": True,
            },
        },
        "pitchedNoteEvents": [
            {
                "id": "p000002",
                "sourceKind": "vocals",
                "startSeconds": 1.125001,
                "endSeconds": 1.625009,
                "midiNote": 69,
                "midiPitch": 69.1732,
                "frequencyHz": 444.42,
                "noteName": "A4",
                "confidence": 0.81,
                "warnings": ["Pitch bends during this candidate."],
            },
            {
                "id": "p000001",
                "sourceKind": "bass",
                "startSeconds": 0.125001,
                "endSeconds": 0.750003,
                "midiNote": 40,
                "midiPitch": 40.187654,
                "frequencyHz": 83.31,
                "noteName": "E2",
                "confidence": 0.91,
                "rawFeatureSummary": {
                    "voicedProbability": 0.91,
                },
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
                    {"kind": "closed_hihat", "confidence": 0.92},
                ],
                "rawFeatureSummary": {
                    "lowBandRatio": 0.1,
                    "midBandRatio": 0.55,
                    "highBandRatio": 0.35,
                },
                "warnings": [
                    "Independent spectral bands support simultaneous broad hit candidates."
                ],
            },
            {
                "id": "r000001",
                "sourceKind": "full_mix",
                "timeSeconds": 0.5,
                "strength": 0.93,
                "hits": [
                    {"kind": "unknown_percussion", "confidence": 0.48}
                ],
            },
        ],
        "alignmentCandidates": [
            {
                "eventId": "p000001",
                "eventType": "pitched",
                "rawTimeSeconds": 0.125001,
                "confidence": 0.0,
                "warnings": [
                    "No beat-grid point is within the local acceptance window; the raw time is unchanged."
                ],
            },
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
                "warnings": [
                    "Timing evidence is weak; review this candidate."
                ],
            },
        ],
        "warnings": [
            "Raw events are candidates and have not been converted into notation."
        ],
    }


def job_record(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "id": JOB_ID,
        "transcription_status": "completed",
        "transcription_stage": "completed",
        "transcription_progress": 100,
        "transcription_message": "Raw event candidates are ready.",
        "transcription_version": "untrusted-db-version",
        "transcription_artifact_file_name": RAW_TRANSCRIPTION_RELATIVE_PATH,
        "transcribed_at": "2026-01-01T00:00:00+00:00",
        "pitched_event_count": 999,
        "percussion_event_count": 998,
        "aligned_event_count": 997,
        "transcription_error": None,
    }
    value.update(overrides)
    return value


def publish(settings: Settings) -> dict:
    payload = artifact_payload()
    write_raw_transcription(JOB_ID, settings, payload)
    return validate_raw_transcription(payload)


def test_completed_details_derive_counts_sources_and_provenance_from_artifact(
    settings: Settings,
) -> None:
    expected = publish(settings)
    job = job_record(
        private_debug_path=str(settings.data_dir / "private" / "trace.log"),
        arbitrary_database_field={"tensor": [1, 2, 3]},
    )
    job_before = copy.deepcopy(job)

    details = load_transcription_details(JOB_ID, settings, job)
    payload = details.payload()

    assert job == job_before
    assert details.available is True
    assert details.status == "completed"
    assert details.stage == "completed"
    assert details.progress == 100
    assert details.transcription_version == expected["transcriptionVersion"]
    assert details.transcribed_at == expected["createdAt"]
    assert details.pitched_event_count == 2
    assert details.percussion_event_count == 2
    assert details.aligned_event_count == 1
    assert details.source_kinds == ("bass", "drums", "full_mix", "vocals")
    assert details.algorithms == expected["algorithms"]
    assert details.warnings == tuple(expected["warnings"])
    assert payload["counts"] == {
        "pitched": 2,
        "percussion": 2,
        "aligned": 1,
    }
    assert payload["pitchedNoteEvents"] == expected["pitchedNoteEvents"]
    assert payload["percussionEvents"] == expected["percussionEvents"]
    assert payload["alignmentCandidates"] == expected["alignmentCandidates"]
    assert len(payload["alignmentCandidates"]) == 2
    assert sum(
        "alignedTimeSeconds" in candidate
        for candidate in payload["alignmentCandidates"]
    ) == 1
    assert any(
        "alignedTimeSeconds" not in candidate
        for candidate in payload["alignmentCandidates"]
    )
    assert payload["downloadFileName"] == "raw-transcription.json"
    encoded = json.dumps(payload, allow_nan=False)
    assert str(settings.data_dir) not in encoded
    assert "private_debug_path" not in encoded
    assert "arbitrary_database_field" not in encoded
    assert "score" not in payload
    assert "notation" not in payload


def test_summary_mode_omits_events_but_preserves_truth(settings: Settings) -> None:
    publish(settings)
    details = load_transcription_details(JOB_ID, settings, job_record())

    summary = details.payload(include_events=False)

    assert summary["available"] is True
    assert summary["version"] == "baseline-pyin-onset-v1"
    assert summary["createdAt"] == CREATED_AT
    assert summary["counts"] == {
        "pitched": 2,
        "percussion": 2,
        "aligned": 1,
    }
    assert summary["sourceKinds"] == [
        "bass",
        "drums",
        "full_mix",
        "vocals",
    ]
    assert summary["algorithms"]
    assert "pitchedNoteEvents" not in summary
    assert "percussionEvents" not in summary
    assert "alignmentCandidates" not in summary


def test_failed_retry_preserves_previous_valid_artifact_and_current_failure(
    settings: Settings,
) -> None:
    expected = publish(settings)
    private = settings.data_dir / "private" / "checkpoint.bin"
    job = job_record(
        transcription_status="failed",
        transcription_stage="failed",
        transcription_progress=43.5,
        transcription_message=f"Retry stopped near {private}.",
        transcription_error=(
            f"Retry failed at {private}; token=super-secret-value"
        ),
    )

    details = load_transcription_details(JOB_ID, settings, job)
    payload = details.payload()

    assert details.available is True
    assert details.status == "failed"
    assert details.stage == "failed"
    assert details.progress == 43.5
    assert details.aligned_event_count == 1
    assert details.pitched_note_events == tuple(expected["pitchedNoteEvents"])
    assert details.percussion_events == tuple(expected["percussionEvents"])
    assert details.alignment_candidates == tuple(expected["alignmentCandidates"])
    assert details.error
    assert str(settings.data_dir) not in details.error
    assert "super-secret-value" not in details.error
    assert str(settings.data_dir) not in (details.message or "")
    assert payload["available"] is True
    assert payload["counts"]["aligned"] == 1
    assert payload["alignmentCandidates"] == expected["alignmentCandidates"]
    assert payload["error"] == details.error


def test_missing_pointer_does_not_scan_an_existing_artifact(
    settings: Settings,
) -> None:
    publish(settings)

    details = load_transcription_details(
        JOB_ID,
        settings,
        job_record(transcription_artifact_file_name=None),
    )

    assert details.available is False
    assert details.pitched_event_count == 0
    assert details.percussion_event_count == 0
    assert details.aligned_event_count == 0
    assert details.payload()["downloadFileName"] is None


def test_noncanonical_database_pointer_is_treated_as_unavailable(
    settings: Settings,
) -> None:
    publish(settings)

    details = load_transcription_details(
        JOB_ID,
        settings,
        job_record(
            transcription_artifact_file_name="../private/raw-events.json"
        ),
    )

    assert details.available is False
    with pytest.raises(TranscriptionArtifactUnavailableError):
        transcription_json_path(
            JOB_ID,
            settings,
            job_record(
                transcription_artifact_file_name="../private/raw-events.json"
            ),
        )


def test_canonical_pointer_with_missing_artifact_raises_bounded_error(
    settings: Settings,
) -> None:
    with pytest.raises(TranscriptionArtifactUnavailableError) as caught:
        load_transcription_details(JOB_ID, settings, job_record())
    assert str(settings.data_dir) not in str(caught.value)


def test_corrupt_canonical_artifact_raises_bounded_validation_error(
    settings: Settings,
) -> None:
    path = settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    path.parent.mkdir()
    path.write_text("{not-json", encoding="utf-8")

    with pytest.raises(TranscriptionArtifactError) as caught:
        load_transcription_details(JOB_ID, settings, job_record())

    assert str(settings.data_dir) not in str(caught.value)
    assert "not-json" not in str(caught.value)


def test_symlinked_canonical_artifact_is_rejected_without_outside_read(
    settings: Settings,
    tmp_path: Path,
) -> None:
    directory = settings.exports_dir / JOB_ID / "transcription"
    directory.mkdir()
    outside = tmp_path / "outside-private.json"
    outside.write_text(json.dumps(artifact_payload()), encoding="utf-8")
    target = directory / "raw-events.json"
    try:
        target.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(TranscriptionArtifactError) as caught:
        load_transcription_details(JOB_ID, settings, job_record())

    assert str(outside) not in str(caught.value)
    assert outside.read_text(encoding="utf-8") == json.dumps(
        artifact_payload()
    )


def test_invalid_job_id_and_nonmapping_job_fail_safely(
    settings: Settings,
) -> None:
    with pytest.raises(TranscriptionArtifactError) as caught:
        load_transcription_details("../private", settings, job_record())
    assert "private" not in str(caught.value)

    with pytest.raises(TranscriptionArtifactError):
        load_transcription_details(JOB_ID, settings, [])  # type: ignore[arg-type]


def test_hostile_database_counts_and_version_cannot_override_validated_truth(
    settings: Settings,
) -> None:
    expected = publish(settings)
    details = load_transcription_details(
        JOB_ID,
        settings,
        job_record(
            pitched_event_count=0,
            percussion_event_count=500_000,
            aligned_event_count=2_147_483_647,
            transcription_version="db-lie-v99",
            transcribed_at="2099-01-01T00:00:00+00:00",
        ),
    )

    assert len(expected["alignmentCandidates"]) == 2
    assert sum(
        "alignedTimeSeconds" in candidate
        for candidate in expected["alignmentCandidates"]
    ) == 1
    assert details.pitched_event_count == 2
    assert details.percussion_event_count == 2
    assert details.aligned_event_count == 1
    assert details.payload(include_events=False)["counts"]["aligned"] == 1
    assert details.transcription_version == "baseline-pyin-onset-v1"
    assert details.transcribed_at == CREATED_AT


def test_download_resolution_returns_only_the_validated_canonical_file(
    settings: Settings,
) -> None:
    publish(settings)
    job = job_record()

    path = transcription_json_path(JOB_ID, settings, job)
    details_payload = load_transcription_details(
        JOB_ID, settings, job
    ).payload()

    assert path == (
        settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    ).resolve(strict=True)
    assert path.is_file()
    assert str(path) not in json.dumps(details_payload, allow_nan=False)
    assert details_payload["downloadFileName"] == "raw-transcription.json"


def test_payload_returns_fresh_event_and_algorithm_copies(
    settings: Settings,
) -> None:
    publish(settings)
    details = load_transcription_details(JOB_ID, settings, job_record())

    first = details.payload()
    first["algorithms"]["pitchTracking"]["version"] = "mutated"
    first["pitchedNoteEvents"][0]["midiPitch"] = 0
    first["percussionEvents"][0]["hits"][0]["kind"] = "mutated"
    second = details.payload()

    assert second["algorithms"]["pitchTracking"]["version"] == "baseline-pyin-v1"
    assert second["pitchedNoteEvents"][0]["midiPitch"] != 0
    assert second["percussionEvents"][0]["hits"][0]["kind"] != "mutated"


def test_safe_future_state_is_preserved_and_invalid_state_is_normalized(
    settings: Settings,
) -> None:
    future = load_transcription_details(
        JOB_ID,
        settings,
        job_record(
            transcription_artifact_file_name=None,
            transcription_status="review_pending",
            transcription_stage="validating_candidates",
            transcription_progress=101.5,
        ),
    )
    assert future.status == "review_pending"
    assert future.stage == "validating_candidates"
    assert future.progress == 100.0

    invalid = load_transcription_details(
        JOB_ID,
        settings,
        job_record(
            transcription_artifact_file_name=None,
            transcription_status="../../private",
            transcription_stage="C:\\private\\stage",
            transcription_progress=math.nan,
        ),
    )
    assert invalid.status == "not_started"
    assert invalid.stage is None
    assert invalid.progress == 0.0


def test_traceback_and_machine_address_are_not_exposed(
    settings: Settings,
) -> None:
    details = load_transcription_details(
        JOB_ID,
        settings,
        job_record(
            transcription_artifact_file_name=None,
            transcription_message="worker object at 0x7ffee1234567",
            transcription_error=(
                "Traceback (most recent call last): File '/tmp/private.py'"
            ),
        ),
    )

    assert "0x7ffee1234567" not in (details.message or "")
    assert details.error == "Raw transcription failed."
    assert "/tmp/private.py" not in details.error


def test_payload_is_finite_json_and_contains_no_event_arrays_in_summary(
    settings: Settings,
) -> None:
    publish(settings)
    details = load_transcription_details(JOB_ID, settings, job_record())

    complete = details.payload()
    summary = details.payload(include_events=False)

    json.dumps(complete, allow_nan=False)
    json.dumps(summary, allow_nan=False)
    assert all(
        math.isfinite(float(value))
        for value in complete["counts"].values()
    )
    assert not {
        "pitchedNoteEvents",
        "percussionEvents",
        "alignmentCandidates",
    } & set(summary)
