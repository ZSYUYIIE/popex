from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app.config import Settings
from app.interpretation_artifacts import (
    InterpretationArtifactError,
    InterpretationArtifactUnavailableError,
    interpretation_json_path,
    load_interpretation_details,
)
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    write_transcription_draft,
)
from app.transcription_events import RAW_TRANSCRIPTION_RELATIVE_PATH


JOB_ID = "c" * 32


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


def draft_payload() -> dict:
    return {
        "schemaVersion": 1,
        "draftVersion": "editable-interpretation-v1",
        "createdAt": "2026-08-07T10:00:00+00:00",
        "sourceTranscription": {
            "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
            "schemaVersion": 1,
            "transcriptionVersion": "raw-transcription-v1",
            "provenance": {"pipelineVersion": "raw-transcription-v1"},
            "sourceEventIndex": [
                {
                    "id": "p1",
                    "eventType": "pitched",
                    "sourceKind": "vocals",
                    "rawStartSeconds": 0.0,
                    "rawEndSeconds": 0.5,
                    "confidence": 0.9,
                    "midiPitch": 69.2,
                }
            ],
        },
        "algorithms": {
            "pitchedPartInference": {"version": "source-phrase-v1"},
            "percussionInterpretation": {"version": "broad-drum-structure-v1"},
            "rhythmInterpretation": {"version": "conservative-grid-v1"},
        },
        "interpretationEvidence": {
            "pitchedPartInference": {
                "version": "source-phrase-v1",
                "parts": [],
                "voices": [],
                "phrases": [],
                "assignments": [{"eventId": "p1", "sourceEventIds": ["p1"]}],
                "unassignedEventIds": [],
                "warnings": [],
                "diagnostics": {"rawTimesPreserved": True},
            },
            "percussionInterpretation": {
                "version": "broad-drum-structure-v1",
                "parts": [],
                "voices": [],
                "groups": [],
                "assignments": [],
                "unresolved_event_ids": [],
                "warnings": [],
                "diagnostics": {"rawTimesPreserved": True},
            },
            "rhythmInterpretation": {
                "version": "conservative-grid-v1",
                "meter_candidates": [],
                "measures": [],
                "event_interpretations": [
                    {"eventId": "p1", "sourceEventIds": ["p1"]}
                ],
                "rest_candidates": [],
                "warnings": [],
                "diagnostics": {"rawTimingAuthoritative": True},
            },
        },
        "parts": [
            {
                "id": "part_vocals",
                "sourceKind": "vocals",
                "role": "pitched",
                "instrumentKind": "source_pitched_line",
                "voiceIds": ["voice_vocals"],
                "sourceEventIds": ["p1"],
                "confidence": 0.9,
            }
        ],
        "voices": [
            {
                "id": "voice_vocals",
                "partId": "part_vocals",
                "voiceKind": "monophonic",
                "sourceEventIds": ["p1"],
                "confidence": 0.9,
            }
        ],
        "measures": [],
        "phrases": [
            {
                "id": "phrase_vocals",
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
                "sourceEventIds": ["p1"],
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 0.5,
                "confidence": 0.8,
            }
        ],
        "pitchedItems": [
            {
                "id": "pitched_p1",
                "interpretationType": "note",
                "placementStatus": "unassigned",
                "partId": "part_vocals",
                "voiceId": "voice_vocals",
                "sourceEventIds": ["p1"],
                "rawStartSeconds": 0.0,
                "rawEndSeconds": 0.5,
                "sourceKind": "vocals",
                "pitch": {
                    "midiNote": 69,
                    "midiPitch": 69.2,
                    "frequencyHz": 445.1,
                    "noteName": "A4",
                },
                "confidence": 0.8,
                "alternatives": [
                    {
                        "id": "alt_p1_duration",
                        "kind": "duration_candidate",
                        "confidence": 0.4,
                    }
                ],
            }
        ],
        "percussionItems": [],
        "alternatives": [],
        "warnings": ["Editable warning."],
    }


def job_mapping(**overrides: object) -> dict:
    value = {
        "interpretation_status": "completed",
        "interpretation_stage": "completed",
        "interpretation_progress": 100,
        "interpretation_message": "Editable interpretation complete.",
        "interpretation_version": "hostile-db-version",
        "interpretation_artifact_file_name": INTERPRETATION_DRAFT_RELATIVE_PATH,
        "interpreted_at": "1900-01-01T00:00:00+00:00",
        "interpretation_part_count": 999,
        "interpretation_phrase_count": 999,
        "interpretation_pitched_item_count": 999,
        "interpretation_percussion_item_count": 999,
        "interpretation_warning_count": 999,
        "interpretation_error": None,
    }
    value.update(overrides)
    return value


def publish(settings: Settings) -> Path:
    return write_transcription_draft(JOB_ID, settings, draft_payload())


def test_no_pointer_returns_safe_unavailable_details(settings: Settings) -> None:
    details = load_interpretation_details(
        JOB_ID,
        settings,
        job_mapping(
            interpretation_status="failed",
            interpretation_artifact_file_name=None,
            interpretation_error="retry later",
        ),
    )
    payload = details.payload(include_items=False)

    assert payload["available"] is False
    assert payload["status"] == "failed"
    assert payload["counts"]["parts"] == 0
    assert payload["downloadFileName"] is None
    assert "parts" not in payload


def test_valid_artifact_drives_summary_instead_of_hostile_db_counts(
    settings: Settings,
) -> None:
    publish(settings)
    details = load_interpretation_details(JOB_ID, settings, job_mapping())
    payload = details.payload(include_items=False)

    assert payload["available"] is True
    assert payload["version"] == "editable-interpretation-v1"
    assert payload["createdAt"] == "2026-08-07T10:00:00+00:00"
    assert payload["counts"] == {
        "parts": 1,
        "voices": 1,
        "measures": 0,
        "phrases": 1,
        "pitched": 1,
        "percussion": 0,
        "warnings": 1,
        "unassignedPitched": 1,
        "unplacedPercussion": 0,
    }
    assert payload["sourceKinds"] == ["vocals"]
    assert payload["warnings"] == ["Editable warning."]
    assert payload["downloadFileName"] == "editable-interpretation.json"
    assert "parts" not in payload
    assert "interpretationEvidence" not in payload


def test_full_payload_preserves_normalized_and_exact_evidence(settings: Settings) -> None:
    publish(settings)
    expected = draft_payload()
    details = load_interpretation_details(JOB_ID, settings, job_mapping())
    payload = details.payload(include_items=True)

    assert payload["parts"] == expected["parts"]
    assert payload["voices"] == expected["voices"]
    assert payload["phrases"] == expected["phrases"]
    assert payload["pitchedItems"] == expected["pitchedItems"]
    assert payload["percussionItems"] == []
    assert payload["interpretationEvidence"] == expected["interpretationEvidence"]


def test_returned_payloads_are_independent_copies(settings: Settings) -> None:
    publish(settings)
    details = load_interpretation_details(JOB_ID, settings, job_mapping())
    first = details.payload(include_items=True)
    second = details.payload(include_items=True)

    first["parts"][0]["sourceKind"] = "mutated"
    first["interpretationEvidence"]["pitchedPartInference"]["assignments"].clear()
    assert second["parts"][0]["sourceKind"] == "vocals"
    assert second["interpretationEvidence"]["pitchedPartInference"]["assignments"]


def test_failed_reinterpretation_keeps_previous_draft_available(
    settings: Settings,
) -> None:
    publish(settings)
    details = load_interpretation_details(
        JOB_ID,
        settings,
        job_mapping(
            interpretation_status="failed",
            interpretation_stage="failed",
            interpretation_progress=71,
            interpretation_message="Retry stopped.",
            interpretation_error="bounded retry failure",
        ),
    )
    payload = details.payload(include_items=False)

    assert payload["available"] is True
    assert payload["status"] == "failed"
    assert payload["progress"] == 71
    assert payload["counts"]["parts"] == 1
    assert payload["error"] == "bounded retry failure"


def test_job_text_is_sanitized(settings: Settings) -> None:
    publish(settings)
    details = load_interpretation_details(
        JOB_ID,
        settings,
        job_mapping(
            interpretation_message="failed at /home/user/private/draft.json token=abc123",
            interpretation_error="see C:\\Users\\me\\secret.json https://bad.invalid/x 0xdeadbeef",
        ),
    )
    encoded = json.dumps(details.payload(include_items=False)).lower()

    assert "/home/user/private" not in encoded
    assert "c:\\users\\me" not in encoded
    assert "abc123" not in encoded
    assert "bad.invalid" not in encoded
    assert "deadbeef" not in encoded


def test_corrupt_saved_draft_is_not_exposed(settings: Settings) -> None:
    path = publish(settings)
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["voices"][0]["partId"] = "missing_part"
    path.write_text(json.dumps(stored), encoding="utf-8")

    with pytest.raises(InterpretationArtifactError, match="failed validation"):
        load_interpretation_details(JOB_ID, settings, job_mapping())


def test_symlinked_draft_is_not_exposed(settings: Settings, tmp_path: Path) -> None:
    path = publish(settings)
    outside = tmp_path / "outside.json"
    path.replace(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(InterpretationArtifactError, match="failed validation"):
        load_interpretation_details(JOB_ID, settings, job_mapping())
    with pytest.raises(InterpretationArtifactError):
        interpretation_json_path(JOB_ID, settings, job_mapping())


def test_download_path_is_canonical_and_validated(settings: Settings) -> None:
    path = publish(settings)
    resolved = interpretation_json_path(JOB_ID, settings, job_mapping())
    assert resolved == path.resolve(strict=True)


def test_download_detects_file_change_during_validation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = publish(settings)
    import app.interpretation_artifacts as module

    original = module._load_published_artifact

    def mutate_after_validation(job_id: str, app_settings: Settings) -> dict:
        result = original(job_id, app_settings)
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return result

    monkeypatch.setattr(module, "_load_published_artifact", mutate_after_validation)
    with pytest.raises(InterpretationArtifactError, match="changed during validation"):
        interpretation_json_path(JOB_ID, settings, job_mapping())


def test_download_rejects_noncanonical_pointer(settings: Settings) -> None:
    publish(settings)
    with pytest.raises(InterpretationArtifactUnavailableError):
        interpretation_json_path(
            JOB_ID,
            settings,
            job_mapping(interpretation_artifact_file_name="interpretation/other.json"),
        )


def test_invalid_job_id_fails_safely(settings: Settings) -> None:
    with pytest.raises(InterpretationArtifactError):
        load_interpretation_details("../bad", settings, job_mapping())
