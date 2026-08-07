from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    InterpretationPipelineError,
    interpret_transcription_job,
)
from app.percussion_interpretation import interpret_percussion
from app.pitched_part_inference import infer_pitched_parts
from app.rhythm_interpretation import interpret_rhythm
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    load_transcription_draft,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    load_raw_transcription,
    write_raw_transcription,
)


JOB_ID = "b" * 32
FIXED_CREATED_AT = "2026-08-07T09:00:00+00:00"


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


def _timing(*, weak_meter: bool = False) -> dict:
    return {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
        "beatConfidence": 0.9,
        "downbeatsSeconds": [0.0, 2.0, 4.0],
        "meter": 4,
        "meterConfidence": 0.2 if weak_meter else 0.85,
    }


def _analysis_payload(*, weak_meter: bool = False) -> dict:
    return {
        "schemaVersion": 1,
        "analysisVersion": "test-analysis-v1",
        "createdAt": "2026-08-07T08:00:00+00:00",
        "sourceAsset": "analysis.wav",
        "timing": _timing(weak_meter=weak_meter),
        "warnings": [],
    }


def _raw_payload() -> dict:
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "test-raw-v1",
        "createdAt": "2026-08-07T08:30:00+00:00",
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": "test-analysis-v1",
        },
        "algorithms": {
            "testRawPipeline": {
                "version": "test-raw-v1",
                "inputMode": "full_mix",
            }
        },
        "pitchedNoteEvents": [
            {
                "id": "p000001",
                "sourceKind": "full_mix",
                "startSeconds": 0.0,
                "endSeconds": 0.75,
                "midiNote": 69,
                "midiPitch": 69.2,
                "frequencyHz": 445.1,
                "noteName": "A4",
                "confidence": 0.91,
                "warnings": [],
            },
            {
                "id": "p000002",
                "sourceKind": "full_mix",
                "startSeconds": 0.25,
                "endSeconds": 1.0,
                "midiNote": 72,
                "midiPitch": 72.17,
                "frequencyHz": 527.4,
                "noteName": "C5",
                "confidence": 0.82,
                "warnings": ["Overlapping full-mix pitch remains editable."],
            },
            {
                "id": "p000003",
                "sourceKind": "full_mix",
                "startSeconds": 1.5,
                "endSeconds": 1.8,
                "midiNote": 64,
                "midiPitch": 64.12,
                "frequencyHz": 331.0,
                "noteName": "E4",
                "confidence": 0.1,
                "warnings": ["Low-confidence pitch remains raw evidence."],
            },
        ],
        "percussionEvents": [
            {
                "id": "r000001",
                "sourceKind": "full_mix",
                "timeSeconds": 0.5,
                "strength": 0.9,
                "hits": [
                    {"kind": "kick", "confidence": 0.92},
                    {"kind": "closed_hihat", "confidence": 0.8},
                ],
                "rawFeatureSummary": {
                    "lowBandRatio": 0.6,
                    "highBandRatio": 0.3,
                    "transientStrength": 0.9,
                },
                "warnings": ["Two simultaneous broad hit families are plausible."],
            },
            {
                "id": "r000002",
                "sourceKind": "full_mix",
                "timeSeconds": 1.2,
                "strength": 0.45,
                "hits": [{"kind": "future_click", "confidence": 0.3}],
                "rawFeatureSummary": {"transientStrength": 0.45},
                "warnings": ["Future percussion kind remains unresolved."],
            },
        ],
        "alignmentCandidates": [
            {
                "eventId": "p000001",
                "eventType": "pitched",
                "rawTimeSeconds": 0.0,
                "beatIndex": 0,
                "subdivision": 1,
                "subdivisionIndex": 0,
                "alignedTimeSeconds": 0.0,
                "offsetSeconds": 0.0,
                "confidence": 0.95,
                "measureIndex": 0,
                "beatInMeasure": 1,
            },
            {
                "eventId": "p000002",
                "eventType": "pitched",
                "rawTimeSeconds": 0.25,
                "beatIndex": 0,
                "subdivision": 2,
                "subdivisionIndex": 1,
                "alignedTimeSeconds": 0.25,
                "offsetSeconds": 0.0,
                "confidence": 0.8,
                "measureIndex": 0,
                "beatInMeasure": 1,
            },
            {
                "eventId": "p000003",
                "eventType": "pitched",
                "rawTimeSeconds": 1.5,
                "confidence": 0.2,
                "warnings": ["No accepted grid point; raw time is unchanged."],
            },
            {
                "eventId": "r000001",
                "eventType": "percussion",
                "rawTimeSeconds": 0.5,
                "beatIndex": 1,
                "subdivision": 1,
                "subdivisionIndex": 0,
                "alignedTimeSeconds": 0.5,
                "offsetSeconds": 0.0,
                "confidence": 0.94,
                "measureIndex": 0,
                "beatInMeasure": 2,
            },
            {
                "eventId": "r000002",
                "eventType": "percussion",
                "rawTimeSeconds": 1.2,
                "confidence": 0.25,
                "warnings": ["No accepted percussion grid point."],
            },
        ],
        "warnings": ["Full-mix transcription remains intentionally broad."],
    }


def _write_job(settings: Settings, *, weak_meter: bool = False) -> None:
    job_dir = settings.exports_dir / JOB_ID
    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(_analysis_payload(weak_meter=weak_meter), allow_nan=False),
        encoding="utf-8",
    )
    write_raw_transcription(JOB_ID, settings, _raw_payload())


def _json_mapping(value: object) -> dict:
    return json.loads(json.dumps(value, allow_nan=False))


def test_full_mix_pipeline_publishes_exact_interpreter_evidence(
    settings: Settings,
) -> None:
    _write_job(settings)
    stages: list[tuple[str, str, float]] = []

    result = interpret_transcription_job(
        JOB_ID,
        settings,
        lambda stage, message, progress: stages.append((stage, message, progress)),
        created_at=FIXED_CREATED_AT,
    )

    assert result.version == INTERPRETATION_PIPELINE_VERSION
    assert result.draft_file_name == INTERPRETATION_DRAFT_RELATIVE_PATH
    assert result.created_at == FIXED_CREATED_AT
    assert result.pitched_item_count == 3
    assert result.percussion_item_count == 2
    assert result.part_count >= 3
    assert result.warning_count == len(result.warnings)

    raw = load_raw_transcription(JOB_ID, settings)
    assert raw is not None
    pitched = infer_pitched_parts(raw["pitchedNoteEvents"], raw["alignmentCandidates"])
    percussion = interpret_percussion(
        raw["percussionEvents"],
        [item for item in raw["alignmentCandidates"] if item["eventType"] == "percussion"],
    )
    rhythm = interpret_rhythm(
        raw["pitchedNoteEvents"],
        raw["percussionEvents"],
        raw["alignmentCandidates"],
        _timing(),
    )
    evidence = result.payload["interpretationEvidence"]
    assert evidence["pitchedPartInference"] == pitched.payload()
    assert evidence["percussionInterpretation"] == _json_mapping(asdict(percussion))
    assert evidence["rhythmInterpretation"] == _json_mapping(asdict(rhythm))

    assert "p000003" in evidence["pitchedPartInference"]["unassignedEventIds"]
    assert len(evidence["pitchedPartInference"]["voices"]) >= 2
    drum_hits = [
        item
        for item in evidence["percussionInterpretation"]["assignments"]
        if item["eventId"] == "r000001"
    ]
    assert len(drum_hits) == 2
    unresolved_rhythm = next(
        item
        for item in evidence["rhythmInterpretation"]["event_interpretations"]
        if item["eventId"] == "r000002"
    )
    assert unresolved_rhythm["placementHypotheses"][0]["kind"] == "unresolved"

    source_ids = {
        item["id"] for item in result.payload["sourceTranscription"]["sourceEventIndex"]
    }
    assert source_ids == {"p000001", "p000002", "p000003", "r000001", "r000002"}
    assert result.payload["sourceTranscription"]["provenance"][
        "sourceSeparationPresent"
    ] is False

    assert [stage for stage, _, _ in stages] == [
        "loading_raw_transcription",
        "loading_analysis_timing",
        "interpreting_pitched_parts",
        "interpreting_percussion",
        "interpreting_rhythm",
        "assembling_interpretation_draft",
        "validating_interpretation_draft",
        "saving_interpretation_draft",
        "completed",
    ]
    assert [progress for _, _, progress in stages] == sorted(
        progress for _, _, progress in stages
    )
    assert stages[-1][2] == 100.0

    stored = load_transcription_draft(JOB_ID, settings)
    assert stored == result.payload


def test_weak_meter_stays_time_relative_without_invented_measures(
    settings: Settings,
) -> None:
    _write_job(settings, weak_meter=True)
    result = interpret_transcription_job(
        JOB_ID,
        settings,
        created_at=FIXED_CREATED_AT,
    )

    evidence = result.payload["interpretationEvidence"]["rhythmInterpretation"]
    assert evidence["measures"] == []
    assert evidence["diagnostics"]["timingMode"] == "beat_relative"
    assert result.payload["measures"] == []
    assert result.payload["sourceTranscription"]["provenance"][
        "sourceSeparationPresent"
    ] is False


def test_same_inputs_and_timestamp_are_byte_stable(settings: Settings) -> None:
    _write_job(settings)
    first = interpret_transcription_job(
        JOB_ID,
        settings,
        created_at=FIXED_CREATED_AT,
    )
    path = settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    first_bytes = path.read_bytes()

    second = interpret_transcription_job(
        JOB_ID,
        settings,
        created_at=FIXED_CREATED_AT,
    )
    assert second.payload == first.payload
    assert path.read_bytes() == first_bytes


def test_missing_raw_artifact_fails_without_creating_draft(settings: Settings) -> None:
    analysis_dir = settings.exports_dir / JOB_ID / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(_analysis_payload()),
        encoding="utf-8",
    )

    with pytest.raises(InterpretationPipelineError, match="unavailable"):
        interpret_transcription_job(JOB_ID, settings, created_at=FIXED_CREATED_AT)
    assert not (
        settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    ).exists()


def test_corrupt_raw_artifact_fails_with_bounded_path_free_error(
    settings: Settings,
) -> None:
    _write_job(settings)
    raw_path = settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    raw_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(InterpretationPipelineError) as caught:
        interpret_transcription_job(JOB_ID, settings, created_at=FIXED_CREATED_AT)
    message = str(caught.value)
    assert "unreadable or unsafe" in message
    assert "/" not in message
    assert "\\" not in message


def test_symlinked_raw_artifact_is_rejected(settings: Settings, tmp_path: Path) -> None:
    _write_job(settings)
    raw_path = settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    outside = tmp_path / "outside-raw.json"
    raw_path.replace(outside)
    try:
        raw_path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(InterpretationPipelineError, match="unreadable or unsafe"):
        interpret_transcription_job(JOB_ID, settings, created_at=FIXED_CREATED_AT)


def test_stale_analysis_provenance_is_rejected(settings: Settings) -> None:
    _write_job(settings)
    analysis_path = settings.exports_dir / JOB_ID / ANALYSIS_JSON_RELATIVE_PATH
    payload = json.loads(analysis_path.read_text(encoding="utf-8"))
    payload["analysisVersion"] = "different-analysis-version"
    analysis_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(InterpretationPipelineError, match="provenance is stale"):
        interpret_transcription_job(JOB_ID, settings, created_at=FIXED_CREATED_AT)


def test_failed_publication_preserves_previous_draft(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_job(settings)
    interpret_transcription_job(
        JOB_ID,
        settings,
        created_at=FIXED_CREATED_AT,
    )
    path = settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    before = path.read_bytes()

    import app.transcription_draft as draft_module

    def fail_replace(source: Path, destination: Path) -> None:
        raise OSError("injected /private/secret publication failure")

    monkeypatch.setattr(draft_module, "_replace_atomic", fail_replace)
    with pytest.raises(InterpretationPipelineError) as caught:
        interpret_transcription_job(
            JOB_ID,
            settings,
            created_at="2026-08-07T09:01:00+00:00",
        )

    assert str(caught.value) == "Editable interpretation could not be published safely."
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".draft.json.*.tmp"))


def test_callback_failure_is_bounded_and_does_not_publish(settings: Settings) -> None:
    _write_job(settings)

    def fail_callback(stage: str, message: str, progress: float) -> None:
        raise RuntimeError("/private/secret callback failure")

    with pytest.raises(InterpretationPipelineError) as caught:
        interpret_transcription_job(
            JOB_ID,
            settings,
            fail_callback,
            created_at=FIXED_CREATED_AT,
        )
    assert str(caught.value) == "Interpretation progress reporting failed."
    assert not (
        settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    ).exists()


def test_created_at_must_be_utc(settings: Settings) -> None:
    _write_job(settings)
    with pytest.raises(InterpretationPipelineError, match="must be UTC"):
        interpret_transcription_job(
            JOB_ID,
            settings,
            created_at="2026-08-07T17:00:00+08:00",
        )
