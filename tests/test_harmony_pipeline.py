from __future__ import annotations

import copy
import json
from dataclasses import replace
from pathlib import Path

import pytest

from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HarmonyArtifactError,
    load_harmony_artifact,
)
from app.harmony_inference import (
    HARMONY_INFERENCE_VERSION,
    HarmonyInferenceError,
    infer_harmony,
)
from app.harmony_pipeline import (
    HARMONY_PIPELINE_VERSION,
    HarmonyPipelineError,
    infer_harmony_job,
)
from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    interpret_transcription_job,
)
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    load_transcription_draft,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    load_raw_transcription,
    write_raw_transcription,
)


JOB_ID = "e" * 32
FIXED_HARMONY_AT = "2026-08-14T04:00:00+00:00"
FIXED_INTERPRETATION_AT = "2026-08-14T03:30:00+00:00"


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


def _timing(*, weak: bool = False) -> dict:
    return {
        "tempoBpm": 120.0,
        "tempoConfidence": 0.9,
        "tempoStable": True,
        "beatsSeconds": [0.0, 1.0, 2.0],
        "beatConfidence": 0.1 if weak else 0.9,
        "downbeatsSeconds": [0.0, 2.0],
        "meter": 4,
        "meterConfidence": 0.8,
    }


def _tonality() -> dict:
    return {
        "tonalCenter": "C",
        "primaryCandidate": {
            "tonalCenter": "C",
            "collection": "ionian",
            "displayName": "C major",
            "confidence": 0.82,
            "supportedByBaseline": True,
        },
        "candidates": [],
        "localRegions": [],
        "chromaticismScore": None,
        "baselineCollections": ["ionian", "aeolian"],
        "key": "C",
        "mode": "major",
        "symbol": "C major",
        "confidence": 0.82,
        "tuningOffsetCents": 3.0,
        "chromaMean": [0.0] * 12,
        "alternatives": [],
    }


def _analysis_payload(
    *,
    version: str = "test-analysis-v1",
    timing_mode: str = "strong",
    malformed_timing: bool = False,
    malformed_tonality: bool = False,
) -> dict:
    payload = {
        "schemaVersion": 1,
        "analysisVersion": version,
        "createdAt": "2026-08-14T03:00:00+00:00",
        "sourceAsset": "analysis.wav",
        "warnings": [],
    }
    if timing_mode != "missing":
        payload["timing"] = "bad" if malformed_timing else _timing(
            weak=timing_mode == "weak"
        )
    payload["tonality"] = "bad" if malformed_tonality else _tonality()
    return payload


def _pitched_events() -> list[dict]:
    return [
        {
            "id": "p_c",
            "sourceKind": "vocals",
            "startSeconds": 0.0,
            "endSeconds": 0.9,
            "midiNote": 60,
            "midiPitch": 60.12,
            "frequencyHz": 264.0,
            "noteName": "C4",
            "confidence": 0.92,
            "warnings": [],
        },
        {
            "id": "p_bass_e",
            "sourceKind": "bass",
            "startSeconds": 0.0,
            "endSeconds": 0.9,
            "midiNote": 52,
            "midiPitch": 52.08,
            "frequencyHz": 165.6,
            "noteName": "E3",
            "confidence": 0.95,
            "warnings": [],
        },
        {
            "id": "p_e",
            "sourceKind": "other",
            "startSeconds": 0.05,
            "endSeconds": 0.9,
            "midiNote": 64,
            "midiPitch": 63.94,
            "frequencyHz": 328.5,
            "noteName": "E4",
            "confidence": 0.9,
            "warnings": [],
        },
        {
            "id": "p_g",
            "sourceKind": "other",
            "startSeconds": 0.1,
            "endSeconds": 0.9,
            "midiNote": 67,
            "midiPitch": 67.04,
            "frequencyHz": 393.0,
            "noteName": "G4",
            "confidence": 0.87,
            "warnings": [],
        },
    ]


def _alignment(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        raw_time = event["startSeconds"]
        result.append(
            {
                "eventId": event["id"],
                "eventType": "pitched",
                "rawTimeSeconds": raw_time,
                "beatIndex": 0,
                "subdivision": 4,
                "subdivisionIndex": min(3, int(round(raw_time * 4))),
                "alignedTimeSeconds": raw_time,
                "offsetSeconds": 0.0,
                "confidence": 0.9,
                "measureIndex": 0,
                "beatInMeasure": 1,
            }
        )
    return result


def _raw_payload(
    *,
    created_at: str = "2026-08-14T03:15:00+00:00",
    analysis_version: str = "test-analysis-v1",
    pitched: bool = True,
) -> dict:
    events = _pitched_events() if pitched else []
    return {
        "schemaVersion": 1,
        "transcriptionVersion": "test-raw-v1",
        "createdAt": created_at,
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": analysis_version,
        },
        "algorithms": {
            "testRawPipeline": {
                "version": "test-raw-v1",
                "inputMode": "full_mix",
            }
        },
        "pitchedNoteEvents": events,
        "percussionEvents": [],
        "alignmentCandidates": _alignment(events),
        "warnings": ["Raw pitch evidence remains editable."],
    }


def _write_inputs(
    settings: Settings,
    *,
    analysis: dict | None = None,
    raw: dict | None = None,
) -> None:
    job_dir = settings.exports_dir / JOB_ID
    analysis_dir = job_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(analysis or _analysis_payload(), allow_nan=False),
        encoding="utf-8",
    )
    write_raw_transcription(JOB_ID, settings, raw or _raw_payload())


def _write_interpretation(settings: Settings) -> dict:
    result = interpret_transcription_job(
        JOB_ID,
        settings,
        created_at=FIXED_INTERPRETATION_AT,
    )
    return result.payload


def _artifact_path(settings: Settings) -> Path:
    return settings.exports_dir / JOB_ID / HARMONY_ARTIFACT_RELATIVE_PATH


def test_raw_only_pipeline_publishes_canonical_artifact_and_progress(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    stages: list[tuple[str, str, float]] = []

    result = infer_harmony_job(
        JOB_ID,
        settings,
        lambda stage, message, progress: stages.append((stage, message, progress)),
        created_at=FIXED_HARMONY_AT,
    )

    assert result.version == HARMONY_PIPELINE_VERSION
    assert result.artifact_file_name == HARMONY_ARTIFACT_RELATIVE_PATH
    assert result.created_at == FIXED_HARMONY_AT
    assert result.event_count == 4
    assert result.segment_count >= 1
    assert result.resolved_segment_count >= 1
    assert result.warning_count == len(result.warnings)
    assert result.used_interpretation_context is False
    assert result.interpretation_version is None
    assert "sourceInterpretation" not in result.payload
    assert result.payload["sourceTranscription"]["fileName"] == (
        RAW_TRANSCRIPTION_RELATIVE_PATH
    )
    assert result.payload["sourceAnalysis"]["fileName"] == (
        ANALYSIS_JSON_RELATIVE_PATH
    )
    raw = {item["id"]: item for item in result.payload["rawEvidence"]}
    assert raw["p_c"]["midiPitch"] == 60.12
    assert raw["p_c"]["rawStartSeconds"] == 0.0
    assert raw["p_c"]["rawEndSeconds"] == 0.9
    assert any(
        segment["primaryCandidate"]
        and segment["primaryCandidate"]["root"] == "C"
        for segment in result.payload["segments"]
    )
    assert load_harmony_artifact(JOB_ID, settings) == result.payload
    assert not (settings.exports_dir / JOB_ID / "stems").exists()

    assert [stage for stage, _, _ in stages] == [
        "loading_raw_transcription",
        "loading_analysis_context",
        "loading_optional_interpretation",
        "inferring_harmonic_context",
        "validating_harmonic_context",
        "saving_harmonic_context",
        "completed",
    ]
    assert [progress for _, _, progress in stages] == sorted(
        progress for _, _, progress in stages
    )
    assert stages[-1][2] == 100.0
    assert all(message and "/" not in message and "\\" not in message for _, message, _ in stages)


def test_current_optional_interpretation_context_is_reused(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    draft = _write_interpretation(settings)

    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.used_interpretation_context is True
    assert result.interpretation_version == INTERPRETATION_PIPELINE_VERSION
    assert result.payload["sourceInterpretation"] == {
        "fileName": INTERPRETATION_DRAFT_RELATIVE_PATH,
        "schemaVersion": 1,
        "draftVersion": INTERPRETATION_PIPELINE_VERSION,
    }
    pitched_context = draft["interpretationEvidence"]["pitchedPartInference"]
    expected_parts = {
        assignment["partId"]
        for assignment in pitched_context["assignments"]
        if assignment["status"] == "assigned"
    }
    observed_parts = {
        part_id
        for segment in result.payload["segments"]
        for part_id in segment["partIds"]
    }
    assert observed_parts == expected_parts


def test_processor_receives_detached_exact_current_inputs(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    draft = _write_interpretation(settings)
    raw = load_raw_transcription(JOB_ID, settings)
    assert raw is not None
    captured: dict[str, object] = {}

    def processor(events, timing, tonality, parts, *, version):
        captured.update(
            events=copy.deepcopy(events),
            timing=copy.deepcopy(timing),
            tonality=copy.deepcopy(tonality),
            parts=copy.deepcopy(parts),
            version=version,
        )
        original_events = copy.deepcopy(events)
        events[0]["midiPitch"] = 0
        if timing is not None:
            timing["beatConfidence"] = 0
        if parts is not None:
            parts["assignments"].clear()
        return infer_harmony(
            original_events,
            captured["timing"],
            captured["tonality"],
            captured["parts"],
            version=version,
        )

    infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
        inference_processor=processor,
    )

    assert captured["events"] == raw["pitchedNoteEvents"]
    assert captured["timing"] == _timing()
    assert captured["tonality"] == _tonality()
    assert captured["parts"] == draft["interpretationEvidence"][
        "pitchedPartInference"
    ]
    assert captured["version"] == HARMONY_INFERENCE_VERSION
    reloaded_raw = load_raw_transcription(JOB_ID, settings)
    assert reloaded_raw is not None
    assert reloaded_raw["pitchedNoteEvents"][0]["midiPitch"] == 60.12
    reloaded_draft = load_transcription_draft(JOB_ID, settings)
    assert reloaded_draft == draft


def test_missing_interpretation_is_normal_and_does_not_warn(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.used_interpretation_context is False
    assert not any("editable interpretation" in warning.lower() for warning in result.warnings)


def test_corrupt_interpretation_falls_back_without_blocking_harmony(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    _write_interpretation(settings)
    draft_path = settings.exports_dir / JOB_ID / INTERPRETATION_DRAFT_RELATIVE_PATH
    draft_path.write_text("{broken", encoding="utf-8")

    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.used_interpretation_context is False
    assert "sourceInterpretation" not in result.payload
    assert any("could not be validated" in warning for warning in result.warnings)
    assert draft_path.read_text(encoding="utf-8") == "{broken"


def test_stale_interpretation_falls_back_to_current_raw_evidence(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    _write_interpretation(settings)
    replacement = _raw_payload(created_at="2026-08-14T03:20:00+00:00")
    write_raw_transcription(JOB_ID, settings, replacement)

    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.used_interpretation_context is False
    assert any("does not match the current raw" in warning for warning in result.warnings)
    assert result.payload["sourceTranscription"]["transcriptionVersion"] == (
        replacement["transcriptionVersion"]
    )


def test_malformed_optional_context_falls_back_safely(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(settings)
    draft = _write_interpretation(settings)
    broken = copy.deepcopy(draft)
    broken["interpretationEvidence"].pop("pitchedPartInference")

    import app.harmony_pipeline as module

    monkeypatch.setattr(module, "load_transcription_draft", lambda *_args: broken)
    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.used_interpretation_context is False
    assert any("lacks usable pitched-part context" in warning for warning in result.warnings)


@pytest.mark.parametrize("timing_mode", ["weak", "missing"])
def test_weak_or_missing_timing_uses_explicit_absolute_fallback(
    settings: Settings,
    timing_mode: str,
) -> None:
    _write_inputs(settings, analysis=_analysis_payload(timing_mode=timing_mode))

    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.payload["diagnostics"]["windowingMode"] == "absolute_time"
    assert result.payload["diagnostics"]["fallbackWindowSeconds"] == 1.0
    assert any("absolute-time fallback" in warning for warning in result.warnings)


def test_stale_analysis_provenance_fails_before_publication(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    analysis_path = settings.exports_dir / JOB_ID / ANALYSIS_JSON_RELATIVE_PATH
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    analysis["analysisVersion"] = "different-analysis-v2"
    analysis_path.write_text(json.dumps(analysis), encoding="utf-8")

    with pytest.raises(HarmonyPipelineError, match="provenance is stale"):
        infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    assert not _artifact_path(settings).exists()


@pytest.mark.parametrize(
    ("analysis", "message"),
    [
        (_analysis_payload(malformed_timing=True), "timing evidence is malformed"),
        (_analysis_payload(malformed_tonality=True), "tonal evidence is malformed"),
    ],
)
def test_malformed_analysis_context_is_rejected(
    settings: Settings,
    analysis: dict,
    message: str,
) -> None:
    _write_inputs(settings, analysis=analysis)
    with pytest.raises(HarmonyPipelineError, match=message):
        infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)


def test_missing_raw_artifact_fails_without_creating_harmony(
    settings: Settings,
) -> None:
    analysis_dir = settings.exports_dir / JOB_ID / "analysis"
    analysis_dir.mkdir(parents=True)
    (analysis_dir / "audio-analysis.json").write_text(
        json.dumps(_analysis_payload()),
        encoding="utf-8",
    )

    with pytest.raises(HarmonyPipelineError, match="unavailable"):
        infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    assert not _artifact_path(settings).exists()


def test_corrupt_raw_artifact_has_bounded_path_free_error(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    raw_path = settings.exports_dir / JOB_ID / RAW_TRANSCRIPTION_RELATIVE_PATH
    raw_path.write_text("{broken", encoding="utf-8")

    with pytest.raises(HarmonyPipelineError) as caught:
        infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    assert str(caught.value) == "Published raw transcription is unreadable or unsafe."
    assert "/" not in str(caught.value)
    assert "\\" not in str(caught.value)


def test_no_pitched_events_publishes_an_explicit_empty_context(
    settings: Settings,
) -> None:
    _write_inputs(settings, raw=_raw_payload(pitched=False))

    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )

    assert result.event_count == 0
    assert result.segment_count == 0
    assert result.unresolved_event_count == 0
    assert result.payload["rawEvidence"] == []
    assert result.payload["segments"] == []
    assert any("no pitched-note evidence" in warning for warning in result.warnings)


def test_inference_error_preserves_previous_harmony(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    def fail(*_args, **_kwargs):
        raise HarmonyInferenceError("failure at /private/audio.wav")

    with pytest.raises(HarmonyPipelineError, match="could not interpret") as caught:
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
            inference_processor=fail,
        )
    assert "/" not in str(caught.value)
    assert "\\" not in str(caught.value)
    assert path.read_bytes() == before


def test_generic_processor_failure_is_bounded_and_preserves_previous(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    def fail(*_args, **_kwargs):
        raise RuntimeError("C:\\Users\\person\\token=private")

    with pytest.raises(HarmonyPipelineError) as caught:
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
            inference_processor=fail,
        )
    assert str(caught.value) == "Harmonic inference failed at a protected boundary."
    assert "/" not in str(caught.value)
    assert "\\" not in str(caught.value)
    assert "token" not in str(caught.value).lower()
    assert path.read_bytes() == before


def test_wrong_processor_result_is_rejected_before_publication(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    with pytest.raises(HarmonyPipelineError, match="invalid result"):
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
            inference_processor=lambda *_args, **_kwargs: {},
        )
    assert path.read_bytes() == before


def test_artifact_validation_failure_preserves_previous(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    def invalid(events, timing, tonality, parts, *, version):
        result = infer_harmony(events, timing, tonality, parts, version=version)
        diagnostics = dict(result.diagnostics)
        diagnostics["eventCount"] = 999
        return replace(result, diagnostics=diagnostics)

    with pytest.raises(HarmonyPipelineError, match="artifact validation"):
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
            inference_processor=invalid,
        )
    assert path.read_bytes() == before


def test_publication_failure_preserves_previous_artifact(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    import app.harmony_artifacts as artifact_module

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise OSError("injected /private/secret publication failure")

    monkeypatch.setattr(artifact_module, "_replace_atomic", fail_replace)
    with pytest.raises(HarmonyPipelineError) as caught:
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
        )
    assert str(caught.value) == "Harmonic context could not be published safely."
    assert path.read_bytes() == before


def test_reload_failure_restores_previous_valid_artifact(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    import app.harmony_artifacts as artifact_module
    import app.harmony_pipeline as pipeline_module

    actual_load = artifact_module.load_harmony_artifact
    calls = 0

    def fail_second_load(job_id: str, app_settings: Settings):
        nonlocal calls
        calls += 1
        if calls == 1:
            return actual_load(job_id, app_settings)
        raise HarmonyArtifactError("verification at /private/secret failed")

    monkeypatch.setattr(pipeline_module, "load_harmony_artifact", fail_second_load)
    with pytest.raises(HarmonyPipelineError, match="could not be verified"):
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
        )
    assert path.read_bytes() == before


def test_reload_mismatch_restores_previous_valid_artifact(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    import app.harmony_artifacts as artifact_module
    import app.harmony_pipeline as pipeline_module

    actual_load = artifact_module.load_harmony_artifact
    calls = 0

    def mismatch_second_load(job_id: str, app_settings: Settings):
        nonlocal calls
        calls += 1
        loaded = actual_load(job_id, app_settings)
        if calls == 2 and loaded is not None:
            loaded["harmonyVersion"] = "unexpected-version"
        return loaded

    monkeypatch.setattr(pipeline_module, "load_harmony_artifact", mismatch_second_load)
    with pytest.raises(HarmonyPipelineError, match="could not be verified"):
        infer_harmony_job(
            JOB_ID,
            settings,
            created_at="2026-08-14T04:01:00+00:00",
        )
    assert path.read_bytes() == before


def test_same_inputs_and_timestamp_are_byte_stable(settings: Settings) -> None:
    _write_inputs(settings)
    first = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )
    path = _artifact_path(settings)
    first_bytes = path.read_bytes()

    second = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )
    assert second.payload == first.payload
    assert path.read_bytes() == first_bytes


def test_progress_callback_failure_does_not_replace_previous(
    settings: Settings,
) -> None:
    _write_inputs(settings)
    infer_harmony_job(JOB_ID, settings, created_at=FIXED_HARMONY_AT)
    path = _artifact_path(settings)
    before = path.read_bytes()

    def callback(stage: str, _message: str, _progress: float) -> None:
        if stage == "validating_harmonic_context":
            raise RuntimeError("callback /private/error")

    with pytest.raises(HarmonyPipelineError, match="progress reporting failed"):
        infer_harmony_job(
            JOB_ID,
            settings,
            callback,
            created_at="2026-08-14T04:01:00+00:00",
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"version": "bad version"}, "pipeline version is invalid"),
        ({"created_at": "2026-08-14T12:00:00+08:00"}, "must be UTC"),
        ({"created_at": "not-a-time"}, "timestamp is invalid"),
        ({"inference_processor": None}, "processor is invalid"),
        ({"stage_callback": "not-callable"}, "callback is invalid"),
    ],
)
def test_invalid_configuration_is_rejected(
    settings: Settings,
    kwargs: dict,
    message: str,
) -> None:
    _write_inputs(settings)
    call_kwargs = dict(kwargs)
    callback = call_kwargs.pop("stage_callback", None)
    with pytest.raises(HarmonyPipelineError, match=message):
        infer_harmony_job(
            JOB_ID,
            settings,
            callback,
            **call_kwargs,
        )


def test_invalid_settings_are_rejected() -> None:
    with pytest.raises(HarmonyPipelineError, match="settings are invalid"):
        infer_harmony_job(JOB_ID, None)  # type: ignore[arg-type]


def test_returned_payload_is_detached_from_saved_artifact(settings: Settings) -> None:
    _write_inputs(settings)
    result = infer_harmony_job(
        JOB_ID,
        settings,
        created_at=FIXED_HARMONY_AT,
    )
    result.payload["rawEvidence"][0]["midiPitch"] = 0

    stored = load_harmony_artifact(JOB_ID, settings)
    assert stored is not None
    assert stored["rawEvidence"][0]["midiPitch"] != 0


def test_pipeline_does_not_claim_final_notation_or_export() -> None:
    source = Path(__file__).resolve().parents[1] / "app" / "harmony_pipeline.py"
    text = source.read_text(encoding="utf-8").lower()
    for forbidden in (
        "musicxml",
        "midi export",
        "tablature",
        "engraving",
        "publication-ready",
        "final chord chart",
    ):
        assert forbidden not in text
