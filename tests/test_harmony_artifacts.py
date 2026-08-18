from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import pytest

from app.config import Settings
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HARMONY_ARTIFACT_SCHEMA_VERSION,
    HarmonyArtifactError,
    HarmonyArtifactUnavailableError,
    HarmonyArtifactValidationError,
    build_harmony_artifact,
    harmony_artifact_path,
    harmony_artifact_scope,
    harmony_attempt_artifact_file_name,
    load_harmony_artifact,
    reconcile_harmony_attempt_artifacts,
    validate_harmony_artifact,
    write_harmony_artifact,
)
from app.harmony_inference import HARMONY_INFERENCE_VERSION, infer_harmony
from app.transcription_draft import INTERPRETATION_DRAFT_RELATIVE_PATH
from app.transcription_events import RAW_TRANSCRIPTION_RELATIVE_PATH


JOB_ID = "d" * 32


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


def event(
    event_id: str,
    midi_note: int,
    *,
    source: str = "other",
    start: float = 0.0,
    end: float = 1.0,
    confidence: float = 0.9,
    midi_pitch: float | None = None,
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
        "warnings": [],
    }


def timing() -> dict:
    return {"beatsSeconds": [0.0, 1.0, 2.0], "beatConfidence": 0.9}


def tonality() -> dict:
    return {
        "primaryCandidate": {
            "tonalCenter": "C",
            "collection": "ionian",
            "displayName": "C ionian",
            "confidence": 0.8,
        }
    }


def resolved_result():
    return infer_harmony(
        [
            event("c", 60, source="vocals", midi_pitch=60.12),
            event("bass_e", 52, source="bass", confidence=0.95, midi_pitch=52.08),
            event("e", 64, source="other", midi_pitch=63.91),
            event("g", 67, source="other", midi_pitch=67.04),
            event(
                "fsharp",
                66,
                source="other",
                confidence=0.18,
                midi_pitch=66.03,
            ),
        ],
        timing(),
        tonality(),
    )


def unresolved_result():
    return infer_harmony([event("c", 60, midi_pitch=60.2)], timing())


def artifact_payload(*, interpretation: bool = False) -> dict:
    return build_harmony_artifact(
        resolved_result(),
        harmony_version="harmonic-context-v1",
        created_at="2026-08-14T03:00:00Z",
        transcription_version="raw-transcription-v1",
        analysis_version="baseline-librosa-v1",
        interpretation_version=(
            "editable-interpretation-v1" if interpretation else None
        ),
    )


def test_constants_are_canonical() -> None:
    assert HARMONY_ARTIFACT_RELATIVE_PATH == "harmony/harmonic-context.json"
    assert HARMONY_ARTIFACT_SCHEMA_VERSION == 1


def test_builder_preserves_raw_evidence_and_canonical_provenance() -> None:
    result = resolved_result()
    before = result.payload()
    artifact = build_harmony_artifact(
        result,
        harmony_version="harmonic-context-v1",
        created_at="2026-08-14T03:00:00Z",
        transcription_version="raw-transcription-v1",
        analysis_version="baseline-librosa-v1",
    )

    assert result.payload() == before
    assert artifact["createdAt"] == "2026-08-14T03:00:00+00:00"
    assert artifact["sourceTranscription"] == {
        "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
        "schemaVersion": 1,
        "transcriptionVersion": "raw-transcription-v1",
    }
    assert artifact["sourceAnalysis"] == {
        "fileName": "analysis/audio-analysis.json",
        "schemaVersion": 1,
        "analysisVersion": "baseline-librosa-v1",
    }
    assert "sourceInterpretation" not in artifact
    assert artifact["algorithms"]["harmonyInference"]["version"] == (
        HARMONY_INFERENCE_VERSION
    )
    raw = {item["id"]: item for item in artifact["rawEvidence"]}
    assert raw["c"]["rawStartSeconds"] == 0.0
    assert raw["c"]["rawEndSeconds"] == 1.0
    assert raw["c"]["midiPitch"] == 60.12
    assert raw["c"]["sourceKind"] == "vocals"
    assert artifact["diagnostics"]["rawEvidenceIncluded"] is True
    assert artifact["diagnostics"]["fractionalPitchPreserved"] is True


def test_builder_can_record_optional_interpretation_provenance() -> None:
    artifact = artifact_payload(interpretation=True)
    assert artifact["sourceInterpretation"] == {
        "fileName": INTERPRETATION_DRAFT_RELATIVE_PATH,
        "schemaVersion": 1,
        "draftVersion": "editable-interpretation-v1",
    }


def test_builder_requires_the_typed_inference_result() -> None:
    with pytest.raises(HarmonyArtifactValidationError, match="HarmonyInferenceResult"):
        build_harmony_artifact(  # type: ignore[arg-type]
            {},
            harmony_version="harmonic-context-v1",
            created_at="2026-08-14T03:00:00+00:00",
            transcription_version="raw-transcription-v1",
            analysis_version="baseline-librosa-v1",
        )


def test_validation_normalizes_unordered_collections_deterministically() -> None:
    expected = artifact_payload()
    shuffled = copy.deepcopy(expected)
    shuffled["rawEvidence"].reverse()
    shuffled["segments"].reverse()
    shuffled["diagnostics"]["sourceKinds"].reverse()
    shuffled["diagnostics"]["chordVocabulary"].reverse()
    for segment in shuffled["segments"]:
        segment["supportingEventIds"].reverse()
        segment["sourceKinds"].reverse()
        segment["partIds"].reverse()
        segment["voiceIds"].reverse()
        segment["unassignedContextEventIds"].reverse()
        segment["observedPitchClasses"].reverse()
        if segment["primaryCandidate"]:
            segment["primaryCandidate"]["evidenceEventIds"].reverse()
        for candidate in segment["alternatives"]:
            candidate["evidenceEventIds"].reverse()
    normalized = validate_harmony_artifact(shuffled)

    assert normalized == expected
    assert validate_harmony_artifact(normalized) == normalized


def test_write_load_and_canonical_path_round_trip(settings: Settings) -> None:
    payload = artifact_payload(interpretation=True)
    path = write_harmony_artifact(JOB_ID, settings, payload)

    assert path == settings.exports_dir / JOB_ID / HARMONY_ARTIFACT_RELATIVE_PATH
    assert load_harmony_artifact(JOB_ID, settings) == payload
    assert harmony_artifact_path(JOB_ID, settings) == path.resolve(strict=True)
    stored = json.loads(path.read_text(encoding="utf-8"))
    assert stored == payload


def test_loaded_artifacts_are_independent_copies(settings: Settings) -> None:
    write_harmony_artifact(JOB_ID, settings, artifact_payload())
    first = load_harmony_artifact(JOB_ID, settings)
    second = load_harmony_artifact(JOB_ID, settings)
    assert first is not None and second is not None

    first["rawEvidence"][0]["midiPitch"] = 0
    first["segments"][0]["supportingEventIds"].clear()
    assert second["rawEvidence"][0]["midiPitch"] != 0
    assert second["segments"][0]["supportingEventIds"]


def test_missing_artifact_is_explicit(settings: Settings) -> None:
    assert load_harmony_artifact(JOB_ID, settings) is None
    with pytest.raises(HarmonyArtifactUnavailableError):
        harmony_artifact_path(JOB_ID, settings)


def test_invalid_replacement_does_not_touch_previous_artifact(
    settings: Settings,
) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    invalid = artifact_payload()
    invalid["diagnostics"]["eventCount"] = 999

    with pytest.raises(HarmonyArtifactValidationError, match="eventCount"):
        write_harmony_artifact(JOB_ID, settings, invalid)
    assert path.read_bytes() == before


def test_pre_replace_failure_preserves_previous_artifact(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"

    import app.harmony_artifacts as module

    def fail_replace(_source: Path, _destination: Path) -> None:
        raise HarmonyArtifactError("simulated publication failure")

    monkeypatch.setattr(module, "_replace_atomic", fail_replace)
    with pytest.raises(HarmonyArtifactError, match="simulated"):
        write_harmony_artifact(JOB_ID, settings, replacement)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_post_replace_failure_restores_previous_and_syncs_recovery(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"

    import app.harmony_artifacts as module

    calls: list[Path] = []

    def fail_then_succeed(directory: Path) -> None:
        calls.append(directory)
        if len(calls) == 1:
            raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_then_succeed)
    with pytest.raises(HarmonyArtifactError, match="publication sync"):
        write_harmony_artifact(JOB_ID, settings, replacement)
    assert len(calls) == 2
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_first_publication_sync_failure_removes_and_syncs_recovery(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    target = settings.exports_dir / JOB_ID / HARMONY_ARTIFACT_RELATIVE_PATH
    calls: list[Path] = []

    def fail_then_succeed(directory: Path) -> None:
        calls.append(directory)
        if len(calls) == 1:
            raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_then_succeed)
    with pytest.raises(HarmonyArtifactError, match="publication sync"):
        write_harmony_artifact(JOB_ID, settings, artifact_payload())
    assert len(calls) == 2
    assert not target.exists()
    assert not list(target.parent.glob(".*.tmp"))


def test_recovery_sync_failure_is_not_reported_as_safe_restoration(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"

    import app.harmony_artifacts as module

    calls = 0

    def fail_sync(_directory: Path) -> None:
        nonlocal calls
        calls += 1
        raise HarmonyArtifactError("simulated sync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_sync)
    with pytest.raises(HarmonyArtifactError, match="could not be restored safely"):
        write_harmony_artifact(JOB_ID, settings, replacement)
    assert calls == 2
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".*.tmp"))


def test_restore_none_removal_syncs_directory(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    target = harmony_attempt_artifact_file_name("a" * 32)
    path = write_harmony_artifact(
        JOB_ID,
        settings,
        artifact_payload(),
        artifact_file_name=target,
    )
    calls: list[Path] = []
    monkeypatch.setattr(module, "_fsync_directory", calls.append)
    module._restore_harmony_artifact(
        JOB_ID,
        settings,
        None,
        artifact_file_name=target,
    )
    assert not path.exists()
    assert calls == [path.parent]


def test_windows_directory_sync_policy_is_explicitly_noop(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    monkeypatch.setattr(module, "_directory_fsync_supported", lambda: False)

    def unexpected_open(*_args, **_kwargs):
        raise AssertionError("Windows policy must not open a directory descriptor")

    monkeypatch.setattr(module.os, "open", unexpected_open)
    module._fsync_directory(settings.exports_dir / JOB_ID)


def test_artifact_scope_is_nested_and_resets_after_exception(settings: Settings) -> None:
    outer = harmony_attempt_artifact_file_name("a" * 32)
    inner = harmony_attempt_artifact_file_name("b" * 32)

    with harmony_artifact_scope(outer):
        outer_path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
        with harmony_artifact_scope(inner):
            inner_path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
        outer_again = write_harmony_artifact(JOB_ID, settings, artifact_payload())

    assert outer_path.name == Path(outer).name
    assert inner_path.name == Path(inner).name
    assert outer_again == outer_path

    with pytest.raises(RuntimeError, match="scope reset"):
        with harmony_artifact_scope(outer):
            raise RuntimeError("scope reset")
    legacy_path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    assert legacy_path.name == "harmonic-context.json"


def test_corrupt_json_and_schema_are_not_exposed(settings: Settings) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(HarmonyArtifactError, match="corrupted"):
        load_harmony_artifact(JOB_ID, settings)

    path.write_text(json.dumps(artifact_payload()), encoding="utf-8")
    stored = json.loads(path.read_text(encoding="utf-8"))
    stored["segments"][0]["supportingEventIds"] = ["missing"]
    path.write_text(json.dumps(stored), encoding="utf-8")
    with pytest.raises(HarmonyArtifactError, match="schema validation"):
        load_harmony_artifact(JOB_ID, settings)


def test_symlinked_file_is_rejected(settings: Settings, tmp_path: Path) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    outside = tmp_path / "outside.json"
    path.replace(outside)
    try:
        path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(HarmonyArtifactError, match="unsafe"):
        load_harmony_artifact(JOB_ID, settings)
    with pytest.raises(HarmonyArtifactError):
        harmony_artifact_path(JOB_ID, settings)


def test_symlinked_harmony_directory_is_rejected(
    settings: Settings,
    tmp_path: Path,
) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = settings.exports_dir / JOB_ID / "harmony"
    try:
        directory.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")

    with pytest.raises(HarmonyArtifactError, match="unsafe"):
        write_harmony_artifact(JOB_ID, settings, artifact_payload())


def test_download_path_detects_change_after_validation(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    import app.harmony_artifacts as module

    original = module.load_harmony_artifact

    def mutate_after_load(job_id: str, app_settings: Settings) -> dict | None:
        loaded = original(job_id, app_settings)
        path.write_text(path.read_text(encoding="utf-8") + " ", encoding="utf-8")
        return loaded

    monkeypatch.setattr(module, "load_harmony_artifact", mutate_after_load)
    with pytest.raises(HarmonyArtifactError, match="changed during validation"):
        harmony_artifact_path(JOB_ID, settings)


def test_invalid_job_id_fails_safely(settings: Settings) -> None:
    with pytest.raises(HarmonyArtifactError, match="invalid"):
        load_harmony_artifact("../bad", settings)
    with pytest.raises(HarmonyArtifactError, match="invalid"):
        write_harmony_artifact("not-a-job", settings, artifact_payload())


@pytest.mark.parametrize(
    ("section", "field", "bad"),
    [
        ("sourceTranscription", "fileName", "transcription/other.json"),
        ("sourceTranscription", "fileName", "../raw-events.json"),
        ("sourceAnalysis", "fileName", "analysis/../audio-analysis.json"),
    ],
)
def test_noncanonical_source_paths_are_rejected(
    section: str,
    field: str,
    bad: str,
) -> None:
    payload = artifact_payload()
    payload[section][field] = bad
    with pytest.raises(HarmonyArtifactValidationError, match="canonical"):
        validate_harmony_artifact(payload)


def test_noncanonical_interpretation_source_is_rejected() -> None:
    payload = artifact_payload(interpretation=True)
    payload["sourceInterpretation"]["fileName"] = "interpretation/other.json"
    with pytest.raises(HarmonyArtifactValidationError, match="canonical"):
        validate_harmony_artifact(payload)


def test_duplicate_and_inconsistent_raw_evidence_is_rejected() -> None:
    payload = artifact_payload()
    payload["rawEvidence"].append(copy.deepcopy(payload["rawEvidence"][0]))
    with pytest.raises(HarmonyArtifactValidationError, match="Duplicate raw evidence"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["rawEvidence"][0]["pitchClass"] = 11
    with pytest.raises(HarmonyArtifactValidationError, match="pitchClass"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["rawEvidence"][0]["pitchName"] = "H"
    with pytest.raises(HarmonyArtifactValidationError, match="pitchName"):
        validate_harmony_artifact(payload)


def test_raw_evidence_warning_bounds_match_canonical_raw_contract() -> None:
    payload = artifact_payload()
    warnings = [f"Review warning {index}." for index in range(128)]
    warnings[8] = "x" * 500
    payload["rawEvidence"][0]["warnings"] = warnings
    validated = validate_harmony_artifact(payload)
    assert validated["rawEvidence"][0]["warnings"] == warnings

    payload["rawEvidence"][0]["warnings"] = warnings + ["overflow"]
    with pytest.raises(HarmonyArtifactValidationError, match="too many warnings"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["rawEvidence"][0]["warnings"] = ["x" * 501]
    with pytest.raises(HarmonyArtifactValidationError):
        validate_harmony_artifact(payload)


def test_legacy_missing_raw_warning_field_preserves_missingness() -> None:
    payload = artifact_payload()
    payload["rawEvidence"][0].pop("warnings")
    validated = validate_harmony_artifact(payload)
    assert "warnings" not in validated["rawEvidence"][0]
    assert validated["rawEvidence"][1]["warnings"] == []


def test_attempt_scoped_artifact_requires_explicit_warning_fields(settings: Settings) -> None:
    payload = artifact_payload()
    payload["rawEvidence"][0].pop("warnings")
    legacy_path = write_harmony_artifact(JOB_ID, settings, payload)
    legacy = load_harmony_artifact(JOB_ID, settings)
    assert legacy_path.name == "harmonic-context.json"
    assert legacy is not None and "warnings" not in legacy["rawEvidence"][0]

    target = harmony_attempt_artifact_file_name("a" * 32)
    with pytest.raises(HarmonyArtifactValidationError, match="explicit warning"):
        write_harmony_artifact(
            JOB_ID,
            settings,
            payload,
            artifact_file_name=target,
        )


def test_reconcile_removes_only_non_durable_non_active_attempt_files(
    settings: Settings,
) -> None:
    legacy = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    durable_name = harmony_attempt_artifact_file_name("a" * 32)
    active_name = harmony_attempt_artifact_file_name("b" * 32)
    orphan_one = harmony_attempt_artifact_file_name("c" * 32)
    orphan_two = harmony_attempt_artifact_file_name("e" * 32)
    for target in (durable_name, active_name, orphan_one, orphan_two):
        write_harmony_artifact(
            JOB_ID,
            settings,
            artifact_payload(),
            artifact_file_name=target,
        )

    removed = reconcile_harmony_attempt_artifacts(
        JOB_ID,
        settings,
        durable_artifact_file_name=durable_name,
        active_attempt_id="b" * 32,
    )
    directory = legacy.parent
    assert removed == 2
    assert legacy.exists()
    assert (directory / Path(durable_name).name).exists()
    assert (directory / Path(active_name).name).exists()
    assert not (directory / Path(orphan_one).name).exists()
    assert not (directory / Path(orphan_two).name).exists()


def test_reconcile_malformed_or_symlink_candidate_fails_closed(
    settings: Settings,
    tmp_path: Path,
) -> None:
    directory = settings.exports_dir / JOB_ID / "harmony"
    directory.mkdir()
    malformed = directory / "harmonic-context.not-a-nonce.json"
    malformed.write_text("leave me", encoding="utf-8")
    with pytest.raises(HarmonyArtifactError, match="reconciled"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id=None,
        )
    assert malformed.read_text(encoding="utf-8") == "leave me"

    malformed.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text("outside", encoding="utf-8")
    candidate = directory / f"harmonic-context.{'f' * 32}.json"
    try:
        candidate.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(HarmonyArtifactError, match="reconciled"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id=None,
        )
    assert candidate.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"


def test_segment_references_and_derived_sources_are_verified() -> None:
    payload = artifact_payload()
    payload["segments"][0]["supportingEventIds"] = ["missing"]
    with pytest.raises(HarmonyArtifactValidationError, match="unknown raw evidence"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["segments"][0]["sourceKinds"] = ["full_mix"]
    with pytest.raises(HarmonyArtifactValidationError, match="sourceKinds"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["segments"][0]["observedPitchClasses"].pop()
    with pytest.raises(HarmonyArtifactValidationError, match="observedPitchClasses"):
        validate_harmony_artifact(payload)


def test_observed_pitch_ratios_must_be_normalized() -> None:
    payload = artifact_payload()
    payload["segments"][0]["observedPitchClasses"][0]["weightRatio"] = 0.9
    with pytest.raises(HarmonyArtifactValidationError, match="sum to 1"):
        validate_harmony_artifact(payload)


def test_candidate_references_must_be_supporting_chord_tones() -> None:
    payload = artifact_payload()
    candidate = payload["segments"][0]["primaryCandidate"]
    assert candidate is not None
    candidate["evidenceEventIds"] = ["missing"]
    with pytest.raises(HarmonyArtifactValidationError, match="unknown raw evidence"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    segment = payload["segments"][0]
    candidate = segment["primaryCandidate"]
    assert candidate is not None
    non_chord = next(
        event["id"]
        for event in payload["rawEvidence"]
        if event["pitchClass"] not in candidate["pitchClasses"]
    )
    segment["supportingEventIds"].append(non_chord)
    segment["supportingEventIds"] = list(dict.fromkeys(segment["supportingEventIds"]))
    segment["sourceKinds"] = sorted(
        {
            event["sourceKind"]
            for event in payload["rawEvidence"]
            if event["id"] in segment["supportingEventIds"]
        }
    )
    candidate["evidenceEventIds"] = [non_chord]
    with pytest.raises(HarmonyArtifactValidationError, match="non-chord"):
        validate_harmony_artifact(payload)


def test_forged_inversion_requires_explicit_matching_bass_evidence() -> None:
    payload = artifact_payload()
    candidate = payload["segments"][0]["primaryCandidate"]
    assert candidate is not None and "inversionCandidate" in candidate
    candidate["inversionCandidate"]["sourceEventIds"] = ["e"]

    with pytest.raises(HarmonyArtifactValidationError, match="bass-source"):
        validate_harmony_artifact(payload)


def test_resolved_and_unresolved_semantics_are_cross_checked() -> None:
    payload = artifact_payload()
    payload["segments"][0]["unresolved"] = True
    with pytest.raises(HarmonyArtifactValidationError, match="must be null"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["segments"][0]["primaryCandidate"] = None
    with pytest.raises(HarmonyArtifactValidationError, match="required when resolved"):
        validate_harmony_artifact(payload)


def test_unresolved_result_retains_raw_evidence_and_no_fabricated_chord() -> None:
    artifact = build_harmony_artifact(
        unresolved_result(),
        harmony_version="harmonic-context-v1",
        created_at="2026-08-14T03:00:00+00:00",
        transcription_version="raw-transcription-v1",
        analysis_version="baseline-librosa-v1",
    )
    segment = artifact["segments"][0]

    assert segment["unresolved"] is True
    assert segment["primaryCandidate"] is None
    assert artifact["unresolvedEventIds"] == ["c"]
    assert artifact["rawEvidence"][0]["midiPitch"] == 60.2


def test_unresolved_ids_and_diagnostic_counts_come_from_artifact_truth() -> None:
    payload = artifact_payload()
    payload["unresolvedEventIds"] = [payload["rawEvidence"][0]["id"]]
    with pytest.raises(HarmonyArtifactValidationError, match="unresolvedEventIds"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["diagnostics"]["segmentCount"] = 999
    with pytest.raises(HarmonyArtifactValidationError, match="segmentCount"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload["diagnostics"]["sourceKinds"] = ["full_mix"]
    with pytest.raises(HarmonyArtifactValidationError, match="sourceKinds"):
        validate_harmony_artifact(payload)


def test_candidate_quality_must_be_declared_in_open_vocabulary() -> None:
    payload = artifact_payload()
    candidate = payload["segments"][0]["primaryCandidate"]
    assert candidate is not None
    candidate["quality"] = "future_extension"
    with pytest.raises(HarmonyArtifactValidationError, match="chordVocabulary"):
        validate_harmony_artifact(payload)

    payload["diagnostics"]["chordVocabulary"].append("future_extension")
    validated = validate_harmony_artifact(payload)
    assert validated["segments"][0]["primaryCandidate"]["quality"] == (
        "future_extension"
    )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf, True, "0.5"])
def test_nonfinite_or_wrong_type_numbers_are_rejected(bad: object) -> None:
    payload = artifact_payload()
    payload["rawEvidence"][0]["midiPitch"] = bad
    with pytest.raises(HarmonyArtifactValidationError):
        validate_harmony_artifact(payload)


def test_json_nan_constants_are_rejected_on_load(settings: Settings) -> None:
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    text = path.read_text(encoding="utf-8")
    text = text.replace('"confidence":0.8', '"confidence":NaN', 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(HarmonyArtifactError, match="corrupted"):
        load_harmony_artifact(JOB_ID, settings)


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload["warnings"].append("debug at /home/user/private.json"),
        lambda payload: payload["warnings"].append("<script>alert(1)</script>"),
        lambda payload: payload["warnings"].append("api_key=private-value"),
        lambda payload: payload["rawEvidence"][0].update(
            warnings=["api_key=private-value"]
        ),
        lambda payload: payload["segments"][0]["primaryCandidate"].update(
            symbol="https://bad.invalid/chord"
        ),
        lambda payload: payload["tonalContext"].update(
            displayName="C ionian\nsecret"
        ),
    ],
)
def test_hostile_text_is_rejected(mutator) -> None:
    payload = artifact_payload()
    mutator(payload)
    with pytest.raises(HarmonyArtifactValidationError, match="unsafe text"):
        validate_harmony_artifact(payload)


def test_artifact_size_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    import app.harmony_artifacts as module

    payload = artifact_payload()
    monkeypatch.setattr(module, "_MAX_ARTIFACT_BYTES", 128)
    with pytest.raises(HarmonyArtifactValidationError, match="too large"):
        validate_harmony_artifact(payload)


def test_unknown_fields_and_non_string_keys_are_rejected() -> None:
    payload = artifact_payload()
    payload["machinePath"] = "/tmp/private"
    with pytest.raises(HarmonyArtifactValidationError, match="unsupported fields"):
        validate_harmony_artifact(payload)

    payload = artifact_payload()
    payload[1] = "bad"  # type: ignore[index]
    with pytest.raises(HarmonyArtifactValidationError):
        validate_harmony_artifact(payload)


def test_no_score_export_or_final_notation_is_generated() -> None:
    artifact = artifact_payload()
    diagnostics = artifact["diagnostics"]
    assert diagnostics["romanNumeralsGenerated"] is False
    assert diagnostics["guitarVoicingsGenerated"] is False
    assert diagnostics["notationGenerated"] is False

    encoded = json.dumps(artifact).lower()
    for forbidden in (
        "musicxml",
        "midi export",
        "tablature",
        "engraving",
        "fret assignment",
        "publication-ready",
    ):
        assert forbidden not in encoded
