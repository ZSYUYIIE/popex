from __future__ import annotations

import copy
import json
import math
import os
import stat
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from threading import Barrier, Event

import pytest

import app.harmony_artifacts as harmony_artifacts_module
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


def test_attempt_publication_is_idempotent_and_never_replaces_content(
    settings: Settings,
) -> None:
    target = harmony_attempt_artifact_file_name("a" * 32)
    payload = artifact_payload()
    path = write_harmony_artifact(
        JOB_ID,
        settings,
        payload,
        artifact_file_name=target,
    )
    before = path.read_bytes()
    before_info = path.stat()

    repeated = write_harmony_artifact(
        JOB_ID,
        settings,
        copy.deepcopy(payload),
        artifact_file_name=target,
    )

    assert repeated == path
    assert path.read_bytes() == before
    assert (path.stat().st_dev, path.stat().st_ino) == (
        before_info.st_dev,
        before_info.st_ino,
    )

    different = artifact_payload()
    different["harmonyVersion"] = "harmonic-context-v2"
    with pytest.raises(HarmonyArtifactError, match="different content"):
        write_harmony_artifact(
            JOB_ID,
            settings,
            different,
            artifact_file_name=target,
        )
    assert path.read_bytes() == before
    assert (path.stat().st_dev, path.stat().st_ino) == (
        before_info.st_dev,
        before_info.st_ino,
    )


def test_idempotent_attempt_winner_survives_first_publishers_sync_failure(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    target = harmony_attempt_artifact_file_name("a" * 32)
    payload = artifact_payload()
    sync_entered = Event()
    release_sync = Event()
    sync_calls = 0
    real_sync = module._fsync_directory

    def fail_only_first_install(directory: Path) -> None:
        nonlocal sync_calls
        sync_calls += 1
        if sync_calls == 1:
            sync_entered.set()
            assert release_sync.wait(5)
            raise HarmonyArtifactError("simulated post-link durability failure")
        real_sync(directory)

    monkeypatch.setattr(module, "_fsync_directory", fail_only_first_install)
    with ThreadPoolExecutor(max_workers=1) as executor:
        first = executor.submit(
            write_harmony_artifact,
            JOB_ID,
            settings,
            payload,
            artifact_file_name=target,
        )
        assert sync_entered.wait(5)
        try:
            winner = write_harmony_artifact(
                JOB_ID,
                settings,
                copy.deepcopy(payload),
                artifact_file_name=target,
            )
            winner_bytes = winner.read_bytes()
        finally:
            release_sync.set()
        with pytest.raises(HarmonyArtifactError, match="post-link durability"):
            first.result(timeout=5)

    assert winner.is_file()
    assert sync_calls == 2
    assert winner.read_bytes() == winner_bytes
    assert load_harmony_artifact(
        JOB_ID,
        settings,
        artifact_file_name=target,
    ) == payload


def test_concurrent_attempt_publications_cannot_overwrite_each_other(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    target = harmony_attempt_artifact_file_name("a" * 32)
    first_payload = artifact_payload()
    second_payload = artifact_payload()
    second_payload["harmonyVersion"] = "harmonic-context-v2"
    publication_barrier = Barrier(2)
    real_link = module._link_atomic

    def synchronized_link(source: Path, destination: Path) -> None:
        publication_barrier.wait()
        real_link(source, destination)

    def publish(payload: dict) -> tuple[str, object]:
        try:
            return (
                "published",
                write_harmony_artifact(
                    JOB_ID,
                    settings,
                    payload,
                    artifact_file_name=target,
                ),
            )
        except HarmonyArtifactError as exc:
            return "conflict", exc

    monkeypatch.setattr(module, "_link_atomic", synchronized_link)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish, (first_payload, second_payload)))

    assert [status for status, _value in results].count("published") == 1
    assert [status for status, _value in results].count("conflict") == 1
    stored = load_harmony_artifact(
        JOB_ID,
        settings,
        artifact_file_name=target,
    )
    assert stored in (first_payload, second_payload)


def test_rollback_refuses_to_replace_a_newer_file_identity(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"
    winner = artifact_payload()
    winner["harmonyVersion"] = "harmonic-context-v3"
    winner_bytes = module._encoded_payload(
        module.validate_harmony_artifact(winner)
    )
    real_restore = module._restore_publication_state

    def install_winner_before_rollback(destination, *args, **kwargs):
        concurrent = destination.with_name(f".{destination.name}.winner.tmp")
        concurrent.write_bytes(winner_bytes)
        os.replace(concurrent, destination)
        return real_restore(destination, *args, **kwargs)

    def fail_sync(_directory: Path) -> None:
        raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(
        module,
        "_restore_publication_state",
        install_winner_before_rollback,
    )
    monkeypatch.setattr(module, "_fsync_directory", fail_sync)

    with pytest.raises(HarmonyArtifactError, match="could not be restored safely"):
        write_harmony_artifact(JOB_ID, settings, replacement)

    assert path.read_bytes() == winner_bytes


def test_rollback_rechecks_identity_at_the_atomic_replace_boundary(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"
    winner = artifact_payload()
    winner["harmonyVersion"] = "harmonic-context-v3"
    winner_bytes = module._encoded_payload(
        module.validate_harmony_artifact(winner)
    )
    real_replace = module._replace_atomic

    def install_winner_in_final_rollback_window(
        source: Path,
        destination: Path,
    ) -> None:
        if source.name.endswith(".restore.tmp"):
            concurrent = destination.with_name(
                f".{destination.name}.winner-window.tmp"
            )
            concurrent.write_bytes(winner_bytes)
            os.replace(concurrent, destination)
        real_replace(source, destination)

    def fail_sync(_directory: Path) -> None:
        raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(module, "_replace_atomic", install_winner_in_final_rollback_window)
    monkeypatch.setattr(module, "_fsync_directory", fail_sync)

    with pytest.raises(HarmonyArtifactError, match="could not be restored safely"):
        write_harmony_artifact(JOB_ID, settings, replacement)

    assert path.read_bytes() == winner_bytes


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX rename and symlink semantics",
)
def test_publication_parent_replacement_never_writes_external_files(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    if not module._descriptor_relative_publication_supported():
        pytest.skip("descriptor-relative publication is unavailable")
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"
    directory = path.parent
    moved_directory = directory.with_name("harmony-before-publication-swap")
    outside = tmp_path / "outside-publication"
    outside.mkdir()
    outside_target = outside / path.name
    outside_target.write_text("external destination sentinel", encoding="utf-8")
    outside_temporary: Path | None = None
    real_replace = module._replace_atomic

    def swap_parent_then_replace(source: Path, destination: Path) -> None:
        nonlocal outside_temporary
        destination.parent.rename(moved_directory)
        destination.parent.symlink_to(outside, target_is_directory=True)
        outside_temporary = outside / source.name
        outside_temporary.write_text("external temporary sentinel", encoding="utf-8")
        real_replace(source, destination)

    monkeypatch.setattr(module, "_replace_atomic", swap_parent_then_replace)

    with pytest.raises(HarmonyArtifactError):
        write_harmony_artifact(JOB_ID, settings, replacement)

    assert outside_target.read_text(encoding="utf-8") == (
        "external destination sentinel"
    )
    assert outside_temporary is not None
    assert outside_temporary.read_text(encoding="utf-8") == (
        "external temporary sentinel"
    )
    assert (moved_directory / path.name).read_bytes() == before


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX rename and symlink semantics",
)
def test_rollback_parent_replacement_never_writes_external_files(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    if not module._descriptor_relative_publication_supported():
        pytest.skip("descriptor-relative publication is unavailable")
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"
    directory = path.parent
    moved_directory = directory.with_name("harmony-before-rollback-swap")
    outside = tmp_path / "outside-rollback"
    outside.mkdir()
    outside_target = outside / path.name
    outside_target.write_text("external destination sentinel", encoding="utf-8")
    outside_temporary: Path | None = None
    replace_calls = 0
    real_replace = module._replace_atomic

    def swap_only_rollback(source: Path, destination: Path) -> None:
        nonlocal outside_temporary, replace_calls
        replace_calls += 1
        if replace_calls == 2:
            destination.parent.rename(moved_directory)
            destination.parent.symlink_to(outside, target_is_directory=True)
            outside_temporary = outside / source.name
            outside_temporary.write_text(
                "external restoration sentinel",
                encoding="utf-8",
            )
        real_replace(source, destination)

    def fail_sync(_directory: Path) -> None:
        raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(module, "_replace_atomic", swap_only_rollback)
    monkeypatch.setattr(module, "_fsync_directory", fail_sync)

    with pytest.raises(HarmonyArtifactError, match="could not be restored safely"):
        write_harmony_artifact(JOB_ID, settings, replacement)

    assert replace_calls == 2
    assert outside_target.read_text(encoding="utf-8") == (
        "external destination sentinel"
    )
    assert outside_temporary is not None
    assert outside_temporary.read_text(encoding="utf-8") == (
        "external restoration sentinel"
    )
    assert (moved_directory / path.name).read_bytes() == before


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX descriptor-relative publication",
)
def test_post_replace_parent_swap_restores_the_pinned_previous_artifact(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    if not module._descriptor_relative_publication_supported():
        pytest.skip("descriptor-relative publication is unavailable")
    path = write_harmony_artifact(JOB_ID, settings, artifact_payload())
    before = path.read_bytes()
    replacement = artifact_payload()
    replacement["harmonyVersion"] = "harmonic-context-v2"
    directory = path.parent
    moved_directory = directory.with_name("harmony-after-replace-swap")
    outside = tmp_path / "outside-after-replace"
    outside.mkdir()
    outside_target = outside / path.name
    outside_target.write_text("external destination sentinel", encoding="utf-8")
    real_replace = module.os.replace
    swapped = False

    def replace_then_swap(source, destination, *args, **kwargs):
        nonlocal swapped
        result = real_replace(source, destination, *args, **kwargs)
        if not swapped and destination == path.name:
            directory.rename(moved_directory)
            directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(module.os, "replace", replace_then_swap)

    with pytest.raises(HarmonyArtifactError):
        write_harmony_artifact(JOB_ID, settings, replacement)

    assert swapped is True
    assert outside_target.read_text(encoding="utf-8") == (
        "external destination sentinel"
    )
    assert (moved_directory / path.name).read_bytes() == before


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX descriptor-relative publication",
)
def test_post_replace_parent_swap_removes_first_publication_from_pinned_directory(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    if not module._descriptor_relative_publication_supported():
        pytest.skip("descriptor-relative publication is unavailable")
    directory = settings.exports_dir / JOB_ID / "harmony"
    moved_directory = directory.with_name("harmony-first-publication-swap")
    outside = tmp_path / "outside-first-publication"
    outside.mkdir()
    outside_target = outside / "harmonic-context.json"
    outside_target.write_text("external destination sentinel", encoding="utf-8")
    real_replace = module.os.replace
    swapped = False

    def replace_then_swap(source, destination, *args, **kwargs):
        nonlocal swapped
        result = real_replace(source, destination, *args, **kwargs)
        if not swapped and destination == "harmonic-context.json":
            directory.rename(moved_directory)
            directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return result

    monkeypatch.setattr(module.os, "replace", replace_then_swap)

    with pytest.raises(HarmonyArtifactError):
        write_harmony_artifact(JOB_ID, settings, artifact_payload())

    assert swapped is True
    assert outside_target.read_text(encoding="utf-8") == (
        "external destination sentinel"
    )
    assert not (moved_directory / "harmonic-context.json").exists()


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
    temporary_files = list(path.parent.glob(".*.tmp"))
    assert bool(temporary_files) is (
        not module._descriptor_relative_cleanup_supported()
    )


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
    descriptor_calls: list[int] = []

    def fail_then_succeed(directory: Path) -> None:
        calls.append(directory)
        if len(calls) == 1:
            raise HarmonyArtifactError("simulated publication sync failure")

    monkeypatch.setattr(module, "_fsync_directory", fail_then_succeed)
    monkeypatch.setattr(
        module,
        "_fsync_directory_descriptor",
        descriptor_calls.append,
    )
    cleanup_supported = module._descriptor_relative_cleanup_supported()
    expected_error = (
        "publication sync" if cleanup_supported else "could not be restored safely"
    )
    with pytest.raises(HarmonyArtifactError, match=expected_error):
        write_harmony_artifact(JOB_ID, settings, artifact_payload())
    assert len(calls) == 1
    assert len(descriptor_calls) == int(cleanup_supported)
    assert target.exists() is (not cleanup_supported)
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
    descriptor_calls: list[int] = []
    monkeypatch.setattr(
        module,
        "_fsync_directory_descriptor",
        descriptor_calls.append,
    )
    module._restore_harmony_artifact(
        JOB_ID,
        settings,
        None,
        artifact_file_name=target,
    )
    cleanup_supported = module._descriptor_relative_cleanup_supported()
    assert path.exists() is (not cleanup_supported)
    assert len(descriptor_calls) == int(cleanup_supported)


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
    if not harmony_artifacts_module._descriptor_relative_cleanup_supported():
        pytest.skip("descriptor-relative cleanup is unavailable")
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
        protection_state_reader=lambda: (
            "processing",
            durable_name,
            "b" * 32,
        ),
        cleanup_lease=lambda: nullcontext(),
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
    if not harmony_artifacts_module._descriptor_relative_cleanup_supported():
        pytest.skip("descriptor-relative cleanup is unavailable")
    directory = settings.exports_dir / JOB_ID / "harmony"
    directory.mkdir()
    valid_orphan = directory / f"harmonic-context.{'a' * 32}.json"
    valid_orphan.write_text("leave me too", encoding="utf-8")
    malformed = directory / "harmonic-context.not-a-nonce.json"
    malformed.write_text("leave me", encoding="utf-8")
    state_reader = lambda: ("processing", None, "f" * 32)
    with pytest.raises(HarmonyArtifactError, match="reconciled"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id="f" * 32,
            protection_state_reader=state_reader,
            cleanup_lease=lambda: nullcontext(),
        )
    assert malformed.read_text(encoding="utf-8") == "leave me"
    assert valid_orphan.read_text(encoding="utf-8") == "leave me too"

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
            active_attempt_id="f" * 32,
            protection_state_reader=state_reader,
            cleanup_lease=lambda: nullcontext(),
        )
    assert candidate.is_symlink()
    assert outside.read_text(encoding="utf-8") == "outside"
    assert valid_orphan.read_text(encoding="utf-8") == "leave me too"


def test_reconcile_skips_cleanup_without_descriptor_relative_support(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = harmony_attempt_artifact_file_name("a" * 32)
    path = write_harmony_artifact(
        JOB_ID,
        settings,
        artifact_payload(),
        artifact_file_name=target,
    )
    monkeypatch.setattr(
        harmony_artifacts_module,
        "_descriptor_relative_cleanup_supported",
        lambda: False,
    )

    def forbidden_remove(*_args, **_kwargs):
        raise AssertionError("lexical cleanup fallback must not run")

    monkeypatch.setattr(
        harmony_artifacts_module,
        "_remove_published_artifact",
        forbidden_remove,
    )
    state_reads = 0

    def state_reader():
        nonlocal state_reads
        state_reads += 1
        return "processing", None, "b" * 32

    removed = reconcile_harmony_attempt_artifacts(
        JOB_ID,
        settings,
        durable_artifact_file_name=None,
        active_attempt_id="b" * 32,
        protection_state_reader=state_reader,
    )

    assert removed == 0
    assert state_reads == 1
    assert path.is_file()
    harmony_artifacts_module.remove_harmony_artifact(
        JOB_ID,
        settings,
        artifact_file_name=target,
    )
    assert path.is_file()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX special files")
@pytest.mark.parametrize("kind", ["directory", "fifo"])
def test_reconcile_non_regular_candidate_fails_before_deleting_peer(
    settings: Settings,
    kind: str,
) -> None:
    if not harmony_artifacts_module._descriptor_relative_cleanup_supported():
        pytest.skip("descriptor-relative cleanup is unavailable")
    directory = settings.exports_dir / JOB_ID / "harmony"
    directory.mkdir()
    valid = directory / f"harmonic-context.{'a' * 32}.json"
    valid.write_text("keep", encoding="utf-8")
    unsafe = directory / f"harmonic-context.{'c' * 32}.json"
    if kind == "directory":
        unsafe.mkdir()
    else:
        os.mkfifo(unsafe)

    with pytest.raises(HarmonyArtifactError, match="reconciled"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id="f" * 32,
            protection_state_reader=lambda: ("processing", None, "f" * 32),
            cleanup_lease=lambda: nullcontext(),
        )

    assert valid.read_text(encoding="utf-8") == "keep"
    assert unsafe.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX device nodes")
def test_reconcile_device_candidate_fails_before_deleting_peer(
    settings: Settings,
) -> None:
    if not harmony_artifacts_module._descriptor_relative_cleanup_supported():
        pytest.skip("descriptor-relative cleanup is unavailable")
    directory = settings.exports_dir / JOB_ID / "harmony"
    directory.mkdir()
    valid = directory / f"harmonic-context.{'a' * 32}.json"
    valid.write_text("keep", encoding="utf-8")
    unsafe = directory / f"harmonic-context.{'c' * 32}.json"
    try:
        os.mknod(unsafe, 0o600 | stat.S_IFCHR, os.makedev(1, 3))
    except (AttributeError, OSError, PermissionError):
        pytest.skip("device-node creation is unavailable")

    with pytest.raises(HarmonyArtifactError, match="reconciled"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id="f" * 32,
            protection_state_reader=lambda: ("processing", None, "f" * 32),
            cleanup_lease=lambda: nullcontext(),
        )

    assert valid.read_text(encoding="utf-8") == "keep"
    assert unsafe.exists()


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX descriptor cleanup")
def test_reconcile_rechecks_protection_state_for_each_deletion(
    settings: Settings,
) -> None:
    if not harmony_artifacts_module._descriptor_relative_cleanup_supported():
        pytest.skip("descriptor-relative cleanup is unavailable")
    target = harmony_attempt_artifact_file_name("a" * 32)
    path = write_harmony_artifact(
        JOB_ID,
        settings,
        artifact_payload(),
        artifact_file_name=target,
    )
    reads = 0

    def state_reader():
        nonlocal reads
        reads += 1
        if reads == 1:
            return "processing", None, "b" * 32
        return "completed", target, None

    with pytest.raises(HarmonyArtifactError, match="protection state changed"):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id="b" * 32,
            protection_state_reader=state_reader,
            cleanup_lease=lambda: nullcontext(),
        )

    assert reads == 2
    assert path.is_file()


@pytest.mark.skipif(
    os.name != "posix",
    reason="requires POSIX rename and symlink semantics",
)
def test_reconcile_parent_replacement_never_unlinks_external_file(
    settings: Settings,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import app.harmony_artifacts as module

    target = harmony_attempt_artifact_file_name("a" * 32)
    write_harmony_artifact(
        JOB_ID,
        settings,
        artifact_payload(),
        artifact_file_name=target,
    )
    directory = settings.exports_dir / JOB_ID / "harmony"
    moved_directory = settings.exports_dir / JOB_ID / "harmony-before-swap"
    outside = tmp_path / "outside-harmony"
    outside.mkdir()
    outside_target = outside / Path(target).name
    outside_target.write_text("external sentinel", encoding="utf-8")
    original_remove = module._remove_published_artifact
    swapped = False

    def swap_parent_then_remove(path, expected_directory, **kwargs):
        nonlocal swapped
        if not swapped:
            expected_directory.rename(moved_directory)
            expected_directory.symlink_to(outside, target_is_directory=True)
            swapped = True
        return original_remove(path, expected_directory, **kwargs)

    monkeypatch.setattr(
        module,
        "_remove_published_artifact",
        swap_parent_then_remove,
    )

    with pytest.raises(HarmonyArtifactError):
        reconcile_harmony_attempt_artifacts(
            JOB_ID,
            settings,
            durable_artifact_file_name=None,
            active_attempt_id="b" * 32,
            protection_state_reader=lambda: ("processing", None, "b" * 32),
            cleanup_lease=lambda: nullcontext(),
        )

    assert swapped is True
    assert outside_target.read_text(encoding="utf-8") == "external sentinel"
    assert (moved_directory / Path(target).name).is_file()


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
