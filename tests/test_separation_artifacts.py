from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.config import Settings
from app.separation import (
    AUDITED_CHECKPOINT_FILE,
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_DEMUCS_VERSION,
    AUDITED_MODEL_NAME,
    AUDITED_MODEL_REPOSITORY,
    AUDITED_MODEL_REVISION,
    REQUIRED_STEM_KINDS,
    STEM_MANIFEST_RELATIVE_PATH,
)
from app.separation_artifacts import (
    StemArtifactError,
    StemKindNotFoundError,
    StemManifestUnavailableError,
    load_stem_details,
    resolve_stem_artifact,
)


JOB_ID = "a" * 32
RUN_ID = "b" * 32
CREATED_AT = "2026-08-04T01:02:03+00:00"
SEPARATED_AT = "2026-08-04T01:03:04+00:00"
SEPARATION_VERSION = "demucs-worker-v3"
LABELS = {
    "vocals": "Vocals",
    "bass": "Bass",
    "drums": "Drums",
    "other": "Other",
}


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    value = Settings(
        data_dir=tmp_path / "data",
        allowed_hosts=("example.com",),
        max_duration_seconds=1800,
        max_filesize_mb=250,
        max_upload_mb=500,
        audio_quality="192",
    )
    value.ensure_directories()
    return value


def _write_wav(
    path: Path,
    *,
    frames: int = 441,
    sample_rate: int = 44100,
    channels: int = 2,
) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.zeros((frames, channels), dtype=np.float32)
    sf.write(path, samples, sample_rate, format="WAV", subtype="PCM_16")
    info = sf.info(str(path))
    return {
        "durationSeconds": float(info.frames / info.samplerate),
        "sampleRate": int(info.samplerate),
        "channels": int(info.channels),
        "sizeBytes": path.stat().st_size,
    }


def _publish(
    settings: Settings,
    *,
    status: str = "completed",
    separated_at: object = SEPARATED_AT,
    separation_error: object = None,
    warnings: list[str] | None = None,
) -> tuple[dict[str, object], Path, dict[str, object]]:
    job_dir = settings.exports_dir / JOB_ID
    run_dir = job_dir / "stems" / "runs" / RUN_ID
    stems: list[dict[str, object]] = []
    for kind in REQUIRED_STEM_KINDS:
        path = run_dir / f"{kind}.wav"
        metadata = _write_wav(path)
        stems.append(
            {
                "kind": kind,
                "label": LABELS[kind],
                "fileName": f"stems/runs/{RUN_ID}/{kind}.wav",
                **metadata,
            }
        )
    manifest: dict[str, object] = {
        "schemaVersion": 3,
        "separationVersion": SEPARATION_VERSION,
        "createdAt": CREATED_AT,
        "sourceAsset": "analysis.wav",
        "runId": RUN_ID,
        "model": {
            "name": AUDITED_MODEL_NAME,
            "packageVersion": AUDITED_DEMUCS_VERSION,
            "runtimeProfile": "linux-x86_64-cpu-cpython313",
            "workerVersion": "1.0.0",
            "torchVersion": "2.13.0+cpu",
            "huggingfaceHubVersion": "1.26.0",
            "repository": AUDITED_MODEL_REPOSITORY,
            "revision": AUDITED_MODEL_REVISION,
            "checkpointFile": AUDITED_CHECKPOINT_FILE,
            "checkpointSha256": AUDITED_CHECKPOINT_SHA256,
            "weightsIdentifier": f"sha256:{AUDITED_CHECKPOINT_SHA256}",
            "device": "cpu",
        },
        "stems": stems,
        "warnings": list(warnings or []),
    }
    manifest_path = job_dir / STEM_MANIFEST_RELATIVE_PATH
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    job: dict[str, object] = {
        "separation_status": status,
        "stem_manifest_file_name": STEM_MANIFEST_RELATIVE_PATH,
        "separation_error": separation_error,
        "separated_at": separated_at,
    }
    return job, job_dir, manifest


def test_valid_manifest_produces_exact_frontend_payload(settings: Settings) -> None:
    job, _, _ = _publish(settings)
    payload = load_stem_details(JOB_ID, settings, job).payload()

    assert list(payload) == [
        "available",
        "status",
        "model",
        "version",
        "separatedAt",
        "warnings",
        "stems",
        "error",
    ]
    assert payload == {
        "available": True,
        "status": "completed",
        "model": AUDITED_MODEL_NAME,
        "version": SEPARATION_VERSION,
        "separatedAt": SEPARATED_AT,
        "warnings": [],
        "stems": payload["stems"],
        "error": None,
    }
    assert isinstance(payload["stems"], list)
    assert [stem["kind"] for stem in payload["stems"]] == list(REQUIRED_STEM_KINDS)
    for stem in payload["stems"]:
        assert list(stem) == [
            "kind",
            "label",
            "fileName",
            "sizeBytes",
            "durationSeconds",
            "sampleRate",
            "channels",
            "previewUrl",
            "downloadUrl",
        ]
        kind = stem["kind"]
        assert stem["previewUrl"] == f"/api/jobs/{JOB_ID}/stems/{kind}/preview"
        assert stem["downloadUrl"] == f"/api/jobs/{JOB_ID}/stems/{kind}/download"
        assert stem["fileName"] == f"stems/runs/{RUN_ID}/{kind}.wav"
    assert str(settings.data_dir) not in json.dumps(payload)


def test_model_version_time_and_warnings_propagate(settings: Settings) -> None:
    warnings = ["Synthetic warning."]
    job, _, _ = _publish(settings, warnings=warnings)
    details = load_stem_details(JOB_ID, settings, job)

    assert details.model == AUDITED_MODEL_NAME
    assert details.version == SEPARATION_VERSION
    assert details.separated_at == SEPARATED_AT
    assert details.warnings == tuple(warnings)


def test_invalid_persisted_time_falls_back_to_manifest_time(settings: Settings) -> None:
    job, _, _ = _publish(settings, separated_at="not-a-time")

    details = load_stem_details(JOB_ID, settings, job)

    assert details.separated_at == CREATED_AT


@pytest.mark.parametrize("status", ["processing", "failed"])
def test_previous_manifest_remains_available_during_retry(
    settings: Settings,
    status: str,
) -> None:
    error = "The latest retry failed." if status == "failed" else None
    job, _, _ = _publish(
        settings,
        status=status,
        separation_error=error,
    )

    details = load_stem_details(JOB_ID, settings, job)

    assert details.available is True
    assert details.status == status
    assert details.error == error
    assert len(details.stems) == 4


def test_database_traceback_or_path_is_not_returned(settings: Settings) -> None:
    raw_error = (
        "Traceback (most recent call last):\n"
        f'  File "{settings.data_dir}/private.py", line 1\n'
        "RuntimeError: failed"
    )
    job, _, _ = _publish(
        settings,
        status="failed",
        separation_error=raw_error,
    )

    payload = load_stem_details(JOB_ID, settings, job).payload()

    assert payload["error"] == "Stem separation failed."
    assert str(settings.data_dir) not in json.dumps(payload)


@pytest.mark.parametrize(
    "pointer",
    [None, "", "stems/other.json", Path(STEM_MANIFEST_RELATIVE_PATH)],
)
def test_missing_or_noncanonical_pointer_is_unavailable(
    settings: Settings,
    pointer: object,
) -> None:
    job = {
        "separation_status": "failed",
        "stem_manifest_file_name": pointer,
        "separation_error": "Retry is available.",
    }

    payload = load_stem_details(JOB_ID, settings, job).payload()

    assert payload == {
        "available": False,
        "status": "failed",
        "model": None,
        "version": None,
        "separatedAt": None,
        "warnings": [],
        "stems": [],
        "error": "Retry is available.",
    }


def test_noncanonical_pointer_cannot_resolve_an_artifact(settings: Settings) -> None:
    job, _, _ = _publish(settings)
    job["stem_manifest_file_name"] = "stems/other.json"

    with pytest.raises(StemManifestUnavailableError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_missing_manifest_is_unavailable(settings: Settings) -> None:
    (settings.exports_dir / JOB_ID).mkdir()
    job = {
        "separation_status": "completed",
        "stem_manifest_file_name": STEM_MANIFEST_RELATIVE_PATH,
    }

    details = load_stem_details(JOB_ID, settings, job)

    assert details.available is False
    with pytest.raises(StemManifestUnavailableError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


@pytest.mark.parametrize(
    "replacement",
    [
        "{not-json",
        json.dumps({"schemaVersion": 999}),
    ],
)
def test_corrupt_or_unsupported_manifest_fails_without_path_leakage(
    settings: Settings,
    replacement: str,
) -> None:
    job, job_dir, _ = _publish(settings)
    (job_dir / STEM_MANIFEST_RELATIVE_PATH).write_text(
        replacement,
        encoding="utf-8",
    )

    with pytest.raises(StemArtifactError) as captured:
        load_stem_details(JOB_ID, settings, job)

    assert str(settings.data_dir) not in str(captured.value)


def test_missing_wav_fails_without_fabricating_stems(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)
    (job_dir / "stems" / "runs" / RUN_ID / "vocals.wav").unlink()

    with pytest.raises(StemArtifactError):
        load_stem_details(JOB_ID, settings, job)


def test_metadata_change_after_publication_is_rejected(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)
    vocals = job_dir / "stems" / "runs" / RUN_ID / "vocals.wav"
    _write_wav(vocals, frames=882)

    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_valid_artifact_resolution_and_safe_download_name(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)

    artifact = resolve_stem_artifact(JOB_ID, "vocals", settings, job)

    assert artifact.kind == "vocals"
    assert artifact.label == "Vocals"
    assert artifact.path == (
        job_dir / "stems" / "runs" / RUN_ID / "vocals.wav"
    ).resolve()
    assert artifact.file_name == f"stems/runs/{RUN_ID}/vocals.wav"
    assert artifact.download_name == "vocals.wav"
    assert artifact.media_type == "audio/wav"
    assert artifact.size_bytes == artifact.path.stat().st_size


@pytest.mark.parametrize(
    "kind",
    [
        "",
        "/vocals",
        "vocals/other",
        r"vocals\other",
        ".",
        "..",
        "../vocals",
        "Vocals",
        "VOCALS",
        "vocals ",
        " vocals",
        "vocals\t",
        "vocals\x00",
        "%2e%2e",
        "%2Fvocals",
        "voc%61ls",
    ],
)
def test_unsafe_kind_forms_are_rejected(
    settings: Settings,
    kind: str,
) -> None:
    job, _, _ = _publish(settings)

    with pytest.raises(StemKindNotFoundError):
        resolve_stem_artifact(JOB_ID, kind, settings, job)


def test_unknown_safe_kind_cannot_select_extra_job_file(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)
    secret = job_dir / "stems" / "runs" / RUN_ID / "guitar.wav"
    _write_wav(secret)

    with pytest.raises(StemKindNotFoundError):
        resolve_stem_artifact(JOB_ID, "guitar", settings, job)

    assert secret.is_file()


def test_symlinked_stem_file_is_rejected(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)
    stem = job_dir / "stems" / "runs" / RUN_ID / "vocals.wav"
    target = job_dir / "replacement.wav"
    _write_wav(target)
    stem.unlink()
    try:
        stem.symlink_to(target)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_symlinked_run_directory_is_rejected(settings: Settings) -> None:
    job, job_dir, _ = _publish(settings)
    runs = job_dir / "stems" / "runs"
    run = runs / RUN_ID
    target = runs / ("c" * 32)
    run.rename(target)
    try:
        run.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_symlinked_stems_directory_is_rejected_at_resolution(
    settings: Settings,
) -> None:
    job, job_dir, _ = _publish(settings)
    stems = job_dir / "stems"
    target = job_dir / "stored-stems"
    stems.rename(target)
    try:
        stems.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")

    assert load_stem_details(JOB_ID, settings, job).available is True
    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_file_replacement_between_details_and_resolution_is_rechecked(
    settings: Settings,
) -> None:
    job, job_dir, _ = _publish(settings)
    details = load_stem_details(JOB_ID, settings, job)
    assert details.available is True

    vocals = job_dir / "stems" / "runs" / RUN_ID / "vocals.wav"
    replacement = job_dir / "replacement.wav"
    _write_wav(replacement, frames=1323)
    os.replace(replacement, vocals)

    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_filename_kind_mismatch_is_rejected(settings: Settings) -> None:
    job, job_dir, manifest = _publish(settings)
    stems = manifest["stems"]
    assert isinstance(stems, list)
    first = stems[0]
    assert isinstance(first, dict)
    first["fileName"] = f"stems/runs/{RUN_ID}/bass.wav"
    (job_dir / STEM_MANIFEST_RELATIVE_PATH).write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )

    with pytest.raises(StemArtifactError):
        resolve_stem_artifact(JOB_ID, "vocals", settings, job)


def test_bad_job_id_fails_without_echoing_input(settings: Settings) -> None:
    bad_job_id = "../../private"
    with pytest.raises(StemArtifactError) as captured:
        load_stem_details(
            bad_job_id,
            settings,
            {"stem_manifest_file_name": None},
        )

    assert bad_job_id not in str(captured.value)
    assert str(settings.data_dir) not in str(captured.value)


def test_module_has_no_fastapi_dependency() -> None:
    source = (
        Path(__file__).parents[1] / "app" / "separation_artifacts.py"
    ).read_text(encoding="utf-8")

    assert "fastapi" not in source.lower()
