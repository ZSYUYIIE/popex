from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "demucs-linux-real-model-e2e.yml"
SCRIPT = ROOT / "scripts" / "validate_demucs_linux_real_model_e2e.py"
DOC = ROOT / "docs" / "runtime" / "demucs-linux-real-model-e2e.md"


def load_validator():
    spec = importlib.util.spec_from_file_location("linux_real_model_e2e", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_workflow_is_manual_only_read_only_and_bounded():
    text = WORKFLOW.read_text(encoding="utf-8")
    trigger = text.split("permissions:", 1)[0]
    assert "workflow_dispatch:" in trigger
    for forbidden in ("pull_request:", "push:", "schedule:", "workflow_run:"):
        assert forbidden not in trigger
    assert "contents: read" in text
    assert "timeout-minutes: 120" in text
    assert "cancel-in-progress: false" in text
    assert "actions/upload-artifact" not in text


def test_workflow_uses_exact_profile_explicit_consent_path_and_base_python():
    text = WORKFLOW.read_text(encoding="utf-8")
    required = (
        "python-version: \"3.13\"",
        "python -m pip install -e '.[dev]'",
        "bash scripts/install_demucs_linux_cpu.sh",
        "python scripts/validate_demucs_linux_real_model_e2e.py",
        "--expected-profile linux-x86_64-cpu-cpython313",
        "--timeout-seconds 3600",
        "allowModelDownload",
        "bf35a81b663819a8255c8fefee17f9d812b786b5",
        "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
        "84025440",
        "vocals",
        "bass",
        "drums",
        "other",
    )
    combined = text + SCRIPT.read_text(encoding="utf-8")
    for value in required:
        assert value in combined
    assert "$POPEX_E2E_RUNTIME/venv/bin/python" not in text


def test_workflow_sets_privacy_guards_and_cleans_every_sensitive_path():
    text = WORKFLOW.read_text(encoding="utf-8")
    for name in (
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_UPDATE_CHECK",
        "HF_HUB_DISABLE_PROGRESS_BARS",
        "HF_TOKEN",
        "HUGGING_FACE_HUB_TOKEN",
    ):
        assert name in text
    cleanup = text.split("if: always()", 1)[1]
    for name in (
        "POPEX_E2E_RUNTIME",
        "POPEX_E2E_CACHE",
        "POPEX_E2E_DATA",
        "POPEX_E2E_SUMMARY",
    ):
        assert name in cleanup
    assert "rm -rf" in cleanup


def test_workflow_never_uploads_or_commits_real_model_outputs():
    text = WORKFLOW.read_text(encoding="utf-8").lower()
    assert "upload-artifact" not in text
    assert "*.th" in text and "*.ckpt" in text
    assert "source-synthetic.wav" in text
    assert "stem-separation.json" in text


def test_validator_constants_are_exact_and_no_custom_model_arguments_exist():
    module = load_validator()
    assert module.EXPECTED_PROFILE == "linux-x86_64-cpu-cpython313"
    assert module.EXPECTED_PROTOCOL_VERSION == 1
    assert module.EXPECTED_WORKER_VERSION == "1.0.0"
    assert module.EXPECTED_DEMUCS_VERSION == "4.1.0"
    assert module.EXPECTED_TORCH_VERSION == "2.13.0+cpu"
    assert module.EXPECTED_MODEL_REPOSITORY == "adefossez/HTDemucs"
    assert module.EXPECTED_MODEL_REVISION == "bf35a81b663819a8255c8fefee17f9d812b786b5"
    assert module.EXPECTED_CHECKPOINT_FILE == "955717e8.safetensors"
    assert module.EXPECTED_CHECKPOINT_SIZE_BYTES == 84_025_440
    assert module.EXPECTED_CHECKPOINT_SHA256 == "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd"
    options = {action.dest for action in module.build_parser()._actions}
    assert not options.intersection(
        {"model", "repository", "revision", "checkpoint", "checkpoint_sha256"}
    )


def test_synthetic_wav_is_deterministic_stereo_pcm(tmp_path: Path):
    module = load_validator()
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first_metadata = module.generate_synthetic_wav(first)
    second_metadata = module.generate_synthetic_wav(second)
    assert first.read_bytes() == second.read_bytes()
    assert first_metadata == second_metadata
    info = sf.info(str(first))
    audio, sample_rate = sf.read(str(first), always_2d=True, dtype="float32")
    assert info.format == "WAV"
    assert info.subtype == "PCM_16"
    assert sample_rate == 44_100
    assert audio.shape == (176_400, 2)
    assert np.isfinite(audio).all()
    assert not np.array_equal(audio[:, 0], audio[:, 1])
    assert float(np.max(np.abs(audio))) > 0.5


def make_args(tmp_path: Path, **overrides):
    runtime = tmp_path / "runtime"
    worker = runtime / "venv" / "bin" / "popex-demucs-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker.chmod(0o755)
    lock = runtime / "runtime-lock.json"
    lock.write_text("{}\n", encoding="utf-8")
    values = {
        "worker": str(worker.resolve()),
        "runtime_lock": str(lock.resolve()),
        "cache_root": str((tmp_path / "cache").resolve()),
        "data_dir": str((tmp_path / "data").resolve()),
        "expected_profile": "linux-x86_64-cpu-cpython313",
        "timeout_seconds": 3600,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_trusted_input_validation_accepts_only_isolated_absolute_roots(tmp_path: Path, monkeypatch):
    module = load_validator()
    monkeypatch.setattr(module, "_supported_platform", lambda: True)
    paths = module.validate_trusted_inputs(make_args(tmp_path))
    assert paths.cache_root == (tmp_path / "cache").resolve()
    assert paths.data_dir == (tmp_path / "data").resolve()

    relative = make_args(tmp_path / "relative", cache_root="cache")
    with pytest.raises(module.E2EValidationError) as caught:
        module.validate_trusted_inputs(relative)
    assert caught.value.code == "INVALID_TRUSTED_PATH"

    overlap_base = tmp_path / "overlap"
    (overlap_base / "cache").mkdir(parents=True)
    overlap_args = make_args(
        overlap_base,
        data_dir=str((overlap_base / "cache" / "data").resolve()),
    )
    with pytest.raises(module.E2EValidationError) as caught:
        module.validate_trusted_inputs(overlap_args)
    assert caught.value.code == "TRUSTED_ROOTS_OVERLAP"


def test_trusted_input_validation_rejects_symlink_and_nonempty_cache(tmp_path: Path, monkeypatch):
    module = load_validator()
    monkeypatch.setattr(module, "_supported_platform", lambda: True)
    target = tmp_path / "actual-cache"
    target.mkdir()
    link = tmp_path / "cache-link"
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(module.E2EValidationError):
        module.validate_trusted_inputs(make_args(tmp_path / "symlink", cache_root=str(link)))

    base = tmp_path / "nonempty"
    args = make_args(base)
    cache = Path(args.cache_root)
    cache.mkdir()
    (cache / "sentinel").write_text("keep", encoding="utf-8")
    with pytest.raises(module.E2EValidationError) as caught:
        module.validate_trusted_inputs(args)
    assert caught.value.code == "TRUSTED_ROOT_NOT_EMPTY"


def test_safe_summary_has_exact_path_free_shape():
    module = load_validator()
    summary = module.build_safe_summary(
        runtime_profile=module.EXPECTED_PROFILE,
        worker_version=module.EXPECTED_WORKER_VERSION,
        demucs_version=module.EXPECTED_DEMUCS_VERSION,
        torch_version=module.EXPECTED_TORCH_VERSION,
        model_revision=module.EXPECTED_MODEL_REVISION,
        checkpoint_size_bytes=module.EXPECTED_CHECKPOINT_SIZE_BYTES,
        checkpoint_sha256=module.EXPECTED_CHECKPOINT_SHA256,
        job_status="completed",
        stem_metrics=[
            {"kind": kind, "durationSeconds": 4.0, "sizeBytes": 1234 + index}
            for index, kind in enumerate(module.EXPECTED_STEM_KINDS)
        ],
        preparation_seconds=12.3456,
        inference_seconds=23.4567,
        peak_rss_mib=512.25,
    )
    serialized = json.dumps(summary, sort_keys=True)
    assert set(summary) == module._SAFE_SUMMARY_KEYS
    assert "/" not in serialized
    assert "\\" not in serialized
    assert "http" not in serialized.lower()
    assert summary["elapsedPreparationSeconds"] == 12.346
    assert summary["elapsedInferenceSeconds"] == 23.457
    assert summary["peakProcessRssMiB"] == 512.2


def test_main_refusal_emits_only_classified_safe_json(tmp_path: Path, capsys, monkeypatch):
    module = load_validator()
    monkeypatch.setattr(module, "_supported_platform", lambda: True)
    args = make_args(tmp_path)
    code = module.main(
        [
            "--worker",
            args.worker,
            "--runtime-lock",
            args.runtime_lock,
            "--cache-root",
            "relative-cache",
            "--data-dir",
            args.data_dir,
            "--expected-profile",
            args.expected_profile,
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.err)
    assert code == 2
    assert captured.out == ""
    assert set(payload) <= module._SAFE_ERROR_KEYS
    assert payload["status"] == "error"
    assert payload["code"] == "INVALID_TRUSTED_PATH"
    assert str(tmp_path) not in captured.err


def test_documentation_records_safety_contract_and_evidence_slot():
    text = DOC.read_text(encoding="utf-8")
    for value in (
        "workflow_dispatch",
        "synthetic",
        "allowModelDownload",
        "linux-x86_64-cpu-cpython313",
        "955717e8.safetensors",
        "84025440",
        "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
        "schema 3",
        "vocals",
        "bass",
        "drums",
        "other",
        "cleanup",
        "Final workflow evidence",
    ):
        assert value in text
