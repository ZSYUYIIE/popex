from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "demucs-windows-real-model-e2e.yml"
VALIDATOR = ROOT / "scripts" / "validate_demucs_windows_real_model_e2e.py"
DOC = ROOT / "docs" / "runtime" / "demucs-windows-real-model-e2e.md"

_SPEC = importlib.util.spec_from_file_location(
    "popex_demucs_windows_real_model_e2e_validator",
    VALIDATOR,
)
assert _SPEC is not None and _SPEC.loader is not None
validator = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = validator
_SPEC.loader.exec_module(validator)


def _workflow() -> dict:
    payload = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_workflow_is_manual_only_and_read_only() -> None:
    payload = _workflow()
    triggers = payload.get("on", payload.get(True))
    assert isinstance(triggers, dict)
    assert set(triggers) == {"workflow_dispatch"}
    assert payload["permissions"] == {"contents": "read"}
    text = WORKFLOW.read_text(encoding="utf-8").lower()
    for forbidden in ("pull_request:", "push:", "schedule:", "repository_dispatch:"):
        assert forbidden not in text


def test_workflow_uses_exact_windows_profile_and_repository_python() -> None:
    payload = _workflow()
    job = payload["jobs"]["windows-real-model-e2e"]
    assert job["runs-on"] == "windows-latest"
    assert 30 <= job["timeout-minutes"] <= 120
    serialized = json.dumps(job)
    assert '"python-version": "3.13"' in serialized
    assert "scripts/install_demucs_windows_cpu.ps1" in serialized
    assert "python scripts/validate_demucs_windows_real_model_e2e.py" in serialized
    assert "venv\\Scripts\\python.exe scripts/validate" not in serialized


def test_workflow_enforces_privacy_cleanup_and_no_artifact_upload() -> None:
    payload = _workflow()
    job = payload["jobs"]["windows-real-model-e2e"]
    text = WORKFLOW.read_text(encoding="utf-8")
    lower = text.lower()
    assert job["env"]["HF_HUB_DISABLE_TELEMETRY"] == "1"
    assert job["env"]["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"
    assert job["env"]["HF_HUB_DISABLE_PROGRESS_BARS"] == "1"
    assert "allowmodeldownload" not in lower
    assert "upload-artifact" not in lower
    assert "actions/upload" not in lower
    assert "hf_hub_download" not in lower
    cleanup = job["steps"][-1]
    assert cleanup["if"] == "always()"
    assert "POPEX_REAL_MODEL_RUNTIME" in cleanup["run"]
    assert "POPEX_REAL_MODEL_CACHE" in cleanup["run"]
    assert "POPEX_REAL_MODEL_DATA" in cleanup["run"]
    assert "Remove-Item" in cleanup["run"]


def test_validator_uses_real_fastapi_consent_path_without_mocks() -> None:
    source = VALIDATOR.read_text(encoding="utf-8")
    lower = source.lower()
    assert "TestClient" in source
    assert "create_app(settings=settings)" in source
    assert 'json={"allowModelDownload": True}' in source
    assert "SeparationRuntimeClient" in source
    assert ".prepare_model(" not in source
    assert "mock" not in lower
    assert "monkeypatch" not in lower
    assert "hf_hub_download" not in lower
    assert "requests.get" not in lower
    assert "urllib.request" not in lower
    assert "AUDITED_MODEL_REPOSITORY" in source
    assert "AUDITED_MODEL_REVISION" in source
    assert "AUDITED_CHECKPOINT_SHA256" in source


def test_validator_generates_deterministic_stereo_44100_pcm(tmp_path: Path) -> None:
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    first_meta = validator._generate_synthetic_audio(first)
    second_meta = validator._generate_synthetic_audio(second)
    assert first.read_bytes() == second.read_bytes()
    assert first_meta == second_meta
    info = sf.info(str(first))
    assert int(info.samplerate) == 44_100
    assert int(info.channels) == 2
    assert str(info.subtype).upper() == "PCM_16"
    assert first_meta["durationSeconds"] == pytest.approx(validator.AUDIO_DURATION_SECONDS)


def test_validator_refuses_relative_and_nonempty_trusted_paths(tmp_path: Path) -> None:
    with pytest.raises(validator.ValidationError, match="absolute"):
        validator._absolute_normalized("relative/path", "cache-root")
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "asset.bin").write_bytes(b"x")
    with pytest.raises(validator.ValidationError, match="new or empty"):
        validator._empty_directory(str(occupied), "cache-root")


def test_validator_refuses_missing_or_empty_worker_files(tmp_path: Path) -> None:
    missing = tmp_path / "missing.exe"
    with pytest.raises(validator.ValidationError, match="existing regular"):
        validator._existing_regular_file(str(missing), "worker")
    empty = tmp_path / "worker.exe"
    empty.write_bytes(b"")
    with pytest.raises(validator.ValidationError, match="must not be empty"):
        validator._existing_regular_file(str(empty), "worker")


def test_safe_error_output_redacts_machine_paths_and_credentials(tmp_path: Path) -> None:
    private = tmp_path / "private" / "runtime-lock.json"
    message = validator._safe_error_message(
        RuntimeError(f"token=secret failed at {private} https://example.invalid/model"),
        (private,),
    )
    lower = message.lower()
    assert str(private).lower() not in lower
    assert "secret" not in lower
    assert "https://" not in lower
    assert "traceback" not in lower


def test_cli_refuses_non_windows_without_traceback_or_path_leak(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(validator.platform, "system", lambda: "Linux")
    paths = [tmp_path / name for name in ("worker.exe", "lock.json", "cache", "data")]
    code = validator.main(
        [
            "--worker",
            str(paths[0]),
            "--runtime-lock",
            str(paths[1]),
            "--cache-root",
            str(paths[2]),
            "--data-dir",
            str(paths[3]),
        ]
    )
    captured = capsys.readouterr()
    assert code == 1
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["status"] == "error"
    assert payload["code"] == "WINDOWS_REAL_MODEL_E2E_FAILED"
    assert "traceback" not in captured.err.lower()
    for path in paths:
        assert str(path).lower() not in captured.err.lower()


def test_documentation_records_real_only_scope_and_evidence_policy() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    assert "workflow_dispatch" in text
    assert "synthetic" in text
    assert "allowmodeldownload" in text
    assert "no mocks" in text or "not replaced with mocks" in text
    assert "no artifact" in text or "not uploaded" in text
    assert "run id" in text
    assert "pending" in text or "passed" in text or "blocked" in text
