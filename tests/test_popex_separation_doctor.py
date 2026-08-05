from __future__ import annotations

import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "popex_separation_doctor.py"
SPEC = importlib.util.spec_from_file_location("popex_separation_doctor", SCRIPT)
assert SPEC and SPEC.loader
DOCTOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = DOCTOR
SPEC.loader.exec_module(DOCTOR)

PROFILE = "linux-x86_64-cpu-cpython313"


def platform_identity():
    return DOCTOR.Identity("linux", "x86_64", "3.13", PROFILE)


def make_paths(tmp_path: Path, *, cache_exists: bool = False):
    worker = tmp_path / "runtime" / "venv" / "bin" / "popex-demucs-worker"
    worker.parent.mkdir(parents=True)
    worker.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    worker.chmod(worker.stat().st_mode | stat.S_IXUSR)
    lock = tmp_path / "runtime" / "runtime-lock.json"
    lock.write_text(
        json.dumps(
            {
                "schemaVersion": 1,
                "runtimeProfile": PROFILE,
                "workerVersion": "1.0.0",
                "packages": {
                    "demucs": "4.1.0",
                    "torch": "2.13.0+cpu",
                    "huggingface_hub": "1.26.0",
                    "safetensors": "0.8.0",
                    "PyYAML": "6.0.3",
                },
            }
        ),
        encoding="utf-8",
    )
    cache = tmp_path / "cache"
    if cache_exists:
        cache.mkdir()
    return worker.resolve(), lock.resolve(), cache.resolve()


def config(tmp_path: Path, *, cache_exists: bool = False, temp_parent=None):
    worker, lock, cache = make_paths(tmp_path, cache_exists=cache_exists)
    return DOCTOR.Config(
        worker=worker,
        runtime_lock=lock,
        cache_root=cache,
        expected_profile=PROFILE,
        temp_parent=temp_parent,
    )


class FakeClient:
    def __init__(self, *, ready: bool):
        self.ready = ready
        self.calls = []

    def runtime_probe(self):
        self.calls.append("runtime-probe")
        return SimpleNamespace(
            runtime_profile=PROFILE,
            worker_version="1.0.0",
            demucs_version="4.1.0",
        )

    def model_probe(self):
        self.calls.append("model-probe")
        if not self.ready:
            error = RuntimeError("model missing")
            error.code = "MODEL_DOWNLOAD_REQUIRED"
            raise error
        return SimpleNamespace(
            runtime_profile=PROFILE,
            worker_version="1.0.0",
            demucs_version="4.1.0",
            offline_ready=True,
        )

    def prepare_model(self, **kwargs):
        self.calls.append("prepare-model")
        raise AssertionError("passive check must never prepare a model")

    def verify_model(self):
        self.calls.append("verify-model")
        raise AssertionError("passive check must never verify a model")

    def separate(self, **kwargs):
        self.calls.append("separate")
        raise AssertionError("passive check must never run separation")


def test_passive_check_reports_download_required_without_download(tmp_path: Path):
    client = FakeClient(ready=False)
    result = DOCTOR.check(
        config(tmp_path),
        client_factory=lambda _: client,
        probe=platform_identity,
    )
    assert result["state"] == "download_required"
    assert result["modelDownloadPerformed"] is False
    assert client.calls == ["runtime-probe", "model-probe"]
    assert (tmp_path / "cache").is_dir()
    assert list((tmp_path / "cache").iterdir()) == []


def test_passive_check_reports_ready_without_mutating_cache(tmp_path: Path):
    client = FakeClient(ready=True)
    value = config(tmp_path, cache_exists=True)
    sentinel = value.cache_root / "existing-cache-entry"
    sentinel.write_text("preserve", encoding="utf-8")
    result = DOCTOR.check(
        value,
        client_factory=lambda _: client,
        probe=platform_identity,
    )
    assert result["state"] == "ready"
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert client.calls == ["runtime-probe", "model-probe"]


def test_missing_worker_maps_to_runtime_missing(tmp_path: Path):
    value = config(tmp_path)
    value.worker.unlink()
    with pytest.raises(DOCTOR.DoctorError) as captured:
        DOCTOR.check(value, probe=platform_identity)
    assert captured.value.state == "runtime_missing"
    assert str(value.worker) not in str(captured.value)


def test_validate_refuses_without_consent_before_client_creation(tmp_path: Path):
    called = False

    def factory(_):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(DOCTOR.DoctorError, match="Explicit"):
        DOCTOR.validate(
            config(tmp_path),
            allowed=False,
            client_factory=factory,
            probe=platform_identity,
        )
    assert called is False


@pytest.mark.parametrize("field", ["worker", "runtime_lock", "cache_root"])
def test_relative_trusted_paths_are_rejected(tmp_path: Path, field: str):
    value = config(tmp_path)
    values = {name: getattr(value, name) for name in value.__dataclass_fields__}
    values[field] = Path("relative-path")
    with pytest.raises(DOCTOR.DoctorError):
        DOCTOR.check(
            DOCTOR.Config(**values),
            client_factory=lambda _: FakeClient(ready=True),
            probe=platform_identity,
        )


def test_symlinked_worker_is_rejected(tmp_path: Path):
    value = config(tmp_path)
    target = tmp_path / "target-worker"
    target.write_text("worker", encoding="utf-8")
    target.chmod(target.stat().st_mode | stat.S_IXUSR)
    value.worker.unlink()
    try:
        value.worker.symlink_to(target)
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(DOCTOR.DoctorError):
        DOCTOR.check(
            value,
            client_factory=lambda _: FakeClient(ready=True),
            probe=platform_identity,
        )


def test_platform_profile_mismatch_fails_before_spawn(tmp_path: Path):
    value = config(tmp_path)
    with pytest.raises(DOCTOR.DoctorError, match="does not match"):
        DOCTOR.check(
            value,
            client_factory=lambda _: (_ for _ in ()).throw(AssertionError()),
            probe=lambda: DOCTOR.Identity(
                "windows", "x86_64", "3.13", "windows-x86_64-cpu-cpython313"
            ),
        )


def test_runtime_lock_exact_package_contract(tmp_path: Path):
    value = config(tmp_path)
    payload = json.loads(value.runtime_lock.read_text(encoding="utf-8"))
    payload["packages"].pop("PyYAML")
    value.runtime_lock.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DOCTOR.DoctorError, match="package set"):
        DOCTOR.check(
            value,
            client_factory=lambda _: FakeClient(ready=True),
            probe=platform_identity,
        )


def test_safe_text_redacts_paths_tokens_urls_and_controls(tmp_path: Path):
    raw = (
        f"token=secret Bearer abc.def https://example.invalid/model "
        f"{tmp_path}/private\nC:\\private\\runtime\\lock.json"
    )
    result = DOCTOR.safe(raw, "fallback")
    assert "secret" not in result
    assert "abc.def" not in result
    assert "example.invalid" not in result
    assert str(tmp_path) not in result
    assert "C:\\private" not in result
    assert "\n" not in result


def test_validation_uses_unique_temp_data_and_preserves_existing_data(tmp_path: Path):
    value = config(tmp_path / "trusted", temp_parent=tmp_path.resolve())
    existing = tmp_path / "existing-popex-data"
    existing.mkdir()
    sentinel = existing / "popex.sqlite3"
    sentinel.write_text("do-not-touch", encoding="utf-8")
    seen = []

    def runner(_config, _client, validation_root):
        seen.append(validation_root)
        assert validation_root.parent == tmp_path.resolve()
        assert validation_root.name.startswith("popex-separation-doctor-")
        (validation_root / "synthetic-only").write_text("ok", encoding="utf-8")
        return {
            "initialState": "ready",
            "modelDownloadPerformed": False,
            "manifestSchemaVersion": 3,
            "stems": ["vocals", "bass", "drums", "other"],
            "previewsVerified": 4,
            "downloadsVerified": 4,
            "sampleRate": 44100,
            "channels": 2,
        }

    first = DOCTOR.validate(
        value,
        allowed=True,
        client_factory=lambda _: object(),
        runner=runner,
        probe=platform_identity,
    )
    second = DOCTOR.validate(
        value,
        allowed=True,
        client_factory=lambda _: object(),
        runner=runner,
        probe=platform_identity,
    )
    assert first["temporaryDataRemoved"] is True
    assert second["temporaryDataRemoved"] is True
    assert seen[0] != seen[1]
    assert not seen[0].exists() and not seen[1].exists()
    assert sentinel.read_text(encoding="utf-8") == "do-not-touch"


def test_cleanup_refuses_non_doctor_directory(tmp_path: Path):
    parent = tmp_path.resolve()
    root = parent / "existing-user-data"
    root.mkdir()
    sentinel = root / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    assert DOCTOR.cleanup(root, parent) is False
    assert sentinel.is_file()


def test_synthetic_wav_is_deterministic_stereo_44100(tmp_path: Path):
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    DOCTOR.synthetic(first)
    DOCTOR.synthetic(second)
    assert first.read_bytes() == second.read_bytes()
    import wave

    with wave.open(str(first), "rb") as source:
        assert source.getnchannels() == 2
        assert source.getframerate() == 44100
        assert source.getsampwidth() == 2
        assert source.getnframes() == 44100


def test_privacy_environment_removes_credentials(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("UNRELATED_VALUE", "preserve")
    DOCTOR.privacy(passive=True)
    assert "HF_TOKEN" not in os.environ
    assert "AWS_SECRET_ACCESS_KEY" not in os.environ
    assert os.environ["UNRELATED_VALUE"] == "preserve"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] == "1"


def test_failure_payload_is_path_free(tmp_path: Path):
    error = DOCTOR.DoctorError(
        "unavailable",
        f"token=secret failed at {tmp_path}/private/runtime-lock.json",
    )
    encoded = json.dumps(DOCTOR.failure("check", error))
    assert "secret" not in encoded
    assert str(tmp_path) not in encoded
    assert "runtime-lock.json" not in encoded


def test_top_level_imports_are_standard_library_only():
    import ast
    import sys as _sys

    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= set(_sys.stdlib_module_names)
    assert "app" not in imported and "fastapi" not in imported


def test_main_refuses_validation_before_path_checks(capsys):
    code = DOCTOR.main([
        "validate",
        "--worker", "/missing/worker",
        "--runtime-lock", "/missing/runtime-lock.json",
        "--cache-root", "/missing/cache",
        "--expected-profile", PROFILE,
    ])
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 2
    assert payload["state"] == "unavailable"
    assert payload["modelDownloadPerformed"] is False
    assert "/missing" not in captured.out
    assert "/missing" not in captured.err


def test_wrapper_and_guide_contracts():
    guide = (ROOT / "docs/runtime/local-separation-validation.md").read_text(encoding="utf-8")
    linux = (ROOT / "docs/runtime/local-separation-validation-linux.sh").read_text(encoding="utf-8")
    windows = (ROOT / "docs/runtime/local-separation-validation-windows.ps1").read_text(encoding="utf-8")

    assert "check" in linux and "validate --allow-model-download" in linux
    assert "-AllowModelDownload" in windows and '"check"' in windows
    assert "find " not in linux and "Get-ChildItem" not in windows
    assert "prepare-model" not in linux and "prepare-model" not in windows
    assert "single PopEx server process" in guide
    assert "84,025,440" in guide
    assert "Source audio stays on the device" in guide
    assert "popex-separation-doctor-" in guide
    assert "existing PopEx data" in guide
    assert ".mp3" not in linux and ".mp3" not in windows
