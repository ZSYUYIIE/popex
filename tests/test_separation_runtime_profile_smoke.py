from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.separation_runtime import WorkerCommandError, WorkerErrorDetail


ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "demucs-runtime-client-smoke.yml"
SCRIPT = ROOT / "scripts" / "validate_separation_runtime_profile.py"


def _workflow_text() -> str:
    assert WORKFLOW.is_file(), "the dedicated runtime-client smoke workflow is missing"
    return WORKFLOW.read_text(encoding="utf-8")


def _script_text() -> str:
    assert SCRIPT.is_file()
    return SCRIPT.read_text(encoding="utf-8")


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("runtime_profile_validator", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_workflow_targets_linux_and_windows_python_313_profiles() -> None:
    text = _workflow_text()
    assert "runs-on: ubuntu-latest" in text
    assert "runs-on: windows-latest" in text
    assert text.count('python-version: "3.13"') == 2
    assert "linux-x86_64-cpu-cpython313" in text
    assert "windows-x86_64-cpu-cpython313" in text


def test_workflow_has_path_focused_triggers_and_dispatch() -> None:
    text = _workflow_text()
    assert "workflow_dispatch:" in text
    for path in (
        ".github/workflows/demucs-runtime-client-smoke.yml",
        "scripts/validate_separation_runtime_profile.py",
        "tests/test_separation_runtime_profile_smoke.py",
        "app/separation_runtime.py",
        "runtimes/demucs_worker/**",
        "runtimes/profiles/linux-cpu/**",
        "runtimes/profiles/windows-cpu/**",
        "scripts/install_demucs_linux_cpu.sh",
        "scripts/install_demucs_windows_cpu.ps1",
    ):
        assert path in text


def test_workflow_permissions_are_read_only() -> None:
    text = _workflow_text()
    permissions = text.split("permissions:", 1)[1].split("jobs:", 1)[0]
    assert "contents: read" in permissions
    assert "write" not in permissions


def test_workflow_invokes_existing_installers_before_base_validator() -> None:
    text = _workflow_text()
    linux_install = text.index("bash scripts/install_demucs_linux_cpu.sh")
    linux_validate = text.index(
        "python scripts/validate_separation_runtime_profile.py",
        linux_install,
    )
    windows_install = text.index("scripts/install_demucs_windows_cpu.ps1")
    windows_validate = text.index(
        "python scripts/validate_separation_runtime_profile.py",
        windows_install,
    )
    assert linux_install < linux_validate
    assert windows_install < windows_validate


def test_workflow_supplies_explicit_worker_lock_cache_and_profile() -> None:
    text = _workflow_text()
    assert text.count("--runtime-lock") == 2
    assert text.count("--worker") == 2
    assert text.count("--cache-root") == 2
    assert text.count("--expected-profile") == 2


def test_workflow_never_runs_model_or_inference_commands() -> None:
    text = _workflow_text().lower()
    for command in ("prepare-model", "verify-model", " separate"):
        assert command not in text
    assert "huggingface.co" not in text
    assert "hf.co" not in text


def test_workflow_cleans_both_platforms_with_always_and_uploads_no_artifacts() -> None:
    text = _workflow_text()
    assert text.count("if: always()") == 2
    assert "Clean temporary Linux runtime and cache" in text
    assert "Clean temporary Windows runtime and cache" in text
    assert "upload-artifact" not in text
    assert "actions/upload" not in text


def test_workflow_sets_hub_privacy_and_offline_variables() -> None:
    text = _workflow_text()
    for name in (
        "HF_HUB_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN",
        "HF_HUB_DISABLE_UPDATE_CHECK",
        "HF_HUB_DISABLE_PROGRESS_BARS",
    ):
        assert name in text


def test_workflow_independently_scans_model_and_readiness_assets() -> None:
    text = _workflow_text()
    for marker in (
        "955717e8.safetensors",
        "*.safetensors",
        "*.th",
        "*.ckpt",
        "htdemucs-bf35a81b-v1.json",
    ):
        assert marker in text
    assert "model cache is not empty" in text.lower()


def test_script_imports_only_standard_library_and_base_runtime_module() -> None:
    tree = ast.parse(_script_text())
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "app.separation_runtime" in imports
    forbidden = {
        "demucs",
        "torch",
        "huggingface_hub",
        "safetensors",
        "yaml",
        "fastapi",
        "popex_demucs_worker",
    }
    assert not {name.split(".", 1)[0] for name in imports} & forbidden


def test_script_calls_only_runtime_and_model_probes() -> None:
    source = _script_text()
    assert "client.runtime_probe()" in source
    assert "client.model_probe()" in source
    for forbidden in (
        "client.prepare_model(",
        "client.verify_model(",
        "client.separate(",
        "return client(",
    ):
        assert forbidden not in source


def test_script_requires_download_required_broad_worker_code_and_exit_20() -> None:
    source = _script_text()
    assert 'exc.code != "MODEL_DOWNLOAD_REQUIRED"' in source
    assert 'exc.detail.worker_code != "MODEL_DOWNLOAD_REQUIRED"' in source
    assert "_MODEL_DOWNLOAD_EXIT_CODE = 20" in source


def test_script_scans_checkpoint_and_readiness_names() -> None:
    source = _script_text()
    assert '"htdemucs-bf35a81b-v1.json"' in source
    for suffix in (".safetensors", ".th", ".ckpt"):
        assert f'"{suffix}"' in source


def test_script_success_output_has_only_safe_path_free_fields(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_validator_module()

    runtime = SimpleNamespace(
        runtime_profile="linux-x86_64-cpu-cpython313",
        worker_version="1.0.0",
        runtime_lock_source="profile",
        demucs_version="4.1.0",
        torch_version="2.13.0+cpu",
    )

    class FakeClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def runtime_probe(self):
            return runtime

        def model_probe(self):
            raise WorkerCommandError(
                WorkerErrorDetail(
                    code="MODEL_DOWNLOAD_REQUIRED",
                    worker_code="MODEL_DOWNLOAD_REQUIRED",
                    message="The verified model is not available.",
                    retryable=True,
                    exit_code=20,
                )
            )

    monkeypatch.setattr(module, "SeparationRuntimeClient", FakeClient)
    cache = tmp_path / "cache"
    cache.mkdir()
    result = module.validate_profile(
        worker=tmp_path / "worker",
        runtime_lock=tmp_path / "runtime-lock.json",
        cache_root=cache,
        expected_profile="linux-x86_64-cpu-cpython313",
    )
    assert set(result) == {
        "schemaVersion",
        "runtimeProfile",
        "workerVersion",
        "demucsVersion",
        "torchVersion",
        "modelState",
        "modelAssetsCreated",
    }
    encoded = json.dumps(result, allow_nan=False)
    assert str(tmp_path) not in encoded
    assert not any("path" in key.lower() for key in result)


def test_script_rejects_relative_or_nonempty_cache_paths(tmp_path: Path) -> None:
    module = _load_validator_module()
    with pytest.raises(module.ProfileValidationError, match="must be absolute"):
        module._trusted_absolute_path("relative", "worker", kind="file")

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "unexpected").write_text("x", encoding="utf-8")
    with pytest.raises(module.ProfileValidationError, match="must be empty"):
        module.validate_profile(
            worker=tmp_path / "worker",
            runtime_lock=tmp_path / "lock",
            cache_root=cache,
            expected_profile="linux-x86_64-cpu-cpython313",
        )
