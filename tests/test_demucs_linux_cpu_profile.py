from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "runtimes" / "profiles" / "linux-cpu"
PROFILE = json.loads((PROFILE_DIR / "profile.json").read_text(encoding="utf-8"))
ARTIFACTS = json.loads((PROFILE_DIR / "artifacts.json").read_text(encoding="utf-8"))
WORKER_LOCK = json.loads((PROFILE_DIR / "worker-runtime-lock.json").read_text(encoding="utf-8"))
REQ_IN = (PROFILE_DIR / "requirements.in").read_text(encoding="utf-8")
REQ_LOCK = (PROFILE_DIR / "requirements.lock").read_text(encoding="utf-8")
TORCH_LOCK = (PROFILE_DIR / "torch.lock").read_text(encoding="utf-8")
INSTALLER = (ROOT / "scripts" / "install_demucs_linux_cpu.sh").read_text(encoding="utf-8")
INSTALL_DOC = (PROFILE_DIR / "INSTALL.md").read_text(encoding="utf-8")
RUNTIME_DOC = (ROOT / "docs" / "runtime" / "demucs-linux-cpu.md").read_text(encoding="utf-8")
INVENTORY = (PROFILE_DIR / "third-party-inventory.md").read_text(encoding="utf-8")


def _pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for match in re.finditer(r"(?mi)^([A-Za-z0-9_.-]+)==([^ \\\n]+)", text):
        pins[match.group(1).lower().replace("_", "-")] = match.group(2)
    return pins


def test_01_profile_schema() -> None:
    assert PROFILE["schemaVersion"] == 1
    assert PROFILE["runtimeProfile"] == "linux-x86_64-cpu-cpython313"
    assert PROFILE["supportedOS"] == ["Linux"]
    assert PROFILE["architecture"] == "x86_64"
    assert PROFILE["pythonImplementation"] == "CPython"
    assert PROFILE["pythonVersionRange"] == ">=3.13.0,<3.14.0"
    assert PROFILE["device"] == "cpu"


def test_02_exact_demucs_pin() -> None:
    assert PROFILE["demucsVersion"] == "4.1.0"
    assert _pins(REQ_IN)["demucs"] == "4.1.0"
    assert _pins(REQ_LOCK)["demucs"] == "4.1.0"
    assert WORKER_LOCK["packages"]["demucs"] == "4.1.0"


def test_03_exact_pytorch_cpu_pin() -> None:
    assert PROFILE["pytorchVersionBuild"] == "2.13.0+cpu"
    assert _pins(TORCH_LOCK) == {"torch": "2.13.0+cpu"}
    assert "3fbf9c9d1f3c10c2d59d04aca426dee9ccc6ceb32d255c61e93acc3b4f75fae6" in TORCH_LOCK


def test_04_official_cpu_index() -> None:
    assert PROFILE["packageIndexUrls"]["pytorchCpu"] == "https://download.pytorch.org/whl/cpu"
    assert "--index-url https://download.pytorch.org/whl/cpu" in INSTALLER
    torch_artifact = next(item for item in ARTIFACTS["artifacts"] if item["name"] == "torch")
    assert torch_artifact["indexUrl"] == "https://download.pytorch.org/whl/cpu"


def test_05_no_cuda_or_nvidia_packages() -> None:
    installed = {name.lower() for name in PROFILE["exactDependencyVersions"]}
    assert not any(name.startswith("nvidia-") for name in installed)
    assert "cuda" not in installed
    assert "torch==2.13.0+cpu" in TORCH_LOCK


def test_06_no_torchaudio() -> None:
    assert "torchaudio" not in _pins(REQ_LOCK)
    assert "torchaudio" not in _pins(TORCH_LOCK)
    assert PROFILE["excludedPackages"]["torchaudio"]["installed"] is False


def test_07_no_training_extras() -> None:
    lock_text = REQ_LOCK + TORCH_LOCK
    assert "demucs[train]" not in lock_text.lower()
    assert "dora-search" not in _pins(lock_text)
    assert "openunmix" not in _pins(lock_text)
    assert PROFILE["excludedPackages"]["dora-search"]["installed"] is False


def test_08_no_models_or_model_urls() -> None:
    install_inputs = REQ_LOCK + TORCH_LOCK + INSTALLER
    assert "huggingface.co/adefossez" not in install_inputs
    assert "955717e8.safetensors" not in install_inputs
    assert PROFILE["modelBundled"] is False
    assert PROFILE["modelDownloadDuringInstallation"] is False


def test_09_no_startup_download() -> None:
    assert PROFILE["startupDownload"] is False
    assert "prepare-model" not in INSTALLER
    assert "model-probe" not in INSTALLER


def test_10_installer_platform_refusal() -> None:
    assert '$(uname -s)' in INSTALLER and '!= "Linux"' in INSTALLER
    assert '$(uname -m)' in INSTALLER
    assert "x86_64|amd64" in INSTALLER
    assert "supports Linux x86-64 only" in INSTALLER


def test_11_installer_python_refusal() -> None:
    assert 'platform.python_implementation() == "CPython"' in INSTALLER
    assert "sys.version_info[:2] == (3, 13)" in INSTALLER
    assert "requires CPython >=3.13.0,<3.14.0" in INSTALLER


def test_12_isolated_venv_behavior() -> None:
    assert '"$PYTHON_BIN" -m venv "$VENV_DIR"' in INSTALLER
    assert 'VENV_PYTHON="$VENV_DIR/bin/python"' in INSTALLER
    assert "POPEX_DEMUCS_LINUX_CPU_DIR" in INSTALLER
    assert "pip install -e '.[dev]'" not in INSTALLER


def test_13_strict_shell_mode() -> None:
    assert INSTALLER.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "IFS=$'\\n\\t'" in INSTALLER
    assert "trap on_exit EXIT" in INSTALLER
    assert "Refusing to install the optional runtime as root" in INSTALLER


def test_14_local_worker_installation_ordering() -> None:
    worker_check = INSTALLER.index('[[ ! -f "$WORKER_DIR/pyproject.toml" ]]')
    create_venv = INSTALLER.index('-m venv "$VENV_DIR"')
    requirements_install = INSTALLER.index('-r "$PROFILE_DIR/requirements.lock"')
    worker_install = INSTALLER.index('"$WORKER_DIR"', requirements_install)
    assert worker_check < create_venv < requirements_install < worker_install
    assert "--no-build-isolation" in INSTALLER and "--no-index" in INSTALLER


def test_15_runtime_probe_only() -> None:
    assert "--protocol-version 1 runtime-probe" in INSTALLER
    final_command = [line for line in INSTALLER.splitlines() if line and not line.startswith("#")][-1]
    assert final_command.endswith("--protocol-version 1 runtime-probe")
    for forbidden in ("prepare-model", "verify-model", "model-probe", " separate"):
        assert forbidden not in INSTALLER


def test_16_removal_instructions() -> None:
    assert "rm -rf" in INSTALL_DOC
    assert "rm -rf" in RUNTIME_DOC
    assert "incomplete isolated runtime" in INSTALLER


def test_17_lock_and_profile_consistency() -> None:
    lock_pins = _pins(REQ_LOCK) | _pins(TORCH_LOCK)
    expected = {
        name.lower().replace("_", "-"): version
        for name, version in PROFILE["exactDependencyVersions"].items()
        if name != "popex-demucs-worker"
    }
    assert lock_pins == expected
    assert all("--hash=sha256:" in block for block in re.split(r"(?m)(?=^[A-Za-z0-9_.-]+==)", REQ_LOCK)[1:])
    assert PROFILE["hashesEnforced"] is True


def test_18_inventory_and_artifacts_are_complete() -> None:
    expected = {
        name.lower().replace("_", "-")
        for name in PROFILE["exactDependencyVersions"]
        if name != "popex-demucs-worker"
    }
    artifact_names = {item["name"].lower().replace("_", "-") for item in ARTIFACTS["artifacts"]}
    assert artifact_names == expected
    for name, version in PROFILE["exactDependencyVersions"].items():
        assert name in INVENTORY or name == "popex-demucs-worker"
        assert version in INVENTORY or name == "popex-demucs-worker"
    for excluded in ("torchaudio", "openunmix", "dora-search", "NVIDIA/CUDA"):
        assert excluded.lower() in INVENTORY.lower()
    assert ARTIFACTS["localPackage"] == {
        "name": "popex-demucs-worker",
        "version": "1.0.0",
        "sourcePath": "runtimes/demucs_worker",
        "indexFetched": False,
    }
