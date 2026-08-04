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
PROVENANCE = (PROFILE_DIR / "lock-provenance.md").read_text(encoding="utf-8")
INVENTORY = (PROFILE_DIR / "third-party-inventory.md").read_text(encoding="utf-8")


def _canon(value: str) -> str:
    return value.lower().replace("_", "-")


def _pins(text: str) -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "==" not in stripped:
            continue
        name, value = stripped.split("==", 1)
        pins[_canon(name)] = value.split()[0].removesuffix("\\")
    return pins


def test_01_profile_schema() -> None:
    assert PROFILE["schemaVersion"] == 1
    assert PROFILE["runtimeProfile"] == "linux-x86_64-cpu-cpython313"
    assert PROFILE["supportedOS"] == ["Linux"]
    assert PROFILE["architecture"] == "x86_64"
    assert PROFILE["pythonImplementation"] == "CPython"
    assert PROFILE["pythonVersionRange"] == ">=3.13.0,<3.14.0"
    assert PROFILE["device"] == "cpu"
    assert PROFILE["workerProtocolVersion"] == 1
    assert PROFILE["workerPackageVersionExpectation"] == "1.0.0"


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
    assert PROFILE["packageIndexUrls"] == {
        "pypi": "https://pypi.org/simple",
        "pytorchCpu": "https://download.pytorch.org/whl/cpu",
    }
    assert "--index-url https://download.pytorch.org/whl/cpu" in INSTALLER
    torch_artifact = next(item for item in ARTIFACTS["artifacts"] if item["name"] == "torch")
    assert torch_artifact["indexUrl"] == "https://download.pytorch.org/whl/cpu"


def test_05_no_cuda_or_nvidia_packages() -> None:
    installed = {_canon(name) for name in PROFILE["exactDependencyVersions"]}
    assert not any(name.startswith("nvidia-") for name in installed)
    assert "cuda" not in installed
    assert "torch==2.13.0+cpu" in TORCH_LOCK
    assert PROFILE["validationEnvironment"]["torchVersionCuda"] is None
    assert PROFILE["validationEnvironment"]["torchCudaAvailable"] is False


def test_06_no_torchaudio() -> None:
    assert "torchaudio" not in _pins(REQ_LOCK)
    assert "torchaudio" not in _pins(TORCH_LOCK)
    assert PROFILE["excludedPackages"]["torchaudio"]["installed"] is False
    assert PROFILE["validationEnvironment"]["forbiddenPackagesPresent"] == []


def test_07_no_training_extras() -> None:
    lock_text = REQ_LOCK + TORCH_LOCK
    assert "demucs[train]" not in lock_text.lower()
    assert "dora-search" not in _pins(lock_text)
    assert "openunmix" not in _pins(lock_text)
    assert PROFILE["excludedPackages"]["dora-search"]["installed"] is False
    assert PROFILE["excludedPackages"]["openunmix"]["installed"] is False


def test_08_no_model_urls_or_weights() -> None:
    install_inputs = REQ_LOCK + TORCH_LOCK + INSTALLER
    assert "huggingface.co/adefossez" not in install_inputs
    assert "955717e8.safetensors" not in install_inputs
    assert PROFILE["modelBundled"] is False
    assert PROFILE["modelDownloadDuringInstallation"] is False
    assert PROFILE["validationEnvironment"]["checkpointAssetsPresent"] == []
    assert PROFILE["validationEnvironment"]["huggingFaceModelCacheCreated"] is False


def test_09_no_startup_download() -> None:
    assert PROFILE["startupDownload"] is False
    assert "prepare-model" not in INSTALLER
    assert "model-probe" not in INSTALLER


def test_10_installer_platform_refusal() -> None:
    assert '$(uname -s)' in INSTALLER and '!= "Linux"' in INSTALLER
    assert '$(uname -m)' in INSTALLER
    assert "x86_64|amd64" in INSTALLER
    assert "supports Linux x86-64 only" in INSTALLER
    refusals = PROFILE["validationEnvironment"]["refusalTests"]
    assert refusals["unsupportedOS"] == 2
    assert refusals["unsupportedArchitecture"] == 2


def test_11_installer_python_and_root_refusal() -> None:
    assert 'platform.python_implementation() == "CPython"' in INSTALLER
    assert "sys.version_info[:2] == (3, 13)" in INSTALLER
    assert "requires CPython >=3.13.0,<3.14.0" in INSTALLER
    assert "Refusing to install the optional runtime as root" in INSTALLER
    refusals = PROFILE["validationEnvironment"]["refusalTests"]
    assert refusals["wrongPython"] == 2
    assert refusals["root"] == 2


def test_12_isolated_venv_behavior() -> None:
    assert '"$PYTHON_BIN" -m venv "$VENV_DIR"' in INSTALLER
    assert 'VENV_PYTHON="$VENV_DIR/bin/python"' in INSTALLER
    assert "POPEX_DEMUCS_LINUX_CPU_DIR" in INSTALLER
    assert "pip install -e '.[dev]'" not in INSTALLER


def test_13_strict_shell_mode() -> None:
    assert INSTALLER.startswith("#!/usr/bin/env bash\nset -euo pipefail\n")
    assert "IFS=$'\\n\\t'" in INSTALLER
    assert "trap on_exit EXIT" in INSTALLER
    assert PROFILE["validationEnvironment"]["shellcheck"] == "0.9.0"


def test_14_local_worker_installation_ordering() -> None:
    worker_check = INSTALLER.index('[[ ! -f "$WORKER_DIR/pyproject.toml" ]]')
    create_venv = INSTALLER.index('-m venv "$VENV_DIR"')
    requirements_install = INSTALLER.index('-r "$PROFILE_DIR/requirements.lock"')
    worker_install = INSTALLER.index('"$WORKER_DIR"', requirements_install)
    assert worker_check < create_venv < requirements_install < worker_install
    assert "--no-build-isolation" in INSTALLER
    assert "--no-index" in INSTALLER
    assert PROFILE["validationEnvironment"]["refusalTests"]["missingWorker"] == 3


def test_15_runtime_probe_rather_than_model_preparation() -> None:
    assert "--protocol-version 1 runtime-probe" in INSTALLER
    final_command = [line for line in INSTALLER.splitlines() if line and not line.startswith("#")][-1]
    assert final_command.endswith("--protocol-version 1 runtime-probe")
    for forbidden in ("prepare-model", "verify-model", "model-probe", " separate"):
        assert forbidden not in INSTALLER


def test_16_removal_and_existing_destination_behavior() -> None:
    assert "rm -rf" in INSTALL_DOC
    assert "rm -rf" in RUNTIME_DOC
    assert "incomplete isolated runtime" in INSTALLER
    assert PROFILE["validationEnvironment"]["refusalTests"]["existingDestination"] == 3


def test_17_lock_and_profile_consistency() -> None:
    lock_pins = _pins(REQ_LOCK) | _pins(TORCH_LOCK)
    expected = {
        _canon(name): version
        for name, version in PROFILE["exactDependencyVersions"].items()
        if name != "popex-demucs-worker"
    }
    assert lock_pins == expected
    requirement_blocks = re.split(r"(?m)(?=^[A-Za-z0-9_.-]+==)", REQ_LOCK)[1:]
    assert requirement_blocks
    assert all("--hash=sha256:" in block for block in requirement_blocks)
    assert PROFILE["hashesEnforced"] is True


def test_18_inventory_and_artifact_identity_complete() -> None:
    expected = {
        _canon(name)
        for name in PROFILE["exactDependencyVersions"]
        if name != "popex-demucs-worker"
    }
    artifacts = ARTIFACTS["artifacts"]
    artifact_names = {_canon(item["name"]) for item in artifacts}
    assert artifact_names == expected
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in artifacts)
    for name, version in PROFILE["exactDependencyVersions"].items():
        assert name == "popex-demucs-worker" or (name in INVENTORY and version in INVENTORY)
    for excluded in ("torchaudio", "openunmix", "dora-search", "NVIDIA/CUDA"):
        assert excluded.lower() in INVENTORY.lower()
    assert ARTIFACTS["localPackage"] == {
        "name": "popex-demucs-worker",
        "version": "1.0.0",
        "sourcePath": "runtimes/demucs_worker",
        "indexFetched": False,
    }


def test_19_worker_runtime_lock_matches_profile() -> None:
    assert WORKER_LOCK["schemaVersion"] == 1
    assert WORKER_LOCK["runtimeProfile"] == PROFILE["runtimeProfile"]
    assert WORKER_LOCK["workerVersion"] == PROFILE["workerPackageVersionExpectation"]
    mapping = {
        "demucs": "demucs",
        "torch": "torch",
        "huggingface_hub": "huggingface-hub",
        "safetensors": "safetensors",
        "PyYAML": "PyYAML",
    }
    assert WORKER_LOCK["packages"] == {
        lock_name: PROFILE["exactDependencyVersions"][profile_name]
        for lock_name, profile_name in mapping.items()
    }


def test_20_integration_evidence_is_exact_and_complete() -> None:
    validation = PROFILE["validationEnvironment"]
    assert PROFILE["validationStatus"] == "validated-ready-for-worker-integration"
    assert validation["integrationEvidenceRun"] == 72
    assert validation["testedWorkerPullRequest"] == 12
    assert validation["testedWorkerHead"] == "ef5d0e41a60f44372e14c4685b51a20cc9acd862"
    assert validation["finalLockedInstallerSmokeTest"] == "passed"
    probe = validation["runtimeProbe"]
    assert probe["compatible"] is True
    assert probe["runtimeProfile"] == PROFILE["runtimeProfile"]
    assert probe["installedVersions"] == probe["lockedVersions"] == WORKER_LOCK["packages"]
    assert validation["forbiddenPackagesPresent"] == []
    assert validation["checkpointAssetsPresent"] == []
    assert validation["huggingFaceModelCacheCreated"] is False
    assert validation["testedWorkerHead"] in PROVENANCE
    assert validation["testedWorkerHead"] in RUNTIME_DOC
    assert "prepare-model was not invoked" in PROVENANCE
