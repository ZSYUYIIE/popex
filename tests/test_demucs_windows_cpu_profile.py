from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "runtimes" / "profiles" / "windows-cpu"
PROFILE = json.loads((PROFILE_DIR / "profile.json").read_text(encoding="utf-8"))
ARTIFACTS = json.loads((PROFILE_DIR / "artifacts.json").read_text(encoding="utf-8"))
WORKER_LOCK = json.loads((PROFILE_DIR / "worker-runtime-lock.json").read_text(encoding="utf-8"))
REQ_IN = (PROFILE_DIR / "requirements.in").read_text(encoding="utf-8")
REQ_LOCK = (PROFILE_DIR / "requirements.lock").read_text(encoding="utf-8")
TORCH_LOCK = (PROFILE_DIR / "torch.lock").read_text(encoding="utf-8")
INSTALLER = (ROOT / "scripts" / "install_demucs_windows_cpu.ps1").read_text(encoding="utf-8")
INSTALL_DOC = (PROFILE_DIR / "INSTALL.md").read_text(encoding="utf-8")
RUNTIME_DOC = (ROOT / "docs" / "runtime" / "demucs-windows-cpu.md").read_text(encoding="utf-8")
PROVENANCE = (PROFILE_DIR / "lock-provenance.md").read_text(encoding="utf-8")
INVENTORY = (PROFILE_DIR / "third-party-inventory.md").read_text(encoding="utf-8")
WORKFLOW = (ROOT / ".github" / "workflows" / "demucs-windows-cpu-validation.yml").read_text(encoding="utf-8")


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


def test_01_required_profile_files_exist() -> None:
    expected = {"profile.json", "requirements.in", "requirements.lock", "torch.lock", "worker-runtime-lock.json", "artifacts.json", "third-party-inventory.md", "lock-provenance.md", "INSTALL.md"}
    assert expected <= {p.name for p in PROFILE_DIR.iterdir() if p.is_file()}


def test_02_profile_schema_and_identity() -> None:
    assert PROFILE["schemaVersion"] == 1
    assert PROFILE["runtimeProfile"] == "windows-x86_64-cpu-cpython313"
    assert PROFILE["supportedOS"] == ["Windows"]
    assert PROFILE["architecture"] == "x86_64"
    assert PROFILE["pythonImplementation"] == "CPython"
    assert PROFILE["pythonVersionRange"] == ">=3.13.0,<3.14.0"
    assert PROFILE["device"] == "cpu"
    assert PROFILE["workerProtocolVersion"] == 1
    assert PROFILE["workerPackageVersionExpectation"] == "1.0.0"


def test_03_exact_demucs_and_torch_pins() -> None:
    assert PROFILE["demucsVersion"] == "4.1.0"
    assert PROFILE["pytorchVersionBuild"] == "2.13.0+cpu"
    assert _pins(REQ_IN)["demucs"] == _pins(REQ_LOCK)["demucs"] == "4.1.0"
    assert _pins(TORCH_LOCK) == {"torch": "2.13.0+cpu"}
    assert "a17ff48608634db245e17e8bb00a9558554a49aeb1e4f5fe6cd039af2a10515b" in TORCH_LOCK


def test_04_official_indexes_only() -> None:
    assert PROFILE["packageIndexUrls"] == {"pypi": "https://pypi.org/simple", "pytorchCpu": "https://download.pytorch.org/whl/cpu"}
    assert "--index-url https://pypi.org/simple" in INSTALLER
    assert "--index-url https://download.pytorch.org/whl/cpu" in INSTALLER
    assert all(item["indexUrl"] in PROFILE["packageIndexUrls"].values() for item in ARTIFACTS["artifacts"])


def test_05_complete_hash_locked_wheel_set() -> None:
    pins = _pins(REQ_LOCK) | _pins(TORCH_LOCK)
    expected = {_canon(name): version for name, version in PROFILE["exactDependencyVersions"].items() if name != "popex-demucs-worker"}
    assert pins == expected
    blocks = re.split(r"(?m)(?=^[A-Za-z0-9_.-]+==)", REQ_LOCK)[1:]
    assert blocks and all("--hash=sha256:" in block for block in blocks)
    assert "--hash=sha256:" in TORCH_LOCK
    assert PROFILE["hashesEnforced"] is True


def test_06_artifact_inventory_matches_locks() -> None:
    artifacts = ARTIFACTS["artifacts"]
    assert {_canon(item["name"]): item["version"] for item in artifacts} == (_pins(REQ_LOCK) | _pins(TORCH_LOCK))
    assert all(item["filename"].endswith(".whl") for item in artifacts)
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in artifacts)
    assert ARTIFACTS["localPackage"] == {"name": "popex-demucs-worker", "version": "1.0.0", "sourcePath": "runtimes/demucs_worker", "indexFetched": False}


def test_07_windows_specific_artifacts_are_recorded() -> None:
    names = {item["filename"] for item in ARTIFACTS["artifacts"]}
    assert "torch-2.13.0+cpu-cp313-cp313-win_amd64.whl" in names
    assert "numpy-2.5.1-cp313-cp313-win_amd64.whl" in names
    assert "pyyaml-6.0.3-cp313-cp313-win_amd64.whl" in names
    assert "sphn-0.2.1-cp313-cp313-win_amd64.whl" in names
    assert "lameenc-1.8.4-cp313-cp313-win_amd64.whl" in names
    assert "colorama-0.4.6-py2.py3-none-any.whl" in names


def test_08_worker_lock_matches_profile() -> None:
    assert WORKER_LOCK["schemaVersion"] == 1
    assert WORKER_LOCK["runtimeProfile"] == PROFILE["runtimeProfile"]
    assert WORKER_LOCK["workerVersion"] == PROFILE["workerPackageVersionExpectation"]
    mapping = {"demucs": "demucs", "torch": "torch", "huggingface_hub": "huggingface_hub", "safetensors": "safetensors", "PyYAML": "PyYAML"}
    assert WORKER_LOCK["packages"] == {lock: PROFILE["exactDependencyVersions"][profile] for lock, profile in mapping.items()}


def test_09_forbidden_dependencies_absent() -> None:
    lock = (REQ_LOCK + TORCH_LOCK).lower()
    for forbidden in ("torchaudio", "openunmix", "dora-search", "demucs[train]", "nvidia-", "cuda"):
        assert forbidden not in lock
    assert not any(_canon(name).startswith("nvidia-") for name in PROFILE["exactDependencyVersions"])


def test_10_no_model_or_startup_download() -> None:
    install_inputs = REQ_LOCK + TORCH_LOCK + INSTALLER + WORKFLOW
    assert "huggingface.co/adefossez" not in install_inputs
    assert "955717e8.safetensors" not in install_inputs
    assert PROFILE["modelBundled"] is False
    assert PROFILE["startupDownload"] is False
    assert PROFILE["modelDownloadDuringInstallation"] is False
    for command in ("prepare-model", "verify-model", " separate"):
        assert command not in INSTALLER


def test_11_strict_powershell_and_refusal_paths() -> None:
    assert "Set-StrictMode -Version Latest" in INSTALLER
    assert '$ErrorActionPreference = "Stop"' in INSTALLER
    assert "This profile supports Windows only" in INSTALLER
    assert "Windows x86-64 only" in INSTALLER
    assert "CPython >=3.13.0,<3.14.0" in INSTALLER
    assert "local popex-demucs-worker source is missing" in INSTALLER
    assert "Refusing to overwrite an existing runtime destination" in INSTALLER


def test_12_user_local_isolated_layout() -> None:
    assert "LOCALAPPDATA" in INSTALLER
    assert "PopEx\\runtimes\\windows-x86_64-cpu-cpython313" in INSTALLER
    assert "-m venv" in INSTALLER
    assert "Scripts\\python.exe" in INSTALLER


def test_13_binary_hash_no_deps_installs() -> None:
    assert INSTALLER.count("--no-deps --require-hashes --only-binary=:all:") == 2
    assert "-m pip check" in INSTALLER


def test_14_worker_installed_last() -> None:
    ordinary = INSTALLER.index("requirements.lock")
    torch = INSTALLER.index("torch.lock", ordinary)
    worker = INSTALLER.index("--no-build-isolation --no-index", torch)
    copy_lock = INSTALLER.index("worker-runtime-lock.json", worker)
    assert ordinary < torch < worker < copy_lock
    assert "--no-deps --no-build-isolation --no-index" in INSTALLER


def test_15_cpu_and_forbidden_package_checks() -> None:
    assert "torch.version.cuda is not None" in INSTALLER
    assert "torch.cuda.is_available()" in INSTALLER
    assert '"torchaudio", "openunmix", "dora-search"' in INSTALLER
    assert 'name.startswith("nvidia-") or "cuda" in name' in INSTALLER


def test_16_runtime_probe_is_final_operation() -> None:
    nonempty = [line.strip() for line in INSTALLER.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    assert nonempty[-1] == "& $WorkerExecutable --protocol-version 1 runtime-probe"
    assert "model-probe" not in INSTALLER


def test_17_safe_cleanup_and_removal_documented() -> None:
    assert "Remove-Item -LiteralPath" in INSTALLER
    assert "incomplete isolated runtime" in INSTALLER
    assert "Remove-Item -LiteralPath" in INSTALL_DOC
    assert "Remove-Item -LiteralPath" in RUNTIME_DOC
    assert "does not delete source media" in INSTALL_DOC.lower()


def test_18_trusted_runtime_layout_documented() -> None:
    for text in (INSTALL_DOC, RUNTIME_DOC):
        assert "popex-demucs-worker.exe" in text
        assert "runtime-lock.json" in text
        assert "%LOCALAPPDATA%\\PopEx\\models" in text
        assert "trusted local configuration" in text


def test_19_inventory_is_complete() -> None:
    for name, version in PROFILE["exactDependencyVersions"].items():
        assert name == "popex-demucs-worker" or (_canon(name) in _canon(INVENTORY) and version in INVENTORY)
    for excluded in ("torchaudio", "openunmix", "dora-search", "NVIDIA/CUDA"):
        assert excluded.lower() in INVENTORY.lower()


def test_20_provenance_records_exact_windows_evidence() -> None:
    assert "30883764913" in PROVENANCE
    assert "8882280475" in PROVENANCE
    assert "Microsoft Windows Server 2025" in PROVENANCE
    assert "CPython `3.13.14`" in PROVENANCE
    assert "a17ff48608634db245e17e8bb00a9558554a49aeb1e4f5fe6cd039af2a10515b" in PROVENANCE
    assert "no model command was invoked" in PROVENANCE.lower()


def test_21_workflow_is_windows_path_focused_and_model_free() -> None:
    assert "runs-on: windows-latest" in WORKFLOW
    assert "workflow_dispatch:" in WORKFLOW
    assert '"runtimes/profiles/windows-cpu/**"' in WORKFLOW
    assert '"scripts/install_demucs_windows_cpu.ps1"' in WORKFLOW
    assert "pytest tests/test_demucs_windows_cpu_profile.py" in WORKFLOW
    for command in ("prepare-model", "verify-model", " separate"):
        assert command not in WORKFLOW


def test_22_workflow_performs_clean_installer_smoke() -> None:
    assert "install_demucs_windows_cpu.ps1" in WORKFLOW
    assert "runtime-probe" in WORKFLOW
    assert "installedVersions" in WORKFLOW and "lockedVersions" in WORKFLOW
    assert "torch.version.cuda" in WORKFLOW and "torch.cuda.is_available" in WORKFLOW
    assert "readiness" in WORKFLOW.lower() and "model-cache" in WORKFLOW
    assert "Remove-Item -Recurse -Force" in WORKFLOW
