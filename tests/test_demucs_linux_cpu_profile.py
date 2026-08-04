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


def test_capture_pr12_linux_cpu_integration_evidence(tmp_path: Path) -> None:
    import os
    import platform
    import shutil
    import subprocess
    import sys
    import tarfile

    import pytest

    if not (
        os.getenv("GITHUB_ACTIONS") == "true"
        and platform.system() == "Linux"
        and platform.machine().lower() in {"x86_64", "amd64"}
        and sys.version_info[:2] == (3, 13)
    ):
        pytest.skip("one-time GitHub Actions integration evidence capture")

    worker_head = "2210ccbd2a8bad6d321d757471871ef434fbc054"
    installer_path = ROOT / "scripts" / "install_demucs_linux_cpu.sh"

    shellcheck = shutil.which("shellcheck")
    assert shellcheck is not None, "shellcheck is unavailable on the integration runner"
    shellcheck_version = subprocess.run(
        [shellcheck, "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    shellcheck_result = subprocess.run(
        [shellcheck, str(installer_path)],
        capture_output=True,
        text=True,
    )
    assert shellcheck_result.returncode == 0, shellcheck_result.stdout + shellcheck_result.stderr

    subprocess.run(
        ["git", "fetch", "--no-tags", "--depth=1", "origin", "pull/12/head"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    fetched_head = subprocess.run(
        ["git", "rev-parse", "FETCH_HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert fetched_head == worker_head

    worker_checkout = tmp_path / "worker-checkout"
    worker_checkout.mkdir()
    archive_path = tmp_path / "worker.tar"
    with archive_path.open("wb") as archive_file:
        subprocess.run(
            ["git", "archive", "--format=tar", fetched_head, "runtimes/demucs_worker"],
            cwd=ROOT,
            check=True,
            stdout=archive_file,
        )
    with tarfile.open(archive_path) as archive:
        archive.extractall(worker_checkout, filter="data")
    worker_dir = worker_checkout / "runtimes" / "demucs_worker"
    assert (worker_dir / "pyproject.toml").is_file()

    base_env = os.environ.copy()
    base_env.update(
        {
            "POPEX_DEMUCS_PYTHON": sys.executable,
            "POPEX_DEMUCS_WORKER_DIR": str(worker_dir),
            "PIP_CACHE_DIR": str(tmp_path / "pip-cache"),
            "HOME": str(tmp_path / "home"),
        }
    )
    (tmp_path / "home").mkdir()

    def run_installer(runtime_dir: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
        env = base_env.copy()
        env["POPEX_DEMUCS_LINUX_CPU_DIR"] = str(runtime_dir)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(installer_path)],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=900,
        )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_uname = fake_bin / "uname"
    fake_uname.write_text(
        "#!/usr/bin/env bash\n"
        "case \"${1:-}\" in\n"
        "  -s) printf '%s\\n' \"${FAKE_UNAME_S:-Linux}\" ;;\n"
        "  -m) printf '%s\\n' \"${FAKE_UNAME_M:-x86_64}\" ;;\n"
        "  *) exec /usr/bin/uname \"$@\" ;;\n"
        "esac\n",
        encoding="utf-8",
    )
    fake_uname.chmod(0o755)
    fake_python = fake_bin / "wrong-python"
    fake_python.write_text("#!/usr/bin/env bash\nexit 1\n", encoding="utf-8")
    fake_python.chmod(0o755)
    fake_path = str(fake_bin) + os.pathsep + base_env["PATH"]

    refusal_evidence: dict[str, dict[str, object]] = {}

    unsupported_os_dir = tmp_path / "unsupported-os"
    unsupported_os = run_installer(
        unsupported_os_dir,
        {"PATH": fake_path, "FAKE_UNAME_S": "Darwin", "FAKE_UNAME_M": "x86_64"},
    )
    assert unsupported_os.returncode == 2
    assert "supports Linux only" in unsupported_os.stderr
    assert not unsupported_os_dir.exists()
    refusal_evidence["unsupportedOS"] = {"returnCode": unsupported_os.returncode, "createdRuntime": False}

    unsupported_arch_dir = tmp_path / "unsupported-arch"
    unsupported_arch = run_installer(
        unsupported_arch_dir,
        {"PATH": fake_path, "FAKE_UNAME_S": "Linux", "FAKE_UNAME_M": "aarch64"},
    )
    assert unsupported_arch.returncode == 2
    assert "supports Linux x86-64 only" in unsupported_arch.stderr
    assert not unsupported_arch_dir.exists()
    refusal_evidence["unsupportedArchitecture"] = {"returnCode": unsupported_arch.returncode, "createdRuntime": False}

    wrong_python_dir = tmp_path / "wrong-python-runtime"
    wrong_python = run_installer(
        wrong_python_dir,
        {"POPEX_DEMUCS_PYTHON": str(fake_python)},
    )
    assert wrong_python.returncode == 2
    assert "requires CPython >=3.13.0,<3.14.0" in wrong_python.stderr
    assert not wrong_python_dir.exists()
    refusal_evidence["wrongPython"] = {"returnCode": wrong_python.returncode, "createdRuntime": False}

    missing_worker_dir = tmp_path / "missing-worker-runtime"
    missing_worker = run_installer(
        missing_worker_dir,
        {"POPEX_DEMUCS_WORKER_DIR": str(tmp_path / "does-not-exist")},
    )
    assert missing_worker.returncode == 3
    assert "worker source is missing" in missing_worker.stderr
    assert not missing_worker_dir.exists()
    refusal_evidence["missingWorker"] = {"returnCode": missing_worker.returncode, "createdRuntime": False}

    existing_dir = tmp_path / "existing-runtime"
    existing_dir.mkdir()
    sentinel = existing_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    existing = run_installer(existing_dir)
    assert existing.returncode == 3
    assert "Refusing to overwrite" in existing.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    refusal_evidence["existingDestination"] = {"returnCode": existing.returncode, "sentinelPreserved": True}

    sudo = shutil.which("sudo")
    assert sudo is not None, "sudo is unavailable for the root-refusal test"
    subprocess.run([sudo, "-n", "true"], check=True, capture_output=True, text=True)
    root_runtime = tmp_path / "root-runtime"
    root_result = subprocess.run(
        [
            sudo,
            "-n",
            "env",
            f"POPEX_DEMUCS_LINUX_CPU_DIR={root_runtime}",
            "bash",
            str(installer_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert root_result.returncode == 2
    assert "Refusing to install the optional runtime as root" in root_result.stderr
    assert not root_runtime.exists()
    refusal_evidence["root"] = {"returnCode": root_result.returncode, "createdRuntime": False}

    runtime_dir = tmp_path / "installed-runtime"
    installation = run_installer(runtime_dir)
    assert installation.returncode == 0, installation.stdout + "\n" + installation.stderr

    json_lines: list[dict[str, object]] = []
    for line in installation.stdout.splitlines():
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            json_lines.append(payload)
    assert len(json_lines) >= 2
    runtime_probe = json_lines[-1]
    assert runtime_probe["protocolVersion"] == 1
    assert runtime_probe["command"] == "runtime-probe"
    assert runtime_probe["status"] == "ok"
    probe_result = runtime_probe["result"]
    assert isinstance(probe_result, dict)
    assert probe_result["compatible"] is True
    assert probe_result["runtimeProfile"] == "linux-x86_64-cpu-cpython313"
    assert probe_result["installedVersions"] == probe_result["lockedVersions"]

    inspect_code = r'''
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import torch

runtime = Path(os.environ["POPEX_RUNTIME_DIR"])
def canon(value: str) -> str:
    return value.lower().replace("_", "-")
packages = {canon(dist.metadata["Name"]): dist.version for dist in metadata.distributions()}
forbidden = sorted(
    name for name in packages
    if name in {"torchaudio", "openunmix", "dora-search", "cuda"} or name.startswith("nvidia-")
)
model_assets = sorted(
    path.relative_to(runtime).as_posix()
    for path in runtime.rglob("*")
    if path.is_file()
    and (
        path.name in {"955717e8.safetensors", "htdemucs.yaml"}
        or path.suffix in {".th", ".ckpt"}
        or "readiness" in path.parts
    )
)
print(json.dumps({
    "packages": packages,
    "torchVersion": torch.__version__,
    "torchVersionCuda": torch.version.cuda,
    "torchCudaAvailable": torch.cuda.is_available(),
    "forbiddenPackages": forbidden,
    "modelAssets": model_assets,
    "huggingFaceCacheExists": (runtime / "huggingface").exists(),
}, sort_keys=True))
'''
    inspection = subprocess.run(
        [str(runtime_dir / "venv" / "bin" / "python"), "-c", inspect_code],
        check=True,
        capture_output=True,
        text=True,
        env={**base_env, "POPEX_RUNTIME_DIR": str(runtime_dir)},
    )
    installed = json.loads(inspection.stdout)
    expected_versions = {
        name.lower().replace("_", "-"): version
        for name, version in PROFILE["exactDependencyVersions"].items()
    }
    installed_subset = {name: installed["packages"].get(name) for name in expected_versions}
    assert installed_subset == expected_versions
    assert installed["torchVersion"] == "2.13.0+cpu"
    assert installed["torchVersionCuda"] is None
    assert installed["torchCudaAvailable"] is False
    assert installed["forbiddenPackages"] == []
    assert installed["modelAssets"] == []
    assert installed["huggingFaceCacheExists"] is False

    evidence = {
        "workerHead": fetched_head,
        "shellcheckVersion": shellcheck_version,
        "shellcheckPassed": True,
        "refusals": refusal_evidence,
        "runtimeProbe": runtime_probe,
        "installedProfileVersions": installed_subset,
        "torchVersionCuda": installed["torchVersionCuda"],
        "torchCudaAvailable": installed["torchCudaAvailable"],
        "forbiddenPackages": installed["forbiddenPackages"],
        "modelAssets": installed["modelAssets"],
        "huggingFaceCacheExists": installed["huggingFaceCacheExists"],
    }
    pytest.fail("POPEX_INTEGRATION_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
