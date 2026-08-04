from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
PROFILE_DIR = ROOT / "runtimes" / "profiles" / "linux-cpu"
PROFILE = json.loads((PROFILE_DIR / "profile.json").read_text(encoding="utf-8"))
ARTIFACTS = json.loads((PROFILE_DIR / "artifacts.json").read_text(encoding="utf-8"))
WORKER_LOCK = json.loads((PROFILE_DIR / "worker-runtime-lock.json").read_text(encoding="utf-8"))
REQ_LOCK = (PROFILE_DIR / "requirements.lock").read_text(encoding="utf-8")
TORCH_LOCK = (PROFILE_DIR / "torch.lock").read_text(encoding="utf-8")
INVENTORY = (PROFILE_DIR / "third-party-inventory.md").read_text(encoding="utf-8")
INSTALLER_PATH = ROOT / "scripts" / "install_demucs_linux_cpu.sh"
INSTALLER = INSTALLER_PATH.read_text(encoding="utf-8")
PR12_HEAD = "2210ccbd2a8bad6d321d757471871ef434fbc054"


def _canon(value: str) -> str:
    return value.lower().replace("_", "-")


def _pins(text: str) -> dict[str, str]:
    return {
        _canon(match.group(1)): match.group(2)
        for match in re.finditer(r"(?mi)^([A-Za-z0-9_.-]+)==([^ \\\n]+)", text)
    }


def test_capture_pr12_linux_cpu_integration_evidence(tmp_path: Path) -> None:
    if not (
        os.getenv("GITHUB_ACTIONS") == "true"
        and platform.system() == "Linux"
        and platform.machine().lower() in {"x86_64", "amd64"}
        and sys.version_info[:2] == (3, 13)
    ):
        pytest.skip("one-time GitHub Actions integration evidence capture")

    expected_versions = {
        _canon(name): version
        for name, version in PROFILE["exactDependencyVersions"].items()
    }
    lock_versions = _pins(REQ_LOCK) | _pins(TORCH_LOCK)
    assert lock_versions == {
        name: version
        for name, version in expected_versions.items()
        if name != "popex-demucs-worker"
    }
    artifact_versions = {
        _canon(item["name"]): item["version"]
        for item in ARTIFACTS["artifacts"]
    }
    assert artifact_versions == lock_versions
    assert WORKER_LOCK == {
        "schemaVersion": 1,
        "runtimeProfile": "linux-x86_64-cpu-cpython313",
        "workerVersion": "1.0.0",
        "packages": {
            "demucs": "4.1.0",
            "torch": "2.13.0+cpu",
            "huggingface_hub": "1.26.0",
            "safetensors": "0.8.0",
            "PyYAML": "6.0.3",
        },
    }
    for name, version in PROFILE["exactDependencyVersions"].items():
        assert name == "popex-demucs-worker" or (name in INVENTORY and version in INVENTORY)

    shellcheck = shutil.which("shellcheck")
    assert shellcheck is not None
    shellcheck_version = subprocess.run(
        [shellcheck, "--version"], check=True, capture_output=True, text=True
    ).stdout.strip()
    lint = subprocess.run(
        [shellcheck, str(INSTALLER_PATH)], capture_output=True, text=True
    )
    assert lint.returncode == 0, lint.stdout + lint.stderr

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
    assert fetched_head == PR12_HEAD

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

    def run_installer(
        runtime_dir: Path,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = base_env.copy()
        env["POPEX_DEMUCS_LINUX_CPU_DIR"] = str(runtime_dir)
        if extra_env:
            env.update(extra_env)
        return subprocess.run(
            ["bash", str(INSTALLER_PATH)],
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

    refusals: dict[str, dict[str, object]] = {}
    refusal_cases = [
        (
            "unsupportedOS",
            tmp_path / "unsupported-os",
            {"PATH": fake_path, "FAKE_UNAME_S": "Darwin", "FAKE_UNAME_M": "x86_64"},
            2,
            "supports Linux only",
        ),
        (
            "unsupportedArchitecture",
            tmp_path / "unsupported-arch",
            {"PATH": fake_path, "FAKE_UNAME_S": "Linux", "FAKE_UNAME_M": "aarch64"},
            2,
            "supports Linux x86-64 only",
        ),
        (
            "wrongPython",
            tmp_path / "wrong-python-runtime",
            {"POPEX_DEMUCS_PYTHON": str(fake_python)},
            2,
            "requires CPython >=3.13.0,<3.14.0",
        ),
        (
            "missingWorker",
            tmp_path / "missing-worker-runtime",
            {"POPEX_DEMUCS_WORKER_DIR": str(tmp_path / "missing-worker")},
            3,
            "worker source is missing",
        ),
    ]
    for name, runtime_dir, extra_env, return_code, message in refusal_cases:
        result = run_installer(runtime_dir, extra_env)
        assert result.returncode == return_code
        assert message in result.stderr
        assert not runtime_dir.exists()
        refusals[name] = {"returnCode": result.returncode, "createdRuntime": False}

    existing_dir = tmp_path / "existing-runtime"
    existing_dir.mkdir()
    sentinel = existing_dir / "sentinel.txt"
    sentinel.write_text("preserve", encoding="utf-8")
    existing = run_installer(existing_dir)
    assert existing.returncode == 3
    assert "Refusing to overwrite" in existing.stderr
    assert sentinel.read_text(encoding="utf-8") == "preserve"
    refusals["existingDestination"] = {
        "returnCode": existing.returncode,
        "sentinelPreserved": True,
    }

    sudo = shutil.which("sudo")
    assert sudo is not None
    subprocess.run([sudo, "-n", "true"], check=True, capture_output=True, text=True)
    root_runtime = tmp_path / "root-runtime"
    root_result = subprocess.run(
        [
            sudo,
            "-n",
            "env",
            f"POPEX_DEMUCS_LINUX_CPU_DIR={root_runtime}",
            "bash",
            str(INSTALLER_PATH),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert root_result.returncode == 2
    assert "Refusing to install the optional runtime as root" in root_result.stderr
    assert not root_runtime.exists()
    refusals["root"] = {"returnCode": root_result.returncode, "createdRuntime": False}

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
    if name in {"torchaudio", "openunmix", "dora-search", "cuda"}
    or name.startswith("nvidia-")
)
checkpoint_assets = sorted(
    path.relative_to(runtime).as_posix()
    for path in runtime.rglob("*")
    if path.is_file()
    and (
        path.name == "955717e8.safetensors"
        or path.suffix in {".th", ".ckpt"}
        or "readiness" in path.parts
    )
)
package_model_metadata = sorted(
    path.relative_to(runtime).as_posix()
    for path in runtime.rglob("htdemucs.yaml")
    if path.is_file()
)
print(json.dumps({
    "packages": packages,
    "torchVersion": torch.__version__,
    "torchVersionCuda": torch.version.cuda,
    "torchCudaAvailable": torch.cuda.is_available(),
    "forbiddenPackages": forbidden,
    "checkpointAssets": checkpoint_assets,
    "packageModelMetadata": package_model_metadata,
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
    installed_subset = {
        name: installed["packages"].get(name)
        for name in expected_versions
    }
    assert installed_subset == expected_versions
    assert installed["torchVersion"] == "2.13.0+cpu"
    assert installed["torchVersionCuda"] is None
    assert installed["torchCudaAvailable"] is False
    assert installed["forbiddenPackages"] == []
    assert installed["checkpointAssets"] == []
    assert installed["huggingFaceCacheExists"] is False
    assert installed["packageModelMetadata"] == [
        "venv/lib/python3.13/site-packages/demucs/remote/htdemucs.yaml"
    ]

    evidence = {
        "workerHead": fetched_head,
        "shellcheckVersion": shellcheck_version,
        "shellcheckPassed": True,
        "internalConsistency": True,
        "refusals": refusals,
        "runtimeProbe": runtime_probe,
        "installedProfileVersions": installed_subset,
        "torchVersionCuda": installed["torchVersionCuda"],
        "torchCudaAvailable": installed["torchCudaAvailable"],
        "forbiddenPackages": installed["forbiddenPackages"],
        "checkpointAssets": installed["checkpointAssets"],
        "packageModelMetadata": installed["packageModelMetadata"],
        "huggingFaceCacheExists": installed["huggingFaceCacheExists"],
    }
    pytest.fail("POPEX_INTEGRATION_EVIDENCE=" + json.dumps(evidence, sort_keys=True))
