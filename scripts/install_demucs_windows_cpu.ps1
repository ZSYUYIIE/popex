[CmdletBinding()]
param(
    [string]$Destination = "",
    [string]$PythonExecutable = "python",
    [string]$WorkerSource = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if (Test-Path variable:PSNativeCommandUseErrorActionPreference) {
    $PSNativeCommandUseErrorActionPreference = $true
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Split-Path -Parent $ScriptDir
$ProfileDir = Join-Path $RepoRoot "runtimes\profiles\windows-cpu"
if ([string]::IsNullOrWhiteSpace($WorkerSource)) {
    $WorkerSource = Join-Path $RepoRoot "runtimes\demucs_worker"
}
if ([string]::IsNullOrWhiteSpace($Destination)) {
    if ([string]::IsNullOrWhiteSpace($env:LOCALAPPDATA)) {
        throw "LOCALAPPDATA is required for the default normal-user runtime location."
    }
    $Destination = Join-Path $env:LOCALAPPDATA "PopEx\runtimes\windows-x86_64-cpu-cpython313"
}

$RuntimeCreated = $false
try {
    if (-not $IsWindows) {
        throw "This profile supports Windows only."
    }
    $Architecture = [System.Runtime.InteropServices.RuntimeInformation]::OSArchitecture
    if ($Architecture -ne [System.Runtime.InteropServices.Architecture]::X64) {
        throw "This profile supports Windows x86-64 only."
    }

    $PythonJson = & $PythonExecutable -c "import json, platform, struct, sys; print(json.dumps({'implementation': platform.python_implementation(), 'version': list(sys.version_info[:3]), 'bits': struct.calcsize('P') * 8}))"
    if ($LASTEXITCODE -ne 0) { throw "CPython 3.13 could not be executed." }
    $PythonInfo = $PythonJson | ConvertFrom-Json
    if ($PythonInfo.implementation -ne "CPython" -or $PythonInfo.version[0] -ne 3 -or $PythonInfo.version[1] -ne 13 -or $PythonInfo.bits -ne 64) {
        throw "This profile requires 64-bit CPython >=3.13.0,<3.14.0."
    }

    if (-not (Test-Path -LiteralPath (Join-Path $WorkerSource "pyproject.toml") -PathType Leaf)) {
        throw "The local popex-demucs-worker source is missing. Merge or check out the worker before installation."
    }
    foreach ($Required in @("profile.json", "requirements.lock", "torch.lock", "worker-runtime-lock.json")) {
        if (-not (Test-Path -LiteralPath (Join-Path $ProfileDir $Required) -PathType Leaf)) {
            throw "A required Windows CPU profile file is missing: $Required"
        }
    }
    if (Test-Path -LiteralPath $Destination) {
        throw "Refusing to overwrite an existing runtime destination. Remove it explicitly before reinstalling."
    }

    New-Item -ItemType Directory -Path $Destination | Out-Null
    $RuntimeCreated = $true
    $VenvDir = Join-Path $Destination "venv"
    & $PythonExecutable -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) { throw "Creating the isolated virtual environment failed." }
    $VenvPython = Join-Path $VenvDir "Scripts\python.exe"

    $env:HF_HOME = Join-Path $Destination "model-cache"
    $env:HF_HUB_CACHE = Join-Path $env:HF_HOME "hub"
    $env:HF_XET_CACHE = Join-Path $env:HF_HOME "xet"
    $env:HF_HUB_OFFLINE = "1"
    $env:HF_HUB_DISABLE_TELEMETRY = "1"
    $env:HF_HUB_DISABLE_IMPLICIT_TOKEN = "1"
    $env:HF_HUB_DISABLE_UPDATE_CHECK = "1"
    $env:HF_HUB_DISABLE_PROGRESS_BARS = "1"
    $env:PYTHONNOUSERSITE = "1"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    Remove-Item Env:HF_TOKEN -ErrorAction SilentlyContinue
    Remove-Item Env:HUGGING_FACE_HUB_TOKEN -ErrorAction SilentlyContinue

    & $VenvPython -m pip install --no-deps --require-hashes --only-binary=:all: --index-url https://pypi.org/simple -r (Join-Path $ProfileDir "requirements.lock")
    if ($LASTEXITCODE -ne 0) { throw "Installing the ordinary hash-locked Windows wheels failed." }
    & $VenvPython -m pip install --no-deps --require-hashes --only-binary=:all: --index-url https://download.pytorch.org/whl/cpu -r (Join-Path $ProfileDir "torch.lock")
    if ($LASTEXITCODE -ne 0) { throw "Installing the exact CPU-only Torch wheel failed." }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "The locked runtime failed pip check." }

    @'
import importlib.metadata as metadata
import os
from pathlib import Path

import demucs
import demucs.api
import demucs.hf
import einops
import huggingface_hub
import julius
import lameenc
import numpy
import safetensors
import sphn
import torch
import tqdm
import yaml

names = {dist.metadata["Name"].lower().replace("_", "-") for dist in metadata.distributions()}
for forbidden in ("torchaudio", "openunmix", "dora-search"):
    if forbidden in names:
        raise SystemExit(f"Forbidden package installed: {forbidden}")
if any(name.startswith("nvidia-") or "cuda" in name for name in names):
    raise SystemExit("A CUDA or NVIDIA package was installed in the CPU profile.")
if torch.__version__ != "2.13.0+cpu" or torch.version.cuda is not None or torch.cuda.is_available():
    raise SystemExit("The installed PyTorch build is not the approved CPU-only build.")
if demucs.__version__ != "4.1.0" or numpy.__version__ != "2.5.1":
    raise SystemExit("The installed Demucs or NumPy version is not approved.")
if Path(os.environ["HF_HOME"]).exists():
    raise SystemExit("Package installation or import unexpectedly created a model cache.")
'@ | & $VenvPython -
    if ($LASTEXITCODE -ne 0) { throw "The locked runtime import and CPU-only verification failed." }

    & $VenvPython -m pip install --no-deps --no-build-isolation --no-index $WorkerSource
    if ($LASTEXITCODE -ne 0) { throw "Installing the local popex-demucs-worker source failed." }
    & $VenvPython -m pip check
    if ($LASTEXITCODE -ne 0) { throw "The completed worker runtime failed pip check." }

    Copy-Item -LiteralPath (Join-Path $ProfileDir "worker-runtime-lock.json") -Destination (Join-Path $Destination "runtime-lock.json")
    $env:POPEX_DEMUCS_RUNTIME_LOCK = Join-Path $Destination "runtime-lock.json"
    $WorkerExecutable = Join-Path $VenvDir "Scripts\popex-demucs-worker.exe"
}
catch {
    [Console]::Error.WriteLine("Windows CPU runtime installation failed: $($_.Exception.Message)")
    if ($RuntimeCreated) {
        [Console]::Error.WriteLine("Review the error, then remove only the incomplete isolated runtime with:")
        [Console]::Error.WriteLine("  Remove-Item -LiteralPath '$Destination' -Recurse -Force")
    }
    else {
        [Console]::Error.WriteLine("No runtime directory was created.")
    }
    exit 1
}

& $WorkerExecutable --protocol-version 1 runtime-probe
