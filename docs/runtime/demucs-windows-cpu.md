# Demucs Windows CPU runtime

## Status

The Windows x86-64 CPython 3.13 CPU-only profile is validated for worker integration. Resolver evidence came from workflow run 1 (`30883764913`), and the clean isolated installer plus runtime probe passed in workflow run 14 (`30884730743`).

## Profile identity

- Runtime profile: `windows-x86_64-cpu-cpython313`
- Validated OS: Microsoft Windows Server 2025 Datacenter, build `10.0.26100`
- Runner image: `windows-2025-vs2026`, version `20260728.188.1`
- Architecture: x86-64 / AMD64
- Python: CPython `>=3.13.0,<3.14.0`; tested `3.13.14`
- Device: CPU
- PyTorch: `2.13.0+cpu`
- Demucs: `4.1.0`
- Worker protocol: `1`
- Worker: `popex-demucs-worker==1.0.0`

This profile does not cover Windows ARM64, 32-bit Windows, CUDA, another Python minor, or another worker protocol.

## Installation boundary

The PowerShell installer creates a new virtual environment outside the base PopEx environment and installs only hash-locked wheels from:

- `https://pypi.org/simple`;
- `https://download.pytorch.org/whl/cpu`.

The worker is installed last from the checked-out local source with no index, dependencies, or build isolation. No model is included or downloaded. Hugging Face offline/privacy variables are set before runtime imports.

## Validation evidence

The clean Windows workflow verified:

- PowerShell syntax;
- all `22` permanent profile tests;
- exact hash-enforced wheel-only installation;
- a complete `31`-distribution runtime inventory excluding pip;
- successful worker protocol-v1 `runtime-probe`;
- identical installed and locked versions for Demucs, Torch, Hugging Face Hub, safetensors, and PyYAML;
- `torch.version.cuda is None` and CUDA unavailable;
- no torchaudio, Open-Unmix, Dora Search, CUDA, or NVIDIA distribution;
- no model checkpoint, readiness manifest, or Hugging Face model cache.

The runtime probe reported:

```json
{
  "runtimeProfile": "windows-x86_64-cpu-cpython313",
  "workerVersion": "1.0.0",
  "pythonVersion": "3.13.14",
  "runtimeLockSource": "profile",
  "installedVersions": {
    "demucs": "4.1.0",
    "torch": "2.13.0+cpu",
    "huggingface_hub": "1.26.0",
    "safetensors": "0.8.0",
    "PyYAML": "6.0.3"
  },
  "compatible": true
}
```

## Excluded packages and assets

- torchaudio
- Open-Unmix
- Dora Search
- Demucs training extras
- CUDA and NVIDIA distributions
- model checkpoints
- readiness manifests
- Hugging Face model cache assets

## Trusted paths

Default worker executable template:

```text
%LOCALAPPDATA%\PopEx\runtimes\windows-x86_64-cpu-cpython313\venv\Scripts\popex-demucs-worker.exe
```

Default runtime lock template:

```text
%LOCALAPPDATA%\PopEx\runtimes\windows-x86_64-cpu-cpython313\runtime-lock.json
```

Recommended private model cache template:

```text
%LOCALAPPDATA%\PopEx\models\windows-x86_64-cpu-cpython313
```

These paths belong in trusted local configuration, not web input or committed machine-specific JSON.

## Install and remove

See `runtimes/profiles/windows-cpu/INSTALL.md` for installation details. After stopping its processes, the default runtime can be removed with:

```powershell
Remove-Item -LiteralPath "$env:LOCALAPPDATA\PopEx\runtimes\windows-x86_64-cpu-cpython313" -Recurse -Force
```

This removes only the optional runtime and does not delete PopEx source media, analysis artifacts, model caches stored elsewhere, or prior stems.

## Limitations

The validation proves package installation and worker compatibility on the recorded Windows image. It does not benchmark end-user hardware or validate Demucs inference quality. Model preparation remains a separate explicit-consent workflow.

## Revalidation boundary

Any package version, hash, Python minor, Torch build, Demucs version, worker protocol, worker implementation, Windows runner image, installer behavior, or package-index change requires a new clean installation and runtime-probe validation.
