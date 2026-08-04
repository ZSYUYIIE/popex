# Demucs Windows CPU runtime

## Status

The Windows x86-64 CPython 3.13 candidate lock has exact official wheel evidence. Final clean installation and worker runtime-probe validation are pending the focused Windows workflow.

## Profile identity

- Runtime profile: `windows-x86_64-cpu-cpython313`
- Validated resolver OS: Microsoft Windows Server 2025 Datacenter, build `10.0.26100`
- Architecture: x86-64 / AMD64
- Python: CPython `>=3.13.0,<3.14.0`
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

The worker is installed last from the checked-out local source. No model is included or downloaded. Hugging Face offline/privacy variables are set before runtime imports.

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

This removes only the optional runtime.

## Revalidation boundary

Any package version, hash, Python minor, Torch build, Demucs version, worker protocol, worker implementation, Windows runner image, or package-index change requires a new clean installation and runtime-probe validation.
