# Install and verify the Windows CPU profile

Run from a PopEx checkout in 64-bit PowerShell with CPython 3.13 available:

```powershell
pwsh -File scripts\install_demucs_windows_cpu.ps1
```

The default isolated runtime is:

```text
%LOCALAPPDATA%\PopEx\runtimes\windows-x86_64-cpu-cpython313
```

Override it without changing the base PopEx environment:

```powershell
pwsh -File scripts\install_demucs_windows_cpu.ps1 `
  -Destination "D:\private\popex-demucs-runtime"
```

The installer refuses non-Windows systems, non-x64 systems, Python other than CPython 3.13, a missing local worker source, and an existing destination before creating the runtime.

## Trusted integration paths

For the default layout:

```text
Worker executable:
%LOCALAPPDATA%\PopEx\runtimes\windows-x86_64-cpu-cpython313\venv\Scripts\popex-demucs-worker.exe

Runtime lock:
%LOCALAPPDATA%\PopEx\runtimes\windows-x86_64-cpu-cpython313\runtime-lock.json

Recommended private model cache root:
%LOCALAPPDATA%\PopEx\models\windows-x86_64-cpu-cpython313
```

Supply the worker executable and runtime lock only through trusted local configuration. Do not accept either path from a web request.

## Verification

Installation finishes with exactly:

```text
popex-demucs-worker --protocol-version 1 runtime-probe
```

A valid result reports `compatible: true`, `runtimeProfile: windows-x86_64-cpu-cpython313`, and identical installed/locked versions. Installation does not prepare, verify, or separate a model.

## Remove

Stop processes using the optional runtime, then remove only the isolated runtime directory:

```powershell
Remove-Item -LiteralPath "$env:LOCALAPPDATA\PopEx\runtimes\windows-x86_64-cpu-cpython313" -Recurse -Force
```

When a custom `-Destination` was used, remove that exact directory. Removal does not delete source media, `analysis.wav`, analysis JSON, prior stems, or a separately managed model cache.
