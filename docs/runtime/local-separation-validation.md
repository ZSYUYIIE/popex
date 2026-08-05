# Local stem-separation validation

## Purpose and safety boundary

The local separation doctor checks a separately installed PopEx Demucs runtime without importing Demucs or PyTorch into the base application. It supports the documented Linux and Windows x86-64 CPython 3.13 CPU profiles.

The two modes are intentionally different:

- `check` is passive and model-free. It calls only `runtime-probe` and `model-probe`. It never authorizes a model download, runs inference, creates a PopEx job, or touches existing exports or SQLite data.
- `validate --allow-model-download` is an explicit end-to-end action. It may download the audited model when the worker reports `download_required`, then uses deterministic synthetic stereo audio in a unique temporary PopEx data directory.

Neither mode accepts a media filename. Source audio stays on the device. The tool never sends trusted worker, runtime-lock, cache, or data paths through the web API or prints them in JSON.

## Prerequisites

1. Set up the normal PopEx base environment and use CPython 3.13.
2. Install one optional CPU runtime with the documented platform installer:
   - Linux: `scripts/install_demucs_linux_cpu.sh`
   - Windows: `scripts/install_demucs_windows_cpu.ps1`
3. Run the doctor from the repository root with the base environment's Python.

Runtime installation and model preparation are separate. The runtime installer does not download the model. Passive checking does not download the model.

## Direct CLI

Linux example using the documented default layout:

```bash
PROFILE=linux-x86_64-cpu-cpython313
RUNTIME_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/$PROFILE"
CACHE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/popex/models/$PROFILE"
python3.13 scripts/popex_separation_doctor.py check \
  --worker "$RUNTIME_DIR/venv/bin/popex-demucs-worker" \
  --runtime-lock "$RUNTIME_DIR/runtime-lock.json" \
  --cache-root "$CACHE_ROOT" \
  --expected-profile "$PROFILE" \
  --device cpu
```

Windows PowerShell example using the documented default layout:

```powershell
$Profile = "windows-x86_64-cpu-cpython313"
$RuntimeDir = Join-Path $env:LOCALAPPDATA "PopEx\runtimes\$Profile"
$CacheRoot = Join-Path $env:LOCALAPPDATA "PopEx\models\$Profile"
python scripts\popex_separation_doctor.py check `
  --worker (Join-Path $RuntimeDir "venv\Scripts\popex-demucs-worker.exe") `
  --runtime-lock (Join-Path $RuntimeDir "runtime-lock.json") `
  --cache-root $CacheRoot `
  --expected-profile $Profile `
  --device cpu
```

`check` prints one compact JSON object to standard output and a path-free human summary to standard error. Its state is exactly one of:

- `ready`
- `download_required`
- `runtime_missing`
- `unavailable`

A `download_required` result is expected on first use. It is not a failed download and does not mean a download was attempted.

## Platform wrappers

Linux:

```bash
bash docs/runtime/local-separation-validation-linux.sh check
```

Explicit synthetic validation:

```bash
bash docs/runtime/local-separation-validation-linux.sh validate --allow-model-download
```

Windows PowerShell:

```powershell
pwsh -NoProfile -File docs\runtime\local-separation-validation-windows.ps1 -Mode check
```

Explicit synthetic validation:

```powershell
pwsh -NoProfile -File docs\runtime\local-separation-validation-windows.ps1 `
  -Mode validate -AllowModelDownload
```

The wrappers use only the documented default runtime layout. Set the shown environment variables or pass the Windows parameters to use other explicit trusted paths. They do not search broad filesystem locations and do not install packages or models silently.

## What explicit validation does

With exact consent, the doctor:

1. validates the current OS, x86-64 architecture, CPython 3.13, worker, runtime lock, cache root, and profile;
2. creates a uniquely named directory under the system temporary directory or `--temporary-root`;
3. generates a deterministic one-second, stereo, 44.1 kHz PCM WAV and a temporary prepared/analyzed job;
4. starts the merged FastAPI app in-process with trusted local settings;
5. sends `allowModelDownload: true` only when the runtime reports `download_required`;
6. validates the schema-3 manifest, vocals, bass, drums, and other stems, plus all preview and download routes;
7. removes the entire unique temporary PopEx data directory.

The separately installed runtime and model cache are deliberately preserved. The doctor does not touch existing PopEx data: SQLite databases, jobs, exports, source media, and stems are never selected or modified.

## Network and disk expectations

Passive `check` does not authorize model network access. Explicit first-use validation may download the audited `955717e8.safetensors` checkpoint of 84,025,440 bytes plus small repository metadata into the private cache. CPU inference also creates temporary four-stem WAV output before the temporary validation directory is removed.

Keep enough free disk space for the installed runtime, model cache, temporary synthetic stems, and normal filesystem overhead. No private audio is used or uploaded.

## Supported operation

This workflow is for one local user and a single PopEx server process. It validates the CPU profile only. Windows ARM64, Linux ARM64, 32-bit systems, CUDA, MPS, multiple Uvicorn workers, public network deployment, and inference quality are outside this doctor contract.

A passing repository CI run proves the doctor logic and fake-client safety tests. It does not prove that a particular user's machine, runtime installation, or model cache is valid. Run the doctor locally after the hosted real-model gate is approved.

## Cleanup and removal

Temporary validation data is removed automatically. After an interrupted run, inspect the chosen temporary parent and remove only a directory whose name starts with `popex-separation-doctor-` after confirming no doctor process is running.

Remove the model cache separately only when you intentionally want the next validation to require a new explicit download.

Linux default cache removal:

```bash
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/popex/models/linux-x86_64-cpu-cpython313"
```

Windows default cache removal:

```powershell
Remove-Item -LiteralPath "$env:LOCALAPPDATA\PopEx\models\windows-x86_64-cpu-cpython313" -Recurse -Force
```

Remove the optional runtime separately with its platform profile guide. Never use the temporary-data cleanup command for the runtime or cache, and never use the runtime/cache removal commands for the normal PopEx data directory.
