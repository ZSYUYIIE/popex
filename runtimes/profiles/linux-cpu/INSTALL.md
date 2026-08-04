# Install and verify the Linux CPU profile

Run from the PopEx repository root after the concurrent worker package exists at `runtimes/demucs_worker`:

```bash
bash scripts/install_demucs_linux_cpu.sh
```

The default isolated runtime is:

```text
${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/linux-x86_64-cpu-cpython313
```

Override it without changing the base environment:

```bash
POPEX_DEMUCS_LINUX_CPU_DIR=/absolute/private/runtime/path \
  bash scripts/install_demucs_linux_cpu.sh
```

The installer accepts only Linux x86-64 and CPython 3.13. It refuses root execution, existing runtime directories, and a missing local worker source directory before creating anything.

## Verification

Installation ends with exactly the worker runtime probe:

```bash
POPEX_DEMUCS_RUNTIME_LOCK=/absolute/runtime/path/runtime-lock.json \
  /absolute/runtime/path/venv/bin/popex-demucs-worker \
  --protocol-version 1 runtime-probe
```

A valid result reports `compatible: true`, `runtimeProfile: linux-x86_64-cpu-cpython313`, and exact installed/locked versions. The installer never invokes `prepare-model`, `model-probe`, `verify-model`, or `separate`.

## Remove

Stop PopEx processes using the runtime, then remove only the isolated runtime directory:

```bash
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/linux-x86_64-cpu-cpython313"
```

If `POPEX_DEMUCS_LINUX_CPU_DIR` was used, remove that exact directory instead. This does not remove PopEx source media, analysis artifacts, model caches stored elsewhere, or prior stems.
