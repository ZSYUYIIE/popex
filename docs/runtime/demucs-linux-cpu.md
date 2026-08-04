# Demucs Linux CPU runtime

## Status

The Linux x86-64 CPython 3.13 dependency set is validated. Final integration is conditional on the local `popex-demucs-worker` version 1.0.0 source directory being present at `runtimes/demucs_worker`.

## Profile identity

- Runtime profile: `linux-x86_64-cpu-cpython313`
- Operating system: Linux
- Architecture: x86-64
- Python: CPython `>=3.13.0,<3.14.0`
- Device: CPU
- PyTorch: `2.13.0+cpu`
- Demucs: `4.1.0`
- Worker protocol: `1`
- Expected worker: `popex-demucs-worker==1.0.0`

This is not a universal profile. Do not use it for ARM, macOS, Windows, CUDA, another Python minor version, or a different worker protocol.

## What installation does

The installer creates a new virtual environment outside the base PopEx environment, installs only hash-locked wheels, installs the local worker source last, and finishes with `runtime-probe`. It contacts:

- `https://pypi.org/simple` for ordinary Python wheels;
- `https://download.pytorch.org/whl/cpu` for the exact CPU-only Torch wheel.

It does not contact Hugging Face for model assets. It sets the Hub to offline mode before any runtime import and refuses installation if an import creates the configured cache path.

## What is excluded

- `torchaudio`
- `openunmix`
- `dora-search`
- Demucs training extras
- CUDA and all `nvidia-*` packages
- model weights and readiness manifests

Open-Unmix 1.3.0 was evaluated only as an exclusion. It is not required by Demucs 4.1.0 and its inference stack uses torchaudio. Dora Search 0.1.12 is a Demucs training dependency and is not installed.

## Verified dependency evidence

On GitHub Actions Ubuntu 24.04.4 with CPython 3.13.14:

- the official CPU wheel resolved as `torch-2.13.0+cpu-cp313-cp313-manylinux_2_28_x86_64.whl`;
- `torch.version.cuda` was `None`;
- `torch.cuda.is_available()` was false;
- Demucs 4.1.0, `demucs.api`, and `demucs.hf` imported successfully after adding the explicit NumPy 2.5.1 pin;
- no torchaudio, Open-Unmix, Dora Search, or NVIDIA distribution was installed;
- installation and imports created no configured Hugging Face model cache.

Demucs 4.1.0 omits NumPy from its Linux package metadata despite importing it at runtime. The explicit NumPy pin is therefore part of this profile, not an optional convenience.

## Install

```bash
bash scripts/install_demucs_linux_cpu.sh
```

The installer refuses unsupported platforms, unsupported Python versions, root execution, an existing destination, and a missing local worker source before creating the runtime.

## Verify manually

```bash
RUNTIME_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/linux-x86_64-cpu-cpython313"
POPEX_DEMUCS_RUNTIME_LOCK="$RUNTIME_DIR/runtime-lock.json" \
  "$RUNTIME_DIR/venv/bin/popex-demucs-worker" \
  --protocol-version 1 runtime-probe
```

Do not substitute `prepare-model` for this verification. Model preparation is a separate explicit-consent workflow.

## Removal

```bash
rm -rf "${XDG_DATA_HOME:-$HOME/.local/share}/popex/runtimes/linux-x86_64-cpu-cpython313"
```

When a custom `POPEX_DEMUCS_LINUX_CPU_DIR` was used, remove that exact directory. Removing the runtime does not remove source media, `analysis.wav`, analysis JSON, prior stems, or a separately managed model cache.

## Known conditions

- The concurrent worker package was not present on base `main` during profile generation, so the actual worker `runtime-probe` must run once that package lands.
- The final installer is intentionally strict and does not update an existing environment in place.
- Any package, Python minor, Torch build, Demucs version, worker protocol, or index change requires regenerating and revalidating this profile.
