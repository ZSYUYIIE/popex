# Demucs Linux CPU runtime

## Status

The Linux x86-64 CPython 3.13 runtime profile is validated for worker integration against `popex-demucs-worker` PR #12 head `ef5d0e41a60f44372e14c4685b51a20cc9acd862`.

This statement does not cover any later untested worker head.

## Profile identity

- Runtime profile: `linux-x86_64-cpu-cpython313`
- Operating system: Linux
- Architecture: x86-64
- Python: CPython `>=3.13.0,<3.14.0`
- Device: CPU
- PyTorch: `2.13.0+cpu`
- Demucs: `4.1.0`
- Worker protocol: `1`
- Worker: `popex-demucs-worker==1.0.0`
- Worker source tested: PR #12 at `ef5d0e41a60f44372e14c4685b51a20cc9acd862`

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
- model checkpoints and readiness manifests

Open-Unmix 1.3.0 was evaluated only as an exclusion. It is not required by Demucs 4.1.0 and its inference stack uses torchaudio. Dora Search 0.1.12 is a Demucs training dependency and is not installed.

## Verified integration evidence

The full installer ran on GitHub Actions Ubuntu 24.04.4 with CPython 3.13.14 against the exact worker head above.

ShellCheck 0.9.0 reported no findings. The installer correctly refused:

- unsupported OS with exit 2;
- unsupported architecture with exit 2;
- wrong Python with exit 2;
- root execution with exit 2;
- an existing destination with exit 3 while preserving its sentinel file;
- missing worker source with exit 3.

The successful isolated installation returned this complete worker envelope:

```json
{
  "command": "runtime-probe",
  "protocolVersion": 1,
  "result": {
    "compatible": true,
    "installedVersions": {
      "PyYAML": "6.0.3",
      "demucs": "4.1.0",
      "huggingface_hub": "1.26.0",
      "safetensors": "0.8.0",
      "torch": "2.13.0+cpu"
    },
    "lockedVersions": {
      "PyYAML": "6.0.3",
      "demucs": "4.1.0",
      "huggingface_hub": "1.26.0",
      "safetensors": "0.8.0",
      "torch": "2.13.0+cpu"
    },
    "pythonVersion": "3.13.14",
    "runtimeLockSource": "profile",
    "runtimeProfile": "linux-x86_64-cpu-cpython313",
    "workerVersion": "1.0.0"
  },
  "status": "ok",
  "warnings": []
}
```

`installedVersions` and `lockedVersions` were identical. The complete installed environment also matched every exact package version in `profile.json`.

Additional results:

- `torch.version.cuda` was `None`;
- `torch.cuda.is_available()` was false;
- no torchaudio, Open-Unmix, Dora Search, CUDA, or NVIDIA package was installed;
- no audited checkpoint, `.th`, `.ckpt`, readiness manifest, or Hugging Face model cache existed;
- no model was downloaded and `prepare-model` was never run.

The Demucs wheel itself includes `demucs/remote/htdemucs.yaml`. That small YAML file is bundled package metadata; it is not the audited model checkpoint and is not evidence of a Hugging Face model download.

## Install

```bash
bash scripts/install_demucs_linux_cpu.sh
```

The installer remains strict: it refuses unsupported platforms, unsupported Python versions, root execution, an existing destination, and a missing local worker source before modifying the runtime.

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

## Revalidation boundary

The profile is ready for integration with the exact worker head tested. Any package version, hash, Python minor, Torch build, Demucs version, worker protocol, worker implementation head, or package-index change requires a new clean installation and runtime-probe validation.
