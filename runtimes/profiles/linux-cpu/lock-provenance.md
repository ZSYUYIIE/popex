# Linux CPU lock provenance

## Scope

This profile is limited to Linux x86-64, CPython 3.13, and CPU-only PyTorch. It is not a universal lock.

## Official indexes

- PyTorch CPU wheels: `https://download.pytorch.org/whl/cpu`
- Python packages: `https://pypi.org/simple`

Only `torch==2.13.0+cpu` is installed from the PyTorch CPU index. All remaining wheel artifacts are pinned to PyPI identities and SHA-256 hashes. This avoids accepting mirror copies from the PyTorch index without a recorded hash.

## Generation method

The lock was derived on 2026-08-04 using GitHub Actions Ubuntu 24.04.4, Linux x86-64, CPython 3.13.14, and pip 26.1.2:

1. Create a clean virtual environment.
2. Install `torch==2.13.0+cpu` from the official CPU index with `pip --report`.
3. Install `demucs==4.1.0` and an explicit `numpy==2.5.1` from PyPI with `pip --report`.
4. Import `torch`, `demucs`, `demucs.api`, `demucs.hf`, `huggingface_hub`, `safetensors`, `yaml`, `sphn`, `julius`, `lameenc`, `einops`, `tqdm`, and NumPy.
5. Verify `torch.version.cuda is None`, `torch.cuda.is_available() is False`, and that no `nvidia-*`, `torchaudio`, `openunmix`, or `dora-search` distribution is installed.
6. Verify no configured Hugging Face cache directory is created by installation or import.
7. Record wheel filenames and SHA-256 digests from pip reports and official PyPI file metadata.
8. Convert the complete resolved set to exact, hash-enforced, wheel-only locks installed with `--no-deps`.

## Important resolver finding

Demucs 4.1.0 imports NumPy in its runtime audio module but does not declare NumPy on Linux. A clean Demucs install failed the worker-facing import probe with `ModuleNotFoundError: numpy`. The profile therefore adds the explicit `numpy==2.5.1` pin.

## Dependency and artifact evidence

`requirements.lock`, `torch.lock`, `artifacts.json`, `profile.json`, `third-party-inventory.md`, and `worker-runtime-lock.json` were compared as one set. The exact package names and versions agree across the lock files, profile, artifact inventory, human-readable inventory, and the worker's five-package compatibility lock. No dependency version or artifact hash was changed during worker integration validation.

## Worker integration validation

The complete installer was exercised against PR #12 at exact head:

`ef5d0e41a60f44372e14c4685b51a20cc9acd862`

The test fetched that exact pull-request head, extracted only `runtimes/demucs_worker`, installed the hash-locked profile into a new isolated runtime, installed `popex-demucs-worker==1.0.0` from the extracted source with no index, dependencies, or build isolation, and invoked only:

```text
popex-demucs-worker --protocol-version 1 runtime-probe
```

The structured result reported:

```json
{
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
}
```

`installedVersions` equalled `lockedVersions` exactly. The complete environment inventory also matched every package and version in `profile.json`.

Additional verified results:

- ShellCheck 0.9.0 reported no findings for `scripts/install_demucs_linux_cpu.sh`.
- Unsupported OS, unsupported architecture, wrong Python, and root execution returned exit 2 without creating a runtime.
- Missing worker source and an existing destination returned exit 3; the existing destination sentinel was preserved.
- `torch.version.cuda` was `None`.
- `torch.cuda.is_available()` was false.
- No torchaudio, Open-Unmix, Dora Search, CUDA, or `nvidia-*` distribution was installed.
- No `955717e8.safetensors`, `.th`, `.ckpt`, readiness manifest, or Hugging Face model cache was present.
- Demucs's installed wheel contains `demucs/remote/htdemucs.yaml`; this is package metadata, not a downloaded checkpoint or Hugging Face cache asset.
- `prepare-model` was not invoked and no model was downloaded.

For static verification: prepare-model was not invoked.

The evidence was captured in GitHub Actions run 72. Validation applies only to the exact PR #12 head above. A later worker head must be tested separately.

## CI evidence

- Run 62: official CPU wheel and Demucs resolution; exposed the missing NumPy declaration.
- Run 63: clean install with explicit NumPy; worker-facing imports passed; 49 repository tests passed.
- Run 64: captured the complete resolved distribution set and artifact hashes.
- Run 65: captured PyPI hashes for the MarkupSafe and setuptools wheels that the PyTorch index report did not expose.
- Run 66: final static profile suite and repository checks passed.
- Run 72: ShellCheck, refusal behavior, exact PR #12 installer integration, runtime probe, package inventory, CUDA exclusion, and model/cache absence passed before the controlled evidence capture assertion.
