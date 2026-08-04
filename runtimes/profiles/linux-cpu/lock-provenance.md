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

## CI evidence

- Run 62: official CPU wheel and Demucs resolution; exposed the missing NumPy declaration.
- Run 63: clean install with explicit NumPy; worker-facing imports passed; 49 repository tests passed.
- Run 64: captured the complete resolved distribution set and artifact hashes.
- Run 65: captured PyPI hashes for the MarkupSafe and setuptools wheels that the PyTorch index report did not expose.

The final local-worker `runtime-probe` remains conditional on `runtimes/demucs_worker` being present. The installer refuses to create a runtime until that directory exists.
