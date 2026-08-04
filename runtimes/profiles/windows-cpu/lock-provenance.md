# Windows CPU lock provenance

## Scope

This profile is limited to Windows x86-64, CPython 3.13, and CPU-only PyTorch. It is not a universal Windows, CUDA, ARM64, or other-Python lock.

## Official indexes

- PyTorch CPU wheels: `https://download.pytorch.org/whl/cpu`
- Python packages: `https://pypi.org/simple`

Only `torch==2.13.0+cpu` is selected from the PyTorch CPU index. Every other artifact is selected from PyPI and recorded with its exact Windows-compatible filename and SHA-256.

## Resolver evidence

GitHub Actions workflow **Demucs Windows CPU profile validation**, run 1 (`30883764913`), used:

- Microsoft Windows Server 2025 Datacenter, build `10.0.26100`;
- runner image `windows-2025-vs2026`, version `20260728.188.1`;
- AMD64;
- CPython `3.13.14`;
- wheel-only resolution from official indexes.

The run downloaded every selected wheel and recomputed its SHA-256. Evidence artifact `8882280475` contained the pip report, exact URLs, filenames, reported digests, and downloaded digests.

## Platform-specific findings

- Official Torch artifact: `torch-2.13.0+cpu-cp313-cp313-win_amd64.whl`, SHA-256 `a17ff48608634db245e17e8bb00a9558554a49aeb1e4f5fe6cd039af2a10515b`.
- Windows adds `colorama==0.4.6` through tqdm.
- Windows-specific wheels and hashes were selected for NumPy, PyYAML, safetensors, sphn, lameenc, hf-xet, and MarkupSafe.
- Linux binary-wheel hashes were not reused for Windows artifacts.
- All selected artifacts were wheels; no source distribution was accepted.

## Remaining validation

The candidate lock still requires the committed PowerShell installer, static suite, clean Windows installation, CPU and forbidden-package checks, model/cache absence checks, and an actual protocol-v1 `runtime-probe`. Until those pass, `profile.json` reports `candidate-awaiting-clean-installer-smoke`.

No model command was invoked during resolver run 1, and no model checkpoint or cache was included in the evidence artifact.
