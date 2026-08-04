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

## Clean installer and worker validation

GitHub Actions workflow run 14 (`30884730743`) exercised the committed installer from a new destination on the same Windows image and Python minor.

Verified results:

- PowerShell parser reported no errors.
- `22` permanent profile tests passed.
- Both index installation phases enforced exact hashes, wheel-only selection, and `--no-deps`.
- The complete installed inventory contained exactly `31` distributions excluding pip and matched `profile.json`.
- `torch.__version__` was `2.13.0+cpu`.
- `torch.version.cuda` was `None` and `torch.cuda.is_available()` was false.
- No torchaudio, Open-Unmix, Dora Search, CUDA, or NVIDIA distribution was installed.
- No `.safetensors`, `.th`, `.ckpt`, readiness manifest, or Hugging Face model cache was created.
- The local `popex-demucs-worker==1.0.0` source installed last with no index, dependencies, or build isolation.
- The worker returned a successful protocol-v1 `runtime-probe` with profile `windows-x86_64-cpu-cpython313` and identical installed and locked versions:

```json
{
  "demucs": "4.1.0",
  "torch": "2.13.0+cpu",
  "huggingface_hub": "1.26.0",
  "safetensors": "0.8.0",
  "PyYAML": "6.0.3"
}
```

The workflow deleted the temporary runtime after validation. No model command was invoked during resolution or installation.

## Revalidation boundary

Any package version, artifact hash, Python minor, Torch build, Demucs version, worker package/protocol, Windows runner image, installer behavior, or package-index origin change requires a new clean Windows installation and worker probe.
