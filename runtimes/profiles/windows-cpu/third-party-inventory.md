# Windows CPU profile third-party inventory

This inventory covers exactly the index-fetched distributions in `requirements.lock` and `torch.lock`. The local `popex-demucs-worker` package is PopEx source under the repository license. Release packaging must preserve the exact license files and notices shipped with every selected wheel.

| Component | Version | License summary | Origin | Runtime purpose |
|---|---:|---|---|---|
| torch | 2.13.0+cpu | BSD-style plus bundled notices | PyTorch CPU index | CPU tensor runtime |
| demucs | 4.1.0 | MIT | PyPI | Source separation implementation |
| numpy | 2.5.1 | BSD-3-Clause plus bundled notices | PyPI | Demucs numerical runtime |
| huggingface-hub | 1.26.0 | Apache-2.0 | PyPI | Pinned model acquisition/cache API; no model fetched at install |
| safetensors | 0.8.0 | Apache-2.0 | PyPI | Checkpoint parsing |
| PyYAML | 6.0.3 | MIT | PyPI | Demucs bag metadata |
| sphn | 0.2.1 | MIT | PyPI | Local audio decoding |
| julius | 0.2.8 | MIT | PyPI | Signal resampling/filtering |
| lameenc | 1.8.4 | MIT | PyPI | Demucs MP3 output support |
| einops | 0.8.2 | MIT | PyPI | Tensor rearrangement |
| tqdm | 4.70.0 | MPL-2.0 and MIT | PyPI | Progress utilities |
| colorama | 0.4.6 | BSD-3-Clause | PyPI | Windows terminal compatibility for tqdm |
| anyio | 4.14.2 | MIT | PyPI | HTTPX concurrency dependency |
| certifi | 2026.7.22 | MPL-2.0 | PyPI | TLS CA bundle |
| click | 8.4.2 | BSD-3-Clause | PyPI | Hugging Face runtime dependency |
| filelock | 3.29.0 | Unlicense | PyPI | Torch file locking |
| fsspec | 2026.4.0 | BSD-3-Clause | PyPI | Torch filesystem abstraction |
| h11 | 0.16.0 | MIT | PyPI | HTTP protocol |
| hf-xet | 1.6.0 | Apache-2.0 | PyPI | Hugging Face transport dependency |
| httpcore | 1.0.9 | BSD-3-Clause | PyPI | HTTPX transport |
| httpx | 0.28.1 | BSD-3-Clause | PyPI | Hugging Face HTTP client |
| idna | 3.18 | BSD-3-Clause | PyPI | Internationalized domain names |
| Jinja2 | 3.1.6 | BSD-3-Clause | PyPI | Torch dependency |
| MarkupSafe | 3.0.3 | BSD-3-Clause | PyPI | Jinja2 dependency |
| mpmath | 1.3.0 | BSD-3-Clause | PyPI | SymPy dependency |
| networkx | 3.6.1 | BSD-3-Clause | PyPI | Torch graph utilities |
| packaging | 26.2 | Apache-2.0 or BSD-2-Clause | PyPI | Version and tag handling |
| setuptools | 78.1.0 | MIT | PyPI | Torch runtime and local worker build backend |
| sympy | 1.14.0 | BSD-3-Clause | PyPI | Torch symbolic math dependency |
| typing-extensions | 4.15.0 | PSF-2.0 | PyPI | Typing backports |

## Evaluated but excluded

| Component | Evaluated version | Status | Reason |
|---|---:|---|---|
| torchaudio | None selected | Not installed | Not required by Demucs 4.1.0 pretrained separation. |
| openunmix | 1.3.0 | Not installed | Not a Demucs 4.1.0 runtime dependency and introduces torchaudio. |
| dora-search | 0.1.12 | Not installed | Demucs training-only dependency. |
| Demucs training extra | 4.1.0 `[train]` | Not installed | Training is outside the local separation runtime. |
| NVIDIA/CUDA packages | None | Not installed | This is a CPU-only profile. |

## Models

No model weights, model URLs, readiness manifests, or Hugging Face model cache assets are included. The audited model is prepared only through a later explicit worker action.
