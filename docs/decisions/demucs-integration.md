# Demucs integration decision

- **Status:** accepted
- **Gate result:** approved with conditions
- **Decision date:** 2026-08-03
- **PopEx baseline:** `main` at `312a20e4d923094768e9482d077113c538a1d8d4`
- **Demucs package:** `demucs==4.1.0`, released 2026-07-11
- **Demucs source revision inspected:** `adefossez/demucs` `main` at `eeac1d15891af95b1288d2884b95baa3e5baa96c`
- **Model:** `htdemucs` from `adefossez/HTDemucs`, revision `bf35a81b663819a8255c8fefee17f9d812b786b5`
- **Checkpoint:** `955717e8.safetensors`, SHA-256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`
- **PyTorch compatibility reference:** `torch==2.13.0` metadata and current official install guidance; this is not a universal PopEx pin

## Official-source references

- Canonical project and maintenance statement: https://github.com/adefossez/demucs
- Demucs 4.1.0 release metadata: https://pypi.org/project/demucs/4.1.0/
- Demucs dependency metadata: https://github.com/adefossez/demucs/blob/main/pyproject.toml
- Demucs source license: https://github.com/adefossez/demucs/blob/main/LICENSE
- Pretrained model resolution: https://github.com/adefossez/demucs/blob/main/demucs/pretrained.py
- Hugging Face loader: https://github.com/adefossez/demucs/blob/main/demucs/hf.py
- Legacy checkpoint loader: https://github.com/adefossez/demucs/blob/main/demucs/repo.py
- Public API and audio fallback: https://github.com/adefossez/demucs/blob/main/demucs/api.py
- CLI behavior and device selection: https://github.com/adefossez/demucs/blob/main/demucs/separate.py
- Windows support: https://github.com/adefossez/demucs/blob/main/docs/windows.md
- Linux support: https://github.com/adefossez/demucs/blob/main/docs/linux.md
- macOS support: https://github.com/adefossez/demucs/blob/main/docs/mac.md
- Official model card: https://huggingface.co/adefossez/HTDemucs
- Exact model revision: https://huggingface.co/adefossez/HTDemucs/tree/bf35a81b663819a8255c8fefee17f9d812b786b5
- Hugging Face cache behavior: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Hugging Face cache, privacy, telemetry, and offline variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- PyTorch install profiles: https://pytorch.org/get-started/locally/
- PyTorch 2.13.0 metadata: https://pypi.org/project/torch/2.13.0/
- PyTorch source license: https://github.com/pytorch/pytorch/blob/main/LICENSE
- TorchAudio source license: https://github.com/pytorch/audio/blob/main/LICENSE
- TorchAudio binary compatibility rule: https://docs.pytorch.org/audio/stable/installation.html
- MUSDB18 provenance and restrictions: https://sigsep.github.io/datasets/musdb.html
- MUSDB18-HQ record and agreement: https://zenodo.org/records/3338373
- HTDemucs paper: https://arxiv.org/abs/2211.08553

## Decision

PopEx may integrate Demucs 4.1.0 and the official `htdemucs` checkpoint as an **optional, local-only source-separation capability**, with these conditions:

1. The base PopEx installation and normal web application remain usable without Demucs, PyTorch, TorchAudio, CUDA, a GPU, model weights, or model-host access.
2. Demucs lives in an explicitly selected separation runtime profile, not default dependencies.
3. No weights are committed, bundled in the base wheel, embedded in Docker, attached to ordinary releases, or downloaded at application startup.
4. The first explicit separation or model-preparation action may initiate an explained download. Later runs reuse the cache.
5. CPU remains supported. CUDA and Apple MPS are optional accelerators.
6. Private audio and generated stems remain local; the only external request is for public model files.
7. Separation failures never invalidate successful source preparation or audio analysis.
8. The integration requests the reviewed Hugging Face model explicitly, preferably `hf://adefossez/htdemucs`, and does not silently fall back to legacy AWS `.th` checkpoints.

This decision does not approve cloud separation, arbitrary model repositories, training dependencies, or model/dataset redistribution.

## Canonical upstream and maintenance status

The canonical project is `adefossez/demucs`. The former `facebookresearch/demucs` repository is archived and points to that fork.

The current project calls itself officially maintained but says active feature development has stopped, replies may be slow, and no new features are expected. PopEx must treat it as a low-activity maintained dependency: pin the exact release, isolate it behind an adapter, and avoid assuming rapid upstream fixes.

## Source-code licensing

Demucs 4.1.0 source is MIT-licensed, copyright Meta Platforms, Inc. and affiliates. The license permits use, modification, publication, distribution, sublicensing, sale, and commercial use, provided the copyright and permission notice accompany copies or substantial portions.

Installing package source and redistributing package source are separate actions:

- A user may install `demucs==4.1.0` into a local optional runtime.
- A PopEx release that redistributes Demucs must include the exact Demucs MIT text and copyright notice.
- PopEx's own MIT license does not replace Demucs's notice.

PyTorch has a BSD-style top-level source license and binary builds contain separately licensed components. Any distributor must preserve the exact license and notice inventory from the selected wheel/native build rather than labeling the whole artifact only as BSD.

TorchAudio is BSD-2-Clause, but Demucs 4.1.0 lists it only in the training extra. Pretrained separation does not require TorchAudio, so PopEx must not add it merely because older Demucs packaging used it. If a future feature needs it, TorchAudio must exactly match the installed PyTorch release.

## Model-weight licensing

The official `adefossez/HTDemucs` repository declares `htdemucs` weights MIT-licensed. The reviewed repository contains one approximately 84 MB safetensors checkpoint, a bag YAML, a model card, and training metadata.

For the evaluated release:

- Demucs source license: MIT.
- `htdemucs` weight license: MIT as declared by the official model repository.
- The source and weight licenses therefore do not differ, but they remain separate artifacts requiring separate records.
- No commercial-use restriction is stated for the released checkpoint.

The model repository currently exposes MIT metadata and a model card rather than a standalone full `LICENSE` file. Runtime download is approved. Redistributing the weights in a PopEx release is not approved by this decision. A later bundling review must capture the exact revision, checkpoint hash, model card, applicable MIT notice, and attribution.

## Dataset and provenance notes

Official Demucs documentation and the HTDemucs paper state that the model was trained on MUSDB/MUSDB-HQ plus an additional collection of 800 songs.

MUSDB18 and MUSDB18-HQ are copyrighted recording datasets, not MIT-licensed audio assets. Their official records limit the material to educational use and prohibit commercial use without permission from the applicable copyright holders. Constituent tracks have multiple provenance paths, including Creative Commons non-commercial share-alike material.

The reviewed Demucs sources do not publish a track-level identity or license inventory for the extra 800 songs. Exported training metadata names internal dataset paths but does not establish redistribution rights.

Therefore:

- PopEx must not redistribute MUSDB, MUSDB-HQ, the extra collection, or excerpts.
- PopEx tests must use synthetic or clearly redistributable audio.
- PopEx must not describe the training datasets as MIT-licensed.
- The model repository's MIT declaration is the stated license for the released checkpoint, while the incomplete 800-song inventory remains a provenance risk for any future public/commercial model bundle.

## Redistribution policy

| Artifact or action | Decision |
| --- | --- |
| Install `demucs==4.1.0` in a user-selected local runtime | Approved |
| Redistribute Demucs package files | Permitted under MIT if the required notice is preserved |
| Install a selected PyTorch CPU/CUDA/MPS build | Approved only through an exact tested runtime profile |
| Redistribute PyTorch | Permitted only with that build's full license/notice inventory |
| Download `htdemucs` after explicit user action | Approved |
| Reuse cached `htdemucs` offline | Approved |
| Download weights during application startup | Rejected |
| Bundle weights in Git, base packages, Docker, installers, or ordinary releases | Rejected |
| Redistribute training datasets | Rejected |
| Upload private audio for separation | Rejected |

## Installation strategy

`pyproject.toml` remains unchanged. No `[project.optional-dependencies].separation` group is added in this cycle.

A universal declaration such as `separation = ["demucs==4.1.0"]` is not defensible yet:

1. Demucs requires `torch>=2.1` on most platforms, but the correct PyTorch distribution depends on operating system and accelerator.
2. Linux's default PyTorch package can include CUDA support and large NVIDIA dependencies. A CPU-only install uses the official CPU index, which a normal PEP 621 dependency cannot safely select.
3. Windows upstream documents CPU as default and requires an explicit CUDA index for NVIDIA acceleration.
4. Apple Silicon uses the standard supported build with MPS.
5. Intel macOS requires `torch>=2.1,<2.3`, `numpy<2`, and Python no newer than 3.12. PopEx requires `numpy>=2,<3`, so a shared environment has an empty NumPy version intersection.
6. An unbounded PyTorch floor would expose PopEx to untested future runtime and packaging changes.

Recommended packaging:

1. Create a separate replaceable separation environment/profile.
2. Install one exact tested PyTorch build from the official CPU or accelerator-specific index.
3. Install `demucs==4.1.0` after PyTorch is present.
4. Exclude TorchAudio and all training extras.
5. Invoke the profile through a narrow adapter or subprocess boundary.

This permits independent Windows CPU, Linux CPU, CUDA, and Apple Silicon MPS profiles without pretending they share one portable wheel set. Intel macOS may be considered later only as an isolated Python 3.12/NumPy 1.x worker, not by lowering PopEx's global requirements.

Exact pins are needed for each supported runtime profile. `demucs==4.1.0` is the approved Demucs pin. This audit intentionally does not choose a universal PyTorch pin; the integration cycle must test and lock one exact build per profile.

## Model-download strategy

Importing Demucs does not itself fetch weights. Constructing `demucs.api.Separator`, or invoking the CLI with a track, loads the selected pretrained model and can download it when the cache is empty. The integration must not construct a separator during startup, health checks, capability detection, job listing, or module import.

On the first explicit user action PopEx should:

1. Detect whether the optional runtime exists.
2. Check free space.
3. Explain the official model source, MIT declaration, approximate 84 MB checkpoint size, cache destination, and network requirement.
4. Explain that PyTorch and generated stems require substantially more storage than the checkpoint.
5. Continue only from the user's explicit separation/model-download action.
6. Configure cache and privacy variables before importing `huggingface_hub`.
7. Request `hf://adefossez/htdemucs` or an equivalently constrained official safetensors path.
8. Record Demucs version, model name, resolved repository revision, checkpoint hash, runtime profile, and device.

The forced Hugging Face path is a security condition. Generic `htdemucs` lookup catches Hugging Face failures and falls back to a legacy remote repository. That fallback calls `torch.hub.load_state_dict_from_url(..., weights_only=False)`. PopEx should load the reviewed safetensors checkpoint and must not silently broaden its source or deserialization format.

## Cache and storage behavior

Demucs 4.1.0 calls `huggingface_hub.hf_hub_download` for the model YAML and checkpoint. The default Hub cache is `~/.cache/huggingface/hub`; `HF_HOME` and `HF_HUB_CACHE` can relocate it. These variables are read at import time.

PopEx should use an identified application-data cache, show its location and size, and provide a safe removal action. Set before importing the Hub library:

- `HF_HUB_CACHE` or `HF_HOME` to the PopEx model-cache directory;
- `HF_HUB_DISABLE_TELEMETRY=1` to avoid adding telemetry;
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so unrelated user credentials are not sent for a public model;
- `HF_HUB_OFFLINE=1` when the user selects offline mode and the required model is already cached.

Expected footprint:

- Demucs wheel: approximately 100.6 kB; source archive: approximately 1.2 MB.
- `htdemucs` checkpoint: approximately 84 MB plus small metadata and cache bookkeeping.
- Reference PyTorch 2.13.0 CPython 3.13 downloads: about 526.6 MB on Linux x86-64, 122.1 MB on Windows x86-64, and 111.2 MB on Apple Silicon, before installation and transitive/native overhead.
- Hugging Face retains revisions until removed; Windows without symlink support can consume extra space.
- Four 44.1 kHz stereo 16-bit WAV stems consume roughly 40 MiB per minute of source audio.

The UI must not present 84 MB as the total feature footprint; PyTorch, native dependencies, cache duplication, and output stems dominate storage.

## Supported runtime matrix

| Platform | Python | CPU | Acceleration | Decision and risk |
| --- | --- | --- | --- | --- |
| Windows x86-64 | 3.10-3.13 | Supported | CUDA through an explicit official index | Approved profile after exact-version testing; 32-bit is unsupported upstream |
| Linux x86-64 | 3.10-3.13 | Supported | CUDA with a matching official build and driver | Approved CPU/CUDA profiles after testing; CPU profile must use the CPU index |
| Linux ARM64 | 3.10-3.13 where wheels exist | Supported in current metadata | Hardware-specific acceleration requires separate validation | Proposed profile; do not claim general support before testing |
| macOS Apple Silicon | 3.10-3.13 | Supported | MPS/Metal officially supported | Approved profile after testing; API integration must explicitly select MPS |
| macOS Intel | At most 3.12 under upstream constraints | CPU upstream | No current MPS path | Blocked in the shared PopEx environment by NumPy constraints; isolated profile only |
| 32-bit systems | Unsupported | Unsupported | Unsupported | Rejected |

Demucs CLI device selection is CUDA, then MPS, then CPU. `demucs.api.Separator` defaults to CUDA when available and otherwise CPU, omitting MPS from its default. A PopEx API caller must explicitly select `mps` when available and expose a CPU override.

## Python-version compatibility

Demucs 4.1.0 declares Python 3.10 or newer, matching PopEx's global floor. Its universal wheel has no upper Python bound. Current PyTorch 2.13.0 publishes CPython 3.13 wheels for Windows, Linux x86-64, Linux ARM64, and Apple Silicon, so wheel absence does not itself block PopEx's Python 3.13 Docker and CI baseline on those platforms.

That is metadata compatibility, not end-to-end validation. Base Docker and CI must continue installing only normal PopEx dependencies. Separation needs an independent runtime matrix.

Intel macOS is the material exception: `torch<2.3` and `numpy<2` conflict with PopEx's `numpy>=2`. Do not reduce PopEx's Python or NumPy declarations to accommodate it.

## CPU, CUDA, and Apple MPS behavior

CPU is an official path and remains mandatory even though it is slower.

CUDA comes from the selected PyTorch build. An NVIDIA driver does not prove that the environment has a compatible CUDA wheel. PopEx needs device preflight, recorded device selection, an actionable mismatch message, and an explicit CPU retry.

Apple Silicon MPS is officially supported through the Demucs CLI. PopEx must explicitly request MPS when `torch.backends.mps.is_available()` and offer CPU fallback for runtime/operator failures.

TorchAudio is not needed for acceleration or pretrained separation.

## Native libraries and FFmpeg

Demucs uses `sphn` for common audio decoding and falls back to FFmpeg for formats `sphn` cannot read. FFmpeg is also required for FLAC output. PyTorch, `sphn`, and related packages contain native binaries even when no additional host library is required.

PopEx already prepares 44.1 kHz PCM `analysis.wav` and already requires FFmpeg for source preparation. Passing this WAV into the Demucs API avoids a new mandatory input-format dependency. Training-only SoundTouch, MUSDB tooling, and TorchAudio remain excluded.

## Failure and offline behavior

After an explicit complete download, cached separation can operate offline. `HF_HUB_OFFLINE=1` prevents HTTP checks and restricts the Hub client to cached files. A missing cache entry must produce a clear offline-cache error.

Required behavior:

- Missing optional runtime: explain the correct platform profile; retain prepared source and analysis.
- Download/network failure: keep the job retryable and retain all prior artifacts.
- Interrupted or partial download: do not mark the model ready until expected files and hashes verify.
- Hash/revision mismatch: reject and request a clean explicit download.
- Low disk: fail before loading/output where possible and report model plus stem requirements.
- CUDA/MPS error: offer a deliberate CPU retry and record the selected device.
- Offline cache hit: make no network request.
- Offline cache miss: fail before audio processing without damaging job state.

## Security and privacy considerations

- Audio processing is local; only model files are fetched.
- Disable Hub telemetry and implicit token sending.
- Use the official repository, expected model name, recorded revision, and hash.
- Prefer safetensors and prevent legacy pickle-capable fallback.
- Treat model files as supply-chain inputs and verify before loading.
- Do not expose arbitrary audio/cache filesystem paths through the web API.
- Do not accept arbitrary Hugging Face repositories or local checkpoint files in the first integration.
- Preserve existing local controls for private source audio, stems, and analysis.

## Packaging alternatives considered

### Default Demucs/PyTorch dependencies

Rejected. They would make a large native ML runtime mandatory and violate the lightweight base product.

### A simple `separation = ["demucs==4.1.0"]` extra

Deferred. It cannot select CPU/CUDA/MPS indexes safely, leaves PyTorch unbounded, and is unsatisfiable with PopEx NumPy on Intel macOS.

### Markers that omit Intel macOS

Rejected for now. A silent no-op on one PopEx platform is not a complete extra and does not solve CPU-versus-CUDA selection.

### Separate runtime profiles or worker environment

Accepted and recommended.

### Bundled `htdemucs` weights

Rejected. Explicit runtime download and cache reuse meet the product need without burdening every release.

### Generic model name with legacy fallback

Rejected. Constrain the source to the reviewed official safetensors repository.

### `htdemucs_ft`

Deferred. It is a four-model bag with roughly four times the weight storage and inference work. Single-model `htdemucs` is the approved baseline.

### Cloud separation

Rejected. It would add audio transfer, privacy risk, operating cost, and service dependency.

## Explicit unresolved questions

1. What are the exact identities and licenses for the extra 800 training songs?
2. Will the official model repository add a standalone weight license/copyright file?
3. Which exact PyTorch release should be locked for each CPU, CUDA, and MPS profile?
4. Should Intel macOS receive an isolated Python 3.12/NumPy 1.x worker or be unsupported for separation?
5. How should PopEx pin a Hub revision when Demucs's high-level loader does not expose a revision argument?
6. What cache lifecycle and maximum-size policy should PopEx expose?
7. Which CUDA versions and minimum drivers will be supported after hardware testing?
8. Will any future release redistribute weights? That requires a new model, provenance, notice, storage, and security review.

## Recommendation for the integration agent

- Target `demucs==4.1.0` and `hf://adefossez/htdemucs`.
- Implement a narrow optional-runtime adapter; do not change base dependencies.
- Instantiate the separator only inside an explicit separation job.
- Configure a PopEx-owned Hub cache before imports.
- Disable telemetry and implicit token sending.
- Record package/model versions, revision, hash, runtime profile, and device.
- Support CPU first, with optional tested CUDA and MPS profiles.
- Pass the existing prepared `analysis.wav` to Demucs.
- Produce `drums`, `bass`, `other`, and `vocals` stems without claiming perfect isolation.
- Preserve source preparation and audio analysis across every separation failure.
- Implement explained first-use download, offline cache, retry, interruption, and low-disk handling.
- Exclude TorchAudio, training dependencies, arbitrary repositories, legacy AWS fallback, bundled weights, and startup downloads.

**APPROVED WITH CONDITIONS**
