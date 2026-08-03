# Demucs integration decision

- **Status:** accepted with conditions
- **Decision date:** 2026-08-03
- **PopEx baseline evaluated:** `main` at `312a20e4d923094768e9482d077113c538a1d8d4`
- **Demucs package evaluated:** `demucs==4.1.0`, released 2026-07-11
- **Demucs source revision inspected:** `adefossez/demucs` `main` at `eeac1d15891af95b1288d2884b95baa3e5baa96c`
- **Model evaluated:** `htdemucs` from `adefossez/HTDemucs`, repository revision `bf35a81b663819a8255c8fefee17f9d812b786b5`
- **Model file evaluated:** `955717e8.safetensors`, SHA-256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`
- **PyTorch compatibility reference:** `torch==2.13.0` release metadata and current official installation guidance; this is not an approved universal PopEx pin

## Official-source references

Primary sources used for this decision:

- Demucs canonical repository and maintenance statement: https://github.com/adefossez/demucs
- Demucs 4.1.0 package metadata and release files: https://pypi.org/project/demucs/4.1.0/
- Demucs dependency metadata: https://github.com/adefossez/demucs/blob/main/pyproject.toml
- Demucs source license: https://github.com/adefossez/demucs/blob/main/LICENSE
- Demucs pretrained-model resolution: https://github.com/adefossez/demucs/blob/main/demucs/pretrained.py
- Demucs Hugging Face loader: https://github.com/adefossez/demucs/blob/main/demucs/hf.py
- Demucs public API and audio fallback behavior: https://github.com/adefossez/demucs/blob/main/demucs/api.py
- Demucs command-line device selection: https://github.com/adefossez/demucs/blob/main/demucs/separate.py
- Windows support: https://github.com/adefossez/demucs/blob/main/docs/windows.md
- Linux support: https://github.com/adefossez/demucs/blob/main/docs/linux.md
- macOS support: https://github.com/adefossez/demucs/blob/main/docs/mac.md
- Official `htdemucs` model repository and model card: https://huggingface.co/adefossez/HTDemucs
- Exact model files and revision: https://huggingface.co/adefossez/HTDemucs/tree/bf35a81b663819a8255c8fefee17f9d812b786b5
- Hugging Face cache behavior: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Hugging Face environment and offline controls: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables
- PyTorch installation profiles: https://pytorch.org/get-started/locally/
- PyTorch 2.13.0 package metadata: https://pypi.org/project/torch/2.13.0/
- PyTorch source license: https://github.com/pytorch/pytorch/blob/main/LICENSE
- TorchAudio source license: https://github.com/pytorch/audio/blob/main/LICENSE
- TorchAudio binary compatibility rule: https://docs.pytorch.org/audio/stable/installation.html
- MUSDB18 provenance and restrictions: https://sigsep.github.io/datasets/musdb.html
- MUSDB18-HQ record and license agreement: https://zenodo.org/records/3338373
- HTDemucs paper: https://arxiv.org/abs/2211.08553

## Decision

PopEx may integrate Demucs 4.1.0 with `htdemucs` as an **optional, local-only separation capability**, subject to all conditions in this record.

Approved product behavior:

1. The normal PopEx installation and web application remain fully usable without Demucs, PyTorch, TorchAudio, CUDA, a GPU, model weights, or network access to a model host.
2. Demucs is installed only into an explicitly selected separation runtime profile.
3. No model weights are committed to Git, copied into the base Python package, embedded in Docker images, or attached to ordinary PopEx releases.
4. No model is downloaded when PopEx starts, when modules are imported, or when an unrelated job is opened.
5. The first explicit user request to prepare or run separation may download the model only after PopEx explains the source, approximate size, license, cache location, and network requirement.
6. Later runs reuse the local model cache.
7. CPU operation remains supported. CUDA and Apple Metal acceleration remain optional.
8. Source audio and generated stems remain local. Model acquisition must not upload audio.
9. Missing dependencies, an unavailable model, insufficient storage, or an offline cache miss must fail only the separation stage and must not destroy prepared source audio or completed analysis.
10. The integration must use the official Hugging Face model path explicitly, preferably `hf://adefossez/htdemucs`, rather than relying on Demucs's generic-name fallback to the legacy AWS checkpoint repository.

This decision does **not** approve a default dependency, a bundled model, a model download during application startup, a generic cloud separation service, or redistribution of training datasets.

## Canonical upstream and maintenance status

The canonical maintained repository is `adefossez/demucs`. The former `facebookresearch/demucs` repository is archived and directs users to the `adefossez` fork.

The canonical repository describes itself as officially maintained, while also stating that active feature development has stopped and that replies may be slow. PopEx should therefore treat Demucs as a low-activity maintained dependency: pin the exact Demucs release used, retain an adapter boundary, and do not assume rapid upstream fixes.

## Source-code licensing

Demucs 4.1.0 source code is licensed under the MIT License, copyright Meta Platforms, Inc. and affiliates. The license permits use, modification, publication, distribution, sublicensing, sale, and commercial use, provided that the copyright and permission notice are included in copies or substantial portions.

The PyPI package is redistributable under those terms. A PopEx distributor that bundles the Demucs package must include the Demucs MIT license and copyright notice in the release's third-party notices or license directory.

PyTorch has a BSD-style top-level source license and its binary distributions contain components under additional licenses. A distributor must preserve the exact license and notice inventory shipped with the selected PyTorch build rather than reducing the entire wheel to a single label.

TorchAudio's source is BSD-2-Clause, but TorchAudio is **not a Demucs 4.1.0 runtime dependency for pretrained separation**. It appears only in Demucs's training extra and must not be added to the PopEx separation runtime without a separate need and exact torch/torchaudio version pairing.

## Model-weight licensing

The official `adefossez/HTDemucs` model repository declares the `htdemucs` weights to be MIT-licensed. The repository contains one 84 MB safetensors checkpoint, a small bag-definition YAML file, a model card, and training metadata.

For the exact version evaluated, the Demucs source license and the `htdemucs` weight license are both MIT. They are still separate artifacts and must be recorded separately. PopEx's MIT license does not automatically apply to the model.

No commercial-use restriction is stated in the official model repository's MIT declaration. The model repository currently exposes license metadata and a model card rather than a separate full `LICENSE` file. Runtime download is approved. Bundling the weights in a PopEx release remains disallowed by product policy unless a later release audit captures the exact model revision, model card, file hash, applicable MIT notice, and attribution in that release.

## Dataset and provenance notes

The Demucs documentation and HTDemucs paper state that `htdemucs` was trained on MUSDB/MUSDB-HQ plus an additional collection of 800 songs.

MUSDB18 and MUSDB18-HQ contain copyrighted multitrack recordings with track-specific provenance. The official dataset record limits the material to educational purposes and says it must not be used commercially without permission from the relevant copyright holders. Some constituent tracks also carry Creative Commons non-commercial share-alike terms.

The official Demucs materials inspected do not identify the 800-song collection track by track, publish its license inventory, or establish redistribution rights for that training data. The model's exported training metadata names internal dataset paths but does not resolve this provenance question.

These dataset restrictions mean:

- PopEx must not redistribute MUSDB, MUSDB-HQ, the unidentified 800-song collection, or excerpts from them.
- PopEx tests must continue to use synthetic or clearly redistributable audio.
- The official model repository's MIT declaration is the stated license for the released weights, but the incomplete training-data inventory remains a provenance risk that should be disclosed and revisited before any public or commercial model bundle is shipped.
- PopEx must not describe the training datasets as MIT-licensed.

## Redistribution policy

The following cases are deliberately separate:

| Artifact or action | Decision |
| --- | --- |
| Install `demucs==4.1.0` source/wheel into a user-selected local environment | Approved, with Demucs MIT notice preserved if redistributed |
| Install a platform-appropriate PyTorch build | Approved only through a documented CPU, CUDA, or MPS runtime profile with the selected build's notices |
| Download `htdemucs` weights after explicit user action | Approved |
| Reuse already cached weights offline | Approved |
| Download weights during PopEx application startup | Rejected |
| Bundle weights in Git, the base wheel, Docker image, installer, or ordinary release | Rejected under this decision |
| Redistribute MUSDB/MUSDB-HQ or the extra 800-song training collection | Rejected |
| Send private audio to Hugging Face or another service | Rejected; the Hub request is for model files only |

## Installation strategy

`pyproject.toml` remains unchanged. No `[project.optional-dependencies].separation` declaration is added in this cycle.

A single cross-platform extra is not currently defensible:

1. Demucs 4.1.0 requires `torch>=2.1` for most platforms, but PyTorch distribution selection is accelerator- and index-specific.
2. On Linux, the default PyTorch package can include CUDA support and large NVIDIA dependencies. A CPU-only installation requires the official CPU package index, which a normal PEP 621 dependency entry cannot select safely.
3. On Windows, upstream documents CPU as the default and requires an explicit CUDA index for NVIDIA acceleration.
4. On Apple Silicon, the normal supported build provides MPS acceleration.
5. On Intel macOS, Demucs requires `torch>=2.1,<2.3`, `numpy<2`, and Python at most 3.12. PopEx currently declares `numpy>=2,<3`, so installing Demucs into the same PopEx environment is unsatisfiable there.
6. An unbounded `torch>=2.1` declaration would allow future major packaging changes without PopEx validation.

The recommended installation design is a separate, replaceable separation runtime profile or worker environment:

1. Create an environment using a supported Python version for that platform.
2. Install one exact, tested PyTorch build from the official CPU or accelerator-specific index.
3. Install `demucs==4.1.0` after PyTorch is present.
4. Keep TorchAudio out of the runtime.
5. Invoke the environment through a narrow PopEx adapter or subprocess boundary.

This preserves the base application and permits different lock data for Windows CPU, Linux CPU, Linux/Windows CUDA, and Apple Silicon MPS without pretending they are one portable wheel set. Intel macOS may be supported later only through an isolated Python 3.12/NumPy 1.x profile, not the current PopEx environment.

Exact pins are required for every supported runtime profile after testing. `demucs==4.1.0` is the approved Demucs pin. This audit does not select a universal PyTorch pin; the integration agent must test and lock one exact PyTorch build per profile.

## Model-download strategy

Importing the Demucs package does not by itself download weights. Constructing `demucs.api.Separator`, or invoking the CLI with a track, loads the selected pretrained model and can initiate a download when the cache is empty.

The integration must therefore avoid constructing a separator during application startup, health checks, module import, job listing, or capability detection.

For the first explicit separation action, PopEx should:

1. Detect whether the optional runtime exists.
2. Check available storage before starting.
3. Explain that the official `adefossez/HTDemucs` model is approximately 84 MB, is declared MIT-licensed, and is downloaded from Hugging Face.
4. Explain that PyTorch and its native runtime are substantially larger than the model and are installed separately.
5. Receive an explicit user action to continue.
6. Set cache, privacy, and telemetry environment variables before importing `huggingface_hub`.
7. Request `hf://adefossez/htdemucs` so a Hugging Face failure does not silently fall back to the older AWS `.th` checkpoint path.
8. Record Demucs version, model name, resolved model repository revision, and checkpoint hash with the separation result.

The forced Hugging Face path is a security condition. Generic `htdemucs` lookup first tries Hugging Face but catches failures and falls back to a legacy remote repository. That fallback uses `torch.hub.load_state_dict_from_url(..., weights_only=False)`. PopEx should prefer the reviewed safetensors artifact and must not silently broaden the download source or deserialization format.

## Cache and storage behavior

Demucs 4.1.0 uses `huggingface_hub.hf_hub_download` for the model YAML and safetensors file. The standard Hugging Face cache defaults to `~/.cache/huggingface/hub`; `HF_HOME` or `HF_HUB_CACHE` can relocate it. Cache environment variables are read at import time.

PopEx should place the cache in a clearly identified local application data directory, expose its location and approximate size, and provide a safe cache-removal action. Recommended environment controls, set before import, are:

- `HF_HUB_CACHE` or `HF_HOME` pointing to the PopEx model-cache directory;
- `HF_HUB_DISABLE_TELEMETRY=1` so the optional capability does not add telemetry;
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` so a user's unrelated Hugging Face credentials are not sent for this public model;
- `HF_HUB_OFFLINE=1` when the user explicitly selects offline mode and the model is already cached.

Expected download/storage effects:

- Demucs wheel: about 100.6 kB; source archive: about 1.2 MB.
- `htdemucs` checkpoint: about 84 MB, plus small metadata and cache bookkeeping.
- PyTorch 2.13.0 reference wheels are approximately 526.6 MB for CPython 3.13 Linux x86-64, 122.1 MB for Windows x86-64, and 111.2 MB for Apple Silicon before installation and transitive/runtime overhead.
- Hugging Face retains cached revisions until explicitly removed; Windows configurations without symlink support can use more disk.
- Four 44.1 kHz stereo 16-bit WAV stems require roughly 40 MiB per minute of source audio, separate from the model cache and retained source artifacts.

The UI must not present 84 MB as the total installation footprint; PyTorch and generated stems dominate storage.

## Supported runtime matrix

| Platform | Python | CPU | Acceleration | Decision and risks |
| --- | --- | --- | --- | --- |
| Windows x86-64 | 3.10-3.13 | Supported | NVIDIA CUDA through an explicit PyTorch CUDA index | Approved runtime profile; 32-bit Windows is unsupported upstream |
| Linux x86-64 | 3.10-3.13 | Supported | NVIDIA CUDA where a matching official build and driver are present | Approved runtime profiles; CPU profile must use the official CPU index to avoid unnecessary CUDA packages |
| Linux ARM64 | 3.10-3.13 where official wheels exist | Supported in current PyTorch metadata | Hardware-specific acceleration requires separate validation | Proposed profile; test before claiming support |
| macOS Apple Silicon | 3.10-3.13 | Supported | MPS/Metal supported upstream | Approved profile after integration testing; API callers must explicitly select `mps` because the CLI and API defaults differ |
| macOS Intel | At most 3.12 for the upstream PyTorch constraint | Upstream can run on CPU | No current supported MPS path | Blocked in the shared PopEx environment because Demucs requires NumPy below 2 while PopEx requires NumPy 2 or newer; consider an isolated worker profile |
| 32-bit operating systems | Not supported | Not supported | Not supported | Rejected |

Demucs's command-line parser selects CUDA, then MPS, then CPU. Its `Separator` API default selects CUDA when available and otherwise CPU. A PopEx API integration must explicitly select MPS on Apple Silicon when available and allow the user to force CPU.

## Python-version compatibility

Demucs 4.1.0 declares Python 3.10 or newer and its universal Python wheel is compatible in metadata with PopEx's global `>=3.10` declaration. Current PyTorch 2.13.0 publishes Python 3.13 wheels for Windows, Linux x86-64, Linux ARM64, and Apple Silicon macOS, so the Python 3.13 CI/Docker baseline is not blocked by wheel absence on those platforms.

This is metadata compatibility, not an end-to-end validation of PopEx separation on every platform. The current Docker and CI images install only base PopEx dependencies and must remain that way. A future separation runtime test matrix should exercise each exact profile independently.

Intel macOS is the material exception: the upstream `torch<2.3` and `numpy<2` markers conflict with PopEx's `numpy>=2`. Do not lower PopEx's global Python or NumPy requirements to accommodate that legacy platform.

## CPU, CUDA, and MPS behavior

CPU separation is an official and required path. It is slower than GPU operation but must remain functional and visible to users without a GPU.

CUDA support comes from the selected PyTorch build, not from a separate Demucs implementation. The integration must not assume that an installed NVIDIA driver guarantees that the Python environment contains a matching CUDA build. Device preflight and actionable fallback are required.

Apple Silicon MPS is officially supported through the Demucs CLI. PopEx must explicitly request `mps` when `torch.backends.mps.is_available()` and provide a CPU retry path for unsupported operations or runtime errors.

Do not install TorchAudio merely to obtain acceleration. It is not required for pretrained Demucs 4.1.0 separation.

## Native libraries and FFmpeg

Demucs 4.1.0 uses `sphn` for common audio decoding and falls back to FFmpeg for formats that `sphn` cannot read. FFmpeg is also required for FLAC output. PyTorch, `sphn`, and related packages contain native binaries even when no separate system library is required.

PopEx already prepares a 44.1 kHz PCM `analysis.wav` and already requires FFmpeg for source preparation. Passing that WAV to the Demucs API avoids introducing a new mandatory media format path. No additional FFmpeg requirement is created beyond PopEx's existing host dependency.

Training-only requirements such as SoundTouch, MUSDB tooling, and TorchAudio are outside the runtime and must not be installed.

## Failure and offline behavior

After the model has been explicitly downloaded and cached, separation can work offline. `HF_HUB_OFFLINE=1` prevents HTTP checks and restricts access to cached files. If the required revision is absent, the operation must fail with an actionable offline-cache message.

Required failure behavior:

- Missing optional runtime: report how to install the correct platform profile; do not mark source preparation or audio analysis failed.
- Network or model-host failure: retain all earlier artifacts and leave separation retryable.
- Interrupted or partial download: do not treat the model as ready until the expected file is present and verified.
- Checksum or revision mismatch: reject the model and request a clean, explicit re-download.
- Insufficient disk space: stop before model loading or stem output where possible and report the cache/output requirements.
- CUDA or MPS runtime failure: offer a deliberate CPU retry; do not silently change devices without recording it.
- Offline cache hit: perform no network request.
- Offline cache miss: fail before audio processing and preserve the job.

## Security and privacy considerations

- Audio processing remains local; only public model files are requested from the network.
- Disable Hugging Face telemetry for the PopEx-managed runtime.
- Do not send an implicit Hugging Face token for the public model.
- Use the reviewed official repository, exact model name, and recorded hash.
- Prefer safetensors and prevent the generic legacy checkpoint fallback.
- Treat downloaded model files as executable-adjacent supply-chain inputs: verify their source and hash before loading.
- Do not expose arbitrary cache or audio filesystem paths through the web API.
- Do not load user-selected arbitrary Hugging Face repositories or local checkpoint files in the first integration.
- Preserve private source audio, stems, and analysis under PopEx's existing local data controls.

## Packaging alternatives considered

### Add Demucs and PyTorch to default dependencies

Rejected. It would make a large native ML runtime mandatory, expand the base attack and license surface, and violate the lightweight local application requirement.

### Add `separation = ["demucs==4.1.0"]` directly to `pyproject.toml`

Deferred. It resolves differently by platform, can pull unnecessary CUDA packages on Linux, leaves PyTorch unbounded, and is unsatisfiable with PopEx's NumPy requirement on Intel macOS.

### Add complex environment markers to exclude Intel macOS

Rejected for now. A marker that silently omits Demucs on one supported PopEx platform is not a complete separation extra and still does not select CPU versus CUDA packages safely.

### Separate runtime profiles or worker environment

Accepted. This is the recommended approach because it isolates heavy native dependencies and permits exact per-platform locks.

### Bundle `htdemucs` weights

Rejected. Runtime download after an explicit user action is sufficient and avoids adding model files to every PopEx distribution.

### Use generic `htdemucs` and accept AWS fallback

Rejected. Use the explicit Hugging Face namespace to keep the download source and safetensors format within the reviewed boundary.

### Use `htdemucs_ft`

Deferred. It is a four-model bag with roughly four times the model storage and inference work. The single-model `htdemucs` is the approved baseline.

### Cloud-hosted separation

Rejected for the current product. It is unnecessary for the local-first stage and would introduce audio transfer, privacy, operating-cost, and service-dependency concerns.

## Explicit unresolved questions

1. What are the exact track-level provenance and licenses for the additional 800-song training collection?
2. Will the official HTDemucs repository add a standalone license/copyright file for the weight artifact, beyond MIT model metadata?
3. Which exact PyTorch release should be locked and tested for each PopEx CPU, CUDA, and MPS profile?
4. Should Intel macOS receive an isolated Python 3.12/NumPy 1.x worker, or be declared unsupported for separation?
5. How should PopEx pin or verify a Hugging Face repository revision when Demucs's high-level loader requests the default branch without a revision parameter?
6. What cache lifecycle and maximum-size policy should the application expose?
7. Which CUDA versions and minimum driver versions will PopEx document after hardware testing?
8. Should a future release redistribute model weights? If so, that release needs a new legal, provenance, storage, and notice review.

## Recommendation for the integration agent

Implement the next separation slice behind an optional-runtime adapter, not as a base dependency.

The integration should:

- target `demucs==4.1.0` and model `hf://adefossez/htdemucs`;
- instantiate the separator only inside an explicit separation job;
- configure a PopEx-owned Hugging Face cache before imports;
- disable telemetry and implicit token sending;
- record package, model, revision, hash, device, and runtime-profile versions;
- support CPU first, with optional CUDA and MPS selection;
- pass the existing prepared `analysis.wav` into Demucs;
- produce four stems (`drums`, `bass`, `other`, `vocals`) without claiming perfect isolation;
- keep source preparation and audio analysis completed when separation fails;
- provide explained first-use download, offline-cache behavior, retry, interruption, and low-disk handling;
- avoid TorchAudio, training dependencies, arbitrary model repositories, legacy AWS fallback, bundled weights, and application-startup downloads;
- add platform-profile installation documentation and tests before declaring separation generally available.

**Gate result: APPROVED WITH CONDITIONS.**
