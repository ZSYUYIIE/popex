# Third-party notices

PopEx itself is licensed under the MIT License. Third-party software, model weights, datasets, media, fonts, and other assets retain their own licenses and copyright notices.

This file is an inventory and maintenance guide. It does not replace upstream license texts. Distributors of PopEx binaries, containers, installers, model bundles, or datasets must review the exact versions they distribute and include all notices required by those versions.

## Current direct Python dependencies

The current dependency declarations are maintained in `pyproject.toml`.

| Component | Current declared use | Upstream license | Upstream project |
| --- | --- | --- | --- |
| FastAPI | Web application and API framework | MIT | https://github.com/fastapi/fastapi |
| python-multipart | Multipart file uploads | Apache-2.0 | https://github.com/Kludex/python-multipart |
| yt-dlp | User-directed supported URL extraction | Unlicense | https://github.com/yt-dlp/yt-dlp |
| librosa | Audio feature, tempo, beat, chroma, and tuning analysis | ISC | https://github.com/librosa/librosa |
| NumPy | Numerical arrays and signal-processing support | BSD-3-Clause, with separately licensed bundled components | https://github.com/numpy/numpy |
| SciPy | Scientific and signal-processing support | BSD-3-Clause, with separately licensed bundled components | https://github.com/scipy/scipy |
| SoundFile | Audio file reading through libsndfile | BSD-3-Clause | https://github.com/bastibe/python-soundfile |
| pytest | Development tests | MIT | https://github.com/pytest-dev/pytest |
| HTTPX | Development and API tests | BSD-3-Clause | https://github.com/encode/httpx |
| setuptools | Python build backend | MIT | https://github.com/pypa/setuptools |

The `fastapi[standard]` extra and other packages introduce transitive dependencies. Their licenses must also be captured when producing a distributable lockfile, container, installer, or binary release.

## External runtime tools

| Component | Use | Licensing note |
| --- | --- | --- |
| FFmpeg / ffprobe | Media probing, extraction, and WAV normalization | FFmpeg may be distributed under LGPL or GPL terms depending on how a particular build is configured. PopEx does not treat every FFmpeg binary as having the same license. Distributors must inspect their chosen build. |
| Node.js | Runtime used by the current URL extraction setup | Node.js is distributed under the MIT License and includes third-party components with their own notices. |
| libsndfile | Native audio I/O used through SoundFile | Distributed under LGPL-2.1-or-later. Packaging must preserve the applicable terms and notices. |

System-installed tools are not relicensed by PopEx.

## Audited optional Demucs separation capability

The licensing and packaging decision is recorded in [docs/decisions/demucs-integration.md](docs/decisions/demucs-integration.md). This audit does not add Demucs or PyTorch to PopEx's default dependencies and does not bundle model weights.

### Software source and packages

| Component evaluated | Intended use | License and required notice | PopEx status |
| --- | --- | --- | --- |
| Demucs `4.1.0` from `adefossez/demucs` | Optional local four-stem separation | MIT; distributions containing Demucs must preserve the Meta Platforms copyright and MIT permission notice from the exact release | Approved only for an optional, platform-specific separation runtime; not currently declared in `pyproject.toml` and not bundled |
| PyTorch, with `2.13.0` reviewed as the current compatibility reference | Tensor runtime used by Demucs | BSD-style top-level source license plus separately licensed bundled components; distributors must include the exact license and notice inventory shipped with the selected wheel or native package | Optional runtime only; no universal version or accelerator build is approved by this audit |
| TorchAudio | Demucs training support, not pretrained separation runtime | BSD-2-Clause; binary releases must match the exact PyTorch version | Not required, not approved for the PopEx separation runtime, and not bundled |
| `huggingface_hub`, safetensors, `sphn`, `julius`, `einops`, `lameenc`, PyYAML, tqdm, and other Demucs runtime dependencies | Model acquisition, safe tensor loading, audio decoding, signal processing, and progress reporting | Each retains its own license and notices | Transitive optional dependencies only; capture exact versions and notices in each platform runtime profile |

Demucs source may be redistributed under MIT terms. PyTorch packages are large native distributions and differ by platform and compute backend. A release containing either component must inventory the exact files it ships; this table is not a substitute for that release inventory.

### `htdemucs` model weights

| Asset evaluated | Official source | Declared license | Size and identity | PopEx status |
| --- | --- | --- | --- | --- |
| `htdemucs` / `955717e8.safetensors` | `adefossez/HTDemucs` on Hugging Face, revision `bf35a81b663819a8255c8fefee17f9d812b786b5` | MIT in the official model repository metadata and model card | Approximately 84 MB; SHA-256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd` | Runtime download after explicit user action is approved; not bundled, not downloaded at startup, and not committed |

For the exact artifacts evaluated, Demucs source and the official `htdemucs` weights are both declared MIT-licensed. They remain separate third-party artifacts. Do not state that the weights are licensed by PopEx or infer their license from the Demucs source alone.

The official model repository currently provides MIT license metadata and a model card rather than a standalone full license file. If a future PopEx release redistributes the weights, that release must capture the exact model revision, model card, checkpoint hash, applicable MIT notice, and attribution. This audit deliberately approves runtime download instead of redistribution.

### Runtime-downloaded assets and cache

Demucs 4.1.0 normally obtains the `htdemucs` YAML and safetensors checkpoint through Hugging Face and stores them in the standard Hugging Face cache. PopEx must:

- request the reviewed official model explicitly, preferably as `hf://adefossez/htdemucs`;
- avoid Demucs's generic-name fallback to legacy AWS `.th` checkpoints;
- initiate the download only after a clear user separation action;
- disclose the model source, license, approximate size, and cache location;
- keep model caches out of Git, base packages, containers, installers, and release archives;
- disable telemetry and implicit credential sending in the PopEx-managed model runtime;
- record the resolved model revision and checkpoint hash with generated results;
- support offline reuse only after the complete model is cached and verified.

Downloading a public model file does not grant PopEx any rights to a user's source audio. Private audio must remain local and must not be sent to the model host.

### Training datasets and provenance

Official Demucs documentation states that `htdemucs` was trained on MUSDB/MUSDB-HQ plus an additional collection of 800 songs.

MUSDB18 and MUSDB18-HQ are not MIT-licensed audio datasets. Their official records restrict the recordings to educational use and prohibit commercial use without permission from the relevant copyright holders. The collection contains material with multiple source-specific terms, including Creative Commons non-commercial share-alike material. The `musdb` Python parser is MIT-licensed, but that license does not apply to the recordings.

The official Demucs sources inspected do not provide a track-level identity or license inventory for the additional 800-song training collection. Exported model metadata contains internal dataset paths, not redistribution permission.

PopEx does not bundle, download, test with, or redistribute these training datasets. The released model repository's MIT declaration is recorded separately from the training-data restrictions. The missing 800-song provenance remains an explicit risk for any future model redistribution or public/commercial packaging decision.

### Commercial use and redistribution summary

- Demucs 4.1.0 source: commercial use and redistribution permitted by MIT, with notice preservation.
- Official `htdemucs` checkpoint: commercial use and redistribution are stated as permitted by the model repository's MIT declaration, but PopEx does not bundle it under this decision.
- PyTorch and transitive packages: redistribution permitted only under each exact component's terms and notices.
- MUSDB/MUSDB-HQ recordings: commercial use is restricted by the dataset agreement; do not redistribute.
- Additional 800-song collection: license inventory unresolved; do not redistribute or characterize as approved.

## Models, datasets, and future components

A component is not approved for bundling merely because it is mentioned in the roadmap.

Before adding a model, dataset, renderer, transcription library, font, or bundled asset, the implementing agent must record:

- exact component and version;
- source-code license;
- model-weight license, if separate;
- dataset or training-data restrictions, where known and relevant;
- whether commercial use is allowed;
- whether redistribution is allowed;
- required attribution or notice text;
- whether the component can be downloaded by the user instead of redistributed;
- compatibility with PopEx's MIT-licensed, free, local-first core.

Current roadmap candidates that still require version-specific review when integrated include:

| Candidate | Planned purpose | Current repository status |
| --- | --- | --- |
| `adefossez/demucs` `4.1.0` with official `htdemucs` weights | Local source separation | Optional architecture approved with conditions; package, PyTorch runtime, and weights are not bundled; no universal `pyproject.toml` extra is currently safe |
| Spotify Basic Pitch | Initial pitched-note event extraction | Candidate only; not yet approved or bundled |
| music21 | Symbolic music and MusicXML construction | Candidate only; not yet approved or bundled |
| pretty_midi | MIDI generation and manipulation | Candidate only; not yet approved or bundled |
| OpenSheetMusicDisplay or Verovio | Browser score rendering | Candidate alternatives; final selection and licensing review pending |
| Drum/percussion transcription model | Percussion-event extraction | No model selected; licensing is part of selection criteria |

## Music and generated content

The PopEx MIT License does not grant rights to:

- uploaded or linked recordings;
- musical compositions;
- lyrics;
- arrangements;
- performances;
- artwork or metadata supplied by third parties;
- generated transcriptions of third-party works.

Users remain responsible for obtaining any rights required for their inputs and uses. This statement does not remove responsibilities that may apply to a future public platform operator. Public publishing, attribution, moderation, takedown, and rights workflows are deferred and are not part of the personal-use MVP.

## Release checklist

Before publishing a release artifact:

1. Generate an exact dependency inventory from the release environment.
2. Review direct and transitive licenses.
3. Review bundled native libraries separately from Python or JavaScript wrappers.
4. Review source-code, model-weight, and dataset licenses independently.
5. Include required license texts and copyright notices.
6. Confirm that no private audio, generated score, model cache, or development dataset is included accidentally.
7. For an optional separation profile, record the exact Demucs, PyTorch, accelerator, model-repository revision, checkpoint hash, and transitive package versions.
8. Confirm that no model is downloaded during application startup and no implicit credentials or telemetry are sent.
9. Update this file when the bundled dependency set changes.

When this file conflicts with an upstream license, the upstream license controls.
