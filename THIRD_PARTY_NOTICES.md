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
| `adefossez/demucs` | Local source separation | Planned; not yet bundled in current `main` |
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
7. Update this file when the bundled dependency set changes.

When this file conflicts with an upstream license, the upstream license controls.