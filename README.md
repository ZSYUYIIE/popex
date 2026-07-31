# PopEx

PopEx is a free and open-source, local-first music-transcription workspace for intermediate musicians. It is intended to convert a **specific recording or arrangement** of a pop song into an editable draft score containing standard notation, guitar and bass tablature, drum and percussion notation, chord symbols, and recording-specific parts.

PopEx is released under the [MIT License](LICENSE). Its core self-hosted functionality must remain usable without a subscription, mandatory paid API, mandatory cloud service, or mandatory hosted GPU.

---

## Product source of truth

This section defines the intended product. It takes precedence over outdated branch descriptions, pull-request summaries, experimental scopes, and implementation shortcuts.

All people and AI agents working on this repository must read this section before changing the project. Product decisions in this section may only be changed following an explicit user-approved decision. Branch-specific documentation should describe implementation status without silently redefining the product.

### Product purpose

PopEx exists to reduce the time musicians spend manually transcribing different versions of pop songs.

The same composition may have substantially different:

- studio arrangements;
- live arrangements;
- acoustic arrangements;
- concert arrangements;
- covers;
- remixes;
- radio edits;
- instrumentation, riffs, fills, grooves, voicings, keys, structures, and endings.

PopEx therefore transcribes the selected **recording version**, not merely a song title or a generic lead sheet.

### Intended user

The primary user is an intermediate musician who:

- reads standard notation, tablature, drum notation, or chord symbols;
- needs to rehearse a particular recording or arrangement;
- can review and correct an AI-generated draft;
- may create their own riffs, fills, voicings, and instrument assignments;
- values saved transcription time more than beginner teaching features.

The MVP is not primarily a beginner chord-learning application. Do not prioritize chord diagrams, capo recommendations, simplified harmony, gamified lessons, basic theory tutorials, or automatic easy-chord substitutions.

### MVP goal

The MVP goal is **version-specific sheet-music generation**.

The defining workflow is:

```text
select a recording
→ prepare analysis-quality audio
→ analyze timing, meter, tuning, and tonal context
→ separate useful stems
→ transcribe pitched notes and percussion events
→ detect chord symbols
→ construct readable measures and rhythms
→ generate a full score and individual parts
→ generate guitar and bass tablature
→ review the score against synchronized audio
→ correct the generated transcription
→ export MusicXML and MIDI
```

A chord-only player is not the MVP. A chord-oriented play-along view may be added later, but it must not replace or redefine the score-generation objective.

### Required MVP score package

For each recording version, PopEx should aim to generate:

- a full combined score;
- individual parts where extraction is sufficiently reliable;
- lead melody or lead-vocal notation;
- bass-clef notation;
- bass tablature;
- guitar standard notation and tablature where viable;
- standard drum-kit notation;
- auxiliary-percussion notation where detectable;
- keyboard or accompaniment reduction for unresolved harmonic layers;
- chord symbols above the score;
- tempo, meter, measure, and section information;
- MIDI export;
- MusicXML export;
- synchronized source-audio review.

PDF export may follow MusicXML rendering. The MVP should produce a rehearsal-ready **draft**, not claim publication-quality engraving without human review.

### Chord symbols

Chord extraction is required, but chords are one layer of the score rather than the complete product.

Chord symbols allow musicians to:

- create alternate voicings;
- add or replace riffs;
- improvise;
- redistribute harmony between instruments;
- simplify or enrich an arrangement;
- build a different live implementation.

The architecture must not be restricted to major and minor triads. It must remain able to represent:

- sevenths and extensions;
- alterations;
- suspensions and added notes;
- inversions and slash chords;
- pedal harmony;
- modal borrowing;
- incomplete or ambiguous harmony;
- honest broad labels when precise extensions are not supported by the audio.

### Tonality, modes, and scales

PopEx must not assume that every recording uses one major or minor scale throughout.

Later harmonic analysis must be able to represent:

- tonal centre;
- ranked scale or mode candidates;
- local tonal regions;
- modulation;
- tonicization;
- modal mixture;
- borrowed harmony;
- chromaticism.

Significant candidate collections include:

- Ionian;
- Dorian;
- Phrygian;
- Lydian;
- Mixolydian;
- Aeolian;
- Locrian;
- harmonic minor;
- melodic minor;
- major and minor pentatonic collections;
- blues collections;
- later altered, whole-tone, diminished, and relevant Japanese collections.

The current baseline may support fewer collections, but persisted schemas and APIs must not permanently type-restrict the product to `major | minor`. A recording must not be forced into one scale when the evidence indicates modal mixture or changing tonal regions.

### Drums and percussion

Rhythm is a first-class transcription output.

The architecture must support both:

- pitched note events;
- percussion hit events.

Drum and percussion output must eventually use real percussion notation rather than ordinary pitched notes placed on a standard staff. The MVP should cover, where detectable:

- kick;
- snare;
- hi-hat;
- toms;
- ride and crash cymbals;
- claps;
- broad auxiliary-percussion events;
- simultaneous hits;
- rests;
- accents;
- open and closed hi-hat states;
- fills and repeated grooves.

When exact classification is uncertain, PopEx should use broader honest labels instead of inventing precision.

### Guitar and bass tablature

Tablature is part of the MVP score objective.

Pitch transcription and tablature generation are separate problems. Once notes are detected, PopEx must assign playable string and fret positions using:

- instrument tuning;
- playable range;
- hand-position continuity;
- chord and phrase context;
- user correction.

Standard notation and tablature must remain synchronized. Generated fingering may require correction, but it should be structurally editable rather than embedded as an irreversible rendering choice.

### Accompaniment reduction and instrument honesty

Dense pop production may contain several overlapping guitars, keyboards, synthesizers, strings, backing vocals, and sound-design layers. PopEx must not fabricate separate instrument parts when the source does not support reliable separation.

When necessary, generate an explicitly labelled **accompaniment reduction** containing the important harmony, rhythm, hooks, and countermelodies. Additional instrument-specific parts should appear only when confidence is sufficient.

### Arrangement and recording identity

The conceptual hierarchy is:

```text
Composition
└── Arrangement
    └── Recording version
        └── Transcription revision
```

Each recording version must retain its own:

- source assets;
- timing map;
- tonal analysis;
- stems;
- pitched note events;
- percussion events;
- chord events;
- instrument parts;
- tablature;
- score documents;
- exports;
- corrections and revisions.

Parts from different versions must never be silently combined.

A practical MVP acceptance scenario is:

```text
import a studio version and a live version of the same song
→ generate separate score packages
→ inspect vocal, bass, drums, accompaniment, tabs, and chord symbols
→ review each score against its own recording
→ correct a manageable number of errors
→ export MusicXML and rehearse from the generated parts
```

### Accuracy principle

PopEx produces an AI-assisted draft, not a guaranteed publication-ready score.

The system must:

- preserve confidence and warnings;
- retain raw model output;
- store user corrections separately;
- support correction, undo, and recovery;
- prefer broad honest outputs over false precision;
- avoid claiming that every instrument in a dense studio mix can be separated exactly;
- record model and analysis versions for reproducibility.

A failed later stage must not destroy successful earlier artifacts.

### Personal-use-first boundary

Development currently targets a private, local-first workflow:

- one local user is sufficient;
- files and generated results remain private;
- processing and storage are local by default;
- no public library is required;
- no account system is required;
- no collaboration is required;
- no public publishing is required;
- private uploads are not used for model training.

Public-product design is deferred. Do not add public catalog, moderation, payment, publishing, or licensing workflows to the personal MVP unless explicitly requested.

### Deferred public direction

A later public version may focus on clearly identifiable popular recording versions and arrangements available through public platforms, with exact attribution to composers, original artists, arrangers, performers, and source versions.

The following decisions are preserved for that later phase:

- public entries must identify the exact arrangement and recording version;
- private uploads must never be automatically added to a public library;
- public availability or citation alone must not be treated as permission to redistribute recordings, lyrics, or generated scores;
- platform integrations must comply with the then-current platform rules;
- public release requires separate design for attribution, rights declarations, privacy, notices, takedowns, moderation, and operator responsibilities.

These public features are not part of the current implementation roadmap.

### Free and open-source commitment

PopEx is intended to be free software available to everyone.

Original PopEx source code and project documentation are released under the MIT License unless a file explicitly states otherwise. Core self-hosted transcription capabilities must not require:

- a subscription;
- a paid account;
- a mandatory paid API;
- a mandatory cloud service;
- a mandatory hosted GPU;
- proprietary infrastructure controlled by the project maintainer.

Optional third-party hosting or paid compute may be supported later for convenience, but local operation must remain a supported path.

“Free for everyone” refers to software access and core functionality. Users may still incur their own hardware, storage, electricity, bandwidth, or optional hosting costs.

The MIT License applies to PopEx code, not automatically to:

- user music;
- recordings;
- compositions;
- arrangements;
- lyrics;
- generated transcriptions of third-party works;
- third-party datasets;
- model weights;
- dependencies;
- fonts or other bundled assets.

Every dependency, model, dataset, and bundled asset retains its own license. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

### Design principles

PopEx adopts the general principles of the MuseScore design handbook and translates them to an accessible web application:

- separate musical data from presentation;
- accessibility from the start;
- clear affordances and natural control placement;
- discoverability over hidden behaviour;
- continuous status feedback;
- user control, undo, and recovery;
- consistent interaction patterns;
- plain musician-facing language;
- minimal interface scope;
- no inactive placeholders for speculative future modules.

The interface should help musicians review and correct transcription results. It should not organize the user experience around worker processes, database rows, model tensors, or other implementation details.

### Architectural invariants

Do not design PopEx around chords alone.

The domain model must remain able to support:

```text
RecordingVersion
├── SourceAssets
├── TimingAnalysis
├── TonalAnalysis
├── Stems
├── PitchedNoteEvents
├── PercussionEvents
├── ChordEvents
├── InstrumentParts
├── Tablature
├── ScoreDocuments
├── Exports
└── TranscriptionRevisions
```

Required separation rules:

- raw model output is separate from cleaned notation;
- user corrections are separate from original predictions;
- analysis data is separate from UI layout;
- pitched events are separate from percussion events;
- score generation is separate from inference;
- tablature generation is separate from pitch recognition;
- every recording version has independent analysis and score data;
- schemas must remain versioned and migratable;
- expensive completed stages should be reusable and retryable.

### Canonical implementation order

The planned order after the audio-analysis foundation is:

1. Demucs stem separation using the maintained `adefossez/demucs` project;
2. raw pitched-note and percussion-event schemas and transcription;
3. baseline chord-event extraction for score symbols;
4. measure, rhythm, MIDI, and MusicXML score construction;
5. drum and auxiliary-percussion notation;
6. guitar and bass tablature generation;
7. synchronized score review and correction;
8. private arrangement/version grouping and comparison;
9. instrument-specific and modal-analysis accuracy improvements;
10. later chord-oriented play-along presentation;
11. separately designed public-library features, if approved.

This order may be adjusted when technical evidence requires it, but the MVP must remain centred on version-specific sheet music.

### Agent rules

Every agent working on this repository must:

1. Read this Product source of truth before beginning work.
2. Inspect current `main` before modifying a branch.
3. Preserve completed functionality unless explicitly instructed otherwise.
4. Keep each cycle narrow, independently testable, and reversible.
5. Avoid speculative empty modules and inactive UI placeholders.
6. Avoid redefining the MVP in branch documentation or PR descriptions.
7. Update the Current implementation status when behaviour changes.
8. Record material product changes here only after explicit user approval.
9. Treat this section as authoritative when older prompts or branch text conflict with it.
10. Clearly state limitations rather than exaggerating transcription accuracy.
11. Preserve the free, local-first, MIT-licensed core.
12. Verify third-party licensing before adding or redistributing models, data, or assets.

See [AGENTS.md](AGENTS.md) for the concise repository workflow.

---

## Current implementation status

Current `main` is an ingestion and baseline audio-analysis foundation. It does **not** yet generate sheet music, stems, notes, drum parts, tabs, or chord symbols.

Implemented capabilities:

- local audio/video upload;
- supported YouTube URL ingestion for user-authorized use;
- safe source retention;
- 44.1 kHz PCM `analysis.wav` normalization;
- persistent SQLite job history;
- separate source-preparation and analysis states;
- deterministic tempo, beat, tonal-centre, chroma, and tuning estimates with librosa;
- versioned analysis JSON;
- retryable analysis;
- source, WAV, metadata, and analysis downloads;
- local Windows, Linux/macOS, and Docker workflows;
- synthetic, offline automated tests.

### Current processing workflow

```text
local upload or supported URL
→ safe source retention
→ analysis.wav normalization
→ signal validation
→ tempo and beat estimation
→ tonal-centre, chroma, and tuning estimation
→ persisted JSON and SQLite summary
```

Supported upload formats: MP3, WAV, FLAC, M4A, AAC, OGG, MP4, MOV, and WebM.

## Analysis output

Every successful analysis writes:

```text
data/exports/{job_id}/analysis/audio-analysis.json
```

The versioned JSON contains:

- duration, sample rate, channels, peak amplitude, RMS, and RMS dBFS;
- silent or near-silent validation;
- global tempo estimate and confidence;
- beat timestamps and beat confidence;
- tentative downbeats and meter only when the baseline has enough evidence;
- tempo-stability estimate;
- global tonal-centre candidate, confidence, and ranked alternatives;
- 12-bin mean chroma vector;
- tuning-offset estimate in cents;
- library and analysis versions;
- warnings for uncertain or weak signals.

Detailed beat timestamps stay in JSON rather than being expanded into SQLite rows. SQLite stores compact summary fields for job lists and API responses.

### Tonal baseline and extensibility

The current deterministic baseline compares 24 profiles only:

- 12 Ionian/major candidates;
- 12 Aeolian/minor candidates.

It does not claim to detect Dorian, Phrygian, Lydian, Mixolydian, modal mixture, or modulation in the current cycle. The persisted schema uses open collection names and includes:

- `tonalCenter`;
- `primaryCandidate`;
- `candidates`;
- `localRegions`;
- `chromaticismScore`.

`localRegions` is reserved for later modulation and modal-mixture analysis. Compatibility fields `key`, `mode`, and `symbol` remain available while clients migrate to the extensible candidate structure.

## Processing status semantics

Source preparation and audio analysis are tracked separately:

- `preparation_status` reports whether the source and `analysis.wav` were created;
- `analysis_status` reports whether timing and tonal analysis is not started, processing, completed, or failed;
- overall progress remains below 100 while analysis is running;
- progress reaches 100 only after analysis succeeds;
- analysis failure does not convert successful ingestion into failed source preparation;
- source, WAV, and metadata artifacts remain available after analysis failure;
- analysis can be retried without re-uploading the source.

## API

- `GET /api/health`
- `POST /api/jobs` with `{ "url": "https://www.youtube.com/watch?v=..." }`
- `POST /api/uploads` with multipart field `file`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `POST /api/jobs/{job_id}/analyze`
- `POST /api/jobs/{job_id}/analyze?force=true`
- `GET /api/jobs/{job_id}/analysis`
- `GET /api/jobs/{job_id}/analysis/download`
- `GET /api/jobs/{job_id}/files/{file_name}`

Historical completed jobs are not analyzed automatically at startup. Use the Analyze action or `POST /api/jobs/{job_id}/analyze`. An identical completed analysis is reused unless `force=true`. Active duplicate analysis requests are rejected.

Artifact downloads are limited to persisted source, normalized-WAV, and metadata filenames. Arbitrary job-directory paths are rejected.

## Local setup

### Windows without Docker

Requirements:

- Python 3.10 or newer; Python 3.12 is recommended;
- FFmpeg and ffprobe on `PATH`;
- Node.js LTS on `PATH` for YouTube extraction.

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

Common free installations:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Gyan.FFmpeg
```

Close and reopen PowerShell after changing `PATH`. Open <http://localhost:8000>; do not browse to `0.0.0.0`.

### Linux or macOS without Docker

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

FFmpeg and ffprobe must be installed separately and available on `PATH`.

### Docker

Docker Desktop or Docker Engine must be running:

```bash
docker compose up --build
```

Open <http://localhost:8000>. The `/data` volume preserves sources, WAV files, JSON analysis, and SQLite history.

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `POPEX_DATA_DIR` | `data` | Local database and artifact root |
| `POPEX_MAX_DURATION_SECONDS` | `1800` | Maximum accepted/analyzed duration |
| `POPEX_MAX_FILESIZE_MB` | `250` | URL source size limit |
| `POPEX_MAX_UPLOAD_MB` | `500` | Direct upload limit |
| `POPEX_FFMPEG_BINARY` | `ffmpeg` | FFmpeg executable or explicit path |
| `POPEX_FFPROBE_BINARY` | `ffprobe` | ffprobe executable or explicit path |
| `AUDIO_ANALYSIS_ENABLED` | `true` | Analyze newly normalized media |
| `AUDIO_ANALYSIS_VERSION` | `baseline-librosa-v1` | Reproducibility and cache key |
| `AUDIO_ANALYSIS_TIMEOUT_SECONDS` | `300` | Analysis timeout guard |
| `AUDIO_SILENCE_RMS_THRESHOLD` | `0.0001` | Effective-silence threshold |

## Validation

```bash
pytest
python -m compileall -q app tests
node --check app/static/app.js
```

Tests generate synthetic click tracks and tonal signals. They do not require internet access or copyrighted recordings.

## Reliability behaviour

- Existing SQLite databases are migrated in place.
- Previously completed jobs remain readable with `analysis_status=not_started`.
- Interrupted analysis retains successful source preparation and becomes retryable.
- Analysis failures retain the source, `analysis.wav`, and metadata.
- JSON is written atomically.
- Technical tracebacks are logged; user-facing errors remain concise.
- Meter and tonal-centre results are estimates and carry confidence values; unavailable values are returned as `null`.

## Known limitations

- No source separation yet.
- No note or percussion-event transcription yet.
- No chord extraction yet.
- No MusicXML, MIDI, PDF, tablature, or score rendering yet.
- Current global tonal estimation evaluates Ionian/major and Aeolian/minor profiles only.
- Dense pop arrangements cannot yet be represented as instrument parts.
- URL ingestion depends on external platform availability and must only be used where the user is authorized to process the source.

## Next planned cycle

The next planned implementation cycle is local Demucs stem separation using the maintained `adefossez/demucs` project and the `htdemucs` model.

The cycle should persist and expose at least:

- vocals;
- bass;
- drums;
- other/accompaniment;
- model and separation versions;
- warnings and failures;
- safe stem preview/download;
- retryable processing without destroying existing analysis.

It must remain local, optional, and compatible with the broader score-generation architecture described above.

## License

PopEx is licensed under the [MIT License](LICENSE).

Third-party dependencies, models, datasets, tools, and assets retain their own licenses. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).