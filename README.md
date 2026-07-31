# PopEx MVP

PopEx is a free, local-first foundation for converting a specific pop-song recording into a reviewable musical score. It accepts supported local audio/video files and YouTube URLs, retains the source artifact, creates a 44.1 kHz PCM `analysis.wav`, and produces deterministic timing and tonal metadata with librosa.

The longer-term score-generation workflow is intended for intermediate musicians and will combine:

- source separation
- pitched-note transcription
- drum and percussion transcription
- measure and rhythm construction
- chord-symbol extraction
- MusicXML generation
- guitar and bass tablature
- synchronized score review against the selected recording version

This cycle does **not** implement Demucs, transcription, chord recognition, notation, tablature, score rendering, or play-along presentation. Its output is confidence-scored timing and tonal metadata for later processing stages.

## Current workflow

```text
local upload or supported URL
→ safe source retention
→ analysis.wav normalization
→ signal validation
→ tempo and beat estimation
→ tonal-center, chroma, and tuning estimation
→ persisted JSON and SQLite summary
```

Supported upload formats: MP3, WAV, FLAC, M4A, AAC, OGG, MP4, MOV, and WebM.

## Analysis output

Every successful analysis writes:

```text
data/exports/{job_id}/analysis/audio-analysis.json
```

The versioned JSON contains:

- duration, sample rate, channels, peak amplitude, RMS, and RMS dBFS
- silent or near-silent validation
- global tempo estimate and confidence
- beat timestamps and beat confidence
- tentative downbeats and meter only when the baseline has enough evidence
- tempo-stability estimate
- global tonal-center candidate, confidence, and ranked alternatives
- 12-bin mean chroma vector
- tuning-offset estimate in cents
- library and analysis versions
- warnings for uncertain or weak signals

Detailed beat timestamps stay in JSON rather than being expanded into SQLite rows. SQLite stores compact summary fields for job lists and API responses.

### Tonal baseline and extensibility

The current deterministic baseline compares 24 profiles only:

- 12 Ionian/major candidates
- 12 Aeolian/minor candidates

It does not claim to detect Dorian, Phrygian, Lydian, Mixolydian, modal mixture, or modulation in this cycle. The persisted schema uses open collection names and includes:

- `tonalCenter`
- `primaryCandidate`
- `candidates`
- `localRegions`
- `chromaticismScore`

`localRegions` is reserved for later modulation and modal-mixture analysis. The compatibility fields `key`, `mode`, and `symbol` remain available while clients migrate to the extensible candidate structure.

## Processing status semantics

Source preparation and audio analysis are tracked separately:

- `preparation_status` reports whether the source and `analysis.wav` were created.
- `analysis_status` reports whether timing and tonal analysis is not started, processing, completed, or failed.
- Overall progress remains below 100 while analysis is running.
- Progress reaches 100 only after analysis succeeds.
- An analysis failure does not convert successful ingestion into a failed source preparation.
- Source, WAV, and metadata artifacts remain available after an analysis failure and analysis can be retried.

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

Historical completed jobs are not analyzed automatically at startup. Use the Analyze button or `POST /api/jobs/{job_id}/analyze`. An identical completed analysis is reused unless `force=true`. Active duplicate analysis requests are rejected.

Artifact downloads are limited to persisted source, normalized-WAV, and metadata filenames. Arbitrary job-directory paths are rejected.

## Windows: run without Docker

Requirements:

- Python 3.10 or newer; Python 3.12 is recommended
- FFmpeg and ffprobe on `PATH`
- Node.js LTS on `PATH` for YouTube extraction

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

The runner creates `.venv` and installs NumPy, SciPy, librosa, and SoundFile from normal Python wheels. CUDA is not required. The first install is larger than the ingestion-only version because it adds local signal-processing dependencies.

Common free installations:

```powershell
winget install -e --id Python.Python.3.12
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Gyan.FFmpeg
```

Close and reopen PowerShell after changing `PATH`. Open <http://localhost:8000>; do not browse to `0.0.0.0`.

## Docker

Docker Desktop must be open with its Linux engine running:

```bash
docker compose up --build
```

Open <http://localhost:8000>. The image installs FFmpeg and `libsndfile1`; the existing `/data` volume preserves sources, WAV files, JSON analysis, and SQLite history.

## Manual local setup

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

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

## Reliability behavior

- Existing SQLite databases are migrated in place.
- Previously completed jobs remain readable with `analysis_status=not_started`.
- Interrupted analysis retains successful source preparation and becomes retryable.
- Analysis failures retain the source, `analysis.wav`, and metadata.
- JSON is written atomically.
- Technical tracebacks are logged; user-facing errors remain concise.
- Meter and tonal-center results are explicitly estimates and carry confidence values; unavailable values are returned as `null`.

## Next planned cycle

The next planned implementation cycle is local Demucs stem separation using the maintained `adefossez/demucs` project and the `htdemucs` model. Chord-symbol extraction remains a later required layer of the generated score rather than the sole product goal.
