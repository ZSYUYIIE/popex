# PopEx MVP

PopEx is the first working slice of a local-first pop-music transcription product. This MVP accepts a supported video URL, extracts an MP3 audio source, stores it persistently, and shows a downloadable job history.

## Current scope

- Submit a YouTube video URL.
- Queue extraction without blocking the browser request.
- Extract the best available audio and convert it to MP3 with FFmpeg.
- Save the MP3 plus normalized metadata under a persistent data directory.
- Track queued, processing, completed, and failed jobs in SQLite.
- Download saved output files from the web interface.
- Enforce a configurable source allowlist, duration limit, and file-size limit.

Chord, note, stem, lyric, and score extraction are intentionally not included in this first slice. The saved audio becomes the input for those later stages.

## Windows: run without Docker

Requirements:

- Python 3.10 or newer
- FFmpeg and ffprobe available on `PATH`
- Node.js LTS available on `PATH`

From PowerShell in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\check_windows.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\run_windows.ps1
```

The runner creates `.venv`, installs the Python dependencies, creates the local `data` directory, and starts PopEx at <http://localhost:8000>.

If PowerShell reports that `ffmpeg`, `ffprobe`, or `node` is missing, install the dependency, close PowerShell, reopen it, and run the dependency check again.

## Run with Docker

Docker Desktop must be open and its Linux engine must be running before this command is used:

```bash
docker compose up --build
```

Open <http://localhost:8000>.

The Compose file mounts a named volume at `/data`, so extracted files and the SQLite database survive container restarts.

On Windows, verify Docker first:

```powershell
docker desktop status
docker version
docker context show
```

The expected context is normally `desktop-linux`. If Docker Desktop is stopped, start it from the Windows Start menu. If it is using Windows containers, switch it to Linux containers.

## Manual local setup: macOS, Linux, or Windows

Requirements:

- Python 3.10+
- FFmpeg and ffprobe available on `PATH`
- A supported JavaScript runtime such as Node.js or Deno for full YouTube extraction support

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e '.[dev]'
uvicorn app.main:app --reload
```

Open <http://localhost:8000>.

## Configuration

Copy `.env.example` values into your environment or Docker deployment settings.

| Variable | Default | Purpose |
| --- | --- | --- |
| `POPEX_DATA_DIR` | `data` | Database and extracted-file root |
| `POPEX_ALLOWED_HOSTS` | YouTube hosts | Comma-separated source host allowlist |
| `POPEX_MAX_DURATION_SECONDS` | `1800` | Reject videos longer than this limit |
| `POPEX_MAX_FILESIZE_MB` | `250` | Maximum source download size |
| `POPEX_AUDIO_QUALITY` | `192` | MP3 bitrate passed to FFmpeg |

## API

- `GET /api/health`
- `POST /api/jobs` with `{ "url": "https://www.youtube.com/watch?v=..." }`
- `GET /api/jobs`
- `GET /api/jobs/{job_id}`
- `GET /api/jobs/{job_id}/files/{file_name}`

## Validation

```bash
pytest
```

## Important usage limitation

Only process media you own or are authorized to download and transform. PopEx does not bypass DRM, authentication, paywalls, or private-access controls.

## Next product slices

1. WAV export for analysis-quality input.
2. Source separation into vocals, drums, bass, and accompaniment using local Demucs `htdemucs`.
3. Chord and beat extraction.
4. Local Whisper lyrics alignment.
5. Basic Pitch note transcription and MusicXML/MIDI export.
