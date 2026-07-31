from __future__ import annotations

import json
import re
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping
from uuid import uuid4

from app.config import Settings


class MediaProcessingError(RuntimeError):
    """Raised when an imported source cannot be prepared for analysis."""


@dataclass(frozen=True)
class MediaResult:
    title: str
    uploader: str | None
    duration_seconds: float | None
    source_format: str | None
    sample_rate: int | None
    channel_count: int | None
    source_file_name: str
    normalized_file_name: str
    files: tuple[str, ...]


ProgressCallback = Callable[[float], None]
StageCallback = Callable[[str, str, float], None]
_processing_lock = threading.Lock()


def process_url(
    job_id: str,
    source_url: str,
    settings: Settings,
    stage_callback: StageCallback,
    progress_callback: ProgressCallback,
) -> MediaResult:
    job_dir = secure_job_dir(settings, job_id, create=True)
    stage_callback("extracting_audio", "Downloading and extracting source audio.", 8)

    try:
        import yt_dlp
    except ImportError as exc:
        cleanup_job_dir(job_dir)
        raise MediaProcessingError("yt-dlp is not installed.") from exc

    def match_filter(
        info: Mapping[str, Any], *, incomplete: bool = False
    ) -> str | None:
        if incomplete:
            return None
        if info.get("is_live"):
            return "Live streams are not supported in this MVP"
        duration = info.get("duration")
        if duration and float(duration) > settings.max_duration_seconds:
            return (
                f"Video duration exceeds the "
                f"{settings.max_duration_seconds}-second limit"
            )
        return None

    def progress_hook(data: Mapping[str, Any]) -> None:
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                progress_callback(
                    min(62.0, max(10.0, downloaded / total * 52 + 10))
                )
        elif data.get("status") == "finished":
            progress_callback(65.0)

    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(job_dir / "source.%(ext)s"),
        "noplaylist": True,
        "overwrites": False,
        "max_filesize": settings.max_filesize_bytes,
        "match_filter": match_filter,
        "progress_hooks": [progress_hook],
        "quiet": True,
        "no_warnings": True,
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": settings.audio_quality,
            }
        ],
    }

    try:
        with _processing_lock:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=True)
    except Exception as exc:
        cleanup_job_dir(job_dir)
        raise MediaProcessingError(
            friendly_error(exc, settings=settings)
        ) from exc

    source_path = job_dir / "source.mp3"
    if not source_path.is_file():
        candidates = sorted(
            path
            for path in job_dir.iterdir()
            if path.is_file() and path.name != "analysis.wav"
        )
        if not candidates:
            cleanup_job_dir(job_dir)
            raise MediaProcessingError(
                "Extraction completed but no source audio file was produced."
            )
        source_path = candidates[0]

    title = str(info.get("title") or "Imported URL audio")
    uploader = info.get("uploader") or info.get("channel")
    return prepare_source(
        job_id=job_id,
        source_path=source_path,
        title=title,
        uploader=str(uploader) if uploader else None,
        settings=settings,
        stage_callback=stage_callback,
        progress_callback=progress_callback,
        metadata_extra={
            "source_type": "url",
            "source_url": source_url,
            "extractor": info.get("extractor_key") or info.get("extractor"),
            "source_id": info.get("id"),
            "webpage_url": info.get("webpage_url") or source_url,
        },
    )


def process_upload(
    job_id: str,
    source_file_name: str,
    original_filename: str,
    settings: Settings,
    stage_callback: StageCallback,
    progress_callback: ProgressCallback,
) -> MediaResult:
    job_dir = secure_job_dir(settings, job_id)
    source_path = resolve_job_file(job_dir, source_file_name)
    if not source_path.is_file():
        raise MediaProcessingError("The uploaded source file is missing.")
    return prepare_source(
        job_id=job_id,
        source_path=source_path,
        title=Path(original_filename).stem or "Uploaded media",
        uploader=None,
        settings=settings,
        stage_callback=stage_callback,
        progress_callback=progress_callback,
        metadata_extra={
            "source_type": "upload",
            "original_filename": original_filename,
        },
    )


def prepare_source(
    *,
    job_id: str,
    source_path: Path,
    title: str,
    uploader: str | None,
    settings: Settings,
    stage_callback: StageCallback,
    progress_callback: ProgressCallback,
    metadata_extra: Mapping[str, Any],
) -> MediaResult:
    job_dir = secure_job_dir(settings, job_id)
    source_path = resolve_job_file(job_dir, source_path.name)

    stage_callback("validating", "Inspecting source media with ffprobe.", 68)
    source_metadata = probe_media(source_path, settings)
    duration = source_metadata["duration_seconds"]
    if duration is not None and duration > settings.max_duration_seconds:
        raise MediaProcessingError(
            f"Media duration exceeds the "
            f"{settings.max_duration_seconds}-second limit."
        )

    analysis_path = job_dir / "analysis.wav"
    stage_callback("normalizing", "Creating 44.1 kHz PCM analysis audio.", 76)
    with _processing_lock:
        normalize_audio(source_path, analysis_path, settings)
    progress_callback(96.0)

    normalized_metadata = probe_media(analysis_path, settings)
    metadata = {
        "job_id": job_id,
        "title": title,
        "uploader": uploader,
        "duration_seconds": duration,
        "source_audio_format": source_metadata["format_name"],
        "source_sample_rate": source_metadata["sample_rate"],
        "source_channel_count": source_metadata["channel_count"],
        "source_file": source_path.name,
        "normalized_wav": analysis_path.name,
        "normalized_sample_rate": normalized_metadata["sample_rate"],
        "normalized_channel_count": normalized_metadata["channel_count"],
        **dict(metadata_extra),
    }
    metadata_path = job_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress_callback(100.0)

    return MediaResult(
        title=title,
        uploader=uploader,
        duration_seconds=duration,
        source_format=source_metadata["format_name"],
        sample_rate=source_metadata["sample_rate"],
        channel_count=source_metadata["channel_count"],
        source_file_name=source_path.name,
        normalized_file_name=analysis_path.name,
        files=(source_path.name, analysis_path.name, metadata_path.name),
    )


def probe_media(path: Path, settings: Settings) -> dict[str, Any]:
    require_binary(settings.ffprobe_binary, "ffprobe")
    command = [
        settings.ffprobe_binary,
        "-v",
        "error",
        "-show_entries",
        "format=duration,format_name:stream=codec_type,sample_rate,channels",
        "-of",
        "json",
        str(path),
    ]
    completed = run_command(command, settings=settings, paths=(path,))
    try:
        payload = json.loads(completed.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise MediaProcessingError(
            "ffprobe returned invalid media metadata."
        ) from exc
    audio_stream = next(
        (
            stream
            for stream in payload.get("streams", [])
            if stream.get("codec_type") == "audio"
        ),
        None,
    )
    if not audio_stream:
        raise MediaProcessingError(
            "The selected file does not contain an audio stream."
        )
    duration_raw = payload.get("format", {}).get("duration")
    sample_rate_raw = audio_stream.get("sample_rate")
    return {
        "duration_seconds": float(duration_raw) if duration_raw else None,
        "format_name": payload.get("format", {}).get("format_name"),
        "sample_rate": int(sample_rate_raw) if sample_rate_raw else None,
        "channel_count": (
            int(audio_stream["channels"])
            if audio_stream.get("channels")
            else None
        ),
    }


def normalize_audio(
    source_path: Path, output_path: Path, settings: Settings
) -> None:
    require_binary(settings.ffmpeg_binary, "FFmpeg")
    output_path.unlink(missing_ok=True)
    command = [
        settings.ffmpeg_binary,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source_path),
        "-map",
        "0:a:0",
        "-vn",
        "-c:a",
        "pcm_s16le",
        "-ar",
        "44100",
        str(output_path),
    ]
    run_command(
        command,
        settings=settings,
        paths=(source_path, output_path),
    )
    if not output_path.is_file() or output_path.stat().st_size == 0:
        output_path.unlink(missing_ok=True)
        raise MediaProcessingError("FFmpeg did not produce the analysis WAV.")


def run_command(
    command: list[str],
    *,
    settings: Settings,
    paths: tuple[Path, ...] = (),
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=max(120, settings.max_duration_seconds * 2),
        )
    except FileNotFoundError as exc:
        raise MediaProcessingError(
            f"Required executable '{command[0]}' is not available on PATH."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise MediaProcessingError("Media processing timed out.") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr or exc.stdout or "The media command failed."
        raise MediaProcessingError(
            friendly_error(detail, settings=settings, paths=paths)
        ) from exc


def require_binary(binary: str, display_name: str) -> None:
    if Path(binary).is_file() or shutil.which(binary):
        return
    raise MediaProcessingError(
        f"{display_name} is unavailable. "
        f"Install it and ensure '{binary}' is on PATH."
    )


def dependency_report(settings: Settings) -> dict[str, Any]:
    return {
        "ffmpeg": bool(
            Path(settings.ffmpeg_binary).is_file()
            or shutil.which(settings.ffmpeg_binary)
        ),
        "ffprobe": bool(
            Path(settings.ffprobe_binary).is_file()
            or shutil.which(settings.ffprobe_binary)
        ),
        "data_directory_writable": _directory_is_writable(settings.data_dir),
    }


def _directory_is_writable(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / f".write-check-{uuid4().hex}"
        probe.write_bytes(b"ok")
        probe.unlink()
        return True
    except OSError:
        return False


def secure_job_dir(
    settings: Settings, job_id: str, *, create: bool = False
) -> Path:
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise MediaProcessingError("Invalid job identifier.")
    root = settings.exports_dir.resolve()
    job_dir = (root / job_id).resolve()
    if root not in job_dir.parents:
        raise MediaProcessingError("Invalid job directory.")
    if create:
        job_dir.mkdir(parents=True, exist_ok=False)
    return job_dir


def resolve_job_file(job_dir: Path, file_name: str) -> Path:
    if Path(file_name).name != file_name:
        raise MediaProcessingError("Invalid stored file name.")
    candidate = (job_dir / file_name).resolve()
    if job_dir.resolve() not in candidate.parents:
        raise MediaProcessingError("Invalid stored file path.")
    return candidate


def cleanup_job_dir(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    shutil.rmtree(job_dir, ignore_errors=True)


def friendly_error(
    error: Exception | str,
    *,
    settings: Settings,
    paths: tuple[Path, ...] = (),
) -> str:
    message = str(error).strip() or "Media processing failed."
    replacements = [
        settings.data_dir.resolve(),
        *[path.resolve() for path in paths],
    ]
    for path in replacements:
        message = message.replace(str(path), "<local media>")
        message = message.replace(
            str(path).replace("\\", "/"), "<local media>"
        )
    message = re.sub(
        r"(?i)(?:[a-z]:\\|/)(?:[^:\n\r]+[\\/])+[^:\n\r ]+",
        "<local path>",
        message,
    )
    message = re.sub(r"\s+", " ", message).strip()
    return message[-800:]


def generated_source_name(extension: str) -> str:
    normalized = extension.lower()
    if not normalized.startswith("."):
        normalized = f".{normalized}"
    return f"source-{uuid4().hex}{normalized}"
