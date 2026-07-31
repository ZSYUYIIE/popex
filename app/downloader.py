from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from app.config import Settings


class ExtractionError(RuntimeError):
    """Raised when a source cannot be converted into a saved audio file."""


@dataclass(frozen=True)
class ExtractionResult:
    title: str
    uploader: str | None
    duration_seconds: float | None
    files: tuple[str, ...]


ProgressCallback = Callable[[float], None]
_download_lock = threading.Lock()


def extract_audio(
    job_id: str,
    source_url: str,
    settings: Settings,
    progress_callback: ProgressCallback,
) -> ExtractionResult:
    """Download one URL and convert its audio stream to MP3.

    The process is intentionally serialized for the single-user MVP to prevent
    several FFmpeg jobs from exhausting a small machine.
    """

    job_dir = (settings.exports_dir / job_id).resolve()
    if settings.exports_dir.resolve() not in job_dir.parents:
        raise ExtractionError("Invalid job directory")
    job_dir.mkdir(parents=True, exist_ok=False)

    def match_filter(info: Mapping[str, Any], *, incomplete: bool = False) -> str | None:
        if incomplete:
            return None
        if info.get("is_live"):
            return "Live streams are not supported in this MVP"
        duration = info.get("duration")
        if duration and float(duration) > settings.max_duration_seconds:
            return (
                f"Video duration exceeds the {settings.max_duration_seconds}-second limit"
            )
        return None

    def progress_hook(data: Mapping[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            downloaded = data.get("downloaded_bytes") or 0
            if total:
                progress_callback(min(90.0, max(1.0, downloaded / total * 90.0)))
        elif status == "finished":
            progress_callback(92.0)

    try:
        import yt_dlp
    except ImportError as exc:
        raise ExtractionError("yt-dlp is not installed on the server") from exc

    options: dict[str, Any] = {
        "format": "bestaudio/best",
        "outtmpl": str(job_dir / "%(title).120B [%(id)s].%(ext)s"),
        "noplaylist": True,
        "windowsfilenames": True,
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
        with _download_lock:
            with yt_dlp.YoutubeDL(options) as downloader:
                info = downloader.extract_info(source_url, download=True)
    except Exception as exc:
        _cleanup_failed_job(job_dir)
        raise ExtractionError(_friendly_error(exc)) from exc

    mp3_files = sorted(path for path in job_dir.iterdir() if path.suffix.lower() == ".mp3")
    if not mp3_files:
        _cleanup_failed_job(job_dir)
        raise ExtractionError("Extraction completed but no MP3 file was produced")

    metadata = {
        "job_id": job_id,
        "source_url": source_url,
        "extractor": info.get("extractor_key") or info.get("extractor"),
        "source_id": info.get("id"),
        "title": info.get("title") or mp3_files[0].stem,
        "uploader": info.get("uploader") or info.get("channel"),
        "duration_seconds": info.get("duration"),
        "webpage_url": info.get("webpage_url") or source_url,
        "audio_file": mp3_files[0].name,
    }
    metadata_path = job_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    progress_callback(100.0)

    return ExtractionResult(
        title=str(metadata["title"]),
        uploader=metadata["uploader"],
        duration_seconds=(
            float(metadata["duration_seconds"])
            if metadata["duration_seconds"] is not None
            else None
        ),
        files=(mp3_files[0].name, metadata_path.name),
    )


def _cleanup_failed_job(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    for path in job_dir.iterdir():
        if path.is_file():
            path.unlink(missing_ok=True)
    try:
        job_dir.rmdir()
    except OSError:
        pass


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message[-500:]
