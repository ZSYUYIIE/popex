from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_ALLOWED_HOSTS = (
    "youtube.com",
    "www.youtube.com",
    "m.youtube.com",
    "music.youtube.com",
    "youtu.be",
)
SUPPORTED_MEDIA_EXTENSIONS = (
    ".mp3",
    ".wav",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".mp4",
    ".mov",
    ".webm",
)


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    allowed_hosts: tuple[str, ...]
    max_duration_seconds: int
    max_filesize_mb: int
    max_upload_mb: int
    audio_quality: str
    ffmpeg_binary: str = "ffmpeg"
    ffprobe_binary: str = "ffprobe"
    audio_analysis_enabled: bool = False
    audio_analysis_version: str = "baseline-librosa-v1"
    audio_analysis_timeout_seconds: int = 300
    audio_silence_rms_threshold: float = 0.0001

    @property
    def database_path(self) -> Path:
        return self.data_dir / "popex.sqlite3"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def max_filesize_bytes(self) -> int:
        return self.max_filesize_mb * 1024 * 1024

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)
        probe = self.data_dir / ".popex-write-test"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            raise RuntimeError(
                f"PopEx data directory is not writable: {self.data_dir}"
            ) from exc

    @classmethod
    def from_env(cls) -> "Settings":
        hosts = tuple(
            host.strip().lower()
            for host in os.getenv(
                "POPEX_ALLOWED_HOSTS", ",".join(DEFAULT_ALLOWED_HOSTS)
            ).split(",")
            if host.strip()
        )
        return cls(
            data_dir=Path(os.getenv("POPEX_DATA_DIR", "data")).expanduser().resolve(),
            allowed_hosts=hosts or DEFAULT_ALLOWED_HOSTS,
            max_duration_seconds=_positive_int("POPEX_MAX_DURATION_SECONDS", 1800),
            max_filesize_mb=_positive_int("POPEX_MAX_FILESIZE_MB", 250),
            max_upload_mb=_positive_int("POPEX_MAX_UPLOAD_MB", 500),
            audio_quality=os.getenv("POPEX_AUDIO_QUALITY", "192").strip() or "192",
            ffmpeg_binary=os.getenv("POPEX_FFMPEG_BINARY", "ffmpeg").strip()
            or "ffmpeg",
            ffprobe_binary=os.getenv("POPEX_FFPROBE_BINARY", "ffprobe").strip()
            or "ffprobe",
            audio_analysis_enabled=_boolean("AUDIO_ANALYSIS_ENABLED", True),
            audio_analysis_version=os.getenv(
                "AUDIO_ANALYSIS_VERSION", "baseline-librosa-v1"
            ).strip()
            or "baseline-librosa-v1",
            audio_analysis_timeout_seconds=_positive_int(
                "AUDIO_ANALYSIS_TIMEOUT_SECONDS", 300
            ),
            audio_silence_rms_threshold=_positive_float(
                "AUDIO_SILENCE_RMS_THRESHOLD", 0.0001
            ),
        )


def _positive_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _positive_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a number") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be greater than zero")
    return value


def _boolean(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise RuntimeError(f"{name} must be true or false")
