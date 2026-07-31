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


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    allowed_hosts: tuple[str, ...]
    max_duration_seconds: int
    max_filesize_mb: int
    audio_quality: str

    @property
    def database_path(self) -> Path:
        return self.data_dir / "popex.sqlite3"

    @property
    def exports_dir(self) -> Path:
        return self.data_dir / "exports"

    @property
    def max_filesize_bytes(self) -> int:
        return self.max_filesize_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.exports_dir.mkdir(parents=True, exist_ok=True)

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
            audio_quality=os.getenv("POPEX_AUDIO_QUALITY", "192").strip() or "192",
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
