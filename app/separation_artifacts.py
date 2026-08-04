from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from app.config import Settings


class StemArtifactError(RuntimeError):
    """Base error for published stem detail and artifact resolution failures."""


class StemManifestUnavailableError(StemArtifactError):
    """Raised when no valid published stem manifest is available."""


class StemKindNotFoundError(StemArtifactError):
    """Raised when a requested stem kind is unsafe or absent from the manifest."""


@dataclass(frozen=True, slots=True)
class ResolvedStemArtifact:
    kind: str
    label: str
    path: Path
    file_name: str
    size_bytes: int
    duration_seconds: float
    sample_rate: int
    channels: int
    media_type: str = "audio/wav"

    @property
    def download_name(self) -> str:
        return f"{self.kind}.wav"


@dataclass(frozen=True, slots=True)
class StemDetails:
    available: bool
    status: str
    model: str | None
    version: str | None
    separated_at: str | None
    warnings: tuple[str, ...]
    stems: tuple[dict[str, object | None], ...]
    error: str | None

    def payload(self) -> dict[str, object | None]:
        raise NotImplementedError


def load_stem_details(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> StemDetails:
    raise NotImplementedError


def resolve_stem_artifact(
    job_id: str,
    kind: str,
    settings: Settings,
    job: Mapping[str, Any],
) -> ResolvedStemArtifact:
    raise NotImplementedError
