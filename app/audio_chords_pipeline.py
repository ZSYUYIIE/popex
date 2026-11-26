"""Hear chords in a job's audio and save them beside its harmony result.

This runs as the last step of the harmony stage. It is best effort: when the
audio, the beats, or the audio libraries are unavailable it saves nothing and
the score falls back to chords named from transcribed notes. A failure here
never fails the harmony stage.
"""

from __future__ import annotations

import logging
import math
import stat
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app import db
from app.audio_chords import (
    ANALYSIS_SAMPLE_RATE,
    AudioChordsError,
    build_audio_chords_artifact,
    load_audio,
    recognize_chords,
    write_audio_chords_artifact,
)
from app.beat_grid import BEAT_GRID_VERSION
from app.config import Settings
from app.separation import STEM_MANIFEST_RELATIVE_PATH, load_stem_manifest
from app.separation_artifacts import resolve_stem_artifact

_LOGGER = logging.getLogger(__name__)
_MIN_AUDIO_SECONDS = 4.0


def _regular_file(path: Path) -> Path | None:
    try:
        info = path.lstat()
    except OSError:
        return None
    if not stat.S_ISREG(info.st_mode) or info.st_size <= 0:
        return None
    return path


def _constant_beats(timing: Mapping[str, Any], duration: float) -> list[float] | None:
    tempo = timing.get("tempoBpm")
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)):
        return None
    tempo = float(tempo)
    if not math.isfinite(tempo) or not 20.0 <= tempo <= 300.0:
        return None
    beat = 60.0 / tempo
    count = int(duration / beat) + 2
    return [index * beat for index in range(count)]


def publish_audio_chords(
    job_id: str,
    settings: Settings,
    *,
    raw_transcription: Mapping[str, Any],
    timing: Mapping[str, Any] | None,
    analysis_version: str,
    harmony_created_at: str,
) -> str | None:
    """Recognize and save chords for one harmony result.

    Returns the job-relative file name, or ``None`` when nothing was saved.
    Never raises.
    """
    try:
        return _publish(
            job_id,
            settings,
            raw_transcription=raw_transcription,
            timing=timing,
            analysis_version=analysis_version,
            harmony_created_at=harmony_created_at,
        )
    except AudioChordsError:
        _LOGGER.warning("Audio chord recognition was skipped for a job.", exc_info=True)
    except Exception:  # a best-effort step must never fail the harmony stage
        _LOGGER.warning("Audio chord recognition failed for a job.", exc_info=True)
    return None


def _publish(
    job_id: str,
    settings: Settings,
    *,
    raw_transcription: Mapping[str, Any],
    timing: Mapping[str, Any] | None,
    analysis_version: str,
    harmony_created_at: str,
) -> str | None:
    if not isinstance(timing, Mapping):
        return None
    job = db.get_job(settings.database_path, job_id)
    if job is None:
        return None
    job_dir = Path(settings.exports_dir) / job_id
    mix_path = _regular_file(job_dir / "analysis.wav")
    if mix_path is None:
        return None
    mix = load_audio(mix_path)
    duration = len(mix) / ANALYSIS_SAMPLE_RATE
    if duration < _MIN_AUDIO_SECONDS:
        return None

    accompaniment = bass = None
    if (
        job.get("separation_status") == "completed"
        and job.get("stem_manifest_file_name") == STEM_MANIFEST_RELATIVE_PATH
    ):
        try:
            if load_stem_manifest(job_id, settings) is not None:
                accompaniment = load_audio(resolve_stem_artifact(job_id, "other", settings, job).path)
                bass = load_audio(resolve_stem_artifact(job_id, "bass", settings, job).path)
        except Exception:  # stems are optional evidence
            _LOGGER.warning("Stems were not usable for chord recognition.", exc_info=True)
            accompaniment = bass = None

    # Imported here so the harmony stage does not import the score layer at load.
    from app.score_api import score_beat_grid, score_meter

    # The same meter decision as the score, so chord changes and bars agree.
    meter, _source, _confidence = score_meter({"timing": timing}, raw_transcription)
    grid = score_beat_grid({"timing": timing}, raw_transcription, beats_per_measure=meter)
    if grid is not None:
        beats = list(grid.times)
        grid_version: str | None = BEAT_GRID_VERSION
    else:
        beats = _constant_beats(timing, duration)
        grid_version = None
        if beats is None:
            return None
    beats = [value for value in beats if value <= duration + 1.0]
    if len(beats) < 6:
        return None

    result = recognize_chords(beats, meter=meter, mix=mix, accompaniment=accompaniment, bass=bass)
    artifact = build_audio_chords_artifact(
        result,
        job_id=job_id,
        created_at=datetime.now(timezone.utc).isoformat(),
        harmony_created_at=harmony_created_at,
        analysis_version=analysis_version,
        grid_version=grid_version,
    )
    return write_audio_chords_artifact(settings, artifact)


__all__ = ["publish_audio_chords"]
