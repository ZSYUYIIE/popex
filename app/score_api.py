"""Read-only score preview built on demand from published job artifacts.

This module is the HTTP-facing half of the score-construction vertical slice.
It combines the already-published raw transcription (pitched-note evidence)
with audio-analysis tempo/meter evidence into the versioned draft score
document from :mod:`app.score_construction` and renders stdlib-only MIDI and
MusicXML bytes. It performs no database writes, no filesystem writes, and no
network access.

Honesty rules:
- transcription and tempo evidence are both required; nothing is invented;
- meter falls back to a 4/4 draft grid only with an explicit warning;
- chord-symbol measure mapping is deferred and reported, not fabricated.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from typing import Any

from app.analysis import AudioAnalysisError, load_analysis
from app.config import Settings
from app.score_construction import (
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    score_to_midi_bytes,
    score_to_musicxml_text,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    load_raw_transcription,
)

_METER_FALLBACK = 4


def _optional_confidence(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    confidence = float(value)
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        return None
    return confidence


class ScorePreviewUnavailableError(RuntimeError):
    """A score preview cannot be built because required evidence is missing."""


class ScorePreviewError(RuntimeError):
    """A score preview cannot be built because saved evidence is invalid."""


def _record_pointer(record: Mapping[str, Any]) -> None:
    if record.get("transcription_status") != "completed":
        raise ScorePreviewUnavailableError(
            "A completed raw transcription is required before score preview."
        )
    if (
        record.get("transcription_artifact_file_name")
        != RAW_TRANSCRIPTION_RELATIVE_PATH
    ):
        raise ScorePreviewUnavailableError(
            "A published raw transcription is required before score preview."
        )


def build_score_preview(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    *,
    include_measures: bool = False,
) -> dict[str, Any]:
    """Build a detached score-preview payload from published artifacts."""
    _record_pointer(record)
    try:
        raw_transcription = load_raw_transcription(job_id, settings)
    except RawTranscriptionError as exc:
        raise ScorePreviewError(
            "Published raw transcription could not be validated."
        ) from exc
    if raw_transcription is None:
        raise ScorePreviewUnavailableError(
            "Published raw transcription is unavailable."
        )
    try:
        analysis = load_analysis(job_id, settings)
    except AudioAnalysisError as exc:
        raise ScorePreviewError(
            "Saved audio analysis could not be validated."
        ) from exc
    if analysis is None:
        raise ScorePreviewUnavailableError(
            "Saved audio analysis is unavailable; tempo evidence is required."
        )
    timing = analysis.get("timing")
    if not isinstance(timing, Mapping):
        raise ScorePreviewError("Saved audio analysis timing is invalid.")
    tempo = timing.get("tempoBpm")
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)):
        raise ScorePreviewUnavailableError(
            "Tempo evidence is unavailable; score preview requires analysis tempo."
        )
    tempo_confidence = _optional_confidence(timing.get("tempoConfidence"))
    meter_confidence = _optional_confidence(timing.get("meterConfidence"))
    tempo_stable = timing.get("tempoStable")
    if not isinstance(tempo_stable, bool):
        tempo_stable = None
    meter = timing.get("meter")
    if type(meter) is int and 1 <= meter <= 12:
        beats_per_measure = meter
        meter_source = "analysis"
    else:
        beats_per_measure = _METER_FALLBACK
        meter_source = "fallback-4/4"

    pitched_inputs = [
        {
            "id": event["id"],
            "sourceKind": event["sourceKind"],
            "startSeconds": event["startSeconds"],
            "endSeconds": event["endSeconds"],
            "midiNote": event["midiNote"],
            "midiPitch": event["midiPitch"],
            "confidence": event["confidence"],
            "warnings": event.get("warnings", []),
        }
        for event in raw_transcription.get("pitchedNoteEvents", ())
    ]
    try:
        document = build_score_document(
            pitched_inputs,
            tempo_bpm=float(tempo),
            beats_per_measure=beats_per_measure,
        )
    except ScoreConstructionError as exc:
        raise ScorePreviewError(
            "Score preview could not be constructed from saved evidence."
        ) from exc

    warnings = list(document["warnings"])
    warnings.append(
        "Pitched events are shown as one draft part; instrument-specific part "
        "assignment is not included."
    )
    if tempo_confidence is None:
        warnings.append(
            "Tempo confidence is unavailable; review score timing and measure placement."
        )
    elif tempo_confidence < 0.50:
        warnings.append(
            "Tempo confidence is below 0.50; review score timing and measure placement."
        )
    if tempo_stable is False:
        warnings.append(
            "The estimated tempo is unstable; review score timing and measure placement."
        )
    if meter_source == "analysis" and (
        meter_confidence is None or meter_confidence < 0.50
    ):
        warnings.append(
            "Meter confidence is low or unavailable; review the measure grouping."
        )
    percussion_event_count = len(raw_transcription.get("percussionEvents", ()))
    if percussion_event_count:
        warnings.append(
            "Percussion events are present but are not rendered in this pitched-note draft."
        )
    if meter_source != "analysis":
        warnings.append(
            "Meter evidence is unavailable or weak; measures use an explicit "
            "4/4 draft grid and require musician review."
        )
    harmony_pointer = record.get("harmony_artifact_file_name")
    if (
        record.get("harmony_status") == "completed"
        and isinstance(harmony_pointer, str)
    ):
        warnings.append(
            "Harmonic context is available but chord-symbol measure mapping "
            "is deferred to a later slice; this draft carries no chord symbols."
        )
    warnings = warnings[:16]

    source_analysis = raw_transcription.get("sourceAnalysis", {})
    payload: dict[str, Any] = {
        "available": True,
        "schemaVersion": SCORE_SCHEMA_VERSION,
        "builderVersion": SCORE_BUILDER_VERSION,
        "tempoBpm": document["tempoBpm"],
        "beatsPerMeasure": document["beatsPerMeasure"],
        "divisions": document["divisions"],
        "meterSource": meter_source,
        "measureCount": document["measureCount"],
        "noteCount": document["noteCount"],
        "percussionEventCount": percussion_event_count,
        "warnings": warnings,
        "timingEvidence": {
            "tempoConfidence": tempo_confidence,
            "tempoStable": tempo_stable,
            "meterConfidence": meter_confidence,
        },
        "provenance": {
            "transcriptionVersion": raw_transcription.get("transcriptionVersion"),
            "transcribedAt": raw_transcription.get("createdAt"),
            "analysisVersion": analysis.get("analysisVersion"),
            "analysisCreatedAt": analysis.get("createdAt"),
            "sourceAnalysisFile": source_analysis.get("fileName")
            if isinstance(source_analysis, Mapping)
            else None,
        },
        "downloadUrls": {
            "midi": f"/api/jobs/{job_id}/score/download?format=midi",
            "musicxml": f"/api/jobs/{job_id}/score/download?format=musicxml",
        },
    }
    if include_measures:
        payload["measures"] = document["measures"]
    return copy.deepcopy(payload)


def render_score_download(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    *,
    format: str,
) -> tuple[bytes, str, str]:
    """Render score bytes for an explicit ``midi`` or ``musicxml`` format."""
    if format not in ("midi", "musicxml"):
        raise ScorePreviewUnavailableError("Unsupported score format.")
    preview = build_score_preview(
        job_id, settings, record, include_measures=True
    )
    document = {
        "schemaVersion": preview["schemaVersion"],
        "tempoBpm": preview["tempoBpm"],
        "beatsPerMeasure": preview["beatsPerMeasure"],
        "divisions": preview["divisions"],
        "measureCount": preview["measureCount"],
        "measures": preview["measures"],
    }
    try:
        if format == "midi":
            return (
                score_to_midi_bytes(document),
                "draft-score.mid",
                "audio/midi",
            )
        text = score_to_musicxml_text(document)
    except ScoreConstructionError as exc:
        raise ScorePreviewError(
            "Score download could not be rendered from saved evidence."
        ) from exc
    return (
        text.encode("utf-8"),
        "draft-score.musicxml",
        "application/vnd.recordare.musicxml+xml",
    )


__all__ = [
    "ScorePreviewError",
    "ScorePreviewUnavailableError",
    "build_score_preview",
    "render_score_download",
]
