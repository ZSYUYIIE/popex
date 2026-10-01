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
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from app.analysis import (
    ANALYSIS_JSON_RELATIVE_PATH,
)
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
_MAX_ANALYSIS_BYTES = 8 * 1024 * 1024


def _analysis_directory_state(paths: tuple[Path, ...]) -> tuple:
    state = []
    for path in paths:
        info = path.lstat()
        if not stat.S_ISDIR(info.st_mode):
            raise ScorePreviewError("Saved audio analysis directory is unsafe.")
        state.append((info.st_dev, info.st_ino, info.st_mode))
    return tuple(state)


def _analysis_file_state(info: os.stat_result) -> tuple:
    state = (info.st_dev, info.st_ino, info.st_mode, info.st_size, info.st_mtime_ns)
    # Windows lstat and fstat can report different creation-time values for
    # the same unchanged file. ctime is a mutation guard only on POSIX.
    return state if os.name == "nt" else (*state, info.st_ctime_ns)


def _reject_analysis_constant(value: str) -> None:
    raise ValueError("Non-finite analysis JSON is invalid.")


def _load_score_analysis(job_id: str, settings: Settings) -> Mapping | None:
    """Read bounded, stable analysis evidence after raw loader validates the ID."""
    job_dir = settings.exports_dir / job_id
    analysis_dir = job_dir / "analysis"
    path = analysis_dir / "audio-analysis.json"
    directories = (settings.exports_dir, job_dir, analysis_dir)
    try:
        directory_state = _analysis_directory_state(directories)
        before = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ScorePreviewError("Saved audio analysis is unavailable.") from exc
    if not stat.S_ISREG(before.st_mode) or before.st_size > _MAX_ANALYSIS_BYTES:
        raise ScorePreviewError("Saved audio analysis is unsafe or too large.")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_BINARY", 0))
    try:
        with os.fdopen(os.open(path, flags), "rb") as reader:
            opened = os.fstat(reader.fileno())
            if _analysis_file_state(opened) != _analysis_file_state(before):
                raise ScorePreviewError("Saved audio analysis changed during validation.")
            data = reader.read(_MAX_ANALYSIS_BYTES + 1)
            after_read = os.fstat(reader.fileno())
        if (
            len(data) > _MAX_ANALYSIS_BYTES
            or _analysis_file_state(before) != _analysis_file_state(after_read)
            or _analysis_file_state(before) != _analysis_file_state(path.lstat())
            or directory_state != _analysis_directory_state(directories)
        ):
            raise ScorePreviewError("Saved audio analysis changed during validation.")
    except OSError as exc:
        raise ScorePreviewError("Saved audio analysis could not be read safely.") from exc
    try:
        payload = json.loads(data.decode("utf-8"), parse_constant=_reject_analysis_constant)
    except (UnicodeError, ValueError, RecursionError) as exc:
        raise ScorePreviewError("Saved audio analysis is unreadable.") from exc
    if (
        not isinstance(payload, Mapping)
        or type(payload.get("schemaVersion")) is not int
        or payload["schemaVersion"] != 1
    ):
        raise ScorePreviewError("Saved audio analysis schema is invalid.")
    return payload


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
    if (
        record.get("preparation_status") != "completed"
        or record.get("analysis_status") != "completed"
    ):
        raise ScorePreviewUnavailableError(
            "Completed source preparation and audio analysis are required before score preview."
        )
    if (
        record.get("analysis_json_file_name") != ANALYSIS_JSON_RELATIVE_PATH
        or not isinstance(record.get("analysis_version"), str)
        or not record.get("analysis_version")
    ):
        raise ScorePreviewUnavailableError(
            "A published versioned audio analysis is required before score preview."
        )
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


def collect_score_evidence(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Load and cross-check the raw transcription and analysis a score uses.

    Returns detached evidence: the validated raw transcription, analysis, the
    builder's pitched-note inputs, and explicit timing evidence. Raises
    :class:`ScorePreviewUnavailableError` for missing evidence and
    :class:`ScorePreviewError` for invalid or mismatched evidence.
    """
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
    source_analysis = raw_transcription.get("sourceAnalysis")
    if (
        not isinstance(source_analysis, Mapping)
        or raw_transcription.get("transcriptionVersion")
        != record.get("transcription_version")
        or raw_transcription.get("createdAt") != record.get("transcribed_at")
        or source_analysis.get("fileName") != ANALYSIS_JSON_RELATIVE_PATH
        or source_analysis.get("analysisVersion") != record.get("analysis_version")
    ):
        raise ScorePreviewError(
            "Published raw transcription does not match the job's current evidence."
        )
    aligned_count = sum(
        1
        for candidate in raw_transcription.get("alignmentCandidates", ())
        if isinstance(candidate, Mapping) and "alignedTimeSeconds" in candidate
    )
    expected_counts = (
        ("pitched_event_count", len(raw_transcription.get("pitchedNoteEvents", ()))),
        ("percussion_event_count", len(raw_transcription.get("percussionEvents", ()))),
        ("aligned_event_count", aligned_count),
    )
    if any(
        isinstance(record.get(field), bool)
        or not isinstance(record.get(field), int)
        or record.get(field) != count
        for field, count in expected_counts
    ):
        raise ScorePreviewError(
            "Published raw transcription counts do not match the job record."
        )
    analysis = _load_score_analysis(job_id, settings)
    if analysis is None:
        raise ScorePreviewUnavailableError(
            "Saved audio analysis is unavailable; tempo evidence is required."
        )
    if not isinstance(analysis, Mapping):
        raise ScorePreviewError("Saved audio analysis is invalid.")
    if (
        analysis.get("analysisVersion") != record.get("analysis_version")
        or analysis.get("createdAt") != record.get("analyzed_at")
        or analysis.get("sourceAsset") != "analysis.wav"
    ):
        raise ScorePreviewError(
            "Saved audio analysis does not match the job's current evidence."
        )
    timing = analysis.get("timing")
    if not isinstance(timing, Mapping):
        raise ScorePreviewError("Saved audio analysis timing is invalid.")
    tempo = timing.get("tempoBpm")
    if isinstance(tempo, bool) or not isinstance(tempo, (int, float)):
        raise ScorePreviewUnavailableError(
            "Tempo evidence is unavailable; score preview requires analysis tempo."
        )
    tempo_stable = timing.get("tempoStable")
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
    return {
        "rawTranscription": raw_transcription,
        "analysis": analysis,
        "pitchedInputs": pitched_inputs,
        "tempoBpm": float(tempo),
        "beatsPerMeasure": beats_per_measure,
        "meterSource": meter_source,
        "tempoConfidence": _optional_confidence(timing.get("tempoConfidence")),
        "meterConfidence": _optional_confidence(timing.get("meterConfidence")),
        "tempoStable": tempo_stable if isinstance(tempo_stable, bool) else None,
        "percussionEventCount": len(raw_transcription.get("percussionEvents", ())),
    }


def score_evidence_warnings(
    document: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    single_draft_part_warning: bool = True,
    percussion_notated: bool = False,
) -> list[str]:
    """Return builder warnings plus explicit timing and omitted-layer warnings."""
    warnings = list(document["warnings"])
    if single_draft_part_warning:
        warnings.append(
            "Pitched events are shown as one draft part; instrument-specific part "
            "assignment is not included."
        )
    tempo_confidence = evidence["tempoConfidence"]
    meter_confidence = evidence["meterConfidence"]
    meter_source = evidence["meterSource"]
    if tempo_confidence is None:
        warnings.append(
            "Tempo confidence is unavailable; review score timing and measure placement."
        )
    elif tempo_confidence < 0.50:
        warnings.append(
            "Tempo confidence is below 0.50; review score timing and measure placement."
        )
    if evidence["tempoStable"] is False:
        warnings.append(
            "The estimated tempo is unstable; review score timing and measure placement."
        )
    if meter_source == "analysis" and (
        meter_confidence is None or meter_confidence < 0.50
    ):
        warnings.append(
            "Meter confidence is low or unavailable; review the measure grouping."
        )
    if evidence["percussionEventCount"] and not percussion_notated:
        warnings.append(
            "Percussion events are present but are not rendered in this pitched-note draft."
        )
    if meter_source != "analysis":
        warnings.append(
            "Meter evidence is unavailable or weak; measures use an explicit "
            "4/4 draft grid and require musician review."
        )
    return warnings


def build_score_preview(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    *,
    include_measures: bool = False,
) -> dict[str, Any]:
    """Build a detached score-preview payload from published artifacts."""
    evidence = collect_score_evidence(job_id, settings, record)
    raw_transcription = evidence["rawTranscription"]
    analysis = evidence["analysis"]
    try:
        document = build_score_document(
            evidence["pitchedInputs"],
            tempo_bpm=evidence["tempoBpm"],
            beats_per_measure=evidence["beatsPerMeasure"],
        )
    except ScoreConstructionError as exc:
        raise ScorePreviewError(
            "Score preview could not be constructed from saved evidence."
        ) from exc

    warnings = score_evidence_warnings(document, evidence)
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
        "meterSource": evidence["meterSource"],
        "measureCount": document["measureCount"],
        "noteCount": document["noteCount"],
        "percussionEventCount": evidence["percussionEventCount"],
        "warnings": warnings,
        "timingEvidence": {
            "tempoConfidence": evidence["tempoConfidence"],
            "tempoStable": evidence["tempoStable"],
            "meterConfidence": evidence["meterConfidence"],
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
    "collect_score_evidence",
    "score_evidence_warnings",
    "render_score_download",
]
