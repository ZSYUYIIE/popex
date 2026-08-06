"""Compose local detectors into one validated raw-transcription artifact."""

from __future__ import annotations

import copy
import math
import re
import stat
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH, AudioAnalysisError, load_analysis
from app.config import Settings
from app.event_alignment import EventAlignmentError, align_raw_events_to_timing
from app.media import MediaProcessingError, secure_job_dir
from app.percussion_transcription import (
    DEFAULT_ALGORITHM_VERSION as DEFAULT_PERCUSSION_VERSION,
    PercussionTranscriptionError,
    transcribe_percussion_audio,
)
from app.pitch_transcription import PitchedTranscriptionError, transcribe_pitched_audio
from app.separation import (
    REQUIRED_STEM_KINDS,
    STEM_MANIFEST_RELATIVE_PATH,
    StemSeparationError,
    StemSeparationResult,
    load_stem_manifest,
)
from app.separation_artifacts import StemArtifactError, resolve_stem_artifact
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RAW_TRANSCRIPTION_SCHEMA_VERSION,
    RawTranscriptionError,
    RawTranscriptionValidationError,
    validate_raw_transcription,
    write_raw_transcription,
)


TRANSCRIPTION_VERSION = "baseline-pyin-onset-v1"
DEFAULT_PITCH_VERSION = "baseline-pyin-v1"

_PITCHED_STEM_KINDS = ("vocals", "bass", "other")
_STEM_SOURCE_ORDER = {"vocals": 0, "bass": 1, "other": 2, "drums": 3, "full_mix": 4}
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._+-]{0,127}")
_CONTROL_RE = re.compile(r"[\x00-\x1f\x7f]+")
_URL_RE = re.compile(r"(?i)https?://[^\s\]\[<>()\"']+")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?:\b[A-Z]:[\\/]|\\\\)[^\s,;\"']+")
_POSIX_PATH_RE = re.compile(r"(?<![:\w])/(?:[^\s,;\"']+)")
_SPACE_RE = re.compile(r"\s+")
_MAX_WARNINGS = 64
_MAX_WARNING_LENGTH = 240

PitchProcessor = Callable[..., Mapping[str, Any]]
PercussionProcessor = Callable[..., Mapping[str, Any]]
AlignmentProcessor = Callable[..., Mapping[str, Any]]
StageCallback = Callable[[str, str, float], None]


class TranscriptionPipelineError(RuntimeError):
    """A job could not publish a safe musician-facing raw transcription."""


@dataclass(frozen=True, slots=True)
class TranscriptionPipelineResult:
    transcription_version: str
    artifact_file_name: str
    transcribed_at: str
    pitched_event_count: int
    percussion_event_count: int
    aligned_event_count: int
    input_mode: str
    warnings: tuple[str, ...]
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _DetectorOutput:
    source_kind: str
    algorithm_version: str
    events: tuple[dict[str, Any], ...]
    warnings: tuple[str, ...]
    input_origin: str


@dataclass(frozen=True, slots=True)
class _InputSelection:
    analysis_audio: Path
    pitched_inputs: tuple[tuple[str, Path], ...]
    percussion_input: tuple[str, Path]
    separation: StemSeparationResult | None
    warnings: tuple[str, ...]


class _SourceFailure(RuntimeError):
    """One detector source failed without invalidating successful sources."""


def transcribe_job(
    job_id: str,
    settings: Settings,
    stage_callback: StageCallback,
    *,
    transcription_version: str = TRANSCRIPTION_VERSION,
    pitch_processor: PitchProcessor = transcribe_pitched_audio,
    percussion_processor: PercussionProcessor = transcribe_percussion_audio,
    alignment_processor: AlignmentProcessor = align_raw_events_to_timing,
) -> TranscriptionPipelineResult:
    """Transcribe one analyzed job with optional stems and full-mix fallback."""

    version = _version(transcription_version, "transcription version")
    if not callable(stage_callback):
        raise TranscriptionPipelineError("A transcription progress callback is required.")
    if not callable(pitch_processor) or not callable(percussion_processor):
        raise TranscriptionPipelineError("Transcription detector configuration is invalid.")
    if not callable(alignment_processor):
        raise TranscriptionPipelineError("Transcription alignment configuration is invalid.")

    _report(
        stage_callback,
        "selecting_transcription_inputs",
        "Selecting safe transcription audio.",
        5,
    )
    job, job_dir, analysis_version, timing = _required_inputs(job_id, settings)
    selection = _select_inputs(job_id, settings, job, job_dir)
    warnings: list[str] = list(selection.warnings)
    fallback_codes: list[str] = []

    _report(
        stage_callback,
        "detecting_pitched_events",
        "Detecting raw pitched-note candidates.",
        25,
    )
    pitch_outputs: list[_DetectorOutput] = []
    used_stem_sources: set[str] = set()
    if selection.separation is not None:
        failed_stem_pitch = 0
        for source_kind, path in selection.pitched_inputs:
            try:
                output = _pitch_output(
                    pitch_processor,
                    path,
                    source_kind,
                    analysis_version,
                    input_origin="stem",
                )
            except _SourceFailure:
                failed_stem_pitch += 1
                _append_warning(
                    warnings,
                    f"Pitched transcription for the {source_kind} stem failed; successful sources were preserved.",
                )
                continue
            pitch_outputs.append(output)
            used_stem_sources.add(source_kind)
            _extend_detector_warnings(warnings, output)

        if failed_stem_pitch == len(selection.pitched_inputs):
            fallback_codes.append("all_pitched_stems_to_full_mix")
            _append_warning(
                warnings,
                "All separated pitched sources failed; the full mix was attempted once for pitched transcription.",
            )
            try:
                output = _pitch_output(
                    pitch_processor,
                    selection.analysis_audio,
                    "full_mix",
                    analysis_version,
                    input_origin="full_mix_fallback",
                )
            except _SourceFailure:
                _append_warning(
                    warnings,
                    "Full-mix pitched transcription also failed; percussion results may still be published.",
                )
            else:
                pitch_outputs.append(output)
                _extend_detector_warnings(warnings, output)
    else:
        try:
            output = _pitch_output(
                pitch_processor,
                selection.analysis_audio,
                "full_mix",
                analysis_version,
                input_origin="full_mix",
            )
        except _SourceFailure:
            _append_warning(
                warnings,
                "Full-mix pitched transcription failed; percussion results may still be published.",
            )
        else:
            pitch_outputs.append(output)
            _extend_detector_warnings(warnings, output)

    _report(
        stage_callback,
        "detecting_percussion_events",
        "Detecting raw percussion candidates.",
        55,
    )
    percussion_outputs: list[_DetectorOutput] = []
    percussion_kind, percussion_path = selection.percussion_input
    percussion_origin = "stem" if selection.separation is not None else "full_mix"
    try:
        output = _percussion_output(
            percussion_processor,
            percussion_path,
            percussion_kind,
            analysis_version,
            input_origin=percussion_origin,
        )
    except _SourceFailure:
        if selection.separation is not None:
            fallback_codes.append("drums_stem_to_full_mix")
            _append_warning(
                warnings,
                "Drums-stem transcription failed; the full mix was attempted once for percussion.",
            )
            try:
                output = _percussion_output(
                    percussion_processor,
                    selection.analysis_audio,
                    "full_mix",
                    analysis_version,
                    input_origin="full_mix_fallback",
                )
            except _SourceFailure:
                _append_warning(
                    warnings,
                    "Full-mix percussion transcription also failed; pitched results may still be published.",
                )
            else:
                percussion_outputs.append(output)
                _extend_detector_warnings(warnings, output)
        else:
            _append_warning(
                warnings,
                "Full-mix percussion transcription failed; pitched results may still be published.",
            )
    else:
        percussion_outputs.append(output)
        if selection.separation is not None:
            used_stem_sources.add("drums")
        _extend_detector_warnings(warnings, output)

    if not pitch_outputs and not percussion_outputs:
        raise TranscriptionPipelineError(
            "Local transcription detectors could not produce a valid result."
        )

    pitched_events = _compose_pitched_events(pitch_outputs)
    percussion_events = _compose_percussion_events(percussion_outputs)
    if not pitched_events:
        _append_warning(warnings, "Transcription completed without pitched-note candidates.")
    if not percussion_events:
        _append_warning(warnings, "Transcription completed without percussion candidates.")

    _report(
        stage_callback,
        "aligning_transcription_events",
        "Aligning raw events to saved timing evidence.",
        75,
    )
    try:
        alignment = alignment_processor(pitched_events, percussion_events, timing)
        alignment_version, alignment_candidates, alignment_warnings, aligned_count = (
            _validate_alignment_output(alignment)
        )
    except (EventAlignmentError, RawTranscriptionValidationError, TypeError, ValueError) as exc:
        raise TranscriptionPipelineError(
            "Saved timing evidence could not align the raw transcription safely."
        ) from exc
    except Exception as exc:
        raise TranscriptionPipelineError(
            "Raw transcription alignment failed at a protected boundary."
        ) from exc
    for item in alignment_warnings:
        _append_warning(warnings, item)

    input_mode = _input_mode(
        selection.separation,
        used_stem_sources,
        pitch_outputs,
        percussion_outputs,
    )
    created_at = _utc_now()
    payload: dict[str, Any] = {
        "schemaVersion": RAW_TRANSCRIPTION_SCHEMA_VERSION,
        "transcriptionVersion": version,
        "createdAt": created_at,
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": analysis_version,
        },
        "algorithms": _algorithms(
            version,
            input_mode,
            fallback_codes,
            pitch_outputs,
            percussion_outputs,
            alignment_version,
            aligned_count,
        ),
        "pitchedNoteEvents": pitched_events,
        "percussionEvents": percussion_events,
        "alignmentCandidates": alignment_candidates,
        "warnings": _warnings(warnings),
    }
    if selection.separation is not None and used_stem_sources:
        payload["sourceSeparation"] = _source_separation(selection.separation)

    try:
        validated = validate_raw_transcription(payload)
    except RawTranscriptionValidationError as exc:
        raise TranscriptionPipelineError(
            "Raw transcription output failed schema validation."
        ) from exc

    _report(
        stage_callback,
        "saving_transcription",
        "Saving the raw transcription artifact.",
        92,
    )
    try:
        write_raw_transcription(job_id, settings, validated)
    except RawTranscriptionError as exc:
        raise TranscriptionPipelineError(
            "Raw transcription could not be published safely."
        ) from exc
    except Exception as exc:
        raise TranscriptionPipelineError(
            "Raw transcription publication failed at a protected boundary."
        ) from exc

    return TranscriptionPipelineResult(
        transcription_version=version,
        artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        transcribed_at=created_at,
        pitched_event_count=len(validated["pitchedNoteEvents"]),
        percussion_event_count=len(validated["percussionEvents"]),
        aligned_event_count=aligned_count,
        input_mode=input_mode,
        warnings=tuple(validated["warnings"]),
        payload=validated,
    )


def _required_inputs(
    job_id: str,
    settings: Settings,
) -> tuple[dict[str, Any], Path, str, Mapping[str, Any]]:
    if not isinstance(settings, Settings):
        raise TranscriptionPipelineError("Transcription settings are invalid.")
    try:
        job = db.get_job(settings.database_path, job_id)
    except Exception as exc:
        raise TranscriptionPipelineError("The transcription job could not be loaded.") from exc
    if job is None:
        raise TranscriptionPipelineError("The transcription job does not exist.")
    if job.get("preparation_status") != "completed" or job.get("analysis_status") != "completed":
        raise TranscriptionPipelineError(
            "Source preparation and audio analysis must be complete before transcription."
        )
    if job.get("analysis_json_file_name") != ANALYSIS_JSON_RELATIVE_PATH:
        raise TranscriptionPipelineError("Saved audio analysis is unavailable or stale.")

    try:
        job_dir = secure_job_dir(settings, job_id)
    except (MediaProcessingError, OSError) as exc:
        raise TranscriptionPipelineError("The transcription job workspace is unavailable.") from exc
    _safe_regular_file(job_dir, PurePosixPath("analysis.wav"), "Analysis audio")
    _safe_regular_file(
        job_dir,
        PurePosixPath(*ANALYSIS_JSON_RELATIVE_PATH.split("/")),
        "Saved audio analysis",
    )
    try:
        analysis = load_analysis(job_id, settings)
    except (AudioAnalysisError, MediaProcessingError, OSError) as exc:
        raise TranscriptionPipelineError("Saved audio analysis is unreadable or unsafe.") from exc
    if not isinstance(analysis, dict):
        raise TranscriptionPipelineError("Saved audio analysis is unavailable.")
    analysis_version = _version(analysis.get("analysisVersion"), "analysis version")
    if job.get("analysis_version") != analysis_version:
        raise TranscriptionPipelineError("Saved audio analysis provenance is stale.")
    if analysis.get("sourceAsset") != "analysis.wav":
        raise TranscriptionPipelineError("Saved audio analysis references an unsupported source.")
    timing = analysis.get("timing")
    if not isinstance(timing, Mapping):
        raise TranscriptionPipelineError("Saved timing evidence is unavailable or malformed.")
    return job, job_dir, analysis_version, timing


def _select_inputs(
    job_id: str,
    settings: Settings,
    job: Mapping[str, Any],
    job_dir: Path,
) -> _InputSelection:
    analysis_audio = _safe_regular_file(
        job_dir,
        PurePosixPath("analysis.wav"),
        "Analysis audio",
    )
    fallback_warning = (
        "Valid separated stems were unavailable; transcription used analysis.wav as the full mix."
    )
    if (
        job.get("separation_status") != "completed"
        or job.get("stem_manifest_file_name") != STEM_MANIFEST_RELATIVE_PATH
    ):
        return _InputSelection(
            analysis_audio=analysis_audio,
            pitched_inputs=(("full_mix", analysis_audio),),
            percussion_input=("full_mix", analysis_audio),
            separation=None,
            warnings=(fallback_warning,),
        )

    try:
        separation = load_stem_manifest(job_id, settings)
        if separation is None:
            raise StemSeparationError("Stem manifest unavailable")
        if (
            job.get("separation_version") != separation.separation_version
            or job.get("separation_model") != separation.model_name
            or tuple(stem.kind for stem in separation.stems) != REQUIRED_STEM_KINDS
        ):
            raise StemSeparationError("Stem provenance is stale")
        resolved = {
            kind: resolve_stem_artifact(job_id, kind, settings, job).path
            for kind in REQUIRED_STEM_KINDS
        }
    except (StemArtifactError, StemSeparationError, MediaProcessingError, OSError):
        return _InputSelection(
            analysis_audio=analysis_audio,
            pitched_inputs=(("full_mix", analysis_audio),),
            percussion_input=("full_mix", analysis_audio),
            separation=None,
            warnings=(
                "Published stems failed current safety or provenance validation; transcription used analysis.wav as the full mix.",
            ),
        )

    return _InputSelection(
        analysis_audio=analysis_audio,
        pitched_inputs=tuple((kind, resolved[kind]) for kind in _PITCHED_STEM_KINDS),
        percussion_input=("drums", resolved["drums"]),
        separation=separation,
        warnings=tuple(
            _safe_warning(f"Stem separation: {item}") for item in separation.warnings
        ),
    )


def _pitch_output(
    processor: PitchProcessor,
    path: Path,
    source_kind: str,
    analysis_version: str,
    *,
    input_origin: str,
) -> _DetectorOutput:
    try:
        result = processor(path, source_kind=source_kind)
    except (PitchedTranscriptionError, Exception) as exc:
        raise _SourceFailure("pitched source failed") from exc
    return _detector_output(
        result,
        source_kind,
        "pitched",
        analysis_version,
        input_origin=input_origin,
    )


def _percussion_output(
    processor: PercussionProcessor,
    path: Path,
    source_kind: str,
    analysis_version: str,
    *,
    input_origin: str,
) -> _DetectorOutput:
    try:
        result = processor(path, source_kind=source_kind)
    except (PercussionTranscriptionError, Exception) as exc:
        raise _SourceFailure("percussion source failed") from exc
    return _detector_output(
        result,
        source_kind,
        "percussion",
        analysis_version,
        input_origin=input_origin,
    )


def _detector_output(
    result: object,
    source_kind: str,
    event_type: str,
    analysis_version: str,
    *,
    input_origin: str,
) -> _DetectorOutput:
    if not isinstance(result, Mapping):
        raise _SourceFailure("detector result is invalid")
    try:
        algorithm_version = _version(result.get("algorithmVersion"), "detector version")
    except TranscriptionPipelineError as exc:
        raise _SourceFailure("detector version is invalid") from exc
    if result.get("sourceKind") != source_kind:
        raise _SourceFailure("detector source identity changed")
    raw_events = result.get("events")
    if isinstance(raw_events, (str, bytes, bytearray)) or not isinstance(raw_events, Sequence):
        raise _SourceFailure("detector events are invalid")
    raw_warnings = result.get("warnings", [])
    if isinstance(raw_warnings, (str, bytes, bytearray)) or not isinstance(raw_warnings, Sequence):
        raise _SourceFailure("detector warnings are invalid")
    warnings: list[str] = []
    for item in raw_warnings:
        if not isinstance(item, str):
            raise _SourceFailure("detector warning is invalid")
        _append_warning(warnings, item)
    diagnostics = result.get("diagnostics", {})
    if not isinstance(diagnostics, Mapping):
        raise _SourceFailure("detector diagnostics are invalid")

    enriched = _enrich_events(raw_events, diagnostics, source_kind, event_type)
    dummy: dict[str, Any] = {
        "schemaVersion": RAW_TRANSCRIPTION_SCHEMA_VERSION,
        "transcriptionVersion": TRANSCRIPTION_VERSION,
        "createdAt": "2000-01-01T00:00:00+00:00",
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "analysisVersion": analysis_version,
        },
        "algorithms": {"detector": {"version": algorithm_version}},
        "pitchedNoteEvents": enriched if event_type == "pitched" else [],
        "percussionEvents": enriched if event_type == "percussion" else [],
        "alignmentCandidates": [],
        "warnings": [],
    }
    try:
        validated = validate_raw_transcription(dummy)
    except RawTranscriptionValidationError as exc:
        raise _SourceFailure("detector events failed schema validation") from exc
    key = "pitchedNoteEvents" if event_type == "pitched" else "percussionEvents"
    return _DetectorOutput(
        source_kind=source_kind,
        algorithm_version=algorithm_version,
        events=tuple(validated[key]),
        warnings=tuple(warnings),
        input_origin=input_origin,
    )


def _enrich_events(
    events: Sequence[Any],
    diagnostics: Mapping[str, Any],
    source_kind: str,
    event_type: str,
) -> list[dict[str, Any]]:
    pitch_evidence = _pitch_evidence(diagnostics) if event_type == "pitched" else {}
    output: list[dict[str, Any]] = []
    for item in events:
        if not isinstance(item, Mapping):
            raise _SourceFailure("detector event is invalid")
        event = copy.deepcopy(dict(item))
        original_id = event.get("id")
        if not isinstance(original_id, str):
            raise _SourceFailure("detector event ID is invalid")
        if event.get("sourceKind") != source_kind:
            raise _SourceFailure("detector event source identity changed")
        feature_key = "rawFeatures" if "rawFeatures" in event else "rawFeatureSummary"
        if "rawFeatures" in event and "rawFeatureSummary" in event:
            raise _SourceFailure("detector event has duplicate raw feature containers")
        existing = event.get(feature_key, {})
        if not isinstance(existing, Mapping):
            raise _SourceFailure("detector raw evidence is invalid")
        feature = copy.deepcopy(dict(existing))
        feature["detectorEventId"] = original_id
        feature["detectorSourceKind"] = source_kind
        for key, value in pitch_evidence.get(original_id, {}).items():
            feature[key] = value
        event[feature_key] = feature
        output.append(event)
    return output


def _pitch_evidence(diagnostics: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    eventing = diagnostics.get("eventing")
    if not isinstance(eventing, Mapping):
        return {}
    values = eventing.get("eventEvidence")
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence):
        return {}
    result: dict[str, dict[str, Any]] = {}
    allowed = {
        "firstFrameIndex",
        "lastFrameIndex",
        "voicedFrameCount",
        "meanVoicedProbability",
        "pitchMadCents",
    }
    for item in values:
        if not isinstance(item, Mapping) or not isinstance(item.get("eventId"), str):
            continue
        evidence: dict[str, Any] = {}
        for key in allowed:
            value = item.get(key)
            if isinstance(value, bool):
                continue
            if isinstance(value, int):
                evidence[key] = value
            elif isinstance(value, float) and math.isfinite(value):
                evidence[key] = value
        result[item["eventId"]] = evidence
    return result


def _compose_pitched_events(outputs: Sequence[_DetectorOutput]) -> list[dict[str, Any]]:
    indexed: list[tuple[float, float, int, str, dict[str, Any]]] = []
    for output in outputs:
        priority = _STEM_SOURCE_ORDER.get(output.source_kind, 99)
        for event in output.events:
            indexed.append(
                (
                    float(event["startSeconds"]),
                    float(event["endSeconds"]),
                    priority,
                    str(event["id"]),
                    copy.deepcopy(event),
                )
            )
    indexed.sort(key=lambda item: item[:4])
    result: list[dict[str, Any]] = []
    for index, (*_, event) in enumerate(indexed, start=1):
        event["id"] = f"p{index:06d}"
        result.append(event)
    return result


def _compose_percussion_events(outputs: Sequence[_DetectorOutput]) -> list[dict[str, Any]]:
    indexed: list[tuple[float, int, str, dict[str, Any]]] = []
    for output in outputs:
        priority = _STEM_SOURCE_ORDER.get(output.source_kind, 99)
        for event in output.events:
            indexed.append(
                (
                    float(event["timeSeconds"]),
                    priority,
                    str(event["id"]),
                    copy.deepcopy(event),
                )
            )
    indexed.sort(key=lambda item: item[:3])
    result: list[dict[str, Any]] = []
    for index, (*_, event) in enumerate(indexed, start=1):
        event["id"] = f"r{index:06d}"
        result.append(event)
    return result


def _validate_alignment_output(
    value: object,
) -> tuple[str, list[dict[str, Any]], tuple[str, ...], int]:
    if not isinstance(value, Mapping):
        raise EventAlignmentError("Alignment output must be a mapping.")
    try:
        version = _version(value.get("alignmentVersion"), "alignment version")
    except TranscriptionPipelineError as exc:
        raise EventAlignmentError("Alignment version is invalid.") from exc
    candidates = value.get("candidates")
    if isinstance(candidates, (str, bytes, bytearray)) or not isinstance(candidates, Sequence):
        raise EventAlignmentError("Alignment candidates must be a sequence.")
    normalized_candidates: list[dict[str, Any]] = []
    for item in candidates:
        if not isinstance(item, Mapping):
            raise EventAlignmentError("Alignment candidate must be a mapping.")
        normalized_candidates.append(copy.deepcopy(dict(item)))
    raw_warnings = value.get("warnings", [])
    if isinstance(raw_warnings, (str, bytes, bytearray)) or not isinstance(raw_warnings, Sequence):
        raise EventAlignmentError("Alignment warnings must be a sequence.")
    warnings: list[str] = []
    for item in raw_warnings:
        if not isinstance(item, str):
            raise EventAlignmentError("Alignment warning must be text.")
        _append_warning(warnings, item)
    aligned_count = sum(1 for item in normalized_candidates if "alignedTimeSeconds" in item)
    diagnostics = value.get("diagnostics", {})
    if isinstance(diagnostics, Mapping):
        claimed = diagnostics.get("alignedCount")
        if type(claimed) is int and claimed >= 0 and claimed != aligned_count:
            raise EventAlignmentError("Alignment diagnostics do not match candidates.")
    return version, normalized_candidates, tuple(warnings), aligned_count


def _algorithms(
    pipeline_version: str,
    input_mode: str,
    fallback_codes: Sequence[str],
    pitch_outputs: Sequence[_DetectorOutput],
    percussion_outputs: Sequence[_DetectorOutput],
    alignment_version: str,
    aligned_count: int,
) -> dict[str, Any]:
    return {
        "eventAlignment": {
            "version": alignment_version,
            "rawTimesPreserved": True,
            "alignedEventCount": aligned_count,
        },
        "percussionDetection": _detector_algorithm_record(
            percussion_outputs,
            DEFAULT_PERCUSSION_VERSION,
        ),
        "pitchTracking": _detector_algorithm_record(
            pitch_outputs,
            DEFAULT_PITCH_VERSION,
        ),
        "transcriptionPipeline": {
            "version": pipeline_version,
            "inputMode": input_mode,
            "fallbacks": list(fallback_codes),
            "demucsRequired": False,
        },
    }


def _detector_algorithm_record(
    outputs: Sequence[_DetectorOutput],
    default_version: str,
) -> dict[str, Any]:
    versions = sorted({item.algorithm_version for item in outputs})
    aggregate = versions[0] if len(versions) == 1 else ("mixed" if versions else default_version)
    sources: dict[str, Any] = {}
    for item in sorted(
        outputs,
        key=lambda value: _STEM_SOURCE_ORDER.get(value.source_kind, 99),
    ):
        sources[item.source_kind] = {
            "version": item.algorithm_version,
            "inputOrigin": item.input_origin,
            "eventCount": len(item.events),
            "warningCount": len(item.warnings),
        }
    return {
        "version": aggregate,
        "sourceVersions": {
            item.source_kind: item.algorithm_version for item in outputs
        },
        "sources": sources,
        "eventCount": sum(len(item.events) for item in outputs),
    }


def _source_separation(result: StemSeparationResult) -> dict[str, Any]:
    model = result.payload.get("model")
    if not isinstance(model, Mapping):
        raise TranscriptionPipelineError("Saved stem provenance is unavailable.")
    return {
        "fileName": STEM_MANIFEST_RELATIVE_PATH,
        "separationVersion": result.separation_version,
        "model": copy.deepcopy(dict(model)),
    }


def _input_mode(
    separation: StemSeparationResult | None,
    used_stem_sources: set[str],
    pitch_outputs: Sequence[_DetectorOutput],
    percussion_outputs: Sequence[_DetectorOutput],
) -> str:
    if separation is None:
        return "full_mix"
    all_outputs = [*pitch_outputs, *percussion_outputs]
    used_full_mix = any(item.input_origin.startswith("full_mix") for item in all_outputs)
    if used_stem_sources and used_full_mix:
        return "stems_with_full_mix_fallback"
    if used_stem_sources:
        return "separated_stems"
    return "full_mix_fallback"


def _safe_regular_file(job_dir: Path, relative: PurePosixPath, label: str) -> Path:
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise TranscriptionPipelineError(f"{label} is unavailable or unsafe.")
    current = job_dir
    try:
        for index, part in enumerate(relative.parts):
            current = current / part
            mode = current.lstat().st_mode
            if stat.S_ISLNK(mode):
                raise TranscriptionPipelineError(f"{label} is unavailable or unsafe.")
            if index < len(relative.parts) - 1 and not stat.S_ISDIR(mode):
                raise TranscriptionPipelineError(f"{label} is unavailable or unsafe.")
        if not stat.S_ISREG(current.lstat().st_mode):
            raise TranscriptionPipelineError(f"{label} is unavailable or unsafe.")
        root = job_dir.resolve(strict=True)
        resolved = current.resolve(strict=True)
    except FileNotFoundError:
        raise TranscriptionPipelineError(f"{label} is missing.") from None
    except TranscriptionPipelineError:
        raise
    except OSError:
        raise TranscriptionPipelineError(f"{label} could not be inspected safely.") from None
    if root not in resolved.parents:
        raise TranscriptionPipelineError(f"{label} is unavailable or unsafe.")
    if resolved.stat().st_size <= 0:
        raise TranscriptionPipelineError(f"{label} is empty.")
    return resolved


def _report(callback: StageCallback, stage: str, message: str, progress: float) -> None:
    try:
        callback(stage, message, progress)
    except Exception as exc:
        raise TranscriptionPipelineError(
            "Transcription progress could not be reported safely."
        ) from exc


def _version(value: object, label: str) -> str:
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not _VERSION_RE.fullmatch(value)
    ):
        raise TranscriptionPipelineError(f"The {label} is invalid.")
    return value


def _extend_detector_warnings(warnings: list[str], output: _DetectorOutput) -> None:
    for item in output.warnings:
        _append_warning(warnings, f"{output.source_kind}: {item}")


def _append_warning(warnings: list[str], value: str) -> None:
    safe = _safe_warning(value)
    if safe and safe not in warnings and len(warnings) < _MAX_WARNINGS:
        warnings.append(safe)


def _warnings(values: Sequence[str]) -> list[str]:
    output: list[str] = []
    for item in values:
        _append_warning(output, item)
    return output


def _safe_warning(value: str) -> str:
    text = _CONTROL_RE.sub(" ", str(value))
    text = _URL_RE.sub("[redacted]", text)
    text = _WINDOWS_PATH_RE.sub("[redacted]", text)
    text = _POSIX_PATH_RE.sub("[redacted]", text)
    text = _SPACE_RE.sub(" ", text).strip()
    if len(text) > _MAX_WARNING_LENGTH:
        text = text[: _MAX_WARNING_LENGTH - 1].rstrip() + "…"
    return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
