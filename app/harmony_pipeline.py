"""Run evidence-aware harmonic inference and publish its canonical artifact."""

from __future__ import annotations

import copy
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Any

from app.analysis import (
    ANALYSIS_JSON_RELATIVE_PATH,
    AudioAnalysisError,
    load_analysis,
)
from app.config import Settings
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HarmonyArtifactError,
    HarmonyArtifactValidationError,
    _restore_harmony_artifact,
    build_harmony_artifact,
    harmony_attempt_artifact_file_name,
    load_harmony_artifact,
    write_harmony_artifact,
)
from app.harmony_inference import (
    HARMONY_INFERENCE_VERSION,
    HarmonyInferenceError,
    HarmonyInferenceResult,
    infer_harmony,
)
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    TranscriptionDraftError,
    load_transcription_draft,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    load_raw_transcription,
)


HARMONY_PIPELINE_VERSION = "harmonic-context-v1"

StageCallback = Callable[[str, str, float], None]
InferenceProcessor = Callable[..., HarmonyInferenceResult]

_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_MAX_PIPELINE_WARNINGS = 64

_INTERPRETATION_UNREADABLE_WARNING = (
    "Published editable interpretation could not be validated; harmony used "
    "canonical raw pitch evidence only."
)
_INTERPRETATION_STALE_WARNING = (
    "Published editable interpretation does not match the current raw "
    "transcription; harmony used canonical raw pitch evidence only."
)
_INTERPRETATION_MALFORMED_WARNING = (
    "Published editable interpretation lacks usable pitched-part context; "
    "harmony used canonical raw pitch evidence only."
)
_NO_PITCHED_EVIDENCE_WARNING = (
    "Raw transcription contains no pitched-note evidence; harmonic context "
    "remains empty."
)


class HarmonyPipelineError(RuntimeError):
    """A job could not publish safe harmonic context."""


@dataclass(frozen=True, slots=True)
class HarmonyPipelineResult:
    version: str
    artifact_file_name: str
    created_at: str
    event_count: int
    segment_count: int
    resolved_segment_count: int
    unresolved_segment_count: int
    unresolved_event_count: int
    used_interpretation_context: bool
    interpretation_version: str | None
    warning_count: int
    warnings: tuple[str, ...]
    payload: dict[str, Any]


def infer_harmony_job(
    job_id: str,
    settings: Settings,
    stage_callback: StageCallback | None = None,
    *,
    version: str = HARMONY_PIPELINE_VERSION,
    created_at: str | None = None,
    inference_processor: InferenceProcessor = infer_harmony,
    attempt_id: str | None = None,
) -> HarmonyPipelineResult:
    """Infer and atomically publish harmonic context from canonical evidence."""
    if not isinstance(settings, Settings):
        raise HarmonyPipelineError("Harmony settings are invalid.")
    pipeline_version = _version(version, "harmony pipeline version")
    timestamp = _created_at(created_at)
    if stage_callback is not None and not callable(stage_callback):
        raise HarmonyPipelineError("Harmony progress callback is invalid.")
    if not callable(inference_processor):
        raise HarmonyPipelineError("Harmony inference processor is invalid.")
    try:
        artifact_file_name = (
            HARMONY_ARTIFACT_RELATIVE_PATH
            if attempt_id is None
            else harmony_attempt_artifact_file_name(attempt_id)
        )
    except HarmonyArtifactError as exc:
        raise HarmonyPipelineError("Harmony attempt identity is invalid.") from exc

    _report(
        stage_callback,
        "loading_raw_transcription",
        "Loading canonical raw pitch evidence.",
        5.0,
    )
    raw = _load_raw(job_id, settings)

    _report(
        stage_callback,
        "loading_analysis_context",
        "Loading matching timing and tonal evidence.",
        18.0,
    )
    analysis_version, timing, tonality = _load_matching_analysis(
        job_id,
        settings,
        raw,
    )

    _report(
        stage_callback,
        "loading_optional_interpretation",
        "Checking optional editable-part context.",
        30.0,
    )
    (
        pitched_part_evidence,
        interpretation_version,
        interpretation_warning,
    ) = _optional_interpretation_context(job_id, settings, raw)

    _report(
        stage_callback,
        "inferring_harmonic_context",
        "Inferring conservative local harmonic candidates.",
        48.0,
    )
    try:
        inference_result = inference_processor(
            copy.deepcopy(raw["pitchedNoteEvents"]),
            copy.deepcopy(timing),
            copy.deepcopy(tonality),
            copy.deepcopy(pitched_part_evidence),
            version=HARMONY_INFERENCE_VERSION,
        )
    except HarmonyInferenceError as exc:
        raise HarmonyPipelineError(
            "Harmonic inference could not interpret the saved pitch evidence safely."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Harmonic inference failed at a protected boundary."
        ) from exc
    if not isinstance(inference_result, HarmonyInferenceResult):
        raise HarmonyPipelineError(
            "Harmony inference processor returned an invalid result."
        )

    pipeline_warnings: list[str] = []
    if interpretation_warning is not None:
        pipeline_warnings.append(interpretation_warning)
    if not raw["pitchedNoteEvents"]:
        pipeline_warnings.append(_NO_PITCHED_EVIDENCE_WARNING)
    combined_warnings = _combined_warnings(
        pipeline_warnings,
        inference_result.warnings,
    )
    inference_result = replace(
        inference_result,
        warnings=tuple(combined_warnings),
    )

    _report(
        stage_callback,
        "validating_harmonic_context",
        "Validating harmonic evidence and provenance.",
        76.0,
    )
    try:
        artifact = build_harmony_artifact(
            inference_result,
            harmony_version=pipeline_version,
            created_at=timestamp,
            transcription_version=raw["transcriptionVersion"],
            analysis_version=analysis_version,
            interpretation_version=interpretation_version,
        )
    except HarmonyArtifactValidationError as exc:
        raise HarmonyPipelineError(
            "Harmonic context failed artifact validation."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Harmonic-context validation failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "saving_harmonic_context",
        "Saving the canonical harmonic-context artifact.",
        92.0,
    )
    reloaded = _publish_and_verify(
        job_id,
        settings,
        artifact,
        artifact_file_name=artifact_file_name,
    )

    _report(
        stage_callback,
        "completed",
        "Harmonic-context artifact complete.",
        100.0,
    )
    diagnostics = reloaded["diagnostics"]
    warnings = tuple(reloaded["warnings"])
    return HarmonyPipelineResult(
        version=pipeline_version,
        artifact_file_name=artifact_file_name,
        created_at=reloaded["createdAt"],
        event_count=diagnostics["eventCount"],
        segment_count=diagnostics["segmentCount"],
        resolved_segment_count=diagnostics["resolvedSegmentCount"],
        unresolved_segment_count=diagnostics["unresolvedSegmentCount"],
        unresolved_event_count=diagnostics["unresolvedEventCount"],
        used_interpretation_context=pitched_part_evidence is not None,
        interpretation_version=interpretation_version,
        warning_count=len(warnings),
        warnings=warnings,
        payload=copy.deepcopy(reloaded),
    )


def _load_raw(job_id: str, settings: Settings) -> dict[str, Any]:
    try:
        raw = load_raw_transcription(job_id, settings)
    except RawTranscriptionError as exc:
        raise HarmonyPipelineError(
            "Published raw transcription is unreadable or unsafe."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Raw transcription loading failed at a protected boundary."
        ) from exc
    if raw is None:
        raise HarmonyPipelineError("Published raw transcription is unavailable.")
    return raw


def _load_matching_analysis(
    job_id: str,
    settings: Settings,
    raw: Mapping[str, Any],
) -> tuple[str, Mapping[str, Any] | None, Mapping[str, Any] | None]:
    try:
        analysis = load_analysis(job_id, settings)
    except AudioAnalysisError as exc:
        raise HarmonyPipelineError(
            "Saved audio analysis is unreadable or unsafe."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Saved audio analysis loading failed at a protected boundary."
        ) from exc
    if not isinstance(analysis, Mapping):
        raise HarmonyPipelineError("Saved audio analysis is unavailable.")
    if analysis.get("schemaVersion") != 1:
        raise HarmonyPipelineError("Saved audio analysis is incompatible.")
    if analysis.get("sourceAsset") != "analysis.wav":
        raise HarmonyPipelineError(
            "Saved audio analysis references an unsupported source."
        )

    source = raw.get("sourceAnalysis")
    if not isinstance(source, Mapping):
        raise HarmonyPipelineError(
            "Raw transcription analysis provenance is malformed."
        )
    if source.get("fileName") != ANALYSIS_JSON_RELATIVE_PATH:
        raise HarmonyPipelineError(
            "Raw transcription analysis provenance is noncanonical."
        )
    analysis_version = _version(
        analysis.get("analysisVersion"),
        "analysis version",
    )
    if source.get("analysisVersion") != analysis_version:
        raise HarmonyPipelineError(
            "Raw transcription analysis provenance is stale."
        )

    timing_value = analysis.get("timing")
    if timing_value is not None and not isinstance(timing_value, Mapping):
        raise HarmonyPipelineError("Saved timing evidence is malformed.")
    tonality_value = analysis.get("tonality")
    if tonality_value is not None and not isinstance(tonality_value, Mapping):
        raise HarmonyPipelineError("Saved tonal evidence is malformed.")
    return analysis_version, timing_value, tonality_value


def _optional_interpretation_context(
    job_id: str,
    settings: Settings,
    raw: Mapping[str, Any],
) -> tuple[dict[str, Any] | None, str | None, str | None]:
    try:
        draft = load_transcription_draft(job_id, settings)
    except TranscriptionDraftError:
        return None, None, _INTERPRETATION_UNREADABLE_WARNING
    except Exception:
        return None, None, _INTERPRETATION_UNREADABLE_WARNING
    if draft is None:
        return None, None, None
    if not _draft_matches_raw(draft, raw):
        return None, None, _INTERPRETATION_STALE_WARNING

    try:
        evidence = draft["interpretationEvidence"]["pitchedPartInference"]
        if not isinstance(evidence, Mapping):
            raise TypeError("invalid pitched-part evidence")
        draft_version = _version(draft["draftVersion"], "interpretation version")
    except (KeyError, TypeError, ValueError, HarmonyPipelineError):
        return None, None, _INTERPRETATION_MALFORMED_WARNING
    return copy.deepcopy(dict(evidence)), draft_version, None


def _draft_matches_raw(
    draft: Mapping[str, Any],
    raw: Mapping[str, Any],
) -> bool:
    try:
        source = draft["sourceTranscription"]
        if not isinstance(source, Mapping):
            return False
        if source.get("fileName") != RAW_TRANSCRIPTION_RELATIVE_PATH:
            return False
        if source.get("schemaVersion") != raw.get("schemaVersion"):
            return False
        if source.get("transcriptionVersion") != raw.get("transcriptionVersion"):
            return False

        provenance = source.get("provenance")
        if not isinstance(provenance, Mapping):
            return False
        expected_algorithms = {
            name: record["version"]
            for name, record in sorted(raw["algorithms"].items())
        }
        if provenance.get("rawSchemaVersion") != raw.get("schemaVersion"):
            return False
        if provenance.get("rawCreatedAt") != raw.get("createdAt"):
            return False
        if provenance.get("analysisVersion") != raw["sourceAnalysis"][
            "analysisVersion"
        ]:
            return False
        if provenance.get("sourceSeparationPresent") != (
            "sourceSeparation" in raw
        ):
            return False
        if provenance.get("rawWarningCount") != len(raw.get("warnings", [])):
            return False
        if provenance.get("algorithmVersions") != expected_algorithms:
            return False
        return source.get("sourceEventIndex") == _source_event_index(raw)
    except (KeyError, TypeError, ValueError):
        return False


def _source_event_index(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in raw["pitchedNoteEvents"]:
        item: dict[str, Any] = {
            "id": event["id"],
            "eventType": "pitched",
            "sourceKind": event["sourceKind"],
            "rawStartSeconds": event["startSeconds"],
            "rawEndSeconds": event["endSeconds"],
            "confidence": event["confidence"],
            "midiNote": event["midiNote"],
            "midiPitch": event["midiPitch"],
            "frequencyHz": event["frequencyHz"],
            "noteName": event["noteName"],
        }
        if "velocity" in event:
            item["velocity"] = event["velocity"]
        if "warnings" in event:
            item["warnings"] = copy.deepcopy(event["warnings"])
        result.append(item)
    for event in raw["percussionEvents"]:
        item = {
            "id": event["id"],
            "eventType": "percussion",
            "sourceKind": event["sourceKind"],
            "rawStartSeconds": event["timeSeconds"],
            "rawEndSeconds": event["timeSeconds"],
            "confidence": event["strength"],
            "strength": event["strength"],
            "hitKinds": [hit["kind"] for hit in event["hits"]],
        }
        if "warnings" in event:
            item["warnings"] = copy.deepcopy(event["warnings"])
        result.append(item)
    result.sort(key=lambda item: (item["rawStartSeconds"], item["id"]))
    return result


def _publish_and_verify(
    job_id: str,
    settings: Settings,
    artifact: Mapping[str, Any],
    *,
    artifact_file_name: str,
) -> dict[str, Any]:
    try:
        previous = load_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=artifact_file_name,
        )
    except HarmonyArtifactError as exc:
        raise HarmonyPipelineError(
            "Existing harmonic context is unreadable or unsafe."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Existing harmonic-context loading failed at a protected boundary."
        ) from exc

    try:
        write_harmony_artifact(
            job_id,
            settings,
            artifact,
            artifact_file_name=artifact_file_name,
        )
    except HarmonyArtifactError as exc:
        raise HarmonyPipelineError(
            "Harmonic context could not be published safely."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Harmonic-context publication failed at a protected boundary."
        ) from exc

    try:
        reloaded = load_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=artifact_file_name,
        )
    except HarmonyArtifactError as exc:
        try:
            _restore_previous(
                job_id,
                settings,
                previous,
                artifact_file_name=artifact_file_name,
            )
        except HarmonyPipelineError as recovery_exc:
            raise recovery_exc from exc
        raise HarmonyPipelineError(
            "Published harmonic context could not be verified."
        ) from exc
    except Exception as exc:
        try:
            _restore_previous(
                job_id,
                settings,
                previous,
                artifact_file_name=artifact_file_name,
            )
        except HarmonyPipelineError as recovery_exc:
            raise recovery_exc from exc
        raise HarmonyPipelineError(
            "Harmonic-context verification failed at a protected boundary."
        ) from exc
    if reloaded is None or reloaded != artifact:
        _restore_previous(
            job_id,
            settings,
            previous,
            artifact_file_name=artifact_file_name,
        )
        raise HarmonyPipelineError(
            "Published harmonic context could not be verified."
        )
    return reloaded


def _restore_previous(
    job_id: str,
    settings: Settings,
    previous: Mapping[str, Any] | None,
    *,
    artifact_file_name: str,
) -> None:
    try:
        _restore_harmony_artifact(
            job_id,
            settings,
            previous,
            artifact_file_name=artifact_file_name,
        )
    except HarmonyArtifactError as exc:
        raise HarmonyPipelineError(
            "Harmonic context verification failed and the previous publication "
            "state could not be restored safely."
        ) from exc
    except Exception as exc:
        raise HarmonyPipelineError(
            "Harmonic-context restoration failed at a protected boundary."
        ) from exc


def _combined_warnings(*groups: Any) -> list[str]:
    output: list[str] = []
    for group in groups:
        for value in group:
            if value not in output:
                output.append(value)
            if len(output) >= _MAX_PIPELINE_WARNINGS:
                return output
    return output


def _version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise HarmonyPipelineError(f"{label.capitalize()} is invalid.")
    return value


def _created_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value or len(value) > 128:
        raise HarmonyPipelineError("Harmony timestamp is invalid.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HarmonyPipelineError("Harmony timestamp is invalid.") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise HarmonyPipelineError("Harmony timestamp must be UTC.")
    return value


def _report(
    callback: StageCallback | None,
    stage: str,
    message: str,
    progress: float,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, message, progress)
    except Exception as exc:
        raise HarmonyPipelineError("Harmony progress reporting failed.") from exc


__all__ = [
    "HARMONY_PIPELINE_VERSION",
    "HarmonyPipelineError",
    "HarmonyPipelineResult",
    "infer_harmony_job",
]
