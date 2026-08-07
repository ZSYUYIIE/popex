"""Compose validated raw transcription into an editable interpretation draft."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from app.analysis import AudioAnalysisError, load_analysis
from app.config import Settings
from app.percussion_interpretation import (
    PERCUSSION_INTERPRETATION_VERSION,
    PercussionInterpretationError,
    PercussionInterpretationResult,
    interpret_percussion,
)
from app.pitched_part_inference import (
    PITCHED_PART_INFERENCE_VERSION,
    PitchedPartInferenceError,
    PitchedPartInferenceResult,
    infer_pitched_parts,
)
from app.rhythm_interpretation import (
    RHYTHM_INTERPRETATION_VERSION,
    RhythmInterpretationError,
    RhythmInterpretationResult,
    interpret_rhythm,
)
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    TranscriptionDraftError,
    TranscriptionDraftValidationError,
    load_transcription_draft,
    validate_transcription_draft,
    write_transcription_draft,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RawTranscriptionError,
    load_raw_transcription,
)


INTERPRETATION_PIPELINE_VERSION = "editable-interpretation-v1"

StageCallback = Callable[[str, str, float], None]

_MAX_SYNTHESIS_WARNINGS = 128
_MAX_ITEM_ALTERNATIVES = 16
_MAX_CONTAINER_REFS = 256
_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,95}")


class InterpretationPipelineError(RuntimeError):
    """A raw transcription could not be converted into an editable draft safely."""


@dataclass(frozen=True, slots=True)
class InterpretationPipelineResult:
    version: str
    draft_file_name: str
    created_at: str
    part_count: int
    phrase_count: int
    pitched_item_count: int
    percussion_item_count: int
    warning_count: int
    warnings: tuple[str, ...]
    payload: dict[str, Any]


def interpret_transcription_job(
    job_id: str,
    settings: Settings,
    stage_callback: StageCallback | None = None,
    *,
    version: str = INTERPRETATION_PIPELINE_VERSION,
    created_at: str | None = None,
) -> InterpretationPipelineResult:
    """Convert one canonical raw-transcription artifact into an editable draft."""
    if not isinstance(settings, Settings):
        raise InterpretationPipelineError("Interpretation settings are invalid.")
    if not isinstance(version, str) or not version or len(version) > 128:
        raise InterpretationPipelineError("Interpretation pipeline version is invalid.")
    if stage_callback is not None and not callable(stage_callback):
        raise InterpretationPipelineError("Interpretation progress callback is invalid.")

    timestamp = _created_at(created_at)

    _report(
        stage_callback,
        "loading_raw_transcription",
        "Loading the validated raw transcription.",
        5.0,
    )
    try:
        raw = load_raw_transcription(job_id, settings)
    except RawTranscriptionError as exc:
        raise InterpretationPipelineError(
            "Published raw transcription is unreadable or unsafe."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Raw transcription loading failed at a protected boundary."
        ) from exc
    if raw is None:
        raise InterpretationPipelineError("Published raw transcription is unavailable.")

    _report(
        stage_callback,
        "loading_analysis_timing",
        "Loading saved timing evidence.",
        15.0,
    )
    timing = _load_matching_timing(job_id, settings, raw)

    pitched_events = list(raw["pitchedNoteEvents"])
    percussion_events = list(raw["percussionEvents"])
    alignments = list(raw["alignmentCandidates"])

    _report(
        stage_callback,
        "interpreting_pitched_parts",
        "Grouping pitched candidates into editable parts and phrases.",
        30.0,
    )
    try:
        pitched_result = infer_pitched_parts(pitched_events, alignments)
    except PitchedPartInferenceError as exc:
        raise InterpretationPipelineError(
            "Pitched-event interpretation could not be completed safely."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Pitched-event interpretation failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "interpreting_percussion",
        "Grouping percussion candidates into broad editable voices.",
        45.0,
    )
    try:
        percussion_result = interpret_percussion(
            percussion_events,
            [
                candidate
                for candidate in alignments
                if candidate.get("eventType") == "percussion"
            ],
        )
    except PercussionInterpretationError as exc:
        raise InterpretationPipelineError(
            "Percussion interpretation could not be completed safely."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Percussion interpretation failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "interpreting_rhythm",
        "Building conservative rhythm and duration hypotheses.",
        60.0,
    )
    try:
        rhythm_result = interpret_rhythm(
            pitched_events,
            percussion_events,
            alignments,
            timing,
        )
    except RhythmInterpretationError as exc:
        raise InterpretationPipelineError(
            "Rhythm interpretation could not be completed safely."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Rhythm interpretation failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "assembling_interpretation_draft",
        "Assembling the editable musical draft.",
        76.0,
    )
    try:
        draft = _build_draft(
            raw=raw,
            pitched_result=pitched_result,
            percussion_result=percussion_result,
            rhythm_result=rhythm_result,
            version=version,
            created_at=timestamp,
        )
    except InterpretationPipelineError:
        raise
    except Exception as exc:
        raise InterpretationPipelineError(
            "Editable draft synthesis failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "validating_interpretation_draft",
        "Validating editable structure and source references.",
        86.0,
    )
    try:
        validated = validate_transcription_draft(draft)
    except TranscriptionDraftValidationError as exc:
        raise InterpretationPipelineError(
            "Editable interpretation failed draft validation."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Editable draft validation failed at a protected boundary."
        ) from exc

    _report(
        stage_callback,
        "saving_interpretation_draft",
        "Saving the editable interpretation draft.",
        95.0,
    )
    try:
        write_transcription_draft(job_id, settings, validated)
        reloaded = load_transcription_draft(job_id, settings)
    except TranscriptionDraftError as exc:
        raise InterpretationPipelineError(
            "Editable interpretation could not be published safely."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Editable interpretation publication failed at a protected boundary."
        ) from exc
    if reloaded is None or reloaded != validated:
        raise InterpretationPipelineError(
            "Published editable interpretation could not be verified."
        )

    _report(
        stage_callback,
        "completed",
        "Editable interpretation draft complete.",
        100.0,
    )
    warnings = tuple(reloaded["warnings"])
    return InterpretationPipelineResult(
        version=version,
        draft_file_name=INTERPRETATION_DRAFT_RELATIVE_PATH,
        created_at=timestamp,
        part_count=len(reloaded["parts"]),
        phrase_count=len(reloaded["phrases"]),
        pitched_item_count=len(reloaded["pitchedItems"]),
        percussion_item_count=len(reloaded["percussionItems"]),
        warning_count=len(warnings),
        warnings=warnings,
        payload=reloaded,
    )


def _load_matching_timing(
    job_id: str,
    settings: Settings,
    raw: Mapping[str, Any],
) -> Mapping[str, Any]:
    try:
        analysis = load_analysis(job_id, settings)
    except AudioAnalysisError as exc:
        raise InterpretationPipelineError(
            "Saved audio analysis is unreadable or unsafe."
        ) from exc
    except Exception as exc:
        raise InterpretationPipelineError(
            "Saved audio analysis loading failed at a protected boundary."
        ) from exc
    if not isinstance(analysis, Mapping):
        raise InterpretationPipelineError("Saved audio analysis is unavailable.")
    if analysis.get("schemaVersion") != 1:
        raise InterpretationPipelineError("Saved audio analysis is incompatible.")
    if analysis.get("sourceAsset") != "analysis.wav":
        raise InterpretationPipelineError(
            "Saved audio analysis references an unsupported source."
        )
    source = raw.get("sourceAnalysis")
    if not isinstance(source, Mapping):
        raise InterpretationPipelineError(
            "Raw transcription analysis provenance is malformed."
        )
    if analysis.get("analysisVersion") != source.get("analysisVersion"):
        raise InterpretationPipelineError(
            "Raw transcription analysis provenance is stale."
        )
    timing = analysis.get("timing")
    if not isinstance(timing, Mapping):
        raise InterpretationPipelineError("Saved timing evidence is unavailable.")
    return timing


def _build_draft(
    *,
    raw: Mapping[str, Any],
    pitched_result: PitchedPartInferenceResult,
    percussion_result: PercussionInterpretationResult,
    rhythm_result: RhythmInterpretationResult,
    version: str,
    created_at: str,
) -> dict[str, Any]:
    pitched_evidence = _json_mapping(pitched_result.payload())
    percussion_evidence = _json_mapping(asdict(percussion_result))
    rhythm_evidence = _json_mapping(asdict(rhythm_result))
    evidence = {
        "pitchedPartInference": pitched_evidence,
        "percussionInterpretation": percussion_evidence,
        "rhythmInterpretation": rhythm_evidence,
    }

    source_index = _source_event_index(raw)
    source_by_id = {item["id"]: item for item in source_index}
    raw_pitched = {event["id"]: event for event in raw["pitchedNoteEvents"]}
    raw_percussion = {event["id"]: event for event in raw["percussionEvents"]}
    raw_alignment = {
        item["eventId"]: item
        for item in raw["alignmentCandidates"]
        if isinstance(item, Mapping)
    }
    rhythm_by_event = {
        item["eventId"]: item
        for item in rhythm_evidence["event_interpretations"]
    }
    pitched_assignment_by_event = {
        item["eventId"]: item
        for item in pitched_evidence["assignments"]
    }
    percussion_assignments_by_event: dict[str, list[dict[str, Any]]] = {}
    for item in percussion_evidence["assignments"]:
        percussion_assignments_by_event.setdefault(item["eventId"], []).append(item)
    for items in percussion_assignments_by_event.values():
        items.sort(key=lambda item: (item.get("hitIndex", 0), item["id"]))

    parts, synthetic_parts = _synthesis_parts(
        pitched_evidence,
        percussion_evidence,
        source_by_id,
        pitched_assignment_by_event,
    )
    voices, synthetic_voices = _synthesis_voices(
        pitched_evidence,
        percussion_evidence,
        source_by_id,
        percussion_assignments_by_event,
        pitched_assignment_by_event,
        synthetic_parts,
    )
    measures, measure_id_by_index = _synthesis_measures(rhythm_evidence)
    phrases, phrase_ids = _synthesis_phrases(
        pitched_evidence,
        measure_id_by_index,
        raw_alignment,
    )

    pitched_items = [
        _pitched_item(
            raw_event=raw_pitched[event_id],
            assignment=pitched_assignment_by_event[event_id],
            rhythm=rhythm_by_event[event_id],
            measure_id_by_index=measure_id_by_index,
            raw_alignment=raw_alignment.get(event_id),
            synthetic_parts=synthetic_parts,
            synthetic_voices=synthetic_voices,
            phrase_ids=phrase_ids,
        )
        for event_id in sorted(
            raw_pitched,
            key=lambda event_id: (
                raw_pitched[event_id]["startSeconds"],
                raw_pitched[event_id]["endSeconds"],
                event_id,
            ),
        )
    ]
    percussion_part_id = (
        percussion_evidence["parts"][0]["id"]
        if percussion_evidence["parts"]
        else _stable_id("part", "percussion")
    )
    percussion_items = [
        _percussion_item(
            raw_event=raw_percussion[event_id],
            assignments=percussion_assignments_by_event[event_id],
            rhythm=rhythm_by_event[event_id],
            voice_evidence=percussion_evidence["voices"],
            measure_id_by_index=measure_id_by_index,
            raw_alignment=raw_alignment.get(event_id),
            percussion_part_id=percussion_part_id,
        )
        for event_id in sorted(
            raw_percussion,
            key=lambda event_id: (raw_percussion[event_id]["timeSeconds"], event_id),
        )
    ]

    algorithms = {
        "interpretationPipeline": {"version": version},
        "rawTranscription": {"version": raw["transcriptionVersion"]},
        "pitchedPartInference": {"version": PITCHED_PART_INFERENCE_VERSION},
        "percussionInterpretation": {"version": PERCUSSION_INTERPRETATION_VERSION},
        "rhythmInterpretation": {"version": RHYTHM_INTERPRETATION_VERSION},
    }
    warnings = _combined_warnings(
        raw.get("warnings", []),
        pitched_evidence.get("warnings", []),
        percussion_evidence.get("warnings", []),
        rhythm_evidence.get("warnings", []),
        ["This artifact is an editable interpretation draft, not final notation."],
    )
    provenance = {
        "rawSchemaVersion": raw["schemaVersion"],
        "rawCreatedAt": raw["createdAt"],
        "analysisVersion": raw["sourceAnalysis"]["analysisVersion"],
        "sourceSeparationPresent": "sourceSeparation" in raw,
        "rawWarningCount": len(raw.get("warnings", [])),
        "algorithmVersions": {
            name: record["version"]
            for name, record in sorted(raw["algorithms"].items())
        },
    }
    return {
        "schemaVersion": 1,
        "draftVersion": version,
        "createdAt": created_at,
        "sourceTranscription": {
            "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
            "schemaVersion": raw["schemaVersion"],
            "transcriptionVersion": raw["transcriptionVersion"],
            "provenance": provenance,
            "sourceEventIndex": source_index,
        },
        "algorithms": algorithms,
        "interpretationEvidence": evidence,
        "parts": parts,
        "voices": voices,
        "measures": measures,
        "phrases": phrases,
        "pitchedItems": pitched_items,
        "percussionItems": percussion_items,
        "alternatives": [],
        "warnings": warnings,
    }


def _source_event_index(raw: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for event in raw["pitchedNoteEvents"]:
        item = {
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


def _synthesis_parts(
    pitched: Mapping[str, Any],
    percussion: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
    pitched_assignment_by_event: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    parts: list[dict[str, Any]] = []
    for item in pitched["parts"]:
        parts.append(
            {
                "id": item["id"],
                "sourceKind": item["sourceKind"],
                "role": item["role"],
                "instrumentKind": "source_pitched_line",
                "voiceIds": _container_refs(item["voiceIds"]),
                "sourceEventIds": _container_refs(item["sourceEventIds"]),
                "confidence": item["confidence"],
                "label": item["label"],
                "warnings": list(item.get("warnings", [])),
            }
        )

    if percussion["parts"]:
        item = percussion["parts"][0]
        refs = list(item["rawEventIds"])
        source_kinds = sorted({source_by_id[event_id]["sourceKind"] for event_id in refs})
        source_kind = source_kinds[0] if len(source_kinds) == 1 else "mixed_percussion"
        parts.append(
            {
                "id": item["id"],
                "sourceKind": source_kind,
                "role": "percussion",
                "instrumentKind": "broad_percussion",
                "voiceIds": _container_refs(item["voiceIds"]),
                "sourceEventIds": _container_refs(refs),
                "confidence": _mean(
                    source_by_id[event_id]["confidence"] for event_id in refs
                ),
                "label": item.get("label", "Percussion"),
                "warnings": [],
            }
        )

    synthetic_parts: dict[str, str] = {}
    unassigned_by_source: dict[str, list[str]] = {}
    for event_id, assignment in pitched_assignment_by_event.items():
        if assignment.get("status") == "assigned":
            continue
        source = source_by_id[event_id]["sourceKind"]
        unassigned_by_source.setdefault(source, []).append(event_id)
    for source_kind in sorted(unassigned_by_source):
        refs = sorted(
            unassigned_by_source[source_kind],
            key=lambda event_id: (source_by_id[event_id]["rawStartSeconds"], event_id),
        )
        part_id = _stable_id("part", "unassigned", source_kind)
        voice_id = _stable_id("voice", "unassigned", source_kind)
        synthetic_parts[source_kind] = part_id
        parts.append(
            {
                "id": part_id,
                "sourceKind": source_kind,
                "role": "unassigned_pitched",
                "instrumentKind": "unresolved_pitched",
                "voiceIds": [voice_id],
                "sourceEventIds": _container_refs(refs),
                "confidence": _mean(
                    source_by_id[event_id]["confidence"] for event_id in refs
                ),
                "warnings": [
                    "Low-confidence pitched evidence remains explicitly unassigned."
                ],
            }
        )
    parts.sort(key=lambda item: item["id"])
    return parts, synthetic_parts


def _synthesis_voices(
    pitched: Mapping[str, Any],
    percussion: Mapping[str, Any],
    source_by_id: Mapping[str, Mapping[str, Any]],
    percussion_assignments_by_event: Mapping[str, Sequence[Mapping[str, Any]]],
    pitched_assignment_by_event: Mapping[str, Mapping[str, Any]],
    synthetic_parts: Mapping[str, str],
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    voices: list[dict[str, Any]] = []
    for item in pitched["voices"]:
        refs = list(item["sourceEventIds"])
        voices.append(
            {
                "id": item["id"],
                "partId": item["partId"],
                "voiceKind": (
                    "monophonic"
                    if item.get("monophonicBaseline") is True
                    else "pitched_voice"
                ),
                "sourceEventIds": _container_refs(refs),
                "confidence": _mean(
                    source_by_id[event_id]["confidence"] for event_id in refs
                ),
                "label": item.get("label", "Pitched voice"),
                "warnings": list(item.get("warnings", [])),
            }
        )

    percussion_assignment_by_id = {
        assignment["id"]: assignment
        for assignments in percussion_assignments_by_event.values()
        for assignment in assignments
    }
    for item in percussion["voices"]:
        assignments = [
            percussion_assignment_by_id[assignment_id]
            for assignment_id in item.get("assignmentIds", [])
            if assignment_id in percussion_assignment_by_id
        ]
        refs = sorted(
            {assignment["eventId"] for assignment in assignments},
            key=lambda event_id: (source_by_id[event_id]["rawStartSeconds"], event_id),
        )
        voices.append(
            {
                "id": item["id"],
                "partId": item["partId"],
                "voiceKind": item["kind"],
                "sourceEventIds": _container_refs(refs),
                "confidence": _mean(
                    assignment["confidence"] for assignment in assignments
                ),
                "label": item.get("label", item["kind"]),
                "broad": True,
            }
        )

    synthetic_voices: dict[str, str] = {}
    unassigned_by_source: dict[str, list[str]] = {}
    for event_id, assignment in pitched_assignment_by_event.items():
        if assignment.get("status") == "assigned":
            continue
        unassigned_by_source.setdefault(source_by_id[event_id]["sourceKind"], []).append(
            event_id
        )
    for source_kind, part_id in sorted(synthetic_parts.items()):
        voice_id = _stable_id("voice", "unassigned", source_kind)
        synthetic_voices[source_kind] = voice_id
        refs = sorted(
            unassigned_by_source[source_kind],
            key=lambda event_id: (source_by_id[event_id]["rawStartSeconds"], event_id),
        )
        voices.append(
            {
                "id": voice_id,
                "partId": part_id,
                "voiceKind": "unassigned_pitched",
                "sourceEventIds": _container_refs(refs),
                "confidence": _mean(
                    source_by_id[event_id]["confidence"] for event_id in refs
                ),
                "warnings": [
                    "This voice holds unresolved pitched evidence without forcing assignment."
                ],
            }
        )
    voices.sort(key=lambda item: item["id"])
    return voices, synthetic_voices


def _synthesis_measures(
    rhythm: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[int, str]]:
    measures: list[dict[str, Any]] = []
    by_index: dict[int, str] = {}
    for item in rhythm["measures"]:
        start = float(item["startSeconds"])
        end = float(item["endSeconds"])
        clean: dict[str, Any] = {
            "id": item["id"],
            "index": item["index"],
            "rawStartSeconds": start,
            "rawEndSeconds": end,
            "interpretedStartSeconds": start,
            "interpretedDurationSeconds": end - start,
            "confidence": item["confidence"],
            "warnings": list(item.get("warnings", [])),
        }
        beat_indices = list(item.get("beatIndices", []))
        if beat_indices:
            clean["startBeatIndex"] = min(beat_indices)
            clean["endBeatIndex"] = max(beat_indices) + 1
        measures.append(clean)
        by_index[item["index"]] = item["id"]
    measures.sort(key=lambda item: (item["index"], item["id"]))
    return measures, by_index


def _synthesis_phrases(
    pitched: Mapping[str, Any],
    measure_id_by_index: Mapping[int, str],
    raw_alignment: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], set[str]]:
    phrases: list[dict[str, Any]] = []
    ids: set[str] = set()
    for item in pitched["phrases"]:
        refs = list(item["sourceEventIds"])
        if len(refs) > _MAX_CONTAINER_REFS:
            continue
        measure_ids = []
        for event_id in refs:
            candidate = raw_alignment.get(event_id)
            if not candidate:
                continue
            measure_index = candidate.get("measureIndex")
            if measure_index in measure_id_by_index:
                measure_ids.append(measure_id_by_index[measure_index])
        clean: dict[str, Any] = {
            "id": item["id"],
            "partId": item["partId"],
            "voiceId": item["voiceId"],
            "sourceEventIds": refs,
            "rawStartSeconds": item["rawStartSeconds"],
            "rawEndSeconds": item["rawEndSeconds"],
            "confidence": min(
                float(item["boundaryConfidence"]),
                float(item["continuityConfidence"]),
            ),
            "boundaryConfidence": item["boundaryConfidence"],
            "continuityConfidence": item["continuityConfidence"],
            "boundaryReasonCodes": list(item.get("boundaryReasonCodes", [])),
            "warnings": list(item.get("warnings", [])),
        }
        unique_measure_ids = sorted(set(measure_ids))
        if unique_measure_ids:
            clean["measureIds"] = unique_measure_ids
        phrases.append(clean)
        ids.add(item["id"])
    phrases.sort(key=lambda item: (item["rawStartSeconds"], item["id"]))
    return phrases, ids


def _pitched_item(
    *,
    raw_event: Mapping[str, Any],
    assignment: Mapping[str, Any],
    rhythm: Mapping[str, Any],
    measure_id_by_index: Mapping[int, str],
    raw_alignment: Mapping[str, Any] | None,
    synthetic_parts: Mapping[str, str],
    synthetic_voices: Mapping[str, str],
    phrase_ids: set[str],
) -> dict[str, Any]:
    event_id = raw_event["id"]
    source_kind = raw_event["sourceKind"]
    assigned = assignment.get("status") == "assigned"
    part_id = assignment.get("partId") if assigned else synthetic_parts[source_kind]
    voice_id = assignment.get("voiceId") if assigned else synthetic_voices[source_kind]
    placement = _preferred_placement(rhythm)
    duration = _preferred_duration(
        rhythm, raw_event["endSeconds"] - raw_event["startSeconds"]
    )
    placed = (
        assigned
        and placement is not None
        and placement.get("status") == "resolved"
        and rhythm.get("unresolved") is False
    )
    item: dict[str, Any] = {
        "id": _stable_id("pitched", event_id),
        "interpretationType": "note" if assigned else "unassigned",
        "placementStatus": "placed" if placed else "unassigned",
        "partId": part_id,
        "voiceId": voice_id,
        "sourceEventIds": [event_id],
        "rawStartSeconds": raw_event["startSeconds"],
        "rawEndSeconds": raw_event["endSeconds"],
        "sourceKind": source_kind,
        "pitch": {
            "midiNote": raw_event["midiNote"],
            "midiPitch": raw_event["midiPitch"],
            "frequencyHz": raw_event["frequencyHz"],
            "noteName": raw_event["noteName"],
        },
        "confidence": min(
            float(raw_event["confidence"]),
            float(rhythm.get("confidence", raw_event["confidence"])),
        ),
        "sharedEvidence": True,
        "warnings": _combined_warnings(
            raw_event.get("warnings", []),
            assignment.get("warnings", []),
            rhythm.get("warnings", []),
        ),
    }
    phrase_id = assignment.get("phraseId")
    if isinstance(phrase_id, str) and phrase_id in phrase_ids:
        item["phraseId"] = phrase_id
    if placement is not None:
        item["interpretedStartSeconds"] = placement["alignedTimeSeconds"]
        item["interpretedDurationSeconds"] = duration
        item["gridPosition"] = _grid_position(
            placement,
            raw_alignment,
            measure_id_by_index,
        )
        if "measureId" in item["gridPosition"]:
            item["measureId"] = item["gridPosition"]["measureId"]

    continuations = list(rhythm.get("continuationHypotheses", []))
    if continuations:
        first = continuations[0]
        tie = {
            "role": "continuation_candidate",
            "confidence": float(first.get("confidence", 0.0)),
        }
        for key in ("boundaryType", "boundaryTimeSeconds", "resolved", "warnings"):
            if key in first:
                tie[key] = copy.deepcopy(first[key])
        item["tieCandidate"] = tie

    item["alternatives"] = _pitched_alternatives(
        item_id=item["id"],
        assignment=assignment,
        rhythm=rhythm,
    )
    return item


def _percussion_item(
    *,
    raw_event: Mapping[str, Any],
    assignments: Sequence[Mapping[str, Any]],
    rhythm: Mapping[str, Any],
    voice_evidence: Sequence[Mapping[str, Any]],
    measure_id_by_index: Mapping[int, str],
    raw_alignment: Mapping[str, Any] | None,
    percussion_part_id: str,
) -> dict[str, Any]:
    voice_kind_by_id = {item["id"]: item["kind"] for item in voice_evidence}
    placement = _preferred_placement(rhythm)
    placed = (
        placement is not None
        and placement.get("status") == "resolved"
        and rhythm.get("unresolved") is False
    )
    hits = []
    for assignment in assignments:
        hits.append(
            {
                "sourceHitIndex": assignment["hitIndex"],
                "rawKind": assignment["rawHitKind"],
                "broadVoice": voice_kind_by_id.get(
                    assignment["voiceId"], "unresolved_percussion"
                ),
                "voiceId": assignment["voiceId"],
                "confidence": assignment["confidence"],
                "resolution": assignment["resolution"],
                "rawHit": copy.deepcopy(assignment["rawHit"]),
            }
        )
    hits.sort(key=lambda item: item["sourceHitIndex"])
    item: dict[str, Any] = {
        "id": _stable_id("percussion", raw_event["id"]),
        "placementStatus": "placed" if placed else "unassigned",
        "partId": percussion_part_id,
        "voiceIds": sorted({assignment["voiceId"] for assignment in assignments}),
        "sourceEventIds": [raw_event["id"]],
        "rawStartSeconds": raw_event["timeSeconds"],
        "rawEndSeconds": raw_event["timeSeconds"],
        "sourceKind": raw_event["sourceKind"],
        "hits": hits,
        "confidence": _mean(assignment["confidence"] for assignment in assignments),
        "sharedEvidence": True,
        "warnings": list(raw_event.get("warnings", [])),
    }
    if placement is not None:
        item["interpretedStartSeconds"] = placement["alignedTimeSeconds"]
        item["interpretedDurationSeconds"] = 0.0
        item["gridPosition"] = _grid_position(
            placement,
            raw_alignment,
            measure_id_by_index,
        )
        if "measureId" in item["gridPosition"]:
            item["measureId"] = item["gridPosition"]["measureId"]
    item["alternatives"] = _placement_alternatives(
        item_id=item["id"],
        rhythm=rhythm,
    )
    return item


def _preferred_placement(rhythm: Mapping[str, Any]) -> Mapping[str, Any] | None:
    for hypothesis in rhythm.get("placementHypotheses", []):
        if (
            hypothesis.get("kind") == "grid"
            and "alignedTimeSeconds" in hypothesis
            and "offsetSeconds" in hypothesis
        ):
            return hypothesis
    return None


def _preferred_duration(rhythm: Mapping[str, Any], raw_duration: float) -> float:
    for hypothesis in rhythm.get("durationHypotheses", []):
        value = hypothesis.get("durationSeconds")
        if (
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            and float(value) >= 0
        ):
            return float(value)
    return float(raw_duration)


def _grid_position(
    placement: Mapping[str, Any],
    raw_alignment: Mapping[str, Any] | None,
    measure_id_by_index: Mapping[int, str],
) -> dict[str, Any]:
    grid = {
        "beatIndex": placement["beatIndex"],
        "subdivision": placement["subdivision"],
        "subdivisionIndex": placement["subdivisionIndex"],
        "alignedTimeSeconds": placement["alignedTimeSeconds"],
        "offsetSeconds": placement["offsetSeconds"],
    }
    if raw_alignment is not None:
        measure_index = raw_alignment.get("measureIndex")
        beat_in_measure = raw_alignment.get("beatInMeasure")
        if (
            isinstance(measure_index, int)
            and measure_index in measure_id_by_index
            and isinstance(beat_in_measure, int)
        ):
            grid.update(
                measureId=measure_id_by_index[measure_index],
                measureIndex=measure_index,
                beatInMeasure=beat_in_measure,
            )
    return grid


def _pitched_alternatives(
    *,
    item_id: str,
    assignment: Mapping[str, Any],
    rhythm: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alternatives: list[dict[str, Any]] = []
    for index, value in enumerate(assignment.get("alternatives", []), start=1):
        alternatives.append(
            {
                "id": _stable_id("alt", item_id, "assignment", str(index)),
                "kind": _slugged(str(value.get("kind", "assignment"))),
                "confidence": float(value.get("confidence", 0.0)),
                "evidence": copy.deepcopy(value),
            }
        )
    for index, value in enumerate(rhythm.get("placementHypotheses", []), start=1):
        alternatives.append(
            {
                "id": _stable_id("alt", item_id, "placement", str(index)),
                "kind": _slugged(f"placement_{value.get('kind', 'candidate')}"),
                "confidence": float(value.get("confidence", 0.0)),
                "evidence": copy.deepcopy(value),
            }
        )
    for index, value in enumerate(rhythm.get("durationHypotheses", []), start=1):
        label = str(value.get("label") or value.get("kind") or "candidate")
        alternatives.append(
            {
                "id": _stable_id("alt", item_id, "duration", str(index)),
                "kind": _slugged(f"duration_{label}"),
                "confidence": float(value.get("confidence", 0.0)),
                "evidence": copy.deepcopy(value),
            }
        )
    for index, value in enumerate(rhythm.get("continuationHypotheses", [])[1:], start=2):
        alternatives.append(
            {
                "id": _stable_id("alt", item_id, "continuation", str(index)),
                "kind": "continuation_candidate",
                "confidence": float(value.get("confidence", 0.0)),
                "evidence": copy.deepcopy(value),
            }
        )
    return alternatives[:_MAX_ITEM_ALTERNATIVES]


def _placement_alternatives(
    *,
    item_id: str,
    rhythm: Mapping[str, Any],
) -> list[dict[str, Any]]:
    alternatives = []
    for index, value in enumerate(rhythm.get("placementHypotheses", []), start=1):
        alternatives.append(
            {
                "id": _stable_id("alt", item_id, "placement", str(index)),
                "kind": _slugged(f"placement_{value.get('kind', 'candidate')}"),
                "confidence": float(value.get("confidence", 0.0)),
                "evidence": copy.deepcopy(value),
            }
        )
    return alternatives[:_MAX_ITEM_ALTERNATIVES]


def _json_mapping(value: object) -> dict[str, Any]:
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise InterpretationPipelineError(
            "Interpretation evidence is not safe JSON."
        ) from exc
    if not isinstance(decoded, dict):
        raise InterpretationPipelineError(
            "Interpretation evidence has an invalid shape."
        )
    return decoded


def _combined_warnings(*groups: Sequence[str]) -> list[str]:
    output: list[str] = []
    for group in groups:
        for item in group:
            if item not in output:
                output.append(item)
            if len(output) >= _MAX_SYNTHESIS_WARNINGS:
                return output
    return output


def _container_refs(values: Sequence[str]) -> list[str]:
    refs = list(values)
    return refs if len(refs) <= _MAX_CONTAINER_REFS else []


def _mean(values: Any) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)


def _stable_id(prefix: str, *components: str) -> str:
    raw = "_".join((prefix, *components))
    normalized = _slugged(raw)
    if len(normalized) <= 96 and _ID_RE.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    keep = max(1, 96 - len(digest) - 1)
    candidate = f"{normalized[:keep].rstrip('_')}_{digest}"
    if not _ID_RE.fullmatch(candidate):
        raise InterpretationPipelineError("A deterministic draft ID could not be created.")
    return candidate


def _slugged(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9_-]+", "_", value.lower()).strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"i_{normalized}"
    return normalized[:96]


def _created_at(value: str | None) -> str:
    if value is None:
        return datetime.now(timezone.utc).isoformat()
    if not isinstance(value, str) or not value or len(value) > 128:
        raise InterpretationPipelineError("Interpretation timestamp is invalid.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise InterpretationPipelineError("Interpretation timestamp is invalid.") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise InterpretationPipelineError("Interpretation timestamp must be UTC.")
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
        raise InterpretationPipelineError(
            "Interpretation progress reporting failed."
        ) from exc


__all__ = [
    "INTERPRETATION_PIPELINE_VERSION",
    "InterpretationPipelineError",
    "InterpretationPipelineResult",
    "interpret_transcription_job",
]
