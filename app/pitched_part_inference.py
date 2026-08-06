from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from app.transcription_events import (
    RawTranscriptionError,
    RawTranscriptionValidationError,
    validate_raw_transcription,
)


PITCHED_PART_INFERENCE_VERSION = "source-phrase-v1"

_ID_PATTERN = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_VERSION_PATTERN = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_MAX_EVENTS = 20_000
_MAX_ALIGNMENT_CANDIDATES = 40_000
_MAX_ALIGNMENT_PER_EVENT = 16
_MAX_PARTS = 128
_MAX_VOICES = 512
_MAX_PHRASES = 20_000
_MAX_ALTERNATIVES = 8
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MIN_ASSIGNMENT_CONFIDENCE = 0.25
_LARGE_GAP_SECONDS = 1.5
_MEASURE_GAP_BOUNDARY = 2
_OVERLAP_EPSILON_SECONDS = 1e-9

_SOURCE_PRIORITY = {"vocals": 0, "bass": 1, "other": 2, "full_mix": 3}
_SOURCE_LABELS = {
    "vocals": "Vocals",
    "bass": "Bass",
    "other": "Other pitched source",
    "full_mix": "Full mix",
}
_SOURCE_ROLES = {
    "vocals": "vocal_line",
    "bass": "bass_line",
    "other": "unresolved_pitched_source",
    "full_mix": "unresolved_pitched_source",
}
_ALIGNMENT_FIELDS = {
    "eventId",
    "eventType",
    "rawTimeSeconds",
    "beatIndex",
    "subdivision",
    "subdivisionIndex",
    "alignedTimeSeconds",
    "offsetSeconds",
    "confidence",
    "measureIndex",
    "beatInMeasure",
    "warnings",
    "collection",
    "eventCollection",
}


class PitchedPartInferenceError(RuntimeError):
    """Pitched part inference could not be completed safely."""


@dataclass(frozen=True, slots=True)
class PitchedPartInferenceResult:
    version: str
    parts: tuple[dict[str, Any], ...]
    voices: tuple[dict[str, Any], ...]
    phrases: tuple[dict[str, Any], ...]
    assignments: tuple[dict[str, Any], ...]
    unassigned_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        result = {
            "version": self.version,
            "parts": copy.deepcopy(list(self.parts)),
            "voices": copy.deepcopy(list(self.voices)),
            "phrases": copy.deepcopy(list(self.phrases)),
            "assignments": copy.deepcopy(list(self.assignments)),
            "unassignedEventIds": list(self.unassigned_event_ids),
            "warnings": list(self.warnings),
            "diagnostics": copy.deepcopy(self.diagnostics),
        }
        _require_safe_json(result)
        return result


def infer_pitched_parts(
    pitched_events: Sequence[Mapping[str, Any]],
    alignment_candidates: Sequence[Mapping[str, Any]] = (),
    *,
    version: str = PITCHED_PART_INFERENCE_VERSION,
) -> PitchedPartInferenceResult:
    """Infer conservative source-aware parts, voices, and phrases."""
    safe_version = _safe_version(version)
    events, alignment_by_event, ignored_alignment_count = _validated_inputs(
        pitched_events,
        alignment_candidates,
    )
    assigned_events = [
        event for event in events if event["confidence"] >= _MIN_ASSIGNMENT_CONFIDENCE
    ]
    unassigned_events = [
        event for event in events if event["confidence"] < _MIN_ASSIGNMENT_CONFIDENCE
    ]

    source_groups: dict[str, list[dict[str, Any]]] = {}
    for event in assigned_events:
        source_groups.setdefault(event["sourceKind"], []).append(event)
    if len(source_groups) > _MAX_PARTS:
        raise PitchedPartInferenceError("Too many pitched source parts were inferred.")

    parts: list[dict[str, Any]] = []
    voices: list[dict[str, Any]] = []
    phrases: list[dict[str, Any]] = []
    assignment_index: dict[str, dict[str, Any]] = {}
    overlap_event_ids: set[str] = set()
    overlap_alternative_voices: dict[str, set[str]] = {}
    broad_source_event_ids: set[str] = set()

    for source_kind in sorted(source_groups, key=_source_sort_key):
        source_events = sorted(source_groups[source_kind], key=_event_sort_key)
        part_id = _stable_id("part", source_kind)
        voice_groups = _partition_voices(source_events)
        if len(voices) + len(voice_groups) > _MAX_VOICES:
            raise PitchedPartInferenceError("Too many pitched voices were inferred.")
        if source_kind in {"other", "full_mix"}:
            broad_source_event_ids.update(event["id"] for event in source_events)

        part_voice_ids: list[str] = []
        part_phrase_ids: list[str] = []
        for voice_index, voice_events in enumerate(voice_groups, start=1):
            voice_id = _stable_id("voice", source_kind, str(voice_index))
            part_voice_ids.append(voice_id)
            phrase_groups = _partition_phrases(voice_events, alignment_by_event)
            if len(phrases) + len(phrase_groups) > _MAX_PHRASES:
                raise PitchedPartInferenceError("Too many pitched phrases were inferred.")
            voice_phrase_ids: list[str] = []
            for phrase_index, phrase_data in enumerate(phrase_groups, start=1):
                phrase_events, boundary_reason, boundary_confidence = phrase_data
                phrase_id = _stable_id(
                    "phrase", source_kind, str(voice_index), str(phrase_index)
                )
                voice_phrase_ids.append(phrase_id)
                part_phrase_ids.append(phrase_id)
                phrases.append(
                    _phrase_payload(
                        phrase_id,
                        part_id,
                        voice_id,
                        phrase_events,
                        alignment_by_event,
                        boundary_reason,
                        boundary_confidence,
                    )
                )
                for event in phrase_events:
                    assignment_index[event["id"]] = _assigned_payload(
                        event,
                        part_id,
                        voice_id,
                        phrase_id,
                        alignment_by_event.get(event["id"], ()),
                    )

            voice_warnings = []
            if len(voice_groups) > 1:
                voice_warnings.append(
                    "Overlapping source events require multiple editable baseline voices."
                )
            voices.append(
                {
                    "id": voice_id,
                    "partId": part_id,
                    "sourceKind": source_kind,
                    "label": (
                        "Primary line"
                        if voice_index == 1
                        else f"Overlapping line {voice_index}"
                    ),
                    "monophonicBaseline": True,
                    "sourceEventIds": [event["id"] for event in voice_events],
                    "phraseIds": voice_phrase_ids,
                    "rawStartSeconds": voice_events[0]["startSeconds"],
                    "rawEndSeconds": max(event["endSeconds"] for event in voice_events),
                    "warnings": voice_warnings,
                }
            )

        _record_overlap_alternatives(
            source_events,
            assignment_index,
            overlap_event_ids,
            overlap_alternative_voices,
        )
        part_warnings = []
        if source_kind in {"other", "full_mix"}:
            part_warnings.append(
                "The source kind remains broad; no precise instrument is inferred."
            )
        if len(voice_groups) > 1:
            part_warnings.append(
                "Material overlap is preserved in separate editable voices."
            )
        parts.append(
            {
                "id": part_id,
                "kind": "pitched_part",
                "sourceKind": source_kind,
                "label": _source_label(source_kind),
                "role": _SOURCE_ROLES.get(source_kind, "source_named_pitched_line"),
                "voiceIds": part_voice_ids,
                "phraseIds": part_phrase_ids,
                "sourceEventIds": [event["id"] for event in source_events],
                "rawStartSeconds": source_events[0]["startSeconds"],
                "rawEndSeconds": max(event["endSeconds"] for event in source_events),
                "confidence": _confidence(
                    sum(event["confidence"] for event in source_events)
                    / len(source_events)
                ),
                "warnings": part_warnings,
            }
        )

    truncated_alternative_count = _attach_alternatives(
        assignment_index,
        parts,
        events,
        overlap_alternative_voices,
        broad_source_event_ids,
    )
    for event in unassigned_events:
        assignment_index[event["id"]] = _unassigned_payload(
            event,
            alignment_by_event.get(event["id"], ()),
        )

    assignments = tuple(assignment_index[event["id"]] for event in events)
    unassigned_event_ids = tuple(event["id"] for event in unassigned_events)
    warnings = []
    if unassigned_events:
        warnings.append("Low-confidence pitched events remain explicitly unassigned.")
    if overlap_event_ids:
        warnings.append(
            "Overlapping pitched events are preserved across separate baseline voices."
        )
    if broad_source_event_ids:
        warnings.append(
            "Broad source kinds remain unresolved and are not renamed as instruments."
        )
    if ignored_alignment_count:
        warnings.append(
            "Non-pitched alignment candidates were ignored by pitched-part inference."
        )
    if truncated_alternative_count:
        warnings.append(
            "Dense overlap produced more voice alternatives than the bounded output can expose."
        )

    diagnostics = {
        "inputEventCount": len(events),
        "assignedEventCount": len(assigned_events),
        "unassignedEventCount": len(unassigned_events),
        "accountedEventCount": len(assignments),
        "partCount": len(parts),
        "voiceCount": len(voices),
        "phraseCount": len(phrases),
        "overlapEventCount": len(overlap_event_ids),
        "alignmentCandidateCount": sum(
            len(candidates) for candidates in alignment_by_event.values()
        ),
        "ignoredNonPitchedAlignmentCount": ignored_alignment_count,
        "truncatedAlternativeCount": truncated_alternative_count,
        "minimumAssignmentConfidence": _MIN_ASSIGNMENT_CONFIDENCE,
        "largeGapSeconds": _LARGE_GAP_SECONDS,
        "timingQuantized": False,
        "pitchQuantized": False,
        "chordsInferred": False,
        "instrumentsInferred": False,
        "tablatureInferred": False,
        "notationGenerated": False,
    }
    if len(assignments) != len(events):
        raise PitchedPartInferenceError("Pitched event accounting failed.")

    result = PitchedPartInferenceResult(
        version=safe_version,
        parts=tuple(parts),
        voices=tuple(voices),
        phrases=tuple(phrases),
        assignments=assignments,
        unassigned_event_ids=unassigned_event_ids,
        warnings=tuple(warnings),
        diagnostics=diagnostics,
    )
    result.payload()
    return result


def _validated_inputs(
    pitched_events: Sequence[Mapping[str, Any]],
    alignment_candidates: Sequence[Mapping[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[dict[str, Any], ...]],
    int,
]:
    event_items = _bounded_sequence(pitched_events, "pitched events", _MAX_EVENTS)
    candidate_items = _bounded_sequence(
        alignment_candidates,
        "alignment candidates",
        _MAX_ALIGNMENT_CANDIDATES,
    )
    copied_events = [_copy_mapping(item, "pitched event") for item in event_items]
    copied_candidates = [
        _copy_mapping(item, "alignment candidate") for item in candidate_items
    ]

    dummy_percussion: dict[str, dict[str, Any]] = {}
    ignored_alignment_count = 0
    for candidate in copied_candidates:
        if set(candidate) - _ALIGNMENT_FIELDS:
            raise PitchedPartInferenceError(
                "An alignment candidate contains unsupported fields."
            )
        event_type = candidate.get("eventType")
        if event_type == "pitched":
            continue
        if event_type != "percussion":
            raise PitchedPartInferenceError("Alignment eventType is invalid.")
        event_id = candidate.get("eventId")
        raw_time = candidate.get("rawTimeSeconds")
        if not isinstance(event_id, str) or not _ID_PATTERN.fullmatch(event_id):
            raise PitchedPartInferenceError("A non-pitched alignment ID is unsafe.")
        if (
            isinstance(raw_time, bool)
            or not isinstance(raw_time, (int, float))
            or not math.isfinite(float(raw_time))
            or float(raw_time) < 0
        ):
            raise PitchedPartInferenceError(
                "A non-pitched alignment raw time is invalid."
            )
        raw_number = float(raw_time)
        previous = dummy_percussion.get(event_id)
        if previous is not None and previous["timeSeconds"] != raw_number:
            raise PitchedPartInferenceError(
                "Non-pitched alignment alternatives disagree on raw time."
            )
        dummy_percussion[event_id] = {
            "id": event_id,
            "sourceKind": "ignored_percussion_alignment",
            "timeSeconds": raw_number,
            "strength": 0.0,
            "hits": [{"kind": "unknown_percussion", "confidence": 0.0}],
        }
        ignored_alignment_count += 1

    payload = {
        "schemaVersion": 1,
        "transcriptionVersion": "pitched-part-inference-input-v1",
        "createdAt": "2026-01-01T00:00:00+00:00",
        "sourceAnalysis": {
            "fileName": "analysis/audio-analysis.json",
            "analysisVersion": "pitched-part-inference-input-v1",
        },
        "algorithms": {
            "pitchedPartInferenceInput": {
                "version": "pitched-part-inference-input-v1"
            }
        },
        "pitchedNoteEvents": copied_events,
        "percussionEvents": list(dummy_percussion.values()),
        "alignmentCandidates": copied_candidates,
        "warnings": [],
    }
    try:
        validated = validate_raw_transcription(payload)
    except (RawTranscriptionValidationError, RawTranscriptionError) as exc:
        raise PitchedPartInferenceError(
            "Pitched part inference input failed validation."
        ) from exc

    events = list(validated["pitchedNoteEvents"])
    grouped: dict[str, list[dict[str, Any]]] = {}
    for candidate in validated["alignmentCandidates"]:
        if candidate["eventType"] != "pitched":
            continue
        grouped.setdefault(candidate["eventId"], []).append(candidate)
    alignment_by_event: dict[str, tuple[dict[str, Any], ...]] = {}
    for event_id, candidates in grouped.items():
        if len(candidates) > _MAX_ALIGNMENT_PER_EVENT:
            raise PitchedPartInferenceError(
                "Too many alignment alternatives reference one pitched event."
            )
        alignment_by_event[event_id] = tuple(
            sorted(candidates, key=_alignment_sort_key)
        )
    return events, alignment_by_event, ignored_alignment_count


def _partition_voices(
    events: Sequence[dict[str, Any]],
) -> list[list[dict[str, Any]]]:
    voices: list[list[dict[str, Any]]] = []
    end_times: list[float] = []
    for event in events:
        chosen: int | None = None
        for index, end_time in enumerate(end_times):
            if event["startSeconds"] >= end_time - _OVERLAP_EPSILON_SECONDS:
                chosen = index
                break
        if chosen is None:
            voices.append([event])
            end_times.append(event["endSeconds"])
        else:
            voices[chosen].append(event)
            end_times[chosen] = event["endSeconds"]
    return voices


def _partition_phrases(
    events: Sequence[dict[str, Any]],
    alignment_by_event: Mapping[str, Sequence[dict[str, Any]]],
) -> list[tuple[list[dict[str, Any]], str, float]]:
    groups: list[tuple[list[dict[str, Any]], str, float]] = []
    current: list[dict[str, Any]] = []
    reason = "source_or_voice_start"
    boundary_confidence = 1.0
    for event in events:
        if not current:
            current = [event]
            continue
        previous = current[-1]
        gap = event["startSeconds"] - previous["endSeconds"]
        next_reason: str | None = None
        next_confidence = 0.0
        if gap >= _LARGE_GAP_SECONDS:
            next_reason = "large_raw_gap"
            next_confidence = min(0.98, 0.72 + min(0.26, gap / 10.0))
        elif _measure_gap_boundary(previous, event, alignment_by_event):
            next_reason = "aligned_measure_gap"
            next_confidence = 0.75
        if next_reason is not None:
            groups.append((current, reason, boundary_confidence))
            current = [event]
            reason = next_reason
            boundary_confidence = next_confidence
        else:
            current.append(event)
    if current:
        groups.append((current, reason, boundary_confidence))
    return groups


def _measure_gap_boundary(
    previous: Mapping[str, Any],
    current: Mapping[str, Any],
    alignment_by_event: Mapping[str, Sequence[dict[str, Any]]],
) -> bool:
    left = _preferred_aligned(alignment_by_event.get(previous["id"], ()))
    right = _preferred_aligned(alignment_by_event.get(current["id"], ()))
    if left is None or right is None:
        return False
    left_measure = left.get("measureIndex")
    right_measure = right.get("measureIndex")
    return (
        isinstance(left_measure, int)
        and isinstance(right_measure, int)
        and right_measure - left_measure >= _MEASURE_GAP_BOUNDARY
    )


def _phrase_payload(
    phrase_id: str,
    part_id: str,
    voice_id: str,
    events: Sequence[dict[str, Any]],
    alignment_by_event: Mapping[str, Sequence[dict[str, Any]]],
    boundary_reason: str,
    boundary_confidence: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    continuity_scores: list[float] = []
    for previous, current in zip(events, events[1:]):
        gap = max(0.0, current["startSeconds"] - previous["endSeconds"])
        pitch_jump = abs(current["midiPitch"] - previous["midiPitch"])
        score = 1.0 - min(0.35, max(0.0, gap - 0.5) / 5.0)
        if pitch_jump > 12.0:
            score -= min(0.25, (pitch_jump - 12.0) / 48.0)
            warnings.append(
                "Large pitch movement lowers continuity confidence but does not alone create a boundary."
            )
        score *= min(previous["confidence"], current["confidence"])
        continuity_scores.append(max(0.0, min(1.0, score)))
    continuity = (
        sum(continuity_scores) / len(continuity_scores)
        if continuity_scores
        else events[0]["confidence"]
    )
    return {
        "id": phrase_id,
        "partId": part_id,
        "voiceId": voice_id,
        "sourceEventIds": [event["id"] for event in events],
        "rawStartSeconds": events[0]["startSeconds"],
        "rawEndSeconds": max(event["endSeconds"] for event in events),
        "boundaryConfidence": _confidence(boundary_confidence),
        "continuityConfidence": _confidence(continuity),
        "boundaryReasonCodes": [boundary_reason],
        "warnings": sorted(set(warnings)),
        "alignedEvidenceEventCount": sum(
            1
            for event in events
            if _preferred_aligned(alignment_by_event.get(event["id"], ()))
            is not None
        ),
    }


def _assigned_payload(
    event: dict[str, Any],
    part_id: str,
    voice_id: str,
    phrase_id: str,
    alignments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output = _raw_fields(event)
    output.update(
        {
            "id": _stable_id("assignment", event["id"]),
            "eventId": event["id"],
            "status": "assigned",
            "partId": part_id,
            "voiceId": voice_id,
            "phraseId": phrase_id,
            "alignmentCandidates": copy.deepcopy(list(alignments)),
            "alternatives": [],
        }
    )
    return output


def _unassigned_payload(
    event: dict[str, Any],
    alignments: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    output = _raw_fields(event)
    warnings = list(output["warnings"])
    warnings.append(
        "Confidence is below the conservative assignment threshold; the event remains unassigned."
    )
    output.update(
        {
            "id": _stable_id("assignment", event["id"]),
            "eventId": event["id"],
            "status": "unassigned",
            "partId": None,
            "voiceId": None,
            "phraseId": None,
            "warnings": warnings,
            "alignmentCandidates": copy.deepcopy(list(alignments)),
            "alternatives": [],
        }
    )
    return output


def _raw_fields(event: Mapping[str, Any]) -> dict[str, Any]:
    output = {
        "sourceKind": event["sourceKind"],
        "rawStartSeconds": event["startSeconds"],
        "rawEndSeconds": event["endSeconds"],
        "midiNote": event["midiNote"],
        "midiPitch": event["midiPitch"],
        "frequencyHz": event["frequencyHz"],
        "noteName": event["noteName"],
        "confidence": event["confidence"],
        "warnings": copy.deepcopy(event.get("warnings", [])),
    }
    for key in ("velocity", "collection", "rawFeatureSummary", "rawFeatures"):
        if key in event:
            output[key] = copy.deepcopy(event[key])
    return output


def _record_overlap_alternatives(
    events: Sequence[dict[str, Any]],
    assignments: Mapping[str, dict[str, Any]],
    overlap_event_ids: set[str],
    overlap_alternative_voices: dict[str, set[str]],
) -> None:
    for index, left in enumerate(events):
        for right in events[index + 1 :]:
            if right["startSeconds"] >= left["endSeconds"] - _OVERLAP_EPSILON_SECONDS:
                break
            overlap_event_ids.update((left["id"], right["id"]))
            left_voice = assignments[left["id"]]["voiceId"]
            right_voice = assignments[right["id"]]["voiceId"]
            if left_voice != right_voice:
                overlap_alternative_voices.setdefault(left["id"], set()).add(
                    right_voice
                )
                overlap_alternative_voices.setdefault(right["id"], set()).add(
                    left_voice
                )


def _attach_alternatives(
    assignments: Mapping[str, dict[str, Any]],
    parts: Sequence[dict[str, Any]],
    events: Sequence[dict[str, Any]],
    overlap_alternative_voices: Mapping[str, set[str]],
    broad_source_event_ids: set[str],
) -> int:
    part_by_source = {part["sourceKind"]: part for part in parts}
    event_by_id = {event["id"]: event for event in events}
    truncated = 0
    for event_id, assignment in assignments.items():
        event = event_by_id[event_id]
        alternatives = [
            {
                "kind": "voice_assignment",
                "voiceId": voice_id,
                "confidence": _confidence(min(0.49, event["confidence"] * 0.5)),
                "reasonCode": "overlap_voice_ambiguity",
            }
            for voice_id in sorted(overlap_alternative_voices.get(event_id, set()))
        ]
        if event_id in broad_source_event_ids:
            part = part_by_source[event["sourceKind"]]
            alternatives.append(
                {
                    "kind": "part_role",
                    "partId": part["id"],
                    "role": "unresolved_pitched_source",
                    "confidence": _confidence(min(0.5, event["confidence"])),
                    "reasonCode": "broad_source_kind",
                }
            )
        truncated += max(0, len(alternatives) - _MAX_ALTERNATIVES)
        assignment["alternatives"] = alternatives[:_MAX_ALTERNATIVES]
    return truncated


def _preferred_aligned(
    candidates: Sequence[dict[str, Any]],
) -> dict[str, Any] | None:
    return next(
        (candidate for candidate in candidates if "alignedTimeSeconds" in candidate),
        None,
    )


def _alignment_sort_key(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        0 if "alignedTimeSeconds" in candidate else 1,
        -float(candidate.get("confidence", 0.0)),
        abs(float(candidate.get("offsetSeconds", 0.0))),
        json.dumps(candidate, ensure_ascii=False, sort_keys=True, allow_nan=False),
    )


def _event_sort_key(event: Mapping[str, Any]) -> tuple[float, float, str]:
    return (event["startSeconds"], event["endSeconds"], event["id"])


def _source_sort_key(source_kind: str) -> tuple[int, str]:
    return (_SOURCE_PRIORITY.get(source_kind, 10), source_kind)


def _source_label(source_kind: str) -> str:
    return _SOURCE_LABELS.get(source_kind, source_kind.replace("_", " "))


def _stable_id(prefix: str, *components: str) -> str:
    raw = "_".join((prefix, *components))
    normalized = re.sub(r"[^a-z0-9_-]+", "_", raw.lower()).strip("_")
    if not normalized or not normalized[0].isalpha():
        normalized = f"i_{normalized}"
    if len(normalized) <= 64 and _ID_PATTERN.fullmatch(normalized):
        return normalized
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    keep = max(1, 64 - len(digest) - 1)
    candidate = f"{normalized[:keep].rstrip('_')}_{digest}"
    if not _ID_PATTERN.fullmatch(candidate):
        raise PitchedPartInferenceError("A deterministic inference ID could not be created.")
    return candidate


def _safe_version(value: object) -> str:
    if not isinstance(value, str) or not _VERSION_PATTERN.fullmatch(value):
        raise PitchedPartInferenceError("The inference version is invalid.")
    return value


def _bounded_sequence(value: object, label: str, limit: int) -> list[Any]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Sequence
    ):
        raise PitchedPartInferenceError(f"{label.capitalize()} must be a sequence.")
    items = list(value)
    if len(items) > limit:
        raise PitchedPartInferenceError(f"Too many {label} were supplied.")
    return items


def _copy_mapping(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or not all(
        isinstance(key, str) for key in value
    ):
        raise PitchedPartInferenceError(f"{label.capitalize()} must be an object.")
    return copy.deepcopy(dict(value))


def _confidence(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 3)


def _require_safe_json(value: Mapping[str, Any]) -> None:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PitchedPartInferenceError(
            "Pitched part inference produced unsafe JSON."
        ) from exc
    if len(encoded) > _MAX_JSON_BYTES:
        raise PitchedPartInferenceError(
            "Pitched part inference output exceeds the size limit."
        )
