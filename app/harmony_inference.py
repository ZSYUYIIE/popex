"""Conservative harmonic-context inference over raw pitched-note evidence.

Raw note timing and pitch remain authoritative. This module aggregates evidence into
bounded chord candidates for review; it does not create notation, Roman numerals,
guitar voicings, or publication-ready harmony.
"""

from __future__ import annotations

import copy
import json
import math
import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any


HARMONY_INFERENCE_VERSION = "pitch-class-window-v1"

PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

_CHORD_TEMPLATES: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("major", (0, 4, 7)),
    ("minor", (0, 3, 7)),
    ("diminished", (0, 3, 6)),
    ("sus2", (0, 2, 7)),
    ("sus4", (0, 5, 7)),
    ("power", (0, 7)),
    ("dominant7", (0, 4, 7, 10)),
    ("major7", (0, 4, 7, 11)),
    ("minor7", (0, 3, 7, 10)),
)

_QUALITY_SUFFIX = {
    "major": "",
    "minor": "m",
    "diminished": "dim",
    "sus2": "sus2",
    "sus4": "sus4",
    "power": "5",
    "dominant7": "7",
    "major7": "maj7",
    "minor7": "m7",
}

_ID_RE = re.compile(r"[a-z][a-z0-9_-]{0,95}")
_SLUG_RE = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")
_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_UNSAFE_TEXT_RE = re.compile(
    r"(?:[A-Za-z]:[\\/]|/(?:home|users|tmp|var|etc|mnt|private|opt|usr)/|\w+://)",
    re.IGNORECASE,
)

_SECRET_TEXT_RE = re.compile(
    r"\b(?:token|password|secret|authorization|api[_-]?key|access[_-]?key)\b\s*[:=]",
    re.IGNORECASE,
)

_MAX_EVENTS = 100_000
_MAX_WINDOWS = 20_000
_MAX_ALTERNATIVES = 3
_MAX_WARNINGS = 32
_MAX_EVENT_WARNINGS = 8
_MAX_WARNING_LENGTH = 240
_MAX_ASSIGNMENTS = 100_000
_FALLBACK_WINDOW_SECONDS = 1.0
_MIN_BEAT_CONFIDENCE = 0.25
_MIN_EVENT_CONFIDENCE = 0.05
_MIN_CANDIDATE_SUPPORT = 0.52
_MIN_RESOLVED_CONFIDENCE = 0.48
_AMBIGUITY_MARGIN = 0.08
_BASS_INVERSION_CONFIDENCE = 0.60


class HarmonyInferenceError(RuntimeError):
    """Raised when harmonic evidence cannot be interpreted safely."""


@dataclass(frozen=True, slots=True)
class HarmonyInferenceResult:
    version: str
    raw_evidence: tuple[dict[str, Any], ...]
    segments: tuple[dict[str, Any], ...]
    tonal_context: dict[str, Any] | None
    unresolved_event_ids: tuple[str, ...]
    warnings: tuple[str, ...]
    diagnostics: dict[str, Any]

    def payload(self) -> dict[str, Any]:
        """Return a detached JSON-compatible mapping for artifact integration."""
        return copy.deepcopy(asdict(self))


@dataclass(frozen=True, slots=True)
class _Event:
    event_id: str
    source_kind: str
    start: float
    end: float
    midi_note: int
    midi_pitch: float
    confidence: float

    @property
    def pitch_class(self) -> int:
        return self.midi_note % 12


@dataclass(frozen=True, slots=True)
class _Window:
    start: float
    end: float
    beat_index: int | None
    mode: str


@dataclass(frozen=True, slots=True)
class _TonalContext:
    root_pc: int
    tonal_center: str
    collection: str
    confidence: float
    display_name: str


@dataclass(frozen=True, slots=True)
class _PartAssignment:
    status: str
    part_id: str | None
    voice_id: str | None


def infer_harmony(
    pitched_events: Sequence[Mapping[str, Any]],
    timing: Mapping[str, Any] | None = None,
    tonality: Mapping[str, Any] | None = None,
    pitched_part_evidence: Mapping[str, Any] | None = None,
    *,
    version: str = HARMONY_INFERENCE_VERSION,
) -> HarmonyInferenceResult:
    """Return conservative harmonic candidates without changing raw evidence."""
    _validate_version(version)
    events = _parse_events(pitched_events)
    timing_data = _parse_timing(timing)
    tonal_context = _parse_tonality(tonality)
    assignments = _parse_part_evidence(pitched_part_evidence, events)
    raw_evidence = tuple(_raw_event_payload(event) for event in events)

    windows, window_mode = _build_windows(events, timing_data)
    warnings: list[str] = []
    if window_mode == "absolute_time":
        _append_warning(
            warnings,
            "Reliable beat evidence is unavailable; harmonic windows use an explicit absolute-time fallback.",
        )
    if tonal_context is None:
        _append_warning(
            warnings,
            "Global tonal context is unavailable or too weak; chord candidates rely on local pitch evidence only.",
        )
    elif tonal_context.confidence < 0.50:
        _append_warning(
            warnings,
            "Global tonal context is weak and is used only as a small advisory score adjustment.",
        )

    source_kinds = sorted({event.source_kind for event in events})
    if source_kinds == ["full_mix"]:
        _append_warning(
            warnings,
            "Full-mix pitch evidence may omit simultaneous chord tones; harmonic candidates remain intentionally conservative.",
        )
    unassigned_context_ids = sorted(
        event_id
        for event_id, assignment in assignments.items()
        if assignment.status != "assigned"
    )
    if unassigned_context_ids:
        _append_warning(
            warnings,
            f"{len(unassigned_context_ids)} pitched event(s) remain unassigned in editable part evidence; raw pitches are still retained.",
        )

    segments: list[dict[str, Any]] = []
    resolved_event_ids: set[str] = set()
    for index, window in enumerate(windows, start=1):
        segment = _segment_for_window(
            segment_id=f"h{index:06d}",
            window=window,
            events=events,
            tonal_context=tonal_context,
            assignments=assignments,
        )
        if segment is None:
            continue
        segments.append(segment)
        if not segment["unresolved"] and segment["primaryCandidate"] is not None:
            resolved_event_ids.update(segment["supportingEventIds"])

    unresolved_event_ids = tuple(
        event.event_id
        for event in events
        if event.event_id not in resolved_event_ids
    )
    diagnostics = {
        "eventCount": len(events),
        "segmentCount": len(segments),
        "resolvedSegmentCount": sum(not segment["unresolved"] for segment in segments),
        "unresolvedSegmentCount": sum(segment["unresolved"] for segment in segments),
        "unresolvedEventCount": len(unresolved_event_ids),
        "sourceKinds": source_kinds,
        "windowingMode": window_mode,
        "fallbackWindowSeconds": (
            _FALLBACK_WINDOW_SECONDS if window_mode == "absolute_time" else None
        ),
        "rawTimingAuthoritative": True,
        "fractionalPitchPreserved": True,
        "rawEvidenceIncluded": True,
        "tonalContextAdvisoryOnly": True,
        "bassSourceRequiredForInversion": True,
        "chordVocabulary": [quality for quality, _ in _CHORD_TEMPLATES],
        "romanNumeralsGenerated": False,
        "guitarVoicingsGenerated": False,
        "notationGenerated": False,
    }
    result = HarmonyInferenceResult(
        version=version,
        raw_evidence=raw_evidence,
        segments=tuple(segments),
        tonal_context=_tonal_payload(tonal_context),
        unresolved_event_ids=unresolved_event_ids,
        warnings=tuple(warnings),
        diagnostics=diagnostics,
    )
    try:
        json.dumps(result.payload(), ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise HarmonyInferenceError("Harmonic inference produced unsafe JSON data.") from exc
    return result


def _raw_event_payload(event: _Event) -> dict[str, Any]:
    return {
        "id": event.event_id,
        "sourceKind": event.source_kind,
        "rawStartSeconds": event.start,
        "rawEndSeconds": event.end,
        "midiNote": event.midi_note,
        "midiPitch": event.midi_pitch,
        "pitchClass": event.pitch_class,
        "pitchName": PITCH_CLASS_NAMES[event.pitch_class],
        "confidence": event.confidence,
    }


def _parse_events(values: object) -> tuple[_Event, ...]:
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise HarmonyInferenceError("Pitched events must be an array.")
    if len(values) > _MAX_EVENTS:
        raise HarmonyInferenceError("Too many pitched events for harmonic inference.")

    seen: set[str] = set()
    events: list[_Event] = []
    for index, value in enumerate(values):
        label = f"pitched event {index}"
        if not isinstance(value, Mapping):
            raise HarmonyInferenceError(f"{label} must be an object.")
        event_id = _safe_id(value.get("id"), f"{label} id")
        if event_id in seen:
            raise HarmonyInferenceError("Pitched event IDs must be unique.")
        seen.add(event_id)
        source_kind = _safe_slug(value.get("sourceKind"), f"{label} sourceKind")
        start = _number(value.get("startSeconds"), f"{label} startSeconds", minimum=0)
        end = _number(value.get("endSeconds"), f"{label} endSeconds", minimum=0)
        if end <= start:
            raise HarmonyInferenceError(f"{label} has an invalid raw time range.")
        midi_note = _integer(value.get("midiNote"), f"{label} midiNote", 0, 127)
        midi_pitch = _number(value.get("midiPitch"), f"{label} midiPitch")
        if not -0.75 <= midi_pitch - midi_note <= 0.75:
            raise HarmonyInferenceError(
                f"{label} fractional pitch is inconsistent with midiNote."
            )
        confidence = _confidence(value.get("confidence"), f"{label} confidence")
        _validate_warnings(value.get("warnings", []), f"{label} warnings")
        events.append(
            _Event(
                event_id=event_id,
                source_kind=source_kind,
                start=start,
                end=end,
                midi_note=midi_note,
                midi_pitch=midi_pitch,
                confidence=confidence,
            )
        )
    events.sort(key=lambda event: (event.start, event.end, event.event_id))
    return tuple(events)


def _parse_timing(value: object) -> dict[str, Any]:
    if value is None:
        return {"beats": (), "beatConfidence": None}
    if not isinstance(value, Mapping):
        raise HarmonyInferenceError("Timing evidence must be an object.")
    beats_value = value.get("beatsSeconds", [])
    if not isinstance(beats_value, Sequence) or isinstance(
        beats_value, (str, bytes, bytearray)
    ):
        raise HarmonyInferenceError("Beat times must be an array.")
    if len(beats_value) > _MAX_WINDOWS + 1:
        raise HarmonyInferenceError("Too many beat times for harmonic inference.")
    beats = tuple(_number(item, "beat time", minimum=0) for item in beats_value)
    if any(later <= earlier for earlier, later in zip(beats, beats[1:])):
        raise HarmonyInferenceError("Beat times must be strictly increasing.")
    confidence = value.get("beatConfidence")
    beat_confidence = (
        None if confidence is None else _confidence(confidence, "beat confidence")
    )
    return {"beats": beats, "beatConfidence": beat_confidence}


def _parse_tonality(value: object) -> _TonalContext | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise HarmonyInferenceError("Tonality evidence must be an object.")
    primary = value.get("primaryCandidate")
    candidate = primary if isinstance(primary, Mapping) else value
    center = candidate.get("tonalCenter") or candidate.get("key") or value.get("tonalCenter")
    collection = candidate.get("collection") or value.get("collection")
    if collection is None:
        legacy_mode = candidate.get("legacyMode") or candidate.get("mode") or value.get("mode")
        if legacy_mode == "major":
            collection = "ionian"
        elif legacy_mode == "minor":
            collection = "aeolian"
    confidence_value = candidate.get("confidence")
    if confidence_value is None:
        confidence_value = value.get("confidence")
    if center is None or collection is None or confidence_value is None:
        return None
    if not isinstance(center, str) or center not in PITCH_CLASS_NAMES:
        raise HarmonyInferenceError("Tonality tonalCenter is unsupported.")
    collection_slug = _safe_slug(collection, "tonality collection")
    confidence = _confidence(confidence_value, "tonality confidence")
    if confidence <= 0:
        return None
    display_name = f"{center} {collection_slug}"
    return _TonalContext(
        root_pc=PITCH_CLASS_NAMES.index(center),
        tonal_center=center,
        collection=collection_slug,
        confidence=confidence,
        display_name=display_name,
    )


def _parse_part_evidence(
    value: object,
    events: Sequence[_Event],
) -> dict[str, _PartAssignment]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise HarmonyInferenceError("Pitched-part evidence must be an object.")
    version = value.get("version")
    if version is not None:
        _validate_open_version(version, "pitched-part evidence version")
    assignments_value = value.get("assignments", [])
    if not isinstance(assignments_value, Sequence) or isinstance(
        assignments_value, (str, bytes, bytearray)
    ):
        raise HarmonyInferenceError("Pitched-part assignments must be an array.")
    if len(assignments_value) > _MAX_ASSIGNMENTS:
        raise HarmonyInferenceError("Too many pitched-part assignments.")
    event_ids = {event.event_id for event in events}
    assignments: dict[str, _PartAssignment] = {}
    for index, item in enumerate(assignments_value):
        if not isinstance(item, Mapping):
            raise HarmonyInferenceError(
                f"Pitched-part assignment {index} must be an object."
            )
        event_id = _safe_id(item.get("eventId"), "pitched-part assignment eventId")
        if event_id not in event_ids:
            raise HarmonyInferenceError(
                "Pitched-part evidence references an unknown raw event."
            )
        if event_id in assignments:
            raise HarmonyInferenceError(
                "Pitched-part evidence contains duplicate primary assignments."
            )
        status = item.get("status")
        if status not in {"assigned", "unassigned"}:
            raise HarmonyInferenceError("Pitched-part assignment status is invalid.")
        part_id = item.get("partId")
        voice_id = item.get("voiceId")
        if part_id is not None:
            part_id = _safe_id(part_id, "pitched-part assignment partId")
        if voice_id is not None:
            voice_id = _safe_id(voice_id, "pitched-part assignment voiceId")
        if status == "assigned" and (part_id is None or voice_id is None):
            raise HarmonyInferenceError(
                "Assigned pitched-part evidence requires partId and voiceId."
            )
        assignments[event_id] = _PartAssignment(
            status=status,
            part_id=part_id,
            voice_id=voice_id,
        )
    unassigned = value.get("unassignedEventIds", [])
    if not isinstance(unassigned, Sequence) or isinstance(
        unassigned, (str, bytes, bytearray)
    ):
        raise HarmonyInferenceError("unassignedEventIds must be an array.")
    for event_id in unassigned:
        safe_event_id = _safe_id(event_id, "unassigned event ID")
        if safe_event_id not in event_ids:
            raise HarmonyInferenceError(
                "Pitched-part evidence references an unknown unassigned event."
            )
        existing = assignments.get(safe_event_id)
        if existing is None:
            assignments[safe_event_id] = _PartAssignment("unassigned", None, None)
        elif existing.status != "unassigned":
            raise HarmonyInferenceError(
                "Pitched-part evidence disagrees about assignment status."
            )
    return assignments


def _build_windows(
    events: Sequence[_Event],
    timing: Mapping[str, Any],
) -> tuple[tuple[_Window, ...], str]:
    if not events:
        return (), "absolute_time"
    beats: tuple[float, ...] = timing["beats"]
    beat_confidence = timing["beatConfidence"]
    if (
        len(beats) >= 2
        and beat_confidence is not None
        and beat_confidence >= _MIN_BEAT_CONFIDENCE
    ):
        windows: list[_Window] = []
        earliest = min(event.start for event in events)
        latest = max(event.end for event in events)
        if earliest < beats[0]:
            windows.append(_Window(earliest, beats[0], None, "absolute_leading"))
        for beat_index, (start, end) in enumerate(zip(beats, beats[1:])):
            if end <= earliest or start >= latest:
                continue
            windows.append(_Window(start, end, beat_index, "beat"))
        if latest > beats[-1]:
            windows.append(_Window(beats[-1], latest, None, "absolute_trailing"))
        if len(windows) > _MAX_WINDOWS:
            raise HarmonyInferenceError("Too many harmonic windows.")
        return tuple(windows), "beat_relative"

    earliest = min(event.start for event in events)
    latest = max(event.end for event in events)
    if latest <= earliest:
        return (), "absolute_time"
    start = math.floor(earliest / _FALLBACK_WINDOW_SECONDS) * _FALLBACK_WINDOW_SECONDS
    windows = []
    index = 0
    while start < latest:
        end = start + _FALLBACK_WINDOW_SECONDS
        windows.append(_Window(start, end, None, "absolute"))
        start = end
        index += 1
        if index > _MAX_WINDOWS:
            raise HarmonyInferenceError("Too many harmonic windows.")
    return tuple(windows), "absolute_time"


def _segment_for_window(
    *,
    segment_id: str,
    window: _Window,
    events: Sequence[_Event],
    tonal_context: _TonalContext | None,
    assignments: Mapping[str, _PartAssignment],
) -> dict[str, Any] | None:
    contributions: list[tuple[_Event, float]] = []
    pitch_class_weights = [0.0] * 12
    for event in events:
        overlap = min(event.end, window.end) - max(event.start, window.start)
        if overlap <= 0 or event.confidence < _MIN_EVENT_CONFIDENCE:
            continue
        weight = overlap * event.confidence
        if weight <= 0:
            continue
        contributions.append((event, weight))
        pitch_class_weights[event.pitch_class] += weight
    if not contributions:
        return None

    total_weight = sum(pitch_class_weights)
    observed = [
        {
            "pitchClass": pitch_class,
            "pitchName": PITCH_CLASS_NAMES[pitch_class],
            "weight": round(weight, 9),
            "weightRatio": round(weight / total_weight, 9),
        }
        for pitch_class, weight in enumerate(pitch_class_weights)
        if weight > 0
    ]
    ordered_events = sorted(
        {event.event_id: event for event, _ in contributions}.values(),
        key=lambda event: (event.start, event.end, event.event_id),
    )
    supporting_ids = [event.event_id for event in ordered_events]
    source_kinds = sorted({event.source_kind for event, _ in contributions})
    part_ids = sorted(
        {
            assignment.part_id
            for event_id in supporting_ids
            if (assignment := assignments.get(event_id)) is not None
            and assignment.status == "assigned"
            and assignment.part_id is not None
        }
    )
    voice_ids = sorted(
        {
            assignment.voice_id
            for event_id in supporting_ids
            if (assignment := assignments.get(event_id)) is not None
            and assignment.status == "assigned"
            and assignment.voice_id is not None
        }
    )
    unassigned_ids = [
        event_id
        for event_id in supporting_ids
        if (assignment := assignments.get(event_id)) is not None
        and assignment.status != "assigned"
    ]

    ranked = _rank_candidates(
        pitch_class_weights=pitch_class_weights,
        contributions=contributions,
        tonal_context=tonal_context,
    )
    primary = ranked[0] if ranked else None
    alternatives = ranked[1 : 1 + _MAX_ALTERNATIVES] if ranked else []
    unresolved = (
        primary is None
        or primary["confidence"] < _MIN_RESOLVED_CONFIDENCE
        or primary["templateCoverage"]
        < (1.0 if primary["quality"] == "power" else 2 / 3)
    )
    segment_warnings: list[str] = []
    if primary is None:
        _append_warning(
            segment_warnings,
            "Local pitch evidence does not support a baseline chord candidate.",
        )
    else:
        if primary["quality"] == "power":
            _append_warning(
                segment_warnings,
                "Power-chord evidence lacks a third, so major/minor quality remains unresolved.",
            )
        if (
            alternatives
            and primary["confidence"] - alternatives[0]["confidence"] < _AMBIGUITY_MARGIN
        ):
            _append_warning(
                segment_warnings,
                "Multiple harmonic candidates have similar support; keep alternatives editable.",
            )
        if primary["nonChordToneRatio"] > 0.25:
            _append_warning(
                segment_warnings,
                "Substantial non-chord pitch evidence is present in this window.",
            )
    if unassigned_ids:
        _append_warning(
            segment_warnings,
            "Some supporting raw events are unassigned in editable pitched-part evidence.",
        )

    result: dict[str, Any] = {
        "id": segment_id,
        "rawStartSeconds": window.start,
        "rawEndSeconds": window.end,
        "windowMode": window.mode,
        "supportingEventIds": supporting_ids,
        "sourceKinds": source_kinds,
        "partIds": part_ids,
        "voiceIds": voice_ids,
        "unassignedContextEventIds": unassigned_ids,
        "observedPitchClasses": observed,
        "primaryCandidate": None if unresolved else primary,
        "alternatives": (
            [copy.deepcopy(primary), *copy.deepcopy(alternatives)]
            if unresolved and primary is not None
            else copy.deepcopy(alternatives)
        )[:_MAX_ALTERNATIVES],
        "unresolved": unresolved,
        "warnings": segment_warnings,
    }
    if window.beat_index is not None:
        result["beatIndex"] = window.beat_index
    return result


def _rank_candidates(
    *,
    pitch_class_weights: Sequence[float],
    contributions: Sequence[tuple[_Event, float]],
    tonal_context: _TonalContext | None,
) -> list[dict[str, Any]]:
    total = sum(pitch_class_weights)
    if total <= 0:
        return []
    observed_count = sum(weight > total * 0.04 for weight in pitch_class_weights)
    if observed_count < 2:
        return []

    raw: list[dict[str, Any]] = []
    for root_pc in range(12):
        for quality, intervals in _CHORD_TEMPLATES:
            chord_pcs = tuple((root_pc + interval) % 12 for interval in intervals)
            chord_weight = sum(pitch_class_weights[pitch_class] for pitch_class in chord_pcs)
            support = chord_weight / total
            present = sum(
                pitch_class_weights[pitch_class] > total * 0.04
                for pitch_class in chord_pcs
            )
            coverage = present / len(chord_pcs)
            minimum_coverage = (
                1.0 if quality == "power" else 0.75 if len(chord_pcs) == 4 else 2 / 3
            )
            if coverage < minimum_coverage or support < _MIN_CANDIDATE_SUPPORT:
                continue
            root_ratio = pitch_class_weights[root_pc] / total
            root_presence = min(1.0, root_ratio / 0.20) if root_ratio > 0 else 0.0
            non_chord_ratio = max(0.0, 1.0 - support)
            source_diversity = min(
                1.0,
                len({event.source_kind for event, _ in contributions}) / 3.0,
            )
            score = (
                0.50 * support
                + 0.28 * coverage
                + 0.14 * root_presence
                + 0.08 * source_diversity
                - 0.10 * non_chord_ratio
            )
            tonal_support = _tonal_support(root_pc, quality, tonal_context)
            score += tonal_support
            if len(chord_pcs) == 3 and coverage < 1.0:
                score -= 0.20
            if len(chord_pcs) == 4 and coverage < 1.0:
                score -= 0.06
            if quality == "power":
                score = min(score, 0.68)
            candidate = {
                "rootPitchClass": root_pc,
                "root": PITCH_CLASS_NAMES[root_pc],
                "quality": quality,
                "symbol": f"{PITCH_CLASS_NAMES[root_pc]}{_QUALITY_SUFFIX[quality]}",
                "pitchClasses": list(chord_pcs),
                "score": _clamp(score),
                "templateCoverage": round(coverage, 9),
                "chordToneWeightRatio": round(support, 9),
                "nonChordToneRatio": round(non_chord_ratio, 9),
                "rootWeightRatio": round(root_ratio, 9),
                "tonalContextSupport": round(tonal_support, 9),
                "evidenceEventIds": sorted(
                    {
                        event.event_id
                        for event, _ in contributions
                        if event.pitch_class in chord_pcs
                    }
                ),
            }
            inversion = _bass_inversion_candidate(
                root_pc=root_pc,
                chord_pcs=chord_pcs,
                candidate_score=candidate["score"],
                contributions=contributions,
            )
            if inversion is not None:
                candidate["inversionCandidate"] = inversion
            raw.append(candidate)

    raw.sort(
        key=lambda candidate: (
            -candidate["score"],
            -candidate["templateCoverage"],
            candidate["rootPitchClass"],
            candidate["quality"],
        )
    )
    if not raw:
        return []
    best_score = raw[0]["score"]
    second_score = raw[1]["score"] if len(raw) > 1 else 0.0
    best_margin = max(0.0, best_score - second_score)
    output: list[dict[str, Any]] = []
    for index, candidate in enumerate(raw[: 1 + _MAX_ALTERNATIVES]):
        if index == 0:
            margin_factor = _clamp(best_margin / 0.12)
            confidence = _clamp(0.68 * candidate["score"] + 0.32 * margin_factor)
        else:
            relative = _clamp(candidate["score"] / max(best_score, 1e-9))
            confidence = _clamp(0.72 * candidate["score"] * relative)
        if candidate["quality"] == "power":
            confidence = min(confidence, 0.60)
        clean = copy.deepcopy(candidate)
        clean["confidence"] = round(confidence, 9)
        output.append(clean)
    return output


def _bass_inversion_candidate(
    *,
    root_pc: int,
    chord_pcs: Sequence[int],
    candidate_score: float,
    contributions: Sequence[tuple[_Event, float]],
) -> dict[str, Any] | None:
    bass_events = [
        event
        for event, _ in contributions
        if event.source_kind == "bass" and event.confidence >= _BASS_INVERSION_CONFIDENCE
    ]
    if not bass_events:
        return None
    bass_events.sort(key=lambda event: (event.midi_pitch, event.start, event.event_id))
    lowest_pitch = bass_events[0].midi_pitch
    lowest = [
        event
        for event in bass_events
        if abs(event.midi_pitch - lowest_pitch) <= 0.35
    ]
    bass_pc = lowest[0].pitch_class
    if bass_pc not in chord_pcs:
        return None
    interval = (bass_pc - root_pc) % 12
    ordered_intervals = [((pitch_class - root_pc) % 12) for pitch_class in chord_pcs]
    try:
        position_index = ordered_intervals.index(interval)
    except ValueError:
        return None
    labels = ("root_position", "first_inversion", "second_inversion", "third_inversion")
    position = (
        labels[position_index]
        if position_index < len(labels)
        else "upper_chord_tone_bass"
    )
    confidence = min(
        candidate_score,
        statistics.mean(event.confidence for event in lowest),
    )
    return {
        "bassPitchClass": bass_pc,
        "bassPitchName": PITCH_CLASS_NAMES[bass_pc],
        "position": position,
        "confidence": round(_clamp(confidence), 9),
        "sourceEventIds": [event.event_id for event in lowest],
    }


def _tonal_support(
    chord_root: int,
    quality: str,
    context: _TonalContext | None,
) -> float:
    if context is None or context.confidence <= 0:
        return 0.0
    if context.collection == "ionian":
        degrees = (0, 2, 4, 5, 7, 9, 11)
        qualities = ("major", "minor", "minor", "major", "major", "minor", "diminished")
    elif context.collection == "aeolian":
        degrees = (0, 2, 3, 5, 7, 8, 10)
        qualities = ("minor", "diminished", "major", "minor", "minor", "major", "major")
    else:
        return 0.0
    relative = (chord_root - context.root_pc) % 12
    for degree, expected_quality in zip(degrees, qualities):
        if relative == degree and quality == expected_quality:
            return 0.04 * context.confidence
    return 0.0


def _tonal_payload(context: _TonalContext | None) -> dict[str, Any] | None:
    if context is None:
        return None
    return {
        "tonalCenter": context.tonal_center,
        "collection": context.collection,
        "displayName": context.display_name,
        "confidence": context.confidence,
        "advisoryOnly": True,
    }


def _validate_version(value: object) -> None:
    _validate_open_version(value, "harmony inference version")


def _validate_open_version(value: object, label: str) -> str:
    if not isinstance(value, str) or not _VERSION_RE.fullmatch(value):
        raise HarmonyInferenceError(f"{label} is invalid.")
    return value


def _safe_id(value: object, label: str) -> str:
    if not isinstance(value, str) or not _ID_RE.fullmatch(value):
        raise HarmonyInferenceError(f"{label} is unsafe.")
    return value


def _safe_slug(value: object, label: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise HarmonyInferenceError(f"{label} is unsafe.")
    return value


def _number(value: object, label: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarmonyInferenceError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise HarmonyInferenceError(f"{label} must be finite.")
    if minimum is not None and number < minimum:
        raise HarmonyInferenceError(f"{label} is below its minimum.")
    return number


def _integer(value: object, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise HarmonyInferenceError(
            f"{label} must be an integer from {minimum} through {maximum}."
        )
    return value


def _confidence(value: object, label: str) -> float:
    number = _number(value, label)
    if not 0.0 <= number <= 1.0:
        raise HarmonyInferenceError(f"{label} must be between 0 and 1.")
    return number


def _validate_warnings(value: object, label: str) -> None:
    if value is None:
        return
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HarmonyInferenceError(f"{label} must be an array.")
    if len(value) > _MAX_EVENT_WARNINGS:
        raise HarmonyInferenceError(f"{label} contains too many warnings.")
    for warning in value:
        if (
            not isinstance(warning, str)
            or not warning
            or len(warning) > _MAX_WARNING_LENGTH
            or _UNSAFE_TEXT_RE.search(warning)
            or _SECRET_TEXT_RE.search(warning)
            or "<" in warning
            or ">" in warning
            or any(ord(character) < 32 or ord(character) == 127 for character in warning)
        ):
            raise HarmonyInferenceError(f"{label} contains unsafe text.")


def _append_warning(warnings: list[str], warning: str) -> None:
    if warning not in warnings and len(warnings) < _MAX_WARNINGS:
        warnings.append(warning)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


__all__ = [
    "HARMONY_INFERENCE_VERSION",
    "HarmonyInferenceError",
    "HarmonyInferenceResult",
    "infer_harmony",
]
