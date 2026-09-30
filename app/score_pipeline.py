"""Build and publish one persisted draft-score document for a score attempt.

The pipeline reuses the reviewed score builder and exporters from
:mod:`app.score_construction` and the evidence checks from
:mod:`app.score_api`. It adds two optional, honestly labelled layers:

- chord symbols from a completed, matching harmonic context: a measure gets a
  symbol only when one resolved candidate covers at least half the measure and
  no competing candidate covers a quarter of it; every overlapping window,
  including unresolved ones, is kept as per-measure review evidence;
- part labels from a completed, matching editable interpretation: a note is
  labelled only when its raw event maps to exactly one interpretation part.

Raw events and interpretation drafts are read, never modified. Exports are
dry-run before publication so a saved score is always downloadable.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.config import Settings
from app.harmony_artifacts import HarmonyArtifactError, load_harmony_artifact
from app.score_api import (
    ScorePreviewError,
    ScorePreviewUnavailableError,
    collect_score_evidence,
    score_evidence_warnings,
)
from app.score_artifacts import (
    SCORE_ARTIFACT_SCHEMA_VERSION,
    SCORE_ARTIFACT_TYPE,
    ScoreArtifactError,
    score_attempt_artifact_file_name,
    write_score_artifact,
)
from app.score_construction import (
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    score_to_midi_bytes,
    score_to_musicxml_text,
)
from app.score_sources import score_source_fingerprint, score_source_identity
from app.transcription_draft import TranscriptionDraftError, load_transcription_draft

SCORE_PIPELINE_VERSION = "score-pipeline-v1"
SCORE_MIDI_EXPORT_VERSION = "smf-type0-v1"
SCORE_MUSICXML_EXPORT_VERSION = "musicxml-3.1-partwise-v1"

_MAX_WARNINGS = 32
_MAX_MEASURE_HARMONY = 16
_DOMINANT_COVERAGE = 0.5
_COMPETING_COVERAGE = 0.25

ProgressCallback = Callable[[str, str, float], None]


class ScorePipelineError(RuntimeError):
    """Score construction failed with a musician-safe message."""


@dataclass(frozen=True, slots=True)
class ScorePipelineResult:
    artifact_file_name: str
    created_at: str
    source_fingerprint: str
    pipeline_version: str
    measure_count: int
    note_count: int
    chord_symbol_count: int
    warning_count: int
    payload: dict[str, Any]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def score_export_document(document: Mapping[str, Any]) -> dict[str, Any]:
    """Return the builder-shaped document the MIDI/MusicXML exporters accept."""
    timing = document["timing"]
    return {
        "schemaVersion": SCORE_SCHEMA_VERSION,
        "tempoBpm": timing["tempoBpm"],
        "beatsPerMeasure": timing["beatsPerMeasure"],
        "divisions": timing["divisions"],
        "measureCount": len(document["measures"]),
        "measures": document["measures"],
    }


def _load_matching_harmony(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    raw_ids: set[str],
) -> tuple[list[dict[str, Any]] | None, str]:
    """Return harmony segments that match current evidence, or an omission note."""
    layer = identity.get("harmony")
    if layer is None:
        return None, (
            "Harmonic context was not complete when this score was built; "
            "no chord symbols are included."
        )
    try:
        artifact = load_harmony_artifact(
            job_id, settings, artifact_file_name=layer["fileName"]
        )
    except HarmonyArtifactError:
        return None, (
            "Saved harmonic context could not be validated; chord symbols were omitted."
        )
    if artifact is None:
        return None, "Saved harmonic context is unavailable; chord symbols were omitted."
    source_transcription = artifact.get("sourceTranscription") or {}
    source_analysis = artifact.get("sourceAnalysis") or {}
    raw_evidence = artifact.get("rawEvidence") or []
    if (
        artifact.get("harmonyVersion") != layer["version"]
        or artifact.get("createdAt") != layer["createdAt"]
        or source_transcription.get("transcriptionVersion")
        != record.get("transcription_version")
        or record.get("harmony_source_transcription_version")
        != record.get("transcription_version")
        or record.get("harmony_source_transcribed_at") != record.get("transcribed_at")
        or source_analysis.get("analysisVersion") != record.get("analysis_version")
        or not {item.get("id") for item in raw_evidence} <= raw_ids
    ):
        return None, (
            "Saved harmonic context does not match the current transcription; "
            "chord symbols were omitted. Re-run harmony, then rebuild the score."
        )
    return list(artifact.get("segments") or []), "Chord symbols come from resolved local harmonic candidates."


def _load_matching_parts(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
    raw_events: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Return interpretation parts and raw-event assignments, or an omission note."""
    layer = identity.get("interpretation")
    if layer is None:
        return None, (
            "Editable interpretation was not complete when this score was built; "
            "notes carry no part labels."
        )
    try:
        draft = load_transcription_draft(job_id, settings)
    except (TranscriptionDraftError, OSError, RuntimeError):
        return None, (
            "Saved editable interpretation could not be validated; part labels were omitted."
        )
    if draft is None:
        return None, "Saved editable interpretation is unavailable; part labels were omitted."
    source = draft.get("sourceTranscription") or {}
    indexed = {
        item["id"]: item
        for item in source.get("sourceEventIndex", ())
        if item.get("eventType") == "pitched"
    }
    if (
        draft.get("draftVersion") != layer["version"]
        or draft.get("createdAt") != layer["createdAt"]
        or source.get("transcriptionVersion") != record.get("transcription_version")
        or any(
            event_id not in raw_events
            or not math.isclose(
                float(item["rawStartSeconds"]),
                float(raw_events[event_id]["startSeconds"]),
                abs_tol=1e-6,
            )
            for event_id, item in indexed.items()
        )
    ):
        return None, (
            "Saved editable interpretation does not match the current transcription; "
            "part labels were omitted. Re-run interpretation, then rebuild the score."
        )
    parts = {part["id"]: part for part in draft.get("parts", ())}
    assignments: dict[str, set[str]] = {}
    for item in draft.get("pitchedItems", ()):
        part_id = item.get("partId")
        if item.get("interpretationType") != "note" or part_id not in parts:
            continue
        for event_id in item.get("sourceEventIds", ()):
            assignments.setdefault(event_id, set()).add(part_id)
    return {"parts": parts, "assignments": assignments}, (
        "Part labels come from the editable interpretation; exports still use "
        "one combined draft part."
    )


def _map_harmony(
    measures: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    *,
    seconds_per_beat: float,
    beats_per_measure: int,
) -> dict[str, int]:
    stats = {"windows": 0, "unresolved": 0, "ambiguous": 0, "beyond": 0, "truncated": 0}
    seconds_per_measure = seconds_per_beat * beats_per_measure
    notated_end = len(measures) * seconds_per_measure
    for segment in segments:
        start = float(segment["rawStartSeconds"])
        if start >= notated_end:
            stats["beyond"] += 1
            continue
        stats["windows"] += 1
        if segment.get("unresolved"):
            stats["unresolved"] += 1
    for measure in measures:
        measure_start = measure["startSeconds"]
        measure_end = measure["endSeconds"]
        coverage: dict[str, float] = {}
        entries = []
        for segment in segments:
            start = max(measure_start, float(segment["rawStartSeconds"]))
            end = min(measure_end, float(segment["rawEndSeconds"]))
            if end <= start:
                continue
            candidate = segment.get("primaryCandidate")
            unresolved = bool(segment.get("unresolved")) or not isinstance(candidate, Mapping)
            symbol = None if unresolved else str(candidate["symbol"])
            if symbol is not None:
                coverage[symbol] = coverage.get(symbol, 0.0) + (end - start)
            entries.append(
                {
                    "segmentId": str(segment["id"]),
                    "startBeat": round((start - measure_start) / seconds_per_beat, 3),
                    "endBeat": round((end - measure_start) / seconds_per_beat, 3),
                    "symbol": symbol,
                    "confidence": None if unresolved else float(candidate["confidence"]),
                    "unresolved": unresolved,
                }
            )
        entries.sort(key=lambda item: (item["startBeat"], item["endBeat"], item["segmentId"]))
        entries = [entry for entry in entries if entry["endBeat"] > entry["startBeat"]]
        if len(entries) > _MAX_MEASURE_HARMONY:
            stats["truncated"] += 1
            entries = entries[:_MAX_MEASURE_HARMONY]
        measure["harmony"] = entries
        measure_seconds = measure_end - measure_start
        ranked = sorted(coverage.items(), key=lambda item: (-item[1], item[0]))
        chosen = None
        if ranked and ranked[0][1] >= _DOMINANT_COVERAGE * measure_seconds and not any(
            share >= _COMPETING_COVERAGE * measure_seconds for _, share in ranked[1:]
        ):
            chosen = ranked[0][0]
        measure["chordSymbol"] = chosen
        if chosen is None and coverage:
            stats["ambiguous"] += 1
    return stats


def construct_score(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    progress: ProgressCallback,
    *,
    attempt_id: str,
    expected_fingerprint: str,
) -> ScorePipelineResult:
    """Build, verify and publish one attempt-scoped draft-score document."""
    identity = score_source_identity(record)
    if identity is None:
        raise ScorePipelineError(
            "Completed audio analysis and raw transcription are required before building a score."
        )
    fingerprint = score_source_fingerprint(identity)
    if fingerprint != expected_fingerprint:
        raise ScorePipelineError(
            "Score evidence changed before construction started; build the score again."
        )
    try:
        artifact_file_name = score_attempt_artifact_file_name(attempt_id)
    except ScoreArtifactError as exc:
        raise ScorePipelineError("The score attempt identity is invalid.") from exc

    progress("loading_evidence", "Checking saved analysis and transcription evidence.", 10)
    try:
        evidence = collect_score_evidence(job_id, settings, record)
    except ScorePreviewUnavailableError as exc:
        raise ScorePipelineError(str(exc)) from exc
    except ScorePreviewError as exc:
        raise ScorePipelineError(str(exc)) from exc

    raw_events = {
        event["id"]: event
        for event in evidence["rawTranscription"].get("pitchedNoteEvents", ())
    }

    progress("building_measures", "Placing notes into measures.", 30)
    try:
        built = build_score_document(
            evidence["pitchedInputs"],
            tempo_bpm=evidence["tempoBpm"],
            beats_per_measure=evidence["beatsPerMeasure"],
        )
    except ScoreConstructionError as exc:
        raise ScorePipelineError(
            "The saved evidence could not be placed into measures safely."
        ) from exc
    measures = built["measures"]
    for measure in measures:
        measure["harmony"] = []
    seconds_per_beat = 60.0 / built["tempoBpm"]

    progress("mapping_harmony", "Mapping harmonic candidates to measures.", 50)
    segments, harmony_note = _load_matching_harmony(
        job_id, settings, record, identity, set(raw_events)
    )
    harmony_stats = {"windows": 0, "unresolved": 0, "ambiguous": 0, "beyond": 0, "truncated": 0}
    if segments is not None:
        harmony_stats = _map_harmony(
            measures,
            segments,
            seconds_per_beat=seconds_per_beat,
            beats_per_measure=built["beatsPerMeasure"],
        )

    progress("labelling_parts", "Applying editable-interpretation part labels.", 65)
    part_data, parts_note = _load_matching_parts(
        job_id, settings, record, identity, raw_events
    )
    part_note_counts: dict[str, int] = {}
    multiply_assigned = 0
    for measure in measures:
        for note in measure["notes"]:
            part_id = None
            if part_data is not None:
                candidates = part_data["assignments"].get(note["id"], set())
                if len(candidates) == 1:
                    part_id = next(iter(candidates))
                    part_note_counts[part_id] = part_note_counts.get(part_id, 0) + 1
                elif len(candidates) > 1:
                    multiply_assigned += 1
            note["partId"] = part_id
    parts = []
    if part_data is not None:
        # List only parts that label pitched notes; percussion-only parts are
        # not part of this pitched-note draft.
        pitched_part_ids = set().union(*part_data["assignments"].values())
        for part_id in sorted(pitched_part_ids):
            part = part_data["parts"][part_id]
            parts.append(
                {
                    "id": part_id,
                    "role": part["role"],
                    "instrumentKind": part["instrumentKind"],
                    "sourceKind": part["sourceKind"],
                    "noteCount": part_note_counts.get(part_id, 0),
                }
            )

    progress("checking_exports", "Checking MIDI and MusicXML exports.", 80)
    timing = {
        "tempoBpm": built["tempoBpm"],
        "beatsPerMeasure": built["beatsPerMeasure"],
        "divisions": built["divisions"],
        "meterSource": evidence["meterSource"],
        "tempoConfidence": evidence["tempoConfidence"],
        "tempoStable": evidence["tempoStable"],
        "meterConfidence": evidence["meterConfidence"],
    }
    export_document = score_export_document({"timing": timing, "measures": measures})
    try:
        score_to_midi_bytes(export_document)
        score_to_musicxml_text(export_document)
    except ScoreConstructionError as exc:
        raise ScorePipelineError(
            "The score could not be exported to MIDI and MusicXML safely."
        ) from exc

    warnings = score_evidence_warnings(built, evidence)
    notes_with_part = sum(part_note_counts.values())
    if part_data is not None:
        # The generic single-part warning is superseded by an explicit note.
        warnings = [
            warning
            for warning in warnings
            if not warning.startswith("Pitched events are shown as one draft part")
        ]
        warnings.append(
            f"{notes_with_part} of {built['noteCount']} note(s) carry an editable-"
            "interpretation part label; MIDI and MusicXML still use one combined draft part."
        )
        if multiply_assigned:
            warnings.append(
                f"{multiply_assigned} note(s) map to more than one interpretation part "
                "and were left unlabelled."
            )
    else:
        warnings.append(parts_note)
    if segments is None:
        warnings.append(harmony_note)
    else:
        chord_count = sum(1 for measure in measures if measure["chordSymbol"] is not None)
        warnings.append(
            f"Chord symbols were placed in {chord_count} of {len(measures)} measure(s); "
            "they are candidates to review, not confirmed harmony."
        )
        if harmony_stats["ambiguous"]:
            warnings.append(
                f"{harmony_stats['ambiguous']} measure(s) contain competing or partial "
                "harmonic candidates and show no chord symbol."
            )
        if harmony_stats["unresolved"]:
            warnings.append(
                f"{harmony_stats['unresolved']} harmonic window(s) were unresolved and "
                "remain listed without a chord name."
            )
        if harmony_stats["beyond"]:
            warnings.append(
                f"{harmony_stats['beyond']} harmonic window(s) fall after the last "
                "notated measure and are not shown."
            )
        if harmony_stats["truncated"]:
            warnings.append(
                f"{harmony_stats['truncated']} measure(s) list only their first "
                f"{_MAX_MEASURE_HARMONY} harmonic windows."
            )
    warnings.append(
        "Tablature and drum notation are not part of this draft score yet."
    )
    if len(warnings) > _MAX_WARNINGS:
        warnings = warnings[: _MAX_WARNINGS - 1] + [
            "Additional warnings were truncated; review the score carefully."
        ]

    chord_symbol_count = sum(1 for measure in measures if measure["chordSymbol"] is not None)
    if segments is not None and chord_symbol_count == 0:
        harmony_note = (
            "Harmonic context was checked, but no measure had one clearly dominant "
            "resolved candidate, so no chord symbols are shown."
        )
    percussion_count = evidence["percussionEventCount"]
    payload = {
        "schemaVersion": SCORE_ARTIFACT_SCHEMA_VERSION,
        "artifactType": SCORE_ARTIFACT_TYPE,
        "jobId": job_id,
        "pipelineVersion": SCORE_PIPELINE_VERSION,
        "builderVersion": SCORE_BUILDER_VERSION,
        "createdAt": _utc_now(),
        "sourceFingerprint": fingerprint,
        "sources": identity,
        "layers": {
            "pitchedNotes": {
                "status": "included",
                "note": "Quantized from raw pitched events to an eighth-note grid.",
            },
            "chordSymbols": {
                "status": "included" if segments is not None else "omitted",
                "note": harmony_note,
            },
            "partLabels": {
                "status": "included" if part_data is not None else "omitted",
                "note": parts_note,
            },
            "percussion": {
                "status": "omitted",
                "note": (
                    f"{percussion_count} raw percussion event(s) are preserved in the "
                    "transcription but not notated yet."
                    if percussion_count
                    else "No percussion events were transcribed; drum notation is not built yet."
                ),
            },
            "tablature": {
                "status": "omitted",
                "note": "Guitar and bass tablature are not generated yet.",
            },
        },
        "timing": timing,
        "parts": parts,
        "measures": measures,
        "counts": {
            "measures": len(measures),
            "notes": built["noteCount"],
            "chordSymbols": chord_symbol_count,
            "harmonyWindows": harmony_stats["windows"],
            "unresolvedHarmonyWindows": harmony_stats["unresolved"],
            "ambiguousHarmonyMeasures": harmony_stats["ambiguous"],
            "percussionEvents": percussion_count,
            "notesWithPart": notes_with_part,
        },
        "warnings": warnings,
        "exports": {
            "midi": SCORE_MIDI_EXPORT_VERSION,
            "musicxml": SCORE_MUSICXML_EXPORT_VERSION,
        },
    }

    progress("saving_score", "Saving the draft score.", 90)
    try:
        write_score_artifact(
            job_id, settings, payload, artifact_file_name=artifact_file_name
        )
    except ScoreArtifactError as exc:
        raise ScorePipelineError("The draft score could not be saved safely.") from exc
    return ScorePipelineResult(
        artifact_file_name=artifact_file_name,
        created_at=payload["createdAt"],
        source_fingerprint=fingerprint,
        pipeline_version=SCORE_PIPELINE_VERSION,
        measure_count=len(measures),
        note_count=built["noteCount"],
        chord_symbol_count=chord_symbol_count,
        warning_count=len(warnings),
        payload=payload,
    )


__all__ = [
    "SCORE_PIPELINE_VERSION",
    "ScorePipelineError",
    "ScorePipelineResult",
    "construct_score",
    "score_export_document",
]
