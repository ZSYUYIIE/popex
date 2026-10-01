"""Build and publish one persisted draft-score document for a score attempt.

The pipeline reuses the reviewed score builder and exporters from
:mod:`app.score_construction` and the evidence checks from
:mod:`app.score_api`. It adds two optional, honestly labelled layers:

- chord symbols from a completed, matching harmonic context: a measure gets a
  symbol only when one resolved candidate covers at least half the measure and
  no competing candidate covers a quarter of it; every overlapping window,
  including unresolved ones, is kept as per-measure review evidence;
- part labels from a completed, matching editable interpretation: a note is
  labelled only when its raw event maps to exactly one interpretation part;
- a separate percussion part of broad drum voices built from the raw
  percussion events: voices come from a completed, matching interpretation,
  otherwise from the documented raw hit-kind table; unresolved hits stay in an
  explicit unresolved lane and are never assigned to a specific drum;
- guitar and bass tablature as a separate fingering layer on already
  quantized notes: bass fingers the separated bass-stem line by default and
  guitar only fingers a line the musician chose; pitches are never changed.

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
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HarmonyArtifactError,
    harmony_raw_evidence,
    harmony_raw_evidence_matches,
    load_harmony_artifact,
)
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
from app.percussion_interpretation import broad_voice_for_hit
from app.score_construction import (
    PERCUSSION_NOTATION,
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    percussion_hit_counts,
    score_to_midi_bytes,
    score_to_musicxml_text,
)
from app.score_sources import (
    effective_tablature_request,
    score_source_fingerprint,
    score_source_identity,
)
from app.tablature import (
    TAB_INSTRUMENT_ORDER,
    TAB_INSTRUMENTS,
    TABLATURE_VERSION,
    TablatureError,
    assign_tablature,
)
from app.transcription_draft import TranscriptionDraftError, load_transcription_draft

SCORE_PIPELINE_VERSION = "score-pipeline-v3"
PRE_DRUM_PIPELINE_VERSIONS = frozenset({"score-pipeline-v1"})
PRE_TABLATURE_PIPELINE_VERSIONS = frozenset({"score-pipeline-v1", "score-pipeline-v2"})
_SOURCE_LABELS = {
    "vocals": "vocal-stem line",
    "bass": "bass-stem line",
    "other": "accompaniment-stem line",
    "full_mix": "full-mix melody line",
}
SCORE_MIDI_EXPORT_VERSION = "smf-type0-v2"
SCORE_MUSICXML_EXPORT_VERSION = "musicxml-3.1-partwise-v2"

_MAX_WARNINGS = 32
_MAX_MEASURE_HARMONY = 16
_MAX_SYMBOL_LENGTH = 64
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


def score_outdated_reason(record: Mapping[str, Any]) -> str | None:
    """Return why a saved score predates a layer this recording would get.

    ``drum-notation``: a ``score-pipeline-v1`` score of a recording whose
    transcription holds percussion events. ``tablature``: a score from before
    tablature for a recording with separated stems, where the default bass
    tablature applies. Such scores stay readable and downloadable.
    """
    version = record.get("score_version")
    count = record.get("percussion_event_count")
    if (
        version in PRE_DRUM_PIPELINE_VERSIONS
        and isinstance(count, int)
        and not isinstance(count, bool)
        and count > 0
    ):
        return "drum-notation"
    if version in PRE_TABLATURE_PIPELINE_VERSIONS and record.get("separation_status") == "completed":
        return "tablature"
    return None


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
    raw_transcription: Mapping[str, Any],
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
    if (
        artifact.get("harmonyVersion") != layer["version"]
        or artifact.get("createdAt") != layer["createdAt"]
        or source_transcription.get("transcriptionVersion")
        != record.get("transcription_version")
        or record.get("harmony_source_transcription_version")
        != record.get("transcription_version")
        or record.get("harmony_source_transcribed_at") != record.get("transcribed_at")
        or source_analysis.get("analysisVersion") != record.get("analysis_version")
        or not harmony_raw_evidence_matches(
            artifact.get("rawEvidence"),
            harmony_raw_evidence(raw_transcription),
            allow_legacy_missing_warnings=layer["fileName"] == HARMONY_ARTIFACT_RELATIVE_PATH,
        )
    ):
        return None, (
            "Saved harmonic context does not match the current transcription; "
            "chord symbols were omitted. Re-run harmony, then rebuild the score."
        )
    return list(artifact.get("segments") or []), "Chord symbols come from resolved local harmonic candidates."


def _load_interpretation(
    job_id: str,
    settings: Settings,
    record: Mapping[str, Any],
    identity: Mapping[str, Any],
) -> tuple[Mapping[str, Any] | None, str | None]:
    """Load the identity's interpretation draft once, or return why it is absent.

    The reason is one of ``incomplete``, ``invalid``, ``unavailable`` or
    ``mismatch``; layer-specific notes are worded by the caller.
    """
    layer = identity.get("interpretation")
    if layer is None:
        return None, "incomplete"
    try:
        draft = load_transcription_draft(job_id, settings)
    except (TranscriptionDraftError, OSError, RuntimeError):
        return None, "invalid"
    if draft is None:
        return None, "unavailable"
    source = draft.get("sourceTranscription") or {}
    if (
        draft.get("draftVersion") != layer["version"]
        or draft.get("createdAt") != layer["createdAt"]
        or source.get("transcriptionVersion") != record.get("transcription_version")
    ):
        return None, "mismatch"
    return draft, None


def _load_matching_parts(
    draft: Mapping[str, Any] | None,
    reason: str | None,
    raw_events: Mapping[str, Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, str]:
    """Return interpretation parts and raw-event assignments, or an omission note."""
    if reason == "incomplete":
        return None, (
            "Editable interpretation was not complete when this score was built; "
            "notes carry no part labels."
        )
    if reason == "invalid":
        return None, (
            "Saved editable interpretation could not be validated; part labels were omitted."
        )
    if reason == "unavailable":
        return None, "Saved editable interpretation is unavailable; part labels were omitted."
    mismatch_note = (
        "Saved editable interpretation does not match the current transcription; "
        "part labels were omitted. Re-run interpretation, then rebuild the score."
    )
    if draft is None:
        return None, mismatch_note
    source = draft.get("sourceTranscription") or {}
    indexed = {
        item["id"]: item
        for item in source.get("sourceEventIndex", ())
        if item.get("eventType") == "pitched"
    }
    if any(
        event_id not in raw_events
        or not math.isclose(
            float(item["rawStartSeconds"]),
            float(raw_events[event_id]["startSeconds"]),
            abs_tol=1e-6,
        )
        for event_id, item in indexed.items()
    ):
        return None, mismatch_note
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


def _raw_percussion_hits(raw_percussion: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Map raw hits through the documented broad-voice table."""
    hits = []
    for event in raw_percussion:
        for hit_index, hit in enumerate(event["hits"]):
            voice, resolved = broad_voice_for_hit(hit["kind"], float(hit["confidence"]))
            hits.append(
                {
                    "eventId": event["id"],
                    "hitIndex": hit_index,
                    "sourceKind": event["sourceKind"],
                    "rawKind": hit["kind"],
                    "broadVoice": voice,
                    "resolved": resolved,
                    "timeSeconds": event["timeSeconds"],
                    "strength": event["strength"],
                    "confidence": hit["confidence"],
                }
            )
    return hits


def _interpreted_percussion_hits(
    draft: Mapping[str, Any],
    raw_percussion: list[Mapping[str, Any]],
) -> list[dict[str, Any]] | None:
    """Return hits voiced by the interpretation, or ``None`` on any mismatch.

    Every raw hit must appear exactly once with the same kind and confidence,
    and every interpreted voice must be a known broad voice. Partial matches
    are rejected instead of being combined with raw evidence.
    """
    raw_by_id = {event["id"]: event for event in raw_percussion}
    source = draft.get("sourceTranscription") or {}
    indexed = {
        item["id"]: item
        for item in source.get("sourceEventIndex", ())
        if item.get("eventType") == "percussion"
    }
    if set(indexed) != set(raw_by_id) or any(
        not math.isclose(
            float(item["rawStartSeconds"]),
            float(raw_by_id[event_id]["timeSeconds"]),
            abs_tol=1e-6,
        )
        for event_id, item in indexed.items()
    ):
        return None
    hits: dict[tuple[str, int], dict[str, Any]] = {}
    for item in draft.get("percussionItems", ()):
        event_ids = item.get("sourceEventIds") or []
        if len(event_ids) != 1 or event_ids[0] not in raw_by_id:
            return None
        event = raw_by_id[event_ids[0]]
        placement = item.get("placementStatus")
        for hit in item.get("hits", ()):
            hit_index = hit.get("sourceHitIndex")
            if (
                not isinstance(hit_index, int)
                or not 0 <= hit_index < len(event["hits"])
                or (event["id"], hit_index) in hits
            ):
                return None
            raw_hit = event["hits"][hit_index]
            voice = hit.get("broadVoice")
            if (
                voice not in PERCUSSION_NOTATION
                or hit.get("rawKind") != raw_hit["kind"]
                or not math.isclose(
                    float(hit.get("confidence", -1.0)),
                    float(raw_hit["confidence"]),
                    abs_tol=1e-6,
                )
            ):
                return None
            hits[(event["id"], hit_index)] = {
                "eventId": event["id"],
                "hitIndex": hit_index,
                "sourceKind": event["sourceKind"],
                "rawKind": raw_hit["kind"],
                "broadVoice": voice,
                "resolved": voice != "unresolved_percussion",
                "timeSeconds": event["timeSeconds"],
                "strength": event["strength"],
                "confidence": raw_hit["confidence"],
                "interpretationPlacement": (
                    placement if placement in {"placed", "unassigned"} else None
                ),
            }
    expected = {
        (event["id"], index)
        for event in raw_percussion
        for index in range(len(event["hits"]))
    }
    if set(hits) != expected:
        return None
    return [hits[key] for key in sorted(hits)]


def _percussion_inputs(
    draft: Mapping[str, Any] | None,
    reason: str | None,
    raw_percussion: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], str, str]:
    """Return ``(hits, voice source, layer note)`` for the percussion part."""
    if not raw_percussion:
        return [], "none", "No percussion events were transcribed, so no drum part is written."
    table_note = (
        "Drum voices use the documented raw hit-kind table: broad voices only, "
        "with unresolved hits kept in their own lane."
    )
    if draft is None:
        if reason == "mismatch":
            table_note = (
                "Saved editable interpretation does not match the current transcription, "
                "so it was not used. " + table_note
            )
        return _raw_percussion_hits(raw_percussion), "raw-hit-kinds", table_note
    interpreted = _interpreted_percussion_hits(draft, raw_percussion)
    if interpreted is None:
        return _raw_percussion_hits(raw_percussion), "raw-hit-kinds", (
            "Saved editable interpretation does not match the current percussion "
            "evidence, so it was not used. " + table_note
            + " Re-run interpretation, then rebuild the score."
        )
    return interpreted, "interpretation", (
        "Drum voices come from the editable interpretation's broad voices; "
        "specific kit pieces are not claimed."
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
    ordered = sorted(
        segments,
        key=lambda item: (float(item["rawStartSeconds"]), str(item["id"])),
    )
    for segment in ordered:
        if float(segment["rawStartSeconds"]) >= notated_end:
            stats["beyond"] += 1
            continue
        stats["windows"] += 1
        if segment.get("unresolved") or _segment_symbol(segment) is None:
            stats["unresolved"] += 1
    # Sweep measures in time order, keeping only windows that can still
    # overlap, so the work is proportional to measures plus overlaps.
    active: list[dict[str, Any]] = []
    next_index = 0
    for measure in measures:
        measure_start = measure["startSeconds"]
        measure_end = measure["endSeconds"]
        while (
            next_index < len(ordered)
            and float(ordered[next_index]["rawStartSeconds"]) < measure_end
        ):
            active.append(ordered[next_index])
            next_index += 1
        active = [
            segment for segment in active
            if float(segment["rawEndSeconds"]) > measure_start
        ]
        coverage: dict[str, float] = {}
        entries = []
        for segment in active:
            start = max(measure_start, float(segment["rawStartSeconds"]))
            end = min(measure_end, float(segment["rawEndSeconds"]))
            if end <= start:
                continue
            symbol = _segment_symbol(segment)
            if symbol is not None:
                coverage[symbol] = coverage.get(symbol, 0.0) + (end - start)
            entries.append(
                {
                    "segmentId": str(segment["id"]),
                    "startBeat": round((start - measure_start) / seconds_per_beat, 3),
                    "endBeat": round((end - measure_start) / seconds_per_beat, 3),
                    "symbol": symbol,
                    "confidence": (
                        None
                        if symbol is None
                        else float(segment["primaryCandidate"]["confidence"])
                    ),
                    "unresolved": symbol is None,
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


def _segment_symbol(segment: Mapping[str, Any]) -> str | None:
    """Return a resolved, displayable chord symbol, or ``None``.

    A symbol longer than the score's text bound is treated as unresolved for
    placement rather than failing the whole score.
    """
    candidate = segment.get("primaryCandidate")
    if segment.get("unresolved") or not isinstance(candidate, Mapping):
        return None
    symbol = candidate.get("symbol")
    if not isinstance(symbol, str) or not symbol or len(symbol) > _MAX_SYMBOL_LENGTH:
        return None
    return symbol


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
    raw_percussion = list(evidence["rawTranscription"].get("percussionEvents", ()))
    draft, draft_reason = _load_interpretation(job_id, settings, record, identity)
    percussion_inputs, voice_source, percussion_note = _percussion_inputs(
        draft, draft_reason, raw_percussion
    )

    progress("building_measures", "Placing notes and drum hits into measures.", 30)
    try:
        built = build_score_document(
            evidence["pitchedInputs"],
            tempo_bpm=evidence["tempoBpm"],
            beats_per_measure=evidence["beatsPerMeasure"],
            percussion_hits=percussion_inputs,
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
        job_id, settings, record, identity, evidence["rawTranscription"]
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
    part_data, parts_note = _load_matching_parts(draft, draft_reason, raw_events)
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

    progress("fingering_tablature", "Suggesting guitar and bass fingerings.", 72)
    tab_request, tab_origin = effective_tablature_request(record)
    tab_instruments = []
    for measure in measures:
        for note in measure["notes"]:
            note["tab"] = None
    for instrument_id in TAB_INSTRUMENT_ORDER:
        source_kind = tab_request[instrument_id]
        if source_kind is None:
            continue
        selected = [
            note
            for measure in measures
            for note in measure["notes"]
            if note["sourceKind"] == source_kind
        ]
        try:
            positions = assign_tablature(selected, instrument_id)
        except TablatureError as exc:
            raise ScorePipelineError("Tablature positions could not be suggested safely.") from exc
        statuses = {"assigned": 0, "out_of_range": 0, "unplayable": 0}
        for note in selected:
            note["tab"] = {"instrument": instrument_id, **positions[note["id"]]}
            statuses[note["tab"]["status"]] += 1
        instrument = TAB_INSTRUMENTS[instrument_id]
        tab_instruments.append(
            {
                "instrument": instrument_id,
                "label": instrument.label,
                "sourceKind": source_kind,
                "tuningName": instrument.tuning_name,
                "strings": list(instrument.strings),
                "frets": instrument.frets,
                "noteCount": len(selected),
                "assignedCount": statuses["assigned"],
                "outOfRangeCount": statuses["out_of_range"],
                "unplayableCount": statuses["unplayable"],
            }
        )
    fingered_count = sum(item["assignedCount"] for item in tab_instruments)
    tab_note_count = sum(item["noteCount"] for item in tab_instruments)

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

    warnings = score_evidence_warnings(
        built,
        evidence,
        single_draft_part_warning=part_data is None,
        percussion_notated=True,
    )
    notes_with_part = sum(part_note_counts.values())
    if part_data is not None:
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
    if percussion_inputs:
        warnings.append(
            "Drum notation shows broad voices on an eighth-note grid; specific kit "
            "pieces, sticking, ghost notes and accents are not claimed."
        )
    for item in tab_instruments:
        line = _SOURCE_LABELS.get(item["sourceKind"], "selected line")
        if not item["noteCount"]:
            if tab_origin == "default":
                continue
            warnings.append(
                f"{item['label']} tablature found no notes on the {line}; choose another line."
            )
            continue
        if item["instrument"] == "guitar":
            warnings.append(
                f"Guitar tablature is a fingering suggestion for the {line}; "
                "PopEx does not detect which part is played on guitar."
            )
        if item["outOfRangeCount"]:
            warnings.append(
                f"{item['outOfRangeCount']} {item['label'].lower()} note(s) are outside "
                f"the {item['tuningName']} range and have no tablature position."
            )
        if item["unplayableCount"]:
            warnings.append(
                f"{item['unplayableCount']} {item['label'].lower()} note(s) could not fit "
                "a playable chord shape and are left without a position."
            )
    if fingered_count:
        warnings.append(
            "Tablature positions are suggestions in standard tuning; held notes, "
            "techniques and alternate fingerings need review."
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
    if fingered_count:
        tablature_note = (
            f"{fingered_count} of {tab_note_count} note(s) have a suggested string and fret "
            "in standard tuning; pitches are unchanged."
        )
    elif tab_instruments and tab_note_count:
        tablature_note = "No selected note fits a playable position, so no tablature is written."
    elif tab_instruments:
        tablature_note = (
            "No notes were transcribed on the selected line"
            + (" (the separated bass stem)" if tab_origin == "default" else "")
            + "; choose a line for bass or guitar tablature, then rebuild."
        )
    else:
        tablature_note = "No line was chosen for bass or guitar tablature."
    percussion_count = evidence["percussionEventCount"]
    placed_hits = [hit for measure in measures for hit in measure["percussionHits"]]
    hit_counts = percussion_hit_counts(placed_hits)
    percussion_summary = {
        "voiceSource": voice_source,
        "voices": [
            {
                "broadVoice": voice,
                "label": PERCUSSION_NOTATION[voice][4],
                "displayStep": PERCUSSION_NOTATION[voice][0],
                "displayOctave": PERCUSSION_NOTATION[voice][1],
                "notehead": PERCUSSION_NOTATION[voice][2],
                "gmNote": PERCUSSION_NOTATION[voice][3],
                "hitCount": hit_counts["byVoice"][voice],
            }
            for voice in PERCUSSION_NOTATION
            if voice in hit_counts["byVoice"]
        ],
    }
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
                "status": "included" if placed_hits else "omitted",
                "note": percussion_note,
            },
            "tablature": {
                "status": "included" if fingered_count else "omitted",
                "note": tablature_note,
            },
        },
        "timing": timing,
        "parts": parts,
        "percussion": percussion_summary,
        "tablature": {
            "version": TABLATURE_VERSION,
            "origin": tab_origin,
            "request": tab_request,
            "instruments": tab_instruments,
        },
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
            "percussionHits": hit_counts["hits"],
            "notatedPercussionHits": hit_counts["notated"],
            "collapsedPercussionHits": hit_counts["collapsed"],
            "unresolvedPercussionHits": hit_counts["unresolved"],
            "offGridPercussionHits": hit_counts["offGrid"],
            "unplacedPercussionHits": hit_counts["unplaced"],
            "tabNotes": tab_note_count,
            "fingeredTabNotes": fingered_count,
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
    "score_outdated_reason",
]
