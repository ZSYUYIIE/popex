"""Drum and percussion notation in the persisted draft score (Cycle 8)."""

from __future__ import annotations

import copy
import os
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.interpretation_pipeline import interpret_transcription_job
from app.main import create_app
from app.percussion_interpretation import (
    BROAD_PERCUSSION_VOICES,
    broad_voice_for_hit,
    broad_voice_label,
)
from app.score_artifacts import (
    ScoreArtifactValidationError,
    load_score_artifact,
    validate_score_artifact,
)
from app.score_construction import (
    PERCUSSION_MIDI_CHANNEL,
    PERCUSSION_NOTATION,
    ScoreConstructionError,
    build_score_document,
    score_to_midi_bytes,
    score_to_musicxml_text,
)
from app.score_sources import current_score_fingerprint, score_source_identity
from app.transcription_events import write_raw_transcription
from test_harmony_api import create_job, make_settings, raw_payload

GM_TABLE = {
    "low_drum": 36,
    "mid_drum": 38,
    "tom_like": 45,
    "closed_high_frequency": 42,
    "open_high_frequency": 46,
    "cymbal_like": 49,
    "unresolved_percussion": 76,
}
DISPLAY_TABLE = {
    "low_drum": ("F", 4, "normal"),
    "mid_drum": ("C", 5, "normal"),
    "tom_like": ("D", 5, "normal"),
    "closed_high_frequency": ("G", 5, "x"),
    "open_high_frequency": ("G", 5, "circle-x"),
    "cymbal_like": ("A", 5, "x"),
    "unresolved_percussion": ("B", 4, "triangle"),
}


def percussion_events() -> list[dict]:
    """Synthetic drum events at 120 BPM: one bar of kick/snare/hat plus extras."""
    return [
        {
            "id": "d_kick_1",
            "sourceKind": "drums",
            "timeSeconds": 0.0,
            "strength": 0.9,
            "hits": [
                {"kind": "kick", "confidence": 0.8},
                {"kind": "closed_hihat", "confidence": 0.6},
            ],
        },
        {
            "id": "d_hat_1",
            "sourceKind": "drums",
            "timeSeconds": 0.26,
            "strength": 0.4,
            "hits": [{"kind": "closed_hihat", "confidence": 0.55}],
        },
        {
            "id": "d_snare_1",
            "sourceKind": "drums",
            "timeSeconds": 0.5,
            "strength": 0.85,
            "hits": [{"kind": "snare", "confidence": 0.7}],
        },
        {
            "id": "d_weak_1",
            "sourceKind": "drums",
            "timeSeconds": 1.0,
            "strength": 0.3,
            "hits": [{"kind": "snare", "confidence": 0.2}],
        },
        {
            "id": "d_unknown_1",
            "sourceKind": "drums",
            "timeSeconds": 1.25,
            "strength": 0.3,
            "hits": [{"kind": "unknown_percussion", "confidence": 0.4}],
        },
        {
            "id": "d_crash_1",
            "sourceKind": "drums",
            "timeSeconds": 2.5,
            "strength": 0.95,
            "hits": [{"kind": "cymbal", "confidence": 0.65}],
        },
        {
            "id": "d_crash_2",
            "sourceKind": "drums",
            "timeSeconds": 2.53,
            "strength": 0.5,
            "hits": [{"kind": "cymbal", "confidence": 0.5}],
        },
    ]


def drum_raw_payload() -> dict:
    payload = raw_payload()
    events = percussion_events()
    payload["percussionEvents"] = events
    payload["alignmentCandidates"] = payload["alignmentCandidates"] + [
        {
            "eventId": event["id"],
            "eventType": "percussion",
            "rawTimeSeconds": event["timeSeconds"],
            "confidence": 0.0,
        }
        for event in events
    ]
    return payload


def create_drum_job(settings) -> str:
    job_id = create_job(settings)
    write_raw_transcription(job_id, settings, drum_raw_payload())
    db.update_job(
        settings.database_path, job_id, percussion_event_count=len(percussion_events())
    )
    return job_id


def publish_real_interpretation(settings, job_id: str) -> dict:
    result = interpret_transcription_job(job_id, settings)
    db.update_job(
        settings.database_path,
        job_id,
        interpretation_status="completed",
        interpretation_stage="completed",
        interpretation_progress=100,
        interpretation_version=result.version,
        interpretation_artifact_file_name=result.draft_file_name,
        interpreted_at=result.created_at,
        interpretation_part_count=result.part_count,
        interpretation_phrase_count=result.phrase_count,
        interpretation_pitched_item_count=result.pitched_item_count,
        interpretation_percussion_item_count=result.percussion_item_count,
        interpretation_warning_count=result.warning_count,
    )
    return result.payload


def build_and_load(settings, job_id: str, *, force: bool = False) -> tuple[dict, dict]:
    with TestClient(create_app(settings)) as client:
        suffix = "?force=true" if force else ""
        response = client.post(f"/api/jobs/{job_id}/score/construct{suffix}")
        assert response.status_code == 202, response.text
        job = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)
        assert job["score"]["status"] == "completed", job["score"]
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
    return job, details


def all_hits(details: dict) -> list[dict]:
    return [hit for measure in details["measures"] for hit in measure["percussionHits"]]


def hit_input(event_id: str, voice: str, seconds: float, **extra) -> dict:
    value = {
        "eventId": event_id,
        "hitIndex": 0,
        "sourceKind": "drums",
        "rawKind": "kick",
        "broadVoice": voice,
        "resolved": voice != "unresolved_percussion",
        "timeSeconds": seconds,
        "strength": 0.5,
        "confidence": 0.7,
    }
    value.update(extra)
    return value


# ---------------------------------------------------------------------------
# Documented tables


def test_notation_tables_cover_exactly_the_broad_voices() -> None:
    assert set(PERCUSSION_NOTATION) == set(BROAD_PERCUSSION_VOICES)
    for voice, (step, octave, notehead, gm_note, label) in PERCUSSION_NOTATION.items():
        assert (step, octave, notehead) == DISPLAY_TABLE[voice]
        assert gm_note == GM_TABLE[voice]
        assert label == broad_voice_label(voice)
    assert PERCUSSION_MIDI_CHANNEL == 9  # General MIDI channel 10, zero-based


def test_raw_hit_kind_table_keeps_low_confidence_and_unknown_hits_unresolved() -> None:
    assert broad_voice_for_hit("kick", 0.8) == ("low_drum", True)
    assert broad_voice_for_hit("closed_hihat", 0.6) == ("closed_high_frequency", True)
    assert broad_voice_for_hit("snare", 0.2) == ("unresolved_percussion", False)
    assert broad_voice_for_hit("unknown_percussion", 0.9) == ("unresolved_percussion", False)
    assert broad_voice_for_hit("cowbell", 0.9) == ("unresolved_percussion", False)


# ---------------------------------------------------------------------------
# Builder and exporters


def test_builder_keeps_percussion_separate_on_the_same_grid() -> None:
    document = build_score_document(
        [{"id": "n1", "startSeconds": 0.0, "endSeconds": 0.5, "midiNote": 60, "confidence": 0.9}],
        tempo_bpm=120.0,
        percussion_hits=[
            hit_input("k1", "low_drum", 0.02),
            hit_input("u1", "unresolved_percussion", 0.74, rawKind="unknown_percussion"),
            hit_input("k2", "low_drum", 2.6),  # beyond the last pitched note
        ],
    )
    assert document["measureCount"] == 2
    assert document["percussionHitCount"] == 3
    first = document["measures"][0]
    assert [note["id"] for note in first["notes"]] == ["n1"]
    assert all("broadVoice" not in note for note in first["notes"])
    kick = first["percussionHits"][0]
    assert kick["quantizedBeat"] == 0.0 and kick["rawTimeSeconds"] == 0.02
    assert kick["quantizationShiftSeconds"] == 0.02
    assert first["percussionHits"][1]["quantizedBeat"] == 1.5
    assert document["measures"][1]["percussionHits"][0]["quantizedBeat"] == 5.0
    assert any("unresolved" in warning for warning in document["warnings"])


def test_builder_without_percussion_is_unchanged() -> None:
    document = build_score_document([], tempo_bpm=100.0)
    assert "percussionHitCount" not in document
    assert all("percussionHits" not in measure for measure in document["measures"])


def test_same_voice_duplicates_collapse_but_stay_as_evidence() -> None:
    document = build_score_document(
        [],
        tempo_bpm=120.0,
        percussion_hits=[
            hit_input("a", "cymbal_like", 0.50, confidence=0.5),
            hit_input("b", "cymbal_like", 0.53, confidence=0.9),
            hit_input("c", "mid_drum", 0.51),
        ],
    )
    hits = document["measures"][0]["percussionHits"]
    states = {hit["eventId"]: hit["notation"] for hit in hits}
    assert states == {"a": "collapsed", "b": "notated", "c": "notated"}
    assert any("share a grid slot" in warning for warning in document["warnings"])


@pytest.mark.parametrize(
    "change",
    [
        {"broadVoice": "cowbell"},
        {"broadVoice": "unresolved_percussion", "resolved": True},
        {"broadVoice": "low_drum", "resolved": False},
        {"interpretationPlacement": "maybe"},
        {"unexpected": 1},
    ],
)
def test_builder_rejects_invalid_percussion_hits(change: dict) -> None:
    hit = hit_input("k", "low_drum", 0.0)
    hit.update(change)
    with pytest.raises(ScoreConstructionError):
        build_score_document([], tempo_bpm=120.0, percussion_hits=[hit])


def test_builder_rejects_duplicate_hit_identity() -> None:
    with pytest.raises(ScoreConstructionError):
        build_score_document(
            [],
            tempo_bpm=120.0,
            percussion_hits=[hit_input("k", "low_drum", 0.0), hit_input("k", "low_drum", 1.0)],
        )


def _midi_channel_events(data: bytes) -> list[tuple[int, int, int, int]]:
    """Return ``(tick, status, key, velocity)`` for channel voice messages."""
    assert data[:4] == b"MThd"
    length = struct.unpack(">I", data[18:22])[0]
    track = data[22 : 22 + length]
    position, tick, events = 0, 0, []
    while position < len(track):
        delta = 0
        while True:
            byte = track[position]
            position += 1
            delta = (delta << 7) | (byte & 0x7F)
            if not byte & 0x80:
                break
        tick += delta
        status = track[position]
        if status == 0xFF:
            meta_length = track[position + 2]
            position += 3 + meta_length
            continue
        events.append((tick, status, track[position + 1], track[position + 2]))
        position += 3
    return events


def test_midi_puts_percussion_on_channel_ten_with_documented_notes() -> None:
    voices = list(PERCUSSION_NOTATION)
    hits = [
        hit_input(f"h{index}", voice, index * 0.25, strength=1.0 if index == 0 else 0.0)
        for index, voice in enumerate(voices)
    ]
    document = build_score_document(
        [{"id": "n1", "startSeconds": 0.0, "endSeconds": 1.0, "midiNote": 64, "confidence": 0.9}],
        tempo_bpm=120.0,
        percussion_hits=hits,
    )
    events = _midi_channel_events(score_to_midi_bytes(document))
    drum_on = [event for event in events if event[1] == 0x99]
    pitched_on = [event for event in events if event[1] & 0xF0 == 0x90 and event[1] != 0x99]
    assert [key for _, _, key, _ in drum_on] == [GM_TABLE[voice] for voice in voices]
    assert drum_on[0][3] == 120 and drum_on[1][3] == 40  # velocity follows strength
    assert pitched_on == [(0, 0x90, 64, 80)]  # pitched channel policy unchanged
    assert sum(1 for event in events if event[1] == 0x89) == len(voices)


def test_midi_skips_collapsed_duplicates() -> None:
    document = build_score_document(
        [],
        tempo_bpm=120.0,
        percussion_hits=[hit_input("a", "low_drum", 0.0), hit_input("b", "low_drum", 0.01)],
    )
    events = _midi_channel_events(score_to_midi_bytes(document))
    assert len([event for event in events if event[1] == 0x99]) == 1


def _musicxml_parts(text: str) -> dict[str, ET.Element]:
    root = ET.fromstring(text)
    return {part.get("id"): part for part in root.findall("part")}


def test_musicxml_writes_unpitched_percussion_part_with_documented_display() -> None:
    voices = list(PERCUSSION_NOTATION)
    document = build_score_document(
        [],
        tempo_bpm=120.0,
        percussion_hits=[
            hit_input(f"h{index}", voice, index * 0.5) for index, voice in enumerate(voices)
        ],
    )
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    score_parts = root.findall("part-list/score-part")
    assert [part.get("id") for part in score_parts] == ["P1", "P2"]
    drum_part = score_parts[1]
    channels = {item.findtext("midi-channel") for item in drum_part.findall("midi-instrument")}
    assert channels == {"10"}
    unpitched = {
        item.get("id"): int(item.findtext("midi-unpitched"))
        for item in drum_part.findall("midi-instrument")
    }
    assert unpitched == {f"P2-I{GM_TABLE[voice]}": GM_TABLE[voice] + 1 for voice in voices}
    names = [item.findtext("instrument-name") for item in drum_part.findall("score-instrument")]
    assert "Unresolved percussion (review)" in names

    part = _musicxml_parts(text)["P2"]
    assert part.find("measure/attributes/clef/sign").text == "percussion"
    notes = part.findall(".//note")
    assert notes and all(note.find("pitch") is None for note in notes)
    by_instrument = {}
    for note in notes:
        instrument = note.find("instrument").get("id")
        by_instrument[instrument] = (
            note.findtext("unpitched/display-step"),
            int(note.findtext("unpitched/display-octave")),
            note.findtext("notehead") or "normal",
        )
    assert by_instrument == {
        f"P2-I{GM_TABLE[voice]}": DISPLAY_TABLE[voice] for voice in voices
    }
    words = [item.text for item in part.iter("words")]
    assert any("unresolved" in (text or "") for text in words)
    # The pitched part keeps its G clef and never uses unpitched notes.
    pitched = _musicxml_parts(text)["P1"]
    assert pitched.find("measure/attributes/clef/sign").text == "G"


def test_musicxml_simultaneous_hits_form_chords_and_low_drum_uses_second_voice() -> None:
    document = build_score_document(
        [],
        tempo_bpm=120.0,
        percussion_hits=[
            hit_input("k", "low_drum", 0.0),
            hit_input("s", "mid_drum", 0.0),
            hit_input("h", "closed_high_frequency", 0.0),
        ],
    )
    part = _musicxml_parts(score_to_musicxml_text(document))["P2"]
    notes = part.findall("measure/note")
    hands = [note for note in notes if note.findtext("voice") == "1"]
    feet = [note for note in notes if note.findtext("voice") == "2"]
    assert len(hands) == 2 and hands[0].find("chord") is None
    assert hands[1].find("chord") is not None
    assert [note.findtext("stem") for note in hands] == ["up", "up"]
    assert len(feet) == 1 and feet[0].findtext("stem") == "down"
    assert part.find("measure/backup") is not None


def test_musicxml_without_hits_has_no_percussion_part() -> None:
    document = build_score_document([], tempo_bpm=120.0, percussion_hits=[])
    assert list(_musicxml_parts(score_to_musicxml_text(document))) == ["P1"]


def test_exporters_reject_two_notated_hits_in_one_slot() -> None:
    document = build_score_document(
        [],
        tempo_bpm=120.0,
        percussion_hits=[hit_input("a", "low_drum", 0.0), hit_input("b", "low_drum", 0.01)],
    )
    for hit in document["measures"][0]["percussionHits"]:
        hit["notation"] = "notated"
    with pytest.raises(ScoreConstructionError):
        score_to_midi_bytes(document)
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(document)


_XSD = os.environ.get("POPEX_MUSICXML_XSD")


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_generated_percussion_examples_validate_against_musicxml_schema() -> None:
    from lxml import etree  # required whenever the schema path is configured

    schema = etree.XMLSchema(etree.parse(_XSD))
    examples = [
        build_score_document([], tempo_bpm=120.0),
        build_score_document(
            [
                {"id": "a", "startSeconds": 0.0, "endSeconds": 3.1, "midiNote": 61, "confidence": 0.9},
                {"id": "b", "startSeconds": 0.0, "endSeconds": 0.5, "midiNote": 66, "confidence": 0.4},
                {"id": "c", "startSeconds": 0.2, "endSeconds": 0.7, "midiNote": 66, "confidence": 0.8},
            ],
            tempo_bpm=120.0,
        ),
        build_score_document(
            [{"id": "n", "startSeconds": 0.0, "endSeconds": 0.9, "midiNote": 61, "confidence": 0.9}],
            tempo_bpm=120.0,
            percussion_hits=[
                hit_input(f"h{index}", voice, index * 0.37)
                for index, voice in enumerate(PERCUSSION_NOTATION)
            ],
        ),
        build_score_document(
            [],
            tempo_bpm=90.0,
            beats_per_measure=3,
            percussion_hits=[
                hit_input("k", "low_drum", 0.0),
                hit_input("s", "mid_drum", 0.0),
                hit_input("u", "unresolved_percussion", 4.1, rawKind="unknown_percussion"),
            ],
        ),
    ]
    for document in examples:
        tree = etree.fromstring(score_to_musicxml_text(document).encode("utf-8"))
        assert schema.validate(tree), schema.error_log


# ---------------------------------------------------------------------------
# Persisted score, freshness and review payloads


def test_score_without_interpretation_maps_raw_hit_kinds(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    job, details = build_and_load(settings, job_id)

    assert details["version"] == "score-pipeline-v4"
    assert details["layers"]["percussion"]["status"] == "included"
    assert "raw hit-kind table" in details["layers"]["percussion"]["note"]
    assert details["percussion"]["voiceSource"] == "raw-hit-kinds"
    hits = all_hits(details)
    assert len(hits) == 8  # every raw hit is kept, including the duplicate crash
    by_id = {(hit["eventId"], hit["hitIndex"]): hit for hit in hits}
    assert by_id[("d_kick_1", 0)]["broadVoice"] == "low_drum"
    assert by_id[("d_kick_1", 1)]["broadVoice"] == "closed_high_frequency"
    weak = by_id[("d_weak_1", 0)]
    assert weak["broadVoice"] == "unresolved_percussion" and weak["resolved"] is False
    assert weak["rawKind"] == "snare"  # raw evidence kept, never assigned to a drum
    assert by_id[("d_unknown_1", 0)]["broadVoice"] == "unresolved_percussion"
    assert by_id[("d_hat_1", 0)]["rawTimeSeconds"] == 0.26
    assert by_id[("d_hat_1", 0)]["strength"] == 0.4
    assert by_id[("d_crash_2", 0)]["notation"] == "collapsed"
    assert all(hit["interpretationPlacement"] is None for hit in hits)

    counts = details["counts"]
    assert counts["percussionEvents"] == 7
    assert counts["percussionHits"] == 8
    assert counts["unresolvedPercussionHits"] == 2
    assert counts["collapsedPercussionHits"] == 1
    assert counts["notatedPercussionHits"] == 7
    assert {voice["broadVoice"]: voice["hitCount"] for voice in details["percussion"]["voices"]} == {
        "low_drum": 1,
        "mid_drum": 1,
        "closed_high_frequency": 2,
        "cymbal_like": 2,
        "unresolved_percussion": 2,
    }
    assert not any("not rendered" in warning for warning in details["warnings"])
    assert details["layers"]["tablature"]["status"] == "omitted"
    assert not any("tablature" in warning.lower() for warning in details["warnings"])
    assert job["score"]["stale"] is False


def test_score_uses_matching_interpretation_voices(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    draft = publish_real_interpretation(settings, job_id)
    assert draft["percussionItems"]
    _, details = build_and_load(settings, job_id)

    assert details["percussion"]["voiceSource"] == "interpretation"
    assert "editable interpretation" in details["layers"]["percussion"]["note"]
    expected = {
        (item["sourceEventIds"][0], hit["sourceHitIndex"]): (
            hit["broadVoice"],
            item["placementStatus"],
        )
        for item in draft["percussionItems"]
        for hit in item["hits"]
    }
    actual = {
        (hit["eventId"], hit["hitIndex"]): (hit["broadVoice"], hit["interpretationPlacement"])
        for hit in all_hits(details)
    }
    assert actual == expected


def test_mismatched_interpretation_is_omitted_not_combined(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    publish_real_interpretation(settings, job_id)
    # Change the raw percussion after interpretation; IDs stay, timing moves.
    changed = drum_raw_payload()
    changed["percussionEvents"][2]["timeSeconds"] = 0.75
    for candidate in changed["alignmentCandidates"]:
        if candidate["eventId"] == "d_snare_1":
            candidate["rawTimeSeconds"] = 0.75
    write_raw_transcription(job_id, settings, changed)
    _, details = build_and_load(settings, job_id)

    assert details["percussion"]["voiceSource"] == "raw-hit-kinds"
    assert "does not match" in details["layers"]["percussion"]["note"]
    snare = next(hit for hit in all_hits(details) if hit["eventId"] == "d_snare_1")
    assert snare["rawTimeSeconds"] == 0.75 and snare["interpretationPlacement"] is None


def test_saved_exports_contain_the_percussion_part(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    build_and_load(settings, job_id)
    with TestClient(create_app(settings)) as client:
        base = f"/api/jobs/{job_id}/score/saved/download?format="
        musicxml = client.get(base + "musicxml")
        midi = client.get(base + "midi")
    assert musicxml.status_code == 200 and midi.status_code == 200
    parts = _musicxml_parts(musicxml.text)
    assert set(parts) == {"P1", "P2"}
    events = _midi_channel_events(midi.content)
    assert len([event for event in events if event[1] == 0x99]) == 7


def test_score_without_percussion_events_omits_the_layer(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    _, details = build_and_load(settings, job_id)
    assert details["layers"]["percussion"]["status"] == "omitted"
    assert details["percussion"] == {"voiceSource": "none", "voices": []}
    assert details["counts"]["percussionHits"] == 0
    assert all(measure["percussionHits"] == [] for measure in details["measures"])


def test_percussion_evidence_is_covered_by_the_source_fingerprint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    record = db.get_job(settings.database_path, job_id)
    identity = score_source_identity(record)
    assert identity["transcription"]["percussionEventCount"] == 7
    before = current_score_fingerprint(record)
    publish_real_interpretation(settings, job_id)
    assert current_score_fingerprint(db.get_job(settings.database_path, job_id)) != before


def _downgrade_to_schema_one(document: dict) -> dict:
    legacy = copy.deepcopy(document)
    legacy["schemaVersion"] = 1
    legacy["pipelineVersion"] = "score-pipeline-v1"
    legacy["builderVersion"] = "score-construction-v1"
    del legacy["percussion"]
    del legacy["tablature"]
    del legacy["tonality"]
    del legacy["counts"]["tabNotes"]
    del legacy["counts"]["fingeredTabNotes"]
    for key in (
        "percussionHits",
        "notatedPercussionHits",
        "collapsedPercussionHits",
        "unresolvedPercussionHits",
        "offGridPercussionHits",
        "unplacedPercussionHits",
    ):
        del legacy["counts"][key]
    for measure in legacy["measures"]:
        del measure["percussionHits"]
        for note in measure["notes"]:
            del note["tab"]
    legacy["layers"]["tablature"] = {
        "status": "omitted",
        "note": "Guitar and bass tablature are not generated yet.",
    }
    legacy["layers"]["percussion"] = {
        "status": "omitted",
        "note": "Raw percussion events are preserved but not notated yet.",
    }
    return legacy


def test_schema_one_scores_stay_readable_and_are_reported_out_of_date(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    build_and_load(settings, job_id)
    record = db.get_job(settings.database_path, job_id)
    current = load_score_artifact(
        job_id, settings, artifact_file_name=record["score_artifact_file_name"]
    )
    legacy = _downgrade_to_schema_one(current)
    assert validate_score_artifact(legacy)["schemaVersion"] == 1

    from app.score_artifacts import write_score_artifact

    pointer = "score/score-document." + "e" * 32 + ".json"
    write_score_artifact(job_id, settings, legacy, artifact_file_name=pointer)
    db.update_job(
        settings.database_path,
        job_id,
        score_artifact_file_name=pointer,
        score_version="score-pipeline-v1",
        scored_at=legacy["createdAt"],
        score_warning_count=len(legacy["warnings"]),
    )
    with TestClient(create_app(settings)) as client:
        job = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert job["score"]["stale"] is True and job["score"]["canRebuild"] is True
    assert details["stale"] is True
    assert details["percussion"] is None
    assert any("before drum notation" in warning for warning in details["warnings"])
    assert musicxml.status_code == 200
    assert list(_musicxml_parts(musicxml.text)) == ["P1"]


def test_schema_one_score_without_percussion_events_is_not_stale(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    build_and_load(settings, job_id)
    db.update_job(settings.database_path, job_id, score_version="score-pipeline-v1")
    from app.score_pipeline import score_outdated_reason

    assert score_outdated_reason(db.get_job(settings.database_path, job_id)) != "drum-notation"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["counts"].update(unresolvedPercussionHits=0),
        lambda doc: doc["percussion"]["voices"][0].update(gmNote=35),
        lambda doc: doc["percussion"].update(voiceSource="none"),
        lambda doc: doc["layers"]["percussion"].update(status="omitted"),
        lambda doc: doc["measures"][0]["percussionHits"][0].update(resolved=False),
        lambda doc: doc["measures"][0]["percussionHits"][0].update(quantizedBeat=0.25),
        lambda doc: doc["measures"][0]["percussionHits"][0].update(broadVoice="kick"),
        lambda doc: doc["measures"][0]["percussionHits"].append(
            copy.deepcopy(doc["measures"][0]["percussionHits"][0])
        ),
        lambda doc: doc.update(schemaVersion=5),
    ],
)
def test_tampered_percussion_documents_are_rejected(tmp_path: Path, mutate) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    build_and_load(settings, job_id)
    record = db.get_job(settings.database_path, job_id)
    document = load_score_artifact(
        job_id, settings, artifact_file_name=record["score_artifact_file_name"]
    )
    assert document["measures"][0]["percussionHits"][0]["resolved"] is True
    mutate(document)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(document)


def test_schema_one_document_cannot_claim_percussion(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    build_and_load(settings, job_id)
    record = db.get_job(settings.database_path, job_id)
    legacy = _downgrade_to_schema_one(
        load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])
    )
    legacy["layers"]["percussion"]["status"] = "included"
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(legacy)
