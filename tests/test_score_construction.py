from __future__ import annotations

import struct
import xml.etree.ElementTree as ET

import pytest

from app.score_construction import (
    SCORE_BUILDER_VERSION,
    SCORE_SCHEMA_VERSION,
    ScoreConstructionError,
    build_score_document,
    score_to_midi_bytes,
    score_to_musicxml_text,
)


def _note(note_id: str, start: float, end: float, midi: int = 69, confidence: float = 0.9) -> dict:
    return {
        "id": note_id,
        "startSeconds": start,
        "endSeconds": end,
        "midiNote": midi,
        "confidence": confidence,
    }


def test_empty_evidence_builds_single_empty_measure_with_warning() -> None:
    document = build_score_document([], tempo_bpm=120.0)
    assert document["schemaVersion"] == SCORE_SCHEMA_VERSION
    assert document["builderVersion"] == SCORE_BUILDER_VERSION
    assert document["measureCount"] == 1
    assert document["noteCount"] == 0
    assert any("empty" in warning.lower() for warning in document["warnings"])


def test_notes_split_across_measures_with_quantization() -> None:
    document = build_score_document(
        [_note("a" * 1 + "1", 0.10, 0.40, 69), _note("b2", 2.10, 2.40, 71)],
        tempo_bpm=120.0,
        beats_per_measure=4,
    )
    assert document["measureCount"] >= 2
    assert document["noteCount"] == 2
    first = document["measures"][0]["notes"][0]
    assert first["quantizedBeat"] == pytest.approx(0.0, abs=0.25)
    assert first["noteName"].startswith("A")


def test_low_confidence_and_large_shift_warn_honestly() -> None:
    document = build_score_document(
        [_note("n1", 0.13, 0.50, 60, confidence=0.10)],
        tempo_bpm=120.0,
    )
    assert any("0.50" in warning for warning in document["warnings"])


def test_duplicate_ids_and_bad_ranges_fail_closed() -> None:
    with pytest.raises(ScoreConstructionError):
        build_score_document(
            [_note("same", 0.0, 0.2), _note("same", 0.3, 0.5)], tempo_bpm=120.0
        )
    with pytest.raises(ScoreConstructionError):
        build_score_document([_note("x1", 0.5, 0.2)], tempo_bpm=120.0)
    with pytest.raises(ScoreConstructionError):
        build_score_document([_note("x2", 0.0, 0.2)], tempo_bpm=5.0)


def test_midi_bytes_have_smf_header_and_note_events() -> None:
    document = build_score_document([_note("m1", 0.0, 0.5, 69)], tempo_bpm=120.0)
    payload = score_to_midi_bytes(document)
    assert payload[:4] == b"MThd"
    assert b"MTrk" in payload
    assert bytes((0x90, 69)) in payload


def test_musicxml_parses_and_marks_draft() -> None:
    document = build_score_document(
        [_note("w1", 0.0, 0.5, 60)],
        tempo_bpm=100.0,
        chord_symbols=["Cmaj"],
    )
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    assert root.tag == "score-partwise"
    assert SCORE_BUILDER_VERSION in text
    assert root.find(".//pitch") is not None
    assert root.find(".//metronome/beat-unit") is not None
    assert root.find(".//metronome/per-minute") is not None


def test_musicxml_preserves_chord_text_without_fabricated_harmony() -> None:
    document = build_score_document(
        [_note("w2", 0.0, 0.5, 60)],
        tempo_bpm=100.0,
        chord_symbols=["Am7"],
    )
    text = score_to_musicxml_text(document)
    root = ET.fromstring(text)
    assert root.find(".//harmony") is None
    words = [node.text for node in root.iter("words")]
    assert "Am7" in words


def test_musicxml_rejects_tampered_measure_index_or_chord() -> None:
    document = build_score_document([_note("w3", 0.0, 0.5, 60)], tempo_bpm=100.0)
    tampered = dict(document)
    tampered_measures = [dict(measure) for measure in document["measures"]]
    tampered_measures[0]["measureIndex"] = 7
    tampered["measures"] = tampered_measures
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(tampered)
    tampered2 = dict(document)
    tampered_measures2 = [dict(measure) for measure in document["measures"]]
    tampered_measures2[0]["chordSymbol"] = "Bad <tag>"
    tampered2["measures"] = tampered_measures2
    with pytest.raises(ScoreConstructionError):
        score_to_musicxml_text(tampered2)
