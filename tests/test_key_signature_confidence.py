"""A key signature from the note collection when the tonal centre is uncertain (cycle 47)."""

from __future__ import annotations

import copy
import xml.etree.ElementTree as ET

import pytest

from app.score_artifacts import ScoreArtifactValidationError, validate_score_artifact
from app.score_construction import build_score_document, score_to_midi_bytes, score_to_musicxml_text
from app.tonal_context import build_tonal_context, key_signature_for, score_key_name

E_MAJOR = [64, 66, 68, 69, 71, 73, 75]  # E F# G# A B C# D#


def measures_from(pitches_per_bar: list[list[int]]) -> list[dict]:
    measures = []
    for index, pitches in enumerate(pitches_per_bar):
        notes = [
            {"midiNote": pitch, "quantizedBeat": float(index * 4 + position), "quantizedDurationBeats": 1.0,
             "confidence": 0.9, "sourceKind": "other"}
            for position, pitch in enumerate(pitches)
        ]
        measures.append({"notes": notes, "harmony": []})
    return measures


def test_the_signature_follows_the_centre_then_the_collection():
    assert key_signature_for(0.8, 0.9, 4, "ionian") == {"fifths": 4, "mode": "major"}
    assert key_signature_for(0.3, 0.9, 4, "ionian") == {"fifths": 4, "mode": "none"}
    assert key_signature_for(0.3, 0.3, 4, "ionian") is None


def test_an_e_major_collection_with_an_unclear_tonic_still_gets_four_sharps():
    # Every note of E major equally often and no bass: the notes are certain,
    # the tonic is not (E Ionian, C# Aeolian, A Lydian share them).
    bars = [[E_MAJOR[(i + j) % 7] for j in range(4)] for i in range(16)]
    tonality = build_tonal_context(measures_from(bars))
    assert tonality["primaryCandidate"]["confidence"] < 0.5
    assert tonality["signatureConfidence"] >= 0.5
    assert tonality["keySignature"] == {"fifths": 4, "mode": "none"}
    assert any("without naming a mode" in note for note in tonality["notes"])
    assert score_key_name({"tonality": tonality}) is None  # no key is named


def test_a_mixed_collection_writes_no_signature():
    # D and D# equally often: E major against E Mixolydian is unresolved.
    mixed = [64, 66, 68, 69, 71, 73, 74, 75]
    bars = [[mixed[(i + j) % 8] for j in range(4)] for i in range(16)]
    tonality = build_tonal_context(measures_from(bars))
    assert tonality["signatureConfidence"] < 0.5
    assert tonality["keySignature"] is None


def _document(key):
    document = build_score_document(
        [{"id": f"n{index}", "startSeconds": index * 0.5, "endSeconds": index * 0.5 + 0.4,
          "midiNote": pitch, "confidence": 0.9} for index, pitch in enumerate(E_MAJOR)],
        tempo_bpm=120.0,
    )
    document["keySignature"] = key
    return document


def test_musicxml_writes_the_signature_without_a_mode_and_fewer_accidentals():
    without = ET.fromstring(score_to_musicxml_text(_document(None)))
    with_key = ET.fromstring(score_to_musicxml_text(_document({"fifths": 4, "mode": "none"})))
    key = with_key.find(".//attributes/key")
    assert (key.findtext("fifths"), key.findtext("mode")) == ("4", "none")
    assert len(without.findall(".//accidental")) == 4  # F#, G#, C# and D# spelled out
    assert len(with_key.findall(".//accidental")) == 0
    midi = score_to_midi_bytes(_document({"fifths": 4, "mode": "none"}))
    assert b"\xff\x59\x02\x04\x00" in midi  # four sharps, major in MIDI terms


def _saved_document(tmp_path):
    from test_score_tablature import _load, build, client_for, create_stem_job
    from test_harmony_api import make_settings

    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    return _load(settings, job_id)


def test_saved_scores_record_the_signature_confidence_and_are_checked(tmp_path):
    document = _saved_document(tmp_path)
    assert document["schemaVersion"] == 10
    assert 0.0 <= document["tonality"]["signatureConfidence"] <= 1.0
    validate_score_artifact(document)
    tampered = copy.deepcopy(document)
    tampered["tonality"]["keySignature"] = {"fifths": 3, "mode": "none"}
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(tampered)
    older = copy.deepcopy(document)
    older["schemaVersion"] = 9
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(older)  # schema 9 has no signatureConfidence
    del older["tonality"]["signatureConfidence"]
    primary = older["tonality"]["primaryCandidate"]
    if primary is None or primary["confidence"] < 0.5:
        older["tonality"]["keySignature"] = None
    validate_score_artifact(older)
