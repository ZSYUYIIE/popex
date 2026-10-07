"""Modal tonal context, key signatures and key-aware spelling (Cycle 12)."""

from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.score_artifacts import (
    ScoreArtifactValidationError,
    load_score_artifact,
    validate_score_artifact,
)
from app.score_construction import build_score_document, score_to_midi_bytes, score_to_musicxml_text
from app.score_pipeline import score_outdated_reason
from app.tonal_context import COLLECTIONS, build_tonal_context, key_signature, root_name
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings
from test_score_tablature import create_stem_job


def bars(pitches: list[int], bass: list[int] | None = None, per_bar: int = 4) -> list[dict]:
    """Quarter-note bars of ``pitches`` with an optional whole-bar bass line."""
    measures, beat = [], 0.0
    for start in range(0, len(pitches), per_bar):
        notes = [
            {"id": f"n{start + j}", "midiNote": midi, "quantizedBeat": beat + j,
             "quantizedDurationBeats": 1.0, "confidence": 0.9, "sourceKind": "full_mix"}
            for j, midi in enumerate(pitches[start : start + per_bar])
        ]
        if bass is not None:
            notes.append(
                {"id": f"b{start}", "midiNote": bass[start // per_bar % len(bass)], "quantizedBeat": beat,
                 "quantizedDurationBeats": 4.0, "confidence": 0.9, "sourceKind": "bass"}
            )
        measures.append({"measureIndex": len(measures), "notes": notes})
        beat += per_bar
    return measures


@pytest.mark.parametrize(
    "pitches,bass,expected",
    [
        ([60, 62, 64, 65, 67, 69, 71, 72], [36, 43, 41, 36], "C Ionian (major)"),
        ([62, 64, 65, 67, 69, 71, 72, 74], [38, 38, 43, 38], "D Dorian"),
        ([64, 65, 67, 69, 71, 72, 74, 76], [40, 41, 40, 40], "E Phrygian"),
        ([65, 67, 69, 71, 72, 74, 76, 77], [41, 43, 41, 41], "F Lydian"),
        ([67, 69, 71, 72, 74, 76, 77, 79], [43, 43, 41, 43], "G Mixolydian"),
        ([57, 59, 60, 62, 64, 65, 67, 69], [45, 41, 43, 45], "A Aeolian (natural minor)"),
        ([57, 59, 60, 62, 64, 65, 68, 69], [45, 40, 45, 45], "A Harmonic minor"),
        ([57, 60, 62, 64, 67, 69, 72, 74], [45], "A Minor pentatonic"),
        ([57, 60, 62, 63, 64, 67], [45], "A Blues"),
        ([67, 69, 71, 74, 76, 79, 81, 83], [43], "G Major pentatonic"),
    ],
)
def test_collections_are_identified_from_notes_and_bass(pitches, bass, expected) -> None:
    result = build_tonal_context(bars(pitches * 4, bass))
    assert result["primaryCandidate"]["displayName"] == expected
    assert result["primaryCandidate"]["confidence"] >= 0.5
    assert result["keySignature"] is not None
    assert len(result["candidates"]) == 8


def test_modulation_is_reported_as_local_regions_without_forcing_one_key() -> None:
    c_major = [60, 62, 64, 65, 67, 69, 71, 72] * 4
    d_flat = [61, 63, 65, 66, 68, 70, 72, 73] * 4
    measures = bars(c_major + d_flat, [36] * 8 + [37] * 8)
    result = build_tonal_context(measures)
    assert result["primaryCandidate"]["confidence"] < 0.5
    assert result["keySignature"] is None  # not forced into one scale
    regions = [(r["startMeasure"], r["endMeasure"], r["displayName"]) for r in result["localRegions"]]
    assert regions[0] == (0, 7, "C Ionian (major)")
    assert regions[-1] == (8, 15, "Db Ionian (major)")
    assert any(r["differsFromWhole"] for r in result["localRegions"])
    assert any("modulation" in note for note in result["notes"])
    assert 0 < result["chromaticismScore"] < 1


def test_relative_modes_without_tonic_evidence_are_ambiguous() -> None:
    # The white notes with no bass and an even distribution: C Ionian vs relatives.
    result = build_tonal_context(bars([60, 62, 64, 65, 67, 69, 71] * 4))
    assert result["ambiguousWith"]
    assert result["primaryCandidate"]["confidence"] < 0.5
    assert any("shares its notes" in note for note in result["notes"])


def test_no_notes_means_no_suggestion() -> None:
    result = build_tonal_context([{"measureIndex": 0, "notes": []}], chroma_mean=[0.1] * 12)
    assert result["primaryCandidate"] is None and result["keySignature"] is None
    assert result["candidates"] == []


def test_key_signatures_follow_the_diatonic_parent() -> None:
    assert key_signature(2, "dorian") == {"fifths": 0, "mode": "dorian"}
    assert key_signature(9, "aeolian") == {"fifths": 0, "mode": "minor"}
    assert key_signature(5, "ionian") == {"fifths": -1, "mode": "major"}
    assert key_signature(7, "mixolydian") == {"fifths": 0, "mode": "mixolydian"}
    assert key_signature(4, "harmonic_minor") == {"fifths": 1, "mode": "minor"}
    assert key_signature(10, "ionian") == {"fifths": -2, "mode": "major"}
    assert root_name(10, "ionian") == "Bb" and root_name(6, "ionian") == "F#"
    assert {"whole_tone"}.isdisjoint(COLLECTIONS)  # later collections are additive


def _export(pitches: list[int], key) -> tuple[ET.Element, bytes]:
    document = build_score_document(
        [{"id": f"n{i}", "startSeconds": i * 0.5, "endSeconds": i * 0.5 + 0.5, "midiNote": m, "confidence": 0.9}
         for i, m in enumerate(pitches)],
        tempo_bpm=120.0,
    )
    document["keySignature"] = key
    return ET.fromstring(score_to_musicxml_text(document)), score_to_midi_bytes(document)


def test_flat_keys_spell_with_flats_and_omit_key_accidentals() -> None:
    root, midi = _export([70, 71, 63, 66], {"fifths": -2, "mode": "major"})  # Bb, B, Eb, Gb
    key = root.find("part/measure/attributes/key")
    assert (key.findtext("fifths"), key.findtext("mode")) == ("-2", "major")
    notes = root.findall("part[@id='P1']/measure/note")
    spelled = [(n.findtext("pitch/step"), n.findtext("pitch/alter"), n.findtext("accidental")) for n in notes]
    assert spelled == [("B", "-1", None), ("B", None, "natural"), ("E", "-1", None), ("G", "-1", "flat")]
    assert b"\xff\x59\x02\xfe\x00" in midi  # two flats, major


def test_without_a_key_signature_output_is_unchanged() -> None:
    root, midi = _export([61], None)
    assert root.find("part/measure/attributes/key") is None
    note = root.find("part[@id='P1']/measure/note")
    assert (note.findtext("pitch/step"), note.findtext("pitch/alter"), note.findtext("accidental")) == ("C", "1", "sharp")
    assert b"\xff\x59" not in midi


def test_modal_key_uses_parent_signature_in_midi() -> None:
    _root, midi = _export([62], {"fifths": 0, "mode": "dorian"})
    assert b"\xff\x59\x02\x00\x00" in midi
    _root, midi = _export([57], {"fifths": 0, "mode": "minor"})
    assert b"\xff\x59\x02\x00\x01" in midi


def test_score_documents_carry_tonal_context_and_key(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        musicxml = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
        comparison_job = client.get(f"/api/jobs/{job_id}").json()
    tonality = details["tonality"]
    assert tonality["version"] == "modal-collections-v1"
    assert tonality["evidence"]["bassSource"] == "bass-stem line"
    assert tonality["primaryCandidate"] == tonality["candidates"][0]
    assert any("Tonal context suggests" in warning for warning in details["warnings"])
    key = ET.fromstring(musicxml.content).find("part/measure/attributes/key")
    assert (key is not None) == (tonality["keySignature"] is not None)
    assert comparison_job["score"]["stale"] is False


def _tonality_document(tmp_path: Path) -> dict:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with TestClient(create_app(settings)) as client:
        client.post(f"/api/jobs/{job_id}/score/construct")
    record = db.get_job(settings.database_path, job_id)
    return load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: doc["tonality"].update(keySignature={"fifths": 3, "mode": "major"}),
        lambda doc: doc["tonality"]["candidates"].reverse(),
        lambda doc: doc["tonality"]["primaryCandidate"].update(collection="hirajoshi"),
        lambda doc: doc["tonality"].update(chromaticismScore=1.5),
        lambda doc: doc["tonality"]["localRegions"].append(
            {"startMeasure": 99, "endMeasure": 99, "tonalCenter": "C", "rootPitchClass": 0,
             "collection": "ionian", "displayName": "C Ionian (major)", "confidence": 0.5,
             "differsFromWhole": True}
        ),
        lambda doc: doc.pop("tonality"),
    ],
)
def test_tampered_tonality_is_rejected(tmp_path: Path, mutate) -> None:
    document = _tonality_document(tmp_path)
    validate_score_artifact(document)
    broken = copy.deepcopy(document)
    mutate(broken)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(broken)


def test_scores_with_notes_from_before_tonal_context_are_out_of_date(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        client.post(f"/api/jobs/{job_id}/score/construct")
        db.update_job(settings.database_path, job_id, score_version="score-pipeline-v3")
        summary = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)["score"]
    assert summary["stale"] is True and summary["staleReason"] == "tonal-context"
    db.update_job(settings.database_path, job_id, score_note_count=0)
    assert score_outdated_reason(db.get_job(settings.database_path, job_id)) is None


def test_tonal_context_panel_renders_candidates_regions_and_notes() -> None:
    detail = {
        "available": True, "createdAt": SUMMARY["createdAt"], "layers": {}, "counts": {}, "warnings": [],
        "parts": [],
        "tonality": {
            "primaryCandidate": {"displayName": "D Dorian", "confidence": 0.72},
            "candidates": [{"displayName": "D Dorian", "confidence": 0.72},
                           {"displayName": "A Minor pentatonic", "confidence": 0.31}],
            "localRegions": [{"startMeasure": 8, "endMeasure": 15, "displayName": "G Mixolydian",
                              "confidence": 0.6, "differsFromWhole": True}],
            "chromaticismScore": 0.08, "keySignature": {"fifths": 0, "mode": "dorian"},
            "notes": ["Some bars suggest a different centre or collection."],
        },
    }
    result = _run_node(
        f"""
t.setDetail("done", {json.dumps(detail)});
const html=t.renderScore({{id:"done",score:{json.dumps(SUMMARY)}}});
const stale=t.renderScore({{id:"s",score:{json.dumps({**SUMMARY, "stale": True, "staleReason": "tonal-context"})}}});
console.log(JSON.stringify({{html,stale}}));
"""
    )
    html = result["html"]
    assert "<strong>Tonal context</strong>" in html
    assert "Most likely: <strong>D Dorian</strong> <span class=\"detail-note\">(72% confidence)</span> · key signature written" in html
    assert "Other candidates: A Minor pentatonic (31%)" in html
    assert "Bars 9–16: G Mixolydian" in html and "differs from the whole score" in html
    assert "8% of note time falls outside the suggested collection." in html
    assert "built before mode and key suggestions were available" in result["stale"]


_XSD = os.environ.get("POPEX_MUSICXML_XSD")


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_key_signature_examples_validate_against_musicxml_schema() -> None:
    from lxml import etree  # required whenever the schema path is configured

    schema = etree.XMLSchema(etree.parse(_XSD))
    for key in ({"fifths": -3, "mode": "minor"}, {"fifths": 2, "mode": "dorian"}, {"fifths": 0, "mode": "locrian"}):
        root, _midi = _export([60, 61, 63, 66, 70, 71], key)
        tree = etree.fromstring(ET.tostring(root))
        assert schema.validate(tree), schema.error_log
