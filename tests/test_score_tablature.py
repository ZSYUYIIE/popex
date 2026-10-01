"""Guitar and bass tablature in the persisted draft score (Cycle 9)."""

from __future__ import annotations

import copy
import json
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
    write_score_artifact,
)
from app.score_sources import current_score_fingerprint
from app.tablature import tab_position_is_consistent
from app.transcription_events import write_raw_transcription
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings, raw_payload


def bass_events() -> list[dict]:
    pitches = [28, 33, 35, 24, 40, 43]  # E1 A1 B1 C1(out of range) E2 G2
    return [
        {
            "id": f"b_{index}",
            "sourceKind": "bass",
            "startSeconds": index * 0.5,
            "endSeconds": index * 0.5 + 0.45,
            "midiNote": midi,
            "midiPitch": float(midi),
            "frequencyHz": 440.0 * 2 ** ((midi - 69) / 12),
            "noteName": "X",
            "confidence": 0.85,
            "warnings": [],
        }
        for index, midi in enumerate(pitches)
    ]


def create_stem_job(settings) -> str:
    job_id = create_job(settings)
    payload = raw_payload()
    events = payload["pitchedNoteEvents"] + bass_events()
    payload["pitchedNoteEvents"] = events
    payload["alignmentCandidates"] = [
        {"eventId": event["id"], "eventType": "pitched", "rawTimeSeconds": event["startSeconds"], "confidence": 0.0}
        for event in events
    ]
    write_raw_transcription(job_id, settings, payload)
    db.update_job(settings.database_path, job_id, pitched_event_count=len(events))
    return job_id


def client_for(settings) -> TestClient:
    return TestClient(create_app(settings))


def job_score(client: TestClient, job_id: str) -> dict:
    return next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)["score"]


def build(client: TestClient, job_id: str, *, force: bool = False) -> dict:
    response = client.post(f"/api/jobs/{job_id}/score/construct{'?force=true' if force else ''}")
    assert response.status_code == 202, response.text
    assert job_score(client, job_id)["status"] == "completed"
    return client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()


def notes_of(details: dict) -> list[dict]:
    return [note for measure in details["measures"] for note in measure["notes"]]


def musicxml(client: TestClient, job_id: str) -> ET.Element:
    response = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert response.status_code == 200
    return ET.fromstring(response.content)


def test_default_tabs_only_the_bass_stem_line(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        details = build(client, job_id)
        root = musicxml(client, job_id)

    assert details["version"] == "score-pipeline-v3"
    tablature = details["tablature"]
    assert tablature["origin"] == "default"
    assert tablature["request"] == {"bass": "bass", "guitar": None}
    assert [item["instrument"] for item in tablature["instruments"]] == ["bass"]
    bass = tablature["instruments"][0]
    assert (bass["noteCount"], bass["assignedCount"], bass["outOfRangeCount"]) == (6, 5, 1)
    assert details["layers"]["tablature"]["status"] == "included"
    assert details["counts"]["tabNotes"] == 6 and details["counts"]["fingeredTabNotes"] == 5

    for note in notes_of(details):
        if note["sourceKind"] == "bass":
            tab = note["tab"]
            assert tab["instrument"] == "bass"
            if note["midiNote"] == 24:
                assert tab == {"instrument": "bass", "status": "out_of_range", "string": None, "fret": None}
            else:
                assert tab_position_is_consistent(note["midiNote"], "bass", tab["string"], tab["fret"])
        else:
            assert note["tab"] is None  # never inferred for other lines
    assert any("outside the Standard 4-string" in warning for warning in details["warnings"])

    names = {part.get("id"): part.findtext("part-name") for part in root.findall("part-list/score-part")}
    assert names == {"P1": "Other Pitched Lines", "P3": "Bass (draft, with TAB)"}
    bass_part = root.find("part[@id='P3']")
    attributes = bass_part.find("measure/attributes")
    assert attributes.findtext("staves") == "2"
    clefs = {clef.get("number"): clef.findtext("sign") for clef in attributes.findall("clef")}
    assert clefs == {"1": "F", "2": "TAB"}
    assert attributes.find("clef[@number='1']/clef-octave-change").text == "-1"
    tuning = [
        (item.findtext("tuning-step"), item.findtext("tuning-octave"))
        for item in attributes.findall("staff-details/staff-tuning")
    ]
    assert tuning == [("E", "1"), ("A", "1"), ("D", "2"), ("G", "2")]
    standard = [note for note in bass_part.iter("note") if note.findtext("staff") == "1"]
    tab_staff = [note for note in bass_part.iter("note") if note.findtext("staff") == "2"]
    assert len(standard) == 6 and len(tab_staff) == 5
    assert all(note.find("notations/technical/fret") is not None for note in tab_staff)
    # Every pitched note appears once on a standard staff across the score.
    pitched_standard = [
        note
        for part_id in ("P1", "P3")
        for note in root.find(f"part[@id='{part_id}']").iter("note")
        if note.findtext("staff") == "1"
    ]
    assert len(pitched_standard) == len(notes_of(details))


def test_without_stems_the_default_writes_no_tablature_and_no_warning(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        details = build(client, job_id)
        root = musicxml(client, job_id)
    assert details["layers"]["tablature"]["status"] == "omitted"
    assert "separated bass stem" in details["layers"]["tablature"]["note"]
    assert details["tablature"]["instruments"][0]["noteCount"] == 0
    assert not any("tablature" in warning.lower() for warning in details["warnings"])
    assert [part.get("id") for part in root.findall("part")] == ["P1"]


def test_musician_choice_is_saved_marks_score_out_of_date_and_rebuilds(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        before = current_score_fingerprint(db.get_job(settings.database_path, job_id))
        response = client.put(
            f"/api/jobs/{job_id}/score/tablature", json={"bass": "bass", "guitar": "full_mix"}
        )
        assert response.status_code == 200
        assert response.json()["origin"] == "musician"
        record = db.get_job(settings.database_path, job_id)
        assert current_score_fingerprint(record) != before
        summary = job_score(client, job_id)
        assert summary["stale"] is True and summary["staleReason"] == "evidence"
        assert summary["tablature"]["request"] == {"bass": "bass", "guitar": "full_mix"}
        details = build(client, job_id, force=True)
        root = musicxml(client, job_id)
        assert job_score(client, job_id)["stale"] is False

    assert details["tablature"]["origin"] == "musician"
    assert details["sources"].get("tablature") is None  # public sources omit inputs
    guitar = next(item for item in details["tablature"]["instruments"] if item["instrument"] == "guitar")
    assert guitar["sourceKind"] == "full_mix" and guitar["assignedCount"] == 3
    assert any("fingering suggestion" in warning for warning in details["warnings"])
    assert {part.get("id") for part in root.findall("part")} == {"P1", "P3", "P4"}
    guitar_attrs = root.find("part[@id='P4']/measure/attributes")
    assert guitar_attrs.find("clef[@number='1']/sign").text == "G"
    assert guitar_attrs.findtext("staff-details/staff-lines") == "6"


def test_restoring_the_default_choice_restores_the_fingerprint(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    original = current_score_fingerprint(db.get_job(settings.database_path, job_id))
    with client_for(settings) as client:
        client.put(f"/api/jobs/{job_id}/score/tablature", json={"bass": None, "guitar": "bass"})
        changed = current_score_fingerprint(db.get_job(settings.database_path, job_id))
        response = client.put(f"/api/jobs/{job_id}/score/tablature", json={"bass": "bass", "guitar": None})
    assert response.json()["origin"] == "default"
    assert db.get_job(settings.database_path, job_id)["score_tablature_request"] is None
    assert changed != original
    assert current_score_fingerprint(db.get_job(settings.database_path, job_id)) == original


@pytest.mark.parametrize(
    "body",
    [
        {"bass": "bass", "guitar": "bass"},
        {"bass": "drums", "guitar": None},
        {"bass": "bass"},
        {"bass": "bass", "guitar": None, "capo": 2},
        {"bass": 1, "guitar": None},
    ],
)
def test_invalid_choices_are_rejected(tmp_path: Path, body: dict) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        response = client.put(f"/api/jobs/{job_id}/score/tablature", json=body)
    assert response.status_code == 422
    assert db.get_job(settings.database_path, job_id)["score_tablature_request"] is None


def test_choices_cannot_change_during_a_build_or_for_unknown_jobs(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        # Claim after startup: restart recovery fails attempts left from before.
        assert db.claim_score_attempt(settings.database_path, job_id, score_version="score-pipeline-v3")
        busy = client.put(f"/api/jobs/{job_id}/score/tablature", json={"bass": None, "guitar": "full_mix"})
        missing = client.put(f"/api/jobs/{'0' * 32}/score/tablature", json={"bass": None, "guitar": None})
        choices = client.get(f"/api/jobs/{job_id}/score/tablature").json()
    assert busy.status_code == 409 and missing.status_code == 404
    assert choices["request"] == {"bass": "bass", "guitar": None}
    assert [item["sourceKind"] for item in choices["sources"]] == ["vocals", "bass", "other", "full_mix"]


def test_job_serialization_hides_the_raw_column(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with client_for(settings) as client:
        client.put(f"/api/jobs/{job_id}/score/tablature", json={"bass": None, "guitar": "full_mix"})
        job = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)
    assert "score_tablature_request" not in json.dumps(job)


def _load(settings, job_id: str) -> dict:
    record = db.get_job(settings.database_path, job_id)
    return load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])


@pytest.mark.parametrize(
    "mutate",
    [
        lambda doc: next(n for n in notes_of(doc) if n["tab"] and n["tab"]["fret"] is not None)["tab"].update(
            fret=19
        ),
        lambda doc: next(n for n in notes_of(doc) if n["tab"] is None).update(
            tab={"instrument": "guitar", "status": "out_of_range", "string": None, "fret": None}
        ),
        lambda doc: doc["tablature"]["instruments"][0].update(assignedCount=6),
        lambda doc: doc["tablature"].update(origin="musician"),
        lambda doc: doc["tablature"]["request"].update(guitar="vocals"),
        lambda doc: doc["counts"].update(fingeredTabNotes=0),
        lambda doc: doc["layers"]["tablature"].update(status="omitted"),
        lambda doc: notes_of(doc)[0].pop("tab"),
    ],
)
def test_tampered_tablature_is_rejected(tmp_path: Path, mutate) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = _load(settings, job_id)
    validate_score_artifact(document)
    mutate(document)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(document)


def test_schema_two_scores_stay_readable_and_report_missing_tablature(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    legacy = copy.deepcopy(_load(settings, job_id))
    legacy["schemaVersion"] = 2
    legacy["pipelineVersion"] = "score-pipeline-v2"
    del legacy["tablature"]
    del legacy["counts"]["tabNotes"], legacy["counts"]["fingeredTabNotes"]
    for note in notes_of(legacy):
        del note["tab"]
    legacy["layers"]["tablature"] = {"status": "omitted", "note": "Not generated yet."}
    pointer = "score/score-document." + "d" * 32 + ".json"
    write_score_artifact(job_id, settings, legacy, artifact_file_name=pointer)
    db.update_job(
        settings.database_path,
        job_id,
        score_artifact_file_name=pointer,
        score_version="score-pipeline-v2",
        score_warning_count=len(legacy["warnings"]),
        separation_status="completed",
    )
    with client_for(settings) as client:
        summary = job_score(client, job_id)
        details = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true").json()
        root = musicxml(client, job_id)
    assert summary["stale"] is True and summary["staleReason"] == "tablature"
    assert details["tablature"] is None
    assert any("before tablature" in warning for warning in details["warnings"])
    assert [part.get("id") for part in root.findall("part")] == ["P1"]


# ---------------------------------------------------------------------------
# Review panel


TAB_DETAIL = {
    "available": True,
    "createdAt": SUMMARY["createdAt"],
    "timing": {"tempoBpm": 120, "beatsPerMeasure": 4, "meterSource": "analysis"},
    "layers": {},
    "counts": {"measures": 1, "notes": 2},
    "warnings": [],
    "parts": [],
    "tablature": {
        "version": "tab-fingering-v1",
        "origin": "musician",
        "request": {"bass": "bass", "guitar": "full_mix"},
        "instruments": [
            {"instrument": "bass", "label": "Bass", "sourceKind": "bass",
             "tuningName": "Standard 4-string (E A D G)", "strings": [43, 38, 33, 28],
             "frets": 20, "noteCount": 6, "assignedCount": 5, "outOfRangeCount": 1,
             "unplayableCount": 0},
        ],
    },
    "measures": [
        {"measureIndex": 0, "chordSymbol": None, "harmony": [], "notes": [
            {"id": "a", "noteName": "A1", "midiNote": 33, "quantizedBeat": 0,
             "quantizedDurationBeats": 1, "confidence": 0.9,
             "tab": {"instrument": "bass", "status": "assigned", "string": 3, "fret": 0}},
            {"id": "c", "noteName": "C1", "midiNote": 24, "quantizedBeat": 1,
             "quantizedDurationBeats": 1, "confidence": 0.9,
             "tab": {"instrument": "bass", "status": "out_of_range", "string": None, "fret": None}},
        ]},
    ],
}


def test_panel_shows_choices_summary_and_positions() -> None:
    summary = {**SUMMARY, "tablature": {"request": {"bass": "bass", "guitar": "full_mix"},
                                        "origin": "musician",
                                        "settingsUrl": "/api/jobs/done/score/tablature"}}
    result = _run_node(
        f"""
t.setDetail("done", {json.dumps(TAB_DETAIL)});
const html=t.renderScore({{id:"done",score:{json.dumps(summary)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert "<legend>Tablature lines</legend>" in html
    assert '<label for="tab-bass-done">Bass tablature from</label>' in html
    assert '<select id="tab-guitar-done" data-action="tablature" data-instrument="guitar"' in html
    assert '<option value="full_mix" selected>Full-mix melody</option>' in html
    assert '<option value="bass" selected>Bass stem</option>' in html
    assert "PopEx does not detect guitar parts" in html
    assert "<strong>Tablature suggestions</strong>" in html
    assert "from the bass stem: 5 of 6 notes fingered" in html
    assert '<span class="score-low">1 outside the range</span>' in html
    assert "A1 · beat 1 · 1 beat · bass string 3, fret 0" in html
    assert '<span class="score-low">outside bass range</span>' in html


def test_choices_are_hidden_while_building() -> None:
    summary = {**SUMMARY, "status": "processing", "canRebuild": False,
               "tablature": {"request": {"bass": "bass", "guitar": None}, "origin": "default",
                             "settingsUrl": "/api/jobs/done/score/tablature"}}
    result = _run_node(
        f"""
const html=t.renderScore({{id:"p",score:{json.dumps(summary)}}});
console.log(JSON.stringify({{html}}));
"""
    )
    assert "Tablature lines" not in result["html"]


def test_changing_a_choice_saves_both_selects() -> None:
    result = _run_node(
        """
const requested=[];
t.setFetch(async (url, options={}) => {requested.push({url,method:options.method||"GET",body:options.body||null});
  return {ok:true,status:200,json:async()=>[]};});
const bass={dataset:{action:"tablature",instrument:"bass",jobId:"job",settingsUrl:"/api/jobs/job/score/tablature"},value:"",disabled:false};
const guitar={dataset:{action:"tablature",instrument:"guitar",jobId:"job",settingsUrl:"/api/jobs/job/score/tablature"},value:"vocals",disabled:false};
const fieldset={querySelectorAll:()=>[bass,guitar]};
guitar.closest=(selector)=>selector==="fieldset"?fieldset:guitar;
await t.getListener("#jobs","change")({target:guitar});
console.log(JSON.stringify({requested,message:t.getElement("#jobs-message").textContent}));
"""
    )
    put = result["requested"][0]
    assert put["url"] == "/api/jobs/job/score/tablature" and put["method"] == "PUT"
    assert json.loads(put["body"]) == {"bass": None, "guitar": "vocals"}
    assert "Rebuild the score" in result["message"]


# ---------------------------------------------------------------------------
# MusicXML schema (runs when POPEX_MUSICXML_XSD points at the 3.1 XSD)

import os  # noqa: E402

from app.score_construction import build_score_document, score_to_musicxml_text  # noqa: E402
from app.tablature import assign_tablature  # noqa: E402

_XSD = os.environ.get("POPEX_MUSICXML_XSD")


def _tab_document() -> dict:
    notes = [
        {"id": f"g{i}", "sourceKind": "other", "startSeconds": i * 0.5, "endSeconds": i * 0.5 + 0.9,
         "midiNote": midi, "confidence": 0.9}
        for i, midi in enumerate([48, 50, 52, 53, 55, 57, 59, 60, 64, 67, 72, 76])
    ]
    notes += [
        {"id": f"c{i}", "sourceKind": "other", "startSeconds": 6.0, "endSeconds": 7.0,
         "midiNote": midi, "confidence": 0.8}
        for i, midi in enumerate([40, 47, 52, 56, 59, 64])
    ]
    notes += [
        {"id": f"b{i}", "sourceKind": "bass", "startSeconds": i * 1.0, "endSeconds": i * 1.0 + 1.6,
         "midiNote": midi, "confidence": 0.9}
        for i, midi in enumerate([28, 33, 35, 24, 40, 43, 46])
    ]
    notes.append({"id": "v", "sourceKind": "vocals", "startSeconds": 0.0, "endSeconds": 2.4,
                  "midiNote": 69, "confidence": 0.8})
    document = build_score_document(notes, tempo_bpm=120.0)
    flat = [note for measure in document["measures"] for note in measure["notes"]]
    for instrument, source in (("guitar", "other"), ("bass", "bass")):
        selected = [note for note in flat if note["sourceKind"] == source]
        positions = assign_tablature(selected, instrument)
        for note in selected:
            note["tab"] = {"instrument": instrument, **positions[note["id"]]}
    for note in flat:
        note.setdefault("tab", None)
    return document


def test_tab_export_writes_each_note_once_with_ties_on_both_staves() -> None:
    root = ET.fromstring(score_to_musicxml_text(_tab_document()))
    bass = root.find("part[@id='P3']")
    tied_tab = [
        note for note in bass.iter("note")
        if note.findtext("staff") == "2" and note.find("tie") is not None
    ]
    assert tied_tab  # sustained bass notes tie across bars on the TAB staff too
    guitar = root.find("part[@id='P4']")
    chord_tab = [note for note in guitar.iter("note") if note.findtext("staff") == "2"]
    assert {int(note.findtext("notations/technical/string")) for note in chord_tab} == {1, 2, 3, 4, 5, 6}


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_tab_examples_validate_against_musicxml_schema() -> None:
    from lxml import etree  # required whenever the schema path is configured

    schema = etree.XMLSchema(etree.parse(_XSD))
    tree = etree.fromstring(score_to_musicxml_text(_tab_document()).encode("utf-8"))
    assert schema.validate(tree), schema.error_log
