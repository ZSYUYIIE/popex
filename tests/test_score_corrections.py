"""Synchronized review and separate, undoable corrections (Cycle 10)."""

from __future__ import annotations

import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.score_artifacts import load_score_artifact, validate_score_artifact
from app.score_corrections import (
    CorrectionError,
    apply_corrections,
    new_operation,
    validate_log,
    validate_operation,
)
from app.transcription_events import write_raw_transcription
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import make_settings, raw_payload
from test_score_percussion import create_drum_job
from test_score_tablature import create_stem_job


def client_for(settings) -> TestClient:
    return TestClient(create_app(settings))


def build(client: TestClient, job_id: str, *, force: bool = False) -> None:
    response = client.post(f"/api/jobs/{job_id}/score/construct{'?force=true' if force else ''}")
    assert response.status_code == 202, response.text


def saved(client: TestClient, job_id: str, view: str = "corrected") -> dict:
    response = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true&view={view}")
    assert response.status_code == 200, response.text
    return response.json()


def notes_of(details: dict) -> list[dict]:
    return [note for measure in details["measures"] for note in measure["notes"]]


def hits_of(details: dict) -> list[dict]:
    return [hit for measure in details["measures"] for hit in measure.get("percussionHits", [])]


def add(client: TestClient, job_id: str, revision: int, operation: dict):
    return client.post(
        f"/api/jobs/{job_id}/score/corrections",
        json={"expectedRevision": revision, "operation": operation},
    )


def stored_document(settings, job_id: str) -> dict:
    record = db.get_job(settings.database_path, job_id)
    return load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])


def score_file_digest(settings, job_id: str) -> str:
    directory = settings.exports_dir / job_id / "score"
    return hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted(directory.iterdir()))
    ).hexdigest()


# ---------------------------------------------------------------------------
# Operation validation


@pytest.mark.parametrize(
    "operation",
    [
        {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 11},
        {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": True},
        {"op": "set_pitch", "target": {"noteId": "p c"}, "midiNote": 60},
        {"op": "set_tab", "target": {"noteId": "x"}, "string": 7, "fret": 0},
        {"op": "set_chord", "target": {"measureIndex": 0}, "symbol": "<C>"},
        {"op": "set_chord", "target": {"measureIndex": 0}, "symbol": " C"},
        {"op": "set_drum_voice", "target": {"eventId": "d", "hitIndex": 0}, "broadVoice": "kick"},
        {"op": "delete_note", "target": {"noteId": "p_c"}, "extra": 1},
        {"op": "transpose_all"},
    ],
)
def test_invalid_operations_are_rejected(operation: dict) -> None:
    with pytest.raises(CorrectionError):
        validate_operation(operation)


def test_log_validation_rejects_duplicates_and_bad_schema() -> None:
    operation = new_operation({"op": "delete_note", "target": {"noteId": "p_c"}})
    with pytest.raises(CorrectionError):
        validate_log({"schemaVersion": 1, "revision": 1, "operations": [operation, operation], "redo": []})
    with pytest.raises(CorrectionError):
        validate_log({"schemaVersion": 2, "revision": 0, "operations": [], "redo": []})


# ---------------------------------------------------------------------------
# Pure application


def test_apply_never_changes_the_prediction_and_reports_originals(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    pristine = copy.deepcopy(document)
    operations = [
        new_operation({"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 62}),
        new_operation({"op": "delete_note", "target": {"noteId": "p_e"}}),
        new_operation({"op": "set_chord", "target": {"measureIndex": 0}, "symbol": "Dm7"}),
        new_operation({"op": "set_drum_voice", "target": {"eventId": "d_weak_1", "hitIndex": 0},
                       "broadVoice": "mid_drum"}),
        new_operation({"op": "delete_hit", "target": {"eventId": "d_crash_1", "hitIndex": 0}}),
        new_operation({"op": "delete_note", "target": {"noteId": "missing"}}),
    ]
    corrected, report = apply_corrections(document, operations)
    assert document == pristine
    validate_score_artifact(corrected)

    by_id = {note["id"]: note for note in notes_of(corrected)}
    assert by_id["p_c"]["midiNote"] == 62 and by_id["p_c"]["noteName"] == "D4"
    assert "p_e" not in by_id
    assert corrected["measures"][0]["chordSymbol"] == "Dm7"
    assert corrected["layers"]["chordSymbols"]["status"] == "included"
    assert corrected["counts"]["notes"] == pristine["counts"]["notes"] - 1
    weak = next(hit for hit in hits_of(corrected) if hit["eventId"] == "d_weak_1")
    assert weak["broadVoice"] == "mid_drum" and weak["resolved"] is True
    crash_2 = next(hit for hit in hits_of(corrected) if hit["eventId"] == "d_crash_2")
    assert crash_2["notation"] == "notated"  # its merged twin was deleted
    assert corrected["counts"]["unresolvedPercussionHits"] == pristine["counts"]["unresolvedPercussionHits"] - 1
    assert report["notes"]["p_c"]["noteName"] == "C4"
    assert report["chords"] == {"0": None}
    assert [item["id"] for item in report["deletedNotes"]] == ["p_e"]
    assert report["notApplicable"] == [
        {"id": operations[-1]["id"], "op": "delete_note", "reason": "The note is not in this score."}
    ]
    assert corrected["warnings"][0].startswith("Musician corrections are applied")


def test_reset_all_ignores_earlier_operations_and_editing_back_clears_the_marker(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    up = new_operation({"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 61})
    back = new_operation({"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 60})
    corrected, report = apply_corrections(document, [up, back])
    assert report["notes"] == {}
    reset = new_operation({"op": "reset_all"})
    corrected, report = apply_corrections(document, [up, reset])
    assert corrected["measures"] == document["measures"]
    assert report["applied"] == []


def test_pitch_changes_refinger_and_tab_corrections_must_sound_the_pitch(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    a1 = next(note for note in notes_of(document) if note["id"] == "b_1")  # A1 = string 3, fret 0
    assert a1["tab"]["string"] == 3 and a1["tab"]["fret"] == 0
    corrected, _ = apply_corrections(
        document, [new_operation({"op": "set_pitch", "target": {"noteId": "b_1"}, "midiNote": 35})]
    )
    moved = next(note for note in notes_of(corrected) if note["id"] == "b_1")
    assert moved["tab"] == {"instrument": "bass", "status": "assigned", "string": 3, "fret": 2}
    good = new_operation({"op": "set_tab", "target": {"noteId": "b_1"}, "string": 4, "fret": 5})
    bad = new_operation({"op": "set_tab", "target": {"noteId": "b_1"}, "string": 4, "fret": 6})
    off_line = new_operation({"op": "set_tab", "target": {"noteId": "p_c"}, "string": 1, "fret": 0})
    corrected, report = apply_corrections(document, [good, bad, off_line])
    assert next(n for n in notes_of(corrected) if n["id"] == "b_1")["tab"]["fret"] == 5
    assert [item["id"] for item in report["notApplicable"]] == [bad["id"], off_line["id"]]
    validate_score_artifact(corrected)
    # Out of range after a pitch change keeps the note but drops the position.
    low, _ = apply_corrections(
        document, [new_operation({"op": "set_pitch", "target": {"noteId": "b_1"}, "midiNote": 20})]
    )
    note = next(n for n in notes_of(low) if n["id"] == "b_1")
    assert note["tab"]["status"] == "out_of_range"
    validate_score_artifact(low)


# ---------------------------------------------------------------------------
# API


def test_add_undo_redo_reset_flow_keeps_the_saved_prediction(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        digest = score_file_digest(settings, job_id)
        state = client.get(f"/api/jobs/{job_id}/score/corrections").json()
        assert state["revision"] == 0 and state["canUndo"] is False

        response = add(client, job_id, 0, {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 62})
        assert response.status_code == 200, response.text
        assert response.json()["revision"] == 1 and response.json()["activeCount"] == 1

        stale = add(client, job_id, 0, {"op": "delete_note", "target": {"noteId": "p_e"}})
        assert stale.status_code == 409

        corrected = saved(client, job_id)
        original = saved(client, job_id, "original")
        assert corrected["view"] == "corrected" and original["view"] == "original"
        assert {n["id"]: n["midiNote"] for n in notes_of(corrected)}["p_c"] == 62
        assert {n["id"]: n["midiNote"] for n in notes_of(original)}["p_c"] == 60
        assert corrected["corrections"]["review"]["notes"]["p_c"]["noteName"] == "C4"

        undo = client.post(f"/api/jobs/{job_id}/score/corrections/undo", json={"expectedRevision": 1})
        assert undo.json()["activeCount"] == 0 and undo.json()["canRedo"] is True
        redo = client.post(f"/api/jobs/{job_id}/score/corrections/redo", json={"expectedRevision": 2})
        assert redo.json()["activeCount"] == 1 and redo.json()["canRedo"] is False
        reset = client.post(f"/api/jobs/{job_id}/score/corrections/reset", json={"expectedRevision": 3})
        assert reset.json()["activeCount"] == 0 and reset.json()["canUndo"] is True
        assert {n["id"]: n["midiNote"] for n in notes_of(saved(client, job_id))}["p_c"] == 60
        restored = client.post(f"/api/jobs/{job_id}/score/corrections/undo", json={"expectedRevision": 4})
        assert restored.json()["activeCount"] == 1
        # Redo re-applies the undone reset, which is itself undoable again.
        again = client.post(f"/api/jobs/{job_id}/score/corrections/redo", json={"expectedRevision": 5})
        assert again.json()["activeCount"] == 0 and again.json()["redoCount"] == 0
        assert again.json()["undoCount"] == 2  # [set_pitch, reset_all]
    assert score_file_digest(settings, job_id) == digest  # predictions untouched


def test_inapplicable_or_invalid_new_corrections_are_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        missing = add(client, job_id, 0, {"op": "delete_note", "target": {"noteId": "nope"}})
        invalid = add(client, job_id, 0, {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 200})
        extra = client.post(
            f"/api/jobs/{job_id}/score/corrections",
            json={"expectedRevision": 0, "operation": {}, "author": "x"},
        )
        empty_undo = client.post(f"/api/jobs/{job_id}/score/corrections/undo", json={"expectedRevision": 0})
        no_score = client.get(f"/api/jobs/{'0' * 32}/score/corrections")
    assert missing.status_code == 422 and "not in this score" in missing.json()["detail"]
    assert invalid.status_code == 422 and extra.status_code == 422
    assert empty_undo.status_code == 409 and no_score.status_code == 404
    assert db.get_score_corrections(settings.database_path, job_id) is None


def test_downloads_offer_corrected_and_original_scores(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        add(client, job_id, 0, {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 62})
        base = f"/api/jobs/{job_id}/score/saved/download?format="
        corrected = client.get(base + "musicxml")
        original = client.get(base + "musicxml&view=original")
        corrected_json = client.get(base + "json").json()
        bad = client.get(base + "musicxml&view=draft")
    assert 'filename="draft-score-original.musicxml"' in original.headers["content-disposition"]

    def first_step(content: bytes) -> str:
        root = ET.fromstring(content)
        return root.find("part[@id='P1']//note/pitch/step").text

    assert first_step(corrected.content) == "D" and first_step(original.content) == "C"
    assert corrected_json["warnings"][0].startswith("Musician corrections are applied")
    assert bad.status_code == 422


def test_corrections_survive_a_rebuild_and_report_vanished_targets(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        add(client, job_id, 0, {"op": "set_pitch", "target": {"noteId": "p_c"}, "midiNote": 62})
        add(client, job_id, 1, {"op": "delete_note", "target": {"noteId": "p_g"}})
        # A new transcription drops p_g; the score is rebuilt from it.
        payload = raw_payload()
        payload["pitchedNoteEvents"] = [e for e in payload["pitchedNoteEvents"] if e["id"] != "p_g"]
        payload["alignmentCandidates"] = [c for c in payload["alignmentCandidates"] if c["eventId"] != "p_g"]
        from test_score_percussion import drum_raw_payload

        drums = drum_raw_payload()
        payload["percussionEvents"] = drums["percussionEvents"]
        payload["alignmentCandidates"] += [c for c in drums["alignmentCandidates"] if c["eventType"] == "percussion"]
        payload["createdAt"] = "2026-08-14T05:00:00+00:00"
        write_raw_transcription(job_id, settings, payload)
        db.update_job(
            settings.database_path, job_id, transcribed_at=payload["createdAt"], pitched_event_count=2
        )
        build(client, job_id, force=True)
        state = client.get(f"/api/jobs/{job_id}/score/corrections").json()
        details = saved(client, job_id)
    assert state["activeCount"] == 2 and state["appliedCount"] == 1
    assert [item["op"] for item in state["notApplicable"]] == ["delete_note"]
    assert {n["id"]: n["midiNote"] for n in notes_of(details)}["p_c"] == 62


def test_corrections_are_never_listed_in_job_serialization(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        add(client, job_id, 0, {"op": "delete_note", "target": {"noteId": "p_e"}})
        job = next(item for item in client.get("/api/jobs").json() if item["id"] == job_id)
    assert "operations" not in json.dumps(job)


# ---------------------------------------------------------------------------
# Review panel


EDIT_DETAIL = {
    "available": True,
    "createdAt": SUMMARY["createdAt"],
    "timing": {"tempoBpm": 120, "beatsPerMeasure": 4, "meterSource": "analysis"},
    "layers": {},
    "counts": {"measures": 1, "notes": 1},
    "warnings": [],
    "parts": [],
    "originalDownloadUrls": {"musicxml": "/api/jobs/edit/score/saved/download?format=musicxml&view=original"},
    "tablature": {"instruments": [{"instrument": "bass", "strings": [43, 38, 33, 28], "frets": 20}]},
    "corrections": {
        "revision": 3, "activeCount": 2, "undoCount": 2, "redoCount": 0, "canUndo": True,
        "canRedo": False, "canReset": True, "notApplicable": [],
        "urls": {"add": "/api/jobs/edit/score/corrections", "undo": "/api/jobs/edit/score/corrections/undo",
                 "redo": "/api/jobs/edit/score/corrections/redo", "reset": "/api/jobs/edit/score/corrections/reset"},
        "review": {"notes": {"b1": {"midiNote": 33, "noteName": "A1", "tab": None}}, "chords": {"0": "C"},
                   "hits": {}, "deletedNotes": [{"id": "x", "measureIndex": 0, "noteName": "E4", "quantizedBeat": 2}],
                   "deletedHits": []},
    },
    "measures": [
        {"measureIndex": 0, "startSeconds": 0, "endSeconds": 2, "chordSymbol": "Dm", "harmony": [],
         "percussionHits": [{"eventId": "d1", "hitIndex": 0, "broadVoice": "low_drum", "quantizedBeat": 0,
                             "notation": "notated"}],
         "notes": [{"id": "b1", "noteName": "B1", "midiNote": 35, "quantizedBeat": 0, "quantizedDurationBeats": 1,
                    "confidence": 0.9, "tab": {"instrument": "bass", "status": "assigned", "string": 3, "fret": 2}}]},
    ],
}
EDIT_JOB = {"id": "edit", "files": [{"kind": "analysis", "preview_url": "/api/jobs/edit/files/analysis.wav"}],
            "score": SUMMARY}


def test_review_rows_markers_and_correction_bar_render() -> None:
    result = _run_node(
        f"""
t.setDetail("edit", {json.dumps(EDIT_DETAIL)});
const html=t.renderScore({json.dumps(EDIT_JOB)});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert "2 corrections applied on top of the prediction" in html
    assert 'aria-pressed="false"' in html and ">Edit score</button>" in html
    assert ">Undo (2)</button>" in html and "Redo</button>" in html
    assert '<label for="review-source-edit">Listen to</label>' in html
    assert 'data-action="play-bar"' in html and 'aria-label="Play bar 1"' in html
    assert '<tr data-review-job="edit" data-start="0" data-end="2">' in html
    assert '<span class="score-edited">edited · was A1</span>' in html
    assert "Dm <span class=\"score-edited\">edited · was C</span>" in html
    assert '<li class="score-deleted"><del>E4</del> · beat 3 · deleted</li>' in html
    assert "Download original prediction (MusicXML)" in html
    assert "Lower B1 by a semitone" not in html  # controls only in edit mode


def test_edit_mode_renders_keyboard_controls() -> None:
    result = _run_node(
        f"""
t.setDetail("edit", {json.dumps(EDIT_DETAIL)});
const toggle={{dataset:{{action:"score-edit-toggle",jobId:"edit",focusKey:"x"}},closest:()=>toggle}};
t.renderScore({json.dumps(EDIT_JOB)});
await t.getListener("#jobs","click")({{target:{{closest:()=>toggle}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)});
console.log(JSON.stringify({{html}}));
"""
    )
    html = result["html"]
    assert 'aria-pressed="true"' in html and ">Finish editing</button>" in html
    assert 'aria-label="Lower B1 by a semitone"' in html and 'data-midi="34"' in html
    assert 'aria-label="Raise B1 by an octave"' in html and 'data-midi="47"' in html
    assert 'aria-label="Delete B1"' in html
    assert '<option value="3:2" selected>string 3, fret 2</option>' in html
    assert '<option value="4:7">string 4, fret 7</option>' in html
    assert 'aria-label="Chord symbol for bar 1"' in html and ">Save chord</button>" in html
    assert 'aria-label="Drum voice at beat 1"' in html
    assert 'aria-label="Delete drum hit at beat 1"' in html


def test_correction_click_posts_with_the_current_revision() -> None:
    result = _run_node(
        f"""
const requested=[];
t.setFetch(async (url, options={{}}) => {{requested.push({{url,method:options.method||"GET",body:options.body||null}});
  return {{ok:true,status:200,json:async()=>({json.dumps(EDIT_DETAIL)})}};}});
t.setDetail("edit", {json.dumps(EDIT_DETAIL)});
const button={{dataset:{{action:"correction",op:"set_pitch",jobId:"edit",noteId:"b1",midi:"34",focusKey:"k"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>button}}}});
console.log(JSON.stringify({{requested,message:t.getElement("#jobs-message").textContent}}));
"""
    )
    post = result["requested"][0]
    assert post["url"] == "/api/jobs/edit/score/corrections" and post["method"] == "POST"
    assert json.loads(post["body"]) == {
        "expectedRevision": 3,
        "operation": {"op": "set_pitch", "target": {"noteId": "b1"}, "midiNote": 34},
    }
    assert result["message"] == "Pitch changed to A#1."


_XSD = __import__("os").environ.get("POPEX_MUSICXML_XSD")


@pytest.mark.skipif(not _XSD, reason="set POPEX_MUSICXML_XSD to the MusicXML 3.1 XSD")
def test_corrected_exports_validate_against_musicxml_schema(tmp_path: Path) -> None:
    from lxml import etree  # required whenever the schema path is configured

    from app.score_construction import score_to_musicxml_text
    from app.score_pipeline import score_export_document

    schema = etree.XMLSchema(etree.parse(_XSD))
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    corrected, _ = apply_corrections(
        document,
        [
            new_operation({"op": "set_pitch", "target": {"noteId": "b_1"}, "midiNote": 36}),
            new_operation({"op": "set_tab", "target": {"noteId": "b_0"}, "string": 4, "fret": 0}),
            new_operation({"op": "delete_note", "target": {"noteId": "p_e"}}),
            new_operation({"op": "set_chord", "target": {"measureIndex": 0}, "symbol": "Am7/G"}),
        ],
    )
    tree = etree.fromstring(score_to_musicxml_text(score_export_document(corrected)).encode("utf-8"))
    assert schema.validate(tree), schema.error_log
