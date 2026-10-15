"""Musician timing corrections remain separate from raw score evidence."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from app.score_artifacts import validate_score_artifact
from app.score_construction import score_to_midi_bytes, score_to_musicxml_text
from app.score_corrections import CorrectionError, apply_corrections, new_operation, validate_operation
from app.score_pipeline import score_export_document
from test_frontend_score import _run_node
from test_harmony_api import make_settings
from test_score_corrections import (EDIT_DETAIL, EDIT_JOB, add, build, client_for,
                                    notes_of, saved, score_file_digest, stored_document)
from test_score_percussion import create_drum_job
from test_score_tablature import create_stem_job


def timing_operation(note_id="p_c", **overrides):
    return {"op": "set_note_timing", "target": {"noteId": note_id},
            "measureIndex": 0, "beat": 0.25, "durationBeats": 1.5, **overrides}


def test_timing_correction_changes_only_notation_and_keeps_prediction(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    pristine = copy.deepcopy(document)
    corrected, report = apply_corrections(document, [new_operation(timing_operation())])
    note = next(n for m in corrected["measures"] for n in m["notes"] if n["id"] == "p_c")
    original = next(n for m in document["measures"] for n in m["notes"] if n["id"] == "p_c")
    assert note["quantizedBeat"] == 0.25
    assert note["quantizedEndBeat"] == 1.75
    assert note["quantizedDurationBeats"] == 1.5
    for key in ("startSeconds", "endSeconds", "startBeat", "endBeat", "rawMidiPitch", "confidence"):
        assert note[key] == original[key]
    assert document == pristine
    assert report["notes"]["p_c"]["quantizedBeat"] == original["quantizedBeat"]
    validate_score_artifact(corrected)


@pytest.mark.parametrize("overrides", [
    {"beat": True}, {"beat": float("nan")}, {"beat": float("inf")},
    {"beat": -0.25}, {"beat": 0.1}, {"beat": "1"}, {"beat": 12},
    {"durationBeats": 0}, {"durationBeats": -1}, {"durationBeats": 0.3},
    {"durationBeats": 4096.25}, {"durationBeats": 10**1000},
    {"measureIndex": False}, {"measureIndex": -1}, {"measureIndex": 2048},
    {"measureIndex": 0.5}, {"unexpected": 1},
])
def test_invalid_timing_is_rejected(overrides):
    with pytest.raises(CorrectionError):
        validate_operation(timing_operation(**overrides))


@pytest.fixture
def prediction(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    return settings, job_id, stored_document(settings, job_id)


def test_cross_bar_move_sustain_exports_and_edit_back(prediction):
    _settings, _job, document = prediction
    # Existing fixture includes two bars. Start late in bar 1 and sustain into 2.
    operation = new_operation(timing_operation(beat=3.75, durationBeats=1.5))
    changed, report = apply_corrections(document, [operation])
    validate_score_artifact(changed)
    exported = score_export_document(changed)
    xml = ET.fromstring(score_to_musicxml_text(exported))
    c_notes = [n for n in xml.findall(".//part/measure/note")
               if n.findtext("pitch/step") == "C" and n.findtext("pitch/octave") == "4"]
    assert any(n.find("tie[@type='start']") is not None for n in c_notes)
    assert any(n.find("tie[@type='stop']") is not None for n in c_notes)
    midi = score_to_midi_bytes(exported)
    assert midi != score_to_midi_bytes(score_export_document(document))
    assert (1800, 0x90, 60) in midi_events(midi)
    assert (2520, 0x80, 60) in midi_events(midi)
    moved, _ = apply_corrections(document, [new_operation(timing_operation(measureIndex=1, beat=0, durationBeats=0.25))])
    assert "p_c" not in {n["id"] for n in moved["measures"][0]["notes"]}
    assert sum(n["id"] == "p_c" for n in notes_of(moved)) == 1
    assert next(n for n in moved["measures"][1]["notes"] if n["id"] == "p_c")["quantizedBeat"] == 4
    validate_score_artifact(moved)
    original = next(n for n in notes_of(document) if n["id"] == "p_c")
    back = new_operation(timing_operation(beat=original["quantizedBeat"], durationBeats=original["quantizedDurationBeats"]))
    restored, report = apply_corrections(document, [operation, back])
    assert report["notes"] == {}
    assert restored["measures"] == document["measures"]


def test_rebuild_extent_missing_target_and_moved_then_deleted(prediction):
    _, _, document = prediction
    outside = new_operation(timing_operation(measureIndex=len(document["measures"])))
    over_end = new_operation(timing_operation(measureIndex=len(document["measures"])-1, beat=3.75, durationBeats=0.5))
    invalid_beat = new_operation(timing_operation(beat=4))
    missing = new_operation(timing_operation("missing"))
    changed, report = apply_corrections(document, [outside, over_end, invalid_beat, missing])
    assert len(report["notApplicable"]) == 4 and not report["applied"]
    assert changed["measures"] == document["measures"]
    move = new_operation(timing_operation(measureIndex=1, beat=0, durationBeats=0.25))
    changed, report = apply_corrections(document, [move, new_operation({"op": "delete_note", "target": {"noteId": "p_c"}})])
    assert not any(n["id"] == "p_c" for n in notes_of(changed))
    assert report["deletedNotes"][0]["measureIndex"] == 1
    # A later rebuild has fewer bars: the same stable ID is not redirected.
    smaller = copy.deepcopy(document)
    smaller["measures"] = smaller["measures"][:1]
    _, report = apply_corrections(smaller, [move])
    assert not report["applied"] and len(report["notApplicable"]) == 1
    assert apply_corrections(document, [move]) == apply_corrections(document, [move])


def test_timing_rejects_overlapping_tablature_hold(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    members = [n for n in notes_of(document) if n.get("tab") and n["tab"]["status"] == "assigned"]
    note, other = members[:2]
    other["tab"] = dict(note["tab"])
    operation = new_operation(timing_operation(note["id"], measureIndex=int(other["quantizedBeat"]//4),
                                              beat=other["quantizedBeat"]%4, durationBeats=0.25))
    changed, report = apply_corrections(document, [operation])
    assert not report["applied"] and "same tablature string" in report["notApplicable"][0]["reason"]
    assert changed["measures"] == document["measures"]


def test_api_undo_redo_restart_exports_and_prediction_hash(prediction):
    settings, job_id, document = prediction
    digest = score_file_digest(settings, job_id)
    raw_files = {p: hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (settings.exports_dir/job_id).rglob("*") if p.is_file()}
    with client_for(settings) as client:
        original_midi = client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi&view=original").content
        response = add(client, job_id, 0, timing_operation(beat=3.75, durationBeats=1.5))
        assert response.status_code == 200, response.text
        assert add(client, job_id, 0, timing_operation()).status_code == 409
        assert add(client, job_id, 1, timing_operation(measureIndex=2047)).status_code == 422
        assert saved(client, job_id)["corrections"]["revision"] == 1
        changed = client.get(f"/api/jobs/{job_id}/score/saved/download?format=json").json()
        assert next(n for n in notes_of(changed) if n["id"] == "p_c")["quantizedBeat"] == 3.75
        assert client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi").content != original_midi
        assert client.get(f"/api/jobs/{job_id}/score/saved/download?format=midi&view=original").content == original_midi
        undo = client.post(f"/api/jobs/{job_id}/score/corrections/undo", json={"expectedRevision": 1})
        assert undo.status_code == 200
        assert saved(client, job_id)["measures"] == saved(client, job_id, "original")["measures"]
        assert client.post(f"/api/jobs/{job_id}/score/corrections/redo", json={"expectedRevision": 2}).status_code == 200
        assert score_file_digest(settings, job_id) == digest
    with client_for(settings) as client:
        after_restart = saved(client, job_id)
        assert next(n for n in notes_of(after_restart) if n["id"] == "p_c")["quantizedBeat"] == 3.75
        build(client, job_id, force=True)
        assert saved(client, job_id)["corrections"]["appliedCount"] == 1
        assert client.post(f"/api/jobs/{job_id}/score/corrections/reset", json={"expectedRevision": 3}).status_code == 200
    # Before rebuilding, correction writes never touched any upstream file.
    for path, hashed in raw_files.items():
        if path.exists() and path.parent.name != "score":
            assert hashlib.sha256(path.read_bytes()).hexdigest() == hashed
    # A fresh rebuild replaces the prediction intentionally; compare its music.
    assert stored_document(settings, job_id)["measures"] == document["measures"]


def test_frontend_timing_labels_and_revision_checked_request():
    result = _run_node(f'''
t.setDetail("edit", {json.dumps(EDIT_DETAIL)});
t.renderScore({json.dumps(EDIT_JOB)});
const toggle={{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>toggle}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)});
const key=html.match(/data-action="correction-timing"[^>]*data-input-key="([^"]+)"/)[1];
for(const [field,value] of [["bar","1"],["beat","1.25"],["duration","1.5"]])
  t.getElement(`[data-timing-input="${{key}}-${{field}}"]`).value=value;
const requested=[];
t.setFetch(async(url,options={{}})=>{{requested.push({{url,body:options.body||null}});return {{ok:true,status:200,json:async()=>({json.dumps(EDIT_DETAIL)})}};}});
const button={{dataset:{{action:"correction-timing",jobId:"edit",noteId:"b1",inputKey:key,focusKey:"save"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>button}}}});
console.log(JSON.stringify({{html,requested}}));
''')
    assert 'aria-label="Bar for B1"' in result["html"]
    assert 'aria-label="Beat for B1"' in result["html"]
    assert 'aria-label="Duration in beats for B1"' in result["html"]
    assert 'class="note-timing" data-open-key="timing-' in result["html"]
    assert json.loads(result["requested"][0]["body"]) == {
        "expectedRevision": 3, "operation": timing_operation("b1")}


def test_timing_form_keys_never_alias_distinct_event_ids():
    detail = copy.deepcopy(EDIT_DETAIL)
    note = detail["measures"][0]["notes"][0]
    detail["measures"][0]["notes"] = [dict(note, id=value) for value in ("a:b", "a-b", "a.b")]
    result = _run_node(f'''
t.setDetail("edit", {json.dumps(detail)});
t.renderScore({json.dumps(EDIT_JOB)});
const toggle={{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>toggle}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)});
const keys=[...html.matchAll(/data-action="correction-timing"[^>]*data-input-key="([^"]+)"/g)].map(match=>match[1]);
console.log(JSON.stringify({{keys}}));
''')
    assert len(result["keys"]) == len(set(result["keys"])) == 3


def midi_events(data):
    """Read generated format-0 events, retaining absolute ticks and note pitch."""
    track = data[22:22+int.from_bytes(data[18:22], "big")]
    offset, tick, events = 0, 0, []
    def vlq():
        nonlocal offset
        value = 0
        while True:
            byte = track[offset]
            offset += 1
            value = (value << 7) | (byte & 127)
            if byte < 128:
                return value
    while offset < len(track):
        tick += vlq()
        status = track[offset]
        offset += 1
        if status == 255:
            offset += 1
            size = vlq()
            offset += size
        else:
            events.append((tick, status & 240, track[offset]))
            offset += 1 if status & 240 in (192, 208) else 2
    return events


@pytest.mark.skipif(not os.environ.get("POPEX_MUSICXML_XSD"), reason="set POPEX_MUSICXML_XSD")
def test_timing_musicxml_schema(prediction):
    from lxml import etree
    _, _, document = prediction
    corrected, _ = apply_corrections(document, [new_operation(timing_operation(beat=3.75, durationBeats=1.5))])
    schema = etree.XMLSchema(etree.parse(os.environ["POPEX_MUSICXML_XSD"]))
    tree = etree.fromstring(score_to_musicxml_text(score_export_document(corrected)).encode())
    assert schema.validate(tree), schema.error_log
