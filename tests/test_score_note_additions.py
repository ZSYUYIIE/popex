"""Missed notes are musician corrections, never fabricated raw predictions."""
from __future__ import annotations

import copy
import json
import os
import xml.etree.ElementTree as ET

import pytest

from app.score_artifacts import ScoreArtifactError, encode_score_artifact, validate_score_artifact
from app.score_corrections import CorrectionError, apply_corrections, new_operation, validate_operation
from app.score_construction import plan_score_parts, score_to_musicxml_text, score_to_midi_bytes
from app.score_pipeline import score_export_document
from app.score_note_entry import note_entry_parts
from app import db
from app.transcription_events import write_raw_transcription
from test_harmony_api import make_settings
from test_score_corrections import (EDIT_DETAIL, EDIT_JOB, add, build, client_for, notes_of,
                                    saved, score_file_digest, stored_document)
from test_score_percussion import create_drum_job, drum_raw_payload
from test_score_tablature import create_stem_job
from test_score_timing_corrections import midi_events
from test_frontend_score import _run_node


def addition(**changes):
    return {"op": "add_note", "target": {"partId": "musician-notes"},
            "measureIndex": 0, "beat": 0.25, "durationBeats": 1.5, "midiNote": 64, **changes}


def test_note_addition_is_a_server_identified_correction():
    operation = new_operation(addition())
    assert operation["op"] == "add_note"
    assert len(operation["id"]) == 32


def test_add_missed_note_preserves_original_prediction(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
        before = saved(client, job_id, "original")
        digest = score_file_digest(settings, job_id)
        response = add(client, job_id, 0, addition())
        assert response.status_code == 200, response.text
        corrected = saved(client, job_id)
        manual = [note for note in notes_of(corrected) if note.get("origin") == "musician"]
        assert len(manual) == 1 and manual[0]["midiNote"] == 64
        assert manual[0]["confidence"] is None
        assert manual[0]["startSeconds"] is None
        assert saved(client, job_id, "original")["measures"] == before["measures"]
        assert score_file_digest(settings, job_id) == digest


@pytest.fixture
def prediction(tmp_path):
    settings = make_settings(tmp_path)
    job = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job)
    return settings, job, stored_document(settings, job)


@pytest.mark.parametrize("changes", [
    {"midiNote": True}, {"midiNote": 11}, {"midiNote": 128}, {"midiNote": "64"},
    {"beat": False}, {"beat": float("nan")}, {"beat": float("inf")}, {"beat": 0.1},
    {"beat": -0.25}, {"beat": 12}, {"durationBeats": 0}, {"durationBeats": 0.3},
    {"durationBeats": 4096.25}, {"durationBeats": 10**1000},
    {"measureIndex": True}, {"measureIndex": -1}, {"measureIndex": 2048},
    {"target": {"partId": "drums"}}, {"target": {"partId": []}},
    {"target": {"partId": "musician-notes", "rawEventId": "fake"}},
    {"confidence": 1}, {"id": "forged"}, {"op": []},
])
def test_invalid_or_forged_addition_is_refused(changes):
    with pytest.raises(CorrectionError):
        validate_operation(addition(**changes))


def test_manual_projection_requires_exact_authorization_and_cannot_be_persisted(prediction):
    _, _, original = prediction
    pristine = copy.deepcopy(original)
    operation = new_operation(addition(beat=3.75, durationBeats=1.5))
    changed, report = apply_corrections(original, [operation])
    note = next(note for note in notes_of(changed) if note.get("origin") == "musician")
    identities = frozenset(report["addedNotes"])
    assert identities == {"user:" + operation["id"]}
    assert note["correctionId"] == operation["id"] and note["entryVersion"] == 1
    assert all(note[key] is None for key in ("confidence", "startSeconds", "endSeconds", "startBeat", "endBeat", "rawMidiPitch", "quantizationShiftSeconds"))
    assert note["partId"] is None
    validate_score_artifact(changed, manual_note_ids=identities)
    with pytest.raises(ScoreArtifactError):
        validate_score_artifact(changed)
    with pytest.raises(ScoreArtifactError):
        encode_score_artifact(changed)
    with pytest.raises(ScoreArtifactError):
        validate_score_artifact(original, manual_note_ids=identities)
    assert original == pristine
    exported = score_export_document(changed)
    root = ET.fromstring(score_to_musicxml_text(exported, only_part="musician-notes"))
    assert root.findtext("part-list/score-part/part-name") == "Musician-entered notes"
    assert root.find(".//tie[@type='start']") is not None
    assert root.find(".//tie[@type='stop']") is not None
    events = midi_events(score_to_midi_bytes(exported))
    assert (1800, 0x90, 64) in events and (2520, 0x80, 64) in events


@pytest.mark.parametrize("changes", [
    {"confidence": 0.9}, {"startSeconds": 1.0}, {"rawMidiPitch": 64.2},
    {"partId": "detected-part"}, {"entryVersion": True}, {"correctionId": "a" * 32},
    {"targetPartId": []}, {"extraEvidence": "fake"},
])
def test_authorized_projection_still_rejects_fabricated_evidence(prediction, changes):
    _, _, document = prediction
    changed, report = apply_corrections(document, [new_operation(addition())])
    next(note for note in notes_of(changed) if note.get("origin") == "musician").update(changes)
    with pytest.raises(ScoreArtifactError):
        validate_score_artifact(changed, manual_note_ids=frozenset(report["addedNotes"]))


def test_existing_part_target_is_not_recreated_or_redirected_after_rebuild(prediction):
    _, _, document = prediction
    part = next(item for item in note_entry_parts(document) if item["id"] != "musician-notes")
    operation = new_operation(addition(target={"partId": part["id"]}))
    changed, report = apply_corrections(document, [operation])
    assert len(report["addedNotes"]) == 1
    emptied = copy.deepcopy(document)
    for measure in emptied["measures"]:
        measure["notes"] = []
    emptied["scoreParts"] = []
    changed, report = apply_corrections(emptied, [operation])
    assert not report["applied"] and not report["addedNotes"]
    assert "no longer" in report["notApplicable"][0]["reason"]
    for invalid in (addition(measureIndex=999), addition(beat=4), addition(beat=3.75, durationBeats=4096)):
        _, report = apply_corrections(document, [new_operation(invalid)])
        assert report["notApplicable"] and not report["addedNotes"]


@pytest.mark.parametrize("instrument,midi", [("guitar", 64), ("bass", 40)])
def test_manual_note_targets_existing_tab_and_keeps_standard_staff_synced(tmp_path, instrument, midi):
    settings = make_settings(tmp_path)
    job = create_stem_job(settings)
    with client_for(settings) as client:
        assert client.put(f"/api/jobs/{job}/score/tablature", json={"guitar": "full_mix", "bass": "bass"}).status_code == 200
        build(client, job)
        response = add(client, job, 0, addition(target={"partId": instrument + "-tab"}, midiNote=midi, measureIndex=1, beat=2, durationBeats=0.5))
        assert response.status_code == 200, response.text
        details = saved(client, job)
        manual = next(note for note in notes_of(details) if note.get("origin") == "musician")
        assert manual["tab"]["instrument"] == instrument and manual["tab"]["status"] == "assigned"
        xml = client.get(f"/api/jobs/{job}/score/saved/download?format=musicxml&part={instrument}-tab")
        assert xml.status_code == 200
        root = ET.fromstring(xml.content)
        assert root.find(".//note[staff='1']") is not None and root.find(".//note[staff='2']/notations/technical/fret") is not None


def test_add_edit_delete_undo_redo_reset_rebuild_and_restart_keep_identity(prediction):
    settings, job, _document = prediction
    with client_for(settings) as client:
        assert add(client, job, 0, addition()).status_code == 200
        identity = next(n["id"] for n in notes_of(saved(client, job)) if n.get("origin") == "musician")
        assert add(client, job, 1, {"op": "set_pitch", "target": {"noteId": identity}, "midiNote": 65}).status_code == 200
        assert add(client, job, 2, {"op": "set_note_timing", "target": {"noteId": identity}, "measureIndex": 1, "beat": 0, "durationBeats": 1}).status_code == 200
        build(client, job, force=True)
        manual = next(n for n in notes_of(saved(client, job)) if n["id"] == identity)
        assert manual["midiNote"] == 65 and manual["quantizedBeat"] == 4 and manual["confidence"] is None
        assert add(client, job, 3, {"op": "delete_note", "target": {"noteId": identity}}).status_code == 200
        assert identity not in {n["id"] for n in notes_of(saved(client, job))}
        for action, revision in (("undo", 4), ("redo", 5), ("undo", 6), ("reset", 7), ("undo", 8)):
            assert client.post(f"/api/jobs/{job}/score/corrections/{action}", json={"expectedRevision": revision}).status_code == 200
        assert identity in {n["id"] for n in notes_of(saved(client, job))}
        stale = add(client, job, 0, addition())
        assert stale.status_code == 409
    with client_for(settings) as client:
        assert [n["id"] for n in notes_of(saved(client, job)) if n.get("origin") == "musician"] == [identity]


def test_frontend_entry_uses_native_fields_and_current_revision():
    detail = copy.deepcopy(EDIT_DETAIL)
    detail["noteEntryParts"] = [{"id": "musician-notes", "name": "Musician-entered notes"}]
    result = _run_node(f'''
t.setDetail("edit",{json.dumps(detail)});
await t.getListener("#jobs","click")({{target:{{closest:()=>({{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}})}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)}),key=html.match(/data-action="correction-add-note"[^>]*data-input-key="([^"]+)"/)[1];
for(const [field,value] of [["part","musician-notes"],["pitch","4"],["octave","4"],["bar","1"],["beat","1.25"],["duration","1.5"]])t.getElement(`[data-note-entry-input="${{key}}-${{field}}"]`).value=value;
const requested=[];t.setFetch(async(url,options={{}})=>{{requested.push({{url,body:options.body||null}});return {{ok:true,status:200,json:async()=>({json.dumps(detail)})}};}});
await t.getListener("#jobs","click")({{target:{{closest:()=>({{dataset:{{action:"correction-add-note",jobId:"edit",inputKey:key}}}})}}}});
console.log(JSON.stringify({{html,requested}}));
''')
    assert "Add a missed note" in result["html"] and 'aria-label="Pitch for new note"' in result["html"]
    assert 'aria-label="Octave for new note"' in result["html"]
    assert json.loads(result["requested"][0]["body"]) == {"expectedRevision": 3, "operation": addition()}


def test_old_unknown_raw_source_named_musician_keeps_generic_part(prediction):
    _, _, document = prediction
    for note in notes_of(document):
        note["sourceKind"] = "musician"
    document["scoreParts"] = plan_score_parts(notes_of(document))
    assert [part["id"] for part in document["scoreParts"]] == ["other-lines"]
    validate_score_artifact(document)
    xml = ET.fromstring(score_to_musicxml_text(score_export_document(document)))
    assert xml.findtext("part-list/score-part/part-name") == "Other pitched lines (draft)"


@pytest.mark.parametrize("schema", [1, 2, 3, 4])
def test_legacy_predictions_accept_typed_additions_without_migrating_files(prediction, schema):
    _, _, document = prediction
    document["schemaVersion"] = schema
    document.pop("scoreParts")
    if schema < 4:
        document.pop("tonality")
    if schema < 3:
        document.pop("tablature")
        for key in ("tabNotes", "fingeredTabNotes"):
            document["counts"].pop(key)
        for note in notes_of(document):
            note.pop("tab")
        document["layers"]["tablature"] = {"status": "omitted", "note": "Legacy."}
    if schema < 2:
        document.pop("percussion")
        for measure in document["measures"]:
            measure.pop("percussionHits")
        for key in ("percussionHits", "notatedPercussionHits", "collapsedPercussionHits", "unresolvedPercussionHits", "offGridPercussionHits", "unplacedPercussionHits"):
            document["counts"].pop(key)
        document["layers"]["percussion"] = {"status": "omitted", "note": "Legacy."}
    validate_score_artifact(document)
    changed, report = apply_corrections(document, [new_operation(addition(target={"partId": "combined"}))])
    validate_score_artifact(changed, manual_note_ids=frozenset(report["addedNotes"]))
    assert len(report["addedNotes"]) == 1 and changed["schemaVersion"] == schema
    assert score_to_midi_bytes(score_export_document(changed))
    assert "<score-partwise" in score_to_musicxml_text(score_export_document(changed))


def test_default_choices_survive_deleting_all_predictions(prediction):
    settings, job, document = prediction
    with client_for(settings) as client:
        original_choices = saved(client, job)["noteEntryParts"]
        for revision, note in enumerate(notes_of(document)):
            assert add(client, job, revision, {"op": "delete_note", "target": {"noteId": note["id"]}}).status_code == 200
        empty = saved(client, job)
        assert empty["counts"]["notes"] == 0 and empty["noteEntryParts"] == original_choices
        target = next(part["id"] for part in original_choices if part["id"] != "musician-notes")
        assert add(client, job, len(notes_of(document)), addition(target={"partId": target})).status_code == 200
        assert saved(client, job)["counts"]["notes"] == 1


def test_export_voice_limit_refuses_write_and_keeps_last_working_view(prediction):
    settings, job, _document = prediction
    with client_for(settings) as client:
        for revision in range(15):
            response = add(client, job, revision, addition(midiNote=72))
            assert response.status_code == 200, response.text
        before = saved(client, job)
        previous_midi = client.get(f"/api/jobs/{job}/score/saved/download?format=midi").content
        response = add(client, job, 15, addition(midiNote=72))
        assert response.status_code == 422
        after = saved(client, job)
        assert after["corrections"]["revision"] == 15 and after["measures"] == before["measures"]
        assert client.get(f"/api/jobs/{job}/score/saved/download?format=midi").content == previous_midi
        # A later valid prediction can make an old correction overlay exceed
        # the MIDI limit. Reads must fail back to the honest original, not 500.
        raw = drum_raw_payload()
        event = raw["pitchedNoteEvents"][0]
        event.update(midiNote=72, midiPitch=72.0, noteName="C5", frequencyHz=440 * 2 ** (3 / 12))
        raw["createdAt"] = "2026-10-16T05:00:00+00:00"
        write_raw_transcription(job, settings, raw)
        db.update_job(settings.database_path, job, transcribed_at=raw["createdAt"])
        build(client, job, force=True)
        rebuilt = saved(client, job)
        assert rebuilt["corrections"]["unusable"] is True
        assert not any(note.get("origin") == "musician" for note in notes_of(rebuilt))
        assert client.get(f"/api/jobs/{job}/score/saved/download?format=midi").status_code == 200


def test_raw_and_manual_id_collision_never_replaces_a_prediction(prediction):
    _, _, document = prediction
    operation = new_operation(addition())
    first = notes_of(document)[0]
    first["id"] = "user:" + operation["id"]
    before = copy.deepcopy(document)
    expected, _ = apply_corrections(document, [])  # normal replay sorts tied IDs
    changed, report = apply_corrections(document, [operation])
    assert changed == expected and document == before
    assert not report["applied"] and "already" in report["notApplicable"][0]["reason"]


def test_frontend_draft_is_preserved_and_vanished_part_requires_selection():
    detail = copy.deepcopy(EDIT_DETAIL)
    detail["noteEntryParts"] = [{"id": "pitched-lines", "name": "Pitched line"}]
    result = _run_node(f'''
t.setDetail("edit",{json.dumps(detail)});
await t.getListener("#jobs","click")({{target:{{closest:()=>({{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}})}}}});
for(const [field,value] of [["part","pitched-lines"],["bar","7"]])t.getListener("#jobs","input")({{target:{{dataset:{{entryJob:"edit",entryField:field}},value}}}});
const kept=t.renderScore({json.dumps(EDIT_JOB)});
const changed={json.dumps(detail)};changed.noteEntryParts=[{{id:"musician-notes",name:"Musician-entered notes"}}];t.setDetail("edit",changed);
const html=t.renderScore({json.dumps(EDIT_JOB)});console.log(JSON.stringify({{kept,html}}));
''')
    assert 'value="7" required' in result["kept"]
    assert '<option value="" selected>Choose a current part</option>' in result["html"]


def test_manual_note_cross_bar_musicxml_schema(prediction):
    schema_path = os.environ.get("POPEX_MUSICXML_XSD")
    if not schema_path:
        pytest.skip("Set POPEX_MUSICXML_XSD to the pinned official MusicXML schema.")
    etree = pytest.importorskip("lxml.etree")
    _, _, document = prediction
    changed, report = apply_corrections(document, [new_operation(addition(beat=3.75, durationBeats=1.5))])
    validate_score_artifact(changed, manual_note_ids=frozenset(report["addedNotes"]))
    schema = etree.XMLSchema(etree.parse(schema_path))
    for part in (None, "musician-notes"):
        xml = score_to_musicxml_text(score_export_document(changed), only_part=part)
        schema.assertValid(etree.fromstring(xml.encode("utf-8")))


def test_addition_outside_preview_still_has_all_edit_controls():
    detail = copy.deepcopy(EDIT_DETAIL)
    note = {**detail["measures"][0]["notes"][0], "id": "user:" + "a" * 32,
            "origin": "musician", "confidence": None, "tab": None, "quantizedBeat": 280}
    detail["measures"] = [{"measureIndex": index, "startSeconds": index * 2, "endSeconds": index * 2 + 2,
                           "notes": [note] if index == 70 else [], "chordSymbol": None, "harmony": [], "percussionHits": []}
                          for index in range(80)]
    detail["counts"]["measures"] = 80
    detail["noteEntryParts"] = [{"id": "musician-notes", "name": "Musician-entered notes"}]
    result = _run_node(f'''
t.setDetail("edit",{json.dumps(detail)});
await t.getListener("#jobs","click")({{target:{{closest:()=>({{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}})}}}});
console.log(JSON.stringify({{html:t.renderScore({json.dumps(EDIT_JOB)})}}));
''')
    html = result["html"]
    assert "Musician notes (1)" in html and "bar 71" in html
    assert f'data-note-id="{note["id"]}"' in html
    assert "Added by musician" in html and "Save timing" in html
    assert "low confidence" not in html


def test_initial_tab_entry_avoids_a_string_held_by_an_earlier_note(tmp_path):
    settings = make_settings(tmp_path)
    job = create_stem_job(settings)
    with client_for(settings) as client:
        build(client, job)
    document = stored_document(settings, job)
    operation = new_operation(addition(target={"partId": "bass-tab"}, midiNote=28,
                                       beat=0.5, durationBeats=0.25))
    changed, report = apply_corrections(document, [operation])
    note = next(item for item in notes_of(changed) if item.get("origin") == "musician")
    assert note["tab"] == {"instrument": "bass", "status": "unplayable", "string": None, "fret": None}
    validate_score_artifact(changed, manual_note_ids=frozenset(report["addedNotes"]))


def test_total_note_limit_still_applies_to_authorized_manual_projection(prediction):
    _, _, document = prediction
    template = copy.deepcopy(notes_of(document)[0])
    for index, measure in enumerate(document["measures"]):
        measure["notes"] = [{**template, "id": f"budget-{index}-{i}", "quantizedBeat": index * 4.0,
                             "quantizedEndBeat": index * 4.0 + 2.0} for i in range(2500)]
    base, _ = apply_corrections(document, [])
    validate_score_artifact(base)
    changed, report = apply_corrections(base, [new_operation(addition())])
    with pytest.raises(ScoreArtifactError):
        validate_score_artifact(changed, manual_note_ids=frozenset(report["addedNotes"]))


def test_add_note_unlocks_before_restoring_focus_and_blocks_duplicate_pending_clicks():
    detail = copy.deepcopy(EDIT_DETAIL)
    detail["noteEntryParts"] = [{"id": "musician-notes", "name": "Musician-entered notes"}]
    result = _run_node(f'''
t.setDetail("edit",{json.dumps(detail)});
await t.getListener("#jobs","click")({{target:{{closest:()=>({{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}})}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)}),key=html.match(/data-action="correction-add-note"[^>]*data-input-key="([^"]+)"/)[1];
for(const [field,value] of [["part","musician-notes"],["pitch","4"],["octave","4"],["bar","1"],["beat","1.25"],["duration","1.5"]])t.getElement(`[data-note-entry-input="${{key}}-${{field}}"]`).value=value;
const current=t.getElement(`[data-focus-key="${{key}}-save"]`);current.disabled=true;current.focus=function(){{if(!this.disabled)this.focused=true;}};
let release,posts=0;const gate=new Promise(resolve=>release=resolve);
t.setFetch(async(url,options={{}})=>{{if(options.method==="POST"){{posts++;await gate;}}return {{ok:true,status:200,json:async()=>({json.dumps(detail)})}};}});
const button={{dataset:{{action:"correction-add-note",jobId:"edit",inputKey:key}}}},click=()=>t.getListener("#jobs","click")({{target:{{closest:()=>button}}}});
const first=click();await click();release();await first;
console.log(JSON.stringify({{posts,disabled:current.disabled,focused:current.focused,busy:current.attributes["aria-busy"]}}));
''')
    assert result == {"posts": 1, "disabled": False, "focused": True, "busy": "false"}
