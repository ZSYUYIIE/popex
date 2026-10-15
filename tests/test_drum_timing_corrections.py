"""Drum placement edits preserve the separate raw percussion prediction."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import xml.etree.ElementTree as ET

import pytest

from app.score_artifacts import validate_score_artifact
from app.score_construction import score_to_midi_bytes, score_to_musicxml_text
from app.score_corrections import CorrectionError, apply_corrections, new_operation, validate_operation
from app.score_pipeline import score_export_document
from test_frontend_score import _run_node
from test_harmony_api import make_settings
from test_score_corrections import EDIT_DETAIL, EDIT_JOB, add, build, client_for, hits_of, saved, stored_document
from test_score_percussion import _midi_channel_events, create_drum_job


def hit_timing(event_id="d_crash_2", hit_index=0, **overrides):
    return {"op": "set_hit_timing", "target": {"eventId": event_id, "hitIndex": hit_index},
            "measureIndex": 0, "beat": 3.5, **overrides}


def test_move_collapsed_hit_keeps_raw_evidence_and_prediction(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = stored_document(settings, job_id)
    pristine = copy.deepcopy(document)
    corrected, report = apply_corrections(document, [new_operation(hit_timing())])
    before = next(h for h in hits_of(document) if h["eventId"] == "d_crash_2")
    after = next(h for h in hits_of(corrected) if h["eventId"] == "d_crash_2")
    assert before["notation"] == "collapsed" and after["notation"] == "notated"
    assert after["quantizedBeat"] == 3.5
    assert {k: v for k, v in before.items() if k not in {"quantizedBeat", "notation"}} == {
        k: v for k, v in after.items() if k not in {"quantizedBeat", "notation"}}
    assert document == pristine
    assert report["hitTiming"]["d_crash_2#0"] == before["quantizedBeat"]
    validate_score_artifact(corrected)


@pytest.fixture
def prediction(tmp_path):
    settings = make_settings(tmp_path)
    job_id = create_drum_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    return settings, job_id, stored_document(settings, job_id)


@pytest.mark.parametrize("overrides", [
    {"beat": True}, {"beat": float("nan")}, {"beat": float("inf")}, {"beat": "1"},
    {"beat": -0.5}, {"beat": 0.25}, {"beat": 0.1}, {"beat": 12}, {"beat": 10**1000},
    {"measureIndex": False}, {"measureIndex": -1}, {"measureIndex": 2048},
    {"measureIndex": 0.5}, {"durationBeats": 0.5},
    {"target": {"eventId": "d", "hitIndex": True}},
    {"target": {"eventId": "d", "hitIndex": 64}},
    {"target": {"noteId": "d"}},
])
def test_invalid_drum_timing_rejected(overrides):
    with pytest.raises(CorrectionError):
        validate_operation(hit_timing(**overrides))


def test_collision_revival_edit_back_and_voice_report_compatibility(prediction):
    _, _, document = prediction
    original_counts = copy.deepcopy(document["counts"])
    # Moving the first cymbal onto the kick's first hat preserves both voices.
    move = new_operation(hit_timing("d_crash_1", beat=0))
    changed, report = apply_corrections(document, [move])
    assert len(hits_of(changed)) == len(hits_of(document))
    assert changed["counts"]["notatedPercussionHits"] == original_counts["notatedPercussionHits"] + 1
    assert report["hits"] == {} and report["hitTiming"] == {"d_crash_1#0": 5}
    validate_score_artifact(changed)
    # Two cymbals in one slot keep every identity, but one is collapsed.
    collision = new_operation(hit_timing("d_crash_2", beat=0))
    changed, report = apply_corrections(document, [move, collision])
    cymbals = [h for h in hits_of(changed) if h["broadVoice"] == "cymbal_like"]
    assert len(cymbals) == 2 and sum(h["notation"] == "notated" for h in cymbals) == 1
    assert changed["counts"]["collapsedPercussionHits"] == original_counts["collapsedPercussionHits"]
    validate_score_artifact(changed)
    back = new_operation(hit_timing("d_crash_1", measureIndex=1, beat=1))
    restored, report = apply_corrections(document, [move, back])
    assert report["hitTiming"] == {} and restored["measures"] == document["measures"]
    # Voice markers remain strings for existing clients, independent of timing.
    voice = new_operation({"op": "set_drum_voice", "target": {"eventId": "d_crash_1", "hitIndex": 0},
                           "broadVoice": "mid_drum"})
    changed, report = apply_corrections(document, [move, voice])
    assert report["hits"]["d_crash_1#0"] == "cymbal_like"
    assert report["hitTiming"]["d_crash_1#0"] == 5
    validate_score_artifact(changed)


def test_rebuild_extent_target_loss_and_moved_then_delete(prediction):
    _, _, document = prediction
    operations = [new_operation(hit_timing(measureIndex=len(document["measures"]))),
                  new_operation(hit_timing(beat=4)), new_operation(hit_timing("missing"))]
    unchanged, report = apply_corrections(document, operations)
    assert len(report["notApplicable"]) == 3 and not report["applied"]
    assert unchanged["measures"] == document["measures"]
    move = new_operation(hit_timing())
    delete = new_operation({"op": "delete_hit", "target": {"eventId": "d_crash_2", "hitIndex": 0}})
    changed, report = apply_corrections(document, [move, delete])
    assert not any(h["eventId"] == "d_crash_2" for h in hits_of(changed))
    assert report["deletedHits"][0]["measureIndex"] == 0
    assert report["deletedHits"][0]["quantizedBeat"] == 3.5
    validate_score_artifact(changed)
    smaller = copy.deepcopy(document)
    smaller["measures"] = smaller["measures"][:1]
    _, report = apply_corrections(smaller, [move])
    assert report["notApplicable"] and not report["applied"]
    assert apply_corrections(document, [move]) == apply_corrections(document, [move])


def test_corrected_drum_exports_use_exact_new_placement(prediction):
    _, _, document = prediction
    changed, _ = apply_corrections(document, [new_operation(hit_timing())])
    exported = score_export_document(changed)
    events = _midi_channel_events(score_to_midi_bytes(exported))
    assert any(tick == 1680 and status == 0x99 and pitch == 49 for tick, status, pitch, _ in events)
    original_events = _midi_channel_events(score_to_midi_bytes(score_export_document(document)))
    assert not any(tick == 1680 and status == 0x99 and pitch == 49 for tick, status, pitch, _ in original_events)
    original_xml = ET.fromstring(score_to_musicxml_text(score_export_document(document)))
    xml = ET.fromstring(score_to_musicxml_text(exported))
    original_first_bar = original_xml.find("part[@id='P2']/measure")
    first_bar = xml.find("part[@id='P2']/measure")
    def cymbals(bar):
        return [n for n in bar.findall("note") if n.findtext("unpitched/display-step") == "A"]
    assert len(cymbals(original_first_bar)) == 0 and len(cymbals(first_bar)) == 1


def test_hit_index_targets_one_sibling_and_keeps_raw_warning_counts(prediction):
    _, _, document = prediction
    changed, _ = apply_corrections(document, [new_operation(hit_timing("d_kick_1", 1))])
    siblings = {h["hitIndex"]: h for h in hits_of(changed) if h["eventId"] == "d_kick_1"}
    assert siblings[0]["quantizedBeat"] == 0 and siblings[1]["quantizedBeat"] == 3.5
    for key in ("offGridPercussionHits", "unplacedPercussionHits", "unresolvedPercussionHits"):
        assert changed["counts"][key] == document["counts"][key]
    validate_score_artifact(changed)


def test_drum_timing_form_keys_do_not_alias_ids_or_hit_indexes():
    detail = copy.deepcopy(EDIT_DETAIL)
    hit = detail["measures"][0]["percussionHits"][0]
    detail["measures"][0]["percussionHits"] = [dict(hit, eventId=value, hitIndex=index)
        for value, index in (("a:b", 0), ("a-b", 0), ("a.b", 0), ("a:b", 1))]
    result = _run_node(f'''
t.setDetail("edit", {json.dumps(detail)});
t.renderScore({json.dumps(EDIT_JOB)});
const toggle={{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>toggle}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)});
console.log(JSON.stringify({{keys:[...html.matchAll(/data-action="correction-hit-timing"[^>]*data-input-key="([^"]+)"/g)].map(m=>m[1])}}));
''')
    assert len(result["keys"]) == len(set(result["keys"])) == 4


def test_api_undo_redo_reset_restart_rebuild_and_prediction_hash(prediction):
    settings, job_id, _ = prediction
    hashes = {p: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (settings.exports_dir/job_id).rglob("*") if p.is_file()}
    with client_for(settings) as client:
        response = add(client, job_id, 0, hit_timing())
        assert response.status_code == 200, response.text
        assert response.json()["review"]["hitTiming"] == {"d_crash_2#0": 5}
        assert add(client, job_id, 0, hit_timing()).status_code == 409
        assert add(client, job_id, 1, hit_timing(beat=4)).status_code == 422
        assert add(client, job_id, 1, hit_timing(beat=0.25)).status_code == 422
        assert saved(client, job_id)["corrections"]["revision"] == 1
        assert client.post(f"/api/jobs/{job_id}/score/corrections/undo", json={"expectedRevision": 1}).status_code == 200
        assert saved(client, job_id)["measures"] == saved(client, job_id, "original")["measures"]
        assert client.post(f"/api/jobs/{job_id}/score/corrections/redo", json={"expectedRevision": 2}).status_code == 200
        for path, digest in hashes.items():
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    with client_for(settings) as client:
        assert saved(client, job_id)["corrections"]["review"]["hitTiming"] == {"d_crash_2#0": 5}
        build(client, job_id, force=True)
        assert saved(client, job_id)["corrections"]["appliedCount"] == 1
        assert client.post(f"/api/jobs/{job_id}/score/corrections/reset", json={"expectedRevision": 3}).status_code == 200
        assert saved(client, job_id)["measures"] == saved(client, job_id, "original")["measures"]


def test_frontend_timing_labels_report_and_revision_checked_save():
    detail = copy.deepcopy(EDIT_DETAIL)
    detail["corrections"]["review"]["hitTiming"] = {"d1#0": 1.5}
    result = _run_node(f'''
t.setDetail("edit", {json.dumps(detail)});
const ordinary=t.renderScore({json.dumps(EDIT_JOB)});
const toggle={{dataset:{{action:"score-edit-toggle",jobId:"edit"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>toggle}}}});
const html=t.renderScore({json.dumps(EDIT_JOB)});
const key=html.match(/data-action="correction-hit-timing"[^>]*data-input-key="([^"]+)"/)[1];
for(const [field,value] of [["bar","1"],["beat","4.5"]])
  t.getElement(`[data-hit-timing-input="${{key}}-${{field}}"]`).value=value;
const requested=[];
t.setFetch(async(url,options={{}})=>{{requested.push({{url,body:options.body||null}});return {{ok:true,status:200,json:async()=>({json.dumps(detail)})}};}});
const button={{dataset:{{action:"correction-hit-timing",jobId:"edit",eventId:"d1",hitIndex:"0",inputKey:key,focusKey:"save"}}}};
await t.getListener("#jobs","click")({{target:{{closest:()=>button}}}});
console.log(JSON.stringify({{html,ordinary,requested}}));
''')
    assert 'aria-label="Bar for Low Drum at beat 1"' in result["html"]
    assert 'aria-label="Beat for Low Drum at beat 1"' in result["html"]
    assert 'data-open-key="hit-timing-' in result["html"]
    assert 'timing edited · was bar 1, beat 2.5' in result["html"]
    assert 'timing edited · was bar 1, beat 2.5' in result["ordinary"]
    assert json.loads(result["requested"][0]["body"]) == {
        "expectedRevision": 3, "operation": hit_timing("d1")}


@pytest.mark.skipif(not os.environ.get("POPEX_MUSICXML_XSD"), reason="set POPEX_MUSICXML_XSD")
def test_drum_timing_musicxml_schema(prediction):
    from lxml import etree
    _, _, document = prediction
    changed, _ = apply_corrections(document, [new_operation(hit_timing())])
    schema = etree.XMLSchema(etree.parse(os.environ["POPEX_MUSICXML_XSD"]))
    tree = etree.fromstring(score_to_musicxml_text(score_export_document(changed)).encode())
    assert schema.validate(tree), schema.error_log
