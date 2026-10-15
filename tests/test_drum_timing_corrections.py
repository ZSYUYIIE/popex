"""Drum placement edits preserve the separate raw percussion prediction."""
from __future__ import annotations

import copy

from app.score_artifacts import validate_score_artifact
from app.score_corrections import apply_corrections, new_operation
from test_harmony_api import make_settings
from test_score_corrections import build, client_for, hits_of, stored_document
from test_score_percussion import create_drum_job


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
