"""Missed notes are musician corrections, never fabricated raw predictions."""
from __future__ import annotations

from app.score_corrections import new_operation
from test_harmony_api import make_settings
from test_score_corrections import add, build, client_for, notes_of, saved, score_file_digest
from test_score_percussion import create_drum_job


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
