"""Musician timing corrections remain separate from raw score evidence."""
from __future__ import annotations

import copy
from pathlib import Path

from app.score_corrections import apply_corrections, new_operation
from test_harmony_api import make_settings
from test_score_corrections import build, client_for, stored_document
from test_score_percussion import create_drum_job


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
