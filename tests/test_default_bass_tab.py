"""Default bass tablature without stems fingers the full mix's lowest line (cycle 40)."""

from __future__ import annotations

from pathlib import Path

import pytest

from app import db
from app.score_artifacts import ScoreArtifactValidationError, validate_score_artifact
from app.score_sources import tablature_line
from app.tablature import tab_position_is_consistent
from app.transcription_events import write_raw_transcription
from test_harmony_api import create_job, make_settings, raw_payload
from test_score_tablature import _load, build, client_for, notes_of


def _event(index: int, kind: str, midi: int) -> dict:
    return {
        "id": f"{kind}_{index}",
        "sourceKind": kind,
        "startSeconds": index * 0.5,
        "endSeconds": index * 0.5 + 0.45,
        "midiNote": midi,
        "midiPitch": float(midi),
        "frequencyHz": 440.0 * 2 ** ((midi - 69) / 12),
        "noteName": "X",
        "confidence": 0.85,
        "warnings": [],
    }


def create_mix_lines_job(settings) -> str:
    job_id = create_job(settings)
    payload = raw_payload()
    events = [_event(i, "mix_melody", midi) for i, midi in enumerate([72, 74, 76, 77, 79, 77])]
    events += [_event(i, "mix_bass", midi) for i, midi in enumerate([28, 33, 35, 36, 40, 43])]
    events += [_event(i, "mix_chords", midi) for i, midi in enumerate([60, 64, 62, 65, 67, 64])]
    payload["pitchedNoteEvents"] = events
    payload["alignmentCandidates"] = [
        {"eventId": event["id"], "eventType": "pitched", "rawTimeSeconds": event["startSeconds"], "confidence": 0.0}
        for event in events
    ]
    write_raw_transcription(job_id, settings, payload)
    db.update_job(settings.database_path, job_id, pitched_event_count=len(events))
    return job_id


def test_the_bass_line_falls_back_only_without_a_bass_stem():
    default = {"bass": "bass", "guitar": None}
    assert tablature_line("bass", default, {"mix_bass", "mix_melody"}) == "mix_bass"
    assert tablature_line("bass", default, {"bass", "mix_bass"}) == "bass"
    assert tablature_line("bass", default, {"full_mix"}) == "bass"
    assert tablature_line("bass", {"bass": "vocals", "guitar": None}, {"mix_bass"}) == "vocals"
    assert tablature_line("bass", {"bass": "bass", "guitar": "mix_bass"}, {"mix_bass"}) == "bass"
    assert tablature_line("guitar", {"bass": None, "guitar": "bass"}, {"mix_bass"}) == "bass"


def test_without_stems_the_default_bass_tab_fingers_the_lowest_line(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_mix_lines_job(settings)
    with client_for(settings) as client:
        details = build(client, job_id)
    tablature = details["tablature"]
    assert tablature["origin"] == "default"
    assert tablature["request"] == {"bass": "bass", "guitar": None}
    bass = tablature["instruments"][0]
    assert bass["sourceKind"] == "mix_bass" and bass["noteCount"] == 6
    assert details["layers"]["tablature"]["status"] == "included"
    for note in notes_of(details):
        if note["sourceKind"] == "mix_bass":
            assert note["tab"]["instrument"] == "bass"
            if note["tab"]["status"] == "assigned":
                assert tab_position_is_consistent(note["midiNote"], "bass", note["tab"]["string"], note["tab"]["fret"])
        else:
            assert note["tab"] is None
    assert any("no separated bass stem, so bass tablature fingers the full-mix lowest line" in warning
               for warning in details["warnings"])
    assert "bass-line" not in [part["id"] for part in details["scoreParts"]]


def test_a_tampered_fallback_line_is_rejected(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_mix_lines_job(settings)
    with client_for(settings) as client:
        build(client, job_id)
    document = _load(settings, job_id)
    validate_score_artifact(document)
    document["tablature"]["instruments"][0]["sourceKind"] = "mix_chords"
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(document)


def test_deleting_every_lowest_line_note_keeps_the_corrected_view_valid(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_mix_lines_job(settings)
    with client_for(settings) as client:
        details = build(client, job_id)
        base = f"/api/jobs/{job_id}/score/corrections"
        revision = client.get(base).json()["revision"]
        for note in notes_of(details):
            if note["sourceKind"] == "mix_bass":
                operation = {"op": "delete_note", "target": {"noteId": note["id"]}}
                response = client.post(base, json={"expectedRevision": revision, "operation": operation})
                assert response.status_code == 200, response.text
                revision = response.json()["revision"]
        corrected = client.get(f"/api/jobs/{job_id}/score/saved?includeMeasures=true")
        download = client.get(f"/api/jobs/{job_id}/score/saved/download?format=musicxml")
    assert corrected.status_code == 200 and download.status_code == 200
    body = corrected.json()
    assert not any(note["sourceKind"] == "mix_bass" for note in notes_of(body))
    assert body["tablature"]["instruments"][0]["noteCount"] == 0
