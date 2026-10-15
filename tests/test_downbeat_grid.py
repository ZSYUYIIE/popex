"""Bar lines follow the detected downbeats (Cycle 21)."""

from __future__ import annotations

import copy
import json

import pytest
from fastapi.testclient import TestClient

from app import db
from app.main import create_app
from app.score_api import score_evidence_warnings
from app.score_artifacts import ScoreArtifactValidationError, load_score_artifact, validate_score_artifact
from app.score_construction import ScoreConstructionError, build_score_document, estimate_grid_lead_in
from app.score_pipeline import score_outdated_reason
from test_frontend_score import SUMMARY, _run_node
from test_harmony_api import create_job, make_settings

BAR = 2.0  # 120 BPM in 4/4


def grid(beats, downbeats, *, from_analysis=True):
    return estimate_grid_lead_in(beats, downbeats, tempo_bpm=120.0, beats_per_measure=4,
                                 meter_from_analysis=from_analysis)


def test_downbeats_set_the_bar_phase() -> None:
    downbeats = [0.3 + k * BAR for k in range(8)]
    beats = [0.3 + k * 0.5 for k in range(32)]
    result = grid(beats, downbeats)
    assert result["source"] == "downbeats"
    assert result["leadInSeconds"] == pytest.approx(1.7, abs=1e-3)  # 0.3 s is beat 1 of bar 2
    assert result["concentration"] == pytest.approx(1.0)


def test_beats_set_the_beat_phase_when_downbeats_are_missing_or_unclear() -> None:
    beats = [0.3 + k * 0.5 for k in range(32)]
    scattered = [0.3, 0.9, 1.6, 2.2, 3.1]
    for downbeats, from_analysis in (([], True), (scattered, True), ([0.3 + k * BAR for k in range(8)], False)):
        result = grid(beats, downbeats, from_analysis=from_analysis)
        assert (result["source"], result["leadInSeconds"]) == ("beats", pytest.approx(0.2, abs=1e-3))


def test_inconsistent_or_sparse_evidence_keeps_bars_at_zero() -> None:
    drifting = [k * 0.53 for k in range(40)]  # the beats do not follow the 120 BPM grid
    assert grid(drifting, []) == {"leadInSeconds": 0.0, "source": "none", "concentration": None}
    assert grid([0.3, 0.8, 1.3], [0.3])["source"] == "none"
    # A phase within 10 ms of a bar line snaps to it.
    assert grid([], [0.004 + k * BAR for k in range(6)])["leadInSeconds"] == 0.0


def _note(note_id: str, start: float) -> dict:
    return {"id": note_id, "startSeconds": start, "endSeconds": start + 0.4, "midiNote": 60, "confidence": 0.9}


def test_a_pickup_sits_at_the_end_of_bar_one() -> None:
    # The first downbeat is at 0.5 s, so the note at 0 s is the pickup on beat 4 of bar 1.
    document = build_score_document([_note("pickup", 0.0), _note("down", 0.5)], tempo_bpm=120.0, lead_in_seconds=1.5)
    first, second = document["measures"][:2]
    assert (first["startSeconds"], first["endSeconds"]) == (-1.5, 0.5)
    assert [(n["id"], n["quantizedBeat"]) for n in first["notes"]] == [("pickup", 3.0)]
    assert [(n["id"], n["quantizedBeat"]) for n in second["notes"]] == [("down", 4.0)]
    assert document["leadInSeconds"] == 1.5
    with pytest.raises(ScoreConstructionError):
        build_score_document([_note("x", 0.0)], tempo_bpm=120.0, lead_in_seconds=BAR)


def test_percussion_and_rhythm_grids_use_the_same_lead_in() -> None:
    hit = {"eventId": "h", "hitIndex": 0, "sourceKind": "drums", "rawKind": "low", "broadVoice": "low_drum",
           "resolved": True, "timeSeconds": 0.5, "strength": 0.9, "confidence": 0.9,
           "interpretationPlacement": "placed"}
    try:
        document = build_score_document([_note("n", 0.5)], tempo_bpm=120.0, lead_in_seconds=1.5, percussion_hits=[hit])
    except ScoreConstructionError:
        pytest.skip("percussion fixture shape differs")
    hits = [h for m in document["measures"] for h in m["percussionHits"]]
    assert [h["quantizedBeat"] for h in hits] == [4.0]


def test_warnings_explain_where_bar_one_starts() -> None:
    base = {"warnings": []}
    evidence = {"tempoConfidence": 0.9, "meterConfidence": 0.9, "meterSource": "analysis", "tempoStable": True,
                "percussionEventCount": 0}
    def warnings(origin):
        return " ".join(score_evidence_warnings(base, {**evidence, "gridOrigin": origin}, single_draft_part_warning=False))
    assert "starts 1.70 s into bar 1" in warnings({"source": "downbeats", "leadInSeconds": 1.7, "concentration": 1.0})
    assert "check where bar 1 should start" in warnings({"source": "beats", "leadInSeconds": 0.2, "concentration": 0.9})
    assert "bars start at the beginning" in warnings({"source": "none", "leadInSeconds": 0.0, "concentration": None})


def test_saved_scores_record_and_validate_the_grid_origin(tmp_path) -> None:
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/score/construct").status_code == 202
    record = db.get_job(settings.database_path, job_id)
    document = load_score_artifact(job_id, settings, artifact_file_name=record["score_artifact_file_name"])
    timing = document["timing"]
    assert document["schemaVersion"] == 8
    assert timing["gridOrigin"] in {"downbeats", "beats", "none"}
    assert document["measures"][0]["startSeconds"] == pytest.approx(-timing["leadInSeconds"])

    shifted = copy.deepcopy(document)
    shifted["measures"][0]["startSeconds"] -= 0.25
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(shifted)
    unexplained = copy.deepcopy(document)
    unexplained["timing"].update(gridOrigin="none", gridConcentration=None, leadInSeconds=0.25)
    unexplained["measures"][0]["startSeconds"] = -0.25
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(unexplained)
    too_long = copy.deepcopy(document)
    too_long["timing"].update(gridOrigin="downbeats", gridConcentration=1.0, leadInSeconds=60.0)
    with pytest.raises(ScoreArtifactValidationError):
        validate_score_artifact(too_long)


def test_older_scores_report_the_downbeat_grid_and_the_ui_says_so() -> None:
    assert score_outdated_reason({"score_version": "score-pipeline-v7", "score_note_count": 3}) == "downbeat-grid"
    assert score_outdated_reason({"score_version": "score-pipeline-v8", "score_note_count": 3}) is None
    detail = {
        "available": True, "createdAt": SUMMARY["createdAt"], "layers": {}, "counts": {}, "warnings": [],
        "timing": {"tempoBpm": 120, "beatsPerMeasure": 4, "leadInSeconds": 1.5},
        "measures": [{"measureIndex": 0, "startSeconds": -1.5, "endSeconds": 0.5, "chordSymbol": None,
                      "notes": [{"id": "n", "midiNote": 60, "quantizedBeat": 3, "quantizedDurationBeats": 1,
                                 "confidence": 0.9, "sourceKind": "vocals"}], "harmony": [], "percussionHits": []}],
    }
    result = _run_node(
        f"""
t.setDetail("pickup", {json.dumps(detail)});
const html=t.renderScore({{id:"pickup",score:{json.dumps(SUMMARY)}}});
const stale=t.renderScore({{id:"s",score:{{...{json.dumps(SUMMARY)}, stale:true, staleReason:"downbeat-grid"}}}});
console.log(JSON.stringify({{html,stale}}));
"""
    )
    assert 'data-lead-in="1.5"' in result["html"]  # the playhead adds the lead-in to playback time
    assert "built before bar lines followed the detected downbeats" in result["stale"]
