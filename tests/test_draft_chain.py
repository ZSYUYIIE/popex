from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import db
from app.draft_chain import DraftStep, DraftTracker, StepResult, wait_for_status
from app.main import create_app
from test_harmony_api import create_job, make_settings

JOB = "a" * 32


def recorder(outcomes: dict[str, StepResult], calls: list[tuple[str, bool]]):
    def make(key: str):
        def run(job_id: str, force: bool) -> StepResult:
            calls.append((key, force))
            return outcomes.get(key, StepResult("reused"))

        return run

    return make


def steps_for(outcomes, calls, required=("transcribe", "score")):
    make = recorder(outcomes, calls)
    return [
        DraftStep(key, key.title(), key in required, make(key))
        for key in ("separate", "transcribe", "interpret", "harmonize", "score")
    ]


def ready(job_id):
    return "ready", None


def run_chain(steps, readiness=ready, timeout=5.0, tracker=None):
    tracker = tracker or DraftTracker(sleep=lambda seconds: None)
    assert tracker.claim(JOB, steps)
    tracker.run(JOB, steps, readiness=readiness, timeout_seconds=timeout)
    return tracker.status(JOB)


# ---------------------------------------------------------------------------
# Chain logic


def test_finished_stages_are_reused_and_nothing_is_forced():
    calls: list = []
    status = run_chain(steps_for({}, calls))
    assert status["state"] == "completed"
    assert [force for _, force in calls] == [False] * 5
    assert [step["outcome"] for step in status["steps"]] == ["reused"] * 5
    assert status["message"] == "The draft score is ready to review."
    assert status["finishedAt"] is not None


def test_everything_after_a_stage_that_ran_is_run_again():
    calls: list = []
    status = run_chain(steps_for({"interpret": StepResult("ran")}, calls))
    assert calls == [
        ("separate", False),
        ("transcribe", False),
        ("interpret", False),
        ("harmonize", True),
        ("score", True),
    ]
    assert status["state"] == "completed"


def test_an_optional_failure_continues_and_is_explained():
    calls: list = []
    outcomes = {"harmonize": StepResult("failed", "Chords could not be found.")}
    status = run_chain(steps_for(outcomes, calls))
    assert status["state"] == "completed"
    assert [key for key, _ in calls][-1] == "score"
    assert "Chords could not be found." in status["message"]
    assert status["steps"][3] == {"key": "harmonize", "label": "Harmonize", "outcome": "failed", "note": "Chords could not be found."}


def test_a_required_failure_stops_the_chain_and_keeps_earlier_results():
    calls: list = []
    outcomes = {"transcribe": StepResult("failed", "Note and drum detection did not finish.")}
    status = run_chain(steps_for(outcomes, calls))
    assert status["state"] == "failed"
    assert status["step"] == "transcribe"
    assert [key for key, _ in calls] == ["separate", "transcribe"]
    assert "Earlier results are kept" in status["message"]
    assert [step["outcome"] for step in status["steps"]][2:] == [None, None, None]


def test_a_step_that_raises_counts_as_failed_without_escaping():
    def explode(job_id, force):
        raise RuntimeError("boom at C:/Users/someone")

    calls: list = []
    steps = steps_for({}, calls)
    steps[1] = DraftStep("transcribe", "Detecting notes", True, explode)
    status = run_chain(steps)
    assert status["state"] == "failed"
    assert "C:/Users" not in status["message"]
    assert status["steps"][1]["outcome"] == "failed"


def test_the_chain_waits_for_analysis_then_runs():
    answers = iter(["waiting", "waiting", "ready"])
    calls: list = []
    status = run_chain(steps_for({}, calls), readiness=lambda job_id: (next(answers), None))
    assert status["state"] == "completed"
    assert len(calls) == 5


def test_failed_analysis_stops_before_any_stage():
    calls: list = []
    status = run_chain(
        steps_for({}, calls),
        readiness=lambda job_id: ("failed", "Audio analysis did not finish."),
    )
    assert status["state"] == "failed"
    assert status["message"] == "Audio analysis did not finish."
    assert calls == []


def test_waiting_too_long_gives_up():
    calls: list = []
    status = run_chain(steps_for({}, calls), readiness=lambda job_id: ("waiting", None), timeout=0.0)
    assert status["state"] == "failed"
    assert "took too long" in status["message"]
    assert calls == []


def test_only_one_chain_per_recording_at_a_time():
    tracker = DraftTracker(sleep=lambda seconds: None)
    steps = steps_for({}, [])
    assert tracker.claim(JOB, steps)
    assert tracker.is_running(JOB)
    assert not tracker.claim(JOB, steps)
    tracker.run(JOB, steps, readiness=ready, timeout_seconds=1.0)
    assert not tracker.is_running(JOB)
    assert tracker.claim(JOB, steps)
    assert tracker.status("b" * 32) is None


def test_status_is_a_detached_copy():
    tracker = DraftTracker(sleep=lambda seconds: None)
    tracker.claim(JOB, steps_for({}, []))
    snapshot = tracker.status(JOB)
    snapshot["steps"][0]["outcome"] = "ran"
    assert tracker.status(JOB)["steps"][0]["outcome"] is None


def test_unknown_outcomes_are_refused():
    with pytest.raises(ValueError):
        StepResult("maybe")


def test_wait_for_status_returns_when_processing_ends():
    values = iter([{"x": "processing"}, {"x": "processing"}, {"x": "completed"}])
    assert wait_for_status(lambda: next(values), "x", sleep=lambda s: None) == "completed"
    assert wait_for_status(lambda: {"x": "processing"}, "x", sleep=lambda s: None, timeout_seconds=0) == "processing"


# ---------------------------------------------------------------------------
# Through the API


def test_one_action_drafts_the_score_from_saved_transcription(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        before = client.get(f"/api/jobs/{job_id}").json()
        assert before["draft"]["canStart"] is True
        assert before["draft"]["startUrl"] == f"/api/jobs/{job_id}/draft"
        response = client.post(f"/api/jobs/{job_id}/draft", json={})
        assert response.status_code == 202
        after = client.get(f"/api/jobs/{job_id}").json()

    progress = after["draft"]["progress"]
    assert progress["state"] == "completed", progress
    outcomes = {step["key"]: step["outcome"] for step in progress["steps"]}
    assert outcomes["separate"] == "skipped"  # not set up in tests
    assert outcomes["transcribe"] == "reused"
    assert outcomes["harmonize"] == "ran"
    assert outcomes["score"] == "ran"
    record = db.get_job(settings.database_path, job_id)
    assert record["score_status"] == "completed"
    assert record["harmony_status"] == "completed"
    assert after["draft"]["running"] is False and after["draft"]["canStart"] is True
    assert "full mix was transcribed" in progress["message"]


def test_drafting_again_reuses_everything(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    with TestClient(create_app(settings)) as client:
        assert client.post(f"/api/jobs/{job_id}/draft").status_code == 202
        first = db.get_job(settings.database_path, job_id)["score_artifact_file_name"]
        assert client.post(f"/api/jobs/{job_id}/draft").status_code == 202
        progress = client.get(f"/api/jobs/{job_id}").json()["draft"]["progress"]
    outcomes = {step["key"]: step["outcome"] for step in progress["steps"]}
    assert outcomes["score"] == "reused"
    assert db.get_job(settings.database_path, job_id)["score_artifact_file_name"] == first


def test_draft_requests_are_validated(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    busy = create_job(settings)
    with TestClient(create_app(settings)) as client:
        # Claimed after start-up, which marks interrupted work as retryable.
        assert db.claim_harmony_attempt(settings.database_path, busy, harmony_version="harmonic-context-v1")
        assert client.post(f"/api/jobs/{'f' * 32}/draft").status_code == 404
        assert client.post(f"/api/jobs/{job_id}/draft", json={"allowModelDownload": "yes"}).status_code == 422
        assert client.post(f"/api/jobs/{job_id}/draft", json={"other": 1}).status_code == 422
        conflict = client.post(f"/api/jobs/{busy}/draft")
        assert conflict.status_code == 409
        assert "already running" in conflict.json()["detail"]
        assert client.get(f"/api/jobs/{busy}").json()["draft"]["canStart"] is False


def test_failed_analysis_cannot_be_drafted(tmp_path: Path):
    settings = make_settings(tmp_path)
    job_id = create_job(settings)
    db.update_job(settings.database_path, job_id, analysis_status="failed")
    with TestClient(create_app(settings)) as client:
        response = client.post(f"/api/jobs/{job_id}/draft")
        assert response.status_code == 409
        assert "Audio analysis did not finish" in response.json()["detail"]
        assert client.get(f"/api/jobs/{job_id}").json()["draft"]["canStart"] is False


def test_browser_offers_one_action_drafting():
    script = (Path(__file__).resolve().parents[1] / "app" / "static" / "app.js").read_text(encoding="utf-8")
    assert "function renderDraft(job)" in script
    assert 'data-action="draft"' in script
    assert "async function startDraftFor(created)" in script
    assert "job.draft?.running===true" in script  # keeps polling while drafting
    assert "data-draft-consent" in script  # model download needs an explicit tick
    styles = (Path(__file__).resolve().parents[1] / "app" / "static" / "styles.css").read_text(encoding="utf-8")
    assert ".draft-panel" in styles and "prefers-reduced-motion" in styles