from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app import db
from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.score_sources import (
    ANALYSIS_FILE_NAME,
    INTERPRETATION_FILE_NAME,
    RAW_TRANSCRIPTION_FILE_NAME,
    current_score_fingerprint,
    score_source_identity,
)
from app.transcription_draft import INTERPRETATION_DRAFT_RELATIVE_PATH
from app.transcription_events import RAW_TRANSCRIPTION_RELATIVE_PATH

VERSION = "score-pipeline-v1"
SCORE_COLUMNS = (
    "score_status",
    "score_stage",
    "score_progress",
    "score_message",
    "score_attempt_id",
    "score_attempt_fingerprint",
    "score_version",
    "score_artifact_file_name",
    "scored_at",
    "score_source_fingerprint",
    "score_measure_count",
    "score_note_count",
    "score_chord_symbol_count",
    "score_warning_count",
    "score_error",
)


def create_ready_job(database: Path, job_id: str = "a" * 32) -> dict:
    db.init_database(database)
    db.create_job(database, job_id, source_type="upload", original_filename="x.wav")
    db.update_job(
        database,
        job_id,
        status="completed",
        stage="completed",
        preparation_status="completed",
        analysis_status="completed",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name=ANALYSIS_JSON_RELATIVE_PATH,
        analyzed_at="2026-08-14T03:55:00+00:00",
        transcription_status="completed",
        transcription_version="raw-transcription-v1",
        transcription_artifact_file_name=RAW_TRANSCRIPTION_RELATIVE_PATH,
        transcribed_at="2026-08-14T04:00:00+00:00",
        pitched_event_count=3,
        percussion_event_count=0,
        aligned_event_count=0,
    )
    job = db.get_job(database, job_id)
    assert job is not None
    return job


def complete(database: Path, job_id: str, attempt_id: str, **overrides) -> db.ScoreCompletion:
    job = db.get_job(database, job_id)
    values = {
        "attempt_id": attempt_id,
        "score_version": VERSION,
        "artifact_file_name": f"score/score-document.{attempt_id}.json",
        "scored_at": "2026-09-30T10:00:00+00:00",
        "source_fingerprint": job["score_attempt_fingerprint"],
        "measure_count": 4,
        "note_count": 3,
        "chord_symbol_count": 1,
        "warning_count": 2,
    }
    values.update(overrides)
    return db.complete_score_attempt(database, job_id, **values)


def test_canonical_pointer_constants_match_owning_modules() -> None:
    assert ANALYSIS_FILE_NAME == ANALYSIS_JSON_RELATIVE_PATH
    assert RAW_TRANSCRIPTION_FILE_NAME == RAW_TRANSCRIPTION_RELATIVE_PATH
    assert INTERPRETATION_FILE_NAME == INTERPRETATION_DRAFT_RELATIVE_PATH


def test_fresh_database_has_score_defaults(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    assert job["score_status"] == "not_started"
    assert job["score_stage"] == "not_started"
    assert job["score_progress"] == 0
    assert all(job[column] is None for column in SCORE_COLUMNS[3:])


def test_legacy_database_gains_score_columns_without_losing_data(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_ready_job(database)
    with sqlite3.connect(database) as connection:
        for column in SCORE_COLUMNS:
            connection.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
    db.init_database(database)
    after = db.get_job(database, before["id"])
    assert {key: after[key] for key in before if not key.startswith("score")} == {
        key: value for key, value in before.items() if not key.startswith("score")
    }
    assert after["score_status"] == "not_started"
    db.init_database(database)  # idempotent
    assert db.get_job(database, before["id"]) == after


def test_inconsistent_legacy_score_rows_are_normalized(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            UPDATE jobs SET score_status = 'completed', score_progress = 100,
                score_attempt_id = ?, score_artifact_file_name = '../escape.json'
            WHERE id = ?
            """,
            ("b" * 32, job["id"]),
        )
    db.init_database(database)
    normalized = db.get_job(database, job["id"])
    assert normalized["score_status"] == "failed"
    assert normalized["score_attempt_id"] is None
    assert normalized["score_artifact_file_name"] is None
    assert normalized["score_error"] == "Saved score metadata is incomplete."
    assert normalized["score_progress"] < 100

    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE jobs SET score_status = ' bogus ', score_stage = '' WHERE id = ?",
            (job["id"],),
        )
    db.init_database(database)
    reset = db.get_job(database, job["id"])
    assert (reset["score_status"], reset["score_stage"], reset["score_progress"]) == (
        "not_started",
        "not_started",
        0,
    )


def test_identity_requires_completed_canonical_evidence(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    identity = score_source_identity(job)
    assert identity is not None
    assert identity["interpretation"] is None and identity["harmony"] is None
    for field, value in (
        ("transcription_status", "processing"),
        ("analysis_json_file_name", "analysis/other.json"),
        ("pitched_event_count", None),
        ("transcribed_at", " "),
    ):
        assert score_source_identity({**job, field: value}) is None
    assert db.claim_score_attempt(database, job["id"], score_version=VERSION) is not None
    blocked = create_ready_job(database, "c" * 32)
    db.update_job(database, blocked["id"], transcription_status="failed")
    assert db.claim_score_attempt(database, blocked["id"], score_version=VERSION) is None


def test_optional_layers_change_the_fingerprint_only_when_completed(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    base = current_score_fingerprint(job)
    running = {**job, "harmony_status": "processing", "harmony_version": "h1",
               "harmonized_at": "2026-08-14T05:00:00+00:00",
               "harmony_artifact_file_name": "harmony/harmonic-context.json"}
    assert current_score_fingerprint(running) == base
    completed = {**running, "harmony_status": "completed"}
    assert current_score_fingerprint(completed) != base
    interpreted = {**job, "interpretation_status": "completed", "interpretation_version": "d1",
                   "interpreted_at": "2026-08-14T05:00:00+00:00",
                   "interpretation_artifact_file_name": INTERPRETATION_FILE_NAME}
    assert current_score_fingerprint(interpreted) not in {base, current_score_fingerprint(completed)}


def test_only_one_concurrent_claim_wins_and_returns_its_identity(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    barrier = Barrier(8)

    def claim() -> str | None:
        barrier.wait()
        return db.claim_score_attempt(database, job["id"], score_version=VERSION)

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: claim(), range(8)))
    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    claimed = db.get_job(database, job["id"])
    assert claimed["score_attempt_id"] == winners[0]
    assert claimed["score_status"] == "processing"
    assert claimed["score_attempt_fingerprint"] == current_score_fingerprint(claimed)


def test_only_one_concurrent_worker_start_wins(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    attempt = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    barrier = Barrier(6)

    def start() -> bool:
        barrier.wait()
        return db.start_score_attempt(database, job["id"], attempt_id=attempt)

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(lambda _: start(), range(6)))
    assert results.count(True) == 1
    with pytest.raises(ValueError):
        db.start_score_attempt(database, job["id"], attempt_id="not-an-attempt")


def test_start_rejects_evidence_changed_after_claim(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    attempt = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    db.update_job(database, job["id"], transcribed_at="2026-08-15T00:00:00+00:00")
    assert not db.start_score_attempt(database, job["id"], attempt_id=attempt)


def test_progress_is_monotonic_and_attempt_bound(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    attempt = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    assert db.start_score_attempt(database, job["id"], attempt_id=attempt)
    kwargs = {"stage": "building_measures", "message": "Placing notes."}
    assert db.update_score_progress(database, job["id"], attempt_id=attempt, progress=30, **kwargs)
    assert not db.update_score_progress(database, job["id"], attempt_id=attempt, progress=20, **kwargs)
    assert not db.update_score_progress(database, job["id"], attempt_id="f" * 32, progress=40, **kwargs)
    with pytest.raises(ValueError):
        db.update_score_progress(database, job["id"], attempt_id=attempt, progress=101, **kwargs)


def test_completion_requires_force_to_rebuild_and_reports_previous(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    first = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    outcome = complete(database, job["id"], first)
    assert outcome == db.ScoreCompletion(True, None)
    assert db.claim_score_attempt(database, job["id"], score_version=VERSION) is None
    second = db.claim_score_attempt(database, job["id"], score_version=VERSION, force=True)
    assert second
    during = db.get_job(database, job["id"])
    assert during["score_artifact_file_name"] == f"score/score-document.{first}.json"
    assert db.claim_score_attempt(database, job["id"], score_version=VERSION, force=True) is None
    outcome = complete(database, job["id"], second)
    assert outcome == db.ScoreCompletion(True, f"score/score-document.{first}.json")


def test_completion_rejects_wrong_pointer_counts_and_stale_evidence(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    attempt = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    assert not complete(
        database, job["id"], attempt, artifact_file_name=f"score/score-document.{'f' * 32}.json"
    ).completed
    with pytest.raises(ValueError):
        complete(database, job["id"], attempt, note_count=-1)
    with pytest.raises(ValueError):
        complete(database, job["id"], attempt, chord_symbol_count=9, measure_count=2)
    with pytest.raises(ValueError):
        complete(database, job["id"], attempt, scored_at="2026-09-30T10:00:00+02:00")
    with pytest.raises(ValueError):
        complete(database, job["id"], attempt, note_count=True)
    db.update_job(database, job["id"], analyzed_at="2026-08-20T00:00:00+00:00")
    assert not complete(database, job["id"], attempt).completed
    assert db.get_job(database, job["id"])["score_status"] == "processing"


def test_failure_is_attempt_scoped_sanitized_and_preserves_success(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    first = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    complete(database, job["id"], first)
    success = db.get_job(database, job["id"])
    second = db.claim_score_attempt(database, job["id"], score_version=VERSION, force=True)
    assert not db.fail_score_attempt(database, job["id"], attempt_id=first, error="late")
    assert db.fail_score_attempt(
        database,
        job["id"],
        attempt_id=second,
        error="Could not open /home/user/private/song.json password=hunter2",
    )
    failed = db.get_job(database, job["id"])
    assert failed["score_status"] == "failed"
    assert failed["score_error"] == "Score construction failed."
    assert failed["score_attempt_id"] is None
    for column in (
        "score_version",
        "score_artifact_file_name",
        "scored_at",
        "score_source_fingerprint",
        "score_measure_count",
        "score_note_count",
    ):
        assert failed[column] == success[column]
    # A failed job can be retried without force.
    assert db.claim_score_attempt(database, job["id"], score_version=VERSION)


def test_restart_recovery_fails_only_the_interrupted_attempt(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    first = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    complete(database, job["id"], first)
    db.claim_score_attempt(database, job["id"], score_version=VERSION, force=True)
    before = db.get_job(database, job["id"])
    db.fail_incomplete_jobs(database)
    after = db.get_job(database, job["id"])
    assert after["score_status"] == "failed"
    assert after["score_attempt_id"] is None
    assert after["score_attempt_fingerprint"] is None
    assert "restart" in after["score_error"]
    assert after["score_artifact_file_name"] == before["score_artifact_file_name"]
    assert after["status"] == before["status"]
    assert after["transcription_status"] == "completed"


def test_cleanup_lease_reports_durable_and_active_ownership(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    with db.score_cleanup_lease(database, job["id"]) as owners:
        assert owners == (None, None)
    first = db.claim_score_attempt(database, job["id"], score_version=VERSION)
    with db.score_cleanup_lease(database, job["id"]) as owners:
        assert owners == (None, first)
    complete(database, job["id"], first)
    with db.score_cleanup_lease(database, job["id"]) as owners:
        assert owners == (f"score/score-document.{first}.json", None)
    with pytest.raises(ValueError):
        with db.score_cleanup_lease(database, "0" * 32):
            pass


def test_update_job_validates_score_fields(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    job = create_ready_job(database)
    for field, value in (
        ("score_status", "done"),
        ("score_attempt_id", "../x"),
        ("score_artifact_file_name", "score/../../etc.json"),
        ("score_source_fingerprint", "abc"),
        ("score_note_count", -3),
        ("score_progress", 150),
    ):
        with pytest.raises(ValueError):
            db.update_job(database, job["id"], **{field: value})
