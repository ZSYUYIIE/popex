from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app import db


RAW_PATH = "transcription/raw-events.json"
DRAFT_PATH = "interpretation/draft.json"
INTERPRETATION_COLUMNS = {
    "interpretation_status",
    "interpretation_stage",
    "interpretation_progress",
    "interpretation_message",
    "interpretation_version",
    "interpretation_artifact_file_name",
    "interpreted_at",
    "interpretation_part_count",
    "interpretation_phrase_count",
    "interpretation_pitched_item_count",
    "interpretation_percussion_item_count",
    "interpretation_warning_count",
    "interpretation_error",
}
COUNT_FIELDS = (
    "interpretation_part_count",
    "interpretation_phrase_count",
    "interpretation_pitched_item_count",
    "interpretation_percussion_item_count",
    "interpretation_warning_count",
)


def create_transcribed_job(database: Path, job_id: str) -> dict:
    db.init_database(database)
    db.create_job(database, job_id, source_type="upload", original_filename="song.wav")
    db.update_job(
        database,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        preparation_status="completed",
        analysis_status="completed",
        source_file_name="source.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-01T00:00:00+00:00",
        separation_status="failed",
        separation_stage="failed",
        separation_progress=44,
        separation_error="preserve separation failure",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Raw transcription complete.",
        transcription_version="raw-transcription-v1",
        transcription_artifact_file_name=RAW_PATH,
        transcribed_at="2026-08-02T00:00:00+00:00",
        pitched_event_count=7,
        percussion_event_count=5,
        aligned_event_count=9,
        transcription_error=None,
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def mark_interpretation_completed(database: Path, job_id: str) -> dict:
    db.update_job(
        database,
        job_id,
        interpretation_status="completed",
        interpretation_stage="completed",
        interpretation_progress=100,
        interpretation_message="Editable interpretation complete.",
        interpretation_version="editable-interpretation-v1",
        interpretation_artifact_file_name=DRAFT_PATH,
        interpreted_at="2026-08-03T00:00:00+00:00",
        interpretation_part_count=4,
        interpretation_phrase_count=3,
        interpretation_pitched_item_count=7,
        interpretation_percussion_item_count=5,
        interpretation_warning_count=2,
        interpretation_error=None,
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def test_fresh_database_has_interpretation_defaults(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)

    with db.connect(database) as connection:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }

    assert INTERPRETATION_COLUMNS <= set(columns)
    assert columns["interpretation_status"]["dflt_value"] == "'not_started'"
    assert columns["interpretation_stage"]["dflt_value"] == "'not_started'"
    assert columns["interpretation_progress"]["dflt_value"] == "0"

    job = db.create_job(database, "fresh-interpretation", source_type="upload")
    assert job["interpretation_status"] == "not_started"
    assert job["interpretation_stage"] == "not_started"
    assert job["interpretation_progress"] == 0
    assert job["interpretation_message"] is None
    assert job["interpretation_version"] is None
    assert job["interpretation_artifact_file_name"] is None
    assert job["interpreted_at"] is None
    assert all(job[field] is None for field in COUNT_FIELDS)
    assert job["interpretation_error"] is None


def test_existing_database_migrates_without_losing_transcription(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, "legacy-transcribed")
    with sqlite3.connect(database) as connection:
        for field in INTERPRETATION_COLUMNS:
            try:
                connection.execute(f"ALTER TABLE jobs DROP COLUMN {field}")
            except sqlite3.OperationalError:
                pass
    db.init_database(database)
    after = db.get_job(database, "legacy-transcribed")

    assert after is not None
    preserved = {
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "analysis_json_file_name",
        "separation_status",
        "separation_error",
        "transcription_status",
        "transcription_stage",
        "transcription_progress",
        "transcription_version",
        "transcription_artifact_file_name",
        "transcribed_at",
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
    }
    assert {key: after[key] for key in preserved} == {
        key: before[key] for key in preserved
    }
    assert after["interpretation_status"] == "not_started"
    assert after["interpretation_stage"] == "not_started"
    assert after["interpretation_progress"] == 0


def test_blank_legacy_interpretation_state_and_negative_counts_are_normalized(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "legacy-invalid")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE jobs
            SET interpretation_status = '   ',
                interpretation_stage = '',
                interpretation_progress = -5,
                interpretation_part_count = -1,
                interpretation_phrase_count = -2,
                interpretation_pitched_item_count = -3,
                interpretation_percussion_item_count = -4,
                interpretation_warning_count = -5
            WHERE id = ?
            """,
            ("legacy-invalid",),
        )

    db.init_database(database)
    job = db.get_job(database, "legacy-invalid")
    assert job is not None
    assert job["interpretation_status"] == "not_started"
    assert job["interpretation_stage"] == "not_started"
    assert job["interpretation_progress"] == 0
    assert all(job[field] is None for field in COUNT_FIELDS)


def test_initialization_remains_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "idempotent")
    first = db.get_job(database, "idempotent")
    db.init_database(database)
    second = db.get_job(database, "idempotent")
    assert first == second


def test_claim_requires_completed_raw_transcription(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)
    db.create_job(database, "not-transcribed", source_type="upload")
    db.update_job(
        database,
        "not-transcribed",
        preparation_status="completed",
        analysis_status="completed",
        transcription_status="failed",
    )

    assert (
        db.claim_interpretation_attempt(
            database,
            "not-transcribed",
            interpretation_version="editable-interpretation-v1",
        )
        is False
    )
    job = db.get_job(database, "not-transcribed")
    assert job is not None
    assert job["interpretation_status"] == "not_started"


def test_first_claim_succeeds_without_mutating_prior_lifecycles(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, "claim-new")

    claimed = db.claim_interpretation_attempt(
        database,
        "claim-new",
        interpretation_version="editable-interpretation-v1",
        message="Preparing editable structure.",
    )
    after = db.get_job(database, "claim-new")

    assert claimed is True
    assert after is not None
    assert after["interpretation_status"] == "processing"
    assert after["interpretation_stage"] == "preparing_interpretation"
    assert after["interpretation_progress"] == 1
    assert after["interpretation_message"] == "Preparing editable structure."
    assert after["interpretation_version"] == "editable-interpretation-v1"
    assert after["interpretation_error"] is None
    preserved = {
        "status",
        "stage",
        "progress",
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "separation_status",
        "separation_error",
        "transcription_status",
        "transcription_stage",
        "transcription_progress",
        "transcription_version",
        "transcription_artifact_file_name",
        "transcribed_at",
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
    }
    assert {key: after[key] for key in preserved} == {
        key: before[key] for key in preserved
    }


def test_only_one_concurrent_claim_wins(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "concurrent")
    barrier = Barrier(8)

    def claim() -> bool:
        barrier.wait()
        return db.claim_interpretation_attempt(
            database,
            "concurrent",
            interpretation_version="editable-interpretation-v1",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: claim(), range(8)))

    assert results.count(True) == 1
    assert results.count(False) == 7
    job = db.get_job(database, "concurrent")
    assert job is not None
    assert job["interpretation_status"] == "processing"


def test_completed_result_requires_force_and_force_preserves_previous_metadata(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "reinterpret")
    previous = mark_interpretation_completed(database, "reinterpret")

    assert (
        db.claim_interpretation_attempt(
            database,
            "reinterpret",
            interpretation_version="editable-interpretation-v2",
        )
        is False
    )
    assert (
        db.claim_interpretation_attempt(
            database,
            "reinterpret",
            interpretation_version="editable-interpretation-v2",
            force=True,
        )
        is True
    )
    current = db.get_job(database, "reinterpret")
    assert current is not None
    assert current["interpretation_status"] == "processing"
    assert current["interpretation_version"] == "editable-interpretation-v2"
    assert current["interpretation_artifact_file_name"] == DRAFT_PATH
    assert current["interpreted_at"] == previous["interpreted_at"]
    assert all(current[field] == previous[field] for field in COUNT_FIELDS)


def test_failed_retry_can_claim_again_and_keeps_previous_draft(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "failed-retry")
    previous = mark_interpretation_completed(database, "failed-retry")
    db.update_job(
        database,
        "failed-retry",
        interpretation_status="failed",
        interpretation_stage="failed",
        interpretation_progress=72,
        interpretation_message="Interpretation stopped.",
        interpretation_error="bounded failure",
    )

    assert db.claim_interpretation_attempt(
        database,
        "failed-retry",
        interpretation_version="editable-interpretation-v2",
    )
    current = db.get_job(database, "failed-retry")
    assert current is not None
    assert current["interpretation_artifact_file_name"] == DRAFT_PATH
    assert current["interpreted_at"] == previous["interpreted_at"]
    assert all(current[field] == previous[field] for field in COUNT_FIELDS)
    assert current["interpretation_error"] is None


def test_restart_marks_processing_interpretation_failed_and_preserves_results(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "restart")
    previous = mark_interpretation_completed(database, "restart")
    assert db.claim_interpretation_attempt(
        database,
        "restart",
        interpretation_version="editable-interpretation-v2",
        force=True,
    )

    db.fail_incomplete_jobs(database)
    current = db.get_job(database, "restart")

    assert current is not None
    assert current["status"] == "completed"
    assert current["preparation_status"] == "completed"
    assert current["analysis_status"] == "completed"
    assert current["transcription_status"] == "completed"
    assert current["transcription_artifact_file_name"] == RAW_PATH
    assert current["interpretation_status"] == "failed"
    assert current["interpretation_stage"] == "failed"
    assert "interrupted" in current["interpretation_error"].lower()
    assert current["interpretation_artifact_file_name"] == DRAFT_PATH
    assert current["interpreted_at"] == previous["interpreted_at"]
    assert all(current[field] == previous[field] for field in COUNT_FIELDS)


def test_update_job_accepts_interpretation_result_fields(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "result-fields")

    result = mark_interpretation_completed(database, "result-fields")
    assert result["interpretation_status"] == "completed"
    assert result["interpretation_version"] == "editable-interpretation-v1"
    assert result["interpretation_artifact_file_name"] == DRAFT_PATH
    assert result["interpretation_part_count"] == 4
    assert result["interpretation_phrase_count"] == 3
    assert result["interpretation_pitched_item_count"] == 7
    assert result["interpretation_percussion_item_count"] == 5
    assert result["interpretation_warning_count"] == 2


@pytest.mark.parametrize("field", COUNT_FIELDS)
@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_update_job_rejects_invalid_interpretation_counts(
    tmp_path: Path,
    field: str,
    bad: object,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, f"bad-{field}-{str(bad)}")

    with pytest.raises(ValueError):
        db.update_job(database, before["id"], **{field: bad})
    assert db.get_job(database, before["id"]) == before


@pytest.mark.parametrize("field", COUNT_FIELDS)
def test_sqlite_rejects_direct_negative_interpretation_counts(
    tmp_path: Path,
    field: str,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, f"constraint-{field}")

    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(database) as connection:
            connection.execute(
                f"UPDATE jobs SET {field} = -1 WHERE id = ?",
                (f"constraint-{field}",),
            )
