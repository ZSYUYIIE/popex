from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app import db


RAW_PATH = "transcription/raw-events.json"
HARMONY_PATH = "harmony/harmonic-context.json"
HARMONY_COLUMNS = {
    "harmony_status",
    "harmony_stage",
    "harmony_progress",
    "harmony_message",
    "harmony_attempt_version",
    "harmony_version",
    "harmony_artifact_file_name",
    "harmonized_at",
    "harmony_source_transcription_version",
    "harmony_source_transcription_artifact_file_name",
    "harmony_source_transcribed_at",
    "harmony_event_count",
    "harmony_segment_count",
    "harmony_resolved_segment_count",
    "harmony_unresolved_segment_count",
    "harmony_unresolved_event_count",
    "harmony_warning_count",
    "harmony_used_interpretation_context",
    "harmony_error",
}
COUNT_FIELDS = (
    "harmony_event_count",
    "harmony_segment_count",
    "harmony_resolved_segment_count",
    "harmony_unresolved_segment_count",
    "harmony_unresolved_event_count",
    "harmony_warning_count",
)
SUCCESS_FIELDS = (
    "harmony_version",
    "harmony_artifact_file_name",
    "harmonized_at",
    *COUNT_FIELDS,
    "harmony_used_interpretation_context",
)
UPSTREAM_FIELDS = (
    "status",
    "stage",
    "progress",
    "preparation_status",
    "analysis_status",
    "analysis_version",
    "analysis_json_file_name",
    "separation_status",
    "separation_stage",
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
    "interpretation_status",
    "interpretation_stage",
    "interpretation_artifact_file_name",
    "interpreted_at",
)


def create_transcribed_job(
    database: Path,
    job_id: str,
    *,
    separation_status: str = "failed",
    interpretation_status: str = "not_started",
) -> dict:
    db.init_database(database)
    db.create_job(
        database,
        job_id,
        source_type="upload",
        original_filename="song.wav",
    )
    db.update_job(
        database,
        job_id,
        status="completed",
        stage="completed",
        progress=100,
        message="Raw transcription complete.",
        preparation_status="completed",
        analysis_status="completed",
        source_file_name="source.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-01T00:00:00+00:00",
        separation_status=separation_status,
        separation_stage=(
            "completed" if separation_status == "completed" else "failed"
        ),
        separation_progress=(100 if separation_status == "completed" else 38),
        separation_error=(
            None if separation_status == "completed" else "preserve separation failure"
        ),
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Raw transcription complete.",
        transcription_version="raw-transcription-v1",
        transcription_artifact_file_name=RAW_PATH,
        transcribed_at="2026-08-02T00:00:00+00:00",
        pitched_event_count=8,
        percussion_event_count=5,
        aligned_event_count=11,
        transcription_error=None,
        interpretation_status=interpretation_status,
        interpretation_stage=(
            "completed" if interpretation_status == "completed" else interpretation_status
        ),
        interpretation_progress=(100 if interpretation_status == "completed" else 0),
        interpretation_message=(
            "Editable interpretation complete."
            if interpretation_status == "completed"
            else None
        ),
        interpretation_version=(
            "editable-interpretation-v1"
            if interpretation_status == "completed"
            else None
        ),
        interpretation_artifact_file_name=(
            "interpretation/draft.json"
            if interpretation_status == "completed"
            else None
        ),
        interpreted_at=(
            "2026-08-03T00:00:00+00:00"
            if interpretation_status == "completed"
            else None
        ),
        interpretation_part_count=(4 if interpretation_status == "completed" else None),
        interpretation_phrase_count=(3 if interpretation_status == "completed" else None),
        interpretation_pitched_item_count=(8 if interpretation_status == "completed" else None),
        interpretation_percussion_item_count=(5 if interpretation_status == "completed" else None),
        interpretation_warning_count=(2 if interpretation_status == "completed" else None),
        interpretation_error=None,
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def complete_harmony(
    database: Path,
    job_id: str,
    *,
    version: str = "harmonic-context-v1",
    used_interpretation_context: bool = True,
) -> dict:
    assert db.claim_harmony_attempt(
        database,
        job_id,
        harmony_version=version,
    )
    assert db.update_harmony_progress(
        database,
        job_id,
        stage="inferring_harmonic_context",
        progress=55,
        message="Inferring harmonic candidates.",
    )
    assert db.complete_harmony_attempt(
        database,
        job_id,
        harmony_version=version,
        artifact_file_name=HARMONY_PATH,
        harmonized_at="2026-08-04T00:00:00+00:00",
        event_count=8,
        segment_count=3,
        resolved_segment_count=2,
        unresolved_segment_count=1,
        unresolved_event_count=1,
        warning_count=2,
        used_interpretation_context=used_interpretation_context,
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def successful_metadata(job: dict) -> dict:
    return {field: job[field] for field in SUCCESS_FIELDS}


def test_fresh_database_has_harmony_defaults(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)

    with db.connect(database) as connection:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }

    assert HARMONY_COLUMNS <= set(columns)
    assert columns["harmony_status"]["dflt_value"] == "'not_started'"
    assert columns["harmony_stage"]["dflt_value"] == "'not_started'"
    assert columns["harmony_progress"]["dflt_value"] == "0"

    job = db.create_job(database, "fresh-harmony", source_type="upload")
    assert job["harmony_status"] == "not_started"
    assert job["harmony_stage"] == "not_started"
    assert job["harmony_progress"] == 0
    assert job["harmony_message"] is None
    assert job["harmony_attempt_version"] is None
    assert job["harmony_version"] is None
    assert job["harmony_artifact_file_name"] is None
    assert job["harmonized_at"] is None
    assert job["harmony_source_transcription_version"] is None
    assert job["harmony_source_transcription_artifact_file_name"] is None
    assert job["harmony_source_transcribed_at"] is None
    assert all(job[field] is None for field in COUNT_FIELDS)
    assert job["harmony_used_interpretation_context"] is None
    assert job["harmony_error"] is None


def test_existing_database_migrates_without_losing_existing_lifecycles(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(
        database,
        "legacy-harmony",
        separation_status="completed",
        interpretation_status="completed",
    )
    with sqlite3.connect(database) as connection:
        for field in HARMONY_COLUMNS:
            try:
                connection.execute(f"ALTER TABLE jobs DROP COLUMN {field}")
            except sqlite3.OperationalError:
                pass

    db.init_database(database)
    after = db.get_job(database, "legacy-harmony")

    assert after is not None
    assert {field: after[field] for field in UPSTREAM_FIELDS} == {
        field: before[field] for field in UPSTREAM_FIELDS
    }
    assert after["harmony_status"] == "not_started"
    assert after["harmony_stage"] == "not_started"
    assert after["harmony_progress"] == 0
    assert all(after[field] is None for field in SUCCESS_FIELDS)


def test_blank_invalid_and_inconsistent_legacy_harmony_state_is_normalized(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "legacy-invalid")
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE jobs
            SET harmony_status = '   ',
                harmony_stage = '',
                harmony_progress = -5,
                harmony_event_count = -1,
                harmony_segment_count = -2,
                harmony_resolved_segment_count = -3,
                harmony_unresolved_segment_count = -4,
                harmony_unresolved_event_count = -5,
                harmony_warning_count = -6,
                harmony_used_interpretation_context = 9
            WHERE id = ?
            """,
            ("legacy-invalid",),
        )
    db.init_database(database)
    job = db.get_job(database, "legacy-invalid")

    assert job is not None
    assert job["harmony_status"] == "not_started"
    assert job["harmony_stage"] == "not_started"
    assert job["harmony_progress"] == 0
    assert all(job[field] is None for field in COUNT_FIELDS)
    assert job["harmony_used_interpretation_context"] is None

    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA ignore_check_constraints = ON")
        connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'completed',
                harmony_stage = 'completed',
                harmony_progress = 150,
                harmony_version = NULL,
                harmony_artifact_file_name = NULL,
                harmonized_at = NULL,
                harmony_error = NULL
            WHERE id = ?
            """,
            ("legacy-invalid",),
        )
    db.init_database(database)
    job = db.get_job(database, "legacy-invalid")
    assert job is not None
    assert job["harmony_status"] == "failed"
    assert job["harmony_stage"] == "failed"
    assert 0 <= job["harmony_progress"] < 100
    assert "incomplete" in job["harmony_error"].lower()


def test_initialization_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "idempotent")
    complete_harmony(database, "idempotent")
    first = db.get_job(database, "idempotent")
    db.init_database(database)
    second = db.get_job(database, "idempotent")
    assert first == second


def test_claim_requires_completed_canonical_raw_transcription(
    tmp_path: Path,
) -> None:
    cases = (
        {"transcription_status": "failed"},
        {"transcription_version": None},
        {"transcription_version": ""},
        {"transcription_artifact_file_name": None},
        {"transcription_artifact_file_name": "transcription/other.json"},
        {"transcribed_at": None},
        {"preparation_status": "failed"},
        {"analysis_status": "failed"},
    )
    for index, fields in enumerate(cases):
        database = tmp_path / f"case-{index}.sqlite3"
        job_id = f"case-{index}"
        create_transcribed_job(database, job_id)
        db.update_job(database, job_id, **fields)
        expected = db.get_job(database, job_id)

        assert not db.claim_harmony_attempt(
            database,
            job_id,
            harmony_version="harmonic-context-v1",
        )
        assert db.get_job(database, job_id) == expected


def test_claim_never_requires_stems_or_interpretation(tmp_path: Path) -> None:
    for index, (separation, interpretation) in enumerate(
        (
            ("not_started", "not_started"),
            ("failed", "failed"),
            ("completed", "not_started"),
        )
    ):
        database = tmp_path / f"optional-{index}.sqlite3"
        job_id = f"optional-{index}"
        before = create_transcribed_job(
            database,
            job_id,
            separation_status=separation,
            interpretation_status=interpretation,
        )
        assert db.claim_harmony_attempt(
            database,
            job_id,
            harmony_version="harmonic-context-v1",
        )
        after = db.get_job(database, job_id)
        assert after is not None
        assert after["harmony_status"] == "processing"
        assert after["harmony_stage"] == "loading_raw_transcription"
        assert after["harmony_attempt_version"] == "harmonic-context-v1"
        assert after["harmony_source_transcription_version"] == before[
            "transcription_version"
        ]
        assert after["harmony_source_transcription_artifact_file_name"] == RAW_PATH
        assert after["harmony_source_transcribed_at"] == before["transcribed_at"]
        assert {field: after[field] for field in UPSTREAM_FIELDS} == {
            field: before[field] for field in UPSTREAM_FIELDS
        }


def test_only_one_concurrent_harmony_claim_wins(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "concurrent")
    barrier = Barrier(8)

    def claim() -> bool:
        barrier.wait()
        return db.claim_harmony_attempt(
            database,
            "concurrent",
            harmony_version="harmonic-context-v1",
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: claim(), range(8)))

    assert results.count(True) == 1
    assert results.count(False) == 7
    job = db.get_job(database, "concurrent")
    assert job is not None
    assert job["harmony_status"] == "processing"


def test_completed_requires_force_and_forced_claim_preserves_last_success(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "reharmonize")
    previous = complete_harmony(database, "reharmonize")

    assert not db.claim_harmony_attempt(
        database,
        "reharmonize",
        harmony_version="harmonic-context-v2",
    )
    assert db.claim_harmony_attempt(
        database,
        "reharmonize",
        harmony_version="harmonic-context-v2",
        force=True,
    )
    current = db.get_job(database, "reharmonize")
    assert current is not None
    assert current["harmony_status"] == "processing"
    assert current["harmony_attempt_version"] == "harmonic-context-v2"
    assert successful_metadata(current) == successful_metadata(previous)
    assert current["harmony_error"] is None


def test_failed_retry_preserves_last_success_until_replacement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "retry")
    previous = complete_harmony(database, "retry")
    assert db.claim_harmony_attempt(
        database,
        "retry",
        harmony_version="harmonic-context-v2",
        force=True,
    )
    assert db.fail_harmony_attempt(
        database,
        "retry",
        error="bounded failure",
    )

    failed = db.get_job(database, "retry")
    assert failed is not None
    assert failed["harmony_status"] == "failed"
    assert successful_metadata(failed) == successful_metadata(previous)
    assert db.claim_harmony_attempt(
        database,
        "retry",
        harmony_version="harmonic-context-v2",
    )
    retrying = db.get_job(database, "retry")
    assert retrying is not None
    assert successful_metadata(retrying) == successful_metadata(previous)
    assert retrying["harmony_error"] is None


def test_progress_is_monotonic_and_bound_to_the_claimed_raw_source(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "progress")
    assert db.claim_harmony_attempt(
        database,
        "progress",
        harmony_version="harmonic-context-v1",
    )

    assert db.update_harmony_progress(
        database,
        "progress",
        stage="inferring_harmonic_context",
        progress=55,
        message="Inferring harmonic candidates.",
    )
    assert not db.update_harmony_progress(
        database,
        "progress",
        stage="loading_analysis_context",
        progress=25,
        message="Regressive progress.",
    )
    current = db.get_job(database, "progress")
    assert current is not None
    assert current["harmony_progress"] == 55
    assert current["harmony_stage"] == "inferring_harmonic_context"

    with db.connect(database) as connection:
        connection.execute(
            "UPDATE jobs SET transcribed_at = ? WHERE id = ?",
            ("2026-08-09T00:00:00+00:00", "progress"),
        )
    assert not db.update_harmony_progress(
        database,
        "progress",
        stage="saving_harmonic_context",
        progress=92,
        message="This worker is stale.",
    )


def test_completion_is_atomic_and_validates_artifact_truth_counts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "complete")
    assert db.claim_harmony_attempt(
        database,
        "complete",
        harmony_version="harmonic-context-v1",
    )
    assert db.complete_harmony_attempt(
        database,
        "complete",
        harmony_version="harmonic-context-v1",
        artifact_file_name=HARMONY_PATH,
        harmonized_at="2026-08-04T00:00:00+00:00",
        event_count=8,
        segment_count=3,
        resolved_segment_count=2,
        unresolved_segment_count=1,
        unresolved_event_count=1,
        warning_count=2,
        used_interpretation_context=False,
    )
    job = db.get_job(database, "complete")

    assert job is not None
    assert job["harmony_status"] == "completed"
    assert job["harmony_stage"] == "completed"
    assert job["harmony_progress"] == 100
    assert job["harmony_attempt_version"] == "harmonic-context-v1"
    assert job["harmony_version"] == "harmonic-context-v1"
    assert job["harmony_artifact_file_name"] == HARMONY_PATH
    assert job["harmony_event_count"] == 8
    assert job["harmony_segment_count"] == 3
    assert job["harmony_resolved_segment_count"] == 2
    assert job["harmony_unresolved_segment_count"] == 1
    assert job["harmony_unresolved_event_count"] == 1
    assert job["harmony_warning_count"] == 2
    assert job["harmony_used_interpretation_context"] == 0
    assert job["harmony_error"] is None


@pytest.mark.parametrize(
    "overrides",
    [
        {"event_count": -1},
        {"segment_count": 4},
        {"resolved_segment_count": -1},
        {"unresolved_segment_count": -1},
        {"unresolved_event_count": 9},
        {"warning_count": True},
        {"used_interpretation_context": 1},
        {"artifact_file_name": "harmony/other.json"},
    ],
)
def test_completion_rejects_invalid_counts_boolean_and_path(
    tmp_path: Path,
    overrides: dict,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "invalid-completion")
    assert db.claim_harmony_attempt(
        database,
        "invalid-completion",
        harmony_version="harmonic-context-v1",
    )
    values = {
        "harmony_version": "harmonic-context-v1",
        "artifact_file_name": HARMONY_PATH,
        "harmonized_at": "2026-08-04T00:00:00+00:00",
        "event_count": 8,
        "segment_count": 3,
        "resolved_segment_count": 2,
        "unresolved_segment_count": 1,
        "unresolved_event_count": 1,
        "warning_count": 2,
        "used_interpretation_context": False,
    }
    values.update(overrides)

    with pytest.raises(ValueError):
        db.complete_harmony_attempt(
            database,
            "invalid-completion",
            **values,
        )
    job = db.get_job(database, "invalid-completion")
    assert job is not None
    assert job["harmony_status"] == "processing"


def test_wrong_attempt_version_and_stale_raw_source_cannot_complete(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "stale")
    assert db.claim_harmony_attempt(
        database,
        "stale",
        harmony_version="harmonic-context-v1",
    )
    common = dict(
        artifact_file_name=HARMONY_PATH,
        harmonized_at="2026-08-04T00:00:00+00:00",
        event_count=8,
        segment_count=3,
        resolved_segment_count=2,
        unresolved_segment_count=1,
        unresolved_event_count=1,
        warning_count=2,
        used_interpretation_context=False,
    )
    assert not db.complete_harmony_attempt(
        database,
        "stale",
        harmony_version="harmonic-context-v2",
        **common,
    )

    db.update_job(
        database,
        "stale",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_version="raw-transcription-v2",
        transcription_artifact_file_name=RAW_PATH,
        transcribed_at="2026-08-05T00:00:00+00:00",
        pitched_event_count=9,
        percussion_event_count=5,
        aligned_event_count=12,
    )
    reset = db.get_job(database, "stale")
    assert reset is not None
    assert reset["harmony_status"] == "not_started"
    assert not db.complete_harmony_attempt(
        database,
        "stale",
        harmony_version="harmonic-context-v1",
        **common,
    )
    assert not db.fail_harmony_attempt(
        database,
        "stale",
        error="stale worker",
    )


def test_failure_sanitizes_error_and_preserves_previous_success(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "failure")
    previous = complete_harmony(database, "failure")
    assert db.claim_harmony_attempt(
        database,
        "failure",
        harmony_version="harmonic-context-v2",
        force=True,
    )
    assert db.fail_harmony_attempt(
        database,
        "failure",
        error="Traceback at C:\\Users\\person\\secret token=private",
    )
    current = db.get_job(database, "failure")

    assert current is not None
    assert current["harmony_status"] == "failed"
    assert current["harmony_stage"] == "failed"
    assert current["harmony_error"] == "Harmonic-context processing failed."
    assert successful_metadata(current) == successful_metadata(previous)
    assert {field: current[field] for field in UPSTREAM_FIELDS} == {
        field: previous[field] for field in UPSTREAM_FIELDS
    }


def test_restart_marks_processing_harmony_failed_and_preserves_everything_else(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(
        database,
        "restart",
        separation_status="failed",
        interpretation_status="completed",
    )
    previous = complete_harmony(database, "restart")
    assert db.claim_harmony_attempt(
        database,
        "restart",
        harmony_version="harmonic-context-v2",
        force=True,
    )

    db.fail_incomplete_jobs(database)
    current = db.get_job(database, "restart")

    assert current is not None
    assert current["status"] == "completed"
    assert current["harmony_status"] == "failed"
    assert current["harmony_stage"] == "failed"
    assert "interrupted" in current["harmony_error"].lower()
    assert successful_metadata(current) == successful_metadata(previous)
    assert {field: current[field] for field in UPSTREAM_FIELDS} == {
        field: previous[field] for field in UPSTREAM_FIELDS
    }


def test_successful_raw_replacement_automatically_invalidates_harmony(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "upstream-success")
    complete_harmony(database, "upstream-success")

    assert db.claim_transcription_attempt(
        database,
        "upstream-success",
        transcription_version="raw-transcription-v2",
        force=True,
    )
    claimed = db.get_job(database, "upstream-success")
    assert claimed is not None
    assert claimed["harmony_status"] == "completed"
    assert claimed["harmony_artifact_file_name"] == HARMONY_PATH

    db.update_job(
        database,
        "upstream-success",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Replacement raw transcription complete.",
        transcription_version="raw-transcription-v2",
        transcription_artifact_file_name=RAW_PATH,
        transcribed_at="2026-08-06T00:00:00+00:00",
        pitched_event_count=10,
        percussion_event_count=6,
        aligned_event_count=14,
        transcription_error=None,
    )
    current = db.get_job(database, "upstream-success")

    assert current is not None
    assert current["transcription_status"] == "completed"
    assert current["harmony_status"] == "not_started"
    assert current["harmony_stage"] == "not_started"
    assert current["harmony_progress"] == 0
    assert current["harmony_message"] is None
    assert current["harmony_attempt_version"] is None
    assert current["harmony_version"] is None
    assert current["harmony_artifact_file_name"] is None
    assert current["harmonized_at"] is None
    assert current["harmony_source_transcription_version"] is None
    assert current["harmony_source_transcription_artifact_file_name"] is None
    assert current["harmony_source_transcribed_at"] is None
    assert all(current[field] is None for field in COUNT_FIELDS)
    assert current["harmony_used_interpretation_context"] is None
    assert current["harmony_error"] is None


def test_failed_raw_retry_preserves_previous_harmony_result(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "upstream-failure")
    previous = complete_harmony(database, "upstream-failure")

    assert db.claim_transcription_attempt(
        database,
        "upstream-failure",
        transcription_version="raw-transcription-v2",
        force=True,
    )
    db.update_job(
        database,
        "upstream-failure",
        transcription_status="failed",
        transcription_stage="failed",
        transcription_progress=70,
        transcription_message="Replacement failed.",
        transcription_error="bounded failure",
    )
    current = db.get_job(database, "upstream-failure")

    assert current is not None
    assert current["transcription_status"] == "failed"
    assert current["harmony_status"] == "completed"
    assert successful_metadata(current) == successful_metadata(previous)


def test_reset_helper_is_source_aware_and_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "reset-helper")
    complete_harmony(database, "reset-helper")

    assert not db.reset_harmony_after_transcription_change(
        database,
        "reset-helper",
    )
    with db.connect(database) as connection:
        connection.execute(
            "UPDATE jobs SET transcribed_at = ? WHERE id = ?",
            ("2026-08-07T00:00:00+00:00", "reset-helper"),
        )
    assert db.reset_harmony_after_transcription_change(
        database,
        "reset-helper",
    )
    assert not db.reset_harmony_after_transcription_change(
        database,
        "reset-helper",
    )
    current = db.get_job(database, "reset-helper")
    assert current is not None
    assert current["harmony_status"] == "not_started"
    assert current["harmony_artifact_file_name"] is None


@pytest.mark.parametrize("field", COUNT_FIELDS)
@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_update_job_rejects_invalid_harmony_counts(
    tmp_path: Path,
    field: str,
    bad: object,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, f"bad-{field}-{str(bad)}")

    with pytest.raises(ValueError):
        db.update_job(database, before["id"], **{field: bad})
    assert db.get_job(database, before["id"]) == before


@pytest.mark.parametrize("bad", [-1, 101, True, float("nan"), "20"])
def test_update_job_rejects_invalid_harmony_progress(
    tmp_path: Path,
    bad: object,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, f"bad-progress-{str(bad)}")

    with pytest.raises(ValueError):
        db.update_job(database, before["id"], harmony_progress=bad)
    assert db.get_job(database, before["id"]) == before


@pytest.mark.parametrize("bad", [0, 1, 2, "true", 1.0])
def test_update_job_rejects_non_boolean_interpretation_context_flag(
    tmp_path: Path,
    bad: object,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_transcribed_job(database, f"bad-flag-{str(bad)}")

    with pytest.raises(ValueError):
        db.update_job(
            database,
            before["id"],
            harmony_used_interpretation_context=bad,
        )
    assert db.get_job(database, before["id"]) == before


def test_update_job_accepts_strict_boolean_context_flag(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "valid-flag")
    db.update_job(
        database,
        "valid-flag",
        harmony_used_interpretation_context=True,
    )
    job = db.get_job(database, "valid-flag")
    assert job is not None
    assert job["harmony_used_interpretation_context"] == 1


@pytest.mark.parametrize("field", COUNT_FIELDS)
def test_sqlite_rejects_direct_negative_harmony_counts(
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


def test_sqlite_rejects_invalid_progress_and_context_flag(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_transcribed_job(database, "constraints")

    for statement in (
        "UPDATE jobs SET harmony_progress = -1 WHERE id = ?",
        "UPDATE jobs SET harmony_progress = 101 WHERE id = ?",
        "UPDATE jobs SET harmony_used_interpretation_context = 2 WHERE id = ?",
    ):
        with pytest.raises(sqlite3.IntegrityError):
            with db.connect(database) as connection:
                connection.execute(statement, ("constraints",))
