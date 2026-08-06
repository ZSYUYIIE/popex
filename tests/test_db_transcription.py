import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import pytest

from app import db


ARTIFACT_PATH = "transcription/raw-events.json"
MANIFEST_PATH = "stems/stem-separation.json"
TRANSCRIPTION_COLUMNS = {
    "transcription_status",
    "transcription_stage",
    "transcription_progress",
    "transcription_message",
    "transcription_version",
    "transcription_artifact_file_name",
    "transcribed_at",
    "pitched_event_count",
    "percussion_event_count",
    "aligned_event_count",
    "transcription_error",
}


def create_original_database(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL,
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                title TEXT,
                uploader TEXT,
                duration_seconds REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs VALUES (
                'legacy-original',
                'https://example.invalid/old',
                'completed',
                100,
                'Original job',
                'Artist',
                10,
                NULL,
                '2026-01-01',
                '2026-01-01'
            )
            """
        )


def create_pre_transcription_database(database: Path) -> None:
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY,
                source_url TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                progress REAL NOT NULL DEFAULT 0,
                title TEXT,
                uploader TEXT,
                duration_seconds REAL,
                error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                source_type TEXT NOT NULL DEFAULT 'url',
                stage TEXT NOT NULL DEFAULT 'queued',
                message TEXT,
                original_filename TEXT,
                source_format TEXT,
                sample_rate INTEGER,
                channel_count INTEGER,
                source_file_name TEXT,
                normalized_file_name TEXT,
                metadata_file_name TEXT,
                preparation_status TEXT NOT NULL DEFAULT 'pending',
                analysis_status TEXT NOT NULL DEFAULT 'not_started',
                analysis_version TEXT,
                tempo_bpm REAL,
                tempo_confidence REAL,
                key_symbol TEXT,
                key_confidence REAL,
                analysis_json_file_name TEXT,
                analyzed_at TEXT,
                analysis_error TEXT,
                separation_status TEXT NOT NULL DEFAULT 'not_started',
                separation_stage TEXT NOT NULL DEFAULT 'not_started',
                separation_progress REAL NOT NULL DEFAULT 0,
                separation_message TEXT,
                separation_version TEXT,
                separation_model TEXT,
                stem_manifest_file_name TEXT,
                separated_at TEXT,
                separation_error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id, source_url, status, progress, title, error,
                created_at, updated_at, source_type, stage, message,
                source_file_name, normalized_file_name, metadata_file_name,
                preparation_status, analysis_status, analysis_version,
                analysis_json_file_name, analyzed_at, analysis_error,
                separation_status, separation_stage, separation_progress,
                separation_message, separation_version, separation_model,
                stem_manifest_file_name, separated_at, separation_error
            ) VALUES (
                'pre-transcription', '', 'completed', 100, 'Preserved job',
                'preserved top-level error', '2026-02-01', '2026-02-02',
                'upload', 'completed', 'Analysis completed.',
                'source.wav', 'analysis.wav', 'metadata.json',
                'completed', 'completed', 'baseline-librosa-v1',
                'analysis/audio-analysis.json',
                '2026-02-02T00:00:00+00:00',
                'preserved analysis error',
                'completed', 'completed', 100, 'Stems completed.',
                'stem-separation-v1', 'htdemucs',
                'stems/stem-separation.json',
                '2026-02-03T00:00:00+00:00',
                'preserved separation error'
            )
            """
        )


def create_analyzed_job(database: Path, job_id: str) -> dict:
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
        message="Analysis completed.",
        preparation_status="completed",
        analysis_status="completed",
        source_file_name="source.wav",
        normalized_file_name="analysis.wav",
        metadata_file_name="metadata.json",
        analysis_version="baseline-librosa-v1",
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-01T00:00:00+00:00",
        error="preserve top-level error",
        analysis_error="preserve analysis error",
        separation_status="completed",
        separation_stage="completed",
        separation_progress=100,
        separation_message="Stems completed.",
        separation_version="stem-separation-v1",
        separation_model="htdemucs",
        stem_manifest_file_name=MANIFEST_PATH,
        separated_at="2026-08-02T00:00:00+00:00",
        separation_error="preserve separation error",
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def test_fresh_database_defaults(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)

    with db.connect(database) as connection:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }

    assert TRANSCRIPTION_COLUMNS <= set(columns)
    assert columns["transcription_status"]["dflt_value"] == "'not_started'"
    assert columns["transcription_stage"]["dflt_value"] == "'not_started'"
    assert columns["transcription_progress"]["dflt_value"] == "0"

    job = db.create_job(database, "fresh", source_type="upload")
    assert job["transcription_status"] == "not_started"
    assert job["transcription_stage"] == "not_started"
    assert job["transcription_progress"] == 0
    assert job["transcription_message"] is None
    assert job["transcription_version"] is None
    assert job["transcription_artifact_file_name"] is None
    assert job["transcribed_at"] is None
    assert job["pitched_event_count"] is None
    assert job["percussion_event_count"] is None
    assert job["aligned_event_count"] is None
    assert job["transcription_error"] is None


def test_original_database_migrates_without_data_loss(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_original_database(database)

    db.init_database(database)
    job = db.get_job(database, "legacy-original")

    assert job is not None
    assert job["title"] == "Original job"
    assert job["progress"] == 100
    assert job["preparation_status"] == "completed"
    assert job["analysis_status"] == "not_started"
    assert job["transcription_status"] == "not_started"
    assert job["transcription_stage"] == "not_started"
    assert job["transcription_progress"] == 0


def test_pre_transcription_database_preserves_analysis_and_stems(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_pre_transcription_database(database)
    before = db.get_job(database, "pre-transcription")

    db.init_database(database)
    after = db.get_job(database, "pre-transcription")

    assert before is not None and after is not None
    preserved = {
        "status",
        "stage",
        "progress",
        "message",
        "error",
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "analysis_json_file_name",
        "analyzed_at",
        "analysis_error",
        "separation_status",
        "separation_stage",
        "separation_progress",
        "separation_message",
        "separation_version",
        "separation_model",
        "stem_manifest_file_name",
        "separated_at",
        "separation_error",
    }
    assert {key: after[key] for key in preserved} == {
        key: before[key] for key in preserved
    }
    assert after["transcription_status"] == "not_started"
    assert after["transcription_stage"] == "not_started"
    assert after["transcription_progress"] == 0


def test_blank_legacy_states_and_invalid_counts_are_normalized(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_pre_transcription_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN transcription_status TEXT")
        connection.execute("ALTER TABLE jobs ADD COLUMN transcription_stage TEXT")
        connection.execute("ALTER TABLE jobs ADD COLUMN transcription_progress REAL")
        connection.execute("ALTER TABLE jobs ADD COLUMN pitched_event_count INTEGER")
        connection.execute("ALTER TABLE jobs ADD COLUMN percussion_event_count INTEGER")
        connection.execute("ALTER TABLE jobs ADD COLUMN aligned_event_count INTEGER")
        connection.execute(
            """
            UPDATE jobs
            SET transcription_status = '  ',
                transcription_stage = NULL,
                transcription_progress = -4,
                pitched_event_count = -1,
                percussion_event_count = -2,
                aligned_event_count = -3
            """
        )

    db.init_database(database)
    job = db.get_job(database, "pre-transcription")

    assert job is not None
    assert job["transcription_status"] == "not_started"
    assert job["transcription_stage"] == "not_started"
    assert job["transcription_progress"] == 0
    assert job["pitched_event_count"] is None
    assert job["percussion_event_count"] is None
    assert job["aligned_event_count"] is None


def test_repeated_initialization_is_idempotent(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_original_database(database)

    db.init_database(database)
    first = db.get_job(database, "legacy-original")
    db.init_database(database)
    second = db.get_job(database, "legacy-original")

    with db.connect(database) as connection:
        columns = [
            row["name"]
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        ]
    assert first == second
    assert len(columns) == len(set(columns))


def test_update_job_accepts_transcription_lifecycle_fields(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "update")

    db.update_job(
        database,
        "update",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Raw transcription completed.",
        transcription_version="cycle-4b-v1",
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=7,
        percussion_event_count=5,
        aligned_event_count=11,
        transcription_error=None,
    )
    job = db.get_job(database, "update")

    assert job is not None
    assert job["transcription_status"] == "completed"
    assert job["transcription_stage"] == "completed"
    assert job["transcription_progress"] == 100
    assert job["transcription_message"] == "Raw transcription completed."
    assert job["transcription_version"] == "cycle-4b-v1"
    assert job["transcription_artifact_file_name"] == ARTIFACT_PATH
    assert job["pitched_event_count"] == 7
    assert job["percussion_event_count"] == 5
    assert job["aligned_event_count"] == 11
    assert job["transcription_error"] is None


@pytest.mark.parametrize("bad", [-1, True, 1.5, "1"])
def test_update_job_rejects_invalid_event_counts(
    tmp_path: Path,
    bad: object,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_analyzed_job(database, "bad-count")

    with pytest.raises(ValueError):
        db.update_job(database, "bad-count", pitched_event_count=bad)

    assert db.get_job(database, "bad-count") == before


def test_database_constraint_rejects_direct_negative_counts(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "constraint")

    with pytest.raises(sqlite3.IntegrityError):
        with db.connect(database) as connection:
            connection.execute(
                "UPDATE jobs SET aligned_event_count = -1 WHERE id = ?",
                ("constraint",),
            )


def test_not_started_claim_succeeds_and_preserves_other_lifecycles(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    before = create_analyzed_job(database, "claim-new")

    claimed = db.claim_transcription_attempt(
        database,
        "claim-new",
        transcription_version="cycle-4b-exact",
        message="Preparing exact raw transcription.",
    )
    after = db.get_job(database, "claim-new")

    assert claimed is True
    assert after is not None
    assert after["transcription_status"] == "processing"
    assert after["transcription_stage"] == "preparing_transcription"
    assert after["transcription_progress"] == 1
    assert after["transcription_message"] == "Preparing exact raw transcription."
    assert after["transcription_version"] == "cycle-4b-exact"
    assert after["transcription_error"] is None
    preserved = {
        "status",
        "stage",
        "progress",
        "message",
        "error",
        "preparation_status",
        "analysis_status",
        "analysis_error",
        "separation_status",
        "stem_manifest_file_name",
        "separated_at",
        "separation_error",
    }
    assert {key: after[key] for key in preserved} == {
        key: before[key] for key in preserved
    }


@pytest.mark.parametrize(
    ("preparation_status", "analysis_status"),
    [
        ("pending", "completed"),
        ("failed", "completed"),
        ("completed", "not_started"),
        ("completed", "processing"),
        ("completed", "failed"),
    ],
)
def test_claim_requires_completed_preparation_and_analysis(
    tmp_path: Path,
    preparation_status: str,
    analysis_status: str,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "incomplete")
    db.update_job(
        database,
        "incomplete",
        preparation_status=preparation_status,
        analysis_status=analysis_status,
    )
    expected = db.get_job(database, "incomplete")

    claimed = db.claim_transcription_attempt(
        database,
        "incomplete",
        transcription_version="cycle-4b-v1",
    )

    assert claimed is False
    assert db.get_job(database, "incomplete") == expected


def test_failed_retry_preserves_last_success_until_replacement(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "retry")
    db.update_job(
        database,
        "retry",
        transcription_status="failed",
        transcription_stage="failed",
        transcription_progress=68,
        transcription_message="Previous attempt failed.",
        transcription_version="old-version",
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=8,
        percussion_event_count=6,
        aligned_event_count=12,
        transcription_error="Current transcription failure.",
    )

    claimed = db.claim_transcription_attempt(
        database,
        "retry",
        transcription_version="new-version",
    )
    job = db.get_job(database, "retry")

    assert claimed is True
    assert job is not None
    assert job["transcription_status"] == "processing"
    assert job["transcription_stage"] == "preparing_transcription"
    assert job["transcription_progress"] == 1
    assert job["transcription_version"] == "new-version"
    assert job["transcription_error"] is None
    assert job["transcription_artifact_file_name"] == ARTIFACT_PATH
    assert job["transcribed_at"] == "2026-08-03T00:00:00+00:00"
    assert job["pitched_event_count"] == 8
    assert job["percussion_event_count"] == 6
    assert job["aligned_event_count"] == 12
    assert job["error"] == "preserve top-level error"
    assert job["analysis_error"] == "preserve analysis error"
    assert job["separation_error"] == "preserve separation error"


def test_completed_claim_requires_explicit_force(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "completed")
    db.update_job(
        database,
        "completed",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Already completed.",
        transcription_version="old-version",
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=4,
        percussion_event_count=3,
        aligned_event_count=6,
    )
    before = db.get_job(database, "completed")

    assert (
        db.claim_transcription_attempt(
            database,
            "completed",
            transcription_version="new-version",
        )
        is False
    )
    assert db.get_job(database, "completed") == before

    assert (
        db.claim_transcription_attempt(
            database,
            "completed",
            transcription_version="new-version",
            force=True,
        )
        is True
    )
    after = db.get_job(database, "completed")
    assert after is not None
    assert after["transcription_status"] == "processing"
    assert after["transcription_version"] == "new-version"
    assert after["transcription_artifact_file_name"] == ARTIFACT_PATH
    assert after["transcribed_at"] == "2026-08-03T00:00:00+00:00"
    assert after["pitched_event_count"] == 4
    assert after["percussion_event_count"] == 3
    assert after["aligned_event_count"] == 6


@pytest.mark.parametrize("force", [False, True])
def test_processing_claim_never_overlaps_active_run(
    tmp_path: Path,
    force: bool,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "processing")
    db.update_job(
        database,
        "processing",
        transcription_status="processing",
        transcription_stage="detecting_events",
        transcription_progress=47,
        transcription_message="Already running.",
        transcription_version="active-version",
        transcription_error="preserve current value",
    )
    before = db.get_job(database, "processing")

    claimed = db.claim_transcription_attempt(
        database,
        "processing",
        transcription_version="replacement-version",
        force=force,
    )

    assert claimed is False
    assert db.get_job(database, "processing") == before


def test_missing_job_claim_is_rejected(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)

    assert (
        db.claim_transcription_attempt(
            database,
            "missing",
            transcription_version="cycle-4b-v1",
        )
        is False
    )
    assert db.get_job(database, "missing") is None


def test_concurrent_claims_allow_exactly_one_winner(tmp_path: Path) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "race")
    db.update_job(
        database,
        "race",
        transcription_status="failed",
        transcription_stage="failed",
        transcription_progress=55,
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=5,
        percussion_event_count=4,
        aligned_event_count=8,
        transcription_error="Retryable failure.",
    )
    barrier = Barrier(2)

    def attempt(number: int) -> bool:
        barrier.wait(timeout=5)
        return db.claim_transcription_attempt(
            database,
            "race",
            transcription_version=f"race-version-{number}",
            message=f"Race attempt {number}.",
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(attempt, (1, 2)))

    job = db.get_job(database, "race")
    assert sorted(results) == [False, True]
    assert job is not None
    assert job["transcription_status"] == "processing"
    winner = job["transcription_version"].removeprefix("race-version-")
    assert job["transcription_message"] == f"Race attempt {winner}."
    assert job["transcription_artifact_file_name"] == ARTIFACT_PATH
    assert job["pitched_event_count"] == 5
    assert job["percussion_event_count"] == 4
    assert job["aligned_event_count"] == 8


def test_restart_marks_only_active_transcription_retryable_and_preserves_outputs(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "restart")
    db.update_job(
        database,
        "restart",
        status="processing",
        stage="transcribing",
        progress=100,
        message="Raw transcription is running.",
        transcription_status="processing",
        transcription_stage="aligning_events",
        transcription_progress=79,
        transcription_message="Aligning raw events.",
        transcription_version="new-version",
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=9,
        percussion_event_count=7,
        aligned_event_count=14,
        transcription_error=None,
    )

    db.fail_incomplete_jobs(database)
    job = db.get_job(database, "restart")

    assert job is not None
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["progress"] == 100
    assert job["preparation_status"] == "completed"
    assert job["analysis_status"] == "completed"
    assert job["analysis_error"] == "preserve analysis error"
    assert job["separation_status"] == "completed"
    assert job["stem_manifest_file_name"] == MANIFEST_PATH
    assert job["separation_error"] == "preserve separation error"
    assert job["transcription_status"] == "failed"
    assert job["transcription_stage"] == "failed"
    assert job["transcription_progress"] == 79
    assert job["transcription_artifact_file_name"] == ARTIFACT_PATH
    assert job["transcribed_at"] == "2026-08-03T00:00:00+00:00"
    assert job["pitched_event_count"] == 9
    assert job["percussion_event_count"] == 7
    assert job["aligned_event_count"] == 14
    assert "remains available" in job["transcription_message"]
    assert "retried" in job["transcription_message"]
    assert job["transcription_error"] == (
        "Raw transcription was interrupted by a server restart."
    )
    assert "/" not in job["transcription_message"]
    assert "\\" not in job["transcription_message"]
    assert "/" not in job["transcription_error"]
    assert "\\" not in job["transcription_error"]
    assert "audio analysis was interrupted" not in job["message"].lower()


def test_restart_leaves_non_active_transcription_states_unchanged(
    tmp_path: Path,
) -> None:
    database = tmp_path / "popex.sqlite3"
    create_analyzed_job(database, "completed-state")
    db.update_job(
        database,
        "completed-state",
        transcription_status="completed",
        transcription_stage="completed",
        transcription_progress=100,
        transcription_message="Done.",
        transcription_artifact_file_name=ARTIFACT_PATH,
        transcribed_at="2026-08-03T00:00:00+00:00",
        pitched_event_count=1,
        percussion_event_count=2,
        aligned_event_count=3,
    )
    before = db.get_job(database, "completed-state")

    db.fail_incomplete_jobs(database)

    assert db.get_job(database, "completed-state") == before
