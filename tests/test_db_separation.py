import sqlite3
from pathlib import Path

from app import db


MANIFEST_PATH = "stems/stem-separation.json"
SEPARATION_COLUMNS = {
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
                'https://youtube.com/watch?v=old',
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


def create_analysis_era_database(database: Path) -> None:
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
                analysis_error TEXT
            )
            """
        )
        connection.execute(
            """
            INSERT INTO jobs (
                id,
                source_url,
                status,
                progress,
                title,
                uploader,
                duration_seconds,
                error,
                created_at,
                updated_at,
                source_type,
                stage,
                message,
                original_filename,
                source_format,
                sample_rate,
                channel_count,
                source_file_name,
                normalized_file_name,
                metadata_file_name,
                preparation_status,
                analysis_status,
                analysis_version,
                tempo_bpm,
                tempo_confidence,
                key_symbol,
                key_confidence,
                analysis_json_file_name,
                analyzed_at,
                analysis_error
            ) VALUES (
                'analysis-era',
                '',
                'completed',
                100,
                'Analyzed job',
                NULL,
                12.5,
                'preserved top-level error',
                '2026-02-01',
                '2026-02-02',
                'upload',
                'completed',
                'Analysis completed.',
                'song.wav',
                'wav',
                44100,
                2,
                'source-abc.wav',
                'analysis.wav',
                'metadata.json',
                'completed',
                'completed',
                'baseline-librosa-v1',
                120.0,
                0.8,
                'A minor',
                0.7,
                'analysis/audio-analysis.json',
                '2026-02-02T00:00:00+00:00',
                'preserved analysis error'
            )
            """
        )


def create_completed_job(database: Path, job_id: str = "completed") -> dict:
    db.init_database(database)
    db.create_job(database, job_id, source_type="upload", original_filename="song.wav")
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
        tempo_bpm=118.0,
        tempo_confidence=0.81,
        key_symbol="C major",
        key_confidence=0.73,
        analysis_json_file_name="analysis/audio-analysis.json",
        analyzed_at="2026-08-01T00:00:00+00:00",
        error="preserve top-level error",
        analysis_error="preserve analysis error",
    )
    result = db.get_job(database, job_id)
    assert result is not None
    return result


def test_fresh_database_defaults(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)

    with db.connect(database) as connection:
        columns = {
            row["name"]: row
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
    assert SEPARATION_COLUMNS <= set(columns)
    assert columns["separation_status"]["dflt_value"] == "'not_started'"
    assert columns["separation_stage"]["dflt_value"] == "'not_started'"
    assert columns["separation_progress"]["dflt_value"] == "0"

    job = db.create_job(database, "fresh", source_type="upload")
    assert job["separation_status"] == "not_started"
    assert job["separation_stage"] == "not_started"
    assert job["separation_progress"] == 0
    assert job["separation_message"] is None
    assert job["stem_manifest_file_name"] is None
    assert job["separation_error"] is None


def test_migration_from_original_pre_analysis_schema(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_original_database(database)

    db.init_database(database)
    job = db.get_job(database, "legacy-original")

    assert job is not None
    assert job["title"] == "Original job"
    assert job["progress"] == 100
    assert job["preparation_status"] == "completed"
    assert job["analysis_status"] == "not_started"
    assert job["separation_status"] == "not_started"
    assert job["separation_stage"] == "not_started"
    assert job["separation_progress"] == 0


def test_migration_from_current_analysis_era_schema(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_analysis_era_database(database)

    db.init_database(database)
    job = db.get_job(database, "analysis-era")

    assert job is not None
    assert job["separation_status"] == "not_started"
    assert job["separation_stage"] == "not_started"
    assert job["separation_progress"] == 0
    assert job["separation_message"] is None
    assert job["separation_version"] is None
    assert job["separation_model"] is None
    assert job["stem_manifest_file_name"] is None
    assert job["separated_at"] is None
    assert job["separation_error"] is None


def test_completed_preparation_and_analysis_values_are_preserved(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_analysis_era_database(database)

    before = db.get_job(database, "analysis-era")
    db.init_database(database)
    after = db.get_job(database, "analysis-era")

    assert before is not None and after is not None
    preserved = {
        "status",
        "progress",
        "title",
        "error",
        "stage",
        "message",
        "source_file_name",
        "normalized_file_name",
        "metadata_file_name",
        "preparation_status",
        "analysis_status",
        "analysis_version",
        "tempo_bpm",
        "tempo_confidence",
        "key_symbol",
        "key_confidence",
        "analysis_json_file_name",
        "analyzed_at",
        "analysis_error",
    }
    assert {key: after[key] for key in preserved} == {
        key: before[key] for key in preserved
    }


def test_interrupted_separation_becomes_retryable_failure(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    before = create_completed_job(database, "interrupted")
    db.update_job(
        database,
        "interrupted",
        status="processing",
        stage="separating_stems",
        message="Separating stems.",
        separation_status="processing",
        separation_stage="separating_stems",
        separation_progress=47.5,
        separation_message="Separating stems.",
        separation_version="stem-separation-v1",
        separation_model="htdemucs",
        separation_error=None,
    )

    db.fail_incomplete_jobs(database)
    job = db.get_job(database, "interrupted")

    assert job is not None
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["progress"] == before["progress"] == 100
    assert job["preparation_status"] == "completed"
    assert job["analysis_status"] == "completed"
    assert job["separation_status"] == "failed"
    assert job["separation_stage"] == "failed"
    assert job["separation_progress"] == 47.5
    assert job["separation_error"] == (
        "Stem separation was interrupted by a server restart."
    )
    assert "remains available" in job["separation_message"]
    assert "retried" in job["separation_message"]
    assert job["source_file_name"] == "source.wav"
    assert job["normalized_file_name"] == "analysis.wav"
    assert job["analysis_json_file_name"] == "analysis/audio-analysis.json"
    assert job["analysis_error"] == "preserve analysis error"
    assert job["error"] == "preserve top-level error"


def test_previous_successful_manifest_survives_interrupted_retry(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_completed_job(database, "retry")
    db.update_job(
        database,
        "retry",
        separation_status="processing",
        separation_stage="separating_stems",
        separation_progress=64,
        separation_version="stem-separation-v2",
        separation_model="htdemucs_ft",
        stem_manifest_file_name=MANIFEST_PATH,
        separated_at="2026-08-02T00:00:00+00:00",
    )

    db.fail_incomplete_jobs(database)
    job = db.get_job(database, "retry")

    assert job is not None
    assert job["status"] == "completed"
    assert job["stage"] == "completed"
    assert job["progress"] == 100
    assert job["message"] == "Analysis completed."
    assert job["error"] == "preserve top-level error"
    assert job["analysis_error"] == "preserve analysis error"
    assert job["separation_status"] == "failed"
    assert job["separation_progress"] == 64
    assert job["stem_manifest_file_name"] == MANIFEST_PATH
    assert job["separated_at"] == "2026-08-02T00:00:00+00:00"


def test_repeated_init_database_is_idempotent(tmp_path: Path):
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
    assert len(columns) == len(set(columns))
    assert first == second


def test_update_job_accepts_new_separation_fields(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    db.init_database(database)
    db.create_job(database, "update", source_type="upload")

    db.update_job(
        database,
        "update",
        separation_status="completed",
        separation_stage="completed",
        separation_progress=100,
        separation_message="Stem separation completed.",
        separation_version="stem-separation-v1",
        separation_model="htdemucs",
        stem_manifest_file_name=MANIFEST_PATH,
        separated_at="2026-08-03T00:00:00+00:00",
        separation_error=None,
    )
    job = db.get_job(database, "update")

    assert job is not None
    assert job["separation_status"] == "completed"
    assert job["separation_stage"] == "completed"
    assert job["separation_progress"] == 100
    assert job["separation_message"] == "Stem separation completed."
    assert job["separation_version"] == "stem-separation-v1"
    assert job["separation_model"] == "htdemucs"
    assert job["stem_manifest_file_name"] == MANIFEST_PATH
    assert job["separated_at"] == "2026-08-03T00:00:00+00:00"
    assert job["separation_error"] is None


def test_unrelated_jobs_are_unchanged(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_completed_job(database, "unrelated")
    db.update_job(
        database,
        "unrelated",
        separation_status="completed",
        separation_stage="completed",
        separation_progress=100,
        separation_message="Already done.",
        stem_manifest_file_name=MANIFEST_PATH,
        separated_at="2026-08-01T00:00:00+00:00",
    )
    before = db.get_job(database, "unrelated")

    db.fail_incomplete_jobs(database)
    after = db.get_job(database, "unrelated")

    assert before == after


def test_restart_does_not_start_separation_automatically(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_completed_job(database, "not-started")

    db.fail_incomplete_jobs(database)
    first = db.get_job(database, "not-started")
    db.init_database(database)
    db.fail_incomplete_jobs(database)
    second = db.get_job(database, "not-started")

    assert first is not None and second is not None
    assert first["separation_status"] == "not_started"
    assert first["separation_stage"] == "not_started"
    assert first["separation_progress"] == 0
    assert second["separation_status"] == "not_started"
    assert second["separation_stage"] == "not_started"
    assert second["separation_progress"] == 0


def test_null_and_empty_separation_state_is_normalized(tmp_path: Path):
    database = tmp_path / "popex.sqlite3"
    create_analysis_era_database(database)
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE jobs ADD COLUMN separation_status TEXT")
        connection.execute("ALTER TABLE jobs ADD COLUMN separation_stage TEXT")
        connection.execute(
            """
            UPDATE jobs
            SET separation_status = NULL,
                separation_stage = ''
            WHERE id = 'analysis-era'
            """
        )

    db.init_database(database)
    job = db.get_job(database, "analysis-era")

    assert job is not None
    assert job["separation_status"] == "not_started"
    assert job["separation_stage"] == "not_started"
