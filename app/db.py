from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


_TRANSCRIPTION_COUNT_FIELDS = frozenset(
    {
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
    }
)

NEW_COLUMNS: dict[str, str] = {
    "source_type": "TEXT NOT NULL DEFAULT 'url'",
    "stage": "TEXT NOT NULL DEFAULT 'queued'",
    "message": "TEXT",
    "original_filename": "TEXT",
    "source_format": "TEXT",
    "sample_rate": "INTEGER",
    "channel_count": "INTEGER",
    "source_file_name": "TEXT",
    "normalized_file_name": "TEXT",
    "metadata_file_name": "TEXT",
    "preparation_status": "TEXT NOT NULL DEFAULT 'pending'",
    "analysis_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "analysis_version": "TEXT",
    "tempo_bpm": "REAL",
    "tempo_confidence": "REAL",
    "key_symbol": "TEXT",
    "key_confidence": "REAL",
    "analysis_json_file_name": "TEXT",
    "analyzed_at": "TEXT",
    "analysis_error": "TEXT",
    "separation_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "separation_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "separation_progress": "REAL NOT NULL DEFAULT 0",
    "separation_message": "TEXT",
    "separation_version": "TEXT",
    "separation_model": "TEXT",
    "stem_manifest_file_name": "TEXT",
    "separated_at": "TEXT",
    "separation_error": "TEXT",
    "transcription_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "transcription_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "transcription_progress": "REAL NOT NULL DEFAULT 0",
    "transcription_message": "TEXT",
    "transcription_version": "TEXT",
    "transcription_artifact_file_name": "TEXT",
    "transcribed_at": "TEXT",
    "pitched_event_count": (
        "INTEGER CHECK (pitched_event_count IS NULL OR pitched_event_count >= 0)"
    ),
    "percussion_event_count": (
        "INTEGER CHECK (percussion_event_count IS NULL OR percussion_event_count >= 0)"
    ),
    "aligned_event_count": (
        "INTEGER CHECK (aligned_event_count IS NULL OR aligned_event_count >= 0)"
    ),
    "transcription_error": "TEXT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def init_database(database_path: Path) -> None:
    with connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS jobs (
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
                separation_error TEXT,
                transcription_status TEXT NOT NULL DEFAULT 'not_started',
                transcription_stage TEXT NOT NULL DEFAULT 'not_started',
                transcription_progress REAL NOT NULL DEFAULT 0,
                transcription_message TEXT,
                transcription_version TEXT,
                transcription_artifact_file_name TEXT,
                transcribed_at TEXT,
                pitched_event_count INTEGER
                    CHECK (pitched_event_count IS NULL OR pitched_event_count >= 0),
                percussion_event_count INTEGER
                    CHECK (
                        percussion_event_count IS NULL
                        OR percussion_event_count >= 0
                    ),
                aligned_event_count INTEGER
                    CHECK (aligned_event_count IS NULL OR aligned_event_count >= 0),
                transcription_error TEXT
            )
            """
        )
        existing = {
            row["name"]
            for row in connection.execute("PRAGMA table_info(jobs)").fetchall()
        }
        for column, definition in NEW_COLUMNS.items():
            if column not in existing:
                connection.execute(
                    f"ALTER TABLE jobs ADD COLUMN {column} {definition}"
                )

        connection.execute(
            """
            UPDATE jobs
            SET source_type = COALESCE(NULLIF(source_type, ''), 'url'),
                metadata_file_name = CASE
                    WHEN normalized_file_name IS NOT NULL
                         AND normalized_file_name != ''
                        THEN COALESCE(NULLIF(metadata_file_name, ''), 'metadata.json')
                    ELSE metadata_file_name
                END,
                preparation_status = CASE
                    WHEN normalized_file_name IS NOT NULL
                         AND normalized_file_name != '' THEN 'completed'
                    WHEN status = 'completed' THEN 'completed'
                    WHEN status = 'failed' THEN 'failed'
                    WHEN preparation_status IS NULL
                         OR preparation_status = '' THEN 'pending'
                    ELSE preparation_status
                END,
                stage = CASE
                    WHEN normalized_file_name IS NOT NULL
                         AND normalized_file_name != ''
                         AND analysis_status = 'failed' THEN 'completed'
                    WHEN status = 'completed'
                         AND (stage IS NULL OR stage = '' OR stage = 'queued')
                        THEN 'completed'
                    WHEN status = 'failed'
                         AND (stage IS NULL OR stage = '' OR stage = 'queued')
                        THEN 'failed'
                    WHEN status = 'processing'
                         AND (stage IS NULL OR stage = '' OR stage = 'queued')
                        THEN 'importing'
                    WHEN stage IS NULL OR stage = '' THEN 'queued'
                    ELSE stage
                END,
                analysis_status = COALESCE(
                    NULLIF(analysis_status, ''),
                    'not_started'
                ),
                separation_status = COALESCE(
                    NULLIF(separation_status, ''),
                    'not_started'
                ),
                separation_stage = COALESCE(
                    NULLIF(separation_stage, ''),
                    'not_started'
                ),
                transcription_status = COALESCE(
                    NULLIF(TRIM(transcription_status), ''),
                    'not_started'
                ),
                transcription_stage = COALESCE(
                    NULLIF(TRIM(transcription_stage), ''),
                    'not_started'
                ),
                transcription_progress = CASE
                    WHEN transcription_progress IS NULL
                         OR transcription_progress < 0 THEN 0
                    ELSE transcription_progress
                END,
                pitched_event_count = CASE
                    WHEN pitched_event_count < 0 THEN NULL
                    ELSE pitched_event_count
                END,
                percussion_event_count = CASE
                    WHEN percussion_event_count < 0 THEN NULL
                    ELSE percussion_event_count
                END,
                aligned_event_count = CASE
                    WHEN aligned_event_count < 0 THEN NULL
                    ELSE aligned_event_count
                END
            """
        )


def fail_incomplete_jobs(database_path: Path) -> None:
    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = CASE
                    WHEN preparation_status = 'completed' THEN 'completed'
                    ELSE 'failed'
                END,
                stage = CASE
                    WHEN preparation_status = 'completed' THEN 'completed'
                    ELSE 'failed'
                END,
                progress = CASE
                    WHEN preparation_status = 'completed' AND progress >= 100
                        THEN 95
                    ELSE progress
                END,
                message = CASE
                    WHEN preparation_status = 'completed'
                        THEN 'Source preparation is complete; audio analysis was interrupted.'
                    ELSE 'Processing was interrupted by a server restart.'
                END,
                error = CASE
                    WHEN preparation_status = 'completed' THEN NULL
                    ELSE 'The server restarted before source preparation completed.'
                END,
                analysis_status = CASE
                    WHEN analysis_status = 'processing' THEN 'failed'
                    ELSE analysis_status
                END,
                analysis_error = CASE
                    WHEN analysis_status = 'processing'
                        THEN 'Audio analysis was interrupted by a server restart.'
                    ELSE analysis_error
                END,
                updated_at = ?
            WHERE status IN ('queued', 'processing')
              AND separation_status != 'processing'
              AND transcription_status != 'processing'
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE jobs
            SET status = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed' THEN 'completed'
                    ELSE status
                END,
                stage = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed' THEN 'completed'
                    ELSE stage
                END,
                message = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                        THEN 'Prepared source and prior analysis remain available; stem separation can be retried.'
                    ELSE message
                END,
                separation_status = 'failed',
                separation_stage = 'failed',
                separation_message = 'Prepared and analyzed audio remains available; stem separation can be retried.',
                separation_error = 'Stem separation was interrupted by a server restart.',
                updated_at = ?
            WHERE separation_status = 'processing'
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE jobs
            SET status = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed' THEN 'completed'
                    ELSE status
                END,
                stage = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed' THEN 'completed'
                    ELSE stage
                END,
                message = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                        THEN 'Prepared and analyzed audio remains available; raw transcription can be retried.'
                    ELSE message
                END,
                transcription_status = 'failed',
                transcription_stage = 'failed',
                transcription_message = 'Prepared and analyzed audio remains available; raw transcription can be retried.',
                transcription_error = 'Raw transcription was interrupted by a server restart.',
                updated_at = ?
            WHERE transcription_status = 'processing'
            """,
            (now,),
        )


def create_job(
    database_path: Path,
    job_id: str,
    *,
    source_type: str,
    source_url: str = "",
    original_filename: str | None = None,
) -> dict[str, Any]:
    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id,
                source_url,
                source_type,
                original_filename,
                status,
                stage,
                progress,
                message,
                preparation_status,
                analysis_status,
                created_at,
                updated_at
            ) VALUES (
                ?, ?, ?, ?,
                'queued', 'queued', 0, 'Waiting to start.',
                'pending', 'not_started', ?, ?
            )
            """,
            (job_id, source_url, source_type, original_filename, now, now),
        )
    job = get_job(database_path, job_id)
    if job is None:
        raise RuntimeError("Job was not created")
    return job


def claim_separation_attempt(
    database_path: Path,
    job_id: str,
    *,
    separation_version: str,
    separation_model: str,
    message: str = "Preparing stem separation.",
) -> bool:
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET separation_status = 'processing',
                separation_stage = 'preparing_separation',
                separation_progress = 1,
                separation_message = ?,
                separation_version = ?,
                separation_model = ?,
                separation_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND preparation_status = 'completed'
              AND separation_status IN ('not_started', 'failed')
            """,
            (
                message,
                separation_version,
                separation_model,
                now,
                job_id,
            ),
        )
        return cursor.rowcount == 1


def claim_transcription_attempt(
    database_path: Path,
    job_id: str,
    *,
    transcription_version: str,
    force: bool = False,
    message: str = "Preparing raw transcription.",
) -> bool:
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET transcription_status = 'processing',
                transcription_stage = 'preparing_transcription',
                transcription_progress = 1,
                transcription_message = ?,
                transcription_version = ?,
                transcription_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND preparation_status = 'completed'
              AND analysis_status = 'completed'
              AND (
                    transcription_status IN ('not_started', 'failed')
                    OR (? = 1 AND transcription_status = 'completed')
              )
            """,
            (
                message,
                transcription_version,
                now,
                job_id,
                int(bool(force)),
            ),
        )
        return cursor.rowcount == 1


def update_job(database_path: Path, job_id: str, **fields: Any) -> None:
    allowed = set(NEW_COLUMNS) | {
        "source_url",
        "status",
        "progress",
        "title",
        "uploader",
        "duration_seconds",
        "error",
    }
    values = {key: value for key, value in fields.items() if key in allowed}
    _validate_transcription_counts(values)
    if not values:
        return
    values["updated_at"] = utc_now()
    assignments = ", ".join(f"{key} = ?" for key in values)
    parameters = [*values.values(), job_id]
    with connect(database_path) as connection:
        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            parameters,
        )


def _validate_transcription_counts(values: dict[str, Any]) -> None:
    for field in _TRANSCRIPTION_COUNT_FIELDS:
        value = values.get(field)
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer or None.")


def get_job(database_path: Path, job_id: str) -> dict[str, Any] | None:
    with connect(database_path) as connection:
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?", (job_id,)
        ).fetchone()
    return dict(row) if row else None


def list_jobs(database_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    with connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
