from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def fail_incomplete_jobs(database_path: Path) -> None:
    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE jobs
            SET status = 'failed',
                error = 'The server restarted before this extraction completed.',
                updated_at = ?
            WHERE status IN ('queued', 'processing')
            """,
            (now,),
        )


def create_job(database_path: Path, job_id: str, source_url: str) -> dict[str, Any]:
    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            INSERT INTO jobs (
                id, source_url, status, progress, created_at, updated_at
            ) VALUES (?, ?, 'queued', 0, ?, ?)
            """,
            (job_id, source_url, now, now),
        )
    job = get_job(database_path, job_id)
    if job is None:
        raise RuntimeError("Job was not created")
    return job


def update_job(database_path: Path, job_id: str, **fields: Any) -> None:
    allowed = {
        "status",
        "progress",
        "title",
        "uploader",
        "duration_seconds",
        "error",
    }
    values = {key: value for key, value in fields.items() if key in allowed}
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


def get_job(database_path: Path, job_id: str) -> dict[str, Any] | None:
    with connect(database_path) as connection:
        row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
    return dict(row) if row else None


def list_jobs(database_path: Path, limit: int = 50) -> list[dict[str, Any]]:
    with connect(database_path) as connection:
        rows = connection.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(row) for row in rows]
