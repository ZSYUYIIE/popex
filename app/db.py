from __future__ import annotations

import math
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.score_artifacts import (
    is_score_artifact_file_name as _is_valid_score_artifact_file_name,
    score_attempt_artifact_file_name,
)
from app.score_sources import current_score_fingerprint, is_score_fingerprint


_TRANSCRIPTION_COUNT_FIELDS = frozenset(
    {
        "pitched_event_count",
        "percussion_event_count",
        "aligned_event_count",
    }
)
_INTERPRETATION_COUNT_FIELDS = frozenset(
    {
        "interpretation_part_count",
        "interpretation_phrase_count",
        "interpretation_pitched_item_count",
        "interpretation_percussion_item_count",
        "interpretation_warning_count",
    }
)
_HARMONY_COUNT_FIELDS = frozenset(
    {
        "harmony_event_count",
        "harmony_segment_count",
        "harmony_resolved_segment_count",
        "harmony_unresolved_segment_count",
        "harmony_unresolved_event_count",
        "harmony_warning_count",
    }
)
_HARMONY_BOOLEAN_FIELDS = frozenset(
    {"harmony_used_interpretation_context"}
)
_HARMONY_ARTIFACT_FILE_NAME = "harmony/harmonic-context.json"
_RAW_TRANSCRIPTION_ARTIFACT_FILE_NAME = "transcription/raw-events.json"
_HARMONY_VERSION_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_HARMONY_ATTEMPT_ID_RE = re.compile(r"[a-f0-9]{32}")
_HARMONY_ATTEMPT_ARTIFACT_RE = re.compile(
    r"harmony/harmonic-context\.([a-f0-9]{32})\.json"
)
_HARMONY_UNSAFE_ERROR_RE = re.compile(
    r"traceback|https?://|(?:^|\s)[A-Za-z]:[\\/]|\\\\|"
    r"(?:^|\s)/(?:[^\s/]+/)+|"
    r"\b(?:token|password|secret|authorization|api[_-]?key|access[_-]?key)"
    r"\b\s*[:=]",
    re.IGNORECASE,
)
_SCORE_COUNT_FIELDS = frozenset(
    {
        "score_measure_count",
        "score_note_count",
        "score_chord_symbol_count",
        "score_warning_count",
    }
)
_SCORE_STATUSES = frozenset({"not_started", "processing", "completed", "failed"})
_SCORE_VERSION_RE = _HARMONY_VERSION_RE

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
    "interpretation_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "interpretation_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "interpretation_progress": "REAL NOT NULL DEFAULT 0",
    "interpretation_message": "TEXT",
    "interpretation_version": "TEXT",
    "interpretation_artifact_file_name": "TEXT",
    "interpreted_at": "TEXT",
    "interpretation_part_count": (
        "INTEGER CHECK (interpretation_part_count IS NULL OR interpretation_part_count >= 0)"
    ),
    "interpretation_phrase_count": (
        "INTEGER CHECK (interpretation_phrase_count IS NULL OR interpretation_phrase_count >= 0)"
    ),
    "interpretation_pitched_item_count": (
        "INTEGER CHECK (interpretation_pitched_item_count IS NULL OR interpretation_pitched_item_count >= 0)"
    ),
    "interpretation_percussion_item_count": (
        "INTEGER CHECK (interpretation_percussion_item_count IS NULL OR interpretation_percussion_item_count >= 0)"
    ),
    "interpretation_warning_count": (
        "INTEGER CHECK (interpretation_warning_count IS NULL OR interpretation_warning_count >= 0)"
    ),
    "interpretation_error": "TEXT",
    "harmony_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "harmony_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "harmony_progress": (
        "REAL NOT NULL DEFAULT 0 "
        "CHECK (harmony_progress >= 0 AND harmony_progress <= 100)"
    ),
    "harmony_message": "TEXT",
    "harmony_attempt_id": "TEXT",
    "harmony_attempt_version": "TEXT",
    "harmony_version": "TEXT",
    "harmony_artifact_file_name": "TEXT",
    "harmonized_at": "TEXT",
    "harmony_source_transcription_version": "TEXT",
    "harmony_source_transcription_artifact_file_name": "TEXT",
    "harmony_source_transcribed_at": "TEXT",
    "harmony_event_count": (
        "INTEGER CHECK (harmony_event_count IS NULL OR harmony_event_count >= 0)"
    ),
    "harmony_segment_count": (
        "INTEGER CHECK (harmony_segment_count IS NULL OR harmony_segment_count >= 0)"
    ),
    "harmony_resolved_segment_count": (
        "INTEGER CHECK (harmony_resolved_segment_count IS NULL "
        "OR harmony_resolved_segment_count >= 0)"
    ),
    "harmony_unresolved_segment_count": (
        "INTEGER CHECK (harmony_unresolved_segment_count IS NULL "
        "OR harmony_unresolved_segment_count >= 0)"
    ),
    "harmony_unresolved_event_count": (
        "INTEGER CHECK (harmony_unresolved_event_count IS NULL "
        "OR harmony_unresolved_event_count >= 0)"
    ),
    "harmony_warning_count": (
        "INTEGER CHECK (harmony_warning_count IS NULL OR harmony_warning_count >= 0)"
    ),
    "harmony_used_interpretation_context": (
        "INTEGER CHECK (harmony_used_interpretation_context IS NULL "
        "OR harmony_used_interpretation_context IN (0, 1))"
    ),
    "harmony_error": "TEXT",
    "score_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "score_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "score_progress": (
        "REAL NOT NULL DEFAULT 0 "
        "CHECK (score_progress >= 0 AND score_progress <= 100)"
    ),
    "score_message": "TEXT",
    "score_attempt_id": "TEXT",
    "score_attempt_fingerprint": "TEXT",
    "score_version": "TEXT",
    "score_artifact_file_name": "TEXT",
    "scored_at": "TEXT",
    "score_source_fingerprint": "TEXT",
    "score_measure_count": (
        "INTEGER CHECK (score_measure_count IS NULL OR score_measure_count >= 0)"
    ),
    "score_note_count": (
        "INTEGER CHECK (score_note_count IS NULL OR score_note_count >= 0)"
    ),
    "score_chord_symbol_count": (
        "INTEGER CHECK (score_chord_symbol_count IS NULL "
        "OR score_chord_symbol_count >= 0)"
    ),
    "score_warning_count": (
        "INTEGER CHECK (score_warning_count IS NULL OR score_warning_count >= 0)"
    ),
    "score_error": "TEXT",
    "score_tablature_request": "TEXT",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def connect(database_path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(database_path, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.create_function(
        "is_valid_harmony_artifact",
        1,
        lambda value: 1 if _is_valid_harmony_artifact_file_name(value) else 0,
    )
    connection.create_function(
        "is_valid_score_artifact",
        1,
        lambda value: 1 if _is_valid_score_artifact_file_name(value) else 0,
    )
    connection.create_function(
        "is_valid_score_fingerprint",
        1,
        lambda value: 1 if is_score_fingerprint(value) else 0,
    )
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
                transcription_error TEXT,
                interpretation_status TEXT NOT NULL DEFAULT 'not_started',
                interpretation_stage TEXT NOT NULL DEFAULT 'not_started',
                interpretation_progress REAL NOT NULL DEFAULT 0,
                interpretation_message TEXT,
                interpretation_version TEXT,
                interpretation_artifact_file_name TEXT,
                interpreted_at TEXT,
                interpretation_part_count INTEGER
                    CHECK (
                        interpretation_part_count IS NULL
                        OR interpretation_part_count >= 0
                    ),
                interpretation_phrase_count INTEGER
                    CHECK (
                        interpretation_phrase_count IS NULL
                        OR interpretation_phrase_count >= 0
                    ),
                interpretation_pitched_item_count INTEGER
                    CHECK (
                        interpretation_pitched_item_count IS NULL
                        OR interpretation_pitched_item_count >= 0
                    ),
                interpretation_percussion_item_count INTEGER
                    CHECK (
                        interpretation_percussion_item_count IS NULL
                        OR interpretation_percussion_item_count >= 0
                    ),
                interpretation_warning_count INTEGER
                    CHECK (
                        interpretation_warning_count IS NULL
                        OR interpretation_warning_count >= 0
                    ),
                interpretation_error TEXT,
                harmony_status TEXT NOT NULL DEFAULT 'not_started',
                harmony_stage TEXT NOT NULL DEFAULT 'not_started',
                harmony_progress REAL NOT NULL DEFAULT 0
                    CHECK (harmony_progress >= 0 AND harmony_progress <= 100),
                harmony_message TEXT,
                harmony_attempt_id TEXT,
                harmony_attempt_version TEXT,
                harmony_version TEXT,
                harmony_artifact_file_name TEXT,
                harmonized_at TEXT,
                harmony_source_transcription_version TEXT,
                harmony_source_transcription_artifact_file_name TEXT,
                harmony_source_transcribed_at TEXT,
                harmony_event_count INTEGER
                    CHECK (harmony_event_count IS NULL OR harmony_event_count >= 0),
                harmony_segment_count INTEGER
                    CHECK (
                        harmony_segment_count IS NULL
                        OR harmony_segment_count >= 0
                    ),
                harmony_resolved_segment_count INTEGER
                    CHECK (
                        harmony_resolved_segment_count IS NULL
                        OR harmony_resolved_segment_count >= 0
                    ),
                harmony_unresolved_segment_count INTEGER
                    CHECK (
                        harmony_unresolved_segment_count IS NULL
                        OR harmony_unresolved_segment_count >= 0
                    ),
                harmony_unresolved_event_count INTEGER
                    CHECK (
                        harmony_unresolved_event_count IS NULL
                        OR harmony_unresolved_event_count >= 0
                    ),
                harmony_warning_count INTEGER
                    CHECK (
                        harmony_warning_count IS NULL
                        OR harmony_warning_count >= 0
                    ),
                harmony_used_interpretation_context INTEGER
                    CHECK (
                        harmony_used_interpretation_context IS NULL
                        OR harmony_used_interpretation_context IN (0, 1)
                    ),
                harmony_error TEXT
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
                END,
                interpretation_status = COALESCE(
                    NULLIF(TRIM(interpretation_status), ''),
                    'not_started'
                ),
                interpretation_stage = COALESCE(
                    NULLIF(TRIM(interpretation_stage), ''),
                    'not_started'
                ),
                interpretation_progress = CASE
                    WHEN interpretation_progress IS NULL
                         OR interpretation_progress < 0 THEN 0
                    ELSE interpretation_progress
                END,
                interpretation_part_count = CASE
                    WHEN interpretation_part_count < 0 THEN NULL
                    ELSE interpretation_part_count
                END,
                interpretation_phrase_count = CASE
                    WHEN interpretation_phrase_count < 0 THEN NULL
                    ELSE interpretation_phrase_count
                END,
                interpretation_pitched_item_count = CASE
                    WHEN interpretation_pitched_item_count < 0 THEN NULL
                    ELSE interpretation_pitched_item_count
                END,
                interpretation_percussion_item_count = CASE
                    WHEN interpretation_percussion_item_count < 0 THEN NULL
                    ELSE interpretation_percussion_item_count
                END,
                interpretation_warning_count = CASE
                    WHEN interpretation_warning_count < 0 THEN NULL
                    ELSE interpretation_warning_count
                END,
                harmony_status = COALESCE(
                    NULLIF(TRIM(harmony_status), ''),
                    'not_started'
                ),
                harmony_stage = COALESCE(
                    NULLIF(TRIM(harmony_stage), ''),
                    'not_started'
                ),
                harmony_progress = CASE
                    WHEN harmony_progress IS NULL OR harmony_progress < 0 THEN 0
                    WHEN harmony_progress > 100 THEN 100
                    ELSE harmony_progress
                END,
                harmony_attempt_id = CASE
                    WHEN harmony_status = 'processing' THEN harmony_attempt_id
                    ELSE NULL
                END,
                harmony_event_count = CASE
                    WHEN harmony_event_count < 0 THEN NULL
                    ELSE harmony_event_count
                END,
                harmony_segment_count = CASE
                    WHEN harmony_segment_count < 0 THEN NULL
                    ELSE harmony_segment_count
                END,
                harmony_resolved_segment_count = CASE
                    WHEN harmony_resolved_segment_count < 0 THEN NULL
                    ELSE harmony_resolved_segment_count
                END,
                harmony_unresolved_segment_count = CASE
                    WHEN harmony_unresolved_segment_count < 0 THEN NULL
                    ELSE harmony_unresolved_segment_count
                END,
                harmony_unresolved_event_count = CASE
                    WHEN harmony_unresolved_event_count < 0 THEN NULL
                    ELSE harmony_unresolved_event_count
                END,
                harmony_warning_count = CASE
                    WHEN harmony_warning_count < 0 THEN NULL
                    ELSE harmony_warning_count
                END,
                harmony_used_interpretation_context = CASE
                    WHEN harmony_used_interpretation_context IN (0, 1)
                        THEN harmony_used_interpretation_context
                    ELSE NULL
                END
            """
        )
        connection.execute(
            """
            UPDATE jobs
            SET harmony_stage = CASE
                    WHEN harmony_status = 'completed' THEN 'completed'
                    WHEN harmony_status = 'failed' THEN 'failed'
                    ELSE harmony_stage
                END,
                harmony_progress = CASE
                    WHEN harmony_status = 'completed' THEN 100
                    ELSE harmony_progress
                END,
                harmony_message = CASE
                    WHEN harmony_status = 'completed'
                         AND (harmony_message IS NULL OR TRIM(harmony_message) = '')
                        THEN 'Harmonic context complete.'
                    WHEN harmony_status = 'failed'
                         AND (harmony_message IS NULL OR TRIM(harmony_message) = '')
                        THEN 'Harmonic context can be retried.'
                    ELSE harmony_message
                END,
                harmony_error = CASE
                    WHEN harmony_status = 'completed' THEN NULL
                    WHEN harmony_status = 'failed'
                         AND (harmony_error IS NULL OR TRIM(harmony_error) = '')
                        THEN 'Harmonic-context processing failed.'
                    ELSE harmony_error
                END
            """
        )
        connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'failed',
                harmony_stage = 'failed',
                harmony_progress = CASE
                    WHEN harmony_progress >= 100 THEN 99
                    ELSE harmony_progress
                END,
                harmony_message = 'Saved harmonic context metadata is incomplete; harmony can be retried.',
                harmony_attempt_id = NULL,
                harmony_error = 'Saved harmonic context metadata is incomplete.'
            WHERE harmony_status = 'completed'
              AND (
                    harmony_attempt_version IS NULL
                    OR TRIM(harmony_attempt_version) = ''
                    OR harmony_version IS NULL
                    OR TRIM(harmony_version) = ''
                    OR harmony_attempt_version != harmony_version
                    OR is_valid_harmony_artifact(harmony_artifact_file_name) != 1
                    OR harmonized_at IS NULL
                    OR TRIM(harmonized_at) = ''
                    OR harmony_source_transcription_version IS NULL
                    OR TRIM(harmony_source_transcription_version) = ''
                    OR harmony_source_transcription_artifact_file_name
                        != 'transcription/raw-events.json'
                    OR harmony_source_transcribed_at IS NULL
                    OR TRIM(harmony_source_transcribed_at) = ''
                    OR harmony_event_count IS NULL
                    OR harmony_segment_count IS NULL
                    OR harmony_resolved_segment_count IS NULL
                    OR harmony_unresolved_segment_count IS NULL
                    OR harmony_unresolved_event_count IS NULL
                    OR harmony_warning_count IS NULL
                    OR harmony_used_interpretation_context IS NULL
                    OR harmony_segment_count
                        != harmony_resolved_segment_count
                           + harmony_unresolved_segment_count
                    OR harmony_unresolved_event_count > harmony_event_count
              )
            """
        )
        _normalize_score_columns(connection)


def _normalize_score_columns(connection: sqlite3.Connection) -> None:
    """Normalize legacy or hand-edited score lifecycle rows in place."""
    connection.execute(
        """
        UPDATE jobs
        SET score_status = CASE
                WHEN TRIM(COALESCE(score_status, '')) IN (
                    'not_started', 'processing', 'completed', 'failed'
                ) THEN TRIM(score_status)
                ELSE 'not_started'
            END,
            score_stage = COALESCE(NULLIF(TRIM(score_stage), ''), 'not_started'),
            score_progress = CASE
                WHEN score_progress IS NULL OR score_progress < 0 THEN 0
                WHEN score_progress > 100 THEN 100
                ELSE score_progress
            END,
            score_artifact_file_name = CASE
                WHEN is_valid_score_artifact(score_artifact_file_name) = 1
                    THEN score_artifact_file_name
                ELSE NULL
            END,
            score_measure_count = CASE
                WHEN score_measure_count < 0 THEN NULL ELSE score_measure_count
            END,
            score_note_count = CASE
                WHEN score_note_count < 0 THEN NULL ELSE score_note_count
            END,
            score_chord_symbol_count = CASE
                WHEN score_chord_symbol_count < 0 THEN NULL
                ELSE score_chord_symbol_count
            END,
            score_warning_count = CASE
                WHEN score_warning_count < 0 THEN NULL ELSE score_warning_count
            END
        """
    )
    connection.execute(
        """
        UPDATE jobs
        SET score_attempt_id = NULL,
            score_attempt_fingerprint = NULL
        WHERE score_status != 'processing'
          AND (score_attempt_id IS NOT NULL OR score_attempt_fingerprint IS NOT NULL)
        """
    )
    connection.execute(
        """
        UPDATE jobs
        SET score_status = CASE
                WHEN score_status = 'completed' THEN 'failed' ELSE score_status
            END,
            score_stage = CASE
                WHEN score_status = 'completed' THEN 'failed' ELSE score_stage
            END,
            score_progress = CASE
                WHEN score_progress >= 100 THEN 99 ELSE score_progress
            END,
            score_message = 'Saved score metadata is incomplete; the score can be rebuilt.',
            score_error = 'Saved score metadata is incomplete.',
            score_artifact_file_name = NULL,
            score_version = NULL,
            scored_at = NULL,
            score_source_fingerprint = NULL,
            score_measure_count = NULL,
            score_note_count = NULL,
            score_chord_symbol_count = NULL,
            score_warning_count = NULL
        WHERE (score_status = 'completed' OR score_artifact_file_name IS NOT NULL)
          AND score_status != 'processing'
          AND (
                score_artifact_file_name IS NULL
                OR score_version IS NULL
                OR TRIM(score_version) = ''
                OR scored_at IS NULL
                OR TRIM(scored_at) = ''
                OR is_valid_score_fingerprint(score_source_fingerprint) != 1
                OR score_measure_count IS NULL
                OR score_note_count IS NULL
                OR score_chord_symbol_count IS NULL
                OR score_warning_count IS NULL
          )
        """
    )
    connection.execute(
        """
        UPDATE jobs
        SET score_stage = CASE
                WHEN score_status = 'completed' THEN 'completed'
                WHEN score_status = 'failed' THEN 'failed'
                WHEN score_status = 'not_started' THEN 'not_started'
                ELSE score_stage
            END,
            score_progress = CASE
                WHEN score_status = 'completed' THEN 100
                WHEN score_status = 'not_started' THEN 0
                ELSE score_progress
            END,
            score_error = CASE
                WHEN score_status = 'failed'
                     AND (score_error IS NULL OR TRIM(score_error) = '')
                    THEN 'Score construction failed.'
                WHEN score_status IN ('completed', 'not_started') THEN NULL
                ELSE score_error
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
              AND interpretation_status != 'processing'
              AND harmony_status != 'processing'
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
        connection.execute(
            """
            UPDATE jobs
            SET status = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed' THEN 'completed'
                    ELSE status
                END,
                stage = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed' THEN 'completed'
                    ELSE stage
                END,
                message = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed'
                        THEN 'Raw transcription and any previous editable draft remain available; interpretation can be retried.'
                    ELSE message
                END,
                interpretation_status = 'failed',
                interpretation_stage = 'failed',
                interpretation_message = 'Raw transcription and any previous editable draft remain available; interpretation can be retried.',
                interpretation_error = 'Editable interpretation was interrupted by a server restart.',
                updated_at = ?
            WHERE interpretation_status = 'processing'
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE jobs
            SET status = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed' THEN 'completed'
                    ELSE status
                END,
                stage = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed' THEN 'completed'
                    ELSE stage
                END,
                message = CASE
                    WHEN status IN ('queued', 'processing')
                         AND preparation_status = 'completed'
                         AND analysis_status = 'completed'
                         AND transcription_status = 'completed'
                        THEN 'Raw transcription and any previous harmonic context remain available; harmony can be retried.'
                    ELSE message
                END,
                harmony_status = 'failed',
                harmony_stage = 'failed',
                harmony_message = 'Raw transcription and any previous harmonic context remain available; harmony can be retried.',
                harmony_attempt_id = NULL,
                harmony_error = 'Harmonic context was interrupted by a server restart.',
                updated_at = ?
            WHERE harmony_status = 'processing'
            """,
            (now,),
        )
        connection.execute(
            """
            UPDATE jobs
            SET score_status = 'failed',
                score_stage = 'failed',
                score_message = 'Earlier results and any previously built score remain available; score construction can be retried.',
                score_attempt_id = NULL,
                score_attempt_fingerprint = NULL,
                score_error = 'Score construction was interrupted by a server restart.',
                updated_at = ?
            WHERE score_status = 'processing'
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


def claim_interpretation_attempt(
    database_path: Path,
    job_id: str,
    *,
    interpretation_version: str,
    force: bool = False,
    message: str = "Preparing editable interpretation.",
) -> bool:
    """Atomically claim one editable-interpretation attempt without clearing prior output."""
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET interpretation_status = 'processing',
                interpretation_stage = 'preparing_interpretation',
                interpretation_progress = 1,
                interpretation_message = ?,
                interpretation_version = ?,
                interpretation_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND preparation_status = 'completed'
              AND analysis_status = 'completed'
              AND transcription_status = 'completed'
              AND (
                    interpretation_status IN ('not_started', 'failed')
                    OR (? = 1 AND interpretation_status = 'completed')
              )
            """,
            (
                message,
                interpretation_version,
                now,
                job_id,
                int(bool(force)),
            ),
        )
        return cursor.rowcount == 1


def claim_harmony_attempt(
    database_path: Path,
    job_id: str,
    *,
    harmony_version: str,
    force: bool = False,
    message: str = "Loading canonical raw pitch evidence.",
) -> str | None:
    """Atomically claim one harmony attempt and return its exact identity."""
    attempt_version = _validate_harmony_version(harmony_version)
    attempt_id = uuid4().hex
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'processing',
                harmony_stage = 'loading_raw_transcription',
                harmony_progress = 1,
                harmony_message = ?,
                harmony_attempt_id = ?,
                harmony_attempt_version = ?,
                harmony_source_transcription_version = transcription_version,
                harmony_source_transcription_artifact_file_name = transcription_artifact_file_name,
                harmony_source_transcribed_at = transcribed_at,
                harmony_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND preparation_status = 'completed'
              AND analysis_status = 'completed'
              AND transcription_status = 'completed'
              AND transcription_version IS NOT NULL
              AND TRIM(transcription_version) != ''
              AND transcription_artifact_file_name = ?
              AND transcribed_at IS NOT NULL
              AND TRIM(transcribed_at) != ''
              AND (
                    harmony_status IN ('not_started', 'failed')
                    OR (? = 1 AND harmony_status = 'completed')
              )
            """,
            (
                safe_message,
                attempt_id,
                attempt_version,
                now,
                job_id,
                _RAW_TRANSCRIPTION_ARTIFACT_FILE_NAME,
                int(bool(force)),
            ),
        )
        return attempt_id if cursor.rowcount == 1 else None


def start_harmony_attempt(
    database_path: Path,
    job_id: str,
    *,
    attempt_id: str,
    message: str = "Starting harmonic-context processing.",
) -> bool:
    """Atomically consume one claimed harmony attempt for worker execution."""
    safe_attempt_id = _validate_harmony_attempt_id(attempt_id)
    if safe_attempt_id is None:
        raise ValueError("harmony_attempt_id is required.")
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_stage = 'loading_raw_transcription',
                harmony_progress = 2,
                harmony_message = ?,
                updated_at = ?
            WHERE id = ?
              AND harmony_status = 'processing'
              AND harmony_stage = 'loading_raw_transcription'
              AND harmony_progress = 1
              AND harmony_attempt_id = ?
              AND transcription_status = 'completed'
              AND transcription_version IS harmony_source_transcription_version
              AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
              AND transcribed_at IS harmony_source_transcribed_at
            """,
            (
                safe_message,
                now,
                job_id,
                safe_attempt_id,
            ),
        )
        return cursor.rowcount == 1


@contextmanager
def harmony_cleanup_lease(
    database_path: Path,
    job_id: str,
    *,
    status: str,
    durable_artifact_file_name: str | None,
    active_attempt_id: str | None,
) -> Iterator[None]:
    """Hold the job's SQLite write lease while one orphan is removed safely."""
    safe_active_attempt_id = _validate_harmony_attempt_id(active_attempt_id)
    if active_attempt_id is not None and safe_active_attempt_id is None:
        raise ValueError("harmony_attempt_id is invalid.")
    if status not in {"not_started", "processing", "completed", "failed"}:
        raise ValueError("harmony status is invalid.")
    if (
        durable_artifact_file_name is not None
        and not _is_valid_harmony_artifact_file_name(durable_artifact_file_name)
    ):
        raise ValueError("harmony artifact file name is invalid.")
    expected = (status, durable_artifact_file_name, safe_active_attempt_id)
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            """
            SELECT harmony_status, harmony_artifact_file_name, harmony_attempt_id
            FROM jobs
            WHERE id = ?
            """,
            (job_id,),
        ).fetchone()
        current = (
            None
            if row is None
            else (
                row["harmony_status"],
                row["harmony_artifact_file_name"],
                row["harmony_attempt_id"],
            )
        )
        if current != expected:
            raise ValueError("harmony cleanup state changed.")
        yield


def update_harmony_progress(
    database_path: Path,
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str,
    attempt_id: str | None = None,
) -> bool:
    """Advance only the active source-matched harmony attempt monotonically."""
    safe_stage = _validate_harmony_text(stage, "harmony stage", 128)
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    safe_progress = _validate_harmony_progress(progress)
    safe_attempt_id = _validate_harmony_attempt_id(attempt_id)
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_stage = ?,
                harmony_progress = ?,
                harmony_message = ?,
                updated_at = ?
            WHERE id = ?
              AND harmony_status = 'processing'
              AND (
                    (? IS NULL AND harmony_attempt_id IS NULL)
                    OR harmony_attempt_id = ?
              )
              AND transcription_status = 'completed'
              AND transcription_version IS harmony_source_transcription_version
              AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
              AND transcribed_at IS harmony_source_transcribed_at
              AND ? >= harmony_progress
            """,
            (
                safe_stage,
                safe_progress,
                safe_message,
                now,
                job_id,
                safe_attempt_id,
                safe_attempt_id,
                safe_progress,
            ),
        )
        return cursor.rowcount == 1


def complete_harmony_attempt(
    database_path: Path,
    job_id: str,
    *,
    harmony_version: str,
    artifact_file_name: str,
    harmonized_at: str,
    event_count: int,
    segment_count: int,
    resolved_segment_count: int,
    unresolved_segment_count: int,
    unresolved_event_count: int,
    warning_count: int,
    used_interpretation_context: bool,
    message: str = "Harmonic context complete.",
    attempt_id: str | None = None,
) -> bool:
    """Atomically replace successful harmony metadata for the active source."""
    successful_version = _validate_harmony_version(harmony_version)
    safe_attempt_id = _validate_harmony_attempt_id(attempt_id)
    safe_artifact_file_name = _validate_completion_artifact_file_name(
        artifact_file_name,
        safe_attempt_id,
    )
    if safe_artifact_file_name is None:
        return False
    safe_timestamp = _validate_harmony_timestamp(harmonized_at)
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    counts = {
        "harmony_event_count": event_count,
        "harmony_segment_count": segment_count,
        "harmony_resolved_segment_count": resolved_segment_count,
        "harmony_unresolved_segment_count": unresolved_segment_count,
        "harmony_unresolved_event_count": unresolved_event_count,
        "harmony_warning_count": warning_count,
    }
    _validate_nonnegative_counts(counts)
    if segment_count != resolved_segment_count + unresolved_segment_count:
        raise ValueError(
            "segment_count must equal resolved and unresolved segment counts."
        )
    if unresolved_event_count > event_count:
        raise ValueError("unresolved_event_count cannot exceed event_count.")
    if type(used_interpretation_context) is not bool:
        raise ValueError("used_interpretation_context must be true or false.")

    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'completed',
                harmony_stage = 'completed',
                harmony_progress = 100,
                harmony_message = ?,
                harmony_attempt_id = NULL,
                harmony_attempt_version = ?,
                harmony_version = ?,
                harmony_artifact_file_name = ?,
                harmonized_at = ?,
                harmony_event_count = ?,
                harmony_segment_count = ?,
                harmony_resolved_segment_count = ?,
                harmony_unresolved_segment_count = ?,
                harmony_unresolved_event_count = ?,
                harmony_warning_count = ?,
                harmony_used_interpretation_context = ?,
                harmony_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND harmony_status = 'processing'
              AND (
                    (? IS NULL AND harmony_attempt_id IS NULL)
                    OR harmony_attempt_id = ?
              )
              AND harmony_attempt_version = ?
              AND transcription_status = 'completed'
              AND transcription_version IS harmony_source_transcription_version
              AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
              AND transcribed_at IS harmony_source_transcribed_at
            """,
            (
                safe_message,
                successful_version,
                successful_version,
                safe_artifact_file_name,
                safe_timestamp,
                event_count,
                segment_count,
                resolved_segment_count,
                unresolved_segment_count,
                unresolved_event_count,
                warning_count,
                int(used_interpretation_context),
                now,
                job_id,
                safe_attempt_id,
                safe_attempt_id,
                successful_version,
            ),
        )
        return cursor.rowcount == 1


def fail_harmony_attempt(
    database_path: Path,
    job_id: str,
    *,
    error: str,
    message: str = "Harmonic context can be retried.",
    attempt_id: str | None = None,
) -> bool:
    """Fail only the active source-matched attempt and preserve prior success."""
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    safe_error = _sanitize_harmony_error(error)
    safe_attempt_id = _validate_harmony_attempt_id(attempt_id)
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'failed',
                harmony_stage = 'failed',
                harmony_message = ?,
                harmony_attempt_id = NULL,
                harmony_error = ?,
                updated_at = ?
            WHERE id = ?
              AND harmony_status = 'processing'
              AND (
                    (? IS NULL AND harmony_attempt_id IS NULL)
                    OR harmony_attempt_id = ?
              )
              AND transcription_status = 'completed'
              AND transcription_version IS harmony_source_transcription_version
              AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
              AND transcribed_at IS harmony_source_transcribed_at
            """,
            (
                safe_message,
                safe_error,
                now,
                job_id,
                safe_attempt_id,
                safe_attempt_id,
            ),
        )
        return cursor.rowcount == 1


def reset_harmony_after_transcription_change(
    database_path: Path,
    job_id: str,
) -> bool:
    """Clear stale harmony metadata only after a successful raw replacement."""
    with connect(database_path) as connection:
        cursor = _reset_stale_harmony_in_connection(connection, job_id)
        return cursor.rowcount == 1


def _reset_stale_harmony_in_connection(
    connection: sqlite3.Connection,
    job_id: str,
    *,
    force: bool = False,
) -> sqlite3.Cursor:
    source_guard = "" if force else """
          AND NOT (
                transcription_version IS harmony_source_transcription_version
                AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
                AND transcribed_at IS harmony_source_transcribed_at
          )
    """
    return connection.execute(
        f"""
        UPDATE jobs
        SET harmony_status = 'not_started',
            harmony_stage = 'not_started',
            harmony_progress = 0,
            harmony_message = NULL,
            harmony_attempt_id = NULL,
            harmony_attempt_version = NULL,
            harmony_version = NULL,
            harmony_artifact_file_name = NULL,
            harmonized_at = NULL,
            harmony_source_transcription_version = NULL,
            harmony_source_transcription_artifact_file_name = NULL,
            harmony_source_transcribed_at = NULL,
            harmony_event_count = NULL,
            harmony_segment_count = NULL,
            harmony_resolved_segment_count = NULL,
            harmony_unresolved_segment_count = NULL,
            harmony_unresolved_event_count = NULL,
            harmony_warning_count = NULL,
            harmony_used_interpretation_context = NULL,
            harmony_error = NULL,
            updated_at = ?
        WHERE id = ?
          AND transcription_status = 'completed'
          {source_guard}
          AND (
                harmony_status != 'not_started'
                OR harmony_stage != 'not_started'
                OR harmony_progress != 0
                OR harmony_message IS NOT NULL
                OR harmony_attempt_id IS NOT NULL
                OR harmony_attempt_version IS NOT NULL
                OR harmony_version IS NOT NULL
                OR harmony_artifact_file_name IS NOT NULL
                OR harmonized_at IS NOT NULL
                OR harmony_source_transcription_version IS NOT NULL
                OR harmony_source_transcription_artifact_file_name IS NOT NULL
                OR harmony_source_transcribed_at IS NOT NULL
                OR harmony_event_count IS NOT NULL
                OR harmony_segment_count IS NOT NULL
                OR harmony_resolved_segment_count IS NOT NULL
                OR harmony_unresolved_segment_count IS NOT NULL
                OR harmony_unresolved_event_count IS NOT NULL
                OR harmony_warning_count IS NOT NULL
                OR harmony_used_interpretation_context IS NOT NULL
                OR harmony_error IS NOT NULL
          )
        """,
        (utc_now(), job_id),
    )


@dataclass(frozen=True, slots=True)
class ScoreCompletion:
    """Outcome of an attempt-scoped score completion."""

    completed: bool
    previous_artifact_file_name: str | None = None


@contextmanager
def _score_write_transaction(
    database_path: Path,
    job_id: str,
) -> Iterator[tuple[sqlite3.Connection, dict[str, Any] | None]]:
    """Hold SQLite's write lock while one score row is read and changed."""
    with connect(database_path) as connection:
        connection.execute("BEGIN IMMEDIATE")
        row = connection.execute(
            "SELECT * FROM jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
        yield connection, (dict(row) if row is not None else None)


def set_score_tablature_request(
    database_path: Path,
    job_id: str,
    encoded_request: str | None,
) -> str:
    """Save one job's tablature choice unless a score build is running.

    Returns ``saved``, ``missing`` or ``busy``. The value is validated by the
    caller with :func:`app.score_sources.encode_tablature_request`; ``None``
    restores the default.
    """
    if encoded_request is not None and (
        not isinstance(encoded_request, str) or len(encoded_request) > 256
    ):
        raise ValueError("score_tablature_request is invalid.")
    with _score_write_transaction(database_path, job_id) as (connection, row):
        if row is None:
            return "missing"
        if row.get("score_status") == "processing":
            return "busy"
        connection.execute(
            "UPDATE jobs SET score_tablature_request = ?, updated_at = ? WHERE id = ?",
            (encoded_request, utc_now(), job_id),
        )
        return "saved"


def claim_score_attempt(
    database_path: Path,
    job_id: str,
    *,
    score_version: str,
    force: bool = False,
    message: str = "Queued score construction.",
) -> str | None:
    """Atomically claim one score attempt and return its exact identity.

    The claim captures the fingerprint of the evidence visible inside the same
    write transaction, so the caller never re-reads mutable state to discover
    which attempt it owns.
    """
    # The version is validated for the caller's contract; the durable version
    # is written only by a successful completion.
    _validate_score_version(score_version)
    safe_message = _validate_harmony_text(message, "score message", 500)
    attempt_id = uuid4().hex
    with _score_write_transaction(database_path, job_id) as (connection, row):
        if row is None:
            return None
        fingerprint = current_score_fingerprint(row)
        status = row.get("score_status")
        if fingerprint is None or not (
            status in {"not_started", "failed"}
            or (bool(force) and status == "completed")
        ):
            return None
        cursor = connection.execute(
            """
            UPDATE jobs
            SET score_status = 'processing',
                score_stage = 'queued',
                score_progress = 1,
                score_message = ?,
                score_attempt_id = ?,
                score_attempt_fingerprint = ?,
                score_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND score_status = ?
            """,
            (
                safe_message,
                attempt_id,
                fingerprint,
                utc_now(),
                job_id,
                status,
            ),
        )
        return attempt_id if cursor.rowcount == 1 else None


def start_score_attempt(
    database_path: Path,
    job_id: str,
    *,
    attempt_id: str,
    message: str = "Loading saved analysis and transcription evidence.",
) -> bool:
    """Consume one queued claim exactly once for worker execution."""
    safe_attempt_id = _validate_score_attempt_id(attempt_id)
    if safe_attempt_id is None:
        raise ValueError("score_attempt_id is required.")
    safe_message = _validate_harmony_text(message, "score message", 500)
    with _score_write_transaction(database_path, job_id) as (connection, row):
        if (
            row is None
            or row.get("score_status") != "processing"
            or row.get("score_stage") != "queued"
            or row.get("score_attempt_id") != safe_attempt_id
        ):
            return False
        if current_score_fingerprint(row) != row.get("score_attempt_fingerprint"):
            # The evidence changed between the claim and the worker start:
            # close this attempt now so it can be retried, never left queued.
            connection.execute(
                """
                UPDATE jobs
                SET score_status = 'failed',
                    score_stage = 'failed',
                    score_message = ?,
                    score_attempt_id = NULL,
                    score_attempt_fingerprint = NULL,
                    score_error = ?,
                    updated_at = ?
                WHERE id = ?
                  AND score_status = 'processing'
                  AND score_attempt_id = ?
                """,
                (
                    "Score construction stopped; earlier results remain available.",
                    "Score evidence changed before construction started; "
                    "build the score again.",
                    utc_now(),
                    job_id,
                    safe_attempt_id,
                ),
            )
            return False
        cursor = connection.execute(
            """
            UPDATE jobs
            SET score_stage = 'loading_evidence',
                score_progress = 2,
                score_message = ?,
                updated_at = ?
            WHERE id = ?
              AND score_status = 'processing'
              AND score_stage = 'queued'
              AND score_attempt_id = ?
            """,
            (safe_message, utc_now(), job_id, safe_attempt_id),
        )
        return cursor.rowcount == 1


def update_score_progress(
    database_path: Path,
    job_id: str,
    *,
    attempt_id: str,
    stage: str,
    progress: float,
    message: str,
) -> bool:
    """Advance only the active attempt, monotonically."""
    safe_attempt_id = _validate_score_attempt_id(attempt_id)
    if safe_attempt_id is None:
        raise ValueError("score_attempt_id is required.")
    safe_stage = _validate_harmony_text(stage, "score stage", 128)
    safe_message = _validate_harmony_text(message, "score message", 500)
    safe_progress = _validate_score_progress(progress)
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET score_stage = ?,
                score_progress = ?,
                score_message = ?,
                updated_at = ?
            WHERE id = ?
              AND score_status = 'processing'
              AND score_attempt_id = ?
              AND ? >= score_progress
            """,
            (
                safe_stage,
                safe_progress,
                safe_message,
                utc_now(),
                job_id,
                safe_attempt_id,
                safe_progress,
            ),
        )
        return cursor.rowcount == 1


def complete_score_attempt(
    database_path: Path,
    job_id: str,
    *,
    attempt_id: str,
    score_version: str,
    artifact_file_name: str,
    scored_at: str,
    source_fingerprint: str,
    measure_count: int,
    note_count: int,
    chord_symbol_count: int,
    warning_count: int,
    message: str = "Draft score saved.",
) -> ScoreCompletion:
    """Publish one attempt's score only if its evidence is still current."""
    safe_attempt_id = _validate_score_attempt_id(attempt_id)
    if safe_attempt_id is None:
        raise ValueError("score_attempt_id is required.")
    safe_version = _validate_score_version(score_version)
    if artifact_file_name != score_attempt_artifact_file_name(safe_attempt_id):
        return ScoreCompletion(False)
    safe_timestamp = _validate_utc_timestamp(scored_at, "scored_at")
    if not is_score_fingerprint(source_fingerprint):
        raise ValueError("score source fingerprint is invalid.")
    counts = {
        "score_measure_count": measure_count,
        "score_note_count": note_count,
        "score_chord_symbol_count": chord_symbol_count,
        "score_warning_count": warning_count,
    }
    _validate_nonnegative_counts(counts)
    if any(value is None for value in counts.values()):
        raise ValueError("score counts are required.")
    if chord_symbol_count > measure_count:
        raise ValueError("chord_symbol_count cannot exceed measure_count.")
    safe_message = _validate_harmony_text(message, "score message", 500)
    with _score_write_transaction(database_path, job_id) as (connection, row):
        if (
            row is None
            or row.get("score_status") != "processing"
            or row.get("score_attempt_id") != safe_attempt_id
            or row.get("score_attempt_fingerprint") != source_fingerprint
            or current_score_fingerprint(row) != source_fingerprint
        ):
            return ScoreCompletion(False)
        previous = row.get("score_artifact_file_name")
        cursor = connection.execute(
            """
            UPDATE jobs
            SET score_status = 'completed',
                score_stage = 'completed',
                score_progress = 100,
                score_message = ?,
                score_attempt_id = NULL,
                score_attempt_fingerprint = NULL,
                score_version = ?,
                score_artifact_file_name = ?,
                scored_at = ?,
                score_source_fingerprint = ?,
                score_measure_count = ?,
                score_note_count = ?,
                score_chord_symbol_count = ?,
                score_warning_count = ?,
                score_error = NULL,
                updated_at = ?
            WHERE id = ?
              AND score_status = 'processing'
              AND score_attempt_id = ?
            """,
            (
                safe_message,
                safe_version,
                artifact_file_name,
                safe_timestamp,
                source_fingerprint,
                measure_count,
                note_count,
                chord_symbol_count,
                warning_count,
                utc_now(),
                job_id,
                safe_attempt_id,
            ),
        )
        if cursor.rowcount != 1:
            return ScoreCompletion(False)
        return ScoreCompletion(
            True,
            previous
            if _is_valid_score_artifact_file_name(previous)
            and previous != artifact_file_name
            else None,
        )


def fail_score_attempt(
    database_path: Path,
    job_id: str,
    *,
    attempt_id: str,
    error: str,
    message: str = "Score construction stopped; earlier results remain available.",
) -> bool:
    """Fail only the active attempt; the previous durable score is untouched."""
    safe_attempt_id = _validate_score_attempt_id(attempt_id)
    if safe_attempt_id is None:
        raise ValueError("score_attempt_id is required.")
    safe_message = _validate_harmony_text(message, "score message", 500)
    safe_error = _sanitize_score_error(error)
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET score_status = 'failed',
                score_stage = 'failed',
                score_message = ?,
                score_attempt_id = NULL,
                score_attempt_fingerprint = NULL,
                score_error = ?,
                updated_at = ?
            WHERE id = ?
              AND score_status = 'processing'
              AND score_attempt_id = ?
            """,
            (safe_message, safe_error, utc_now(), job_id, safe_attempt_id),
        )
        return cursor.rowcount == 1


@contextmanager
def score_cleanup_lease(
    database_path: Path,
    job_id: str,
) -> Iterator[tuple[str | None, str | None]]:
    """Yield the durable pointer and active attempt while holding the write lock.

    Artifact cleanup runs inside this lease so a concurrent completion cannot
    make a file durable between the ownership check and its removal.
    """
    with _score_write_transaction(database_path, job_id) as (_connection, row):
        if row is None:
            raise ValueError("score job is unavailable.")
        durable = row.get("score_artifact_file_name")
        active = (
            row.get("score_attempt_id")
            if row.get("score_status") == "processing"
            else None
        )
        yield (
            durable if _is_valid_score_artifact_file_name(durable) else None,
            active if isinstance(active, str) else None,
        )


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
    _validate_nonnegative_counts(values)
    _validate_harmony_values(values)
    _validate_score_values(values)
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
        if values.get("transcription_status") == "completed":
            _reset_stale_harmony_in_connection(
                connection,
                job_id,
                force=True,
            )


def _validate_nonnegative_counts(values: dict[str, Any]) -> None:
    for field in (
        _TRANSCRIPTION_COUNT_FIELDS
        | _INTERPRETATION_COUNT_FIELDS
        | _HARMONY_COUNT_FIELDS
        | _SCORE_COUNT_FIELDS
    ):
        if field not in values:
            continue
        value = values[field]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{field} must be a non-negative integer or None.")


def _validate_harmony_values(values: dict[str, Any]) -> None:
    if "harmony_progress" in values:
        values["harmony_progress"] = _validate_harmony_progress(
            values["harmony_progress"]
        )
    if "harmony_attempt_id" in values:
        values["harmony_attempt_id"] = _validate_harmony_attempt_id(
            values["harmony_attempt_id"]
        )
    if "harmony_artifact_file_name" in values:
        pointer = values["harmony_artifact_file_name"]
        if pointer is not None:
            values["harmony_artifact_file_name"] = _validate_harmony_artifact_pointer(
                pointer
            )
    for field in _HARMONY_BOOLEAN_FIELDS:
        if field not in values:
            continue
        value = values[field]
        if value is None:
            continue
        if type(value) is not bool:
            raise ValueError(f"{field} must be true, false, or None.")
        values[field] = int(value)


def _validate_harmony_progress(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("harmony progress must be a number from 0 through 100.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError("harmony progress must be a number from 0 through 100.")
    return number


def _validate_harmony_version(value: Any) -> str:
    if not isinstance(value, str) or not _HARMONY_VERSION_RE.fullmatch(value):
        raise ValueError("harmony_version is invalid.")
    return value


def _validate_harmony_attempt_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _HARMONY_ATTEMPT_ID_RE.fullmatch(value):
        raise ValueError("harmony_attempt_id is invalid.")
    return value


def _is_valid_harmony_artifact_file_name(value: Any) -> bool:
    return isinstance(value, str) and (
        value == _HARMONY_ARTIFACT_FILE_NAME
        or _HARMONY_ATTEMPT_ARTIFACT_RE.fullmatch(value) is not None
    )


def _validate_harmony_artifact_pointer(value: Any) -> str:
    if not _is_valid_harmony_artifact_file_name(value):
        raise ValueError("harmony_artifact_file_name is invalid.")
    return value


def _validate_completion_artifact_file_name(
    value: Any,
    attempt_id: str | None,
) -> str | None:
    pointer = _validate_harmony_artifact_pointer(value)
    expected = (
        _HARMONY_ARTIFACT_FILE_NAME
        if attempt_id is None
        else f"harmony/harmonic-context.{attempt_id}.json"
    )
    return pointer if pointer == expected else None


def _validate_harmony_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text.")
    text = " ".join(value.split())
    if not text or len(text) > maximum:
        raise ValueError(f"{label} is invalid.")
    return text


def _validate_harmony_timestamp(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError("harmonized_at must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("harmonized_at must be a UTC timestamp.") from exc
    if (
        parsed.tzinfo is None
        or parsed.utcoffset() is None
        or parsed.utcoffset().total_seconds() != 0
    ):
        raise ValueError("harmonized_at must be a UTC timestamp.")
    return value


def _sanitize_harmony_error(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("harmony error must be text.")
    text = " ".join(value.split())
    if not text or _HARMONY_UNSAFE_ERROR_RE.search(text):
        return "Harmonic-context processing failed."
    if len(text) > 500:
        text = text[:499].rstrip() + "…"
    return text


def _validate_score_version(value: Any) -> str:
    if not isinstance(value, str) or not _SCORE_VERSION_RE.fullmatch(value):
        raise ValueError("score_version is invalid.")
    return value


def _validate_score_attempt_id(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _HARMONY_ATTEMPT_ID_RE.fullmatch(value):
        raise ValueError("score_attempt_id is invalid.")
    return value


def _validate_score_progress(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("score progress must be a number from 0 through 100.")
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 100:
        raise ValueError("score progress must be a number from 0 through 100.")
    return number


def _validate_utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 128:
        raise ValueError(f"{label} must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} must be a UTC timestamp.") from exc
    if parsed.utcoffset() is None or parsed.utcoffset().total_seconds() != 0:
        raise ValueError(f"{label} must be a UTC timestamp.")
    return value


def _sanitize_score_error(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("score error must be text.")
    text = " ".join(value.split())
    if not text or _HARMONY_UNSAFE_ERROR_RE.search(text):
        return "Score construction failed."
    if len(text) > 500:
        text = text[:499].rstrip() + "…"
    return text


def _validate_score_values(values: dict[str, Any]) -> None:
    if "score_status" in values and values["score_status"] not in _SCORE_STATUSES:
        raise ValueError("score_status is invalid.")
    if "score_progress" in values:
        values["score_progress"] = _validate_score_progress(values["score_progress"])
    if "score_attempt_id" in values:
        values["score_attempt_id"] = _validate_score_attempt_id(
            values["score_attempt_id"]
        )
    pointer = values.get("score_artifact_file_name")
    if pointer is not None and not _is_valid_score_artifact_file_name(pointer):
        raise ValueError("score_artifact_file_name is invalid.")
    for field in ("score_attempt_fingerprint", "score_source_fingerprint"):
        value = values.get(field)
        if value is not None and not is_score_fingerprint(value):
            raise ValueError(f"{field} is invalid.")


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
