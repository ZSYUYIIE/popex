from pathlib import Path

path = Path("app/db.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    text = text.replace(old, new, 1)


if "def claim_harmony_attempt(" in text:
    required = (
        '"harmony_status": "TEXT NOT NULL DEFAULT \'not_started\'"',
        "def complete_harmony_attempt(",
        "def reset_harmony_after_transcription_change(",
        "harmony_source_transcription_artifact_file_name",
    )
    if all(marker in text for marker in required):
        print("harmony job state already patched")
        raise SystemExit(0)
    raise SystemExit("partial harmony job-state patch detected")

replace_once(
    "import sqlite3\n",
    "import math\nimport re\nimport sqlite3\n",
    "imports",
)

replace_once(
    '''        "interpretation_warning_count",
    }
)

NEW_COLUMNS''',
    '''        "interpretation_warning_count",
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
_HARMONY_UNSAFE_ERROR_RE = re.compile(
    r"traceback|https?://|(?:^|\\s)[A-Za-z]:[\\\\/]|\\\\\\\\|"
    r"(?:^|\\s)/(?:[^\\s/]+/)+|"
    r"\\b(?:token|password|secret|authorization|api[_-]?key|access[_-]?key)"
    r"\\b\\s*[:=]",
    re.IGNORECASE,
)

NEW_COLUMNS''',
    "harmony constants",
)

replace_once(
    '''    "interpretation_error": "TEXT",
}''',
    '''    "interpretation_error": "TEXT",
    "harmony_status": "TEXT NOT NULL DEFAULT 'not_started'",
    "harmony_stage": "TEXT NOT NULL DEFAULT 'not_started'",
    "harmony_progress": (
        "REAL NOT NULL DEFAULT 0 "
        "CHECK (harmony_progress >= 0 AND harmony_progress <= 100)"
    ),
    "harmony_message": "TEXT",
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
}''',
    "new harmony columns",
)

replace_once(
    '''                interpretation_error TEXT
            )''',
    '''                interpretation_error TEXT,
                harmony_status TEXT NOT NULL DEFAULT 'not_started',
                harmony_stage TEXT NOT NULL DEFAULT 'not_started',
                harmony_progress REAL NOT NULL DEFAULT 0
                    CHECK (harmony_progress >= 0 AND harmony_progress <= 100),
                harmony_message TEXT,
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
            )''',
    "create table harmony columns",
)

replace_once(
    '''                interpretation_warning_count = CASE
                    WHEN interpretation_warning_count < 0 THEN NULL
                    ELSE interpretation_warning_count
                END
            """
        )


def fail_incomplete_jobs''',
    '''                interpretation_warning_count = CASE
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
                harmony_error = 'Saved harmonic context metadata is incomplete.'
            WHERE harmony_status = 'completed'
              AND (
                    harmony_attempt_version IS NULL
                    OR TRIM(harmony_attempt_version) = ''
                    OR harmony_version IS NULL
                    OR TRIM(harmony_version) = ''
                    OR harmony_attempt_version != harmony_version
                    OR harmony_artifact_file_name != 'harmony/harmonic-context.json'
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


def fail_incomplete_jobs''',
    "harmony normalization",
)

replace_once(
    '''              AND interpretation_status != 'processing'
            """,''',
    '''              AND interpretation_status != 'processing'
              AND harmony_status != 'processing'
            """,''',
    "restart base exclusion",
)

replace_once(
    '''            WHERE interpretation_status = 'processing'
            """,
            (now,),
        )


def create_job''',
    '''            WHERE interpretation_status = 'processing'
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
                harmony_error = 'Harmonic context was interrupted by a server restart.',
                updated_at = ?
            WHERE harmony_status = 'processing'
            """,
            (now,),
        )


def create_job''',
    "restart harmony state",
)

harmony_functions = '''

def claim_harmony_attempt(
    database_path: Path,
    job_id: str,
    *,
    harmony_version: str,
    force: bool = False,
    message: str = "Loading canonical raw pitch evidence.",
) -> bool:
    """Atomically claim one harmony attempt while preserving prior success."""
    attempt_version = _validate_harmony_version(harmony_version)
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
                attempt_version,
                now,
                job_id,
                _RAW_TRANSCRIPTION_ARTIFACT_FILE_NAME,
                int(bool(force)),
            ),
        )
        return cursor.rowcount == 1


def update_harmony_progress(
    database_path: Path,
    job_id: str,
    *,
    stage: str,
    progress: float,
    message: str,
) -> bool:
    """Advance only the active source-matched harmony attempt monotonically."""
    safe_stage = _validate_harmony_text(stage, "harmony stage", 128)
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    safe_progress = _validate_harmony_progress(progress)
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
) -> bool:
    """Atomically replace successful harmony metadata for the active source."""
    successful_version = _validate_harmony_version(harmony_version)
    if artifact_file_name != _HARMONY_ARTIFACT_FILE_NAME:
        raise ValueError("artifact_file_name must be the canonical harmony path.")
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
                artifact_file_name,
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
) -> bool:
    """Fail only the active source-matched attempt and preserve prior success."""
    safe_message = _validate_harmony_text(message, "harmony message", 500)
    safe_error = _sanitize_harmony_error(error)
    now = utc_now()
    with connect(database_path) as connection:
        cursor = connection.execute(
            """
            UPDATE jobs
            SET harmony_status = 'failed',
                harmony_stage = 'failed',
                harmony_message = ?,
                harmony_error = ?,
                updated_at = ?
            WHERE id = ?
              AND harmony_status = 'processing'
              AND transcription_status = 'completed'
              AND transcription_version IS harmony_source_transcription_version
              AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
              AND transcribed_at IS harmony_source_transcribed_at
            """,
            (safe_message, safe_error, now, job_id),
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
) -> sqlite3.Cursor:
    return connection.execute(
        """
        UPDATE jobs
        SET harmony_status = 'not_started',
            harmony_stage = 'not_started',
            harmony_progress = 0,
            harmony_message = NULL,
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
          AND NOT (
                transcription_version IS harmony_source_transcription_version
                AND transcription_artifact_file_name
                    IS harmony_source_transcription_artifact_file_name
                AND transcribed_at IS harmony_source_transcribed_at
          )
          AND (
                harmony_status != 'not_started'
                OR harmony_stage != 'not_started'
                OR harmony_progress != 0
                OR harmony_message IS NOT NULL
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
'''

replace_once(
    "\n\ndef update_job(database_path: Path, job_id: str, **fields: Any) -> None:\n",
    harmony_functions
    + "\n\ndef update_job(database_path: Path, job_id: str, **fields: Any) -> None:\n",
    "harmony lifecycle functions",
)

replace_once(
    '''    _validate_nonnegative_counts(values)
    if not values:
''',
    '''    _validate_nonnegative_counts(values)
    _validate_harmony_values(values)
    if not values:
''',
    "update validation",
)

replace_once(
    '''        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            parameters,
        )


def _validate_nonnegative_counts''',
    '''        connection.execute(
            f"UPDATE jobs SET {assignments} WHERE id = ?",
            parameters,
        )
        if values.get("transcription_status") == "completed":
            _reset_stale_harmony_in_connection(connection, job_id)


def _validate_nonnegative_counts''',
    "automatic upstream invalidation",
)

replace_once(
    '''    for field in _TRANSCRIPTION_COUNT_FIELDS | _INTERPRETATION_COUNT_FIELDS:
''',
    '''    for field in (
        _TRANSCRIPTION_COUNT_FIELDS
        | _INTERPRETATION_COUNT_FIELDS
        | _HARMONY_COUNT_FIELDS
    ):
''',
    "harmony count validation",
)

helper_functions = '''

def _validate_harmony_values(values: dict[str, Any]) -> None:
    if "harmony_progress" in values:
        values["harmony_progress"] = _validate_harmony_progress(
            values["harmony_progress"]
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
'''

replace_once(
    '''            raise ValueError(f"{field} must be a non-negative integer or None.")


def get_job''',
    '''            raise ValueError(f"{field} must be a non-negative integer or None.")'''
    + helper_functions
    + "\n\ndef get_job",
    "harmony validation helpers",
)

path.write_text(text, encoding="utf-8")
