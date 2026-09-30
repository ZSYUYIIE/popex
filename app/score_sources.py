"""Exact upstream-evidence identity for persisted score construction.

A persisted score is derived from several independently versioned layers
(audio analysis, raw transcription, and optionally the editable interpretation
and harmonic context). This module turns a job record into one canonical
identity mapping and a SHA-256 fingerprint of that mapping. The fingerprint is
captured when a score attempt is claimed, re-checked atomically when the
attempt completes, stored with the published score, and compared on every read
so a score built from older evidence is reported as out of date instead of
being silently combined with newer layers.

The module is intentionally dependency-free so the database layer can use it
inside a write transaction without importing the analysis stack.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from typing import Any

SCORE_SOURCE_IDENTITY_VERSION = 1

# Canonical artifact pointers. Tests pin these to the owning modules'
# constants; they are duplicated here to keep this module import-light.
ANALYSIS_FILE_NAME = "analysis/audio-analysis.json"
RAW_TRANSCRIPTION_FILE_NAME = "transcription/raw-events.json"
INTERPRETATION_FILE_NAME = "interpretation/draft.json"
_HARMONY_FILE_RE = re.compile(r"harmony/harmonic-context(?:\.[a-f0-9]{32})?\.json")
_FINGERPRINT_RE = re.compile(r"[a-f0-9]{64}")


def _text(record: Mapping[str, Any], key: str) -> str | None:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        return None
    return value


def _count(record: Mapping[str, Any], key: str) -> int | None:
    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def score_source_identity(record: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the evidence identity a score would use, or ``None`` if unready.

    Analysis and raw transcription are required. Interpretation and harmony
    are included only when their stage is completed with a canonical pointer;
    otherwise they are recorded as ``None`` (an explicitly omitted layer).
    """
    if not isinstance(record, Mapping):
        return None
    if (
        record.get("preparation_status") != "completed"
        or record.get("analysis_status") != "completed"
        or record.get("analysis_json_file_name") != ANALYSIS_FILE_NAME
        or record.get("transcription_status") != "completed"
        or record.get("transcription_artifact_file_name")
        != RAW_TRANSCRIPTION_FILE_NAME
    ):
        return None
    analysis_version = _text(record, "analysis_version")
    analyzed_at = _text(record, "analyzed_at")
    transcription_version = _text(record, "transcription_version")
    transcribed_at = _text(record, "transcribed_at")
    counts = {
        "pitchedEventCount": _count(record, "pitched_event_count"),
        "percussionEventCount": _count(record, "percussion_event_count"),
        "alignedEventCount": _count(record, "aligned_event_count"),
    }
    if (
        analysis_version is None
        or analyzed_at is None
        or transcription_version is None
        or transcribed_at is None
        or any(value is None for value in counts.values())
    ):
        return None

    interpretation = None
    if (
        record.get("interpretation_status") == "completed"
        and record.get("interpretation_artifact_file_name")
        == INTERPRETATION_FILE_NAME
        and _text(record, "interpretation_version") is not None
        and _text(record, "interpreted_at") is not None
    ):
        interpretation = {
            "fileName": INTERPRETATION_FILE_NAME,
            "version": record["interpretation_version"],
            "createdAt": record["interpreted_at"],
        }

    harmony = None
    harmony_file = record.get("harmony_artifact_file_name")
    if (
        record.get("harmony_status") == "completed"
        and isinstance(harmony_file, str)
        and _HARMONY_FILE_RE.fullmatch(harmony_file) is not None
        and _text(record, "harmony_version") is not None
        and _text(record, "harmonized_at") is not None
    ):
        harmony = {
            "fileName": harmony_file,
            "version": record["harmony_version"],
            "createdAt": record["harmonized_at"],
        }

    return {
        "identityVersion": SCORE_SOURCE_IDENTITY_VERSION,
        "analysis": {
            "fileName": ANALYSIS_FILE_NAME,
            "version": analysis_version,
            "createdAt": analyzed_at,
        },
        "transcription": {
            "fileName": RAW_TRANSCRIPTION_FILE_NAME,
            "version": transcription_version,
            "createdAt": transcribed_at,
            **counts,
        },
        "interpretation": interpretation,
        "harmony": harmony,
    }


def score_source_fingerprint(identity: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hex digest of one source identity mapping."""
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def current_score_fingerprint(record: Mapping[str, Any] | None) -> str | None:
    """Return the fingerprint of the record's current evidence, if ready."""
    identity = score_source_identity(record)
    return None if identity is None else score_source_fingerprint(identity)


def is_score_fingerprint(value: object) -> bool:
    return isinstance(value, str) and _FINGERPRINT_RE.fullmatch(value) is not None


__all__ = [
    "SCORE_SOURCE_IDENTITY_VERSION",
    "current_score_fingerprint",
    "is_score_fingerprint",
    "score_source_fingerprint",
    "score_source_identity",
]
