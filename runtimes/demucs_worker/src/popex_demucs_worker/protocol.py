from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any

EXIT_SUCCESS = 0
EXIT_RUNTIME_INCOMPATIBLE = 10
EXIT_MODEL_DOWNLOAD_REQUIRED = 20
EXIT_MODEL_VERIFICATION_FAILED = 21
EXIT_MODEL_DOWNLOAD_FAILED = 22
EXIT_INVALID_REQUEST = 30
EXIT_SEPARATION_FAILED = 40
EXIT_CANCELLED = 41
EXIT_TIMEOUT = 42
EXIT_INTERNAL = 50

_URL_RE = re.compile(r"(?i)https?://[^\s\"']+")
_TOKEN_RE = re.compile(
    r"(?i)(?:bearer\s+)?(?:hf_|ghp_|github_pat_|sk-)[A-Za-z0-9_\-]{6,}"
)
_AWS_RE = re.compile(r"\bAKIA[0-9A-Z]{12,}\b")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:[\\/][^\s\"']+")
_POSIX_PATH_RE = re.compile(r"(?<![A-Za-z0-9_])/(?:[^\s\"']+/)*[^\s\"']+")


@dataclass(slots=True)
class WorkerError(Exception):
    code: str
    message: str
    exit_code: int
    retryable: bool = False

    def __str__(self) -> str:
        return self.message


def sanitize_diagnostic(value: object) -> str:
    text = str(value).replace("\x00", "[redacted]")
    text = _URL_RE.sub("[redacted-url]", text)
    text = _TOKEN_RE.sub("[redacted-token]", text)
    text = _AWS_RE.sub("[redacted-token]", text)
    text = _WINDOWS_PATH_RE.sub("[redacted-path]", text)
    text = _POSIX_PATH_RE.sub("[redacted-path]", text)
    return text


def success_envelope(command: str, result: dict, warnings: list[str] | None = None) -> dict:
    return {
        "protocolVersion": 1,
        "command": command,
        "status": "ok",
        "result": result,
        "warnings": list(warnings or []),
    }


def error_envelope(command: str, error: WorkerError, warnings: list[str] | None = None) -> dict:
    return {
        "protocolVersion": 1,
        "command": command,
        "status": "error",
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
        "warnings": list(warnings or []),
    }


def emit_json(payload: dict[str, Any]) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
    )
    sys.stdout.write(encoded + "\n")
    sys.stdout.flush()
