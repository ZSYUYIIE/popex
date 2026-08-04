from __future__ import annotations

import importlib.util
from pathlib import Path


_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "check_agent_handoff.py"
_SPEC = importlib.util.spec_from_file_location("popex_check_agent_handoff", _SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

MARKER = _MODULE.MARKER
validate_event = _MODULE.validate_event

BASE_SHA = "a" * 40
HEAD_SHA = "b" * 40


def handoff_body(*, status: str = "running", ci_status: str = "pending", ci_run=None) -> str:
    return f"""{MARKER}
```json
{{
  "schemaVersion": 1,
  "agent": "Agent 1",
  "cycle": "cycle-3",
  "status": "{status}",
  "branch": "agent/example",
  "baseSha": "{BASE_SHA}",
  "headSha": "{HEAD_SHA}",
  "pr": 14,
  "ownedPaths": ["app/example.py"],
  "unauthorizedFilesChanged": false,
  "ci": {{"status": "{ci_status}", "run": {"null" if ci_run is None else ci_run}}},
  "mergeRecommendation": "hold"
}}
```

## Scope

Focused work.

## Files changed

- `app/example.py`

## Public or integration contract

A contract.

## Validation

Tests.

## Integration notes

Notes.

## Risks and blockers

None identified.

## Final recommendation

READY FOR INTEGRATION REVIEW
"""


def pull_request_event(body: str | None = None, *, number: int = 14, branch: str = "agent/example"):
    return {
        "number": number,
        "pull_request": {
            "number": number,
            "body": handoff_body() if body is None else body,
            "head": {"ref": branch, "sha": HEAD_SHA},
            "base": {"ref": "main", "sha": BASE_SHA},
        },
    }


def test_valid_running_handoff_passes():
    assert validate_event(pull_request_event()) == []


def test_valid_ready_handoff_requires_passing_ci():
    body = handoff_body(status="ready_for_review", ci_status="passed", ci_run=81)
    assert validate_event(pull_request_event(body)) == []


def test_missing_marker_fails():
    errors = validate_event(pull_request_event("## Scope\n"))
    assert any("missing required marker" in error for error in errors)


def test_head_sha_must_match_pr_head():
    event = pull_request_event()
    event["pull_request"]["head"]["sha"] = "c" * 40
    errors = validate_event(event)
    assert "headSha must match the pull request head SHA" in errors


def test_ready_handoff_without_green_ci_fails():
    body = handoff_body(status="ready_for_review", ci_status="pending", ci_run=None)
    errors = validate_event(pull_request_event(body))
    assert "ready_for_review requires a passing CI run number" in errors


def test_legacy_agent_pr_is_exempt():
    event = pull_request_event("", number=13)
    assert validate_event(event, legacy_through=13) == []


def test_non_agent_branch_is_exempt():
    event = pull_request_event("", branch="docs/example")
    assert validate_event(event) == []


def test_unauthorized_changes_require_blocked_status():
    body = handoff_body().replace(
        '"unauthorizedFilesChanged": false',
        '"unauthorizedFilesChanged": true',
    )
    errors = validate_event(pull_request_event(body))
    assert "unauthorized file changes require status=blocked" in errors


def test_missing_required_section_fails():
    body = handoff_body().replace("## Integration notes", "## Other notes")
    errors = validate_event(pull_request_event(body))
    assert "missing required Markdown section: ## Integration notes" in errors


def test_duplicate_json_key_fails():
    body = handoff_body().replace(
        '"schemaVersion": 1,',
        '"schemaVersion": 1,\n  "schemaVersion": 1,',
    )
    errors = validate_event(pull_request_event(body))
    assert any("duplicate JSON key" in error for error in errors)
