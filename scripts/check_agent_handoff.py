from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

MARKER = "<!-- popex-agent-handoff:v1 -->"
AGENT_BRANCH_PREFIX = "agent/"
SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_SECTIONS = (
    "## Scope",
    "## Files changed",
    "## Public or integration contract",
    "## Validation",
    "## Integration notes",
    "## Risks and blockers",
    "## Final recommendation",
)

REQUIRED_TOP_LEVEL_KEYS = {
    "schemaVersion",
    "agent",
    "cycle",
    "status",
    "branch",
    "baseSha",
    "headSha",
    "pr",
    "ownedPaths",
    "unauthorizedFilesChanged",
    "ci",
    "mergeRecommendation",
}

ALLOWED_STATUSES = {"running", "blocked", "ready_for_review", "superseded"}
ALLOWED_CI_STATUSES = {"pending", "passed", "failed"}
ALLOWED_MERGE_RECOMMENDATIONS = {"hold", "merge"}


class DuplicateKeyError(ValueError):
    pass


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DuplicateKeyError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _extract_handoff(body: str) -> tuple[dict[str, Any] | None, list[str]]:
    errors: list[str] = []
    if MARKER not in body:
        return None, [f"missing required marker: {MARKER}"]

    after_marker = body.split(MARKER, 1)[1]
    match = re.search(r"```json\s*(\{.*?\})\s*```", after_marker, re.DOTALL)
    if match is None:
        return None, ["missing JSON handoff block immediately after the marker"]

    try:
        payload = json.loads(match.group(1), object_pairs_hook=_strict_object)
    except (json.JSONDecodeError, DuplicateKeyError) as exc:
        return None, [f"invalid handoff JSON: {exc}"]

    if not isinstance(payload, dict):
        return None, ["handoff JSON must be an object"]

    missing = REQUIRED_TOP_LEVEL_KEYS - set(payload)
    unknown = set(payload) - REQUIRED_TOP_LEVEL_KEYS
    if missing:
        errors.append(f"missing handoff keys: {', '.join(sorted(missing))}")
    if unknown:
        errors.append(f"unknown handoff keys: {', '.join(sorted(unknown))}")
    return payload, errors


def validate_event(event: dict[str, Any], *, legacy_through: int = 13) -> list[str]:
    pull_request = event.get("pull_request")
    if not isinstance(pull_request, dict):
        return ["event does not contain a pull_request object"]

    number = event.get("number")
    if not isinstance(number, int):
        number = pull_request.get("number")
    if not isinstance(number, int):
        return ["pull request number is missing"]

    head = pull_request.get("head")
    base = pull_request.get("base")
    if not isinstance(head, dict) or not isinstance(base, dict):
        return ["pull request head/base metadata is missing"]

    head_ref = head.get("ref")
    head_sha = head.get("sha")
    base_sha = base.get("sha")
    if not isinstance(head_ref, str):
        return ["pull request head ref is missing"]

    if not head_ref.startswith(AGENT_BRANCH_PREFIX):
        return []
    if number <= legacy_through:
        print(
            f"agent handoff check: legacy PR #{number} is exempt through #{legacy_through}",
            file=sys.stderr,
        )
        return []

    body = pull_request.get("body")
    if not isinstance(body, str):
        body = ""

    payload, errors = _extract_handoff(body)
    if payload is None:
        return errors

    if payload.get("schemaVersion") != 1:
        errors.append("schemaVersion must be exactly 1")
    if not _nonempty_string(payload.get("agent")):
        errors.append("agent must be a non-empty string")
    if not _nonempty_string(payload.get("cycle")):
        errors.append("cycle must be a non-empty string")

    status = payload.get("status")
    if status not in ALLOWED_STATUSES:
        errors.append(f"status must be one of: {', '.join(sorted(ALLOWED_STATUSES))}")

    if payload.get("branch") != head_ref:
        errors.append("branch must exactly match the pull request head ref")

    recorded_base_sha = payload.get("baseSha")
    recorded_head_sha = payload.get("headSha")
    if not isinstance(recorded_base_sha, str) or SHA_RE.fullmatch(recorded_base_sha) is None:
        errors.append("baseSha must be a 40-character lowercase hexadecimal SHA")
    elif recorded_base_sha != base_sha:
        errors.append("baseSha must match the pull request base SHA")
    if not isinstance(recorded_head_sha, str) or SHA_RE.fullmatch(recorded_head_sha) is None:
        errors.append("headSha must be a 40-character lowercase hexadecimal SHA")
    elif recorded_head_sha != head_sha:
        errors.append("headSha must match the pull request head SHA")

    if payload.get("pr") != number:
        errors.append("pr must equal the pull request number")

    owned_paths = payload.get("ownedPaths")
    if (
        not isinstance(owned_paths, list)
        or not owned_paths
        or any(not _nonempty_string(item) for item in owned_paths)
    ):
        errors.append("ownedPaths must be a non-empty array of non-empty strings")

    unauthorized = payload.get("unauthorizedFilesChanged")
    if type(unauthorized) is not bool:
        errors.append("unauthorizedFilesChanged must be a boolean")
    elif unauthorized and status != "blocked":
        errors.append("unauthorized file changes require status=blocked")

    ci = payload.get("ci")
    if not isinstance(ci, dict) or set(ci) != {"status", "run"}:
        errors.append("ci must contain exactly status and run")
        ci_status = None
        ci_run = None
    else:
        ci_status = ci.get("status")
        ci_run = ci.get("run")
        if ci_status not in ALLOWED_CI_STATUSES:
            errors.append(
                f"ci.status must be one of: {', '.join(sorted(ALLOWED_CI_STATUSES))}"
            )
        if ci_run is not None and (type(ci_run) is not int or ci_run <= 0):
            errors.append("ci.run must be null or a positive integer")

    recommendation = payload.get("mergeRecommendation")
    if recommendation not in ALLOWED_MERGE_RECOMMENDATIONS:
        errors.append(
            "mergeRecommendation must be one of: "
            + ", ".join(sorted(ALLOWED_MERGE_RECOMMENDATIONS))
        )

    if status == "ready_for_review":
        if ci_status != "passed" or type(ci_run) is not int:
            errors.append("ready_for_review requires a passing CI run number")
    if recommendation == "merge":
        if status != "ready_for_review" or ci_status != "passed":
            errors.append("merge recommendation requires ready_for_review and passing CI")
        if unauthorized is True:
            errors.append("merge recommendation is forbidden with unauthorized file changes")

    for section in REQUIRED_SECTIONS:
        if section not in body:
            errors.append(f"missing required Markdown section: {section}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate a PopEx agent PR handoff")
    parser.add_argument("--event", type=Path, required=True)
    parser.add_argument("--legacy-through", type=int, default=13)
    args = parser.parse_args(argv)

    try:
        event = json.loads(args.event.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        print(f"could not read pull-request event: {exc}", file=sys.stderr)
        return 2

    errors = validate_event(event, legacy_through=args.legacy_through)
    if errors:
        print("Agent handoff validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    print("Agent handoff validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())