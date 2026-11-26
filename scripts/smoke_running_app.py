"""Smoke test of an already running PopEx over HTTP, the way a browser uses it.

Use it against a server started by any documented route (``docker compose
up``, ``run_windows.ps1``, ``uvicorn``). It waits for the page, uploads a
short synthetic recording, starts **Draft the score** exactly as the page
does, waits for the draft, and checks the saved score, its downloads and
the notation page. Only synthetic audio is used.

Usage::

    python scripts/smoke_running_app.py [--base-url http://127.0.0.1:8000] [--timeout 900]

``--expect-jobs N`` instead only checks that the server answers and already
lists at least N recordings with a saved score (for a restart check).
Prints a JSON summary and exits non-zero on failure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from smoke_score_pipeline import synthetic_recording  # noqa: E402


def _request(url: str, *, data: bytes | None = None, headers: dict[str, str] | None = None,
             method: str | None = None, timeout: float = 30.0) -> tuple[int, bytes]:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def _json(url: str, **kwargs: Any) -> tuple[int, Any]:
    status, body = _request(url, **kwargs)
    try:
        return status, json.loads(body or b"null")
    except ValueError:
        return status, None


def _multipart(field: str, filename: str, content: bytes, content_type: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    head = (
        f"--{boundary}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
        f"Content-Type: {content_type}\r\n\r\n"
    ).encode()
    return head + content + f"\r\n--{boundary}--\r\n".encode(), f"multipart/form-data; boundary={boundary}"


def wait_for_page(base: str, timeout: float) -> str:
    deadline = time.monotonic() + timeout
    last = "no answer"
    while time.monotonic() < deadline:
        try:
            status, body = _request(base + "/", timeout=5)
            if status == 200 and b"Add a recording" in body:
                return "ok"
            last = f"HTTP {status}"
        except OSError as error:
            last = str(error)
        time.sleep(3)
    raise SystemExit(f"PopEx did not serve its page at {base}/ ({last}).")


def run(base: str, timeout: float) -> dict[str, Any]:
    wait_for_page(base, timeout)
    body, content_type = _multipart("file", "smoke-running-app.wav", synthetic_recording(), "audio/wav")
    status, created = _json(base + "/api/uploads", data=body, headers={"Content-Type": content_type}, method="POST",
                            timeout=120)
    if status not in (200, 201, 202) or not isinstance(created, dict) or "id" not in created:
        raise SystemExit(f"Upload was rejected: HTTP {status} {str(created)[:300]}")
    job_id = created["id"]
    # The page starts drafting right after the upload; drafting waits for analysis.
    draft = json.dumps({"allowModelDownload": False}).encode()
    for _ in range(20):
        status, _payload = _json(f"{base}/api/jobs/{job_id}/draft", data=draft,
                                 headers={"Content-Type": "application/json"}, method="POST")
        if status == 202:
            break
        time.sleep(3)
    else:
        raise SystemExit(f"Drafting could not be started: HTTP {status}")

    deadline = time.monotonic() + timeout
    job: dict[str, Any] = {}
    while time.monotonic() < deadline:
        _status, jobs = _json(base + "/api/jobs")
        job = next((item for item in jobs or [] if item.get("id") == job_id), {})
        score = job.get("score") or {}
        draft_state = (job.get("draft") or {}).get("status")
        if score.get("status") == "completed" and score.get("available"):
            break
        if score.get("status") == "failed" or draft_state == "failed" or job.get("status") == "failed":
            raise SystemExit(f"Drafting failed: {json.dumps(job)[:1500]}")
        time.sleep(3)
    else:
        raise SystemExit(f"The draft did not finish in {timeout:.0f}s: {json.dumps(job)[:1500]}")

    _status, details = _json(f"{base}/api/jobs/{job_id}/score/saved?includeMeasures=false")
    downloads = {}
    for name in ("musicxml", "midi"):
        code, content = _request(f"{base}/api/jobs/{job_id}/score/saved/download?format={name}")
        downloads[name] = {"status": code, "bytes": len(content)}
    notation_status, _ = _request(f"{base}/jobs/{job_id}/notation")
    summary = {
        "jobId": job_id,
        "scoreVersion": details.get("version"),
        "layers": {key: value.get("status") for key, value in (details.get("layers") or {}).items()},
        "parts": [part.get("id") for part in details.get("scoreParts") or []],
        "counts": {key: (details.get("counts") or {}).get(key) for key in ("measures", "notes", "percussionHits")},
        "downloads": downloads,
        "notationPage": notation_status,
    }
    problems = []
    if summary["layers"].get("pitchedNotes") != "included" or not summary["counts"]["notes"]:
        problems.append("no pitched notes were written")
    if summary["layers"].get("percussion") != "included":
        problems.append("no drum part was written")
    if any(item["status"] != 200 or item["bytes"] < 100 for item in downloads.values()):
        problems.append("a score download failed")
    if notation_status != 200:
        problems.append("the notation page did not load")
    summary["problems"] = problems
    return summary


def check_saved(base: str, expected: int, timeout: float) -> dict[str, Any]:
    wait_for_page(base, timeout)
    _status, jobs = _json(base + "/api/jobs")
    saved = [job for job in jobs or [] if (job.get("score") or {}).get("available")]
    summary = {"recordingsWithScores": len(saved), "problems": []}
    if len(saved) < expected:
        summary["problems"].append(f"expected at least {expected} saved score(s), found {len(saved)}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--expect-jobs", type=int, default=None)
    args = parser.parse_args()
    base = args.base_url.rstrip("/")
    if args.expect_jobs is not None:
        summary = check_saved(base, args.expect_jobs, args.timeout)
    else:
        summary = run(base, args.timeout)
    print(json.dumps(summary, indent=2))
    return 1 if summary["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
