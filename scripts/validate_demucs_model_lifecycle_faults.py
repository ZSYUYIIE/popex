#!/usr/bin/env python3
"""Run deterministic offline Demucs model-lifecycle fault validation.

This command never prepares or downloads the audited model. It delegates to the
permanent pytest characterization suite, forces Hub offline/privacy controls,
and emits one bounded path-free JSON summary after pytest exits.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
FOCUSED_TEST = "tests/test_demucs_model_lifecycle_faults.py"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run offline deterministic Demucs model lifecycle fault tests."
    )
    parser.add_argument(
        "--maxfail",
        type=int,
        default=0,
        help="Optional pytest max-failure count; zero runs the full characterization.",
    )
    return parser


def _environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "HF_HUB_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
            "HF_HUB_DISABLE_UPDATE_CHECK": "1",
            "HF_HUB_DISABLE_PROGRESS_BARS": "1",
            "PYTHONNOUSERSITE": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONUTF8": "1",
        }
    )
    for key in tuple(environment):
        upper = key.upper()
        if any(marker in upper for marker in ("TOKEN", "SECRET", "PASSWORD", "API_KEY")):
            environment.pop(key, None)
    return environment


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.maxfail < 0:
        print("maxfail must not be negative", file=sys.stderr)
        return 2

    command = [sys.executable, "-m", "pytest", "-q", FOCUSED_TEST]
    if args.maxfail:
        command.extend(["--maxfail", str(args.maxfail)])
    completed = subprocess.run(
        command,
        cwd=ROOT,
        env=_environment(),
        check=False,
    )
    summary = {
        "schemaVersion": 1,
        "suite": "demucs-model-lifecycle-faults",
        "status": "passed" if completed.returncode == 0 else "blocked",
        "pytestExitCode": completed.returncode,
        "realModelDownloaded": False,
        "modelHostAccessAllowed": False,
        "gpuRequired": False,
    }
    print(
        json.dumps(
            summary,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
