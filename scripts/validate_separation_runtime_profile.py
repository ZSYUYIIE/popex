#!/usr/bin/env python3
"""Validate one installed optional Demucs runtime through the base client.

The script intentionally imports only Python's standard library and
``app.separation_runtime``. It never prepares, verifies, downloads, or uses a
model. Success means the installed runtime matches its profile lock and the
model probe reports that an explicit download is still required.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import stat
import sys
from pathlib import Path
from typing import Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from app.separation_runtime import (  # noqa: E402
    DEMUCS_VERSION,
    SeparationRuntimeClient,
    SeparationRuntimeError,
    WorkerCommandError,
)

_SCHEMA_VERSION = 1
_EXPECTED_WORKER_VERSION = "1.0.0"
_MODEL_DOWNLOAD_EXIT_CODE = 20
_PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_READINESS_FILE = "htdemucs-bf35a81b-v1.json"
_MODEL_SUFFIXES = frozenset({".safetensors", ".th", ".ckpt"})


class ProfileValidationError(RuntimeError):
    """The installed runtime did not satisfy the smoke-test contract."""


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _trusted_absolute_path(value: str, label: str, *, kind: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ProfileValidationError(f"{label} must be a non-empty local path")
    path = Path(value)
    if not path.is_absolute():
        raise ProfileValidationError(f"{label} must be absolute")
    normalized = Path(os.path.normpath(str(path)))
    if not _same_path(path, normalized):
        raise ProfileValidationError(f"{label} must be normalized")
    try:
        lexical = path.lstat()
        resolved = path.resolve(strict=True)
        resolved_info = resolved.lstat()
    except (FileNotFoundError, NotADirectoryError, OSError) as exc:
        raise ProfileValidationError(f"{label} must exist") from exc
    if stat.S_ISLNK(lexical.st_mode) or stat.S_ISLNK(resolved_info.st_mode):
        raise ProfileValidationError(f"{label} must not be a symlink")
    if not _same_path(path, resolved):
        raise ProfileValidationError(f"{label} must not contain symlink components")
    if kind == "file" and not stat.S_ISREG(resolved_info.st_mode):
        raise ProfileValidationError(f"{label} must be a regular file")
    if kind == "directory" and not stat.S_ISDIR(resolved_info.st_mode):
        raise ProfileValidationError(f"{label} must be a directory")
    return resolved


def _validate_expected_profile(value: str) -> str:
    if not isinstance(value, str) or _PROFILE_PATTERN.fullmatch(value) is None:
        raise ProfileValidationError("expected profile identifier is invalid")
    return value


def _model_assets(root: Path) -> tuple[str, ...]:
    assets: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if name == _READINESS_FILE or path.suffix.lower() in _MODEL_SUFFIXES:
            assets.append(path.relative_to(root).as_posix())
    return tuple(sorted(assets))


def validate_profile(
    *,
    worker: Path,
    runtime_lock: Path,
    cache_root: Path,
    expected_profile: str,
) -> dict[str, object]:
    if any(cache_root.iterdir()):
        raise ProfileValidationError("cache root must be empty before probing")

    client = SeparationRuntimeClient(
        worker_executable=worker,
        cache_root=cache_root,
        runtime_lock_path=runtime_lock,
        expected_runtime_profile=expected_profile,
    )

    runtime = client.runtime_probe()
    if runtime.runtime_profile != expected_profile:
        raise ProfileValidationError("runtime probe returned an unexpected profile")
    if runtime.runtime_lock_source != "profile":
        raise ProfileValidationError("runtime probe did not use the supplied profile lock")
    if runtime.worker_version != _EXPECTED_WORKER_VERSION:
        raise ProfileValidationError("runtime probe returned an unexpected worker version")
    if runtime.demucs_version != DEMUCS_VERSION:
        raise ProfileValidationError("runtime probe returned an unexpected Demucs version")

    try:
        client.model_probe()
    except WorkerCommandError as exc:
        if (
            exc.code != "MODEL_DOWNLOAD_REQUIRED"
            or exc.detail.worker_code != "MODEL_DOWNLOAD_REQUIRED"
            or exc.detail.exit_code != _MODEL_DOWNLOAD_EXIT_CODE
        ):
            raise ProfileValidationError(
                "model probe returned an unexpected structured worker failure"
            ) from None
    except SeparationRuntimeError as exc:
        raise ProfileValidationError(
            f"model probe returned unexpected runtime error {exc.code}"
        ) from None
    else:
        raise ProfileValidationError(
            "model probe unexpectedly reported that the model is ready"
        )

    if _model_assets(cache_root):
        raise ProfileValidationError("model or readiness assets were created")
    if any(cache_root.iterdir()):
        raise ProfileValidationError("model probe created cache content")

    return {
        "schemaVersion": _SCHEMA_VERSION,
        "runtimeProfile": runtime.runtime_profile,
        "workerVersion": runtime.worker_version,
        "demucsVersion": runtime.demucs_version,
        "torchVersion": runtime.torch_version,
        "modelState": "download_required",
        "modelAssetsCreated": False,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an installed PopEx Demucs runtime through the base client."
    )
    parser.add_argument("--worker", required=True)
    parser.add_argument("--runtime-lock", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--expected-profile", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        worker = _trusted_absolute_path(args.worker, "worker", kind="file")
        runtime_lock = _trusted_absolute_path(
            args.runtime_lock, "runtime lock", kind="file"
        )
        cache_root = _trusted_absolute_path(
            args.cache_root, "cache root", kind="directory"
        )
        expected_profile = _validate_expected_profile(args.expected_profile)
        result = validate_profile(
            worker=worker,
            runtime_lock=runtime_lock,
            cache_root=cache_root,
            expected_profile=expected_profile,
        )
    except ProfileValidationError as exc:
        print(f"runtime profile validation failed: {exc}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as exc:
        print(
            f"runtime profile validation failed: {type(exc).__name__}",
            file=sys.stderr,
        )
        return 1

    print(
        json.dumps(
            result,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
