from __future__ import annotations

import os
import re
import stat
from pathlib import Path, PurePosixPath

from .protocol import EXIT_INVALID_REQUEST, EXIT_MODEL_VERIFICATION_FAILED, WorkerError

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")


def trusted_root(raw: str, *, create: bool = False, code: str = "INVALID_PATH") -> Path:
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise WorkerError(code, "The trusted root is invalid.", EXIT_INVALID_REQUEST)
    path = Path(raw)
    if not path.is_absolute():
        raise WorkerError(code, "The trusted root must be absolute.", EXIT_INVALID_REQUEST)
    if path.exists() and path.is_symlink():
        raise WorkerError(code, "The trusted root may not be a symbolic link.", EXIT_INVALID_REQUEST)
    if create:
        path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise WorkerError(code, "The trusted root is not an available directory.", EXIT_INVALID_REQUEST)
    return path.resolve(strict=True)


def safe_posix_relative(raw: object, *, error_code: str = "UNSAFE_PATH") -> PurePosixPath:
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        raise WorkerError(
            error_code,
            "A relative asset path is unsafe.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    path = PurePosixPath(raw)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkerError(
            error_code,
            "A relative asset path is unsafe.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    return path


def resolve_contained(
    root: Path,
    relative: object,
    *,
    require_regular_file: bool = False,
    require_directory: bool = False,
    error_code: str = "UNSAFE_PATH",
    exit_code: int = EXIT_MODEL_VERIFICATION_FAILED,
) -> Path:
    try:
        rel = safe_posix_relative(relative, error_code=error_code)
    except WorkerError as exc:
        exc.exit_code = exit_code
        raise
    lexical = root.joinpath(*rel.parts)
    try:
        resolved = lexical.resolve(strict=require_regular_file or require_directory)
    except OSError as exc:
        raise WorkerError(error_code, "A required contained path is unavailable.", exit_code) from exc
    if not resolved.is_relative_to(root):
        raise WorkerError(error_code, "A contained path escapes its trusted root.", exit_code)
    if require_regular_file:
        try:
            mode = resolved.stat().st_mode
        except OSError as exc:
            raise WorkerError(error_code, "A required model asset is unavailable.", exit_code) from exc
        if not stat.S_ISREG(mode):
            raise WorkerError(error_code, "A required model asset is not a regular file.", exit_code)
    if require_directory and not resolved.is_dir():
        raise WorkerError(error_code, "A required contained directory is unavailable.", exit_code)
    return resolved


def relative_asset_path(root: Path, path: Path) -> str:
    lexical = Path(os.path.abspath(path))
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            "A downloaded model asset is unavailable.",
            EXIT_MODEL_VERIFICATION_FAILED,
        ) from exc
    if not lexical.is_relative_to(root) or not resolved.is_relative_to(root):
        raise WorkerError(
            "MODEL_ASSET_INVALID",
            "A downloaded model asset escaped the cache root.",
            EXIT_MODEL_VERIFICATION_FAILED,
        )
    relative = lexical.relative_to(root).as_posix()
    safe_posix_relative(relative)
    return relative


def validate_input_file(workspace_root: Path, input_relative: str) -> Path:
    if input_relative != "analysis.wav":
        raise WorkerError(
            "INVALID_INPUT",
            "The worker accepts only analysis.wav as input.",
            EXIT_INVALID_REQUEST,
        )
    path = resolve_contained(
        workspace_root,
        input_relative,
        require_regular_file=True,
        error_code="INVALID_INPUT",
        exit_code=EXIT_INVALID_REQUEST,
    )
    lexical = workspace_root / input_relative
    if lexical.is_symlink():
        raise WorkerError(
            "INVALID_INPUT",
            "The input audio may not be a symbolic link.",
            EXIT_INVALID_REQUEST,
        )
    return path


def validate_output_directory(workspace_root: Path, output_relative: str) -> Path:
    try:
        rel = safe_posix_relative(output_relative, error_code="INVALID_OUTPUT")
    except WorkerError as exc:
        exc.exit_code = EXIT_INVALID_REQUEST
        raise
    if (
        len(rel.parts) != 4
        or rel.parts[0:2] != ("stems", "runs")
        or not _RUN_ID_PATTERN.fullmatch(rel.parts[2])
        or rel.parts[3] != "worker-output"
    ):
        raise WorkerError(
            "INVALID_OUTPUT",
            "The output directory is not an allocated worker run directory.",
            EXIT_INVALID_REQUEST,
        )
    lexical = workspace_root.joinpath(*rel.parts)
    resolved = lexical.resolve(strict=False)
    if not resolved.is_relative_to(workspace_root):
        raise WorkerError(
            "INVALID_OUTPUT",
            "The output directory escapes the workspace root.",
            EXIT_INVALID_REQUEST,
        )
    current = workspace_root
    for part in rel.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise WorkerError(
                "INVALID_OUTPUT",
                "The output directory may not use symbolic links.",
                EXIT_INVALID_REQUEST,
            )
    if lexical.exists():
        if not lexical.is_dir() or any(lexical.iterdir()):
            raise WorkerError(
                "INVALID_OUTPUT",
                "The output directory must be new or empty.",
                EXIT_INVALID_REQUEST,
            )
    else:
        lexical.mkdir(parents=True, exist_ok=False)
    return lexical.resolve(strict=True)


def atomic_write_json(path: Path, encoded: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        try:
            directory_fd = os.open(path.parent, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()
