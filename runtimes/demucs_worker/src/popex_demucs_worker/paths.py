from __future__ import annotations

import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

from .protocol import EXIT_INVALID_REQUEST, EXIT_MODEL_VERIFICATION_FAILED, WorkerError

_RUN_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
_REPLACE_SUPPORTS_DIR_FD = os.replace in getattr(os, "supports_dir_fd", set())
_STAT_SUPPORTS_DIR_FD = os.stat in getattr(os, "supports_dir_fd", set())
_UNLINK_SUPPORTS_DIR_FD = os.unlink in getattr(os, "supports_dir_fd", set())


@dataclass(frozen=True, slots=True)
class FileIdentity:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    ctime_ns: int

    @classmethod
    def from_stat(cls, value: os.stat_result) -> "FileIdentity":
        return cls(
            device=int(value.st_dev),
            inode=int(value.st_ino),
            mode=int(value.st_mode),
            size=int(value.st_size),
            mtime_ns=int(value.st_mtime_ns),
            ctime_ns=int(value.st_ctime_ns),
        )

    def same_object(self, other: "FileIdentity") -> bool:
        if self.device != other.device:
            return False
        if self.inode and other.inode:
            return self.inode == other.inode
        return (
            stat.S_IFMT(self.mode) == stat.S_IFMT(other.mode)
            and self.size == other.size
            and self.mtime_ns == other.mtime_ns
            and self.ctime_ns == other.ctime_ns
        )


@dataclass(slots=True)
class AtomicPublication:
    path: Path
    identity: FileIdentity
    parent: Path
    parent_identity: FileIdentity
    directory_fd: int | None
    _closed: bool = False

    def remove(self) -> bool:
        """Remove only the exact file published by this token."""
        if self._closed:
            return False
        current = _child_identity(self.parent, self.path.name, self.directory_fd)
        if current is None or not current.same_object(self.identity):
            return False
        try:
            if self.directory_fd is not None and _UNLINK_SUPPORTS_DIR_FD:
                os.unlink(self.path.name, dir_fd=self.directory_fd)
            else:
                _validate_directory_identity(self.parent, self.parent_identity)
                self.path.unlink()
            _fsync_directory(self.parent, self.directory_fd)
        except FileNotFoundError:
            return False
        return True

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.directory_fd is not None:
            os.close(self.directory_fd)
            self.directory_fd = None


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


def atomic_write_json(cache_root: Path, path: Path, encoded: str) -> AtomicPublication:
    """Publish JSON beneath a revalidated root without following parent symlinks."""
    root = trusted_root(str(cache_root))
    lexical = Path(os.path.abspath(path))
    if not lexical.is_relative_to(root) or lexical == root:
        raise _publication_error("The readiness target escapes the trusted cache root.")
    parent, parent_identity = _ensure_safe_directory(root, lexical.parent)
    directory_fd = _open_directory_fd(parent, parent_identity)
    temporary = parent / f".{lexical.name}.{os.getpid()}.{uuid4().hex}.tmp"
    temporary_identity: FileIdentity | None = None
    published = False
    try:
        _validate_directory_identity(parent, parent_identity)
        _validate_final_target(parent, lexical.name, directory_fd)
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            opened = FileIdentity.from_stat(os.fstat(handle.fileno()))
            anchored = _child_identity(parent, temporary.name, directory_fd)
            if (
                anchored is None
                or not stat.S_ISREG(opened.mode)
                or not anchored.same_object(opened)
            ):
                raise _publication_error("The readiness temporary file is unsafe.")
            temporary_identity = opened
            handle.write(encoded)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            after_write = FileIdentity.from_stat(os.fstat(handle.fileno()))
            anchored_after = _child_identity(parent, temporary.name, directory_fd)
            if (
                anchored_after is None
                or not after_write.same_object(opened)
                or not anchored_after.same_object(opened)
                or not stat.S_ISREG(after_write.mode)
            ):
                raise _publication_error("The readiness temporary file changed unexpectedly.")

        _validate_directory_identity(parent, parent_identity)
        _validate_final_target(parent, lexical.name, directory_fd)
        _replace_child(parent, temporary.name, lexical.name, directory_fd)
        final_identity = _child_identity(parent, lexical.name, directory_fd)
        if final_identity is None or not stat.S_ISREG(final_identity.mode):
            raise _publication_error("The published readiness manifest is unsafe.")
        try:
            _validate_directory_identity(parent, parent_identity)
        except WorkerError:
            _unlink_exact(parent, lexical.name, final_identity, directory_fd)
            raise
        _fsync_directory(parent, directory_fd)
        published = True
        return AtomicPublication(
            path=lexical,
            identity=final_identity,
            parent=parent,
            parent_identity=parent_identity,
            directory_fd=directory_fd,
        )
    finally:
        if not published:
            if temporary_identity is not None:
                _unlink_exact(parent, temporary.name, temporary_identity, directory_fd)
            elif os.path.lexists(temporary):
                try:
                    temporary.unlink()
                except OSError:
                    pass
            if directory_fd is not None:
                os.close(directory_fd)


def _publication_error(message: str) -> WorkerError:
    return WorkerError(
        "READINESS_PUBLICATION_UNSAFE",
        message,
        EXIT_MODEL_VERIFICATION_FAILED,
    )


def _same_path(left: Path, right: Path) -> bool:
    return os.path.normcase(str(left)) == os.path.normcase(str(right))


def _ensure_safe_directory(root: Path, target: Path) -> tuple[Path, FileIdentity]:
    if not target.is_relative_to(root):
        raise _publication_error("The readiness parent escapes the trusted cache root.")
    current = root
    root_identity = FileIdentity.from_stat(os.lstat(root))
    if stat.S_ISLNK(root_identity.mode) or not stat.S_ISDIR(root_identity.mode):
        raise _publication_error("The trusted cache root is unsafe.")
    for part in target.relative_to(root).parts:
        current = current / part
        if os.path.lexists(current):
            identity = FileIdentity.from_stat(os.lstat(current))
            if stat.S_ISLNK(identity.mode):
                raise _publication_error("A readiness parent component is unsafe.")
            if not stat.S_ISDIR(identity.mode):
                # Preserve the long-standing direct-function exception contract
                # while the CLI still maps unexpected filesystem failures safely.
                raise NotADirectoryError(str(current))
        else:
            try:
                current.mkdir(mode=0o700)
            except OSError as exc:
                raise _publication_error("A readiness parent component could not be created.") from exc
            identity = FileIdentity.from_stat(os.lstat(current))
            if stat.S_ISLNK(identity.mode) or not stat.S_ISDIR(identity.mode):
                raise _publication_error("A readiness parent component is unsafe.")
        try:
            resolved = current.resolve(strict=True)
        except OSError as exc:
            raise _publication_error("A readiness parent component is unavailable.") from exc
        if not resolved.is_relative_to(root) or not _same_path(resolved, current):
            raise _publication_error("A readiness parent component escapes the trusted cache root.")
    identity = FileIdentity.from_stat(os.lstat(target))
    return target, identity


def _validate_directory_identity(path: Path, expected: FileIdentity) -> None:
    try:
        current = FileIdentity.from_stat(os.lstat(path))
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise _publication_error("The readiness parent changed unexpectedly.") from exc
    if (
        stat.S_ISLNK(current.mode)
        or not stat.S_ISDIR(current.mode)
        or not current.same_object(expected)
        or not _same_path(resolved, path)
    ):
        raise _publication_error("The readiness parent changed unexpectedly.")


def _open_directory_fd(parent: Path, expected: FileIdentity) -> int | None:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(parent, flags)
    except (OSError, TypeError, NotImplementedError):
        return None
    current = FileIdentity.from_stat(os.fstat(descriptor))
    if not stat.S_ISDIR(current.mode) or not current.same_object(expected):
        os.close(descriptor)
        raise _publication_error("The readiness parent could not be anchored safely.")
    return descriptor


def _child_identity(parent: Path, name: str, directory_fd: int | None) -> FileIdentity | None:
    try:
        if directory_fd is not None and _STAT_SUPPORTS_DIR_FD:
            value = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        else:
            value = os.lstat(parent / name)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise _publication_error("A readiness file could not be inspected safely.") from exc
    return FileIdentity.from_stat(value)


def _validate_final_target(parent: Path, name: str, directory_fd: int | None) -> None:
    current = _child_identity(parent, name, directory_fd)
    if current is None:
        return
    if stat.S_ISLNK(current.mode) or not stat.S_ISREG(current.mode):
        raise _publication_error("The readiness target is not a safe regular file.")


def _replace_child(parent: Path, source: str, destination: str, directory_fd: int | None) -> None:
    if directory_fd is not None and _REPLACE_SUPPORTS_DIR_FD:
        try:
            os.replace(
                source,
                destination,
                src_dir_fd=directory_fd,
                dst_dir_fd=directory_fd,
            )
            return
        except TypeError:
            # Preserve compatibility with platforms or deterministic tests whose
            # replacement hook accepts only the two path arguments.
            pass
    os.replace(parent / source, parent / destination)


def _unlink_exact(
    parent: Path,
    name: str,
    expected: FileIdentity,
    directory_fd: int | None,
) -> bool:
    try:
        current = _child_identity(parent, name, directory_fd)
        if current is None or not current.same_object(expected):
            return False
        if directory_fd is not None and _UNLINK_SUPPORTS_DIR_FD:
            os.unlink(name, dir_fd=directory_fd)
        else:
            (parent / name).unlink()
        return True
    except OSError:
        return False


def _fsync_directory(parent: Path, directory_fd: int | None) -> None:
    if directory_fd is not None:
        try:
            os.fsync(directory_fd)
        except OSError:
            return
        return
    try:
        descriptor = os.open(parent, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        return
    finally:
        os.close(descriptor)
