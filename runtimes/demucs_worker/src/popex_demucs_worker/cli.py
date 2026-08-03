from __future__ import annotations

import argparse
import contextlib
import importlib
import io
import sys
from collections.abc import Sequence

from .constants import PROTOCOL_VERSION
from .probes import model_probe, runtime_probe
from .protocol import (
    EXIT_CANCELLED,
    EXIT_INTERNAL,
    EXIT_INVALID_REQUEST,
    EXIT_TIMEOUT,
    WorkerError,
    emit_json,
    error_envelope,
    sanitize_diagnostic,
    success_envelope,
)

_COMMANDS = {
    "runtime-probe",
    "model-probe",
    "prepare-model",
    "verify-model",
    "separate",
}


class ProtocolParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        raise WorkerError(
            "INVALID_REQUEST",
            "The worker request is invalid.",
            EXIT_INVALID_REQUEST,
        )


def _parser() -> ProtocolParser:
    parser = ProtocolParser(prog="popex-demucs-worker", add_help=True)
    parser.add_argument("--protocol-version", type=int, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("runtime-probe")

    model_probe_parser = subparsers.add_parser("model-probe")
    model_probe_parser.add_argument("--cache-root", required=True)

    prepare_parser = subparsers.add_parser("prepare-model")
    prepare_parser.add_argument("--cache-root", required=True)

    verify_parser = subparsers.add_parser("verify-model")
    verify_parser.add_argument("--cache-root", required=True)

    separate_parser = subparsers.add_parser("separate")
    separate_parser.add_argument("--cache-root", required=True)
    separate_parser.add_argument("--workspace-root", required=True)
    separate_parser.add_argument("--input-relative", required=True)
    separate_parser.add_argument("--output-relative", required=True)
    separate_parser.add_argument("--device", required=True)
    return parser


def _guess_command(argv: Sequence[str]) -> str:
    for value in argv:
        if value in _COMMANDS:
            return value
    return "unknown"


def _dispatch(args: argparse.Namespace) -> dict:
    if args.protocol_version != PROTOCOL_VERSION:
        raise WorkerError(
            "UNSUPPORTED_PROTOCOL",
            "The requested worker protocol version is not supported.",
            EXIT_INVALID_REQUEST,
        )
    if args.command == "runtime-probe":
        return runtime_probe()
    if args.command == "model-probe":
        return model_probe(args.cache_root)

    commands = importlib.import_module("popex_demucs_worker.commands")
    if args.command == "prepare-model":
        return commands.prepare_model(args.cache_root)
    if args.command == "verify-model":
        return commands.verify_model(args.cache_root)
    if args.command == "separate":
        return commands.separate(
            args.cache_root,
            args.workspace_root,
            args.input_relative,
            args.output_relative,
            args.device,
        )
    raise WorkerError(
        "INVALID_REQUEST",
        "The worker command is invalid.",
        EXIT_INVALID_REQUEST,
    )


def main(argv: Sequence[str] | None = None) -> int:
    values = list(sys.argv[1:] if argv is None else argv)
    command = _guess_command(values)
    diagnostics = io.StringIO()
    try:
        args = _parser().parse_args(values)
        command = args.command
        with contextlib.redirect_stdout(diagnostics):
            result = _dispatch(args)
        envelope = success_envelope(command, result)
        exit_code = 0
    except WorkerError as exc:
        envelope = error_envelope(command, exc)
        exit_code = exc.exit_code
    except KeyboardInterrupt:
        error = WorkerError(
            "CANCELLED",
            "The worker operation was cancelled.",
            EXIT_CANCELLED,
            retryable=True,
        )
        envelope = error_envelope(command, error)
        exit_code = EXIT_CANCELLED
    except TimeoutError:
        error = WorkerError(
            "WORKER_TIMEOUT",
            "The worker operation timed out.",
            EXIT_TIMEOUT,
            retryable=True,
        )
        envelope = error_envelope(command, error)
        exit_code = EXIT_TIMEOUT
    except SystemExit as exc:
        error = WorkerError(
            "INVALID_REQUEST",
            "The worker request is invalid.",
            EXIT_INVALID_REQUEST,
        )
        envelope = error_envelope(command, error)
        exit_code = EXIT_INVALID_REQUEST if exc.code else 0
    except BaseException as exc:
        diagnostic = sanitize_diagnostic(f"{type(exc).__name__}: {exc}")
        if diagnostic:
            diagnostics.write(diagnostic)
        error = WorkerError(
            "INTERNAL_ERROR",
            "The worker encountered an unexpected internal failure.",
            EXIT_INTERNAL,
            retryable=False,
        )
        envelope = error_envelope(command, error)
        exit_code = EXIT_INTERNAL

    captured = diagnostics.getvalue().strip()
    if captured:
        sys.stderr.write(sanitize_diagnostic(captured) + "\n")
        sys.stderr.flush()
    try:
        emit_json(envelope)
    except (TypeError, ValueError):
        fallback = WorkerError(
            "INTERNAL_ERROR",
            "The worker produced an invalid structured result.",
            EXIT_INTERNAL,
            retryable=False,
        )
        emit_json(error_envelope(command, fallback))
        return EXIT_INTERNAL
    return exit_code
