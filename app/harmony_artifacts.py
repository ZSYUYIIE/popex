from __future__ import annotations

import copy
import json
import math
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from app.analysis import ANALYSIS_JSON_RELATIVE_PATH
from app.config import Settings
from app.harmony_inference import HarmonyInferenceResult
from app.media import MediaProcessingError, secure_job_dir
from app.transcription_draft import (
    INTERPRETATION_DRAFT_RELATIVE_PATH,
    INTERPRETATION_DRAFT_SCHEMA_VERSION,
)
from app.transcription_events import (
    RAW_TRANSCRIPTION_RELATIVE_PATH,
    RAW_TRANSCRIPTION_SCHEMA_VERSION,
)


HARMONY_ARTIFACT_RELATIVE_PATH = "harmony/harmonic-context.json"
HARMONY_ARTIFACT_SCHEMA_VERSION = 1

_PITCH_CLASS_NAMES = (
    "C",
    "C#",
    "D",
    "D#",
    "E",
    "F",
    "F#",
    "G",
    "G#",
    "A",
    "A#",
    "B",
)

_JOB_ID = re.compile(r"[a-f0-9]{32}")
_ATTEMPT_ARTIFACT = re.compile(r"harmony/harmonic-context\.([a-f0-9]{32})\.json")
_ID = re.compile(r"[a-z][a-z0-9_-]{0,95}")
_SLUG = re.compile(r"[a-z0-9][a-z0-9_-]{0,95}")
_VERSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}")
_URI = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://")
_WINDOWS_PATH = re.compile(r"(?i)\b[A-Z]:[\\/][^\s,;]+")
_UNIX_PATH = re.compile(
    r"(?i)(?<![A-Za-z0-9])/(?:home|users|tmp|var|etc|mnt|volumes|private|opt|usr)"
    r"(?:/[^\s,;]+)+"
)
_SECRET = re.compile(
    r"(?i)\b(?:token|password|secret|api[_-]?key|access[_-]?key|authorization|bearer)"
    r"\b\s*(?:=|:)\s*[^\s,;]+"
)

_MAX_ARTIFACT_BYTES = 24 * 1024 * 1024
_MAX_RAW_EVENTS = 100_000
_MAX_RAW_EVENT_WARNINGS = 128
_MAX_SEGMENTS = 20_000
_MAX_ALTERNATIVES = 16
_MAX_REFERENCES = 100_000
_MAX_WARNINGS = 128
_MAX_WARNING_LENGTH = 500
_MAX_TEXT_LENGTH = 512
_MAX_ALGORITHMS = 128
_MAX_VOCABULARY = 128
_ARTIFACT_SCOPE: ContextVar[str | None] = ContextVar(
    "harmony_artifact_scope",
    default=None,
)


class HarmonyArtifactError(RuntimeError):
    """Base error for canonical harmonic-context artifact operations."""


class HarmonyArtifactValidationError(HarmonyArtifactError, ValueError):
    """Raised when a schema-1 harmonic-context payload is invalid."""


class HarmonyArtifactUnavailableError(HarmonyArtifactError):
    """Raised when no canonical harmonic-context artifact is available."""


HarmonyProtectionState = tuple[str, str | None, str | None]
HarmonyProtectionStateReader = Callable[[], HarmonyProtectionState | None]
HarmonyCleanupAuthorizer = Callable[[], None]
HarmonyCleanupLease = Callable[[], AbstractContextManager[None]]
HarmonyDirectorySnapshot = tuple[int, int, int]
HarmonyFileSnapshot = tuple[int, int, int, int, int, int]
HarmonyFileIdentity = tuple[int, int, int]
HarmonyPublicationContext = tuple[
    int,
    Path,
    Path,
    HarmonyDirectorySnapshot,
]
_PUBLICATION_DIRECTORY: ContextVar[HarmonyPublicationContext | None] = ContextVar(
    "harmony_publication_directory",
    default=None,
)
_ANY_DESTINATION_IDENTITY = object()
_EXPECTED_REPLACE_DESTINATION: ContextVar[object] = ContextVar(
    "harmony_expected_replace_destination",
    default=_ANY_DESTINATION_IDENTITY,
)
_PUBLICATION_RECOVERY: ContextVar[bool] = ContextVar(
    "harmony_publication_recovery",
    default=False,
)


def harmony_attempt_artifact_file_name(attempt_id: str) -> str:
    """Return the isolated canonical relative artifact name for one attempt."""
    if not isinstance(attempt_id, str) or not _JOB_ID.fullmatch(attempt_id):
        raise HarmonyArtifactError("Harmony attempt identity is invalid.")
    return f"harmony/harmonic-context.{attempt_id}.json"


@contextmanager
def harmony_artifact_scope(artifact_file_name: str) -> Iterator[None]:
    """Make legacy artifact calls resolve to one isolated attempt target."""
    canonical = _canonical_artifact_file_name(artifact_file_name)
    token = _ARTIFACT_SCOPE.set(canonical)
    try:
        yield
    finally:
        _ARTIFACT_SCOPE.reset(token)


def build_harmony_artifact(
    result: HarmonyInferenceResult,
    *,
    harmony_version: str,
    created_at: str,
    transcription_version: str,
    analysis_version: str,
    interpretation_version: str | None = None,
) -> dict[str, Any]:
    """Build and validate a canonical artifact from detached inference evidence."""
    if not isinstance(result, HarmonyInferenceResult):
        raise HarmonyArtifactValidationError(
            "Harmony result must be a HarmonyInferenceResult."
        )

    payload: dict[str, Any] = {
        "schemaVersion": HARMONY_ARTIFACT_SCHEMA_VERSION,
        "harmonyVersion": harmony_version,
        "createdAt": created_at,
        "sourceTranscription": {
            "fileName": RAW_TRANSCRIPTION_RELATIVE_PATH,
            "schemaVersion": RAW_TRANSCRIPTION_SCHEMA_VERSION,
            "transcriptionVersion": transcription_version,
        },
        "sourceAnalysis": {
            "fileName": ANALYSIS_JSON_RELATIVE_PATH,
            "schemaVersion": 1,
            "analysisVersion": analysis_version,
        },
        "algorithms": {
            "harmonyInference": {"version": result.version},
        },
        "rawEvidence": copy.deepcopy(list(result.raw_evidence)),
        "segments": copy.deepcopy(list(result.segments)),
        "tonalContext": copy.deepcopy(result.tonal_context),
        "unresolvedEventIds": list(result.unresolved_event_ids),
        "warnings": list(result.warnings),
        "diagnostics": copy.deepcopy(result.diagnostics),
    }
    if interpretation_version is not None:
        payload["sourceInterpretation"] = {
            "fileName": INTERPRETATION_DRAFT_RELATIVE_PATH,
            "schemaVersion": INTERPRETATION_DRAFT_SCHEMA_VERSION,
            "draftVersion": interpretation_version,
        }
    return validate_harmony_artifact(payload)


def validate_harmony_artifact(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deterministic, internally consistent schema-1 artifact copy."""
    value = _mapping(payload, "harmonic-context artifact")
    required = {
        "schemaVersion",
        "harmonyVersion",
        "createdAt",
        "sourceTranscription",
        "sourceAnalysis",
        "algorithms",
        "rawEvidence",
        "segments",
        "tonalContext",
        "unresolvedEventIds",
        "warnings",
        "diagnostics",
    }
    optional = {"sourceInterpretation"}
    _keys(value, required, optional, "harmonic-context artifact")
    if _integer(value["schemaVersion"], "schemaVersion", 0, 2_147_483_647) != 1:
        raise HarmonyArtifactValidationError(
            "Unsupported harmonic-context schema version."
        )

    source_transcription = _source_transcription(value["sourceTranscription"])
    source_analysis = _source_analysis(value["sourceAnalysis"])
    algorithms = _algorithms(value["algorithms"])
    raw_evidence, raw_index, raw_order = _raw_evidence(value["rawEvidence"])
    segments = _segments(value["segments"], raw_index, raw_order)
    tonal_context = _tonal_context(value["tonalContext"])
    unresolved = _unresolved_ids(
        value["unresolvedEventIds"],
        raw_index,
        raw_order,
        segments,
    )
    diagnostics = _diagnostics(
        value["diagnostics"],
        raw_evidence=raw_evidence,
        segments=segments,
        unresolved=unresolved,
    )

    vocabulary = set(diagnostics["chordVocabulary"])
    for segment in segments:
        candidates = [segment["primaryCandidate"], *segment["alternatives"]]
        for candidate in candidates:
            if candidate is not None and candidate["quality"] not in vocabulary:
                raise HarmonyArtifactValidationError(
                    "Candidate quality is missing from diagnostics.chordVocabulary."
                )

    result: dict[str, Any] = {
        "schemaVersion": HARMONY_ARTIFACT_SCHEMA_VERSION,
        "harmonyVersion": _version(value["harmonyVersion"], "harmonyVersion"),
        "createdAt": _utc_timestamp(value["createdAt"], "createdAt"),
        "sourceTranscription": source_transcription,
        "sourceAnalysis": source_analysis,
    }
    if "sourceInterpretation" in value:
        result["sourceInterpretation"] = _source_interpretation(
            value["sourceInterpretation"]
        )
    result.update(
        algorithms=algorithms,
        rawEvidence=raw_evidence,
        segments=segments,
        tonalContext=tonal_context,
        unresolvedEventIds=unresolved,
        warnings=_warnings(value["warnings"], "warnings", _MAX_WARNINGS),
        diagnostics=diagnostics,
    )
    _encoded_payload(result)
    return result


def write_harmony_artifact(
    job_id: str,
    settings: Settings,
    payload: Mapping[str, Any],
    *,
    artifact_file_name: str | None = None,
) -> Path:
    """Validate and atomically publish one canonical harmonic-context artifact."""
    target = _resolve_artifact_file_name(artifact_file_name)
    validated = validate_harmony_artifact(payload)
    if target != HARMONY_ARTIFACT_RELATIVE_PATH and any(
        "warnings" not in item for item in validated["rawEvidence"]
    ):
        raise HarmonyArtifactValidationError(
            "Attempt-scoped raw evidence requires explicit warning lists."
        )
    encoded = _encoded_payload(validated)
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=True)
    assert directory is not None
    leaf = _artifact_leaf(target)
    destination = directory / leaf
    with _publication_directory_scope(directory, job_dir):
        return _publish_harmony_artifact(
            destination,
            directory,
            job_dir,
            encoded,
            immutable=target != HARMONY_ARTIFACT_RELATIVE_PATH,
        )


def _publish_harmony_artifact(
    destination: Path,
    directory: Path,
    job_dir: Path,
    encoded: bytes,
    *,
    immutable: bool,
) -> Path:
    directory_snapshot = _directory_snapshot(directory, job_dir)
    _validate_existing_destination(destination, directory)
    previous = _existing_artifact_bytes(destination, directory)
    previous_identity = (
        _published_file_identity(destination, directory)
        if previous is not None
        else None
    )
    if immutable and previous is not None:
        return _accept_existing_immutable_artifact(
            destination,
            directory,
            job_dir,
            directory_snapshot,
            encoded,
        )

    temporary = directory / f".{destination.name}.{uuid4().hex}.tmp"
    installed = False
    installed_identity: HarmonyFileIdentity | None = None
    try:
        temporary_identity = _write_exclusive_regular_file(
            temporary,
            encoded,
            directory,
        )
        _assert_publication_directory_current(
            directory,
            job_dir,
            directory_snapshot,
        )
        if immutable:
            try:
                _link_atomic(temporary, destination)
            except FileExistsError:
                return _accept_existing_immutable_artifact(
                    destination,
                    directory,
                    job_dir,
                    directory_snapshot,
                    encoded,
                )
        else:
            _replace_if_destination(
                temporary,
                destination,
                expected_destination_identity=previous_identity,
            )
        installed = True
        installed_identity = temporary_identity
        _assert_publication_directory_current(
            directory,
            job_dir,
            directory_snapshot,
        )
        if _published_file_identity(destination, directory) != installed_identity:
            raise HarmonyArtifactError(
                "Published harmonic context changed during publication."
            )
        _fsync_directory(directory)
    except (HarmonyArtifactError, OSError) as exc:
        if installed and not immutable and installed_identity is not None:
            try:
                _restore_publication_state(
                    destination,
                    directory,
                    job_dir,
                    directory_snapshot,
                    previous,
                    installed_identity=installed_identity,
                )
            except HarmonyArtifactError as recovery_exc:
                raise recovery_exc
        if isinstance(exc, HarmonyArtifactError):
            raise
        raise HarmonyArtifactError(
            "Harmonic context could not be published safely."
        ) from exc
    finally:
        _remove_temporary(temporary, directory, job_dir)
    _assert_publication_directory_current(
        directory,
        job_dir,
        directory_snapshot,
    )
    return destination.resolve(strict=True)


def _accept_existing_immutable_artifact(
    destination: Path,
    directory: Path,
    job_dir: Path,
    expected_directory_snapshot: HarmonyDirectorySnapshot,
    encoded: bytes,
) -> Path:
    _assert_publication_directory_current(
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    before_identity = _published_file_identity(destination, directory)
    if _existing_artifact_bytes(destination, directory) != encoded:
        raise HarmonyArtifactError(
            "An attempt-scoped harmonic context is already published with "
            "different content."
        )
    if _published_file_identity(destination, directory) != before_identity:
        raise HarmonyArtifactError(
            "Attempt-scoped harmonic context changed while being accepted."
        )
    _fsync_directory(directory)
    _assert_publication_directory_current(
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    if (
        _published_file_identity(destination, directory) != before_identity
        or _existing_artifact_bytes(destination, directory) != encoded
    ):
        raise HarmonyArtifactError(
            "Attempt-scoped harmonic context changed while being accepted."
        )
    return destination.resolve(strict=True)


def load_harmony_artifact(
    job_id: str,
    settings: Settings,
    *,
    artifact_file_name: str | None = None,
) -> dict[str, Any] | None:
    """Load and revalidate one published canonical artifact."""
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=False)
    if directory is None:
        return None
    target = _resolve_artifact_file_name(artifact_file_name)
    destination = directory / _artifact_leaf(target)
    try:
        destination.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HarmonyArtifactError(
            "Saved harmonic context is unavailable."
        ) from exc
    data = _read_stable_regular_file(destination, directory)
    try:
        payload = json.loads(
            data.decode("utf-8"),
            parse_constant=_reject_json_constant,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise HarmonyArtifactError(
            "Saved harmonic context is unreadable or corrupted."
        ) from exc
    try:
        validated = validate_harmony_artifact(payload)
    except HarmonyArtifactValidationError as exc:
        raise HarmonyArtifactError(
            "Saved harmonic context failed schema validation."
        ) from exc
    if target != HARMONY_ARTIFACT_RELATIVE_PATH and any(
        "warnings" not in item for item in validated["rawEvidence"]
    ):
        raise HarmonyArtifactError(
            "Saved attempt-scoped harmonic context failed schema validation."
        )
    return validated


def harmony_artifact_path(
    job_id: str,
    settings: Settings,
    *,
    artifact_file_name: str | None = None,
) -> Path:
    """Resolve one canonical JSON file only if it is stable through validation."""
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=False)
    if directory is None:
        raise HarmonyArtifactUnavailableError(
            "Published harmonic context is unavailable."
        )
    target = _resolve_artifact_file_name(artifact_file_name)
    path = directory / _artifact_leaf(target)
    before = _regular_file_snapshot(path, directory, unavailable=True)
    if target == HARMONY_ARTIFACT_RELATIVE_PATH and _ARTIFACT_SCOPE.get() is None:
        artifact = load_harmony_artifact(job_id, settings)
    else:
        artifact = load_harmony_artifact(
            job_id,
            settings,
            artifact_file_name=target,
        )
    if artifact is None:
        raise HarmonyArtifactUnavailableError(
            "Published harmonic context is unavailable."
        )
    final_directory = _artifact_directory(job_dir, create=False)
    if final_directory is None or final_directory != directory:
        raise HarmonyArtifactError(
            "Published harmonic context changed during validation."
        )
    after = _regular_file_snapshot(path, directory, unavailable=True)
    if before != after:
        raise HarmonyArtifactError(
            "Published harmonic context changed during validation."
        )
    return path.resolve(strict=True)


def remove_harmony_artifact(
    job_id: str,
    settings: Settings,
    *,
    artifact_file_name: str,
    cleanup_authorizer: HarmonyCleanupAuthorizer | None = None,
) -> None:
    """Remove one canonical artifact only through a pinned directory handle."""
    leaf_name = _artifact_leaf(artifact_file_name)
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=False)
    if directory is None or not _descriptor_relative_cleanup_supported():
        return
    descriptor, snapshot = _open_cleanup_directory(directory, job_dir)
    try:
        removed = _remove_published_artifact(
            directory / leaf_name,
            directory,
            directory_descriptor=descriptor,
            expected_directory_snapshot=snapshot,
            job_dir=job_dir,
            leaf_name=leaf_name,
            cleanup_authorizer=cleanup_authorizer,
        )
        if removed:
            _fsync_directory_descriptor(descriptor)
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
    finally:
        os.close(descriptor)


def reconcile_harmony_attempt_artifacts(
    job_id: str,
    settings: Settings,
    *,
    durable_artifact_file_name: str | None,
    active_attempt_id: str | None,
    protection_state_reader: HarmonyProtectionStateReader | None = None,
    cleanup_lease: HarmonyCleanupLease | None = None,
) -> int:
    """Remove orphan attempts using current DB protection and a pinned directory."""
    job_dir = _secure_job_root(job_id, settings)
    expected_state: HarmonyProtectionState = (
        "processing",
        durable_artifact_file_name,
        active_attempt_id,
    )
    if protection_state_reader is None:
        return 0

    def authorize_cleanup() -> None:
        _require_current_protection_state(protection_state_reader, expected_state)

    authorize_cleanup()
    protected: set[str] = {"harmonic-context.json"}
    if durable_artifact_file_name is not None:
        protected.add(
            _artifact_leaf(_canonical_artifact_file_name(durable_artifact_file_name))
        )
    if (
        active_attempt_id is None
        or cleanup_lease is None
        or not _descriptor_relative_cleanup_supported()
    ):
        return 0
    protected.add(
        _artifact_leaf(harmony_attempt_artifact_file_name(active_attempt_id))
    )

    directory = _artifact_directory(job_dir, create=False)
    if directory is None:
        return 0

    descriptor, snapshot = _open_cleanup_directory(directory, job_dir)
    try:
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
        try:
            names = sorted(os.listdir(descriptor))
        except (OSError, TypeError, ValueError) as exc:
            raise HarmonyArtifactError(
                "Harmony attempt artifacts could not be reconciled safely."
            ) from exc

        candidates: list[tuple[str, HarmonyFileSnapshot]] = []
        unsafe = False
        for name in names:
            if not isinstance(name, str) or name.startswith("."):
                continue
            if not name.startswith("harmonic-context."):
                continue
            
            # First, check if this is a regular file. This catches symlinks and
            # other non-regular files regardless of whether they're protected.
            try:
                file_snapshot = _relative_regular_file_snapshot(descriptor, name)
            except HarmonyArtifactError:
                unsafe = True
                continue
            if file_snapshot is None:
                continue
            if not stat.S_ISREG(file_snapshot[2]):
                unsafe = True
                continue
            
            # Protected files (active attempt, durable artifact) are validated
            # but not removed. They will be protected by state checks.
            if name in protected:
                continue
            
            # Only attempt artifacts need to match the regex
            relative = f"harmony/{name}"
            if not _ATTEMPT_ARTIFACT.fullmatch(relative):
                unsafe = True
                continue
            
            candidates.append((name, file_snapshot))
        
        if unsafe:
            raise HarmonyArtifactError(
                "Harmony attempt artifacts could not be reconciled safely."
            )

        removed = 0
        for name, file_snapshot in candidates:
            # Revalidate protection state before each removal
            authorize_cleanup()
            try:
                with cleanup_lease():
                    if _remove_published_artifact(
                        directory / name,
                        directory,
                        directory_descriptor=descriptor,
                        expected_directory_snapshot=snapshot,
                        job_dir=job_dir,
                        leaf_name=name,
                        expected_file_snapshot=file_snapshot,
                        cleanup_authorizer=authorize_cleanup,
                    ):
                        removed += 1
            except HarmonyArtifactError:
                raise
            except Exception as exc:
                raise HarmonyArtifactError(
                    "Harmony cleanup protection state could not be verified."
                ) from exc

        # Final state validation
        authorize_cleanup()
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
        if removed:
            _fsync_directory_descriptor(descriptor)
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
        return removed
    finally:
        os.close(descriptor)


def _source_transcription(value: Any) -> dict[str, Any]:
    source = _mapping(value, "sourceTranscription")
    _keys(
        source,
        {"fileName", "schemaVersion", "transcriptionVersion"},
        set(),
        "sourceTranscription",
    )
    if _integer(source["schemaVersion"], "sourceTranscription.schemaVersion", 0, 10) != 1:
        raise HarmonyArtifactValidationError(
            "sourceTranscription.schemaVersion must be 1."
        )
    return {
        "fileName": _canonical_relative_path(
            source["fileName"],
            RAW_TRANSCRIPTION_RELATIVE_PATH,
            "sourceTranscription.fileName",
        ),
        "schemaVersion": RAW_TRANSCRIPTION_SCHEMA_VERSION,
        "transcriptionVersion": _version(
            source["transcriptionVersion"],
            "sourceTranscription.transcriptionVersion",
        ),
    }


def _source_analysis(value: Any) -> dict[str, Any]:
    source = _mapping(value, "sourceAnalysis")
    _keys(
        source,
        {"fileName", "schemaVersion", "analysisVersion"},
        set(),
        "sourceAnalysis",
    )
    if _integer(source["schemaVersion"], "sourceAnalysis.schemaVersion", 0, 10) != 1:
        raise HarmonyArtifactValidationError(
            "sourceAnalysis.schemaVersion must be 1."
        )
    return {
        "fileName": _canonical_relative_path(
            source["fileName"],
            ANALYSIS_JSON_RELATIVE_PATH,
            "sourceAnalysis.fileName",
        ),
        "schemaVersion": 1,
        "analysisVersion": _version(
            source["analysisVersion"], "sourceAnalysis.analysisVersion"
        ),
    }


def _source_interpretation(value: Any) -> dict[str, Any]:
    source = _mapping(value, "sourceInterpretation")
    _keys(
        source,
        {"fileName", "schemaVersion", "draftVersion"},
        set(),
        "sourceInterpretation",
    )
    if _integer(source["schemaVersion"], "sourceInterpretation.schemaVersion", 0, 10) != 1:
        raise HarmonyArtifactValidationError(
            "sourceInterpretation.schemaVersion must be 1."
        )
    return {
        "fileName": _canonical_relative_path(
            source["fileName"],
            INTERPRETATION_DRAFT_RELATIVE_PATH,
            "sourceInterpretation.fileName",
        ),
        "schemaVersion": INTERPRETATION_DRAFT_SCHEMA_VERSION,
        "draftVersion": _version(
            source["draftVersion"], "sourceInterpretation.draftVersion"
        ),
    }


def _algorithms(value: Any) -> dict[str, Any]:
    algorithms = _mapping(value, "algorithms")
    if not algorithms or len(algorithms) > _MAX_ALGORITHMS:
        raise HarmonyArtifactValidationError("Invalid algorithm records.")
    if "harmonyInference" not in algorithms:
        raise HarmonyArtifactValidationError(
            "algorithms.harmonyInference is required."
        )
    output: dict[str, Any] = {}
    for name in sorted(algorithms):
        if not isinstance(name, str) or not _VERSION.fullmatch(name):
            raise HarmonyArtifactValidationError("Unsafe algorithm name.")
        record = _mapping(algorithms[name], f"algorithms.{name}")
        _keys(record, {"version"}, set(), f"algorithms.{name}")
        output[name] = {
            "version": _version(record["version"], f"algorithms.{name}.version")
        }
    return output


def _raw_evidence(
    value: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, int]]:
    records = _sequence(value, "rawEvidence")
    if len(records) > _MAX_RAW_EVENTS:
        raise HarmonyArtifactValidationError("Too many raw harmony evidence events.")
    output = [_raw_event(item, index) for index, item in enumerate(records)]
    output.sort(
        key=lambda item: (
            item["rawStartSeconds"],
            item["rawEndSeconds"],
            item["id"],
        )
    )
    index: dict[str, dict[str, Any]] = {}
    for item in output:
        if item["id"] in index:
            raise HarmonyArtifactValidationError("Duplicate raw evidence event ID.")
        index[item["id"]] = item
    order = {item["id"]: position for position, item in enumerate(output)}
    return output, index, order


def _raw_event(value: Any, index: int) -> dict[str, Any]:
    label = f"rawEvidence[{index}]"
    item = _mapping(value, label)
    required = {
        "id",
        "sourceKind",
        "rawStartSeconds",
        "rawEndSeconds",
        "midiNote",
        "midiPitch",
        "pitchClass",
        "pitchName",
        "confidence",
    }
    _keys(item, required, {"warnings"}, label)
    start = _number(item["rawStartSeconds"], f"{label}.rawStartSeconds", minimum=0)
    end = _number(item["rawEndSeconds"], f"{label}.rawEndSeconds", minimum=0)
    if end <= start:
        raise HarmonyArtifactValidationError(f"{label} has an invalid raw range.")
    midi_note = _integer(item["midiNote"], f"{label}.midiNote", 0, 127)
    midi_pitch = _number(item["midiPitch"], f"{label}.midiPitch")
    if not -0.75 <= midi_pitch - midi_note <= 0.75:
        raise HarmonyArtifactValidationError(
            f"{label}.midiPitch is inconsistent with midiNote."
        )
    pitch_class = _integer(item["pitchClass"], f"{label}.pitchClass", 0, 11)
    if pitch_class != midi_note % 12:
        raise HarmonyArtifactValidationError(
            f"{label}.pitchClass is inconsistent with midiNote."
        )
    pitch_name = item["pitchName"]
    if pitch_name != _PITCH_CLASS_NAMES[pitch_class]:
        raise HarmonyArtifactValidationError(
            f"{label}.pitchName is inconsistent with pitchClass."
        )
    output = {
        "id": _id(item["id"], f"{label}.id"),
        "sourceKind": _slug(item["sourceKind"], f"{label}.sourceKind"),
        "rawStartSeconds": start,
        "rawEndSeconds": end,
        "midiNote": midi_note,
        "midiPitch": midi_pitch,
        "pitchClass": pitch_class,
        "pitchName": pitch_name,
        "confidence": _confidence(item["confidence"], f"{label}.confidence"),
    }
    if "warnings" in item:
        output["warnings"] = _warnings(
            item["warnings"],
            f"{label}.warnings",
            _MAX_RAW_EVENT_WARNINGS,
        )
    return output


def _segments(
    value: Any,
    raw_index: Mapping[str, Mapping[str, Any]],
    raw_order: Mapping[str, int],
) -> list[dict[str, Any]]:
    records = _sequence(value, "segments")
    if len(records) > _MAX_SEGMENTS:
        raise HarmonyArtifactValidationError("Too many harmonic segments.")
    output = [
        _segment(item, index, raw_index, raw_order)
        for index, item in enumerate(records)
    ]
    output.sort(
        key=lambda item: (
            item["rawStartSeconds"],
            item["rawEndSeconds"],
            item["id"],
        )
    )
    ids = [item["id"] for item in output]
    if len(ids) != len(set(ids)):
        raise HarmonyArtifactValidationError("Duplicate harmonic segment ID.")
    return output


def _segment(
    value: Any,
    index: int,
    raw_index: Mapping[str, Mapping[str, Any]],
    raw_order: Mapping[str, int],
) -> dict[str, Any]:
    label = f"segments[{index}]"
    item = _mapping(value, label)
    required = {
        "id",
        "rawStartSeconds",
        "rawEndSeconds",
        "windowMode",
        "supportingEventIds",
        "sourceKinds",
        "partIds",
        "voiceIds",
        "unassignedContextEventIds",
        "observedPitchClasses",
        "primaryCandidate",
        "alternatives",
        "unresolved",
        "warnings",
    }
    optional = {"beatIndex"}
    _keys(item, required, optional, label)
    start = _number(item["rawStartSeconds"], f"{label}.rawStartSeconds", minimum=0)
    end = _number(item["rawEndSeconds"], f"{label}.rawEndSeconds", minimum=0)
    if end <= start:
        raise HarmonyArtifactValidationError(f"{label} has an invalid raw range.")
    mode = _slug(item["windowMode"], f"{label}.windowMode")
    if mode == "beat":
        if "beatIndex" not in item:
            raise HarmonyArtifactValidationError(
                f"{label}.beatIndex is required for beat windows."
            )
        beat_index: int | None = _integer(
            item["beatIndex"], f"{label}.beatIndex", 0, 2_147_483_647
        )
    else:
        if "beatIndex" in item:
            raise HarmonyArtifactValidationError(
                f"{label}.beatIndex is only valid for beat windows."
            )
        beat_index = None

    supporting = _event_references(
        item["supportingEventIds"],
        f"{label}.supportingEventIds",
        raw_index,
        raw_order,
        allow_empty=False,
    )
    for event_id in supporting:
        event = raw_index[event_id]
        overlap = min(end, event["rawEndSeconds"]) - max(
            start, event["rawStartSeconds"]
        )
        if overlap <= 0:
            raise HarmonyArtifactValidationError(
                f"{label} references raw evidence outside its time range."
            )
    derived_sources = sorted({raw_index[event_id]["sourceKind"] for event_id in supporting})
    sources = _slug_list(
        item["sourceKinds"], f"{label}.sourceKinds", _MAX_REFERENCES
    )
    if sources != derived_sources:
        raise HarmonyArtifactValidationError(
            f"{label}.sourceKinds does not match supporting raw evidence."
        )

    unassigned = _event_references(
        item["unassignedContextEventIds"],
        f"{label}.unassignedContextEventIds",
        raw_index,
        raw_order,
        allow_empty=True,
    )
    if not set(unassigned).issubset(supporting):
        raise HarmonyArtifactValidationError(
            f"{label}.unassignedContextEventIds must be supporting events."
        )

    observed = _observed_pitch_classes(
        item["observedPitchClasses"],
        f"{label}.observedPitchClasses",
    )
    observed_classes = {entry["pitchClass"] for entry in observed}
    expected_classes = {raw_index[event_id]["pitchClass"] for event_id in supporting}
    if observed_classes != expected_classes:
        raise HarmonyArtifactValidationError(
            f"{label}.observedPitchClasses does not match supporting raw evidence."
        )

    primary_value = item["primaryCandidate"]
    primary = (
        None
        if primary_value is None
        else _candidate(
            primary_value,
            f"{label}.primaryCandidate",
            raw_index,
            set(supporting),
        )
    )
    alternatives_value = _sequence(item["alternatives"], f"{label}.alternatives")
    if len(alternatives_value) > _MAX_ALTERNATIVES:
        raise HarmonyArtifactValidationError(
            f"{label}.alternatives contains too many candidates."
        )
    alternatives = [
        _candidate(
            candidate,
            f"{label}.alternatives[{candidate_index}]",
            raw_index,
            set(supporting),
        )
        for candidate_index, candidate in enumerate(alternatives_value)
    ]
    candidate_keys = [
        _candidate_identity(candidate)
        for candidate in ([primary] if primary is not None else []) + alternatives
    ]
    if len(candidate_keys) != len(set(candidate_keys)):
        raise HarmonyArtifactValidationError(
            f"{label} contains duplicate harmonic candidates."
        )

    unresolved = _boolean(item["unresolved"], f"{label}.unresolved")
    if unresolved and primary is not None:
        raise HarmonyArtifactValidationError(
            f"{label}.primaryCandidate must be null when unresolved."
        )
    if not unresolved and primary is None:
        raise HarmonyArtifactValidationError(
            f"{label}.primaryCandidate is required when resolved."
        )

    output: dict[str, Any] = {
        "id": _id(item["id"], f"{label}.id"),
        "rawStartSeconds": start,
        "rawEndSeconds": end,
        "windowMode": mode,
        "supportingEventIds": supporting,
        "sourceKinds": sources,
        "partIds": _id_list(item["partIds"], f"{label}.partIds", _MAX_REFERENCES),
        "voiceIds": _id_list(item["voiceIds"], f"{label}.voiceIds", _MAX_REFERENCES),
        "unassignedContextEventIds": unassigned,
        "observedPitchClasses": observed,
        "primaryCandidate": primary,
        "alternatives": alternatives,
        "unresolved": unresolved,
        "warnings": _warnings(item["warnings"], f"{label}.warnings", _MAX_WARNINGS),
    }
    if beat_index is not None:
        output["beatIndex"] = beat_index
    return output


def _observed_pitch_classes(value: Any, label: str) -> list[dict[str, Any]]:
    records = _sequence(value, label)
    if not records or len(records) > 12:
        raise HarmonyArtifactValidationError(f"{label} must contain 1 through 12 items.")
    output: list[dict[str, Any]] = []
    seen: set[int] = set()
    total_ratio = 0.0
    for index, raw in enumerate(records):
        item_label = f"{label}[{index}]"
        item = _mapping(raw, item_label)
        _keys(item, {"pitchClass", "pitchName", "weight", "weightRatio"}, set(), item_label)
        pitch_class = _integer(item["pitchClass"], f"{item_label}.pitchClass", 0, 11)
        if pitch_class in seen:
            raise HarmonyArtifactValidationError(f"{label} has duplicate pitch classes.")
        seen.add(pitch_class)
        if item["pitchName"] != _PITCH_CLASS_NAMES[pitch_class]:
            raise HarmonyArtifactValidationError(
                f"{item_label}.pitchName is inconsistent with pitchClass."
            )
        weight = _number(item["weight"], f"{item_label}.weight", minimum=0, exclusive_minimum=True)
        ratio = _confidence(item["weightRatio"], f"{item_label}.weightRatio")
        total_ratio += ratio
        output.append(
            {
                "pitchClass": pitch_class,
                "pitchName": item["pitchName"],
                "weight": weight,
                "weightRatio": ratio,
            }
        )
    if not math.isclose(total_ratio, 1.0, rel_tol=0, abs_tol=1e-6):
        raise HarmonyArtifactValidationError(f"{label}.weightRatio values must sum to 1.")
    output.sort(key=lambda item: item["pitchClass"])
    return output


def _candidate(
    value: Any,
    label: str,
    raw_index: Mapping[str, Mapping[str, Any]],
    supporting_ids: set[str],
) -> dict[str, Any]:
    item = _mapping(value, label)
    required = {
        "rootPitchClass",
        "root",
        "quality",
        "symbol",
        "pitchClasses",
        "score",
        "templateCoverage",
        "chordToneWeightRatio",
        "nonChordToneRatio",
        "rootWeightRatio",
        "tonalContextSupport",
        "evidenceEventIds",
        "confidence",
    }
    optional = {"inversionCandidate"}
    _keys(item, required, optional, label)
    root_pc = _integer(item["rootPitchClass"], f"{label}.rootPitchClass", 0, 11)
    if item["root"] != _PITCH_CLASS_NAMES[root_pc]:
        raise HarmonyArtifactValidationError(
            f"{label}.root is inconsistent with rootPitchClass."
        )
    pitch_classes_value = _sequence(item["pitchClasses"], f"{label}.pitchClasses")
    pitch_classes = [
        _integer(value, f"{label}.pitchClasses[{index}]", 0, 11)
        for index, value in enumerate(pitch_classes_value)
    ]
    if not pitch_classes or pitch_classes[0] != root_pc or len(pitch_classes) != len(set(pitch_classes)):
        raise HarmonyArtifactValidationError(
            f"{label}.pitchClasses must be unique and begin with the root."
        )
    evidence = _event_references(
        item["evidenceEventIds"],
        f"{label}.evidenceEventIds",
        raw_index,
        {event_id: index for index, event_id in enumerate(sorted(raw_index))},
        allow_empty=False,
    )
    if not set(evidence).issubset(supporting_ids):
        raise HarmonyArtifactValidationError(
            f"{label}.evidenceEventIds must be supporting events."
        )
    if any(raw_index[event_id]["pitchClass"] not in pitch_classes for event_id in evidence):
        raise HarmonyArtifactValidationError(
            f"{label}.evidenceEventIds contain non-chord pitch evidence."
        )

    output: dict[str, Any] = {
        "rootPitchClass": root_pc,
        "root": item["root"],
        "quality": _slug(item["quality"], f"{label}.quality"),
        "symbol": _safe_text(item["symbol"], f"{label}.symbol", _MAX_TEXT_LENGTH),
        "pitchClasses": pitch_classes,
        "score": _confidence(item["score"], f"{label}.score"),
        "templateCoverage": _confidence(item["templateCoverage"], f"{label}.templateCoverage"),
        "chordToneWeightRatio": _confidence(item["chordToneWeightRatio"], f"{label}.chordToneWeightRatio"),
        "nonChordToneRatio": _confidence(item["nonChordToneRatio"], f"{label}.nonChordToneRatio"),
        "rootWeightRatio": _confidence(item["rootWeightRatio"], f"{label}.rootWeightRatio"),
        "tonalContextSupport": _confidence(item["tonalContextSupport"], f"{label}.tonalContextSupport"),
        "evidenceEventIds": evidence,
        "confidence": _confidence(item["confidence"], f"{label}.confidence"),
    }
    if "inversionCandidate" in item:
        output["inversionCandidate"] = _inversion(
            item["inversionCandidate"],
            f"{label}.inversionCandidate",
            raw_index,
            supporting_ids,
            set(pitch_classes),
        )
    return output


def _inversion(
    value: Any,
    label: str,
    raw_index: Mapping[str, Mapping[str, Any]],
    supporting_ids: set[str],
    chord_pitch_classes: set[int],
) -> dict[str, Any]:
    item = _mapping(value, label)
    _keys(
        item,
        {"bassPitchClass", "bassPitchName", "position", "confidence", "sourceEventIds"},
        set(),
        label,
    )
    bass_pc = _integer(item["bassPitchClass"], f"{label}.bassPitchClass", 0, 11)
    if bass_pc not in chord_pitch_classes or item["bassPitchName"] != _PITCH_CLASS_NAMES[bass_pc]:
        raise HarmonyArtifactValidationError(
            f"{label} is inconsistent with the chord pitch classes."
        )
    sources = _event_references(
        item["sourceEventIds"],
        f"{label}.sourceEventIds",
        raw_index,
        {event_id: index for index, event_id in enumerate(sorted(raw_index))},
        allow_empty=False,
    )
    if not set(sources).issubset(supporting_ids):
        raise HarmonyArtifactValidationError(
            f"{label}.sourceEventIds must be supporting events."
        )
    for event_id in sources:
        event = raw_index[event_id]
        if event["sourceKind"] != "bass" or event["pitchClass"] != bass_pc:
            raise HarmonyArtifactValidationError(
                f"{label} requires explicit matching bass-source evidence."
            )
    return {
        "bassPitchClass": bass_pc,
        "bassPitchName": item["bassPitchName"],
        "position": _slug(item["position"], f"{label}.position"),
        "confidence": _confidence(item["confidence"], f"{label}.confidence"),
        "sourceEventIds": sources,
    }


def _candidate_identity(candidate: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        candidate["rootPitchClass"],
        candidate["quality"],
        tuple(candidate["pitchClasses"]),
    )


def _tonal_context(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    item = _mapping(value, "tonalContext")
    _keys(
        item,
        {"tonalCenter", "collection", "displayName", "confidence", "advisoryOnly"},
        set(),
        "tonalContext",
    )
    center = item["tonalCenter"]
    if center not in _PITCH_CLASS_NAMES:
        raise HarmonyArtifactValidationError("tonalContext.tonalCenter is unsupported.")
    if _boolean(item["advisoryOnly"], "tonalContext.advisoryOnly") is not True:
        raise HarmonyArtifactValidationError("tonalContext must remain advisory only.")
    return {
        "tonalCenter": center,
        "collection": _slug(item["collection"], "tonalContext.collection"),
        "displayName": _safe_text(item["displayName"], "tonalContext.displayName", _MAX_TEXT_LENGTH),
        "confidence": _confidence(item["confidence"], "tonalContext.confidence"),
        "advisoryOnly": True,
    }


def _unresolved_ids(
    value: Any,
    raw_index: Mapping[str, Mapping[str, Any]],
    raw_order: Mapping[str, int],
    segments: Sequence[Mapping[str, Any]],
) -> list[str]:
    provided = _event_references(
        value,
        "unresolvedEventIds",
        raw_index,
        raw_order,
        allow_empty=True,
    )
    resolved_ids: set[str] = set()
    for segment in segments:
        if not segment["unresolved"] and segment["primaryCandidate"] is not None:
            resolved_ids.update(segment["supportingEventIds"])
    expected = [
        event_id
        for event_id, _ in sorted(raw_order.items(), key=lambda item: item[1])
        if event_id not in resolved_ids
    ]
    if provided != expected:
        raise HarmonyArtifactValidationError(
            "unresolvedEventIds does not match resolved segment evidence."
        )
    return provided


def _diagnostics(
    value: Any,
    *,
    raw_evidence: Sequence[Mapping[str, Any]],
    segments: Sequence[Mapping[str, Any]],
    unresolved: Sequence[str],
) -> dict[str, Any]:
    item = _mapping(value, "diagnostics")
    required = {
        "eventCount",
        "segmentCount",
        "resolvedSegmentCount",
        "unresolvedSegmentCount",
        "unresolvedEventCount",
        "sourceKinds",
        "windowingMode",
        "fallbackWindowSeconds",
        "rawTimingAuthoritative",
        "fractionalPitchPreserved",
        "rawEvidenceIncluded",
        "tonalContextAdvisoryOnly",
        "bassSourceRequiredForInversion",
        "chordVocabulary",
        "romanNumeralsGenerated",
        "guitarVoicingsGenerated",
        "notationGenerated",
    }
    _keys(item, required, set(), "diagnostics")
    resolved_count = sum(not segment["unresolved"] for segment in segments)
    unresolved_count = len(segments) - resolved_count
    expected_counts = {
        "eventCount": len(raw_evidence),
        "segmentCount": len(segments),
        "resolvedSegmentCount": resolved_count,
        "unresolvedSegmentCount": unresolved_count,
        "unresolvedEventCount": len(unresolved),
    }
    for name, expected in expected_counts.items():
        actual = _integer(item[name], f"diagnostics.{name}", 0, 2_147_483_647)
        if actual != expected:
            raise HarmonyArtifactValidationError(
                f"diagnostics.{name} does not match artifact truth."
            )
    sources = _slug_list(item["sourceKinds"], "diagnostics.sourceKinds", _MAX_REFERENCES)
    expected_sources = sorted({event["sourceKind"] for event in raw_evidence})
    if sources != expected_sources:
        raise HarmonyArtifactValidationError(
            "diagnostics.sourceKinds does not match raw evidence."
        )
    mode = item["windowingMode"]
    if mode not in {"beat_relative", "absolute_time"}:
        raise HarmonyArtifactValidationError("diagnostics.windowingMode is invalid.")
    fallback = item["fallbackWindowSeconds"]
    if mode == "absolute_time":
        fallback_value: float | None = _number(
            fallback,
            "diagnostics.fallbackWindowSeconds",
            minimum=0,
            exclusive_minimum=True,
        )
    else:
        if fallback is not None:
            raise HarmonyArtifactValidationError(
                "Beat-relative diagnostics must not declare a fallback window."
            )
        fallback_value = None
    vocabulary = _slug_list(
        item["chordVocabulary"],
        "diagnostics.chordVocabulary",
        _MAX_VOCABULARY,
    )
    if not vocabulary:
        raise HarmonyArtifactValidationError(
            "diagnostics.chordVocabulary must not be empty."
        )
    true_flags = (
        "rawTimingAuthoritative",
        "fractionalPitchPreserved",
        "rawEvidenceIncluded",
        "tonalContextAdvisoryOnly",
        "bassSourceRequiredForInversion",
    )
    false_flags = (
        "romanNumeralsGenerated",
        "guitarVoicingsGenerated",
        "notationGenerated",
    )
    for name in true_flags:
        if _boolean(item[name], f"diagnostics.{name}") is not True:
            raise HarmonyArtifactValidationError(f"diagnostics.{name} must remain true.")
    for name in false_flags:
        if _boolean(item[name], f"diagnostics.{name}") is not False:
            raise HarmonyArtifactValidationError(f"diagnostics.{name} must remain false.")
    return {
        **expected_counts,
        "sourceKinds": sources,
        "windowingMode": mode,
        "fallbackWindowSeconds": fallback_value,
        "rawTimingAuthoritative": True,
        "fractionalPitchPreserved": True,
        "rawEvidenceIncluded": True,
        "tonalContextAdvisoryOnly": True,
        "bassSourceRequiredForInversion": True,
        "chordVocabulary": vocabulary,
        "romanNumeralsGenerated": False,
        "guitarVoicingsGenerated": False,
        "notationGenerated": False,
    }


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HarmonyArtifactValidationError(f"{label} must be an object.")
    return value


def _sequence(value: Any, label: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise HarmonyArtifactValidationError(f"{label} must be an array.")
    return value


def _keys(
    value: Mapping[str, Any],
    required: set[str],
    optional: set[str],
    label: str,
) -> None:
    if not all(isinstance(key, str) for key in value):
        raise HarmonyArtifactValidationError(f"{label} contains a non-string field.")
    keys = set(value)
    missing = required - keys
    unknown = keys - required - optional
    if missing:
        raise HarmonyArtifactValidationError(
            f"{label} is missing required fields: {', '.join(sorted(missing))}."
        )
    if unknown:
        raise HarmonyArtifactValidationError(
            f"{label} contains unsupported fields: {', '.join(sorted(unknown))}."
        )


def _id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise HarmonyArtifactValidationError(f"{label} is unsafe.")
    return value


def _slug(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SLUG.fullmatch(value):
        raise HarmonyArtifactValidationError(f"{label} is unsafe.")
    return value


def _version(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _VERSION.fullmatch(value):
        raise HarmonyArtifactValidationError(f"{label} is invalid.")
    return value


def _safe_text(value: Any, label: str, maximum: int) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > maximum:
        raise HarmonyArtifactValidationError(f"{label} is invalid.")
    if (
        any(ord(character) < 32 or ord(character) == 127 for character in value)
        or "<" in value
        or ">" in value
        or "\x00" in value
        or _URI.search(value)
        or _WINDOWS_PATH.search(value)
        or _UNIX_PATH.search(value)
        or _SECRET.search(value)
    ):
        raise HarmonyArtifactValidationError(f"{label} contains unsafe text.")
    return value


def _warnings(value: Any, label: str, maximum_count: int) -> list[str]:
    records = _sequence(value, label)
    if len(records) > maximum_count:
        raise HarmonyArtifactValidationError(f"{label} contains too many warnings.")
    return [
        _safe_text(item, f"{label}[{index}]", _MAX_WARNING_LENGTH)
        for index, item in enumerate(records)
    ]


def _number(
    value: Any,
    label: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    exclusive_minimum: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise HarmonyArtifactValidationError(f"{label} must be a number.")
    number = float(value)
    if not math.isfinite(number):
        raise HarmonyArtifactValidationError(f"{label} must be finite.")
    if minimum is not None:
        if exclusive_minimum and number <= minimum:
            raise HarmonyArtifactValidationError(f"{label} is below its minimum.")
        if not exclusive_minimum and number < minimum:
            raise HarmonyArtifactValidationError(f"{label} is below its minimum.")
    if maximum is not None and number > maximum:
        raise HarmonyArtifactValidationError(f"{label} is above its maximum.")
    return number


def _confidence(value: Any, label: str) -> float:
    return _number(value, label, minimum=0, maximum=1)


def _integer(value: Any, label: str, minimum: int, maximum: int) -> int:
    if type(value) is not int or not minimum <= value <= maximum:
        raise HarmonyArtifactValidationError(
            f"{label} must be an integer from {minimum} through {maximum}."
        )
    return value


def _boolean(value: Any, label: str) -> bool:
    if type(value) is not bool:
        raise HarmonyArtifactValidationError(f"{label} must be true or false.")
    return value


def _id_list(value: Any, label: str, maximum: int) -> list[str]:
    records = _sequence(value, label)
    if len(records) > maximum:
        raise HarmonyArtifactValidationError(f"{label} contains too many IDs.")
    output = [_id(item, f"{label}[{index}]") for index, item in enumerate(records)]
    if len(output) != len(set(output)):
        raise HarmonyArtifactValidationError(f"{label} contains duplicate IDs.")
    return sorted(output)


def _slug_list(value: Any, label: str, maximum: int) -> list[str]:
    records = _sequence(value, label)
    if len(records) > maximum:
        raise HarmonyArtifactValidationError(f"{label} contains too many values.")
    output = [_slug(item, f"{label}[{index}]") for index, item in enumerate(records)]
    if len(output) != len(set(output)):
        raise HarmonyArtifactValidationError(f"{label} contains duplicate values.")
    return sorted(output)


def _event_references(
    value: Any,
    label: str,
    raw_index: Mapping[str, Mapping[str, Any]],
    raw_order: Mapping[str, int],
    *,
    allow_empty: bool,
) -> list[str]:
    records = _sequence(value, label)
    if len(records) > _MAX_REFERENCES:
        raise HarmonyArtifactValidationError(f"{label} contains too many event references.")
    output = [_id(item, f"{label}[{index}]") for index, item in enumerate(records)]
    if not allow_empty and not output:
        raise HarmonyArtifactValidationError(f"{label} must not be empty.")
    if len(output) != len(set(output)):
        raise HarmonyArtifactValidationError(f"{label} contains duplicate event references.")
    unknown = set(output) - set(raw_index)
    if unknown:
        raise HarmonyArtifactValidationError(f"{label} references unknown raw evidence.")
    return sorted(output, key=lambda event_id: raw_order[event_id])


def _canonical_relative_path(value: Any, expected: str, label: str) -> str:
    if not isinstance(value, str) or value != expected:
        raise HarmonyArtifactValidationError(f"{label} is not canonical.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or ".." in parsed.parts or "." in parsed.parts:
        raise HarmonyArtifactValidationError(f"{label} is not canonical.")
    return value


def _canonical_artifact_file_name(value: Any) -> str:
    leaf = _artifact_leaf(value)
    return f"harmony/{leaf}"


def _resolve_artifact_file_name(value: str | None) -> str:
    if value is not None:
        return _canonical_artifact_file_name(value)
    scoped = _ARTIFACT_SCOPE.get()
    if scoped is not None:
        return scoped
    return HARMONY_ARTIFACT_RELATIVE_PATH


def _artifact_leaf(value: Any) -> str:
    if not isinstance(value, str):
        raise HarmonyArtifactError("Harmony artifact file name is invalid.")
    if value == HARMONY_ARTIFACT_RELATIVE_PATH:
        return "harmonic-context.json"
    if not _ATTEMPT_ARTIFACT.fullmatch(value):
        raise HarmonyArtifactError("Harmony artifact file name is invalid.")
    parsed = PurePosixPath(value)
    if parsed.is_absolute() or parsed.parts != ("harmony", parsed.name):
        raise HarmonyArtifactError("Harmony artifact file name is invalid.")
    return parsed.name


def _utc_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 64:
        raise HarmonyArtifactValidationError(f"{label} must be a UTC timestamp.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise HarmonyArtifactValidationError(f"{label} must be a UTC timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() != timezone.utc.utcoffset(parsed):
        raise HarmonyArtifactValidationError(f"{label} must be a UTC timestamp.")
    return parsed.astimezone(timezone.utc).isoformat()


def _encoded_payload(value: Mapping[str, Any]) -> bytes:
    try:
        data = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError, UnicodeError) as exc:
        raise HarmonyArtifactValidationError(
            "Harmonic context could not be serialized safely."
        ) from exc
    if len(data) > _MAX_ARTIFACT_BYTES:
        raise HarmonyArtifactValidationError("Harmonic context artifact is too large.")
    return data


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"Invalid JSON constant: {value}")


def _validate_job_id(job_id: str) -> None:
    if not isinstance(job_id, str) or not _JOB_ID.fullmatch(job_id):
        raise HarmonyArtifactError("The harmonic-context artifact request is invalid.")


def _secure_job_root(job_id: str, settings: Settings) -> Path:
    _validate_job_id(job_id)
    try:
        resolved_job = secure_job_dir(settings, job_id)
        exports_root = settings.exports_dir.resolve(strict=True)
        lexical_job = exports_root / job_id
        info = lexical_job.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HarmonyArtifactError("Harmonic-context job directory is unsafe.")
        resolved = lexical_job.resolve(strict=True)
        if resolved != resolved_job.resolve(strict=True) or exports_root not in resolved.parents:
            raise HarmonyArtifactError("Harmonic-context job directory is unsafe.")
        return resolved
    except HarmonyArtifactError:
        raise
    except (MediaProcessingError, OSError, RuntimeError) as exc:
        raise HarmonyArtifactError("Harmonic-context job directory is unsafe.") from exc


def _artifact_directory(job_dir: Path, *, create: bool) -> Path | None:
    directory = job_dir / "harmony"
    try:
        info = directory.lstat()
    except FileNotFoundError:
        if not create:
            return None
        try:
            directory.mkdir(mode=0o700)
        except FileExistsError:
            pass
        except OSError as exc:
            raise HarmonyArtifactError(
                "Harmony artifact directory could not be created safely."
            ) from exc
        try:
            info = directory.lstat()
        except OSError as exc:
            raise HarmonyArtifactError(
                "Harmony artifact directory could not be inspected safely."
            ) from exc
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be inspected safely."
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HarmonyArtifactError("Harmony artifact directory is unsafe.")
    root = job_dir.resolve(strict=True)
    try:
        resolved = directory.resolve(strict=True)
    except OSError as exc:
        raise HarmonyArtifactError("Harmony artifact directory is unsafe.") from exc
    if root not in resolved.parents:
        raise HarmonyArtifactError("Harmony artifact directory is unsafe.")
    return directory


def _directory_snapshot(directory: Path, job_dir: Path) -> HarmonyDirectorySnapshot:
    try:
        info = directory.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise HarmonyArtifactError("Harmony artifact directory is unsafe.")
        if job_dir.resolve(strict=True) not in directory.resolve(strict=True).parents:
            raise HarmonyArtifactError("Harmony artifact directory is unsafe.")
        return info.st_dev, info.st_ino, info.st_mode
    except HarmonyArtifactError:
        raise
    except OSError as exc:
        raise HarmonyArtifactError("Harmony artifact directory is unsafe.") from exc


def _descriptor_relative_cleanup_supported() -> bool:
    """Return whether cleanup can stay confined to a pinned directory handle."""
    return (
        os.name == "posix"
        and bool(getattr(os, "O_DIRECTORY", 0))
        and bool(getattr(os, "O_NOFOLLOW", 0))
        and os.listdir in getattr(os, "supports_fd", set())
        and os.open in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_dir_fd", set())
        and os.stat in getattr(os, "supports_follow_symlinks", set())
        and os.unlink in getattr(os, "supports_dir_fd", set())
    )


def _descriptor_relative_publication_supported() -> bool:
    """Return whether publication can stay confined to a pinned directory.

    ``os.replace`` is not registered in ``os.supports_dir_fd`` by CPython even
    though it accepts ``src_dir_fd``/``dst_dir_fd``: it is defined as
    ``os.replace = os.rename`` in ``Lib/os.py`` and both share
    ``internal_rename()``, so the renameat-based dir_fd capability is gated by
    ``HAVE_RENAMEAT`` and registered on ``os.rename`` only. Check the shared
    primitive instead of the unregistered alias.
    """
    return (
        _descriptor_relative_cleanup_supported()
        and os.rename in getattr(os, "supports_dir_fd", set())
        and os.link in getattr(os, "supports_dir_fd", set())
        and os.link in getattr(os, "supports_follow_symlinks", set())
    )


@contextmanager
def _publication_directory_scope(
    directory: Path,
    job_dir: Path,
) -> Iterator[HarmonyPublicationContext | None]:
    """Pin a publication directory when the platform exposes safe primitives."""
    if not _descriptor_relative_publication_supported():
        if os.name == "posix":
            raise HarmonyArtifactError(
                "Harmony publication requires descriptor-relative filesystem support."
            )
        yield None
        return
    descriptor, snapshot = _open_cleanup_directory(directory, job_dir)
    context: HarmonyPublicationContext = (
        descriptor,
        directory,
        job_dir,
        snapshot,
    )
    token = _PUBLICATION_DIRECTORY.set(context)
    try:
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
        yield context
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            snapshot,
        )
    finally:
        _PUBLICATION_DIRECTORY.reset(token)
        os.close(descriptor)


def _active_publication_directory(
    directory: Path,
) -> HarmonyPublicationContext | None:
    context = _PUBLICATION_DIRECTORY.get()
    if context is None:
        return None
    descriptor, expected_directory, job_dir, snapshot = context
    if expected_directory != directory:
        raise HarmonyArtifactError("Harmony publication directory is inconsistent.")
    _assert_pinned_publication_directory(descriptor, snapshot)
    if not _PUBLICATION_RECOVERY.get():
        _assert_cleanup_directory_current(
            descriptor,
            expected_directory,
            job_dir,
            snapshot,
        )
    return context


def _assert_pinned_publication_directory(
    descriptor: int,
    expected: HarmonyDirectorySnapshot,
) -> None:
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be verified safely."
        ) from exc
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_ino <= 0
        or (opened.st_dev, opened.st_ino, opened.st_mode) != expected
    ):
        raise HarmonyArtifactError(
            "Harmony artifact directory changed during publication."
        )


def _assert_publication_directory_current(
    directory: Path,
    job_dir: Path,
    expected: HarmonyDirectorySnapshot,
) -> None:
    context = _active_publication_directory(directory)
    if context is not None:
        if context[3] != expected:
            raise HarmonyArtifactError(
                "Harmony artifact directory changed during publication."
            )
        return
    if _directory_snapshot(directory, job_dir) != expected:
        raise HarmonyArtifactError(
            "Harmony artifact directory changed during publication."
        )


def _published_file_identity(
    path: Path,
    directory: Path,
) -> HarmonyFileIdentity:
    context = _active_publication_directory(directory)
    if context is not None:
        if path.parent != directory:
            raise HarmonyArtifactError("Harmonic context file is unsafe.")
        snapshot = _relative_regular_file_snapshot(context[0], path.name)
        if snapshot is None:
            raise HarmonyArtifactError("Harmonic context file is unsafe.")
        return _harmony_file_identity(snapshot)
    info = _require_regular_file(path, directory)
    return info.st_dev, info.st_ino, stat.S_IFMT(info.st_mode)


def _current_published_file_identity(
    path: Path,
    directory: Path,
) -> HarmonyFileIdentity | None:
    context = _active_publication_directory(directory)
    if context is not None:
        if path.parent != directory:
            raise HarmonyArtifactError("Harmonic context file is unsafe.")
        snapshot = _relative_regular_file_snapshot(context[0], path.name)
        return None if snapshot is None else _harmony_file_identity(snapshot)
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HarmonyArtifactError("Harmonic context file is unsafe.") from exc
    return _published_file_identity(path, directory)


def _open_cleanup_directory(
    directory: Path,
    job_dir: Path,
) -> tuple[int, HarmonyDirectorySnapshot]:
    if directory.parent != job_dir or directory.name != "harmony":
        raise HarmonyArtifactError("Harmony artifact directory is unsafe.")
    expected = _directory_snapshot(directory, job_dir)
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    flags |= getattr(os, "O_CLOEXEC", 0)
    try:
        job_info = job_dir.lstat()
        if stat.S_ISLNK(job_info.st_mode) or not stat.S_ISDIR(job_info.st_mode):
            raise HarmonyArtifactError("Harmonic-context job directory is unsafe.")
        expected_job = (job_info.st_dev, job_info.st_ino, job_info.st_mode)
        job_descriptor = os.open(job_dir, flags)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmonic-context job directory could not be pinned safely."
        ) from exc
    descriptor: int | None = None
    try:
        try:
            opened_job = os.fstat(job_descriptor)
            if (
                not stat.S_ISDIR(opened_job.st_mode)
                or opened_job.st_ino <= 0
                or (opened_job.st_dev, opened_job.st_ino, opened_job.st_mode)
                != expected_job
            ):
                raise HarmonyArtifactError(
                    "Harmonic-context job directory changed during cleanup."
                )
            descriptor = os.open("harmony", flags, dir_fd=job_descriptor)
        except HarmonyArtifactError:
            raise
        except OSError as exc:
            raise HarmonyArtifactError(
                "Harmony artifact directory could not be pinned safely."
            ) from exc
    finally:
        os.close(job_descriptor)
    assert descriptor is not None
    try:
        _assert_cleanup_directory_current(
            descriptor,
            directory,
            job_dir,
            expected,
        )
    except BaseException:
        os.close(descriptor)
        raise
    return descriptor, expected


def _assert_cleanup_directory_current(
    descriptor: int,
    directory: Path,
    job_dir: Path,
    expected: HarmonyDirectorySnapshot,
) -> None:
    try:
        opened = os.fstat(descriptor)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be verified safely."
        ) from exc
    opened_snapshot = (opened.st_dev, opened.st_ino, opened.st_mode)
    if (
        not stat.S_ISDIR(opened.st_mode)
        or opened.st_ino <= 0
        or opened_snapshot != expected
        or _directory_snapshot(directory, job_dir) != expected
    ):
        raise HarmonyArtifactError(
            "Harmony artifact directory changed during cleanup."
        )


def _assert_removal_directory_current(
    descriptor: int,
    directory: Path,
    job_dir: Path,
    expected: HarmonyDirectorySnapshot,
) -> None:
    """Permit rollback only against the exact pinned publication directory."""
    context = _PUBLICATION_DIRECTORY.get()
    if (
        _PUBLICATION_RECOVERY.get()
        and context is not None
        and context == (descriptor, directory, job_dir, expected)
    ):
        _assert_pinned_publication_directory(descriptor, expected)
        return
    _assert_cleanup_directory_current(
        descriptor,
        directory,
        job_dir,
        expected,
    )


def _relative_regular_file_snapshot(
    directory_descriptor: int,
    leaf_name: str,
) -> HarmonyFileSnapshot | None:
    if not isinstance(leaf_name, str) or Path(leaf_name).name != leaf_name:
        raise HarmonyArtifactError("Harmony cleanup target is invalid.")
    try:
        info = os.stat(
            leaf_name,
            dir_fd=directory_descriptor,
            follow_symlinks=False,
        )
    except FileNotFoundError:
        return None
    except (NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise HarmonyArtifactError(
            "Harmony cleanup target could not be inspected safely."
        ) from exc
    if not stat.S_ISREG(info.st_mode) or info.st_ino <= 0:
        raise HarmonyArtifactError("Harmony cleanup target is not a regular file.")
    return _harmony_file_snapshot(info)


def _harmony_file_snapshot(info: os.stat_result) -> HarmonyFileSnapshot:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def _harmony_file_identity(snapshot: HarmonyFileSnapshot) -> HarmonyFileIdentity:
    return snapshot[0], snapshot[1], stat.S_IFMT(snapshot[2])


def _harmony_content_snapshot(
    snapshot: HarmonyFileSnapshot,
) -> tuple[int, int, int, int, int]:
    return (
        snapshot[0],
        snapshot[1],
        stat.S_IFMT(snapshot[2]),
        snapshot[3],
        snapshot[4],
    )


def _read_stable_regular_file_at(
    directory_descriptor: int,
    leaf_name: str,
) -> bytes:
    before = _relative_regular_file_snapshot(directory_descriptor, leaf_name)
    if before is None:
        raise HarmonyArtifactError("Saved harmonic context is unavailable.")
    if before[3] > _MAX_ARTIFACT_BYTES:
        raise HarmonyArtifactError("Saved harmonic context is too large.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(leaf_name, flags, dir_fd=directory_descriptor)
    except (OSError, TypeError, ValueError) as exc:
        raise HarmonyArtifactError("Saved harmonic context is unavailable.") from exc
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or _harmony_content_snapshot(_harmony_file_snapshot(opened))
            != _harmony_content_snapshot(before)
        ):
            raise HarmonyArtifactError(
                "Saved harmonic context changed during validation."
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total),
            )
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES:
                raise HarmonyArtifactError("Saved harmonic context is too large.")
        if (
            _harmony_content_snapshot(_harmony_file_snapshot(os.fstat(descriptor)))
            != _harmony_content_snapshot(before)
        ):
            raise HarmonyArtifactError(
                "Saved harmonic context changed during validation."
            )
    except HarmonyArtifactError:
        raise
    except OSError as exc:
        raise HarmonyArtifactError("Saved harmonic context is unavailable.") from exc
    finally:
        os.close(descriptor)
    after = _relative_regular_file_snapshot(directory_descriptor, leaf_name)
    if (
        after is None
        or _harmony_content_snapshot(after) != _harmony_content_snapshot(before)
    ):
        raise HarmonyArtifactError(
            "Saved harmonic context changed during validation."
        )
    return b"".join(chunks)


def _require_current_protection_state(
    reader: HarmonyProtectionStateReader,
    expected: HarmonyProtectionState,
) -> None:
    try:
        current = reader()
    except Exception as exc:
        raise HarmonyArtifactError(
            "Harmony cleanup protection state could not be verified."
        ) from exc
    if current != expected:
        raise HarmonyArtifactError(
            "Harmony cleanup protection state changed during reconciliation."
        )


def _validate_existing_destination(path: Path, directory: Path) -> None:
    context = _active_publication_directory(directory)
    if context is not None:
        if path.parent != directory:
            raise HarmonyArtifactError("Existing harmonic context is unsafe.")
        _relative_regular_file_snapshot(context[0], path.name)
        return
    try:
        path.lstat()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise HarmonyArtifactError("Existing harmonic context is unsafe.") from exc
    _require_regular_file(path, directory)


def _existing_artifact_bytes(path: Path, directory: Path) -> bytes | None:
    context = _active_publication_directory(directory)
    if context is not None:
        if path.parent != directory:
            raise HarmonyArtifactError("Existing harmonic context is unsafe.")
        if _relative_regular_file_snapshot(context[0], path.name) is None:
            return None
        return _read_stable_regular_file_at(context[0], path.name)
    try:
        path.lstat()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise HarmonyArtifactError("Existing harmonic context is unsafe.") from exc
    return _read_stable_regular_file(path, directory)


def _require_regular_file(path: Path, directory: Path) -> os.stat_result:
    try:
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
            raise HarmonyArtifactError("Harmonic context file is unsafe.")
        if directory.resolve(strict=True) != path.resolve(strict=True).parent:
            raise HarmonyArtifactError("Harmonic context file is unsafe.")
        return info
    except HarmonyArtifactError:
        raise
    except OSError as exc:
        raise HarmonyArtifactError("Harmonic context file is unsafe.") from exc


def _write_exclusive_regular_file(
    path: Path,
    data: bytes,
    directory: Path,
) -> HarmonyFileIdentity:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    context = _active_publication_directory(directory)
    if context is not None and path.parent != directory:
        raise HarmonyArtifactError("Temporary harmonic-context file is unsafe.")
    try:
        descriptor = (
            os.open(path.name, flags, 0o600, dir_fd=context[0])
            if context is not None
            else os.open(path, flags, 0o600)
        )
    except OSError as exc:
        raise HarmonyArtifactError(
            "Temporary harmonic-context file could not be created safely."
        ) from exc
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise HarmonyArtifactError("Temporary harmonic-context file is unsafe.")
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        written = os.fstat(descriptor)
        if not stat.S_ISREG(written.st_mode) or written.st_ino <= 0:
            raise HarmonyArtifactError("Temporary harmonic-context file is unsafe.")
        if context is not None:
            snapshot = _relative_regular_file_snapshot(context[0], path.name)
            if snapshot is None or _harmony_file_identity(snapshot) != (
                written.st_dev,
                written.st_ino,
                stat.S_IFMT(written.st_mode),
            ):
                raise HarmonyArtifactError(
                    "Temporary harmonic-context file changed during publication."
                )
        else:
            _require_regular_file(path, directory)
        return written.st_dev, written.st_ino, stat.S_IFMT(written.st_mode)
    except HarmonyArtifactError:
        raise
    except OSError as exc:
        raise HarmonyArtifactError(
            "Temporary harmonic-context file could not be written safely."
        ) from exc
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass


def _replace_if_destination(
    source: Path,
    destination: Path,
    *,
    expected_destination_identity: HarmonyFileIdentity | None,
) -> None:
    token = _EXPECTED_REPLACE_DESTINATION.set(expected_destination_identity)
    try:
        _replace_atomic(source, destination)
    finally:
        _EXPECTED_REPLACE_DESTINATION.reset(token)


def _replace_atomic(source: Path, destination: Path) -> None:
    context = _active_publication_directory(source.parent)
    expected_destination_identity = _EXPECTED_REPLACE_DESTINATION.get()
    if context is not None:
        if destination.parent != source.parent:
            raise HarmonyArtifactError("Harmony publication target is unsafe.")
        if expected_destination_identity is not _ANY_DESTINATION_IDENTITY:
            current = _current_published_file_identity(destination, source.parent)
            if current != expected_destination_identity:
                raise HarmonyArtifactError(
                    "Harmonic context changed before atomic replacement."
                )
        try:
            os.replace(
                source.name,
                destination.name,
                src_dir_fd=context[0],
                dst_dir_fd=context[0],
            )
        except OSError as exc:
            raise HarmonyArtifactError(
                "Harmonic context could not be published atomically."
            ) from exc
        return
    if expected_destination_identity is not _ANY_DESTINATION_IDENTITY:
        current = _current_published_file_identity(destination, source.parent)
        if current != expected_destination_identity:
            raise HarmonyArtifactError(
                "Harmonic context changed before atomic replacement."
            )
    try:
        os.replace(source, destination)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmonic context could not be published atomically."
        ) from exc


def _link_atomic(source: Path, destination: Path) -> None:
    context = _active_publication_directory(source.parent)
    if destination.parent != source.parent:
        raise HarmonyArtifactError("Harmony publication target is unsafe.")
    try:
        if context is not None:
            os.link(
                source.name,
                destination.name,
                src_dir_fd=context[0],
                dst_dir_fd=context[0],
                follow_symlinks=False,
            )
        else:
            os.link(source, destination, follow_symlinks=False)
    except FileExistsError:
        raise
    except (NotImplementedError, OSError, TypeError, ValueError) as exc:
        raise HarmonyArtifactError(
            "Harmonic context could not be published without replacement."
        ) from exc


@contextmanager
def _publication_recovery_scope() -> Iterator[None]:
    token = _PUBLICATION_RECOVERY.set(True)
    try:
        yield
    finally:
        _PUBLICATION_RECOVERY.reset(token)


def _restore_publication_state(
    destination: Path,
    directory: Path,
    job_dir: Path,
    expected_directory_snapshot: HarmonyDirectorySnapshot,
    previous: bytes | None,
    *,
    installed_identity: HarmonyFileIdentity,
) -> None:
    with _publication_recovery_scope():
        _restore_publication_state_in_scope(
            destination,
            directory,
            job_dir,
            expected_directory_snapshot,
            previous,
            installed_identity=installed_identity,
        )


def _restore_publication_state_in_scope(
    destination: Path,
    directory: Path,
    job_dir: Path,
    expected_directory_snapshot: HarmonyDirectorySnapshot,
    previous: bytes | None,
    *,
    installed_identity: HarmonyFileIdentity,
) -> None:
    _assert_publication_directory_current(
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    context = _active_publication_directory(directory)
    current_snapshot = (
        _relative_regular_file_snapshot(context[0], destination.name)
        if context is not None
        else None
    )
    if context is not None:
        if (
            current_snapshot is None
            or _harmony_file_identity(current_snapshot) != installed_identity
        ):
            raise HarmonyArtifactError(
                "Published harmonic context changed before rollback."
            )
    elif _published_file_identity(destination, directory) != installed_identity:
        raise HarmonyArtifactError(
            "Published harmonic context changed before rollback."
        )
    if previous is None:
        if not _descriptor_relative_cleanup_supported():
            raise HarmonyArtifactError(
                "Harmony publication rollback requires confined cleanup support."
            )
        owns_descriptor = context is None
        if context is None:
            descriptor, snapshot = _open_cleanup_directory(directory, job_dir)
        else:
            descriptor, snapshot = context[0], context[3]
        try:
            if snapshot != expected_directory_snapshot:
                raise HarmonyArtifactError(
                    "Harmony artifact directory changed while restoring publication state."
                )
            baseline = _relative_regular_file_snapshot(
                descriptor,
                destination.name,
            )
            if (
                baseline is None
                or _harmony_file_identity(baseline) != installed_identity
            ):
                raise HarmonyArtifactError(
                    "Published harmonic context changed before rollback."
                )
            removed = _remove_published_artifact(
                destination,
                directory,
                directory_descriptor=descriptor,
                expected_directory_snapshot=snapshot,
                job_dir=job_dir,
                leaf_name=destination.name,
                expected_file_snapshot=baseline,
            )
            if removed:
                _fsync_directory_descriptor(descriptor)
        finally:
            if owns_descriptor:
                os.close(descriptor)
        return

    temporary = directory / f".{destination.name}.{uuid4().hex}.restore.tmp"
    try:
        _write_exclusive_regular_file(temporary, previous, directory)
        _assert_publication_directory_current(
            directory,
            job_dir,
            expected_directory_snapshot,
        )
        if _published_file_identity(destination, directory) != installed_identity:
            raise HarmonyArtifactError(
                "Published harmonic context changed before rollback."
            )
        _replace_if_destination(
            temporary,
            destination,
            expected_destination_identity=installed_identity,
        )
        _assert_publication_directory_current(
            directory,
            job_dir,
            expected_directory_snapshot,
        )
        if _existing_artifact_bytes(destination, directory) != previous:
            raise HarmonyArtifactError(
                "Previous harmonic context could not be restored exactly."
            )
        _fsync_directory(directory)
    finally:
        _remove_temporary(temporary, directory, job_dir)


def _remove_published_artifact(
    path: Path,
    directory: Path,
    *,
    directory_descriptor: int | None = None,
    expected_directory_snapshot: HarmonyDirectorySnapshot | None = None,
    job_dir: Path | None = None,
    leaf_name: str | None = None,
    expected_file_snapshot: HarmonyFileSnapshot | None = None,
    cleanup_authorizer: HarmonyCleanupAuthorizer | None = None,
) -> bool:
    if (
        directory_descriptor is None
        or expected_directory_snapshot is None
        or job_dir is None
        or leaf_name is None
        or path.parent != directory
        or path.name != leaf_name
    ):
        raise HarmonyArtifactError("Harmony cleanup target is invalid.")
    _assert_removal_directory_current(
        directory_descriptor,
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    first = _relative_regular_file_snapshot(directory_descriptor, leaf_name)
    if first is None:
        return False
    baseline = expected_file_snapshot if expected_file_snapshot is not None else first
    if first != baseline:
        raise HarmonyArtifactError(
            "Harmony cleanup target changed before removal."
        )
    _assert_removal_directory_current(
        directory_descriptor,
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    if cleanup_authorizer is not None:
        try:
            cleanup_authorizer()
        except HarmonyArtifactError:
            raise
        except Exception as exc:
            raise HarmonyArtifactError(
                "Harmony cleanup authorization could not be verified."
            ) from exc
    _assert_removal_directory_current(
        directory_descriptor,
        directory,
        job_dir,
        expected_directory_snapshot,
    )
    second = _relative_regular_file_snapshot(directory_descriptor, leaf_name)
    if second is None:
        return False
    if second != baseline:
        raise HarmonyArtifactError(
            "Harmony cleanup target changed before removal."
        )
    try:
        os.unlink(leaf_name, dir_fd=directory_descriptor)
    except FileNotFoundError:
        return False
    except (OSError, TypeError, ValueError) as exc:
        raise HarmonyArtifactError(
            "Unverified harmonic context could not be removed safely."
        ) from exc
    if _relative_regular_file_snapshot(directory_descriptor, leaf_name) is None:
        return True
    raise HarmonyArtifactError(
        "Unverified harmonic context removal could not be verified."
    )


def _restore_harmony_artifact(
    job_id: str,
    settings: Settings,
    previous: Mapping[str, Any] | None,
    *,
    artifact_file_name: str | None = None,
) -> None:
    """Restore the pre-attempt target, or remove a first publication."""
    target = _resolve_artifact_file_name(artifact_file_name)
    if previous is not None:
        if target == HARMONY_ARTIFACT_RELATIVE_PATH and _ARTIFACT_SCOPE.get() is None:
            write_harmony_artifact(job_id, settings, previous)
        else:
            write_harmony_artifact(
                job_id,
                settings,
                previous,
                artifact_file_name=target,
            )
        return
    job_dir = _secure_job_root(job_id, settings)
    directory = _artifact_directory(job_dir, create=False)
    if directory is None:
        return
    remove_harmony_artifact(
        job_id,
        settings,
        artifact_file_name=target,
    )


def _remove_temporary(path: Path, directory: Path, job_dir: Path) -> None:
    if not _descriptor_relative_cleanup_supported():
        return
    context = _PUBLICATION_DIRECTORY.get()
    owns_descriptor = context is None
    if context is not None:
        descriptor, expected_directory, expected_job_dir, snapshot = context
        if expected_directory != directory or expected_job_dir != job_dir:
            return
    else:
        try:
            descriptor, snapshot = _open_cleanup_directory(directory, job_dir)
        except HarmonyArtifactError:
            return
    try:
        removed = _remove_published_artifact(
            path,
            directory,
            directory_descriptor=descriptor,
            expected_directory_snapshot=snapshot,
            job_dir=job_dir,
            leaf_name=path.name,
        )
        if removed:
            _fsync_directory_descriptor(descriptor)
    except HarmonyArtifactError:
        pass
    finally:
        if owns_descriptor:
            os.close(descriptor)


def _directory_fsync_supported() -> bool:
    return os.name != "nt"


def _fsync_directory(directory: Path) -> None:
    if not _directory_fsync_supported():
        return
    context = _active_publication_directory(directory)
    if context is not None:
        _fsync_directory_descriptor(context[0])
        _active_publication_directory(directory)
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(directory, flags)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be synchronized."
        ) from exc
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be synchronized."
        ) from exc
    finally:
        os.close(descriptor)


def _fsync_directory_descriptor(descriptor: int) -> None:
    try:
        os.fsync(descriptor)
    except OSError as exc:
        raise HarmonyArtifactError(
            "Harmony artifact directory could not be synchronized."
        ) from exc


def _regular_file_snapshot(
    path: Path,
    directory: Path,
    *,
    unavailable: bool,
) -> tuple[int, int, int, int]:
    try:
        info = _require_regular_file(path, directory)
    except HarmonyArtifactError as exc:
        if unavailable:
            try:
                path.lstat()
            except FileNotFoundError as missing:
                raise HarmonyArtifactUnavailableError(
                    "Published harmonic context is unavailable."
                ) from missing
            except OSError:
                pass
        raise exc
    return _snapshot(info)


def _read_stable_regular_file(path: Path, directory: Path) -> bytes:
    before = _regular_file_snapshot(path, directory, unavailable=False)
    if before[2] > _MAX_ARTIFACT_BYTES:
        raise HarmonyArtifactError("Saved harmonic context is too large.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise HarmonyArtifactError("Saved harmonic context is unavailable.") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or _snapshot(opened) != before:
            raise HarmonyArtifactError(
                "Saved harmonic context changed during validation."
            )
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(1024 * 1024, _MAX_ARTIFACT_BYTES + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_ARTIFACT_BYTES:
                raise HarmonyArtifactError("Saved harmonic context is too large.")
        after_open = os.fstat(descriptor)
        if _snapshot(after_open) != before:
            raise HarmonyArtifactError(
                "Saved harmonic context changed during validation."
            )
    except HarmonyArtifactError:
        raise
    except OSError as exc:
        raise HarmonyArtifactError("Saved harmonic context is unavailable.") from exc
    finally:
        os.close(descriptor)
    after_path = _regular_file_snapshot(path, directory, unavailable=False)
    if after_path != before:
        raise HarmonyArtifactError(
            "Saved harmonic context changed during validation."
        )
    return b"".join(chunks)


def _snapshot(info: os.stat_result) -> tuple[int, int, int, int]:
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


__all__ = [
    "HARMONY_ARTIFACT_RELATIVE_PATH",
    "HARMONY_ARTIFACT_SCHEMA_VERSION",
    "HarmonyArtifactError",
    "HarmonyArtifactUnavailableError",
    "HarmonyArtifactValidationError",
    "build_harmony_artifact",
    "harmony_artifact_path",
    "harmony_artifact_scope",
    "harmony_attempt_artifact_file_name",
    "load_harmony_artifact",
    "reconcile_harmony_attempt_artifacts",
    "remove_harmony_artifact",
    "validate_harmony_artifact",
    "write_harmony_artifact",
]
