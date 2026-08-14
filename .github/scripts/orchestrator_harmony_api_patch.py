from pathlib import Path

path = Path("app/main.py")
text = path.read_text(encoding="utf-8")


def replace_once(old: str, new: str, label: str) -> None:
    global text
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    text = text.replace(old, new, 1)


if "def _run_harmony_job(" in text:
    required = (
        '"/api/jobs/{job_id}/harmonize"',
        '"/api/jobs/{job_id}/harmony"',
        "def _serialize_harmony(",
        "_INTERNAL_HARMONY_FIELDS",
    )
    if all(marker in text for marker in required):
        print("harmony API already patched")
        raise SystemExit(0)
    raise SystemExit("partial harmony API patch detected")

replace_once(
    '''from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    InterpretationPipelineError,
    InterpretationPipelineResult,
    interpret_transcription_job,
)
''',
    '''from app.interpretation_pipeline import (
    INTERPRETATION_PIPELINE_VERSION,
    InterpretationPipelineError,
    InterpretationPipelineResult,
    interpret_transcription_job,
)
from app.harmony_artifacts import (
    HARMONY_ARTIFACT_RELATIVE_PATH,
    HarmonyArtifactError,
    HarmonyArtifactUnavailableError,
    harmony_artifact_path,
    load_harmony_artifact,
)
from app.harmony_pipeline import (
    HARMONY_PIPELINE_VERSION,
    HarmonyPipelineError,
    HarmonyPipelineResult,
    infer_harmony_job,
)
''',
    "harmony imports",
)

replace_once(
    '''InterpretationProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    InterpretationPipelineResult,
]
''',
    '''InterpretationProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    InterpretationPipelineResult,
]
HarmonyProcessor = Callable[
    [str, Settings, Callable[[str, str, float], None]],
    HarmonyPipelineResult,
]
''',
    "harmony processor type",
)

replace_once(
    '''_INTERNAL_INTERPRETATION_FIELDS = frozenset(
    {
        "interpretation_status",
        "interpretation_stage",
        "interpretation_progress",
        "interpretation_message",
        "interpretation_version",
        "interpretation_artifact_file_name",
        "interpreted_at",
        "interpretation_part_count",
        "interpretation_phrase_count",
        "interpretation_pitched_item_count",
        "interpretation_percussion_item_count",
        "interpretation_warning_count",
        "interpretation_error",
    }
)
''',
    '''_INTERNAL_INTERPRETATION_FIELDS = frozenset(
    {
        "interpretation_status",
        "interpretation_stage",
        "interpretation_progress",
        "interpretation_message",
        "interpretation_version",
        "interpretation_artifact_file_name",
        "interpreted_at",
        "interpretation_part_count",
        "interpretation_phrase_count",
        "interpretation_pitched_item_count",
        "interpretation_percussion_item_count",
        "interpretation_warning_count",
        "interpretation_error",
    }
)
_INTERNAL_HARMONY_FIELDS = frozenset(
    {
        "harmony_status",
        "harmony_stage",
        "harmony_progress",
        "harmony_message",
        "harmony_attempt_version",
        "harmony_version",
        "harmony_artifact_file_name",
        "harmonized_at",
        "harmony_source_transcription_version",
        "harmony_source_transcription_artifact_file_name",
        "harmony_source_transcribed_at",
        "harmony_event_count",
        "harmony_segment_count",
        "harmony_resolved_segment_count",
        "harmony_unresolved_segment_count",
        "harmony_unresolved_event_count",
        "harmony_warning_count",
        "harmony_used_interpretation_context",
        "harmony_error",
    }
)
''',
    "internal harmony fields",
)

replace_once(
    '''    transcription_processor: TranscriptionProcessor = transcribe_job,
    interpretation_processor: InterpretationProcessor = interpret_transcription_job,
) -> FastAPI:
''',
    '''    transcription_processor: TranscriptionProcessor = transcribe_job,
    interpretation_processor: InterpretationProcessor = interpret_transcription_job,
    harmony_processor: HarmonyProcessor = infer_harmony_job,
) -> FastAPI:
''',
    "create_app harmony injection",
)

harmony_start_route = '''

    @app.post(
        "/api/jobs/{job_id}/harmonize",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def harmonize_existing_job(
        job_id: str,
        background_tasks: BackgroundTasks,
        force: str | None = Query(None),
    ) -> dict:
        force_value = _strict_query_bool(force, field="force", default=False)
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        if (
            record.get("preparation_status") != "completed"
            or record.get("analysis_status") != "completed"
            or record.get("analysis_json_file_name")
            != ANALYSIS_JSON_RELATIVE_PATH
            or not isinstance(record.get("analysis_version"), str)
            or not record.get("analysis_version")
            or record.get("transcription_status") != "completed"
            or record.get("transcription_artifact_file_name")
            != RAW_TRANSCRIPTION_RELATIVE_PATH
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "Completed canonical audio analysis and raw transcription "
                    "are required before harmonic context."
                ),
            )
        try:
            transcription_json_path(job_id, app_settings, record)
        except (
            TranscriptionArtifactUnavailableError,
            TranscriptionArtifactError,
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid published raw transcription is required before "
                    "harmonic context."
                ),
            ) from None
        try:
            analysis = load_analysis(job_id, app_settings)
        except (AudioAnalysisError, MediaProcessingError):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid matching audio analysis is required before "
                    "harmonic context."
                ),
            ) from None
        if (
            not isinstance(analysis, dict)
            or analysis.get("analysisVersion") != record.get("analysis_version")
        ):
            raise HTTPException(
                status_code=409,
                detail=(
                    "A valid matching audio analysis is required before "
                    "harmonic context."
                ),
            )

        harmony_status = record.get("harmony_status") or "not_started"
        if harmony_status == "processing":
            raise HTTPException(
                status_code=409,
                detail="Harmonic-context processing is already running.",
            )
        if harmony_status == "completed" and not force_value:
            raise HTTPException(
                status_code=409,
                detail=(
                    "Harmonic context is already complete; use force=true to "
                    "re-harmonize."
                ),
            )
        claimed = db.claim_harmony_attempt(
            app_settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
            force=force_value,
        )
        if not claimed:
            raise HTTPException(
                status_code=409,
                detail="Harmonic context could not be started in the current state.",
            )
        background_tasks.add_task(
            _run_harmony_job,
            job_id,
            app_settings,
            harmony_processor,
        )
        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)
'''

replace_once(
    '''        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)

    @app.get("/api/jobs/{job_id}/analysis")
''',
    '''        current = db.get_job(app_settings.database_path, job_id)
        return serialize_job(current or record)'''
    + harmony_start_route
    + '''

    @app.get("/api/jobs/{job_id}/analysis")
''',
    "harmony start route",
)

harmony_read_routes = '''

    @app.get("/api/jobs/{job_id}/harmony")
    def get_harmony(
        job_id: str,
        include_segments: str | None = Query(None, alias="includeSegments"),
    ) -> dict:
        include_segments_value = _strict_query_bool(
            include_segments,
            field="includeSegments",
            default=False,
        )
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        artifact = _load_harmony_for_http(job_id, app_settings, record)
        return _harmony_details_payload(
            artifact,
            record,
            include_segments=include_segments_value,
        )

    @app.get("/api/jobs/{job_id}/harmony/download")
    def download_harmony(job_id: str) -> FileResponse:
        record = db.get_job(app_settings.database_path, job_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Job not found")
        _load_harmony_for_http(job_id, app_settings, record)
        try:
            path = harmony_artifact_path(job_id, app_settings)
        except HarmonyArtifactUnavailableError:
            raise HTTPException(
                status_code=404,
                detail="Published harmonic context is unavailable.",
            ) from None
        except HarmonyArtifactError:
            logging.exception(
                "Published harmonic context failed path validation for job %s",
                job_id,
            )
            raise HTTPException(
                status_code=500,
                detail="Published harmonic context could not be validated.",
            ) from None
        return FileResponse(
            path,
            filename="harmonic-context.json",
            media_type="application/json",
        )
'''

replace_once(
    '''        return FileResponse(
            path,
            filename="editable-interpretation.json",
            media_type="application/json",
        )

    @app.get("/api/jobs/{job_id}/stems")
''',
    '''        return FileResponse(
            path,
            filename="editable-interpretation.json",
            media_type="application/json",
        )'''
    + harmony_read_routes
    + '''

    @app.get("/api/jobs/{job_id}/stems")
''',
    "harmony read routes",
)

harmony_worker = '''


def _run_harmony_job(
    job_id: str,
    settings: Settings,
    processor: HarmonyProcessor,
) -> None:
    last_progress = 1.0

    def update_stage(stage: str, message: str, progress: float) -> None:
        nonlocal last_progress
        try:
            numeric_progress = float(progress)
        except (TypeError, ValueError):
            numeric_progress = last_progress
        numeric_progress = max(
            last_progress,
            min(99.0, max(1.0, numeric_progress)),
        )
        next_progress = round(numeric_progress, 1)
        updated = db.update_harmony_progress(
            settings.database_path,
            job_id,
            stage=stage,
            progress=next_progress,
            message=message,
        )
        if not updated:
            raise HarmonyPipelineError(
                "Harmonic-context attempt is no longer active."
            )
        last_progress = next_progress

    try:
        result = processor(job_id, settings, update_stage)
    except HarmonyPipelineError as exc:
        _record_harmony_failure(
            settings,
            job_id,
            _safe_harmony_error(str(exc), settings),
        )
    except Exception:
        logging.exception(
            "Unexpected harmonic-context failure for job %s",
            job_id,
        )
        _record_harmony_failure(
            settings,
            job_id,
            "Unexpected harmonic-context failure. Check server logs.",
        )
    else:
        if not _validate_harmony_processor_result(
            job_id,
            settings,
            result,
        ):
            logging.error(
                "Harmony processor returned an invalid result for %s",
                job_id,
            )
            _record_harmony_failure(
                settings,
                job_id,
                "Harmonic context returned an invalid result.",
            )
            return
        completed = db.complete_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=result.version,
            artifact_file_name=result.artifact_file_name,
            harmonized_at=result.created_at,
            event_count=result.event_count,
            segment_count=result.segment_count,
            resolved_segment_count=result.resolved_segment_count,
            unresolved_segment_count=result.unresolved_segment_count,
            unresolved_event_count=result.unresolved_event_count,
            warning_count=result.warning_count,
            used_interpretation_context=result.used_interpretation_context,
        )
        if not completed:
            logging.warning(
                "Discarded stale harmonic-context completion for job %s",
                job_id,
            )


def _validate_harmony_processor_result(
    job_id: str,
    settings: Settings,
    result: object,
) -> bool:
    if not isinstance(result, HarmonyPipelineResult):
        return False
    counts = (
        result.event_count,
        result.segment_count,
        result.resolved_segment_count,
        result.unresolved_segment_count,
        result.unresolved_event_count,
        result.warning_count,
    )
    if (
        result.artifact_file_name != HARMONY_ARTIFACT_RELATIVE_PATH
        or not isinstance(result.version, str)
        or not result.version
        or not isinstance(result.created_at, str)
        or not result.created_at
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in counts
        )
        or result.segment_count
        != result.resolved_segment_count + result.unresolved_segment_count
        or result.unresolved_event_count > result.event_count
        or type(result.used_interpretation_context) is not bool
        or (
            result.used_interpretation_context
            and (
                not isinstance(result.interpretation_version, str)
                or not result.interpretation_version
            )
        )
        or (
            not result.used_interpretation_context
            and result.interpretation_version is not None
        )
        or not isinstance(result.warnings, tuple)
        or result.warning_count != len(result.warnings)
        or not isinstance(result.payload, dict)
    ):
        return False
    try:
        artifact = load_harmony_artifact(job_id, settings)
    except HarmonyArtifactError:
        return False
    if artifact is None or artifact != result.payload:
        return False
    diagnostics = artifact.get("diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    expected_counts = {
        "eventCount": result.event_count,
        "segmentCount": result.segment_count,
        "resolvedSegmentCount": result.resolved_segment_count,
        "unresolvedSegmentCount": result.unresolved_segment_count,
        "unresolvedEventCount": result.unresolved_event_count,
        "warningCount": result.warning_count,
    }
    if (
        artifact.get("harmonyVersion") != result.version
        or artifact.get("createdAt") != result.created_at
        or any(diagnostics.get(name) != value for name, value in expected_counts.items())
        or tuple(artifact.get("warnings", ())) != result.warnings
        or ("sourceInterpretation" in artifact)
        != result.used_interpretation_context
    ):
        return False
    if result.used_interpretation_context:
        source = artifact.get("sourceInterpretation")
        if (
            not isinstance(source, dict)
            or source.get("draftVersion") != result.interpretation_version
        ):
            return False
    return True


def _record_harmony_failure(
    settings: Settings,
    job_id: str,
    error: str,
) -> None:
    db.fail_harmony_attempt(
        settings.database_path,
        job_id,
        error=error,
    )


def _safe_harmony_error(value: str, settings: Settings) -> str:
    text = str(value)
    lowered = text.lower()
    if "traceback (most recent call last)" in lowered or "stack trace" in lowered:
        return "Harmonic-context processing failed."
    try:
        cleaned = friendly_error(text, settings=settings)
    except (OSError, RuntimeError, ValueError):
        return "Harmonic-context processing failed."
    cleaned = re.sub(
        r"(?i)\\b(?:https?|file)://[^\\s]+",
        "<external location>",
        cleaned,
    )
    cleaned = re.sub(
        r"(?i)\\b(?:token|password|secret|api[_-]?key|authorization|bearer)"
        r"\\s*(?:=|:)?\\s*[^\\s,;]+",
        "<redacted>",
        cleaned,
    )
    cleaned = re.sub(r"(?i)0x[0-9a-f]{6,}", "<address>", cleaned)
    cleaned = " ".join(cleaned.replace("\\x00", "").split()).strip()
    if not cleaned or "/" in cleaned or "\\\\" in cleaned:
        return "Harmonic-context processing failed."
    return cleaned[:500]
'''

replace_once(
    '''    return cleaned[:500]


def _record_analysis_failure(
''',
    '''    return cleaned[:500]'''
    + harmony_worker
    + '''


def _record_analysis_failure(
''',
    "harmony worker",
)

replace_once(
    '''        if key not in _INTERNAL_SEPARATION_FIELDS
        and key not in _INTERNAL_TRANSCRIPTION_FIELDS
        and key not in _INTERNAL_INTERPRETATION_FIELDS
''',
    '''        if key not in _INTERNAL_SEPARATION_FIELDS
        and key not in _INTERNAL_TRANSCRIPTION_FIELDS
        and key not in _INTERNAL_INTERPRETATION_FIELDS
        and key not in _INTERNAL_HARMONY_FIELDS
''',
    "hide harmony fields",
)

replace_once(
    '''    interpretation = _serialize_interpretation(job)
    if interpretation is not None:
        payload["interpretation"] = interpretation
    return payload
''',
    '''    interpretation = _serialize_interpretation(job)
    if interpretation is not None:
        payload["interpretation"] = interpretation
    harmony = _serialize_harmony(job)
    if harmony is not None:
        payload["harmony"] = harmony
    return payload
''',
    "serialize harmony contract",
)

serialize_harmony = '''


def _serialize_harmony(job: dict) -> dict[str, Any] | None:
    status_value = job.get("harmony_status")
    status = (
        status_value
        if isinstance(status_value, str) and status_value
        else "not_started"
    )
    should_expose = (
        job.get("transcription_status") == "completed"
        or status != "not_started"
        or job.get("harmony_artifact_file_name") is not None
    )
    if not should_expose:
        return None
    job_id = job["id"]
    available = (
        job.get("harmony_artifact_file_name")
        == HARMONY_ARTIFACT_RELATIVE_PATH
    )
    ready = (
        job.get("analysis_status") == "completed"
        and job.get("analysis_json_file_name")
        == ANALYSIS_JSON_RELATIVE_PATH
        and job.get("transcription_status") == "completed"
        and job.get("transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
    )
    context_value = job.get("harmony_used_interpretation_context")
    used_context = (
        bool(context_value)
        if type(context_value) is int and context_value in {0, 1}
        else None
    )
    return {
        "enabled": True,
        "status": status,
        "stage": job.get("harmony_stage") or "not_started",
        "progress": _safe_progress(job.get("harmony_progress")),
        "message": job.get("harmony_message"),
        "attemptVersion": job.get("harmony_attempt_version"),
        "version": job.get("harmony_version"),
        "createdAt": job.get("harmonized_at"),
        "available": available,
        "counts": {
            "events": _safe_count(job.get("harmony_event_count")),
            "segments": _safe_count(job.get("harmony_segment_count")),
            "resolved": _safe_count(
                job.get("harmony_resolved_segment_count")
            ),
            "unresolved": _safe_count(
                job.get("harmony_unresolved_segment_count")
            ),
            "unresolvedEvents": _safe_count(
                job.get("harmony_unresolved_event_count")
            ),
            "warnings": _safe_count(job.get("harmony_warning_count")),
        },
        "usedInterpretationContext": used_context,
        "canStart": ready and status in {"not_started", "failed"},
        "canReharmonize": ready and status == "completed",
        "startUrl": f"/api/jobs/{job_id}/harmonize",
        "detailsUrl": f"/api/jobs/{job_id}/harmony?includeSegments=false",
        "fullDetailsUrl": f"/api/jobs/{job_id}/harmony?includeSegments=true",
        "downloadUrl": (
            f"/api/jobs/{job_id}/harmony/download" if available else None
        ),
        "error": job.get("harmony_error"),
    }
'''

replace_once(
    '''

def _strict_query_bool(value: str | None, *, field: str, default: bool) -> bool:
''',
    serialize_harmony
    + '''


def _strict_query_bool(value: str | None, *, field: str, default: bool) -> bool:
''',
    "serialize harmony helper",
)

harmony_http_helpers = '''


def _load_harmony_for_http(
    job_id: str,
    settings: Settings,
    record: dict,
) -> dict[str, Any]:
    if (
        record.get("harmony_artifact_file_name")
        != HARMONY_ARTIFACT_RELATIVE_PATH
    ):
        raise HTTPException(
            status_code=404,
            detail="Published harmonic context is unavailable.",
        )
    try:
        artifact = load_harmony_artifact(job_id, settings)
    except HarmonyArtifactError:
        logging.exception(
            "Published harmonic context failed validation for job %s",
            job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Published harmonic context could not be validated.",
        ) from None
    if artifact is None:
        raise HTTPException(
            status_code=404,
            detail="Published harmonic context is unavailable.",
        )
    if not _harmony_artifact_matches_record(artifact, record):
        logging.error(
            "Published harmonic context metadata mismatch for job %s",
            job_id,
        )
        raise HTTPException(
            status_code=500,
            detail="Published harmonic context could not be validated.",
        )
    return artifact


def _harmony_artifact_matches_record(
    artifact: dict[str, Any],
    record: dict,
) -> bool:
    diagnostics = artifact.get("diagnostics")
    transcription = artifact.get("sourceTranscription")
    analysis = artifact.get("sourceAnalysis")
    if not all(
        isinstance(value, dict)
        for value in (diagnostics, transcription, analysis)
    ):
        return False
    context_value = record.get("harmony_used_interpretation_context")
    if type(context_value) is not int or context_value not in {0, 1}:
        return False
    expected_counts = {
        "eventCount": record.get("harmony_event_count"),
        "segmentCount": record.get("harmony_segment_count"),
        "resolvedSegmentCount": record.get(
            "harmony_resolved_segment_count"
        ),
        "unresolvedSegmentCount": record.get(
            "harmony_unresolved_segment_count"
        ),
        "unresolvedEventCount": record.get(
            "harmony_unresolved_event_count"
        ),
        "warningCount": record.get("harmony_warning_count"),
    }
    return (
        artifact.get("harmonyVersion") == record.get("harmony_version")
        and artifact.get("createdAt") == record.get("harmonized_at")
        and transcription.get("fileName")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
        and transcription.get("transcriptionVersion")
        == record.get("harmony_source_transcription_version")
        and record.get("transcription_version")
        == record.get("harmony_source_transcription_version")
        and record.get("transcription_artifact_file_name")
        == record.get("harmony_source_transcription_artifact_file_name")
        == RAW_TRANSCRIPTION_RELATIVE_PATH
        and record.get("transcribed_at")
        == record.get("harmony_source_transcribed_at")
        and analysis.get("fileName") == ANALYSIS_JSON_RELATIVE_PATH
        and analysis.get("analysisVersion")
        == record.get("analysis_version")
        and all(
            diagnostics.get(name) == value
            and isinstance(value, int)
            and not isinstance(value, bool)
            and value >= 0
            for name, value in expected_counts.items()
        )
        and len(artifact.get("warnings", ()))
        == record.get("harmony_warning_count")
        and ("sourceInterpretation" in artifact) == bool(context_value)
    )


def _harmony_details_payload(
    artifact: dict[str, Any],
    record: dict,
    *,
    include_segments: bool,
) -> dict[str, Any]:
    diagnostics = artifact["diagnostics"]
    payload: dict[str, Any] = {
        "available": True,
        "status": record.get("harmony_status") or "not_started",
        "schemaVersion": artifact["schemaVersion"],
        "harmonyVersion": artifact["harmonyVersion"],
        "createdAt": artifact["createdAt"],
        "sourceTranscription": artifact["sourceTranscription"],
        "sourceAnalysis": artifact["sourceAnalysis"],
        "algorithms": artifact["algorithms"],
        "tonalContext": artifact["tonalContext"],
        "counts": {
            "events": diagnostics["eventCount"],
            "segments": diagnostics["segmentCount"],
            "resolved": diagnostics["resolvedSegmentCount"],
            "unresolved": diagnostics["unresolvedSegmentCount"],
            "unresolvedEvents": diagnostics["unresolvedEventCount"],
            "warnings": diagnostics["warningCount"],
        },
        "usedInterpretationContext": "sourceInterpretation" in artifact,
        "warnings": artifact["warnings"],
        "diagnostics": diagnostics,
        "segmentsIncluded": include_segments,
        "downloadUrl": f"/api/jobs/{record['id']}/harmony/download",
    }
    if "sourceInterpretation" in artifact:
        payload["sourceInterpretation"] = artifact[
            "sourceInterpretation"
        ]
    if include_segments:
        payload["rawEvidence"] = artifact["rawEvidence"]
        payload["segments"] = artifact["segments"]
        payload["unresolvedEventIds"] = artifact[
            "unresolvedEventIds"
        ]
    return payload
'''

replace_once(
    '''

def _stem_artifact_or_http_error(
''',
    harmony_http_helpers
    + '''


def _stem_artifact_or_http_error(
''',
    "harmony HTTP helpers",
)

path.write_text(text, encoding="utf-8")
