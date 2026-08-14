from pathlib import Path

script_path = Path(".github/scripts/orchestrator_harmony_api_patch.py")
source = script_path.read_text(encoding="utf-8")

call_start = source.index(
    "replace_once(\n    '''_INTERNAL_INTERPRETATION_FIELDS = frozenset("
)
call_end = source.index(
    "\n\nreplace_once(\n    '''    transcription_processor:",
    call_start,
)

replacement = """interpretation_start = text.index(
    \"_INTERNAL_INTERPRETATION_FIELDS = frozenset(\"
)
insertion_point = text.index(\"class JobCreate\", interpretation_start)
if \"_INTERNAL_HARMONY_FIELDS\" not in text[interpretation_start:insertion_point]:
    harmony_fields = '''_INTERNAL_HARMONY_FIELDS = frozenset(
    {
        \"harmony_status\",
        \"harmony_stage\",
        \"harmony_progress\",
        \"harmony_message\",
        \"harmony_attempt_version\",
        \"harmony_version\",
        \"harmony_artifact_file_name\",
        \"harmonized_at\",
        \"harmony_source_transcription_version\",
        \"harmony_source_transcription_artifact_file_name\",
        \"harmony_source_transcribed_at\",
        \"harmony_event_count\",
        \"harmony_segment_count\",
        \"harmony_resolved_segment_count\",
        \"harmony_unresolved_segment_count\",
        \"harmony_unresolved_event_count\",
        \"harmony_warning_count\",
        \"harmony_used_interpretation_context\",
        \"harmony_error\",
    }
)
'''
    text = text[:insertion_point] + harmony_fields + \"\\n\\n\" + text[insertion_point:]
"""

patched_source = source[:call_start] + replacement + source[call_end:]
exec(compile(patched_source, str(script_path), "exec"), {"__name__": "__main__"})
