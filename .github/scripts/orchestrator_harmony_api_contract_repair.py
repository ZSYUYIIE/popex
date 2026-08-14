from pathlib import Path

main_path = Path("app/main.py")
test_path = Path("tests/test_harmony_api.py")
main = main_path.read_text(encoding="utf-8")
tests = test_path.read_text(encoding="utf-8")


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"{label}: expected one marker, found {count}")
    return text.replace(old, new, 1)


old_result_counts = '''    expected_counts = {
        "eventCount": result.event_count,
        "segmentCount": result.segment_count,
        "resolvedSegmentCount": result.resolved_segment_count,
        "unresolvedSegmentCount": result.unresolved_segment_count,
        "unresolvedEventCount": result.unresolved_event_count,
        "warningCount": result.warning_count,
    }
'''
new_result_counts = '''    expected_counts = {
        "eventCount": result.event_count,
        "segmentCount": result.segment_count,
        "resolvedSegmentCount": result.resolved_segment_count,
        "unresolvedSegmentCount": result.unresolved_segment_count,
        "unresolvedEventCount": result.unresolved_event_count,
    }
'''
old_record_counts = '''    expected_counts = {
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
'''
new_record_counts = '''    expected_counts = {
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
    }
'''
old_details_warning = '''            "warnings": diagnostics["warningCount"],
'''
new_details_warning = '''            "warnings": len(artifact["warnings"]),
'''
old_test_publish = '''        harmony_warning_count=diagnostics["warningCount"],
'''
new_test_publish = '''        harmony_warning_count=len(payload["warnings"]),
'''
old_test_summary = '''        "warnings": artifact["diagnostics"]["warningCount"],
'''
new_test_summary = '''        "warnings": len(artifact["warnings"]),
'''
old_fixture = '''        interpretation_status=interpretation_status,
        interpretation_stage=interpretation_status,
        harmony_status=harmony_status,
        harmony_stage=harmony_status,
    )
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    return job_id
'''
new_fixture = '''        interpretation_status=interpretation_status,
        interpretation_stage=interpretation_status,
    )
    if transcribed:
        write_raw_transcription(job_id, settings, raw_payload())
    if harmony_status == "processing":
        assert db.claim_harmony_attempt(
            settings.database_path,
            job_id,
            harmony_version=HARMONY_PIPELINE_VERSION,
        )
    elif harmony_status != "not_started":
        db.update_job(
            settings.database_path,
            job_id,
            harmony_status=harmony_status,
            harmony_stage=harmony_status,
        )
    return job_id
'''

changed = False
for target, old, new, label in (
    ("main", old_result_counts, new_result_counts, "processor result counts"),
    ("main", old_record_counts, new_record_counts, "record match counts"),
    ("main", old_details_warning, new_details_warning, "details warning count"),
    ("tests", old_test_publish, new_test_publish, "test publish warning count"),
    ("tests", old_test_summary, new_test_summary, "test summary warning count"),
    ("tests", old_fixture, new_fixture, "active conflict fixture"),
):
    current = main if target == "main" else tests
    if old in current:
        current = replace_once(current, old, new, label)
        changed = True
    elif new not in current:
        raise SystemExit(f"{label}: marker not found")
    if target == "main":
        main = current
    else:
        tests = current

if changed:
    main_path.write_text(main, encoding="utf-8")
    test_path.write_text(tests, encoding="utf-8")
