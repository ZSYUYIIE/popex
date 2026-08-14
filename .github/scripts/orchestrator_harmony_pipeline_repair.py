from pathlib import Path

path = Path("tests/test_harmony_pipeline.py")
text = path.read_text(encoding="utf-8")

old_alignment = '''def _alignment(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        raw_time = event["startSeconds"]
        result.append(
            {
                "eventId": event["id"],
                "eventType": "pitched",
                "rawTimeSeconds": raw_time,
                "beatIndex": 0,
                "subdivision": 4,
                "subdivisionIndex": min(3, int(round(raw_time * 4))),
                "alignedTimeSeconds": raw_time,
                "offsetSeconds": 0.0,
                "confidence": 0.9,
                "measureIndex": 0,
                "beatInMeasure": 1,
            }
        )
    return result
'''
new_alignment = '''def _alignment(events: list[dict]) -> list[dict]:
    result = []
    for event in events:
        raw_time = event["startSeconds"]
        result.append(
            {
                "eventId": event["id"],
                "eventType": "pitched",
                "rawTimeSeconds": raw_time,
                "beatIndex": 0,
                "subdivision": 1,
                "subdivisionIndex": 0,
                "alignedTimeSeconds": 0.0,
                "offsetSeconds": raw_time,
                "confidence": 0.9,
                "measureIndex": 0,
                "beatInMeasure": 1,
            }
        )
    return result
'''
old_assertion = '''    assert reloaded_raw["pitchedNoteEvents"][0]["midiPitch"] == 60.12
'''
new_assertion = '''    assert next(
        event["midiPitch"]
        for event in reloaded_raw["pitchedNoteEvents"]
        if event["id"] == "p_c"
    ) == 60.12
'''

changed = False
if old_alignment in text:
    if text.count(old_alignment) != 1:
        raise SystemExit("alignment fixture marker is ambiguous")
    text = text.replace(old_alignment, new_alignment, 1)
    changed = True
elif new_alignment not in text:
    raise SystemExit("alignment fixture marker not found")

if old_assertion in text:
    if text.count(old_assertion) != 1:
        raise SystemExit("raw event assertion marker is ambiguous")
    text = text.replace(old_assertion, new_assertion, 1)
    changed = True
elif new_assertion not in text:
    raise SystemExit("raw event assertion marker not found")

if changed:
    path.write_text(text, encoding="utf-8")
