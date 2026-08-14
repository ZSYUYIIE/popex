from pathlib import Path

path = Path("tests/test_harmony_pipeline.py")
text = path.read_text(encoding="utf-8")
old = '''def _alignment(events: list[dict]) -> list[dict]:
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
new = '''def _alignment(events: list[dict]) -> list[dict]:
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
if old in text:
    if text.count(old) != 1:
        raise SystemExit("alignment fixture marker is ambiguous")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
elif new not in text:
    raise SystemExit("alignment fixture marker not found")
