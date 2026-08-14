from pathlib import Path

path = Path("app/static/app.js")
text = path.read_text(encoding="utf-8")

old_helper = '''function harmonyMidi(value){const number=Number(value);return Number.isFinite(number)&&number>=0&&number<=127?number.toFixed(2):"unresolved"}
'''
new_helper = '''function harmonyMidi(value){const number=Number(value);return Number.isFinite(number)&&number>=0&&number<=127?number.toFixed(2):"unresolved"}
function safeHarmonyDate(value){const text=safeUiText(value,"",80);if(!text||!/^\\d{4}-\\d{2}-\\d{2}T/.test(text))return"";const date=new Date(text);return Number.isNaN(date.getTime())?"":text}
'''
old_created = '''createdAt=safeUiText(detailObject?.createdAt??harmony.createdAt,"",80)'''
new_created = '''createdAt=safeHarmonyDate(detailObject?.createdAt??harmony.createdAt)'''

changed = False
if "function safeHarmonyDate(value)" not in text:
    if text.count(old_helper) != 1:
        raise SystemExit(
            f"harmony date helper marker: expected one, found {text.count(old_helper)}"
        )
    text = text.replace(old_helper, new_helper, 1)
    changed = True

if old_created in text:
    if text.count(old_created) != 1:
        raise SystemExit(
            f"harmony createdAt marker: expected one, found {text.count(old_created)}"
        )
    text = text.replace(old_created, new_created, 1)
    changed = True
elif new_created not in text:
    raise SystemExit("harmony createdAt marker not found")

if changed:
    path.write_text(text, encoding="utf-8")
