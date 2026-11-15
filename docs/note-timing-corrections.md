# Note timing corrections

The draft score's Edit score controls include a Timing disclosure per pitched
note. Bar and Beat are one-based in the interface; Duration is measured in
quarter-note beats. Enter quarter-beat steps (one sixteenth note), then Save
timing. This is a musician's explicit edit, not a claim that inference detected
sixteenth-note precision. Playback review still uses the original recording;
the piano roll and corrected JSON, MIDI and MusicXML use the edited notation.
Cross-bar holds export with ties through the existing exporter.

The existing correction endpoint accepts this strict operation:

```json
{"op":"set_note_timing","target":{"noteId":"p_c"},"measureIndex":0,"beat":0.25,"durationBeats":1.5}
```

The surrounding request includes the current expectedRevision, exactly like
pitch edits. The bar index and local beat are zero-based in JSON. The operation
is stored in the existing schema-1 correction log; no database or score schema
migration is needed. Old operation kinds remain valid. Older app versions do
not understand the new kind and must not rewrite this log after a downgrade.

The target is a stable raw event ID. Only quantized notation placement changes;
raw seconds, raw beat coordinates, fractional pitch, confidence, source warnings
and original quantization-shift evidence remain untouched. The review report
includes the original quantized start/duration. The saved prediction file is
immutable; original exports, undo, redo and reset remain available.

Starts and durations must fit existing bars: local beat is less than the meter,
duration is positive, and the end cannot extend beyond the saved score. No
clipping, new empty bars or missing-note insertion is performed. Duration is
bounded to 4096 beats; steps are 0.25 beats. NaN, infinity, booleans, strings and
unsupported fields are rejected. A timing edit overlapping another assigned
note on the same tablature instrument/string is refused, not silently refingered.
It does not repair pre-existing fingering conflicts or validate instrument technique.

On rebuild, the operation is reapplied to the stable ID and saved bar grid, not
to absolute seconds. If the note or requested extent no longer exists it is
reported as not applicable, never dropped or redirected. If the automatic grid
phase changes, a musician should review the corrected bar placement again.
Automatic inference warnings and tonal context remain evidence from the saved
prediction; corrections do not rerun those analyses.

This narrow change does not add notes, edit percussion timing, infer tuplets,
create an engraving view or change automatic rhythm/downbeat detection.

Validation includes focused timing/correction/API/frontend tests, full pytest,
compileall and browser JS syntax, a real browser keyboard/save/undo flow, and
the synthetic upload-to-score smoke with a timing edit and original hash checks.
