# Drum timing corrections

In Edit score, each listed drum hit has a Timing disclosure with Bar and Beat
fields and an explicit Save drum timing button. Bar and Beat count from 1 in
the interface. Beats are quarter-note beats; drum placement uses the existing
half-beat (eighth-note) grid. This does not claim that inference found a more
precise rhythm. There is no drum-duration, accent, new-hit or kit-piece editor.

The existing correction endpoint accepts this strict operation inside a request
with the current `expectedRevision`:

```json
{"op":"set_hit_timing","target":{"eventId":"d_crash_2","hitIndex":0},"measureIndex":0,"beat":3.5}
```

JSON bar index and local beat are zero-based. The bar must exist and the local
beat must be less than the saved meter. Half-beat steps only; invalid fields,
booleans, strings, non-finite numbers and out-of-range values are rejected.
No bar is silently added. The operation uses the existing schema-1 correction
log without a database or saved-score migration. Existing logs remain readable;
older apps cannot interpret this operation and must not rewrite the log after
a downgrade.

Only notation placement changes. Raw seconds/beats, source identity, raw kind,
strength, confidence, interpretation placement and original quantization-shift
evidence remain unchanged. Original warnings and off-grid/unplaced counts refer
to that prediction evidence, not the size of a musician's subsequent edit.
The additive `corrections.review.hitTiming` map records original absolute
quantized beats under `eventId#hitIndex`. Existing `review.hits` values remain
original voice strings. The UI marks changed timing in normal and editing views.

Every colliding hit remains in the score. Two hits of the same voice in one slot
produce one notated hit and collapsed evidence; moving one away can restore it
to notation. Deterministic selection prefers edited hits, then confidence and
the original grid-shift evidence. Different voices can still sound together.
Corrected MusicXML and MIDI use the edited slot; original exports remain
available. Undo, redo and reset use the existing revision-checked workflow.

Rebuilds replay by stable event ID and hit index. If a target or requested bar
disappears the correction is reported as not applicable, never dropped or
redirected. Edits are bar-relative, not absolute seconds; review alignment if a
future automatic tempo/meter/grid-origin rebuild changes the bar grid.

This change does not rerun inference, change the drum grid, add notation
engraving or merge parts from different recordings.
