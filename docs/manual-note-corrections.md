# Add a missed note

Open a recording's saved draft, choose **Edit score**, then **Add a missed note**.
Choose its part, pitch, octave (C4 is middle C), bar, beat and duration. Add note
saves a separate musician correction. Undo removes it; redo restores the same
identity. Pitch, timing, tablature position and deletion controls also work on
added notes. Reset to prediction is itself undoable.

The **Musician notes** selector exposes every active addition, including those
outside the first 64 bars/24 notes shown in the main review-table preview. Choose
one to use the same pitch, timing, TAB and deletion controls there.

The form uses quarter-note beats in quarter-beat steps. A duration can cross bar
lines but must fit the existing score. Extend/rebuild the score before entering
notes outside its bars. Native controls are keyboard accessible; unfinished
values remain through polling refreshes and an unavailable part requires a new
choice rather than silently selecting another part.

## Parts and honest provenance

Existing pitched parts and guitar/bass TAB parts are available. Schema-5 scores
can also create a **Musician-entered notes** part; it appears only after a note
is actually added. Older scores retain their combined pitched part. No detected
instrument is invented. If the selected part disappears after a rebuild, the
addition is kept in the log and reported as not applicable, not redirected.

The added note's `origin` is `musician`, `entryVersion` is 1, and `correctionId`
identifies the server-generated operation. Its stable `user:<operation-id>` ID
survives rebuilds/restarts and can be targeted by later corrections. Raw seconds,
raw beats, fractional model pitch, quantization shift and detector confidence are
null: there was no prediction for this entry. `partId` is null rather than a fake
inferred interpretation label; `targetPartId` names the musician's export-part
choice. `sourceKind` is routing metadata only, not new detector evidence.

The structured review labels additions **Added by musician**. Corrected JSON
retains that versioned provenance; original JSON/MIDI/MusicXML and immutable
prediction artifacts remain unchanged. Analysis, harmonic candidates and their
confidence are not recomputed from manual notes.

TAB placement is a suggestion on the existing instrument/tuning, separate from
pitch. Initial entry avoids strings occupied by overlapping assigned notes. If
no position is available, the note remains honestly `unplayable` or `out_of_range`
rather than being dropped or transposed. Existing fingering corrections and
limitations still apply; this is not a complete physical guitar-performance model.

## API and persistence

Saved-score details add `noteEntryParts`, derived from the immutable base so
deleting all predictions does not remove the available entry targets.

```json
{
  "expectedRevision": 0,
  "operation": {
    "op": "add_note",
    "target": {"partId": "musician-notes"},
    "measureIndex": 0,
    "beat": 0.25,
    "durationBeats": 1.5,
    "midiNote": 64
  }
}
```

POST this to `/api/jobs/{job_id}/score/corrections`. Bar/beat indexes are zero
based in the API, one based in the interface. Pitch is an integer MIDI note from
C0 (12) through G9 (127); booleans/coerced strings/nonfinite values, unsupported
fields, fake evidence/IDs and percussion targets are refused. The server owns
the operation ID. A stale revision returns 409 without adding another note.

The additive operation uses the existing versioned correction log and requires
no database/prediction-artifact migration. A musician note is a **correction
projection**, not a persisted prediction event. Default artifact loading and
encoding reject it; corrected-view validation explicitly authorizes only the
active replay's exact manual IDs, then checks complete counts, parts, provenance
and exports. Older raw source labels are not reclassified as musician notes.

MIDI/MusicXML limits are checked before saving an addition-containing projection
and again after replay against a rebuilt score. A rejected correction leaves the
prior log/downloads intact. If a rebuild makes a replay unexportable, the original
remains available and corrections are explicitly reported unusable. Limits remain
2,000 correction operations and 5,000 score notes; unsupported overlapping MIDI
unisons fail rather than being truncated or sent to the percussion channel.

Drum-hit insertion, arbitrary instrument/part design, tuning/capo changes and
automatic recognition improvements are separate work. No dependency, model,
dataset, recording upload, paid API or hosted service is introduced here.
