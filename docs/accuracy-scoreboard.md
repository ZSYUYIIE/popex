# Accuracy scoreboard

The scoreboard compares a saved draft score with a small reference you write by hand for the same recording. It prints the same numbers every time, so you can tell whether a change to timing, harmony, or transcription made the draft better or worse.

It reads files only. It does not change predictions, corrections, or saved scores, needs no audio, uses no network, and adds no dependency.

## Run it

1. Build a score for the recording in PopEx and download **score data (JSON)**. The saved-score artifact and the `GET /api/jobs/{job_id}/score/saved?includeMeasures=true` payload work too.
2. Write a reference for that recording (format below).
3. Run:

```bash
python scripts/accuracy_scoreboard.py --score score.json --reference my-song.reference.json --label main
```

Options:

| Option | Purpose |
| --- | --- |
| `--analysis audio-analysis.json` | Also judge the key shown by the audio analysis. |
| `--label NAME` | Name the run, for example `main` or `beat-grid fix`. |
| `--save results.json` | Add the run to a results file. A run with the same reference and label is replaced. |
| `--json` | Print the result as JSON (`popex-accuracy-result`, schema 1) instead of text. |
| `--table results.json` | Print the saved runs side by side. |

To compare a change, save one run before it and one after it with different labels, then print the table.

The command exits with 0 on success and 2 when an input cannot be used. On failure it prints one plain message and writes nothing.

## Write a reference

A reference holds facts about one recording. It never holds audio or lyrics. See [`accuracy-reference.example.json`](accuracy-reference.example.json), which describes no real recording.

```json
{
  "artifactType": "popex-accuracy-reference",
  "schemaVersion": 1,
  "id": "my-song-studio",
  "title": "Optional display title",
  "source": "Where these facts come from",
  "timing": {"tempoBpm": 120, "beatsPerMeasure": 4, "firstDownbeatSeconds": 0.52},
  "key": {"tonalCenter": "G", "collections": ["ionian", "mixolydian"]},
  "chords": {"bars": [{"startBar": 5, "endBar": 36, "pattern": ["Am", "Am", "C", "C", "G", "G", "G", "G"]}]},
  "notes": {"bass": [{"startSeconds": 6.52, "midiNote": 45}]}
}
```

| Field | Required | Meaning |
| --- | --- | --- |
| `id` | yes | Lowercase letters, digits, and hyphens. Identifies the recording in saved runs. |
| `timing.tempoBpm`, `timing.beatsPerMeasure` | yes | The recording's tempo and meter. |
| `timing.firstDownbeatSeconds` | for bar patterns | Time of the first beat of bar 1. Needed to place bar patterns in time and to report aligned bars. |
| `key` | no | Tonal centre and every collection you accept as right. Known collections are the ones the score's tonal context reports: `ionian`, `dorian`, `phrygian`, `lydian`, `mixolydian`, `aeolian`, `locrian`, `harmonic_minor`, `melodic_minor`, `major_pentatonic`, `minor_pentatonic`, `blues`. `major` and `minor` are accepted as names for `ionian` and `aeolian`. |
| `chords.bars` | no | Bar ranges (1-based, inclusive) with a repeating pattern. One entry is one bar. Use a list such as `["G", "D"]` to split a bar into equal parts, and `"N"` for a bar without a chord. Ranges must not overlap. Suits recordings with a steady tempo. |
| `chords.segments` | no | The alternative to `bars`: `{"startSeconds", "endSeconds", "symbol"}` items, for recordings whose tempo moves. Give `bars` or `segments`, not both. |
| `notes` | no | Reference notes per source line (`vocals`, `bass`, `other`, `full_mix`), or `all` to compare against every note in the draft. Each note has `startSeconds` and `midiNote`. |

Leave out what you do not know. A missing group is reported as not measured, never as wrong.

### Keep references for real recordings out of the repository

Do not commit a reference for a real recording, the recording itself, or a score generated from it. Keep them in a private folder. A test fails if a tracked reference does not have an id beginning with `synthetic-`.

## What the numbers mean

Accuracy against the reference and indicators are printed in separate groups and must not be mixed.

### Timing

| Number | Meaning |
| --- | --- |
| Tempo error | Draft tempo against the reference, in percent. A half-tempo or double-tempo reading is flagged. |
| Meter | Whether the beats per bar match. |
| Bars | Bars written, and roughly how many the reference tempo implies for the same time span. |
| Bar-grid drift | Beats the draft notated minus beats that passed at the reference tempo, by the end of the draft and at its worst point. It does not depend on where bar 1 starts. A draft that holds one slightly wrong tempo for a whole recording drifts steadily: 1% over 400 beats is 4 beats, a whole bar in 4/4. |
| Bars starting on a reference downbeat | Share of bars that start within a quarter of a beat of a reference downbeat. Reported only when the reference gives `firstDownbeatSeconds`. |

### Key

For the score's tonal context and, when supplied, the analysis key: whether the tonal centre is right, whether the collection is one you accepted, and whether the predicted scale has the same notes as an accepted one (E minor against G major has the same notes and the wrong centre). The confidence and whether a key signature was written are shown beside them.

### Chords

Chords are compared in real time, as shares of the reference chord time. A stretch with no symbol counts as wrong.

| Number | Meaning |
| --- | --- |
| Covered | Share of the reference chord time where the draft shows any chord. |
| Root right | Share where the root matches. |
| Root and major/minor right | Share where the root and the major, minor, or other quality match. Sevenths, extensions, and slash basses are dropped first, so `Am7` equals `Am`; suspended, diminished, augmented, and power chords are grouped as other. |
| Root right where shown | Root accuracy over only the time the draft shows a chord. |

Two rows are printed. **Symbols shown on bars** is what the musician sees. **Harmony windows** scores the underlying candidates, including ones the draft did not place on a bar. Where overlapping windows disagree, a moment counts as right if any window is right, so this row is an upper bound on what a perfect choice between windows could show. A large gap between the rows means the candidates are better than what is placed on the page.

Because chords are compared in real time, a drifting bar grid lowers chord accuracy even when each bar carries a sensible symbol.

### Notes

Reported only when the reference has notes. A draft note is found when a reference note of the same pitch starts within 50 ms of it; each reference note can be matched once. Precision, recall, and F1 are given three ways:

- **as detected**: using the time the note was detected in the audio;
- **ignoring octave**: the same, counting a note in the wrong octave as found;
- **as notated on the grid**: using the time the note has on the page, which also reflects bar-grid errors.

### Indicators

Indicators describe the draft without reference notes: notes per source line, the share flagged low confidence, the share inside the reference key, how often the bass sits on the reference chord root, and percussion hit counts. They help explain a result. They are not accuracy: a draft can sit entirely in the right key and still be the wrong notes.

## What the scoreboard cannot tell you

- It measures agreement with your reference. A wrong reference gives wrong numbers.
- Without reference notes it does not measure note accuracy at all.
- It does not judge rests, note lengths, voicing, drum voices, tablature positions, or engraving.
- Chord quality is reduced to major, minor, or other; it does not check extensions.
- Bar patterns assume one steady tempo. Use timed segments when the tempo moves.
- One recording is one data point. Compare runs on the same recordings, and use several recordings before drawing a conclusion.
