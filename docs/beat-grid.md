# Bar grid: bars follow the tracked beats

A draft score needs bar lines that sit where a musician counts them. PopEx places them with a beat grid (`beat-grid-v1`, [`app/beat_grid.py`](../app/beat_grid.py)) built from the beats the audio analysis tracked.

## Why not one tempo

Earlier scores used a single tempo for the whole recording and started bar 1 at 0 s. A tempo estimate that is 1% off moves the bar lines by 4 beats over 400 beats, a whole bar in 4/4, and a recording whose tempo moves never fits. On the first real recording measured with the [accuracy scoreboard](accuracy-scoreboard.md), the reported tempo was 0.94% high while the tracked beats themselves were right; bars drifted 6 beats by the end and only 8.6% of them started on a real downbeat. With the beat grid the drift is 0.02 beats and every bar starts on a downbeat.

## How the grid is built

1. **Tracked beats.** The analysis beat times are checked (finite, increasing, between 20 and 300 beats per minute). Where the tracker skipped beats, evenly spaced beats are filled in.
2. **Steady or following.** If one straight line explains the beats (residual spread at most 30 ms), the grid is that line; this removes tracker jitter from a recording made to a click. Otherwise each grid beat is a local straight-line fit over the surrounding beats, so the grid follows a tempo that moves.
3. **Settle on the played beat.** Beat trackers tend to report beats a little late. When at least 16 detected drum onsets fall near grid beats, the grid moves by their median offset (at most a fifth of a beat). Off-beat onsets are ignored, so swung eighths do not pull the grid.
4. **Find beat 1.** In 4/4, snare-like hits on beats 2 and 4 and kick-like hits favouring beat 1 identify the first beat of the bar. If the drums do not settle it, analysis downbeats are used only when the analysis meter confidence is at least 0.5 and at least 80% of them agree. Otherwise the grid starts on the first tracked beat.
5. **Start of the recording.** A first downbeat within half a beat of 0 s starts bar 1. A later first downbeat means the recording opens with a pickup; the pickup then gets its own first bar, shown from 0 s.

Beats before the first tracked beat and after the last are extrapolated at the local tempo.

## What a score records

Score documents use schema 6. `timing.beatGrid` is `null` when too few beats were tracked (the score then uses one constant tempo and warns about it), or:

| Field | Meaning |
| --- | --- |
| `version` | `beat-grid-v1` |
| `mode` | `steady` or `following` |
| `barPhaseSource` | `drums`, `analysis`, or `none` |
| `barPhaseConfidence` | 0 to 1, or `null` when the source is `none` |
| `onsetShiftSeconds` | how far detected onsets moved the grid |
| `trackedBeats` | beats the analysis supplied |
| `firstBeatSeconds` | time of beat 1 of bar 1; negative when a pickup bar begins before the recording |

`timing.tempoBpm` is the grid's tempo. Each measure's `startSeconds` and `endSeconds` are the real times of its bar lines, so bars may differ in length. Note and hit `startSeconds`/`rawTimeSeconds` are the detected times and are never changed; `startBeat`/`rawBeat` are positions on the grid and can be slightly negative for sound just before bar 1.

Scores built by `score-pipeline-v5` or earlier stay readable and downloadable and are flagged for rebuild.

## Warnings a musician sees

- the tempo moves and bars follow the tracked beats;
- beat 1 could not be confirmed and bars start on the first tracked beat;
- beat 1 comes from the audio analysis alone;
- too few beats were tracked, so bars use one constant tempo.

## Limits

- Bars are only as good as the beat tracking. Half-tempo or double-tempo tracking, or a lost beat in a free passage, gives wrong bars there.
- The backbeat rule covers 4/4 with a recognisable kick and snare. Four-on-the-floor without a backbeat, other meters, and music without drums fall back to the analysis or to "unconfirmed".
- Meter is one value for the whole recording; changes of meter are not followed.
- MIDI export carries one tempo, so it plays in notated time rather than following the recording.
