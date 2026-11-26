# Bar grid: bars follow the tracked beats

A draft score needs bar lines that sit where a musician counts them. PopEx places them with a beat grid (`beat-grid-v1`, [`app/beat_grid.py`](../app/beat_grid.py)) built from the beats the audio analysis tracked.

## Why not one tempo

Earlier scores used a single tempo for the whole recording and started bar 1 at 0 s. A tempo estimate that is 1% off moves the bar lines by 4 beats over 400 beats, a whole bar in 4/4, and a recording whose tempo moves never fits. On the first real recording measured with the [accuracy scoreboard](accuracy-scoreboard.md), the reported tempo was 0.94% high while the tracked beats themselves were right; bars drifted 6 beats by the end and only 8.6% of them started on a real downbeat. With the beat grid the drift is 0.02 beats and every bar starts on a downbeat.

## How the grid is built

0. **Beats per bar.** The drums and bass decide when they clearly repeat every three or four beats. For each of 3 and 4, and for each of kick-like hits, snare-like hits, and bass-line note onsets (the bass stem, or the full mix's lowest line) with at least 8 hits near a beat, the accent contrast is how far the busiest beat position stands out from the average position, (busiest − average) / busiest. A waltz kick on every third beat scores high for 3 and zero for 4; a rock kick on 1 and 3 with snare on 2 and 4 scores for 4 and not for 3; a kick on every beat says nothing. The contrasts are summed, and a meter is chosen when it scores at least 0.4 and beats the other by at least 0.25 (`meterSource: drums-and-bass`). Otherwise the analysis meter is used, and without one the score falls back to 4/4 and says so. The harmony stage uses the same decision, so chords and bars agree.
1. **Tracked beats.** The analysis beat times are checked (finite, increasing, between 20 and 300 beats per minute). Where the tracker skipped beats, evenly spaced beats are filled in.
2. **Steady or following.** If one straight line explains the beats (residual spread at most 30 ms), the grid is that line; this removes tracker jitter from a recording made to a click. Otherwise each grid beat is a local straight-line fit over the surrounding beats, so the grid follows a tempo that moves.
3. **Settle on the played beat.** Beat trackers tend to report beats a little late. When at least 16 detected drum onsets fall near grid beats, the grid moves by their median offset (at most a fifth of a beat). Off-beat onsets are ignored, so swung eighths do not pull the grid.
4. **Find beat 1.** In 4/4, snare-like hits on beats 2 and 4 and kick-like hits favouring beat 1 identify the first beat of the bar. In other meters, the beat position with the most kick-like hits, and fewest snare-like hits, is beat 1. If the drums do not settle it, analysis downbeats are used only when the analysis meter confidence is at least 0.5 and at least 80% of them agree. Otherwise the grid starts on the first tracked beat.
5. **Start of the recording.** A first downbeat within half a beat of 0 s starts bar 1. A later first downbeat means the recording opens with a pickup; the pickup then gets its own first bar, shown from 0 s.

Beats before the first tracked beat and after the last are extrapolated at the local tempo.

## What a score records

Score documents use schema 6. `timing.beatGrid` is `null` when too few beats were tracked (the score then uses one constant tempo and warns about it), or:

| Field | Meaning |
| --- | --- |
| `version` | `beat-grid-v2` (`beat-grid-v1` before beats per bar came from the drums and bass) |
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
- The backbeat rule covers 4/4 with a recognisable kick and snare, and the kick accent covers 3/4. Four-on-the-floor without a backbeat and music without drums fall back to the analysis or to "unconfirmed".
- Only 3 and 4 beats per bar are inferred from the drums and bass. A compound meter such as 6/8, tracked at the eighth note, is written as bars of three eighths when the analysis finds groups of three; 5/4, 7/8 and 12/8 are not recognised.
- Meter is one value for the whole recording; changes of meter are not followed.
- MIDI export carries one tempo, so it plays in notated time rather than following the recording.
