# Rhythm grids

A draft score writes each pitched source line (lead vocal, bass, accompaniment, full-mix melody) on its own rhythm grid. Eighth notes are the default. A line moves to a sixteenth-note grid only when its note onsets clearly show sixteenths.

## The rule

For each line, PopEx looks at the onsets of its confident notes (confidence at least 0.50) and counts how many sit on an off-eighth sixteenth, the "e" or "a" of a beat:

- an onset counts when it is within **30 ms** of that position;
- positions are measured on the score's own bar grid, which follows the tracked beats, so a recording whose tempo moves is judged against its own beats rather than one constant tempo.

The line gets a sixteenth-note grid when all of these hold:

- at least **3** confident onsets count;
- they are at least **15%** of the line's confident onsets;
- a sixteenth lasts at least **90 ms** at the score's tempo (up to about 166 BPM). Faster than that, sixteenths are hard to read and easy to mis-detect, so the line keeps eighths and the score says the tempo was too fast. When the tempo moves, an onset only counts where the local sixteenth lasts 90 ms, so a fast passage cannot earn sixteenths that would be unreadable there.

Otherwise the line keeps the eighth-note grid, so a few stray early or late notes never turn a straight line into a busy one.

## What is saved

Each line's decision is saved with the score as `rhythmGrids`: the grid unit, the number of confident onsets, the number that sat on an "e" or "a", and the reason (`off-eighth-onsets`, `insufficient-evidence` or `tempo-too-fast`). The standard-notation layer note on the recording says which lines use sixteenths.

Before a score is saved, every predicted note is checked against its own line's grid. Musician corrections can still move any note in sixteenth steps, whatever its line's grid, and the original prediction is kept.

Drum notation is unchanged: broad voices on an eighth-note grid, with off-grid hits counted and warned about.

## Limits

- Triplets, swing and thirty-second notes are not detected. They are written on the nearest straight grid, and the score warns when notes shift by more than 50 ms.
- The decision is per line for the whole recording, not per section.
- Onset evidence comes from the note model, so a line whose notes are poorly detected gets an eighth-note grid.
