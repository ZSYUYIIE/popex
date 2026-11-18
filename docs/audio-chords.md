# Chords heard in the audio

PopEx names chord symbols by listening to the recording (`audio-chords-v1`, [`app/audio_chords.py`](../app/audio_chords.py)), not only from the notes its pitch tracker found.

## Why

The first harmony pass names a chord from the transcribed notes inside a time window. A single-line pitch tracker finds few of the notes of a chord, so most of a real recording stayed without a symbol. On the first real recording measured with the [accuracy scoreboard](accuracy-scoreboard.md), chord symbols appeared on 2 of 162 bars and the root was right for 1.5% of the reference chord time. Listening to the audio puts a symbol on 148 of 160 bars with the root right for 87.9% of that time.

## How it works

1. **Beats.** The same [beat grid](beat-grid.md) the score uses supplies the beat times. If too few beats were tracked, beats come from the analysis tempo.
2. **Sound.** Two sources are used when available:
   - the separated accompaniment and bass stems, which are free of drums and voice;
   - the harmonic part of the full mix.

   Where the stems are loud enough they decide; where they fall quiet (a verse carried by voice and drums) the full mix takes over. Without separated stems the full mix is used throughout.
3. **Profile per beat.** Each beat gets a pitch-class profile for the chord tones and one for the bass register.
4. **Compare.** Each beat is compared with major and minor triads on all twelve roots and with "no chord". The bass supports the root. Triads inside the recording's overall scale get a small preference, which settles major against minor where the third is weak.
5. **No chord.** A stretch that is silent, or mostly percussive over about two bars (a drum-only intro), is "no chord".
6. **One path.** A single path through the beats is chosen that prefers to change chord on bar lines, then on half bars.
7. **Confidence.** Each segment carries a confidence from how clearly its chord beat the best rival root. A segment below 0.4 is left unnamed.

Recognition is deterministic and runs locally with the libraries PopEx already uses. No model is downloaded.

## What is deliberately not claimed

- Only major, minor, or no chord. Sevenths, suspensions, added notes, diminished and augmented chords, and slash basses are not named.
- A chord in a passage without accompaniment is implied by the voice and whatever else sounds. It is usually right and less certain; its confidence is lower.
- The overall scale preference can pull a borrowed chord toward a chord inside the scale.

## Where it runs and what is saved

Recognition is the last step of the harmony stage. It is best effort: if the audio or a tempo is unavailable, or anything fails, nothing is saved, the harmony stage still completes, and the score uses chords named from transcribed notes as before.

The result is saved beside the job as `chords/audio-chords.<id>.json` (`popex-audio-chords`, schema 1), tied to the harmony result it was made with. The two newest results are kept, so a harmony run that is replaced keeps its chords until the new one is confirmed.

| Field | Meaning |
| --- | --- |
| `source` | `stems+mix`, `stems`, or `mix` |
| `scaleTonic` | tonic of the major-scale pitch set that holds the most energy |
| `segments[].startSeconds`, `endSeconds`, `beats` | where the segment sits |
| `segments[].symbol` | `C`, `Am`, …, or `null` |
| `segments[].confidence` | 0.4 to 1 for a named chord, otherwise `null` |
| `segments[].noChord` | `true` for a silent or drum-only stretch |
| `segments[].stemShare` | how much of the decision came from the separated stems (0 to 1) |

## In the score

When a saved result matches the job's current harmony run and analysis, the score maps its segments onto bars by time. A bar shows a symbol when one chord covers at least half of it and no other covers a quarter, as before; a bar split evenly between two chords shows none. The chord layer note says the symbols were heard in the audio and from which source. Re-run **Harmonize**, then rebuild the score, to add them to an existing recording.

## Limits

- Tuned and measured on one real recording plus synthetic progressions. More recordings are needed before the 87.9% figure can be read as typical.
- One symbol per bar is shown, so faster chord changes are not yet on the page.
- Wrong beats or bar lines move the chord changes with them.
