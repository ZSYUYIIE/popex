# Note detection with Basic Pitch

When a recording has separated stems, PopEx detects the notes on the vocal, bass and accompaniment stems with **Basic Pitch** (`basic-pitch-icassp2022-v1`), a small neural network from Spotify that hears several notes at once. The full mix, used when stems are not available, still uses the pYIN pitch tracker.

## Why

pYIN follows one pitch at a time. On the first real recording measured with the [accuracy scoreboard](accuracy-scoreboard.md) it found a bass note in only a third of the bars and turned every accompaniment chord into a single note. With Basic Pitch on the same stems:

| Indicator (same recording, same reference chords) | pYIN | Basic Pitch |
| --- | --- | --- |
| Bass notes per bar | 0.54 | 2.02 |
| Bars with any bass note | 33.7% | 71.3% |
| Bass notes on the reference chord root | 59.7% | 76.9% |
| Vocal notes flagged low confidence | 50.8% | 30.9% |
| Vocal notes inside the key | 89.0% | 92.6% |
| Accompaniment | one note at a time | chords kept |

These are indicators, not note accuracy: the reference for that recording has chords but no notes. They show the bass line now follows the harmony and the accompaniment keeps its chords.

## How it runs

- The model file is bundled (`app/models/basic_pitch/nmp.onnx`, 230 KB) and checked against its SHA-256 before use. It runs on the CPU with ONNX Runtime. Nothing is downloaded.
- Audio is read at 22,050 Hz and processed in overlapping two-second windows, exactly as in Basic Pitch 0.4.0. The decoding of notes from the model output is a port of Basic Pitch's own code and was checked to reproduce its note events exactly on a real stem.
- Per stem:

  | Stem | Range | Shortest note | Result |
  | --- | --- | --- | --- |
  | Bass | 30 to 400 Hz | 80 ms | one line: where notes overlap, the strongest is kept |
  | Vocals | 80 to 1200 Hz | 100 ms | one line |
  | Accompaniment | 60 to 2100 Hz | 80 ms | chords kept; at most 2400 notes, the strongest |

- Each note's confidence is the model's average activation over the note. Notes under 0.45 carry a weak-activation warning.
- If ONNX Runtime or the model is unavailable, or the model fails on a stem, that stem uses pYIN and the transcription says so.

## Limits

- Basic Pitch was trained on solo and small-ensemble recordings and synthesised multitracks. Separated pop stems contain bleed and artefacts, so expect extra or missing notes, especially in the accompaniment.
- One-line reduction keeps the strongest note; a backing vocal louder than the lead can win.
- Drums are not transcribed by this model.
- The full mix still uses pYIN, which finds only a dominant melody.

## Licence

Code and weights: Apache License 2.0, Copyright 2022 Spotify AB. The licence and NOTICE ship next to the model. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for training-data provenance.
