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

- Without separated stems, the model hears the full mix once (30 to 2100 Hz, shortest note 80 ms) and its notes are split by register into three estimated lines, each labelled as a line of the mix, never as a stem:

  | Line | Taken from | Rule |
  | --- | --- | --- |
  | Lowest line (`mix_bass`) | notes at or below E3 | where notes overlap, the lowest is kept: the loudest low note is often the bass's octave overtone |
  | Top line (`mix_melody`) | remaining notes at or above G3 | where notes overlap, the highest is kept |
  | Chords (`mix_chords`) | every other note | kept as heard; nothing detected is dropped |

  Both single lines ignore notes shorter than 100 ms and weak notes that start at least 20 ms inside a note more than 1.67 times as strong (usually its overtone or undertone); those stay in the chords line. Two notes clash only when they overlap by more than a quarter of the shorter one; a smaller legato overlap keeps both, and the earlier note ends where the next begins. The score writes the top line as **Melody (top line of the full mix, draft)**, the lowest line on the **Bass line** staff, and the chords line in the **Accompaniment reduction** on a grand staff. The key-finding step uses the lowest line as its bass evidence.

  On the synthetic test song (melody, bass, chords and drums mixed, with known notes) this found the melody with recall 0.98 and precision 0.86 and the bass with recall 0.70 and precision 0.68. pYIN on the same mix had written 30 notes that followed the bass and no melody.
- Each note's confidence is the model's average activation over the note. Notes under 0.45 carry a weak-activation warning.
- If ONNX Runtime or the model is unavailable, or the model fails on a stem or the full mix, that source uses pYIN and the transcription says so.

## Limits

- Basic Pitch was trained on solo and small-ensemble recordings and synthesised multitracks. Separated pop stems contain bleed and artefacts, so expect extra or missing notes, especially in the accompaniment.
- One-line reduction keeps the strongest note; a backing vocal louder than the lead can win.
- Drums are not transcribed by this model.
- The full-mix lines are estimates by register. The top line can follow a harmony part or a chord's top note; the lowest line can pick up low chord notes or a ghost note below the bass; instruments are not identified. Separated stems give better lines when they are available.

## Licence

Code and weights: Apache License 2.0, Copyright 2022 Spotify AB. The licence and NOTICE ship next to the model. See [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) for training-data provenance.
