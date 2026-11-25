# Song sections

A draft score now marks where the song's sections change and which sections sound alike, so a musician can go straight to "the second time that full-band section comes round" instead of counting bars.

## How sections are found

Sections are measured on the score's own bar grid, so every section starts on a bar line.

For each bar, PopEx measures:

- how loud each separated stem is (vocals, drums, bass, accompaniment), when the recording has stems;
- the harmony of the mix (chroma);
- the timbre of the mix (MFCC).

A novelty curve over the bar-to-bar similarity matrix (a checkerboard kernel four bars wide) marks where the sound changes. A section is at least four bars long. Sections whose average sound is close enough share a letter, assigned in order of first appearance.

Each section also lists the stems clearly playing in it. A stem counts as playing when its level in that section is within 15 dB of its own loud level (90th percentile over the recording) and above -55 dBFS. The UI shows this as "drums only", "full band", "vocals, drums, accompaniment" and so on.

Method version: `bar-sections-v1`, in `app/song_sections.py`.

## What the letters mean

Letters mean "these bars sound alike". They are not verse or chorus names, and PopEx does not claim to know a song's form; the musician can name sections (see below). Two sections with the same chords can get different letters when the arrangement changes (for example the bass comes in), and a verse and chorus over one loop can share a letter when they sound alike.

Letters are per recording. An "A" in a live version and an "A" in the studio version are unrelated.

## Naming sections

Under the strip, **Name the sections** lists every section with a box for its name, such as "Intro", "Verse 1", "Chorus" or "Bridge". Names are musician corrections: they are saved in the score's correction log beside the detected letters, can be undone like any other correction, and never change the original prediction. Leave a box empty to clear a name.

When a letter repeats and only one of its sections gets a new name, **Give every section with the same letter the same name** (on by default) copies that name to the others. Turn it off to name repeats separately, for example "Chorus 1" and "Chorus 2".

Named sections show the name first and keep the letter: on the strip ("Chorus, D · 29-44"), on the notation page's section buttons ("Chorus (D)"), and as the rehearsal mark text in the corrected MusicXML. The original-prediction download keeps the letters. The in-app notation view draws the marks too, including on the first bar of the range you render.

Correction: `{"op": "set_section_name", "target": {"startMeasure": 28}, "name": "Chorus"}` (name 1 to 40 characters on one line, or `null` to clear). It does not apply when no section starts at that bar, for example after the score is drafted again with different sections.

## When there are no sections

The score carries a note instead of sections when:

- the score has fewer than 16 bars;
- the sound does not change enough between bars;
- the analysis audio cannot be read.

Finding sections never stops a score from being built.

## Where sections appear

- **Score document** (schema 7): `sections` with `version`, `note`, and `sections` (`startMeasure`, `endMeasure` exclusive, `label`, `boundaryStrength`, `heard`). Schema 1 to 6 scores stay readable without sections.
- **Saved-score API**: `sections` adds bar counts, start and end times in seconds, `heardText`, and an explanation of the letters.
- **Recording card**: a proportional strip under the draft; each section opens the notation at its bars.
- **Notation page**: "Go to a section" buttons render that section's bars. The page also accepts `?bars=17-32`.
- **MusicXML**: a boxed rehearsal mark with the section letter at the first bar of each section, in the first part.
- **Library comparison**: each version's section sequence, for example `A B C D E C`.

Scores built before sections are reported as out of date (reason `sections`) when they have at least 16 bars; drafting again adds sections.

## Limits

- Boundaries come from changes in sound. A section that changes only in lyrics, or a gradual build, may not get its own boundary.
- Without stems, sections rely on the mix alone and say nothing about which instruments play.
- Bars before the first tracked beat (a pickup) are measured from the start of the recording.
