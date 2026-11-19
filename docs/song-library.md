# Song library

The library keeps every version of a song together: the studio take, a live recording, an acoustic session, a cover. Each version keeps its own analysis, draft score, sections and corrections. The library groups and compares them and never merges one version's parts into another.

Everything stays on this computer. Filing reads only the recording's own title or file name and your existing song titles; nothing is looked up online.

## Adding a recording

While adding a recording (file or link), the optional **Add it to a song in your library** panel takes the song, artist, version and label up front; the new recording is filed as soon as it is created, using the same matching as below.

An ungrouped recording shows **Add to your library**. Its fields start with a suggestion taken from the recording's name:

| Name | Song | Artist or credits | Version | Label |
| --- | --- | --- | --- | --- |
| `Coldplay - Yellow (Live in Buenos Aires).mp3` | Yellow | Coldplay | Live | Live in Buenos Aires |
| `Yellow - Acoustic` | Yellow | | Acoustic | Acoustic |
| `Taylor Swift - Shake It Off (Official Video)` | Shake It Off | Taylor Swift | | |
| `shake-it-off.mp3` | Shake It Off | | | |

Suggestions drop common noise such as "(Official Video)", "(Lyrics)" or a download site's name, read "Artist - Title", recognise live, acoustic, unplugged, remix, cover, concert and radio-edit versions, and tidy file-style names. They are only a starting point: check them before adding.

When an existing song's title matches closely, the form says so. Matching ignores case, spacing, punctuation and a leading "The", "A" or "An", and tolerates a small typo.

**Add to library** reuses the song with the same title (with its first arrangement) or creates a new song with a "Main arrangement". To move a version to another arrangement or song later, use **Recording version** on that recording.

The library also lists recordings that are **Not in a song yet**, each with a link that opens its form.

## Library rows

Each version shows:

- its kind and label, for example "Live · Live in Buenos Aires";
- length, tempo, key and bars;
- its section sequence, for example `sections A B C D E C …`;
- **Open notation**, **MusicXML** and **MIDI** once a draft score is saved, or "no draft yet".

The key comes from the draft score's tonal context when the score was confident enough to write a key signature. Otherwise the audio analysis's major/minor estimate is shown, marked "rough estimate".

**Find a song or version** filters by song title, credits, arrangement, version title, label and kind. Every word must match, so "live owls" finds the live versions of songs by Night Owls.

**Compare versions** places each version's estimates side by side: tempo, tonal context, bars, chords, chords only in that version, drum hits, tab notes, corrections and the section sequence.

## API

- `GET /api/library`: songs, arrangements and versions. Each version carries `durationSeconds`, `tempoBpm`, `keySymbol`, `keySource` (`score`, `analysis` or `null`), `sectionForm`, `bars`, `notationUrl` and `downloadUrls` (the last three only when a score is saved). Each entry in `unassigned` also carries `suggestion` (`title`, `credits`, `kind`, `label`, `matchingSongId`) and `fileUrl`.
- `POST /api/jobs/{job_id}/library` with strict JSON `{"title": "...", "credits": null, "kind": null, "label": null}`: file the recording in one step. Returns `songId`, `arrangementId`, `createdSong`, the updated `job` and the `library`. 404 for an unknown recording, 422 for invalid text or an unknown kind.

The score's key name and section sequence are copied to the recording when its score is saved, so the library does not read every score file. Scores saved before this show the analysis key until they are drafted again.

## Limits

- Suggestions only parse names; a recording called "track01" gets "Track01" as its song.
- There is no shared or public library yet. Publishing, rights and moderation remain deferred.
