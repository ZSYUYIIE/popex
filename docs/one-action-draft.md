# Draft the score in one action

A recording's score is built by several stages: separating instruments, detecting notes and drum hits, organising parts and rhythm, listening for chords, and writing the score. Each stage can still be started on its own, but a musician should not have to know the order. **Draft the score** runs them all.

## What happens

Adding a recording (a file or a link) starts drafting straight away. Drafting waits for source preparation and audio analysis, then runs:

| Step | Required | If it cannot run |
| --- | --- | --- |
| Separating vocals, bass, drums and accompaniment | no | The full mix is transcribed instead. Skipped when separation is not set up, or when its model has not been downloaded and you did not tick the box to allow it. |
| Detecting notes and drum hits | yes | Drafting stops; earlier results stay saved. |
| Organising parts and rhythm | no | Notes carry no part labels. |
| Listening for chords | no | The score has no chord symbols. |
| Writing the score | yes | Drafting stops; earlier results stay saved. |

The recording shows each step as it runs, with a mark for done, already done, skipped, or did not finish, and a short reason for anything skipped or unfinished.

## Reuse

A step that already finished is reused. When a step runs again, every step after it runs again too, because its results would otherwise describe old evidence. A saved score is also rebuilt when it was made from older results or by an older version of PopEx. Harmony saved before chords were heard in the audio is run again so the score gets those chords.

So pressing **Draft again** on a finished recording is cheap: it only redoes what is out of date.

## Model download

Stem separation uses a verified model that is downloaded once to a local cache. Drafting never downloads it unless you tick **Separate instruments first** next to the button. Your audio never leaves the computer.

## API

`POST /api/jobs/{job_id}/draft` with optional strict JSON `{"allowModelDownload": true | false}` returns 202 and the job. It returns 409 while the recording is already being drafted, while one of its stages is running, or when preparation or analysis failed.

Every job includes `draft`:

| Field | Meaning |
| --- | --- |
| `startUrl` | where to post |
| `canStart` | whether drafting can start now |
| `running` | whether a draft is waiting or running |
| `progress` | `null`, or `state` (`waiting`, `running`, `completed`, `failed`), `step`, `message`, `startedAt`, `finishedAt`, and `steps` with `key`, `label`, `outcome` (`ran`, `reused`, `skipped`, `failed` or `null`) and `note` |

## Limits

- Drafting progress is kept in memory. After a restart it shows as not running; every finished step stays saved, and drafting again picks up from there.
- Steps run one after another in the background of the local server. Two recordings can draft at the same time, but stem separation runs one recording at a time.
