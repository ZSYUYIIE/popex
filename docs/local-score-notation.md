# Local score notation

Implementation checkpoint for issue #151. No renderer is bundled at this
checkpoint and no notation-view capability is claimed yet.

## Selection and approval

The human approved local OpenSheetMusicDisplay 2.2.1 integration on 2026-10-16,
conditional on verifying bundled dependencies/fonts and including all notices.
The official GitHub release API and npm registry agree on version 2.2.1, released
2026-10-12, with declared BSD-3-Clause licensing. This is third-party software,
not PopEx MIT code. No model weights or training dataset are involved.

Primary evidence:

- https://github.com/opensheetmusicdisplay/opensheetmusicdisplay/releases/tag/2.2.1
- https://github.com/opensheetmusicdisplay/opensheetmusicdisplay/blob/2.2.1/LICENSE
- https://github.com/opensheetmusicdisplay/opensheetmusicdisplay/blob/2.2.1/package.json

The official minified browser release is 1,420,498 bytes. Its hash, exact bundled
runtime dependency versions and all embedded font/glyph licenses still need
verification before any redistribution. A roadmap mention alone is not approval.

## Intended workflow

Open notation from one recording's saved score. Choose full score or an existing
individual part, corrected or original, then explicitly render its existing
MusicXML locally. Keep structured review/corrections and downloads available.
No score construction, inference or private recording upload to a service occurs.
No runtime CDN, account, telemetry, paid API or model download is permitted.

The notation view must show honest warnings, preserve original artifacts and
offer retry on rendering failure. The SVG is a visual aid, not a replacement for
the structured accessible review or a publication-quality engraving guarantee.
