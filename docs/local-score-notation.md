# Local score notation

Implementation and licensing record for issue #151 / PR #152. The notation
view reuses saved exports; it does not construct or infer music.

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

## Verified redistribution inventory

The unmodified npm 2.2.1 browser asset is 1,420,444 bytes, SHA-256
`224ee5bee61d31ff3d07ade025f931a6cecd394cb54d2e39b1e7af2f0c7f1f5d`.
The npm archive integrity is recorded in the adjacent vendor manifest. The
official GitHub browser asset is 1,420,498 bytes, SHA-256
`537ed08f9c082ce0d785d21b80485b8d443cfb92ddc7a4ba4977804f09408cf4`;
the only difference is its source-map URL trailer. The npm asset is used so the
browser does not request an absent source map. Neither asset was minified or
rebuilt by PopEx. Git attributes disable newline normalization for this file.

The official release source map identifies the runtime modules. Its JSZip,
loglevel and typescript-collections sources were compared with exact npm package
files. JSZip's browserified submodules were compared with their package sources
(immediate and setimmediate include only the Browserify global wrapper). Every
one of pako's 16 modules matched 1.0.5 exactly, not an assumed latest version.
VexFlow is pinned 1.2.93 by OSMD; OSMD's prebuild copies its tagged VexFlowPatch
over that source. The embedded glyph source matches that tagged patch exactly.

| Component actually redistributed | Version | Applicable license |
| --- | --- | --- |
| OpenSheetMusicDisplay | 2.2.1 | BSD-3-Clause, Copyright 2019 PhonicScore |
| VexFlow with OSMD's tagged patch | 1.2.93 / OSMD 2.2.1 | MIT / BSD-3-Clause |
| loglevel | 1.9.2 matching source | MIT |
| typescript-collections | 1.3.3 matching source | MIT |
| JSZip browser distribution | 3.10.1 | MIT option explicitly selected, not GPL |
| lie within JSZip | 3.3.0 | MIT |
| immediate within JSZip | 3.0.6 | MIT |
| pako within JSZip | 1.0.5 | MIT plus zlib notices on ported zlib files |
| setimmediate within JSZip | 1.0.5 | MIT |
| VexFlow-18/Gonville glyph outlines | tagged VexFlow/OSMD glyph data | Gonville font files unrestricted; additions retain MIT/BSD notices |

JSZip's Node stream dependency resolves to an empty browser stub, not the Node
readable-stream package and its transitive dependencies. Type declarations and
OSMD development tools are not included. No demo/test recordings, standalone
web fonts, Bravura/SMuFL assets, model weights or datasets are bundled.

The upstream min.js points to a LICENSE.txt absent from the release/npm files.
PopEx supplies that exact filename with complete BSD/MIT/zlib notices copied
from the verified packages and source files, plus Gonville attribution and its
author's unrestricted font permission. Commercial use and redistribution are
permitted subject to preserving those notices and not implying endorsement.
These components do not inherit PopEx's MIT license. No paid infrastructure is
required. Self-hosting the bundle is chosen over a CDN to retain offline use and
avoid sending a visitor's address or private score to a rendering host.

Font source/permission: https://www.chiark.greenend.org.uk/~sgtatham/gonville/
Runtime sources: exact npm packages listed above, OSMD 2.2.1 tagged source/map,
VexFlow 1.2.93 source and JSZip v3.10.1 distribution. All checked 2026-10-16.

## Intended workflow

Open notation from one recording's saved score. Choose full score or an existing
individual part, corrected or original, then explicitly render its existing
MusicXML locally. Keep structured review/corrections and downloads available.
No score construction, inference or private recording upload to a service occurs.
No runtime CDN, account, telemetry, paid API or model download is permitted.

The view renders a selected window of 1–128 bars, with the exact range shown;
change the range to read the rest of a long score. Downloads always contain the
whole score/part. Browser XML reads are capped at 8 MiB and 20,000 notes; larger
scores keep working through downloads and the existing structured review.
The MusicXML is parsed as a Document, never passed to a URL-loading code path.
DTD/entity declarations are refused. Same-origin fetches reject redirects;
the page's CSP refuses remote scripts/connections and framing. Renderer scripts
are loaded only by the explicit notation page, not the ordinary app view.

Saved export responses add X-PopEx-Score-Created-At, X-PopEx-Correction-Revision
and X-PopEx-Score-View headers from the actual exported snapshot. The view checks
these against loaded metadata and refuses a render if another window changed
the score in between, rather than labelling a different revision. Download
semantics and files are unchanged; responses are no-store. Missing parts/ranges
are reported, not silently replaced. A failed render retains any previous visual
and all original saved files/corrections. Reload then Render to see new edits.

The notation view must show honest warnings, preserve original artifacts and
offer retry on rendering failure. The SVG is a visual aid, not a replacement for
the structured accessible review or a publication-quality engraving guarantee.

## Recording review beside notation

After a successful Render, the native audio player uses only that recording's
existing normalized WAV preview. Loading, rendering, changing selectors and
refreshing information never start playback. Play shown bars seeks to the first
displayed bar and stops at the end of the last. Choose a bar and Play selected
bar to hear it alone; Repeat shown bars repeats only a range started by Play
shown bars. Stop pauses. Native controls can seek elsewhere in the recording;
the position cue reports when it is outside the displayed bars or in a gap.

The strict optional `includeMeasureTiming=true` saved-score query returns only
measure index/start/end from the same saved corrected/original document. It does
not add notes to the default payload or estimate a new uniform timing grid.
Negative pickup starts are clamped to zero for playback. Timing is copied and
bound to the successful rendered snapshot, not mutable controls or a later
metadata refresh. A new successful render pauses/rebinds playback; a failed
render retains the previous visual and its timing. No source or correction is
written. Missing audio never prevents notation or downloads.

The current bar is shown as text; the already-bundled renderer's public cursor
API also shades that bar when available. Cursor iteration is bounded and does
not automatically scroll or animate the page. Cursor failure is cosmetic and
cannot stop the recording. Audio failures and rejected Play requests are
retryable, and Stop/new render supersede pending playback requests.

This is estimated draft synchronization, not verified alignment or sample-accurate
stopping. Browser media events/timers and background-tab throttling can affect
boundary precision. The view plays the normalized full recording, not generated
MIDI or a selected part's stem. Existing structured review still offers stem
selection and corrections. No new dependency, vendor file, font or license was
added for this playback layer.
