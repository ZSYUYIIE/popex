# PopEx — Project Status and Next Steps

Date: 2026-09-10. Base: `main@0498e50` ("Close Cycle 6 harmony cross-layer
audit blockers (#103)", CI run `33139522686` green). The canonical product
definition is the **Product source of truth** section in `README.md`; this
document only summarizes status and plans work. It must not redefine the
product.

## 1. What PopEx is

PopEx is a free, open-source (MIT), local-first music-transcription workspace
for intermediate musicians. It converts a **specific recording or
arrangement** of a pop song into an editable draft score with standard
notation, guitar and bass tablature, drum/percussion notation, chord symbols,
and recording-specific parts. The MVP is version-specific sheet-music
generation; a chord-only play-along interface is later scope, not the MVP.
The core must never require a paid API or hosted service. Different recording
versions are separate arrangements with separate analysis and scores.

## 2. Product invariants (must hold for every change)

- MIT-licensed, free and open source; local-first core with no paid/hosted dependency.
- MVP: standard notation, drum/percussion notation, guitar and bass tablature, chord symbols.
- Separate arrangements stay separate (analysis and scores per recording version).
- Architecture supports pitched notes, percussion events, chords, parts, tabs, scores, revisions.
- Raw model predictions and user corrections stay separate.
- Dense/uncertain arrangements get honest reductions and warnings, never fabricated precision.
- Tonal schemas stay extensible beyond major/minor.
- Private inputs stay private (personal-use phase).
- Public-library, publishing, rights, moderation, and payment systems are deferred.

## 3. Architecture (as built)

Separation of concerns enforced throughout: domain data vs presentation
state; raw events kept before quantization/score cleanup; pitched-note and
percussion-event representations distinct; score construction separate from
inference; tablature fingering separate from pitch transcription; original
predictions preserved when users edit; earlier artifacts preserved when a
later stage fails; expensive stages retryable/idempotent; exact model and
analysis versions stored; no arbitrary local filesystem paths through the API.

## 4. Canonical-order progress

| # | Canonical step | State |
|---|---|---|
| 1 | Audio ingest, source separation, beat tracking | Done (analysis + Demucs stems) |
| 2 | Raw pitched-note / percussion-event transcription | Done (`baseline-pyin-onset-v1`, schema 1; pitched + percussion + advisory beat alignment) |
| 3 | Baseline chord-event extraction for score symbols | Done as evidence-aware harmonic context (`pitch-class-window-v1`, schema 1; conservative chord candidates with unresolved accounting) |
| 4 | Measure, rhythm, MIDI, MusicXML score construction | **In progress** — pure builder + read-only preview/downloads done locally, unpushed (issues #109, #110) |
| 5 | Drum/percussion notation | Not started |
| 6 | Guitar/bass tablature generation | Not started |
| 7 | Synchronized score review and correction | Partial (transcription/interpretation/harmony review panels exist; score review not started) |
| 8 | Revision history within a recording arrangement | Not started |

Later scope (not MVP): chord-only play-along view, public library/publishing/rights/moderation/payments.

## 5. Pipeline and artifact inventory (all in `main`)

- Analysis: `baseline-librosa-v1` → `data/exports/{job}/analysis/audio-analysis.json`
  (tempo, beats, tentative downbeats/meter, tonal centre + candidates, chroma, tuning, warnings).
- Stems (optional, worker-isolated Demucs `4.1.0`): schema-3 manifest
  `data/exports/{job}/stems/stem-separation.json` (vocals/bass/drums/other).
- Raw transcription: `baseline-pyin-onset-v1`, schema 1
  (`transcription/raw-events.json`; separate pitched/percussion events).
- Interpretation: `editable-interpretation-v1`, draft schema 1 (pitched
  parts/phrases, `conservative-grid-v1` rhythm, `broad-drum-structure-v1` drums).
- Harmony: `pitch-class-window-v1`, artifact schema 1, with attempt-scoped
  nonce-bound publication, previous-result preservation, orphan reconciliation
  (Cycle 6 closure, PR #103).
- Score (local branches only): `score-construction-v1`, schema 1 —
  stdlib-only measures + SMF type-0 MIDI + partwise MusicXML, explicit 8th-note
  quantization with shift/confidence warnings; chord symbols deferred honestly.

Job lifecycle states (independent per stage with atomic one-winner claims):
`preparation`, `analysis`, `separation`, `transcription`, `interpretation`,
`harmony` (`not_started`/`processing`/`completed`/`failed`, retryable,
failures preserve prior success).

## 6. API inventory (`main` + local score branch)

`GET /api/health` · `POST /api/jobs` · `POST /api/uploads` · `GET /api/jobs` ·
`GET /api/jobs/{id}` · analyze → `/analysis`, `/analysis/download` ·
separate → `/stems`, `/stems/{kind}/preview|download` ·
transcribe → `/transcription`, `/transcription/download` ·
interpret → `/interpretation`, `/interpretation/download` ·
harmonize → `/harmony`, `/harmony/download` ·
**score branch adds:** `GET /api/jobs/{id}/score` (`?includeMeasures`) and
`GET /api/jobs/{id}/score/download?format=midi|musicxml` (read-only, built on
demand from published transcription + analysis tempo; 404 when evidence is
missing, 422 on bad format, 500 on corrupt evidence).

## 7. Open GitHub state (2026-09-10)

- #102 (integration, Cycle 6 races) — OPEN but work merged via PR #103; needs verification comment + close.
- #94/#95/#96 (Cycle 6 audits), #104/#105/#106 (Cycle 6 reviews) — OPEN, pre-merge artifacts; orchestrator to close or carry.
- #31/#32 (Linux/Windows real-model separation proofs), #42 (UI system) — OPEN, deferred.
- #107 (Windows-only harmony rollback message expectation) — fix ready locally.
- #108 (README status reconciliation) — docs ready locally.
- #109 (score vertical slice) / #110 (score persistence + API) — spike + read-only API ready locally.

## 8. Local unpushed work (push tool-denied in this workspace)

| Branch | Commits over `main@0498e50` | Validation |
|---|---|---|
| `agent/cycle6-windows-expectation-fix` | `901d0da` test-expectation fix | Harmony suites green on Windows; `compileall`, `node --check` pass |
| `agent/docs-reconcile-implementation-status` | `f382039` README status/workflow/API/reliability/limitations/next-cycle | `compileall`, `node --check` pass; product truth untouched |
| `agent/score-construction-slice` | `9be8829` stdlib score builder; `72e9b2b` honesty hardening; `d66b4b6` read-only score API + 8 API tests | 16 score tests + harmony/transcription/interpretation/`test_app` suites pass; `compileall`, `node --check` pass |
| `agent/project-status-summary` | this document | n/a |

Full `pytest` on Windows shows unrelated platform failures (`resource` module
missing, symlink tests); Linux CI is canonical and green on `main`.

## 9. Next steps (ordered)

1. **Publish backlog** (needs push rights): push the 4 branches above, open
   draft PRs vs `main` with `popex-agent-handoff:v1` blocks, CI, review.
2. **Merge in order**: #107 → #108 → score slice; resolve the one-line README
   overlap between #108 and the score branch at merge time (keep #108 text,
   add the score-preview sentence).
3. **Close stale Cycle 6 bookkeeping**: verify #102 acceptance on `main`,
   comment with evidence, close #102; orchestrator dispositions #94–#96,
   #104–#106.
4. **Finish issue #110**: persisted versioned score documents, `score_status`
   lifecycle with atomic publication + previous-result preservation (mirror the
   harmony attempt/nonce pattern), score/MIDI/MusicXML artifact files,
   failure-preservation + stale-retry tests.
5. **Score review UI**: read-only draft rendering + warnings panel, keyboard
   accessible, non-colour status indicators, near the content it affects.
6. **Drum/percussion notation slice** (canonical step 5), reusing the
   percussion-event + drum-structure evidence already stored.
7. **Guitar/bass tablature slice** (canonical step 6), fingering strictly
   separate from pitch transcription, synchronized with standard notation.
8. **Revision history within an arrangement** (canonical step 8).
9. **Deferred**: chord-only play-along, public library/publishing/rights/
   moderation/payments, PDF export (after MusicXML rendering).

## 10. Standing rules for the next agent

- Read `README.md` (product truth), `AGENTS.md`, `THIRD_PARTY_NOTICES.md`
  before adding any dependency/model/dataset/font/asset; `music21`,
  `pretty_midi`, renderers stay candidate-only until version-specific review.
- One narrow, independently testable cycle at a time; no speculative modules,
  fake data, or unrelated reformatting.
- GitHub-first: one issue per agent, draft PR as first checkpoint, full
  handoff in PR body (base/head SHAs, files, validation counts, CI, risks,
  merge recommendation); keep PRs unmerged unless assigned merge ownership.
- Baseline validation: `pytest`, `python -m compileall -q app tests`,
  `node --check app/static/app.js`, plus an end-to-end local smoke test for
  media/analysis changes. Synthetic or redistributable audio only in tests.
