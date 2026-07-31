# PopEx agent instructions

This file is the concise operating guide for people and AI agents working on PopEx.

The canonical product definition is the **Product source of truth** section in [README.md](README.md). Read it before planning, coding, reviewing, or merging work.

## Precedence

For repository work, use this order when project instructions conflict:

1. the user's latest explicit decision;
2. the Product source of truth in `README.md`;
3. this `AGENTS.md` workflow;
4. the current cycle or branch prompt;
5. pull-request descriptions and comments;
6. older branch documentation and code comments.

Do not silently resolve a material conflict by redefining the product. Report it and preserve the canonical goal.

## Product invariants

Every change must preserve these decisions unless the user explicitly changes them:

- PopEx is free and open source under the MIT License.
- The core product is local-first and must not require a paid API or hosted service.
- The MVP is version-specific sheet-music generation for intermediate musicians.
- The MVP includes standard notation, drum/percussion notation, guitar and bass tablature, and chord symbols.
- A chord-only play-along interface is later scope, not the MVP.
- Different recording versions remain separate arrangements with separate analysis and scores.
- The architecture must support pitched notes, percussion events, chords, parts, tabs, scores, and revisions.
- Raw model predictions and user corrections remain separate.
- Dense or uncertain arrangements must produce honest reductions and warnings rather than fabricated precision.
- Tonal schemas must remain extensible beyond major and minor.
- Private inputs remain private in the personal-use phase.
- Public-library, publishing, rights, moderation, and payment systems are deferred.

## Before starting work

1. Read `README.md`, especially Product source of truth, Current implementation status, Known limitations, and Next planned cycle.
2. Read `THIRD_PARTY_NOTICES.md` before adding any dependency, model, dataset, font, or asset.
3. Fetch and inspect current `main`.
4. Compare the working branch with current `main`.
5. Identify files shared with other active branches.
6. State the narrow goal, non-goals, acceptance criteria, and validation commands.
7. Reuse completed artifacts and modules instead of rebuilding them unnecessarily.

## Scope discipline

- Keep one cycle narrow and independently testable.
- Do not add speculative empty modules, disabled navigation, fake data, or placeholders for distant features.
- Do not redesign unrelated UI while implementing backend analysis.
- Do not reformat unrelated files or create large formatting-only diffs.
- Do not change the Product source of truth to make a branch appear compliant.
- Update implementation-status documentation when behaviour changes.
- Record material product changes only after explicit user approval.

## Data and architecture rules

- Separate domain data from presentation state.
- Version persisted schemas and analysis artifacts.
- Keep raw events before quantization and score cleanup.
- Keep pitched-note and percussion-event representations distinct.
- Keep score construction separate from inference.
- Keep tablature fingering separate from pitch transcription.
- Preserve original predictions when users edit results.
- Preserve successful earlier artifacts when a later stage fails.
- Make expensive stages retryable and idempotent where practical.
- Store exact model and analysis versions.
- Never expose arbitrary local filesystem paths through the API.

## Dependency and licensing rules

Before adding or redistributing a third-party component:

1. identify its exact version;
2. verify the source-code license;
3. verify model-weight and dataset licenses separately;
4. confirm local use and redistribution are allowed;
5. avoid mandatory paid or proprietary infrastructure;
6. update `THIRD_PARTY_NOTICES.md`;
7. include required license and copyright notices;
8. do not describe third-party material as MIT-licensed unless it is.

A roadmap mention is not approval to bundle a dependency.

## UI and accessibility rules

Apply the design principles recorded in `README.md`:

- use semantic and standard controls where possible;
- make core workflows keyboard accessible;
- provide visible focus and non-colour status indicators;
- keep controls near the content they affect;
- show honest progress and recovery actions;
- use musician-facing language rather than backend terminology;
- preserve source artifacts and corrections through failures;
- avoid modal interruptions for routine status;
- do not hide essential actions behind hover-only or context-menu-only interactions.

## Testing and validation

Every branch must run the checks relevant to its scope. The current baseline is:

```bash
pytest
python -m compileall -q app tests
node --check app/static/app.js
```

Add tests for new behaviour, migrations, retry paths, failure preservation, and backward compatibility.

Use synthetic or clearly redistributable audio in tests. Never commit copyrighted recordings, private user files, model caches, generated personal scores, or secrets.

For media and analysis changes, perform at least one end-to-end local smoke test in addition to unit tests.

## Pull-request and merge rules

- Rebase or merge current `main` before declaring a branch ready when shared files changed.
- Resolve shared frontend files deliberately; never choose one branch wholesale when both contain required behaviour.
- Keep PR descriptions aligned with README product language.
- Report known limitations and deferred work.
- Do not merge another agent's branch unless explicitly instructed.
- Do not bypass failing required checks.
- After merge, update the next agent's base and scope.

## Documentation ownership

The Product source of truth in `README.md` is canonical and should remain stable.

Ordinary feature agents may update:

- Current implementation status;
- workflow and API documentation;
- setup and configuration;
- tests and validation;
- known limitations;
- roadmap progress.

They must not alter the product purpose, MVP definition, licensing commitment, intended user, or deferred public direction without explicit approval.

When uncertain, preserve the canonical section and ask for a product decision.