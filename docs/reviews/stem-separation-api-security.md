# Stem separation API security and reliability review

## Status and reviewed revisions

This review is the durable security and reliability gate for issue #26.

Final reviewed state:

- merged base before the API merge: `main` at `ae505b2d703b66da2bb05b77061681f4f14d09be`;
- API assignment: issue #23;
- API draft: PR #27 at final reviewed head `8ec51979c3df97936bc8a6905ccedac552e4f49f`;
- API files reviewed: `app/config.py`, `app/main.py`, `app/separation_service.py`, `.env.example`, `README.md`, `tests/test_separation_service.py`, and `tests/test_stem_api.py`;
- cross-module contract tests: merged PR #28;
- real Linux and Windows runtime-client smoke: merged PR #29;
- final API CI: run `148`, successful;
- final API test result: `448 passed, 1 warning`;
- final API compile and browser JavaScript syntax checks: successful.

The initial review found SR-001 through SR-006. The remediation was reviewed against the final API head, not the earlier blocked head. All six findings now have code changes and focused regression coverage.

## Threat model

The intended deployment is a private, local-first, one-user application. The review assumes:

- malformed or adversarial uploaded media;
- arbitrary job identifiers and stem kinds received through HTTP routes;
- duplicate clicks, concurrent requests, multiple analyzed jobs, process restart, and accidental multi-worker startup;
- missing, replaced, incompatible, or compromised optional worker executables;
- stale or replaced runtime locks and model readiness metadata;
- partial model downloads, interrupted inference, worker timeout, invalid worker JSON, and corrupted stem output;
- SQLite locking, transient I/O failure, disk-full conditions, permission errors, and process death between claim, schedule, publication, and final persistence;
- local filesystem changes between validation and response streaming;
- accidental network exposure of an otherwise local application.

An administrator or same-account attacker with unrestricted filesystem/process access, a deliberately malicious executable already trusted by local configuration, and physical compromise remain outside the practical protection boundary. PopEx must still contain worker output, avoid path and credential disclosure, preserve earlier artifacts, and fail closed at protocol boundaries.

## Executive decision

The optional stem-separation API is ready to merge for the documented single-process, one-user, local-first CPU MVP.

The final implementation provides:

- explicit first-use model-download consent;
- a fresh-cache path that reaches `download_required` safely;
- one shared preparation/inference slot across jobs;
- optional configuration that cannot break required ingestion and analysis startup;
- fail-closed handling of unknown persisted separation states;
- one sanitized nested public separation payload with internal SQLite fields removed;
- an outer background-task safety boundary and retryable persistence recovery;
- preservation of successful source preparation, audio analysis, prior manifest pointers, timestamps, and stem bytes;
- manifest-authoritative details, preview, and download endpoints;
- exact external runtime locks and real passive client/runtime validation on Linux and Windows CPU profiles.

No unresolved blocker or high-severity finding remains for the API merge.

## Merge blockers

None.

## High-priority findings

No unresolved high-priority finding remains. SR-001 through SR-006 are closed by the final PR #27 implementation and tests.

## Accepted local-MVP limitations

The following limitations are accepted and must remain documented:

- one application process and one local user are the supported operating mode;
- no account system or per-job authorization is provided;
- non-loopback or public deployment is unsupported without authentication and Host/Origin protections;
- historical successful stem run directories are not garbage-collected automatically;
- a same-account attacker can replace trusted local files or executables and already has broader access than PopEx can prevent;
- artifact validation returns a pathname before `FileResponse` opens it, leaving a narrow same-user TOCTOU window;
- the runtime client terminates the direct worker process but does not yet prove descendant-process cleanup on every platform;
- model quality, full checkpoint preparation, inference time, peak memory, and end-user performance remain outside ordinary repository CI.

## Detailed findings

### SR-001 — Fresh cache root prevents first-use consent flow

- **Original severity:** blocker
- **Final status:** resolved
- **Final evidence:** enabled service initialization safely creates and validates the PopEx-owned cache root before passive probing. Disabled mode performs no cache creation. Unsafe components become a safe unavailable capability rather than crashing the base app.
- **Regression evidence:** a nonexistent default cache reaches `download_required`, strict consent prepares exactly once, and separation proceeds.
- **Merge disposition:** closed.

### SR-002 — Cross-job model/inference concurrency is unbounded

- **Original severity:** high
- **Final status:** resolved for the CPU local MVP
- **Final evidence:** `SeparationService` owns one shared worker slot around model preparation and inference. Waiting jobs remain claimed and processing. A second first-use job re-probes inside the slot and skips redundant preparation when the first job already prepared the model.
- **Regression evidence:** two ready jobs never exceed one active operation; two simultaneous first-use jobs prepare once and separate twice.
- **Residual limitation:** the lock is in-process; distributed and multi-process coordination remains future work.
- **Merge disposition:** closed for the supported operating mode.

### SR-003 — Disabled mode can be broken by unused runtime environment values

- **Original severity:** high
- **Final status:** resolved
- **Final evidence:** `STEM_SEPARATION_ENABLED` is parsed before runtime-only values. Disabled mode ignores stale device, timeout, worker, lock, and cache values. Enabled invalid optional configuration is represented by a generic unavailable state rather than failing required startup.
- **Regression evidence:** disabled malformed runtime settings keep health available and perform zero runtime/cache I/O; enabled invalid settings serialize `unavailable` without path leakage.
- **Merge disposition:** closed.

### SR-004 — Unknown persisted status fails open as not-started

- **Original severity:** high
- **Final status:** resolved
- **Final evidence:** unknown or future persisted statuses are non-actionable, serialize as a generic failed/unavailable state, expose no start URL, and are rejected by POST before capability refresh, claim, or scheduling. The original database value is preserved for inspection or migration.
- **Regression evidence:** summary and POST tests cover a future status value.
- **Merge disposition:** closed.

### SR-005 — Raw persisted separation errors leak beside sanitized summary

- **Original severity:** high
- **Final status:** resolved
- **Final evidence:** all internal separation persistence columns are removed from top-level public job JSON. The nested summary is the only public separation state and sanitizes traceback text, known and arbitrary paths, URLs, credentials, bearer-like values, runtime-lock/cache assignments, and control characters.
- **Regression evidence:** list and single-job tests seed Windows/POSIX paths, lock/cache names, credentials, and traceback text and assert that none appear anywhere in JSON.
- **Merge disposition:** closed.

### SR-006 — Background persistence failure can leave processing state indefinitely

- **Original severity:** high
- **Final status:** resolved for recoverable local storage failures
- **Final evidence:** the background task has an outer safety boundary. Failure persistence is retried once. Callback-write, failure-write, and final-completion-write failures are handled and logged. A completion-write failure restores the prior canonical manifest or removes the newly published pointer when no prior success existed, then records a retryable failed state when SQLite recovers.
- **Regression evidence:** one-shot callback, failure-recording, and completion-recording fault-injection tests preserve preparation, analysis, prior manifest pointers, timestamps, manifests, and stem bytes.
- **Residual limitation:** sustained database unavailability can only be logged until storage becomes writable or the process restarts.
- **Merge disposition:** closed.

### SR-007 — Multi-worker startup recovery can interfere with live work

- **Severity:** medium
- **Status:** accepted local-MVP limitation
- **Evidence:** startup recovery is global and does not use leases or ownership identifiers.
- **Required operating rule:** run one PopEx application process. Multi-process support requires a durable queue or lease/heartbeat design.

### SR-008 — Artifact validation-to-streaming race remains

- **Severity:** medium
- **Status:** accepted local-MVP limitation
- **Evidence:** the artifact resolver validates containment, symlinks, metadata, and before/after stat snapshots, then `FileResponse` opens the returned pathname.
- **Later hardening:** portable no-follow descriptor-based streaming or an immediate reopen/recheck boundary.

### SR-009 — Local-only request trust is not enforced by authentication

- **Severity:** medium
- **Status:** accepted local-MVP limitation
- **Required operating rule:** use loopback or a trusted local network only. Public or untrusted-network deployment requires authentication and Host/Origin protections.

### SR-010 — Worker timeout does not prove descendant cleanup

- **Severity:** medium
- **Status:** pre-real-model validation item
- **Evidence:** the client terminates and then kills the direct worker process; descendant-tree behavior has not been demonstrated on both platforms.
- **Required action:** observe process trees during real timeout and cancellation tests on Linux and Windows, then add process-group or Job Object handling if needed.

### SR-011 — Successful and partial artifact/cache growth is not quota-managed

- **Severity:** medium
- **Status:** accepted local-MVP limitation
- **Required action:** add retention, cleanup, and disk diagnostics in a later lifecycle cycle without deleting the currently published result unexpectedly.

### SR-012 — Consent boundary is correctly structured

- **Severity:** informational
- **Status:** accepted control
- **Evidence:** strict Boolean request parsing, request-time capability refresh, consent before claim, exact-True runtime enforcement, no preparation for ready state, and no preparation from startup, health, listing, polling, details, or artifact routes.

### SR-013 — Supply-chain and runtime trust boundaries are strong

- **Severity:** informational
- **Status:** accepted control
- **Evidence:** trusted absolute configuration, per-spawn runtime-lock validation, minimal credential-free child environments, strict versioned JSON protocol, bounded output/time, exact model identity, full checkpoint digest, offline verification/separation, and exact Linux/Windows CPU profile locks.

### SR-014 — Manifest-backed artifact access is substantially fail closed

- **Severity:** informational
- **Status:** accepted control
- **Evidence:** canonical DB pointer, schema-3 validation, exact provenance, safe run paths, fixed current stem inventory, WAV metadata revalidation, symlink containment, conservative kind identifiers, and no client-supplied filesystem paths.

## Test coverage map

| Boundary | Final evidence | Remaining later evidence |
| --- | --- | --- |
| Explicit consent | strict body tests; fresh-cache first-use test; ready-state no-prepare test | real checkpoint preparation |
| Runtime/profile trust | merged client/worker tests; real Linux/Windows passive profile smoke | real model preparation and inference |
| Same-job concurrency | atomic SQLite claim and duplicate request tests | cross-process ownership |
| Cross-job concurrency | shared one-operation service slot; ready and first-use tests | multi-process queue/lease |
| Restart recovery | DB and API restart tests | overlapping process startup |
| Stage progress | canonical stage validation and sub-100 clamp | real inference timing |
| Failure preservation | publisher, retry, callback, persistence, manifest restoration tests | disk-full on real runtime |
| Artifact resolution | manifest/symlink/metadata helper and route tests | no-follow response streaming |
| Error privacy | complete list/single JSON path/credential/traceback assertions | network-facing audit mode |
| Process cleanup | direct timeout/cancel tests | descendant-tree observation |
| Disk lifecycle | failed-run cleanup and prior-result preservation | quota and retention policy |

## Required pre-merge tests

All required pre-merge API tests are present and passed on final head `8ec51979c3df97936bc8a6905ccedac552e4f49f` in CI run `148`.

## Required pre-real-model tests

Before claiming real-model production confidence:

- run explicit first-use preparation using the audited checkpoint and verify exact repository, revision, size, and SHA-256;
- run one synthetic CPU inference on Linux and Windows using the exact validated profiles;
- interrupt model download and verify no readiness manifest and no usable corrupt checkpoint;
- test timeout, cancellation, server termination, and restart while observing process trees;
- inject disk-full and permission failures during download, stem output, manifest replacement, and final SQLite completion;
- record peak memory, CPU, runtime, and generated artifact size for one supported synthetic input;
- verify no source audio content or path is transmitted to the model host and no credentials are inherited;
- confirm prior source, analysis, manifest, and stem bytes remain unchanged after failed retries.

## Later hardening backlog

- durable task ownership or a separate queue for multi-process support;
- authenticated non-loopback mode with Host/Origin/CSRF protections;
- no-follow descriptor-based artifact streaming;
- process-group or Windows Job Object descendant cleanup;
- cache and historical-run retention with user-visible disk diagnostics;
- executable/package attestation beyond trusted local path configuration;
- bounded request bodies and rate limits for network-exposed deployments;
- audit records for consent, preparation, verification, separation, and cleanup.

## Final gate

**APPROVED FOR API MERGE**

Reason: PR #27 final head `8ec51979c3df97936bc8a6905ccedac552e4f49f` resolves SR-001 through SR-006, includes focused regression coverage, integrates the merged cross-module and real Linux/Windows passive runtime evidence, changes only authorized files, and passes CI run `148`. Remaining items are documented local-MVP limitations or pre-real-model validation work rather than API merge blockers.
