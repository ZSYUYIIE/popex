# Stem separation API security and reliability review

## Status and reviewed revisions

This review is the durable handoff for issue #26.

Reviewed repository state:

- merged base: `main` at `0a124bc0b533852a2d9e59f381d1c767c8ba1492`;
- API assignment: issue #23;
- API draft: PR #27, reviewed through head `798995cdf700fa5f7d87853beda09da5a996c5af`;
- API changed files reviewed at that head: `app/config.py`, `app/main.py`, `app/separation_service.py`, `.env.example`, `tests/test_separation_service.py`, and `tests/test_stem_api.py`;
- supplementary real-profile/client evidence: PR #29 at `20e6e5ad3ec357b3f230ce3b67ac52127090de46`, with Linux and Windows passive smoke stopping at `MODEL_DOWNLOAD_REQUIRED` and creating no model assets.

Blocker and high-severity API findings were posted directly to PR #27 as SR-001 through SR-006. The API branch remains draft and under active repair. This document must be rechecked against the final API head before its gate can be upgraded.

## Threat model

The intended deployment is a private, local-first, one-user application. The review nevertheless assumes the following realistic failure and adversary conditions:

- malformed or adversarial uploaded media;
- arbitrary job identifiers and stem kinds received through HTTP routes;
- duplicate clicks, concurrent requests, multiple analyzed jobs, process restart, development reload, and accidental multi-worker startup;
- missing, replaced, incompatible, or compromised optional worker executables;
- stale or replaced runtime locks and model readiness metadata;
- partial model downloads, interrupted inference, worker timeout, child-process failure, invalid worker JSON, and corrupted stem output;
- SQLite locking, transient I/O failure, disk-full conditions, permission errors, and process death between claim, schedule, publication, and final persistence;
- a browser page from another origin attempting to reach a loopback or non-loopback PopEx service;
- local filesystem changes between validation and response streaming.

The following are not considered fully preventable inside this local MVP: an administrator or same-account attacker with unrestricted filesystem/process access, a deliberately malicious worker executable already trusted by local configuration, and physical compromise of the host. The application must still contain worker output, avoid path and credential disclosure, preserve earlier artifacts, and fail closed at protocol boundaries.

## Executive decision

The merged runtime client, worker protocol, platform locks, atomic separation publisher, persistence claim, capability mapper, and artifact resolver provide a strong foundation. Exact model identity, explicit download consent, strict subprocess JSON, environment minimization, offline verification/separation, manifest authority, path containment, symlink checks, and prior-artifact preservation are materially implemented and tested.

The API integration at reviewed head `798995cdf700fa5f7d87853beda09da5a996c5af` is **not ready to merge**. One first-use blocker and five high-priority defects remain in the API/configuration/service layer. The most important defect makes a fresh default cache classify as unavailable, preventing the user from ever reaching the explicit-consent model preparation path.

## Merge blockers

- **SR-001:** a fresh enabled installation with the default, not-yet-created cache root cannot reach `download_required`; passive `model_probe` receives a nonexistent root and fails as an invalid worker request. The explicit first-use path is therefore unreachable with the normal default configuration.

No blocker was found in the merged worker model allowlist, runtime lock boundary, strict protocol parser, SQLite claim primitive, manifest publication, or artifact resolver.

## High-priority findings

- **SR-002:** worker-backed model preparation and inference are unbounded across different jobs.
- **SR-003:** disabled mode still parses unused runtime-only environment settings and can fail base application startup.
- **SR-004:** unknown persisted separation status is converted to actionable `not_started` instead of failing closed.
- **SR-005:** raw top-level separation persistence fields bypass the sanitized nested serializer.
- **SR-006:** stage/failure/final SQLite write errors can escape a background task and leave a row indefinitely processing.

## Accepted local-MVP limitations

These limitations are acceptable only when documented and kept outside the final security claim:

- one application process and one local user are the supported operating mode;
- no account system or per-job authorization is provided;
- the official run script binds to `127.0.0.1`, but non-loopback/Docker exposure must be treated as an unsupported trusted-network deployment unless authentication and origin/host controls are added;
- historical successful stem run directories are not garbage-collected automatically;
- a same-account attacker can replace files or executables and already has broader access than PopEx can prevent;
- the current client terminates the direct worker process but does not prove descendant-process cleanup on every platform;
- model quality, inference time, peak memory, and full checkpoint download behavior remain outside passive API tests.

## Detailed findings

### SR-001 — Fresh cache root prevents first-use consent flow

- **Severity:** blocker
- **Affected boundary:** configuration → service capability probe → worker `model-probe`
- **Exact evidence:** PR #27 `Settings.from_env()` defaults an enabled missing cache setting to `data_dir / "runtime-cache" / "demucs"` without creating it. `SeparationService.initialize()` immediately calls `refresh_capability()`, while `_cache_root()` only returns the path. The merged worker `model_probe()` calls `trusted_root(cache_root_text)` without `create=True`, so a nonexistent root produces exit 30 rather than `MODEL_DOWNLOAD_REQUIRED`.
- **Realistic scenario:** a user installs a validated worker/runtime profile, enables separation, supplies worker and runtime-lock paths, and has never downloaded a model. The normal cache directory does not exist. The UI receives `unavailable`, not `download_required`; POST rejects before consent and cannot call `prepare_model`.
- **Current mitigation:** `prepare-model` itself can create a trusted cache root, but the API cannot reach it because capability is non-actionable first.
- **Residual risk:** complete first-use feature failure under the intended default configuration.
- **Required action and owner:** API/service owner must create and validate the enabled cache root during service initialization, not settings parsing. Creation must reject symlinks and non-directories without affecting disabled base startup.
- **Merge disposition:** must fix before API merge. Posted to PR #27 as comment `5176019589`.

### SR-002 — Cross-job model/inference concurrency is unbounded

- **Severity:** high
- **Affected boundary:** API claim/scheduling → optional worker processes → shared cache and host resources
- **Exact evidence:** `db.claim_separation_attempt()` serializes attempts only per job. `SeparationService.request_start()` schedules each successful job independently. The service `RLock` protects capability cache mutation only; no semaphore or queue surrounds `prepare_model()` or the processor call.
- **Realistic scenario:** several analyzed jobs are started quickly. Multiple CPU Torch subprocesses allocate large memory concurrently. Two first-use jobs can also prepare the same model cache at the same time.
- **Current mitigation:** same-job duplicate requests have exactly one SQLite claim winner; Hugging Face and atomic manifest helpers provide some internal locking/atomicity.
- **Residual risk:** memory exhaustion, severe CPU contention, redundant network/cache operations, worker termination, and multiple failed/stuck rows.
- **Required action and owner:** API/service owner should provide a shared bounded execution slot, with one concurrent preparation/inference operation preferred for the CPU MVP. Cross-job and cross-consent tests are required. If deferred from API merge, it blocks real-model testing and must be explicitly documented.
- **Merge disposition:** high-priority required patch before real-model use; posted to PR #27 as comment `5176034977`.

### SR-003 — Disabled mode can be broken by unused runtime environment values

- **Severity:** high
- **Affected boundary:** environment parsing → required base application startup
- **Exact evidence:** PR #27 `Settings.from_env()` always calls `_device`, `_positive_int` for the separation timeout, and `_optional_path` for runtime paths even when `STEM_SEPARATION_ENABLED` is false.
- **Realistic scenario:** a stale `STEM_SEPARATION_DEVICE=gpu`, malformed timeout, or unusable path remains in a shell profile while separation is disabled. `Settings.from_env()` raises before the application can serve ingestion, analysis, or health.
- **Current mitigation:** missing worker configuration after successful settings construction is mapped to `runtime_missing` or `unavailable` by the service.
- **Residual risk:** optional functionality violates the required base-app availability boundary.
- **Required action and owner:** config/API owner must ignore runtime-only values when disabled. When enabled, invalid optional runtime configuration should become a safe unavailable capability rather than crash startup. A malformed enable Boolean may remain a deliberate top-level configuration error.
- **Merge disposition:** must fix before API merge. Posted to PR #27 as comment `5176049369`.

### SR-004 — Unknown persisted status fails open as not-started

- **Severity:** high
- **Affected boundary:** SQLite persistence → job serialization and POST eligibility
- **Exact evidence:** PR #27 `_safe_status()` maps every value outside the four current statuses to `"not_started"`. `serialize_job()` uses that value for `canStart`, and `request_start()` uses it for pre-claim checks.
- **Realistic scenario:** a future migration introduces `paused`, or a damaged row contains an unknown state. The UI is offered a start action instead of a generic non-actionable state. The underlying atomic SQL claim currently rejects the unknown raw row, so POST returns a confusing state-change conflict rather than actually overlapping the attempt.
- **Current mitigation:** the database claim SQL only accepts raw `not_started` or `failed`, preventing the worst duplicate execution in the present schema.
- **Residual risk:** fail-open UI state, misleading consent/action behavior, and unsafe forward compatibility if claim logic later expands.
- **Required action and owner:** service owner must preserve a safely bounded unknown state or emit `unknown`, set `canStart=false`, and reject POST before capability probing/claiming.
- **Merge disposition:** must fix before API merge. Posted to PR #27 as comment `5176055336`.

### SR-005 — Raw persisted separation errors leak beside sanitized summary

- **Severity:** high
- **Affected boundary:** SQLite row → `/api/jobs` and `/api/jobs/{id}` JSON
- **Exact evidence:** merged `_serialize_job()` starts with `payload = dict(job)`. PR #27 adds a sanitized nested `separation.error`, but does not remove the raw top-level `separation_error` or `separation_message` copied from SQLite.
- **Realistic scenario:** a prior version, injected test, unexpected dependency error, or manually repaired database stores a Windows/POSIX worker, cache, job, or lock path in `separation_error`. The nested field is redacted while the raw top-level field exposes it unchanged.
- **Current mitigation:** newly generated known service errors are generally static or passed through `friendly_error`; runtime-client messages are separately sanitized.
- **Residual risk:** the public API contract can still disclose absolute paths, tracebacks, or credential-like text through legacy/unexpected persisted values.
- **Required action and owner:** API serializer owner must remove internal separation persistence columns from public payloads or sanitize every public copy. Tests must inspect the complete JSON, not only the nested object.
- **Merge disposition:** must fix before API merge. Posted to PR #27 as comment `5176087321`.

### SR-006 — Background persistence failure can leave processing state indefinitely

- **Severity:** high
- **Affected boundary:** processor callbacks/finalization → SQLite → retry lifecycle
- **Exact evidence:** `run_attempt()` catches processor/runtime errors, but the success `db.update_job(... completed ...)` occurs in the `else` block outside a further safety boundary. `_record_failure()` inside exception handlers can itself raise. `_update_stage()` database failures propagate through the processor callback.
- **Realistic scenario:** a one-shot SQLite lock, disk-full event, permission change, or transient I/O failure occurs during a callback, failure write, or final completion write. The background task exception is unobserved by the request path and the row may remain processing until restart. If the stem manifest was already published, the new successful run can remain orphaned from the database pointer.
- **Current mitigation:** startup `fail_incomplete_jobs()` makes processing rows retryable after a process restart and preserves earlier artifacts.
- **Residual risk:** no-restart wedge, misleading polling, orphaned run data, and loss of immediate retry availability.
- **Required action and owner:** service owner must add an outer task safety boundary, explicit logging, and best-effort one-shot recovery persistence. Synthetic callback, failure-write, and final-write failure tests are required.
- **Merge disposition:** must fix before API merge. Posted to PR #27 as comment `5176101841`.

### SR-007 — Multi-worker startup recovery can interfere with live work

- **Severity:** medium
- **Affected boundary:** application lifespan → global SQLite recovery → concurrent application processes
- **Exact evidence:** every `create_app()` lifespan invokes `db.fail_incomplete_jobs()` globally. No worker ownership, lease, heartbeat, or process identifier distinguishes genuinely interrupted attempts from work active in another process.
- **Realistic scenario:** a second Uvicorn worker starts, or a rolling/reload process overlaps an existing worker. The new process marks the existing worker’s live separation failed. A user may retry while the first worker continues, producing two runs and last-writer-wins manifest/database state.
- **Current mitigation:** official Windows development script launches one Uvicorn worker; Docker also launches one process. Atomic per-job claims prevent duplicate starts until recovery changes the status.
- **Residual risk:** unsupported multi-worker use can create false failures and duplicate same-job processing.
- **Required action and owner:** documentation/API owner must state single-process support. Multi-worker support requires leases/ownership or a separate durable queue before it is enabled.
- **Merge disposition:** acceptable local-MVP limitation only when documented before merge.

### SR-008 — Artifact validation-to-streaming race remains

- **Severity:** medium
- **Affected boundary:** manifest artifact resolver → Starlette `FileResponse`
- **Exact evidence:** `resolve_stem_artifact()` performs containment, lstat, metadata, and before/after inode snapshots, then returns a pathname. `FileResponse` opens that pathname after the helper returns.
- **Realistic scenario:** a same-user process replaces the validated file with a symlink or another file between helper return and response open.
- **Current mitigation:** strong pre-return symlink/containment/metadata checks; route kind never supplies a path; the optional worker is already a same-account trusted executable boundary.
- **Residual risk:** a compromised same-user process can exploit a narrow pathname race, although it already has broader direct filesystem access.
- **Required action and owner:** later hardening should stream from a no-follow file descriptor or re-open/recheck immediately before streaming where portable.
- **Merge disposition:** accepted local-MVP limitation; not an API merge blocker.

### SR-009 — Local-only request trust is not enforced by the application

- **Severity:** medium
- **Affected boundary:** browser/network origin → unauthenticated API and private artifacts
- **Exact evidence:** `app.main` has no authentication, `TrustedHostMiddleware`, or explicit Origin/Host enforcement. Strict JSON and no permissive CORS protect normal cross-origin form/fetch CSRF, but DNS rebinding or direct LAN access remains possible. The official Windows script binds loopback, while the Dockerfile binds `0.0.0.0`.
- **Realistic scenario:** a user exposes PopEx on a LAN or public interface. Another client can list private jobs, fetch stems, and submit explicit model-download/inference requests by API.
- **Current mitigation:** separation is disabled by default; the product is personal-use-first; normal browser cross-origin JSON fetches require a CORS preflight that is not allowed.
- **Residual risk:** DNS rebinding and direct network access if the service is exposed beyond loopback.
- **Required action and owner:** README/API owner must clearly restrict the unauthenticated MVP to loopback/trusted local access. Non-loopback deployments require authentication plus Host/Origin protections.
- **Merge disposition:** documentation required before merge; code hardening may follow if network deployment becomes supported.

### SR-010 — Worker timeout does not prove descendant cleanup

- **Severity:** medium
- **Affected boundary:** runtime client process lifecycle → OS child processes and resources
- **Exact evidence:** `SeparationRuntimeClient._run_subprocess()` terminates and then kills the direct `Popen` process. It does not establish a Unix process group, Windows Job Object, or another descendant-tree cleanup mechanism.
- **Realistic scenario:** Demucs/Torch or a dependency spawns a child process. Parent timeout/cancellation kills only the worker, leaving descendants consuming CPU, memory, handles, or files.
- **Current mitigation:** the current worker implementation is primarily single-process and client timeouts are bounded; failed run cleanup removes safely contained output when possible.
- **Residual risk:** platform-specific orphan processes have not been disproved with the real runtime.
- **Required action and owner:** pre-real-model lifecycle tests must observe process trees on Linux and Windows. Add process-group/Job-Object cleanup if descendants occur.
- **Merge disposition:** not a synthetic API merge blocker; blocks production confidence until tested.

### SR-011 — Successful and partial artifact/cache growth is not quota-managed

- **Severity:** medium
- **Affected boundary:** repeated retries/downloads → local disk
- **Exact evidence:** each successful separation allocates a new run directory and atomically repoints the manifest; older successful runs remain. Failed cleanup is best-effort. Hugging Face may retain partial cache files after an interrupted authorized download.
- **Realistic scenario:** repeated successful retries or interrupted first-use downloads consume disk until later writes fail.
- **Current mitigation:** failed unpublished runs are removed best-effort; no model is downloaded without explicit consent; readiness is published only after full verification.
- **Residual risk:** disk exhaustion and accumulation of unreachable historical runs/partial cache content.
- **Required action and owner:** later lifecycle work should add explicit cleanup/retention and disk-space diagnostics without deleting the currently published or previous valid artifacts unexpectedly.
- **Merge disposition:** accepted local-MVP limitation with documentation; include disk-full tests before real-model confidence.

### SR-012 — Consent boundary is correctly structured

- **Severity:** info
- **Affected boundary:** frontend → FastAPI body → service → runtime client
- **Exact evidence:** frontend sends `allowModelDownload=true` only for rendered `download_required`; PR #27 uses `StrictBool` and forbids extra body fields; service refreshes capability at POST time; only `run_attempt(... prepare_model=True)` calls `client.prepare_model(allow_model_download=True)`; ready state ignores redundant true.
- **Realistic scenario:** missing, false, string, numeric, or stale consent requests.
- **Current mitigation:** strict validation, request-time capability refresh, claim after consent check, and runtime-client exact-True enforcement.
- **Residual risk:** network-exposed unauthenticated callers remain outside the local-only trust boundary described in SR-009.
- **Required action and owner:** retain these tests and add a real missing-cache first-use case after SR-001.
- **Merge disposition:** accepted control; no finding against merge.

### SR-013 — Supply-chain and runtime trust boundaries are strong

- **Severity:** info
- **Affected boundary:** trusted configuration → runtime client → isolated worker → model cache
- **Exact evidence:** merged client requires trusted absolute worker/cache/lock configuration, revalidates the lock per spawn, uses a minimal credential-free environment, requires strict UTF-8 one-object protocol JSON, caps output/time, and sets offline mode for verify/separate. Worker locks exact package/model identity and uses exact repository/revision/checkpoint size/full SHA-256 before atomic readiness publication. Linux/Windows profile-client smoke in PR #29 passed without model assets.
- **Realistic scenario:** missing runtime, replaced lock, incompatible package, malformed worker envelope, wrong model family, or unexpected output.
- **Current mitigation:** fail-closed protocol, model allowlist, full verification, no fallback loading, exact output inventory, path containment, and safe broad/detailed errors.
- **Residual risk:** a deliberately compromised executable trusted by local configuration has same-user privileges; executable replacement is not re-attested cryptographically per spawn.
- **Required action and owner:** preserve exact locks and repeat profile/client workflows after any runtime/profile change. Consider executable identity attestation later if the threat model expands.
- **Merge disposition:** control accepted.

### SR-014 — Manifest-backed artifact access is substantially fail closed

- **Severity:** info
- **Affected boundary:** SQLite pointer → manifest → WAV metadata → preview/download route
- **Exact evidence:** canonical DB pointer is required; manifest schema, exact model provenance, safe run path, required stem order, WAV regular-file status, symlink containment, size/rate/channel/duration, and before/after stat snapshots are validated. Route input is a conservative kind identifier and never a filename/path.
- **Realistic scenario:** traversal, percent-encoding, extra job files, corrupted manifest, symlinked run/stem, or replaced WAV.
- **Current mitigation:** dedicated resolver and stable 404/500 route mapping without path detail.
- **Residual risk:** only the post-validation FileResponse race described in SR-008.
- **Required action and owner:** retain focused helper tests and add route-level symlink/replacement tests where platforms permit.
- **Merge disposition:** control accepted.

## Test coverage map

| Boundary | Existing evidence | Missing evidence at reviewed API head |
| --- | --- | --- |
| Explicit consent | frontend contract, strict `StrictBool`, service fake-client tests | real client with nonexistent default cache; duplicate first-use requests across jobs |
| Runtime/profile trust | merged client/worker tests; PR #29 Linux and Windows passive smoke | real checkpoint preparation and inference lifecycle |
| Same-job concurrency | atomic DB claim and two-request service test | schedule/claim crash boundary and cross-process behavior |
| Cross-job concurrency | none | bounded global preparation/inference test |
| Restart recovery | DB tests and API synthetic restart test | overlapping multi-worker startup/recovery |
| Stage progress | canonical-stage validation and clamp-below-100 test | SQLite callback failure injection |
| Failure preservation | publisher cleanup, DB preservation, retry tests | disk-full/permission failure during final DB update and manifest publication |
| Artifact resolution | extensive manifest/symlink/metadata helper tests; route happy/corrupt cases | response-open race/no-follow streaming and route-level symlink timing |
| Error privacy | runtime/capability/artifact sanitizer tests | complete job JSON assertion covering raw copied SQLite fields |
| Browser request safety | strict JSON body; no permissive CORS | Host/Origin/DNS-rebinding and non-loopback deployment tests |
| Process cleanup | direct worker timeout/cancel tests | descendant process-tree observation on Linux and Windows |
| Disk lifecycle | failed-run best-effort cleanup | quota, historical run retention, partial model cache cleanup |

## Required pre-merge tests

The API branch must add and pass tests for:

1. enabled real-client construction with a nonexistent default cache root yielding `download_required`, followed by strict consent, one preparation call, and one separation scheduling path;
2. disabled mode with malformed runtime-only environment values still starting and preserving ingestion/analysis health;
3. enabled invalid trusted runtime configuration becoming safe `unavailable`, without startup failure or path leakage;
4. unknown/future persisted status remaining non-actionable in summary and POST;
5. complete list/single-job JSON containing no raw path, traceback, credential, cache, worker, or runtime-lock text from separation persistence fields;
6. one-shot callback, failure-persistence, and final-completion SQLite write failures producing logged, retryable, artifact-preserving outcomes;
7. all strict body cases: omitted, false, true, string, number, null, extra field, malformed JSON, and wrong content type;
8. schedule failure after claim returning a stable conflict and persisted retryable failure;
9. details/preview/download behavior for missing job, no pointer, corrupt manifest, unknown/encoded kind, symlink, metadata replacement, and prior manifest during processing/failed retry;
10. unchanged disabled `/api/jobs` shape and zero runtime/filesystem activity.

## Required pre-real-model tests

Before downloading the real checkpoint or running inference:

- serialize or bound cross-job preparation and inference, then test multiple jobs and duplicate first-use consent;
- run the exact profile/client smoke after API integration and verify only the consented path can reach model preparation;
- interrupt model download and verify no readiness manifest, no usable corrupt checkpoint, safe retry, and bounded partial cache behavior;
- run timeout, cancellation, server termination, and restart scenarios while observing process trees on Linux and Windows;
- exercise disk-full and permission errors during download, stem output, manifest replacement, and final SQLite completion;
- confirm peak memory/CPU behavior for one supported CPU separation and enforce the chosen concurrency limit;
- verify no source audio path/content is sent to the model host and no credentials are inherited;
- confirm prior source, analysis, and published stems remain byte-identical after every failed retry scenario.

## Later hardening backlog

- durable task ownership/lease or a separate job queue for multi-process support;
- authenticated non-loopback mode with Host/Origin/CSRF protections;
- no-follow descriptor-based artifact streaming;
- process-group or Windows Job Object descendant cleanup;
- cache and historical-run retention policy with user-visible disk diagnostics;
- executable/package attestation beyond trusted local path configuration;
- rate limits and bounded request bodies for any network-exposed deployment;
- audit/event records for consent, preparation, verification, separation, and cleanup.

## Final gate

**BLOCKED — SECURITY OR RELIABILITY ISSUE**

Reason: PR #27 at reviewed head `798995cdf700fa5f7d87853beda09da5a996c5af` has an unresolved first-use blocker (SR-001) and high-priority API/configuration/persistence defects SR-002 through SR-006. The branch must remain draft and unmerged. This gate must be re-evaluated against the final API head after the required patches and tests land.
