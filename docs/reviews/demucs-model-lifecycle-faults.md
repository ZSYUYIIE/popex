# Demucs model lifecycle fault validation

## Status

**BLOCKED — production fault characterizations fail.**

This review validates the merged optional-worker model lifecycle without downloading the real 84 MB checkpoint, contacting a model host, installing GPU packages, or modifying production code. Deterministic fake Hub and filesystem boundaries exercise the real worker preparation, verification, readiness, CLI envelope, base service, SQLite, and prior-artifact preservation paths.

## Scope and method

The permanent suite uses:

- the real `popex_demucs_worker.commands.prepare_model` and `verify_model` functions;
- the real exact-revision Hub call construction;
- the real bag validation, size/hash checks, readiness schema, atomic JSON publication, containment checks, CLI envelopes, and diagnostic sanitization;
- the real `SeparationService` failure and persistence path;
- tiny deterministic JSON/YAML and checkpoint bytes;
- patched expected checkpoint size and digest in test process only;
- no real checkpoint, model host, inference, GPU, or arbitrary URL.

The workflow is Linux-only because all injected boundaries are deterministic and platform-native inference is out of scope. Hub offline/privacy variables are set before all commands, no cache or runtime artifact is uploaded, and temporary state is removed with `always()` cleanup.

## Production defects

### 1. Readiness publication follows a symlinked parent

`atomic_write_json()` creates and replaces the readiness file through `path.parent` without revalidating that every parent remains a contained non-symlink directory. If `cache/readiness` becomes a symlink after asset download and before manifest publication, preparation follows it and writes `htdemucs-bf35a81b-v1.json` outside the trusted cache root.

The failing characterization is:

```text
test_prepare_model_rejects_symlinked_readiness_parent_without_external_write
```

Expected contract:

- preparation fails with a structured worker error;
- no external readiness file is written;
- no authoritative readiness exists under the cache root.

Observed production behavior before a fix:

- preparation can return success;
- the readiness manifest is written through the symlink outside the trusted root.

This violates containment and atomic-publication requirements. The test is intentionally not weakened or marked xfail.

### 2. Passive model probe does not rehash the checkpoint

`load_readiness_manifest()` validates the manifest identity and checkpoint size but does not recompute the checkpoint SHA-256. A same-size checkpoint replacement can therefore remain accepted by passive `model_probe()` even when its bytes no longer match the audited digest.

The failing characterization is:

```text
test_passive_model_probe_rehashes_same_size_checkpoint_replacement
```

Expected contract:

- passive readiness probing detects the same-size replacement;
- the model is not reported ready;
- the detailed error identifies a checkpoint hash mismatch.

Observed production behavior before a fix:

- `model_probe()` can report `offlineReady: true` for the replaced bytes.

A related characterization, `test_replacement_during_verification_must_not_publish_ready_state`, exercises replacement after hashing and before readiness publication. The safe fix must close both the passive-probe integrity gap and the verification/publication time-of-check/time-of-use window.

## Validated non-defect behavior

The suite also characterizes the following expected behavior:

- exact repository `adefossez/HTDemucs`, full revision, allowlisted bag/checkpoint filenames, `token=False`, and no arbitrary URL;
- network failure before bytes, YAML interruption, checkpoint interruption, short checkpoint, disk-full/write failures, fsync failure, rename failure, read-only publication boundary, non-directory readiness parent, cancellation, and timeout;
- no readiness publication after failed preparation;
- temporary download files remain contained and non-authoritative;
- existing valid readiness and checkpoint bytes survive a download failure that occurs before replacement;
- changed checkpoint size, corrupt readiness JSON, traversal, and symlinked readiness are rejected;
- a repaired transient condition can succeed on a later explicitly authorized retry;
- no stale `.partial` or readiness `.tmp` file influences retry;
- worker/client/service-visible diagnostics redact seeded tokens, bearer values, URLs, Windows paths, POSIX paths, and cache paths;
- a claimed service failure becomes retryable separation failure while source media, `analysis.wav`, metadata, audio-analysis JSON, prior stem manifest, pointer, and prior stem bytes remain unchanged;
- a request rejected before claim does not alter SQLite or artifacts.

## Existing verified cache semantics

Current production behavior preserves the readiness-manifest bytes after many failures rather than deleting them. Corruption detected by size/schema/path checks makes readiness logically unavailable but does not erase the stored manifest. That is acceptable only when every passive probe revalidates the complete checkpoint identity. The missing passive digest check currently breaks that assumption.

A failed re-preparation before asset replacement leaves a prior valid snapshot usable. A future fix must also ensure that a failed re-preparation cannot overwrite or replace assets referenced by an existing valid readiness manifest before the new candidate has been fully verified and atomically published.

## Required production follow-up

A focused production patch should:

1. make readiness publication reject symlinked or non-directory parent components immediately before temporary-file creation and again before replacement;
2. ensure the temporary and final readiness paths remain contained beneath the resolved trusted cache root;
3. recompute or otherwise cryptographically validate the checkpoint during passive readiness probing;
4. close replacement races between verification and readiness publication, for example through immutable candidate paths plus final identity revalidation;
5. preserve the existing safe error envelopes, privacy controls, exact source identity, prior cache snapshot, and retry semantics;
6. make the committed failing characterizations pass without editing their security expectations.

## Validation commands

```bash
pytest tests/test_demucs_model_lifecycle_faults.py
pytest
python -m compileall -q app tests scripts/validate_demucs_model_lifecycle_faults.py
node --check app/static/app.js
```

The dedicated workflow is:

```text
.github/workflows/demucs-model-lifecycle-faults.yml
```

It is expected to remain red while the production defects are unfixed. The PR must remain draft, blocked, and unmerged.

## Gate result

**BLOCKED — failing production characterizations preserved for a focused patch.**
