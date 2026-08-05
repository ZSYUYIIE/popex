# Demucs Linux CPU real-model end-to-end validation

## Purpose

This manual validation proves the merged PopEx product path, not only an isolated worker or runtime client. It installs the exact Linux CPU profile, starts the real FastAPI application with trusted local settings, observes `download_required`, sends the strict explicit-consent request `{"allowModelDownload": true}`, prepares the audited model, performs CPU inference on generated synthetic stereo PCM audio, publishes schema 3, and retrieves vocals, bass, drums, and other through the public details, preview, and download endpoints.

The workflow is intentionally `workflow_dispatch` only. Pushes and pull requests do not repeatedly download the checkpoint.

## Locked identity

- Profile: `linux-x86_64-cpu-cpython313`
- CPython: 3.13
- Worker protocol: 1
- Worker: `1.0.0`
- Demucs: `4.1.0`
- Torch: `2.13.0+cpu`
- Repository identity: `adefossez/HTDemucs`
- Revision: `bf35a81b663819a8255c8fefee17f9d812b786b5`
- Checkpoint: `955717e8.safetensors`
- Checkpoint size: `84025440` bytes
- SHA-256: `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`

No fallback repository, mutable revision, legacy `.th`, alternate checkpoint, or relaxed hash is accepted.

## Validation path

The workflow:

1. installs the lightweight base package with Python 3.13;
2. runs the permanent offline contract tests;
3. creates isolated temporary runtime, cache, application-data, and safe-summary paths;
4. installs the exact profile with `scripts/install_demucs_linux_cpu.sh`;
5. invokes `scripts/validate_demucs_linux_real_model_e2e.py` with trusted absolute paths;
6. confirms passive startup remains download-only and creates no model before consent;
7. sends the exact Boolean `allowModelDownload` consent through FastAPI;
8. requires one audited preparation and one worker-backed separation;
9. validates completed SQLite state, schema 3 provenance, the physical checkpoint size/hash, and all four manifest-backed endpoints;
10. prints one path-free JSON summary with versions, revision/hash, stem durations/sizes, preparation/inference elapsed time, and peak RSS;
11. performs cleanup under `if: always()`.

## Privacy and artifact policy

The input is a deterministic four-second stereo 44.1 kHz PCM WAV generated from sine components and seeded transients. It contains no copyrighted or private recording.

The validation refuses untrusted relative, overlapping, symlinked, or nonempty roots. Runtime and model subprocesses receive the merged minimal environment. Hostile `HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN` sentinel values are present in the workflow only to prove they do not appear in the product output. Hub implicit-token use and telemetry are disabled.

No workflow artifact is uploaded. The runtime, checkpoint cache, source WAV, generated stems, SQLite database, summary file, and machine paths remain temporary. The final cleanup removes all of them even after failure.

## Safe failure evidence

The validator never prints raw exceptions, tracebacks, URLs, tokens, or local paths. A failure emits a classified JSON object containing only a phase, stable validation code, and—when the merged runtime client supplied it—safe runtime/worker/exit codes. If exact model-host access, checkpoint verification, or inference fails, the PR must remain blocked rather than substituting fake inference.

## Local static validation

The permanent tests are offline and do not download or import the optional runtime:

```bash
pytest tests/test_demucs_linux_real_model_e2e.py
pytest
python -m compileall -q app tests scripts/validate_demucs_linux_real_model_e2e.py
node --check app/static/app.js
```

## Final workflow evidence

Status: blocked before dispatch.

Repository CI run `171` (run ID `30973494047`) passed on the complete implementation head with `459 passed`, compile-all success, and JavaScript syntax success. The manual real-model workflow has no run number or run ID because GitHub only permits `workflow_dispatch` for a workflow definition present on the default branch. This new workflow exists only on the required unmerged feature branch, and this agent is not authorized to merge it or modify an existing default-branch workflow solely to bootstrap dispatch.

No checkpoint was downloaded and no real inference was claimed. An orchestrator-controlled default-branch or integration checkpoint is required before the exact workflow can be dispatched. After that checkpoint, the workflow must run against the unchanged validation contract and either record its safe success summary and cleanup evidence or preserve a genuine model-host/runtime/inference failure as a blocked result.
