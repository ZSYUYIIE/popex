# Windows real-model stem-separation validation

This manual gate proves the exact Windows x86-64 CPU profile can use the merged FastAPI product to prepare the audited htdemucs model after explicit consent and publish four manifest-backed stems from deterministic synthetic audio.

## Safety boundary

The workflow is `workflow_dispatch` only. Normal pull-request and push CI never install the optional runtime or download the checkpoint.

The validation uses:

- `windows-latest`;
- 64-bit CPython 3.13;
- runtime profile `windows-x86_64-cpu-cpython313`;
- worker package `1.0.0`, protocol `1`;
- Demucs `4.1.0`;
- CPU-only Torch `2.13.0+cpu`;
- repository `adefossez/HTDemucs`;
- revision `bf35a81b663819a8255c8fefee17f9d812b786b5`;
- checkpoint `955717e8.safetensors`;
- checkpoint size `84025440` bytes;
- SHA-256 `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`.

The workflow installs the isolated runtime with `scripts/install_demucs_windows_cpu.ps1`. The validator runs from repository Python and starts the actual FastAPI application with trusted worker, runtime-lock, cache, and temporary data paths.

## Validated product sequence

The validator must observe this exact sequence:

1. passive startup and runtime probing create no checkpoint or readiness manifest;
2. the job capability reports `download_required`;
3. the validator submits exactly `{"allowModelDownload": true}` to the job separation endpoint;
4. the application performs model preparation only through that consented API request;
5. the application performs real CPU inference; failed real inference is not replaced with mocks;
6. SQLite records completed separation state and the canonical schema-3 manifest pointer;
7. the manifest records exact audited provenance and one contained run;
8. details, preview, and download endpoints return vocals, bass, drums, and other WAV files;
9. public API responses and the compact summary expose no machine path, cache location, credential, or traceback.

Only generated deterministic stereo 44.1 kHz PCM-16 audio is used. No private or copyrighted recording is read or committed.

## Workflow operation

Run **Demucs Windows real-model E2E** from the GitHub Actions workflow-dispatch UI after the workflow file is present on the repository default branch. The job has read-only repository permissions and a bounded timeout.

The workflow does not upload an artifact. Its safe JSON summary and GitHub step summary contain only versions, audited model identity, stem metadata, elapsed times, measurable process-memory values, and privacy booleans.

An `always()` cleanup step removes the temporary runtime, model cache, application data, database, generated audio, readiness manifest, checkpoint, and stems.

## Permanent offline checks

`tests/test_demucs_windows_real_model_e2e.py` proves without a real model download that:

- the workflow is manual-only and read-only;
- the exact Windows installer and repository Python validator are used;
- privacy controls and cleanup are present;
- no artifact upload or direct model-host command exists in the workflow;
- the validator uses the real FastAPI consent route and contains no mocks;
- synthetic audio is deterministic stereo 44.1 kHz PCM-16;
- unsafe or nonempty trusted paths are refused;
- failure output is path-safe and traceback-free.

Repository CI run #166 (run ID `30973370606`) passed all 458 tests, Python compilation, and browser JavaScript syntax on the corrected authorized implementation. This is offline/static evidence only and is not a substitute for real model preparation and inference.

## Final workflow evidence

- Pull request: `#35`
- Candidate branch: `agent/windows-real-model-e2e`
- Candidate branch head: recorded in the current `popex-agent-handoff:v1` block
- Manual workflow run number: none
- Manual workflow run ID: none
- Result: **blocked before dispatch**
- Blocker: GitHub requires a `workflow_dispatch` workflow to exist on the default branch before the manual dispatch UI/API can run it. This new workflow is intentionally confined to the unmerged issue branch, and this agent is not authorized to merge it or alter an existing default-branch workflow.
- Runtime profile: `windows-x86_64-cpu-cpython313`
- Real model preparation performed: no
- Real CPU inference performed: no
- Inference replacement with mocks: prohibited and not performed
- Runtime, cache, database, model, and media artifact upload: none

The orchestrator must first make the manual workflow available from the default branch through an authorized integration step, then dispatch the exact integrated code. If that real run exposes a runtime, model-host, checkpoint, or inference incompatibility, preserve its safe logs and keep the pull request blocked. Do not weaken versions, hashes, model identity, CPU-only requirements, or replace inference with a fake result.
