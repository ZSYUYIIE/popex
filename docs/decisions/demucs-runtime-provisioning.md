# Demucs runtime provisioning decision

- **Status:** accepted
- **Decision date:** 2026-08-03
- **PopEx baseline:** `main` at `053d1a5d8fcf9bc4b35e9dc9b24c1de4ec28dfc5`
- **Prior licensing decision:** PR #6 merged as `a952abe20380daf7dc117e0df948ec1af956a02e`
- **Separation-service contract inspected:** draft PR #8 at `e8119657c44f44d4c4cee3e6610789e24fbe955c`
- **Demucs release:** `demucs==4.1.0`, released 2026-07-11
- **Demucs release tag:** `v4.1.0`, commit `6a604bb002d12c4fbabb303ba64db40b5c5743f0`
- **Audited model repository:** `adefossez/HTDemucs`
- **Audited model revision:** `bf35a81b663819a8255c8fefee17f9d812b786b5`
- **Audited checkpoint:** `955717e8.safetensors`
- **Audited checkpoint size:** `84025440` bytes
- **Audited checkpoint SHA-256:** `d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd`

## Official-source references

Exact-release evidence:

- Demucs 4.1.0 release metadata and distribution hashes: https://pypi.org/project/demucs/4.1.0/
- Demucs 4.1.0 tag: https://github.com/adefossez/demucs/tree/v4.1.0
- Release version: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/__init__.py
- Hugging Face loader: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/hf.py
- Pretrained model routing: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/pretrained.py
- Local and legacy repositories: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/repo.py
- Public separation API: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/api.py
- Audio output helpers: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/audio.py
- Bag-of-models class: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/apply.py
- CLI entry point and parser: https://github.com/adefossez/demucs/blob/v4.1.0/demucs/__main__.py and https://github.com/adefossez/demucs/blob/v4.1.0/demucs/separate.py
- Package metadata: https://github.com/adefossez/demucs/blob/v4.1.0/pyproject.toml

Model and cache evidence:

- Official model repository: https://huggingface.co/adefossez/HTDemucs
- Exact audited revision: https://huggingface.co/adefossez/HTDemucs/tree/bf35a81b663819a8255c8fefee17f9d812b786b5
- Exact bag definition: https://huggingface.co/adefossez/HTDemucs/blob/bf35a81b663819a8255c8fefee17f9d812b786b5/htdemucs.yaml
- Exact checkpoint: https://huggingface.co/adefossez/HTDemucs/blob/bf35a81b663819a8255c8fefee17f9d812b786b5/955717e8.safetensors
- Revision-pinned downloads: https://huggingface.co/docs/huggingface_hub/en/guides/download
- `hf_hub_download` reference: https://huggingface.co/docs/huggingface_hub/en/package_reference/file_download
- Cache refs, blobs, and snapshots: https://huggingface.co/docs/huggingface_hub/en/guides/manage-cache
- Cache, offline, token, and telemetry environment variables: https://huggingface.co/docs/huggingface_hub/en/package_reference/environment_variables

PopEx evidence:

- Merged licensing and packaging gate: https://github.com/ZSYUYIIE/popex/pull/6
- Draft separation service and manifest contract: https://github.com/ZSYUYIIE/popex/pull/8

## Problem statement

PopEx needs a local optional runtime that can prove it is loading the audited checkpoint rather than merely loading a model called `htdemucs`. The mechanism must:

1. contact the model host only after explicit user authorization;
2. request only the two required files from the exact audited commit;
3. verify the checkpoint SHA-256 before any tensor or model loading;
4. avoid Demucs's generic-name fallback to legacy AWS `.th` files;
5. reuse the exact verified files without HTTP when offline;
6. keep Demucs and PyTorch isolated from the base PopEx environment;
7. constrain all input and output paths;
8. return structured provenance that can be persisted in the stem manifest;
9. preserve all completed earlier artifacts and prior successful stems after every failure.

This spike decides the mechanism only. It does not implement the worker, backend routes, dependency installation, runtime profiles, user interface, or production configuration.

## Exact 4.1.0 source findings

The exact release tag reports `__version__ = "4.1.0"`. PyPI identifies the release date as 2026-07-11 and publishes a universal wheel plus source archive. The exact Git tag resolves to `6a604bb002d12c4fbabb303ba64db40b5c5743f0`.

Current upstream `main` is three documentation-only commits ahead of `v4.1.0`. The relevant Python source files inspected for this decision have the same Git blob identities at the release tag and current `main`. The source-level conclusions below therefore come from the exact release and do not rely on later upstream behavior.

### Hugging Face routing

`demucs.pretrained.get_model` treats names beginning with `hf://` specially. It directly calls `demucs.hf.get_hf_model` and returns the result. It does not enter the generic-name exception handler or legacy repository path. Therefore an `hf://` model name prevents generic AWS fallback.

A generic name such as `htdemucs` first attempts Hugging Face, catches every exception, and then falls back to the legacy remote repository. PopEx must not use the generic name as its trust boundary.

### Missing revision support

`demucs.hf.get_hf_model` calls:

```python
hf_hub_download(repo_id, f"{name}.yaml")
hf_hub_download(repo_id, f"{sig}.safetensors")
```

Neither call supplies `revision`, `cache_dir`, `local_files_only`, or `token`. Demucs's `hf://` syntax accepts a namespace and model name, but the Demucs parser does not define a model-revision argument and does not pass a revision through to the Hub client.

Consequently, `python -m demucs -n hf://adefossez/htdemucs ...` avoids AWS fallback but does not pin the audited revision.

### Exact audited bag

At the audited model revision, `htdemucs.yaml` contains one model signature: `955717e8`. The corresponding checkpoint is `955717e8.safetensors`, with the audited SHA-256 and size recorded above.

The worker must reject a bag that includes any different or additional signature. It must not discover checkpoint names from a mutable branch and then trust them dynamically.

### Safetensors loading

`demucs.hf.load_safetensors_model(path)` uses `safetensors.safe_open`, reads model structure metadata, imports the named installed Demucs class, reconstructs the state, and calls Demucs's model-state loader. It does not download a file and does not use PyTorch pickle deserialization.

The audited file hash pins the entire safetensors file, including its metadata. The worker must still validate the loaded result before inference: expected class family, stereo input, 44.1 kHz sample rate, and the ordered sources `drums`, `bass`, `other`, `vocals`.

### Local `--repo` behavior

`demucs.repo.LocalRepo.scan` recognizes only files whose suffix is `.th`. `BagOnlyRepo` can read local YAML files, but every model signature in a bag is resolved through that `.th`-only repository. The exact 4.1.0 `--repo` path cannot load the audited safetensors checkpoint.

Converting the audited safetensors file into `.th` would create a new, unaudited artifact and restore a pickle-style loading path. That is not an acceptable substitute.

### Legacy loader

`demucs.repo.RemoteRepo` uses:

```python
torch.hub.load_state_dict_from_url(
    url,
    map_location="cpu",
    check_hash=True,
    weights_only=False,
)
```

This is the legacy path that the integration must exclude. Its filename hash checking does not make it equivalent to the audited Hugging Face safetensors artifact.

## Alternatives evaluated

| Alternative | Exact revision pinned | Audited SHA verified before load | Guaranteed offline selection | Legacy fallback excluded | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| A. Direct high-level CLI with `hf://` | No | No | No | Yes | Rejected |
| B. Pre-populated Hub cache plus ordinary `hf://` loader | Not reliably | Possible during preparation only | Depends on mutable cache ref | Yes | Rejected |
| C. Local Demucs `--repo` | Not applicable | Not for audited safetensors | Yes for `.th` only | Yes | Rejected |
| D. Narrow PopEx optional-runtime worker | Yes | Yes | Yes | Yes | Recommended |
| E. Upstream patch or maintained fork | Yes after modification | Possible | Yes | Yes | Deferred last resort |

### A. Direct high-level CLI

Evaluated form:

```text
python -m demucs -n hf://adefossez/htdemucs ...
```

Positive finding: `hf://` bypasses the generic exception-and-fallback path.

Blocking findings:

- Demucs 4.1.0 exposes no way to pass the audited commit hash.
- The internal downloads omit `revision`, which officially means the current `main` revision.
- An online invocation can refresh the Hub cache's `refs/main` mapping to a newer commit.
- An offline invocation selects whatever commit the local `refs/main` currently identifies; the model name alone does not prove that this is the audited commit.
- The CLI does not return the resolved repository commit or checkpoint digest in a stable machine-readable result.

Decision: reject the direct CLI as the production trust path. It remains useful only for unpinned manual experimentation outside PopEx's audited workflow.

### B. Pre-populated Hugging Face cache

Evaluated sequence:

1. use `hf_hub_download(..., revision=<audited commit>)` to pre-populate the exact snapshot;
2. verify the checkpoint;
3. set offline mode;
4. invoke the ordinary Demucs `hf://` loader, whose internal calls omit `revision`.

Official cache documentation separates:

- `blobs`, which hold file content;
- `snapshots/<commit>`, which expose files for a commit;
- `refs/main`, which maps the branch name to the latest known commit.

A full commit hash can resolve its own snapshot without consulting `refs/main`. The ordinary Demucs loader does not request that commit hash; it requests the default branch.

A temporary no-network experiment with `huggingface_hub==1.16.1` created two synthetic cached revisions and no model weights. It confirmed:

- an exact commit request resolved the exact snapshot;
- an unqualified request failed when no `refs/main` existed;
- an unqualified request selected the audited snapshot when `refs/main` was manually pointed to it;
- moving `refs/main` caused the same unqualified request to select the other snapshot.

This experiment illustrates the documented cache model; it is not a substitute for official documentation and does not select a runtime dependency version.

Making `refs/main` point to the audited commit would be manual cache manipulation. It is not an API contract for pinning, can be changed by a later online branch lookup, and creates a hidden dependency between the preparation code and Demucs's unqualified loader.

Decision: reject cache manipulation. PopEx may use the official Hub cache, but every worker lookup must use the full audited commit explicitly.

### C. Local Demucs `--repo`

The exact release's local repository scans `.th` files only. A YAML bag can refer only to signatures found as `.th` files. It cannot use `955717e8.safetensors` through the supported `--repo` path.

Decision: reject. Do not convert, rename, or reserialize the checkpoint to fit this interface.

### D. Narrow PopEx optional-runtime worker

The recommended worker runs only inside a separately installed and locked Demucs/PyTorch runtime profile. The base PopEx application communicates with it through a subprocess protocol and never imports Demucs, PyTorch, safetensors, or Hugging Face.

The worker performs these operations:

1. set cache, privacy, token, telemetry, and offline variables before any Hub import;
2. call the public `huggingface_hub.hf_hub_download` function for exactly:
   - repository `adefossez/HTDemucs`;
   - revision `bf35a81b663819a8255c8fefee17f9d812b786b5`;
   - file `htdemucs.yaml`;
   - file `955717e8.safetensors`;
   - explicit PopEx cache root;
   - `token=False`;
3. parse the YAML with `yaml.safe_load` and require exactly `models: ["955717e8"]` with no unexpected model signature;
4. require checkpoint size `84025440` bytes and compute the complete SHA-256 before loading;
5. atomically publish a local readiness manifest only after verification;
6. on every separation, resolve the same full commit in offline/local-only mode and recompute the checkpoint digest before model loading;
7. load the local file through the exact-release `demucs.hf.load_safetensors_model` function;
8. construct `demucs.apply.BagOfModels` from the verified single model and verified bag data;
9. use the public `demucs.api.Separator` separation and audio-loading behavior through a minimal version-pinned adapter that injects the already loaded model;
10. write stems using `demucs.api.save_audio` into only the allocated unpublished output directory;
11. return one structured JSON result containing exact runtime and model provenance.

#### Required upstream functions

Public API:

- `huggingface_hub.hf_hub_download`
- `demucs.api.Separator`
- `demucs.api.save_audio`

Semi-public or internal API:

- `demucs.hf.load_safetensors_model`
- `demucs.apply.BagOfModels`
- `Separator._load_model` as the narrow subclass injection hook

`load_safetensors_model` and `BagOfModels` are named, importable functions/classes without leading underscores, but they are not documented as stable external APIs. `_load_model` is explicitly private. Their use is acceptable only because the runtime pins `demucs==4.1.0`, the adapter is small, and exact source-contract tests will fail before any upgrade is accepted.

The recommended injection adapter subclasses `Separator`, stores the preloaded verified bag before calling `Separator.__init__`, and overrides `_load_model` only to set `_model`, `_audio_channels`, and `_samplerate` from that bag. It then uses the inherited `separate_audio_file` method. This avoids the built-in model lookup entirely.

No upstream inference or audio algorithm needs to be copied. The adapter itself is original PopEx code. If a later implementation copies substantial Demucs source instead, it must preserve the Demucs MIT copyright and permission notice in that copied file and in release notices.

Decision: approve this architecture. It is safer and more explicit than cache-ref manipulation, and it does not require PopEx to distribute a modified Demucs package.

### E. Upstream patch or maintained fork

The minimum useful upstream change would add keyword-only `revision`, `cache_dir`, `local_files_only`, and `token` controls to `get_hf_model`, pass them to both `hf_hub_download` calls, and thread a model revision through `get_model`, `Separator`, and a CLI option such as `--model-revision`.

A PopEx fork would then need to:

- publish and pin a modified package;
- preserve the Demucs MIT notice;
- maintain the patch against a low-activity upstream;
- repeat source, package, and security review on every rebase;
- ensure the CLI reports resolved provenance and verifies PopEx's independently audited digest.

A narrow upstream contribution may be worthwhile, but PopEx does not need to wait for it. Maintaining a fork solely for revision plumbing costs more than the isolated adapter.

Decision: defer. Reconsider only if the exact-release semi-public loading functions prove unusable during implementation testing.

## Security analysis

The worker trust boundary is an allowlist, not a user-selectable model browser.

Required constants:

```text
repository = adefossez/HTDemucs
revision = bf35a81b663819a8255c8fefee17f9d812b786b5
bag file = htdemucs.yaml
model signatures = [955717e8]
checkpoint file = 955717e8.safetensors
checkpoint size = 84025440
checkpoint SHA-256 = d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd
```

The web client must not supply or override these values.

The worker must:

- pass `token=False` to every public-model download;
- remove inherited `HF_TOKEN` and legacy token variables from its environment;
- set `HF_HUB_DISABLE_IMPLICIT_TOKEN=1` before import;
- set `HF_HUB_DISABLE_TELEMETRY=1` before import;
- set `HF_HUB_DISABLE_UPDATE_CHECK=1` before import;
- never log environment variables, request headers, tokens, absolute paths, or Hub cache internals;
- hash the completed regular file before opening it with safetensors;
- reject symlinks or paths that resolve outside the configured cache or output root;
- reject modified YAML, additional model signatures, unexpected source labels, unexpected sample rate, and unexpected channel count;
- use no generic Demucs model name and no legacy `--repo` or AWS path;
- publish readiness and success records atomically;
- sanitize exception text before returning it to the base application.

Safetensors avoids the legacy `.th` pickle-loading path. The exact Demucs loader imports a class name contained in safetensors metadata. Because the complete file digest is pinned, that metadata is part of the audited artifact. The worker must not load a different checkpoint merely because it is also a safetensors file.

## Revision-pinning analysis

The full 40-character commit hash is the only approved revision identifier. A branch, tag, short hash, cache ref, or current model repository head is insufficient.

Preparation uses:

```python
hf_hub_download(
    repo_id="adefossez/HTDemucs",
    filename="htdemucs.yaml",
    revision="bf35a81b663819a8255c8fefee17f9d812b786b5",
    cache_dir=cache_root,
    token=False,
)
```

and the equivalent call for `955717e8.safetensors`.

Verification and separation use the same repository, files, and full revision with `local_files_only=True`, after setting `HF_HUB_OFFLINE=1` before importing `huggingface_hub`.

The worker uses the returned exact-snapshot paths directly. It does not call Demucs's unqualified Hub loader and does not read or write `refs/main` itself.

## Cache and offline analysis

The cache is an implementation detail owned by the optional runtime. The base application knows only:

- the configured cache root;
- the relative path of the readiness manifest;
- readiness status and approximate storage;
- the operation to remove the model cache safely.

The readiness manifest records cache-root-relative POSIX paths. It must reject absolute paths, `..`, NUL characters, alternate path separators, and symlink escape. These paths are not returned through the web API.

Offline verification must satisfy all of the following:

1. `HF_HUB_OFFLINE=1` is set before importing `huggingface_hub`;
2. exact-revision `hf_hub_download` calls use `local_files_only=True`;
3. returned paths remain inside the configured cache root;
4. both expected regular files exist;
5. YAML content identifies only `955717e8`;
6. checkpoint size and full SHA-256 match;
7. the readiness manifest matches the installed runtime profile and package versions.

If any condition fails, `offlineReady` is false and separation is not started. No online fallback is permitted from an offline command.

Deleting the model cache invalidates readiness but must not delete source media, `analysis.wav`, metadata, audio-analysis JSON, stem manifests, or prior stem WAVs.

## Recommended architecture

Use an executable named `popex-demucs-worker`, installed only in each optional separation runtime profile. The base application receives the executable path from trusted local configuration.

The base application invokes the worker with `shell=False`, a minimal environment, captured UTF-8 stdout/stderr, a timeout, and no inherited credentials. It never invokes `python -m demucs` directly for production separation.

The worker emits exactly one JSON object on stdout. Progress and sanitized diagnostics may use stderr or a later dedicated progress channel, but stdout remains machine-readable. The base application rejects extra stdout text, invalid UTF-8, non-finite JSON numbers, unsupported protocol versions, and unknown result fields when they affect trust decisions.

PR #8's isolated run-directory, immutable prior-run, output validation, path containment, and atomic publication behavior remain valid. The future integration changes its runner from the direct Demucs CLI to this worker and obtains model provenance from the worker result rather than using the model name as the weight identifier.

## Rejected architectures

The following are explicitly rejected:

- generic `htdemucs` CLI invocation;
- direct `hf://adefossez/htdemucs` CLI invocation as the audited production path;
- manually pinning or rewriting Hub `refs/main`;
- relying on whatever revision is newest in a pre-populated cache;
- using Demucs `--repo` with converted or renamed checkpoint files;
- loading legacy AWS `.th` checkpoints;
- loading `.th` with `weights_only=False`;
- bundling the checkpoint in Git, base packages, Docker images, or ordinary releases;
- importing Demucs, PyTorch, or Hugging Face in the base application merely to probe capability;
- copying Demucs inference logic when the version-pinned adapter can call the installed implementation;
- maintaining a Demucs fork before the worker path has been implementation-tested.

## Worker protocol

### Envelope

Every completed worker command writes one object:

```json
{
  "protocolVersion": 1,
  "command": "runtime-probe",
  "status": "ok",
  "result": {},
  "warnings": []
}
```

Failures use:

```json
{
  "protocolVersion": 1,
  "command": "separate",
  "status": "error",
  "error": {
    "code": "MODEL_NOT_READY",
    "message": "The verified htdemucs model is not available in this runtime.",
    "retryable": true
  },
  "warnings": []
}
```

Messages must contain no absolute paths, tokens, request URLs containing credentials, or raw tracebacks.

### Runtime availability probe

```text
popex-demucs-worker --protocol-version 1 runtime-probe
```

Requirements:

- uses only the standard library and `importlib.metadata`;
- does not import Demucs, PyTorch, safetensors, YAML, or Hugging Face;
- performs no network request;
- returns Python version, worker version, runtime profile identifier, and installed distribution versions for Demucs, PyTorch, Hugging Face Hub, safetensors, and PyYAML;
- returns an incompatibility error when `demucs!=4.1.0` or the profile does not match its lock record.

If the configured executable does not exist or cannot start, the base application records `RUNTIME_MISSING` without a worker exit code.

### Model readiness probe

```text
popex-demucs-worker --protocol-version 1 model-probe --cache-root <trusted-root>
```

Requirements:

- standard-library-only;
- no import of Demucs, PyTorch, or Hugging Face;
- no network request;
- validates the readiness-manifest schema, profile, safe relative asset paths, file type, containment, expected size, and manifest digest values;
- may return `MODEL_DOWNLOAD_REQUIRED` without treating it as a failed separation;
- separation still performs full offline hash verification immediately before loading.

### Explicit prepare/download action

```text
popex-demucs-worker --protocol-version 1 prepare-model --cache-root <trusted-root>
```

This command may be invoked only after an explicit user authorization recorded by the base application.

Requirements:

- set environment controls before Hub import;
- download only the two allowlisted files from the full commit;
- use `token=False`;
- verify YAML, size, and SHA-256;
- publish the readiness manifest atomically;
- return `MODEL_READY` and exact provenance;
- not start separation unless the base application's explicit action requested both preparation and separation.

### Offline verification action

```text
popex-demucs-worker --protocol-version 1 verify-model --cache-root <trusted-root>
```

Requirements:

- set offline mode before import;
- use exact revision and `local_files_only=True`;
- make no HTTP request;
- fully rehash the checkpoint;
- update `verifiedAt` atomically only after successful verification;
- return `MODEL_DOWNLOAD_REQUIRED` for an absent cache and `MODEL_VERIFICATION_FAILED` for a mismatched or unsafe cache.

### Separation invocation

```text
popex-demucs-worker --protocol-version 1 separate \
  --cache-root <trusted-cache-root> \
  --workspace-root <trusted-job-root> \
  --input-relative analysis.wav \
  --output-relative stems/runs/<run-id>/demucs-output \
  --device <cpu|cuda|mps>
```

Requirements:

- exact offline verification before model load;
- input must be the regular file `analysis.wav` inside the trusted workspace root;
- output must be a new or empty allocated directory inside the current unpublished run;
- no arbitrary filename template, repository, revision, checkpoint, device string, or path from a web client;
- output only `vocals.wav`, `bass.wav`, `drums.wav`, and `other.wav` for this profile;
- use the installed Demucs 4.1.0 separation implementation and normal 16-bit WAV output behavior;
- return exact model/runtime provenance and job-root-relative output names;
- never publish or replace the job's stem manifest itself; PR #8's base-side service retains atomic publication ownership.

### Structured success result

A successful separation result includes at least:

```json
{
  "runtimeProfile": "linux-cpu-v1",
  "workerVersion": "1.0.0",
  "demucsVersion": "4.1.0",
  "torchVersion": "<locked-version>",
  "huggingfaceHubVersion": "<locked-version>",
  "modelRepository": "adefossez/HTDemucs",
  "modelRevision": "bf35a81b663819a8255c8fefee17f9d812b786b5",
  "checkpointFile": "955717e8.safetensors",
  "checkpointSha256": "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
  "device": "cpu",
  "outputs": ["vocals.wav", "bass.wav", "drums.wav", "other.wav"]
}
```

### Exit codes

| Exit code | Stable meaning |
| ---: | --- |
| `0` | Command succeeded |
| `10` | Runtime profile missing required or compatible package versions |
| `20` | Model download required or readiness manifest absent |
| `21` | Model verification, checksum, schema, or containment failure |
| `22` | Authorized model download failed because of network, remote, or storage error |
| `30` | Invalid request, unsupported protocol, unsafe path, or invalid input |
| `40` | Demucs separation or output generation failed |
| `41` | Worker observed cancellation |
| `42` | Worker-enforced timeout |
| `50` | Unexpected internal worker failure |

A missing executable is detected by the base application and mapped to `RUNTIME_MISSING`; it is not exit code 10.

### Environment variables

The launcher supplies a minimal environment. Set before importing Hub or Demucs modules:

- `HF_HOME` to the PopEx-owned model-runtime directory;
- `HF_HUB_CACHE` to its Hub subdirectory;
- `HF_XET_CACHE` to its Xet subdirectory;
- `HF_HUB_DISABLE_IMPLICIT_TOKEN=1`;
- `HF_HUB_DISABLE_TELEMETRY=1`;
- `HF_HUB_DISABLE_UPDATE_CHECK=1`;
- `HF_HUB_DISABLE_PROGRESS_BARS=1` for machine-readable operation;
- `PYTHONNOUSERSITE=1`;
- `HF_HUB_OFFLINE=1` for `verify-model` and `separate`.

The launcher removes `HF_TOKEN`, `HUGGING_FACE_HUB_TOKEN`, and other unrelated credential variables. Public downloads also pass `token=False` explicitly.

### Cache directory semantics

The cache root belongs to one installed runtime profile. Its readiness manifest is stored outside mutable Hub `refs` but inside the PopEx-owned model-runtime directory. The manifest refers to Hub assets only with cache-root-relative paths.

Several profiles may share an exact read-only model snapshot only after cross-platform path and locking tests. The first implementation should use one cache root per profile to avoid concurrent cache mutation and Windows symlink differences.

### Cancellation and timeout

The base application owns the process timeout. On cancellation it first requests graceful termination, then force-kills after a bounded grace period. POSIX and Windows process-group behavior must be tested separately.

The worker installs a signal handler and uses Demucs's chunk callback to raise cancellation at a safe boundary. It writes only to the current unpublished output directory. The caller removes only that failed run after containment checks. Prior published runs and their manifest remain untouched.

If the process is killed before returning JSON, the base application maps the observed termination to cancellation or timeout and sanitizes all diagnostics.

### No-download capability checks

`runtime-probe` and `model-probe` are the only startup or passive checks. They import no network-capable optional library and cannot download a model. `verify-model` and `separate` are explicitly offline. Only `prepare-model` may make an HTTP request.

## Readiness-manifest schema

Recommended file name:

```text
readiness/htdemucs-bf35a81b-v1.json
```

Schema:

```json
{
  "schemaVersion": 1,
  "protocolVersion": 1,
  "runtimeProfile": "linux-cpu-v1",
  "workerVersion": "1.0.0",
  "demucsVersion": "4.1.0",
  "torchVersion": "<locked-version>",
  "huggingfaceHubVersion": "<locked-version>",
  "modelRepository": "adefossez/HTDemucs",
  "modelRevision": "bf35a81b663819a8255c8fefee17f9d812b786b5",
  "bagFile": "htdemucs.yaml",
  "bagModelSignatures": ["955717e8"],
  "checkpointFile": "955717e8.safetensors",
  "checkpointSizeBytes": 84025440,
  "checkpointSha256": "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
  "verifiedAt": "2026-08-03T00:00:00Z",
  "cacheAssets": {
    "bag": "hub/models--adefossez--HTDemucs/snapshots/bf35a81b663819a8255c8fefee17f9d812b786b5/htdemucs.yaml",
    "checkpoint": "hub/models--adefossez--HTDemucs/snapshots/bf35a81b663819a8255c8fefee17f9d812b786b5/955717e8.safetensors"
  },
  "offlineReady": true,
  "warnings": []
}
```

`verifiedAt` records the most recent complete verification, not merely download time. The implementation may add a bag-file digest and installed runtime lock identifier.

The web API may expose readiness, versions, model identity, approximate size, verification time, offline readiness, and warnings. It must not expose `cacheAssets`, the cache root, or any absolute filesystem path.

## Stem-manifest provenance

PR #8's draft schema version 1 records `model.name`, `packageVersion`, `weightsIdentifier`, and `device`, with `weightsIdentifier` equal to the supplied model name. That is insufficient for the audited runtime.

The integration cycle should write a new stem-manifest schema version containing:

```json
{
  "model": {
    "name": "htdemucs",
    "packageVersion": "4.1.0",
    "runtimeProfile": "linux-cpu-v1",
    "repository": "adefossez/HTDemucs",
    "revision": "bf35a81b663819a8255c8fefee17f9d812b786b5",
    "checkpointFile": "955717e8.safetensors",
    "checkpointSha256": "d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
    "weightsIdentifier": "sha256:d9fa14133cfcc034a6758923bb3a8ca9f8dfd0b582134643bbf83f72c17576dd",
    "device": "cpu"
  }
}
```

If PR #8 merges before integration, readers must remain backward-compatible with schema version 1 while new successful runs write the new schema. If PR #8 is revised before merge, it may adopt the stronger provenance contract directly. In either case, provenance values come from the worker's verified result, not from client input or the base application's package environment.

## Explicit-consent flow

The exact first-use sequence is:

```text
base application runs runtime-probe and model-probe with no network
→ user sees model source, exact revision, checkpoint size, total runtime/storage warning,
  license distinction, cache behavior, and confirmation that audio stays local
→ user explicitly authorizes model preparation or separation
→ base application records that action and invokes prepare-model
→ worker downloads only htdemucs.yaml and 955717e8.safetensors at the full audited revision
→ worker verifies the exact bag, file size, and full SHA-256 before any model load
→ worker atomically publishes the readiness manifest
→ worker returns MODEL_READY
→ base application either returns ready state or, when the same explicit action requested
  separation, allocates a new run and invokes separate
→ every separation performs exact offline verification again before loading
→ later runs reuse the exact cached snapshot
→ offline mode performs no HTTP request and fails closed on a cache miss
```

Application startup, health checks, job-list requests, and passive UI polling never invoke `prepare-model`.

## Failure behavior

Provisioning and separation are independent retryable stages. Every failure must preserve:

- retained source media;
- `analysis.wav`;
- source metadata;
- audio-analysis JSON;
- database preparation and analysis state;
- any previously published stem manifest;
- all stem files referenced by the previously published manifest.

Preparation failures do not publish or update readiness. A failed checksum deletes or quarantines only the newly obtained unverified asset after containment checks; it does not trust or load it.

A failed separation writes only inside its allocated unpublished run directory. The base-side separation service may remove only that failed run. It must not replace the prior successful stem manifest.

Failure categories remain distinct and actionable:

- runtime missing;
- runtime version incompatible;
- model download required;
- network unavailable;
- insufficient storage;
- checksum or model-schema mismatch;
- unsafe cache path;
- offline cache miss;
- unsupported device;
- out of memory;
- cancellation;
- timeout;
- invalid or missing `analysis.wav`;
- Demucs inference failure;
- invalid or incomplete stem output.

No failure may trigger generic-name lookup, AWS fallback, alternate checkpoint selection, cloud audio upload, or automatic redownload without explicit authorization.

## Portability implications

The worker protocol is platform-independent; runtime installations are not.

Each supported profile must separately lock and test:

- worker package;
- Python version;
- `demucs==4.1.0`;
- exact PyTorch build and package index;
- exact `huggingface-hub` version;
- safetensors, PyYAML, sphn, and other transitive versions;
- device preflight and fallback behavior.

Minimum initial profiles:

- Linux CPU;
- Windows CPU;
- Apple Silicon MPS.

CUDA profiles require independent driver and wheel-index testing. Intel macOS remains blocked in the shared PopEx environment by the NumPy constraint recorded in the licensing decision.

The worker must not rely on Unix-only symlink behavior. Windows cache behavior, file locking, cancellation, long paths, and process groups require dedicated tests. Cache paths remain private even when symlink support is unavailable and files are duplicated.

No universal `[project.optional-dependencies].separation` declaration is approved by this spike.

## Implementation files proposed for the next cycle

Exact ownership and naming must be assigned before implementation. A narrow next cycle is expected to need:

- an isolated worker package with its own entry point and platform-profile packaging metadata;
- a base-side worker client that runs probes, preparation, verification, and separation with the structured protocol;
- a focused update to PR #8's separation runner and stem-manifest provenance contract;
- configuration for the trusted worker executable and private cache root;
- platform-specific runtime lock or installation definitions after exact-version tests;
- synthetic protocol, containment, cancellation, and failure-preservation tests.

Illustrative paths, not authorized by this documentation branch:

```text
runtime/demucs_worker/pyproject.toml
runtime/demucs_worker/popex_demucs_worker/__main__.py
runtime/demucs_worker/popex_demucs_worker/worker.py
app/separation_runtime.py
app/separation.py
tests/test_demucs_worker_protocol.py
tests/test_separation_runtime.py
```

The integration owner must coordinate ownership with active branches before changing any shared file.

## Tests required

### Source-contract tests

- Assert the installed Demucs version is exactly 4.1.0.
- Assert the expected semi-public functions and private injection hook exist with compatible signatures.
- Assert generic-name lookup is never called by the worker.
- Assert no code path reaches `RemoteRepo` or `torch.hub.load_state_dict_from_url`.
- Assert the exact bag creates one model with the expected four sources, sample rate, and channel count.

### Download and cache tests

- Mock Hub downloads and assert exact repository, full revision, filenames, cache root, and `token=False`.
- Reject a short hash, branch, tag, different repository, different filename, extra bag signature, wrong size, and wrong SHA-256.
- Demonstrate that moving synthetic `refs/main` cannot affect exact-revision worker resolution.
- Prohibit all socket/HTTP access in `runtime-probe`, `model-probe`, `verify-model`, and `separate` tests.
- Test absent, partial, corrupt, symlinked, escaped, and concurrently modified cache states.
- Test atomic readiness publication and safe cache deletion.

### Worker protocol tests

- Validate every command's JSON success and failure schema.
- Validate stable exit codes.
- Reject extra stdout, invalid UTF-8, NaN/Infinity, unsupported protocol versions, unknown commands, arbitrary paths, and unsupported devices.
- Ensure diagnostics redact absolute paths, credentials, and environment values.
- Test missing executable mapping in the base client.

### Separation tests

- Retain PR #8's synthetic output-validation, containment, and failed-retry preservation tests.
- Verify worker provenance is copied exactly into the stem manifest.
- Verify a checksum failure occurs before model loading.
- Verify prior successful stems survive runtime, model, device, timeout, cancellation, and output failures.
- Test cancellation at a Demucs callback boundary and forced termination after grace timeout.
- Test CPU output through one manual or dedicated integration profile using the audited model after explicit download; do not place the model in ordinary CI caches or artifacts.

### Platform tests

- Linux CPU runtime install and separation smoke test.
- Windows CPU runtime install, cache, process-group, and cancellation test.
- Apple Silicon MPS test plus explicit CPU fallback.
- Separate CUDA tests only for profiles PopEx actually supports.
- Python 3.13 test before claiming it for the optional runtime; base-application CI alone is insufficient.

Normal repository CI must continue to run without installing Demucs, PyTorch, or model weights.

## Temporary validation performed

- Inspected exact source at Demucs tag `v4.1.0`; no model file was downloaded.
- Compared `v4.1.0` with current upstream `main`; post-release changes were documentation-only.
- Inspected the official exact model revision, bag, checkpoint LFS metadata, size, and published SHA-256 without downloading the 84 MB checkpoint.
- Ran a synthetic local Hub-cache experiment using `huggingface_hub==1.16.1`; it created only tiny text blobs and confirmed exact-revision versus mutable-ref selection behavior.
- Attempted to obtain the PyPI source distribution in the local container; the environment could not resolve the file host. The exact Git release tag and official PyPI distribution metadata/hashes were used instead. This network limitation does not alter the source conclusions.

## Remaining blockers

1. Select and lock exact runtime dependency sets for each supported platform profile, including an exact Hugging Face Hub version.
2. Implement the worker and pass all source-contract, security, cache, protocol, cancellation, and preservation tests.
3. Reconcile PR #8's direct CLI invocation and schema version 1 with the worker protocol and exact provenance fields.
4. Add explicit-consent UX and base-side authorization recording before enabling `prepare-model`.
5. Complete at least one CPU end-to-end smoke test with the audited checkpoint without adding the checkpoint to normal CI or release artifacts.
6. Re-review any Demucs, model revision, checkpoint, worker protocol, or semi-public upstream API change before upgrade.

These are implementation acceptance conditions. They do not block beginning the recommended worker cycle. No safe direct high-level CLI or cache-manipulation path was confirmed, but a safe pinned local loading path was confirmed through the exact-revision worker architecture.

## Final gate result

APPROVED — IMPLEMENT THE RECOMMENDED WORKER