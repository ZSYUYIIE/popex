# Demucs runtime-client profile smoke

## Status

The draft pull request was opened before any hosted Linux or Windows runtime installation. The base-client validator is implemented; hosted profile installation evidence remains pending.

## Scope

This work validates the merged base `SeparationRuntimeClient` against the separately installed Linux and Windows CPU profiles without downloading or preparing the audited model.

The validation chain is:

```text
install exact CPU profile
-> installer runtime-probe succeeds
-> base-client runtime_probe succeeds
-> base-client model_probe reports MODEL_DOWNLOAD_REQUIRED
-> confirm no model/checkpoint/readiness/cache assets
-> remove temporary runtime and cache
```

## Base-client validator

The repository Python invokes:

```text
python scripts/validate_separation_runtime_profile.py \
  --worker <absolute installed worker executable> \
  --runtime-lock <absolute copied profile lock> \
  --cache-root <absolute empty private cache directory> \
  --expected-profile <profile identifier>
```

The validator imports only Python's standard library and `app.separation_runtime`. It requires absolute normalized non-symlink paths, constructs `SeparationRuntimeClient`, and accepts success only when:

- the runtime profile matches the requested Linux or Windows identifier;
- `runtimeLockSource` is `profile`;
- worker version is `1.0.0`;
- Demucs version is `4.1.0`;
- the client accepts the installed-versus-locked version set;
- `model_probe()` raises structured broad code `MODEL_DOWNLOAD_REQUIRED` with exit code `20`;
- the cache root remains empty and contains no readiness manifest or checkpoint file.

The script never calls `prepare_model`, `verify_model`, `separate`, or the client callable. Its only stdout is one compact JSON object containing safe version/profile state and no local paths.

## Expected success JSON

```json
{
  "demucsVersion": "4.1.0",
  "modelAssetsCreated": false,
  "modelState": "download_required",
  "runtimeProfile": "linux-x86_64-cpu-cpython313",
  "schemaVersion": 1,
  "torchVersion": "2.13.0+cpu",
  "workerVersion": "1.0.0"
}
```

The Windows job reports the corresponding `windows-x86_64-cpu-cpython313` profile.

## Privacy and cleanup

All workflow steps will set Hugging Face offline/privacy variables before installation and probing. The dedicated workflow will scan both the temporary runtime and cache independently and remove both in an `always()` cleanup step. It will not upload the runtime or any cache as an artifact.

## Current evidence

- Draft PR opened before runtime installation: yes.
- Validator implementation pushed: yes.
- Linux hosted installation: not started.
- Windows hosted installation: not started.
- Model preparation or download: not started and not permitted.
