# Demucs runtime-client profile smoke

## Status

Validated on hosted Linux and Windows runners. The draft pull request was opened before either runtime installation, and no audited model asset was downloaded, prepared, verified, or used.

## Scope

This work validates the merged base `SeparationRuntimeClient` against the separately installed Linux and Windows CPU profiles without downloading or preparing the audited model.

The validated chain is:

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
- `model_probe()` raises structured broad and worker code `MODEL_DOWNLOAD_REQUIRED` with exit code `20`;
- the cache root remains empty and contains no readiness manifest or checkpoint file.

The script never calls `prepare_model`, `verify_model`, `separate`, or the client callable. Its only stdout is one compact JSON object containing safe version/profile state and no local paths.

## Hosted evidence

Dedicated workflow `Demucs runtime client profile smoke` run 2, ID `30888687504`, validated both jobs from checkpoint `ba17463b8e9718e23ff2af29cd8273013375af06`.

### Linux

- Job ID: `91925512985`
- Runner: Ubuntu 24.04.4 LTS
- Python: CPython 3.13.14
- Profile: `linux-x86_64-cpu-cpython313`
- Installer: `scripts/install_demucs_linux_cpu.sh`
- Installer runtime probe: compatible; lock source `profile`; installed versions equal locked versions
- Worker: `1.0.0`
- Demucs: `4.1.0`
- PyTorch: `2.13.0+cpu`
- Hugging Face Hub: `1.26.0`
- safetensors: `0.8.0`
- PyYAML: `6.0.3`

Base-client result:

```json
{"demucsVersion":"4.1.0","modelAssetsCreated":false,"modelState":"download_required","runtimeProfile":"linux-x86_64-cpu-cpython313","schemaVersion":1,"torchVersion":"2.13.0+cpu","workerVersion":"1.0.0"}
```

Independent runtime/cache scanning passed, the private cache remained empty, and the `always()` cleanup completed.

### Windows

- Job ID: `91925512948`
- Runner: Microsoft Windows Server 2025 Datacenter 10.0.26100
- Runner image: `windows-2025-vs2026` version `20260728.188.1`
- Python: CPython 3.13.14
- Profile: `windows-x86_64-cpu-cpython313`
- Installer: `scripts/install_demucs_windows_cpu.ps1`
- Installer runtime probe: compatible; lock source `profile`; installed versions equal locked versions
- Worker: `1.0.0`
- Demucs: `4.1.0`
- PyTorch: `2.13.0+cpu`
- Hugging Face Hub: `1.26.0`
- safetensors: `0.8.0`
- PyYAML: `6.0.3`

Base-client result:

```json
{"demucsVersion":"4.1.0","modelAssetsCreated":false,"modelState":"download_required","runtimeProfile":"windows-x86_64-cpu-cpython313","schemaVersion":1,"torchVersion":"2.13.0+cpu","workerVersion":"1.0.0"}
```

Independent runtime/cache scanning passed, the private cache remained empty, and the `always()` cleanup completed.

## Repository validation

Repository CI run 129, ID `30888687516`, passed on the same implementation checkpoint:

- `pytest`: `388 passed, 1 warning in 26.60s`
- `python -m compileall -q app tests`: passed
- `node --check app/static/app.js`: passed

The warning is the existing Starlette/FastAPI test-client deprecation concerning a future `httpx2` migration.

The 15 permanent cases in `tests/test_separation_runtime_profile_smoke.py` are included in the full passing test count. The validator module is imported by those tests and compiled successfully.

## Privacy and cleanup conclusions

- Hugging Face offline and privacy variables were set before all installation and probe commands.
- `prepare-model`, `verify-model`, and `separate` were not invoked.
- No `955717e8.safetensors`, other `.safetensors`, `.th`, `.ckpt`, or `htdemucs-bf35a81b-v1.json` was created.
- No model cache content was created.
- The temporary runtime, cache, and safe summary were removed on both platforms.
- No runtime or cache artifact was uploaded.
- No base PopEx dependency, profile lock, hash, worker source, client source, or installer protection was changed.

## Integration conclusion

The merged base client can launch both validated external CPU runtimes, accept each exact profile lock, and safely identify the missing audited model as a user-actionable download requirement. The base application remains Demucs/PyTorch-free, and model preparation remains a separate explicit-consent operation.
