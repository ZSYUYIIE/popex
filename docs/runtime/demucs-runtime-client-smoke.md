# Demucs runtime-client profile smoke

## Status

Implementation checkpoint for issue #25. The draft pull request must exist before any hosted Linux or Windows runtime installation begins.

## Scope

This work will validate the merged base `SeparationRuntimeClient` against the separately installed Linux and Windows CPU profiles without downloading or preparing the audited model.

The final validation chain is:

```text
install exact CPU profile
-> base-client runtime probe
-> base-client model probe reports MODEL_DOWNLOAD_REQUIRED
-> confirm no model/checkpoint/readiness/cache assets
-> remove temporary runtime and cache
```

No runtime installation evidence has been collected at this checkpoint.
