<!-- popex-agent-handoff:v1 -->
```json
{
  "schemaVersion": 1,
  "agent": "Agent <number or role>",
  "cycle": "<cycle identifier>",
  "status": "running",
  "branch": "agent/<branch>",
  "baseSha": "<40-character lowercase SHA>",
  "headSha": "<40-character lowercase SHA>",
  "pr": 0,
  "ownedPaths": ["path/or/glob"],
  "unauthorizedFilesChanged": false,
  "ci": {
    "status": "pending",
    "run": null
  },
  "mergeRecommendation": "hold"
}
```

> Keep this block current. Valid status values are `running`, `blocked`, `ready_for_review`, and `superseded`. Set `mergeRecommendation` to `merge` only when the final head is reviewed and CI is green.

## Scope

What this PR implements and deliberately does not implement.

## Files changed

- `path` — created/modified/deleted — purpose

## Public or integration contract

Exact functions, schemas, endpoints, commands, artifacts, or UI fields that another branch must consume.

## Validation

- `exact command`
  - Result:
  - Count/output:
  - Environment:
  - Warnings or skipped checks:

## Integration notes

How the orchestrator or next owner should use this work. State dependencies on other PRs explicitly.

## Risks and blockers

Concrete remaining risks, blockers, compatibility limits, and assumptions. Use `None identified` only after checking.

## Final recommendation

Choose one:

- `READY FOR MERGE`
- `READY FOR INTEGRATION REVIEW`
- `COMPLETE WITH CONDITIONS`
- `BLOCKED`

Include the reason and confirm whether the branch remains unmerged.