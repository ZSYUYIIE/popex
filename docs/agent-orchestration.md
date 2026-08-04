# GitHub-first agent orchestration

PopEx uses GitHub—not chat transcripts—as the canonical coordination layer for concurrent agent work.

The user should not copy long prompts, progress logs, or handoff reports between agents and the orchestrator. Chat may contain a one-line pointer, but the repository issue and pull request hold the complete task and result.

## Cycle workflow

1. The orchestrator creates one GitHub issue per agent, up to four issues per cycle.
2. Each issue contains the full assignment: branch, base, owned paths, prohibited paths, acceptance criteria, validation, and reporting contract.
3. An agent can be started with only: `Work ZSYUYIIE/popex issue #<number>. Use GitHub as the canonical state and do not return a long chat report.`
4. The agent opens a draft pull request as its first remote checkpoint, before long-running installs, downloads, or experiments.
5. The agent keeps the pull-request body current. It records status, base and head SHAs, changed files, contracts, validation, blockers, and integration notes there.
6. When blocked, the agent pushes verified partial work and changes the handoff status to `blocked`; it does not remain invisible with only uncommitted local work.
7. When complete, the agent changes the handoff status to `ready_for_review` and records the exact final CI run.
8. The orchestrator reads issues, pull requests, diffs, comments, reviews, and CI directly through GitHub; it does not ask the user to relay those materials.
9. The orchestrator reviews, requests focused patches, marks ready, merges, and creates the next cycle's issues.

## Canonical locations

| Information | Canonical location |
| --- | --- |
| Full assignment | GitHub issue |
| Branch and file ownership | GitHub issue and PR handoff block |
| Current status | PR handoff JSON |
| Progress/checkpoints | Updated PR body; concise PR comments only when useful |
| Code and artifacts | Agent branch |
| Tests and CI | PR body plus GitHub Actions |
| Review findings | PR review or issue comment |
| Final handoff | PR body |
| Merge decision | GitHub review and merge state |

A chat response such as “PR #21 is ready” is sufficient. The full report must not be duplicated into chat unless the user explicitly asks for it.

## Required pull-request handoff

Agent PRs use the marker:

```text
<!-- popex-agent-handoff:v1 -->
```

Immediately after it, the PR body contains a JSON object matching the repository template. The JSON is the compact machine-readable state; the Markdown sections hold evidence and explanation.

Status values:

- `running`: work is in progress and a remote checkpoint exists;
- `blocked`: verified partial work is pushed and the blocker is documented;
- `ready_for_review`: implementation and required validation are complete;
- `superseded`: another branch or decision replaced the work.

The PR body must include:

- Scope
- Files changed
- Public or integration contract
- Validation
- Integration notes
- Risks and blockers
- Final recommendation

The head SHA in the handoff must match the PR head. A `ready_for_review` handoff requires passing CI on that head.

## Long-running work

Before a long dependency resolution, package installation, model download, build, or integration experiment, the agent must push a checkpoint and update the PR body. If the experiment fails or stalls, the repository still contains the verified partial evidence.

Model weights, caches, private audio, secrets, and temporary runtime directories remain prohibited from Git.

## Orchestrator responsibilities

The orchestrator must:

- create issues instead of asking the user to copy full prompts;
- inspect GitHub directly instead of asking for pasted reports;
- verify claims against diffs and CI;
- keep no more than four concurrent agents per cycle;
- avoid overlapping file ownership;
- merge safe foundations directly when authorized;
- leave dependent work draft until its contracts align;
- create the next issues from the new verified `main` SHA.

## Transitional rule

PRs numbered 13 and below predate this protocol and are exempt from the automated body check. Their existing descriptions remain usable. All later `agent/*` pull requests must use the versioned handoff block.