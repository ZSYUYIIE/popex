---
name: Agent task
about: Assign one isolated task to a concurrent PopEx agent
title: "[Agent] "
labels: ""
assignees: ""
---

<!-- popex-agent-task:v1 -->

## Assignment identity

- Cycle:
- Agent:
- Repository: `ZSYUYIIE/popex`
- Base branch: `main`
- Base SHA:
- Branch to create:
- Work independently and simultaneously: Yes
- Merge your own PR: No

## Goal

State one narrow, independently testable outcome.

## Context and invariants

Link the relevant README, AGENTS, decision, API, or product constraints.

## Exclusive ownership

May create or modify:

- `path`

Must not modify:

- `path`

## Required contract

Specify exact functions, schemas, commands, endpoints, artifacts, states, or UI fields.

## Acceptance criteria

1. 
2. 
3. 

## Validation

```bash
pytest
python -m compileall -q app tests
node --check app/static/app.js
```

Add focused commands required by the scope.

## GitHub reporting contract

- Open a draft PR as the first remote checkpoint.
- Use the repository PR template and keep its `popex-agent-handoff:v1` JSON current.
- Before long-running dependency or integration work, push a checkpoint.
- Put the full progress and final handoff in the PR body, not in chat.
- When blocked, push verified partial work and set status to `blocked`.
- When complete, set status to `ready_for_review`, record exact final CI, and keep the PR unmerged.
- A chat response may contain only the PR number, final head, and one-line status.

## Deliverable

- Focused commits
- Pushed branch
- Draft PR against `main`
- Current PR handoff body
- No unauthorized files, secrets, caches, model weights, private audio, or copyrighted recordings