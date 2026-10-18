# Batching Guide

Use this guide when converting a user task into an execution plan.

## Planning objective

Produce batches that are:

- small enough to verify quickly
- large enough to deliver a meaningful increment
- ordered by dependency and risk
- explicit about what the user must test or review
- explicit about which knowledge artifact must be updated

## First-pass context gathering

Before planning an existing project or initiative:

1. Read `.codex/knowledge/INDEX.md`.
2. Identify the matching artifact file.
3. Read that artifact before defining batches.
4. Carry the artifact path into every batch as the required update target.

## When to split into multiple batches

Split when any of these apply:

- more than one subsystem or package boundary is affected
- refactoring and feature work are mixed
- data/schema changes are involved
- public APIs or contracts change
- verification requires different test layers
- the user likely needs an approval gate before irreversible follow-up work

## Recommended batch types

### Structural batch

Use first when package boundaries, dependency direction, or refactoring foundations are the main risk.

### Feature batch

Use after structure is ready. Keep behavior changes localized.

### Integration batch

Use when wiring new behavior through API, UI, CLI, jobs, or persistence layers.

### Hardening batch

Use for tests, cleanup, documentation, or edge-case fixes that depend on previous batches landing.

## Stack-aware planning

### Spring Boot

- Isolate packaging or dependency-direction refactors before adding behavior when services or packages are already bloated.
- Separate controller, service, persistence, and transaction changes when they can be reviewed independently.
- Treat schema changes, repository query changes, and API contract changes as explicit batch boundaries.
- Prefer verification progression like unit tests -> slice tests -> integration tests.

### Python

- Separate pure logic extraction from framework or I/O wiring changes.
- Isolate module-boundary cleanup before broader feature work when imports or ownership are unclear.
- Treat CLI/API/job-runner boundary changes as explicit integration batches.
- Prefer verification progression like unit tests -> adapter/integration tests -> end-to-end flows when needed.

## Required batch template

Use this exact structure when the task is non-trivial:

### Batch N - Title

- Goal:
- Scope:
- Touched areas:
- Stack context:
- Dependencies:
- Implementation notes for $development-agent:
- Verification:
- Knowledge artifact to update:
- User review gate:
- Out of scope:

## Verification policy

- Prefer automated verification when available.
- Name the exact command, suite, or manual check expected.
- If verification cannot be run by Codex, say whether the user must test, review, or both.
- Do not mark a batch complete without a verification statement.

## Knowledge update policy

After each completed batch:

- append a short dated note to the artifact file
- capture what changed
- capture the key decision or constraint if one was introduced
- keep notes terse and cumulative
- do not defer the note to a later batch

## Handoff policy

When selecting the next batch, include a compact prompt the development agent can execute directly. It should contain:

- the batch title
- the goal
- hard constraints
- exact files or areas to inspect first, if known
- required verification
- required knowledge artifact update path
