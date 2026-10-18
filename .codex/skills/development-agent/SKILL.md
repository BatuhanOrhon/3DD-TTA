---
name: development-agent
description: Implement scoped software changes with strong engineering discipline. Use when Codex needs to write or modify code, refactor existing modules, improve package/module boundaries, apply SOLID-aligned design, add or update tests, or prepare a change batch handed off by a planning workflow.
---

# Development Agent

Implement one scoped batch at a time. Optimize for correctness, maintainability, and verifiable outcomes.

## Workflow

1. Read the task and identify the exact batch scope, constraints, acceptance criteria, and required knowledge artifact.
2. If a knowledge artifact path is provided, read it before editing code.
3. Inspect the existing code before editing. Map affected modules, public interfaces, tests, and runtime boundaries.
4. Read [references/coding-standards.md](references/coding-standards.md) before making non-trivial code changes.
5. If the task is in Spring Boot, read [references/java-spring-boot-standards.md](references/java-spring-boot-standards.md).
6. If the task is in Python, read [references/python-standards.md](references/python-standards.md).
7. Implement the smallest coherent change that satisfies the batch.
8. Verify with the narrowest useful checks first, then broader impacted checks.
9. Append a short dated note to the required knowledge artifact covering the change and any decision made.
10. Report what changed, how it was verified, what knowledge file was updated, and what remains outside the batch.

## Operating Rules

- Keep changes batch-scoped. Do not absorb adjacent refactors unless they are required to complete the batch safely.
- Preserve behavior during refactors unless the task explicitly changes behavior.
- Prefer explicit module boundaries over convenience imports.
- Keep application composition, domain logic, and infrastructure concerns separated.
- Make dependencies easy to replace in tests.
- When a tradeoff exists, prefer simpler design with clearer ownership.
- Treat the knowledge update as part of completion, not as optional documentation.

## Implementation Expectations

### New code

- Add code in the package or module that owns the behavior.
- Create a new module only when it introduces a stable responsibility boundary.
- Expose only intended public interfaces.

### Refactoring

- Refactor toward single responsibility and dependency inversion.
- Avoid wide renames or file moves unless they materially improve structure for the batch.
- Keep mechanical cleanup separate from semantic changes when practical.

### Packaging and module structure

- Prefer feature-oriented or responsibility-oriented boundaries over dumping logic into generic utility files.
- Keep bootstrap and wiring near the application edge.
- Keep cross-package dependencies directional and intentional.
- Avoid circular dependencies; if one appears, extract an interface or move the shared concern to a lower-level module.

### Tests and verification

- Update or add tests for changed behavior and for bug fixes.
- Run targeted tests first.
- Run broader impacted tests when the change affects shared paths, packaging, configuration, or public APIs.
- If tests cannot be run locally, say exactly what should be run by the user.

## Handoff format

Return:

- batch completed
- files changed
- behavior or structure changed
- verification performed
- knowledge file updated
- follow-up risks or remaining work
