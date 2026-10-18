# Coding Standards

Use these standards whenever implementing or refactoring code with this skill.

## 1. Usage

- Apply this file as the default baseline.
- Read `java-spring-boot-standards.md` for Spring Boot work.
- Read `python-standards.md` for Python work.
- When a repository mixes both stacks, keep the boundary between them explicit.

## 2. Design principles

- Enforce single responsibility at function, class, and module level.
- Keep modules open for extension through composition, not fragile inheritance trees.
- Preserve substitutability when working with interfaces or shared abstractions.
- Split interfaces when consumers need only a subset of behavior.
- Depend on abstractions at boundaries where infrastructure or external systems vary.

## 3. Package and module boundaries

- Group code by feature or responsibility first.
- Keep domain rules independent from transport, framework, storage, and UI details where practical.
- Avoid large `utils` or `helpers` modules that accumulate unrelated logic.
- Export a deliberate public surface. Treat the rest as internal.
- Keep configuration loading and dependency wiring at the edge of the system.

## 4. Refactoring rules

- Change structure with minimal behavior change.
- When a refactor also changes behavior, make the behavioral change explicit in the task summary and tests.
- Prefer extraction over duplication only when the abstraction is real and already proven by repeated use.
- Do not generalize early. Duplicate once if that keeps the design clearer.

## 5. Function and class quality

- Keep functions focused and predictable.
- Make data flow explicit through parameters and return values.
- Avoid hidden global state.
- Prefer plain, intention-revealing names over abbreviated names.
- Keep side effects near the edges and make them obvious.

## 6. Dependency management

- Invert dependencies where business logic currently depends on infrastructure details.
- Prefer constructor or function injection over hard-coded imports when replacement in tests matters.
- Do not introduce a dependency container unless the project already uses one or complexity justifies it.

## 7. Testing expectations

- Add regression tests for bug fixes.
- Add characterization tests before risky refactors when behavior is unclear.
- Test public behavior and contracts more than private implementation details.
- Keep test setup close to the behavior under test.

## 8. Change discipline

- Keep one batch coherent and reviewable.
- Separate formatting-only churn from logic changes when possible.
- Document assumptions when code context is incomplete.
- State any recommended follow-up as optional work, not as an implicit part of the completed batch.
