# Python Standards

Use this file for Python repositories and services.

## 1. Module structure

- Group modules by feature or responsibility.
- Keep package `__init__.py` exports deliberate and minimal.
- Avoid giant `utils.py`, `helpers.py`, or `common.py` files.
- Separate domain logic from web framework, CLI, persistence, and integration code.

## 2. Design and dependency boundaries

- Prefer composition and small functions over deep inheritance trees.
- Use classes where they carry state or enforce a boundary; do not wrap simple stateless behavior in classes without a reason.
- Make I/O boundaries explicit: filesystem, network, database, clock, env, and process calls should be easy to isolate in tests.
- Prefer dependency injection through function parameters, constructors, or clearly owned adapters.

## 3. Typing and interfaces

- Add type hints for public functions, methods, and module boundaries.
- Use `Protocol`, abstract base classes, or typed callables only when they create a useful seam.
- Do not over-abstract internal one-off flows.
- Keep data models explicit with `dataclass`, Pydantic, or typed dicts when the project already uses them and they improve clarity.

## 4. Error handling

- Raise domain-relevant exceptions instead of returning ambiguous sentinel values.
- Keep exception translation at the boundary layer: API, CLI, job runner, or integration adapter.
- Do not swallow exceptions without converting them into a clear outcome or log signal.

## 5. Packaging and imports

- Prefer absolute imports inside larger applications.
- Keep import direction stable and acyclic.
- Avoid runtime side effects at import time beyond lightweight constant or logger setup.
- Keep settings and bootstrapping in dedicated modules, not scattered across business code.

## 6. Framework use

- Keep framework objects at the edge: FastAPI request objects, Django ORM objects, Click contexts, Celery tasks, and similar runtime types should not spread through core logic unless the codebase intentionally couples them.
- Move serialization, transport validation, and adapter glue to boundary modules.

## 7. Testing

- Use `pytest` style unless the repository already standardizes on something else.
- Prefer focused unit tests with fixtures that expose intent.
- Use parametrization for rule variations.
- Mock external systems at the adapter boundary.
- Add integration tests for persistence, API, or job flows only where the unit tests cannot prove the behavior.

## 8. Refactoring guidance

- Split long functions by responsibility, not by arbitrary line count.
- Extract pure functions first when untangling mixed business logic and I/O.
- Replace boolean flag APIs with clearer call sites or separate functions when behavior branches materially.
- Remove incidental shared helpers if they hide ownership more than they reduce duplication.
