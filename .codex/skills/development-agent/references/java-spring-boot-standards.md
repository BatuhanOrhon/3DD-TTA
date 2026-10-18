# Java Spring Boot Standards

Use this file for Java and Spring Boot repositories.

## 1. Project structure

- Prefer package-by-feature over package-by-layer when the codebase is not already locked into a different structure.
- Keep feature packages narrow and cohesive: controller or API, application service, domain model, repository or adapter, mapper only if needed.
- Keep `config` packages for framework wiring and infrastructure configuration, not business logic.
- Avoid a shared `common` package that becomes a dumping ground.

## 2. Layering

- Keep controllers thin. They should validate transport concerns, map request or response models, and delegate.
- Place business orchestration in application services.
- Keep domain rules in domain objects or domain services, not in controllers, repositories, or entity annotations.
- Keep repository code focused on persistence access. Do not bury business decisions in queries unless the decision is inherently data access logic.

## 3. Spring dependency injection

- Prefer constructor injection only.
- Do not use field injection.
- Keep bean wiring explicit when the dependency graph is non-trivial.
- Introduce interfaces only when they provide a real seam: multiple implementations, boundary isolation, or testability.
- Avoid needless interface-plus-impl pairs for trivial internal classes.

## 4. Entities, DTOs, and mapping

- Do not expose JPA entities directly from controllers.
- Separate request or response DTOs from persistence entities.
- Keep mapping explicit. Use MapStruct or dedicated mappers only when mapping volume justifies it.
- Avoid putting API-specific annotations or serialization behavior into domain types unless the architecture intentionally uses them as boundary models.

## 5. Transactions and persistence

- Place transaction boundaries in the service layer unless there is a strong reason not to.
- Keep transactional methods small and intention-revealing.
- Avoid mixing remote calls and long-running non-database work inside transactions.
- Prefer repository methods with clear intent over highly generic query helpers.
- Watch for N+1 and lazy loading issues at module boundaries.

## 6. Error handling

- Centralize HTTP error translation with controller advice or an equivalent mechanism.
- Use domain-specific exceptions for business failures where they improve clarity.
- Do not leak persistence or framework exceptions directly to API responses.

## 7. Testing

- Prefer plain unit tests for domain logic.
- Use slice tests selectively: `@WebMvcTest`, repository tests, or other focused slices when they materially reduce setup cost.
- Use `@SpringBootTest` only when integration coverage is actually required.
- Mock across architectural boundaries, not inside the unit under test.
- Keep test fixtures local and readable.

## 8. Refactoring guidance

- Break apart fat services by use case or domain responsibility.
- Extract reusable policy or rule objects before introducing inheritance.
- If a package has bidirectional dependencies, move the shared abstraction downward or split the feature boundary.
- Remove dead configuration and stale beans during structural refactors only if they are clearly in scope.
