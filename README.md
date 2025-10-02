BU Superagent – Architecture Basics

Goal: Strict Clean Architecture skeleton (Domain → Application → Infrastructure → Interface → Config)
for a RAG engine (German language), without “vibecoding”. Domain stays IO-free and deterministic.

Layers

domain/: Entities, Value Objects, Policies, pure Services. No I/O, no globals, no external libs.

application/: Use Cases + Ports (Interfaces). Orchestrates domain. No direct Infra access.

infrastructure/: Adapters implementing Ports. Wraps external libs/IO.

interface/: CLI/HTTP – only parsing/formatting + delegation to Use Cases.

config/: Composition Root & Settings – only place for Env/DI/Wiring.

Rules (Short)

Dependencies only inward (see .importlinter).

Errors: typed (DomainErrors/Result), no silent failures.

Tests: Domain & Use Cases without Infra (Stubs/Fakes).
