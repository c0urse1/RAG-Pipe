# Copilot – Project Guardrails (Strict Architecture Mode)

- No business logic in `interface/` or `infrastructure/`.
- Domain is IO-free: no external imports (except stdlib typing/dataclasses), no env, no FS/network.
- Application uses only Ports (Interfaces in `application/ports/*`). Never direct imports from infrastructure.
- Infrastructure implements Ports and wraps external libs (Chroma, HF, Filesystem).
- Config is the only place for Env/Settings and Wiring (Dependency Injection).
- Errors: use `Result[T,E]` or specific `*Error` classes; never silent failure.
- Tests: For new Domain/Use Case features add Unit Tests. No E2E shortcuts.
- Naming: Ports: `*Port`, Adapters: `*Adapter` or technology-based (e.g. `ChromaVectorStore`).

These minimal rules already ensure that Copilot produces architecture-compliant suggestions.

In Step 2 we’ll expand the document with examples, Do/Don’t snippets, and prompt hints for Copilot.
