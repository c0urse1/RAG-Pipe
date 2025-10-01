# BU Superagent – Architekturgrundlagen

Ziel: Striktes Clean-Architecture-Gerüst (Domain → Application → Infrastructure → Interface → Config)
für eine RAG-Engine (deutsche Sprache), ohne „Vibecoding“. Domain bleibt IO-frei und deterministisch.

## Schichten

- domain/: Entitäten, Value Objects, Policies, pure Services. Kein I/O, keine Globals, keine externen Libs.
- application/: Use-Cases + Ports (Interfaces). Orchestriert Domain. Kein direkter Infra-Zugriff.
- infrastructure/: Adapter, die Ports implementieren. Wrappt externe Libs/IO.
- interface/: CLI/HTTP – nur Parsing/Formatting + Delegation an Use-Cases.
- config/: Composition Root & Settings – einziger Ort für Env/DI/Wiring.

## Regeln (Kurz)

- Abhängigkeiten nur nach innen (siehe `.importlinter`).
- Fehler: typisiert (DomainErrors/Result), keine stillen Fails.
- Tests: Domain & Use-Cases ohne Infra (Stubs/Fakes).

