from bu_superagent.config.composition import build_embedding_adapter, build_llm_adapter
from bu_superagent.config.settings import AppSettings


def test_builders_return_instances_without_side_effects():
    s = AppSettings()
    emb = build_embedding_adapter(s)
    llm = build_llm_adapter(s)
    # Nur Existenz prüfen; keine Netzwerk- oder Modell-Ladevorgänge
    assert emb is not None
    assert llm is not None
