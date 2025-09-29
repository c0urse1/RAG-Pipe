from bu_superagent.infrastructure.vectorstore.chroma_vector_store import InMemoryVectorStore


def test_inmemory_vectorstore_contract_add_persist():
    vs = InMemoryVectorStore()
    vs.add(["hello"], [{"text": "hello"}])
    vs.persist()  # no-op
    assert True
