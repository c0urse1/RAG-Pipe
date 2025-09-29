from bu_superagent.infrastructure.vectorstore.chroma_vector_store import InMemoryVectorStore


def test_inmemory_vectorstore_contract():
    vs = InMemoryVectorStore()
    vs.upsert(["a"], [[1.0, 0.0]], [{"text": "hello"}])
    res = vs.query([1.0, 0.0], top_k=1)
    assert res and res[0][0] == "a"
