import sys


def test_chroma_adapter_with_fake_module(monkeypatch):
    # Fake chromadb module
    class FakeCollection:
        def __init__(self):
            self._added = []

        def add(self, ids, metadatas, documents, embeddings):
            self._added.append((ids, metadatas, documents, embeddings))

        def query(self, query_embeddings, n_results):
            # Return the first doc we added
            if not self._added:
                return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}
            ids, metas, docs, _emb = self._added[0]
            return {"ids": [ids[:1]], "documents": [docs[:1]], "metadatas": [metas[:1]], "distances": [[0.1]]}

    class FakeClient:
        def __init__(self, path):
            self.path = path
            self._coll = FakeCollection()

        def get_or_create_collection(self, name, metadata=None):
            return self._coll

    class FakeChromadb:
        def PersistentClient(self, path):
            return FakeClient(path)

    monkeypatch.setitem(sys.modules, "chromadb", FakeChromadb())

    from bu_superagent.infrastructure.vectorstore.chroma_vector_store import (
        ChromaVectorStoreAdapter,
    )

    vs = ChromaVectorStoreAdapter(persist_dir="/tmp/chroma", collection="c")
    vs.ensure_collection("c", dim=3)
    vs.upsert(ids=["a"], vectors=[[0.0, 1.0, 0.0]], payloads=[{"text": "doc"}])
    res = vs.search([0.0, 1.0, 0.0], top_k=1)
    assert res and res[0].id == "a" and res[0].text == "doc"


def test_qdrant_adapter_with_fake_module(monkeypatch):
    # Fake qdrant_client module
    class FakePoint:
        def __init__(self, id, payload, score=0.2):
            self.id = id
            self.score = score
            self.payload = payload

    class FakeClient:
        def __init__(self, host, port):
            self.host = host
            self.port = port
            self._points = []

        def recreate_collection(self, collection_name, vectors_config):
            return None

        def upsert(self, collection_name, points):
            self._points = points

        def search(self, collection_name, query_vector, limit):
            return [FakePoint(points[0].id, points[0].payload, 0.2) for points in [self._points]]

    class FakeModels:
        class Distance:
            COSINE = "cosine"

        class VectorParams:
            def __init__(self, size, distance):
                self.size = size
                self.distance = distance

        class PointStruct:
            def __init__(self, id, vector, payload):
                self.id = id
                self.vector = vector
                self.payload = payload

    import types
    fake_qdrant = types.ModuleType("qdrant_client")
    fake_qdrant.QdrantClient = FakeClient
    fake_models = types.ModuleType("qdrant_client.models")
    fake_models.Distance = FakeModels.Distance
    fake_models.VectorParams = FakeModels.VectorParams
    fake_models.PointStruct = FakeModels.PointStruct
    monkeypatch.setitem(sys.modules, "qdrant_client", fake_qdrant)
    monkeypatch.setitem(sys.modules, "qdrant_client.models", fake_models)

    from bu_superagent.infrastructure.vectorstore.qdrant_vector_store import (
        QdrantVectorStoreAdapter,
    )

    vs = QdrantVectorStoreAdapter(host="h", port=6333, collection="c")
    vs.ensure_collection("c", dim=3)
    vs.upsert(ids=["x"], vectors=[[1.0, 0.0, 0.0]], payloads=[{"text": "t"}])
    res = vs.search([1.0, 0.0, 0.0], top_k=1)
    assert res and res[0].id == "x" and res[0].text == "t"


def test_faiss_adapter_with_fake_modules(monkeypatch):
    class FakeArray:
        def __init__(self, data):
            self._data = data
            # best-effort shape
            n = len(data)
            self.shape = (n,) if not data or not hasattr(data[0], "__len__") else (n, len(data[0]))

    class FakeNP:
        float32 = float

        def array(self, data, dtype=None):  # noqa: ARG002
            return FakeArray(data)

    class FakeIndex:
        def add(self, arr):  # noqa: ARG002
            return None

        def search(self, q, top_k):  # noqa: ARG002
            return [[0.95]], [[0]]

    class FakeFaiss:
        def IndexFlatIP(self, dim):  # noqa: ARG002
            return FakeIndex()

    monkeypatch.setitem(sys.modules, "numpy", FakeNP())
    monkeypatch.setitem(sys.modules, "faiss", FakeFaiss())

    from bu_superagent.infrastructure.vectorstore.faiss_vector_store import (
        FaissVectorStoreAdapter,
    )

    vs = FaissVectorStoreAdapter(collection="c")
    vs.ensure_collection("c", dim=3)
    vs.upsert(ids=["y"], vectors=[[0.0, 1.0, 0.0]], payloads=[{"text": "doc2"}])
    res = vs.search([0.0, 1.0, 0.0], top_k=1)
    assert res and res[0].id == "y" and res[0].text == "doc2"
