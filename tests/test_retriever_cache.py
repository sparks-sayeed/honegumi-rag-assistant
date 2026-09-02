"""Tests for the vector-store load cache.

Embeddings run locally now, so loading them is expensive: the
sentence-transformers weights are read from disk and the FAISS index is
deserialized. The planner fans out up to seven retrievers per run, so without
this cache a single run would repeat that work seven times.
"""

import pytest

from honegumi_rag_assistant.nodes import retriever


@pytest.fixture(autouse=True)
def clear_cache():
    """The cache is module-level state; isolate each test from the others."""
    retriever._STORE_CACHE.clear()
    yield
    retriever._STORE_CACHE.clear()


@pytest.fixture
def counting_loaders(monkeypatch):
    """Replace the embedding and FAISS loaders with call counters.

    The fake records the order of operations as well as the counts, because
    *when* the warmup inference happens matters: it has to run before the store
    is published to the cache, or the parallel retrievers race into cold-start
    kernel compilation.
    """
    calls = {"embeddings": 0, "load_local": 0, "warmups": 0, "order": []}

    class FakeEmbeddings:
        def __init__(self, **kwargs):
            calls["embeddings"] += 1
            calls["last_kwargs"] = kwargs
            calls["order"].append("construct")
            self.model_name = kwargs.get("model_name")

        def embed_query(self, text):
            calls["warmups"] += 1
            calls["order"].append(f"embed:{text}")
            return [0.0] * 8

        def __repr__(self):
            return f"embeddings::{self.model_name}"

    class FakeFAISS:
        @staticmethod
        def load_local(path, embeddings, **kwargs):
            calls["load_local"] += 1
            calls["order"].append("load_local")
            return f"store::{path}::{embeddings}"

    monkeypatch.setattr(retriever, "HuggingFaceEmbeddings", FakeEmbeddings)
    monkeypatch.setattr(retriever, "FAISS", FakeFAISS)
    return calls


class TestLoadVectorstore:
    def test_loads_once_then_serves_from_cache(self, counting_loaders):
        first = retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        for _ in range(6):
            again = retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
            assert again is first

        assert counting_loaders["embeddings"] == 1
        assert counting_loaders["load_local"] == 1

    def test_different_model_is_a_separate_entry(self, counting_loaders):
        a = retriever._load_vectorstore("/tmp/store", "model-a")
        b = retriever._load_vectorstore("/tmp/store", "model-b")
        assert a != b
        assert counting_loaders["load_local"] == 2

    def test_different_path_is_a_separate_entry(self, counting_loaders):
        a = retriever._load_vectorstore("/tmp/store-a", "model")
        b = retriever._load_vectorstore("/tmp/store-b", "model")
        assert a != b
        assert counting_loaders["load_local"] == 2

    def test_embeddings_are_normalised(self, counting_loaders):
        """BGE models are trained for cosine similarity, so vectors must be unit."""
        retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        kwargs = counting_loaders["last_kwargs"]
        assert kwargs["model_name"] == "BAAI/bge-large-en-v1.5"
        assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}

    def test_device_is_pinned_not_auto_selected(self, counting_loaders):
        """Auto-selection picks mps on Apple Silicon, which deadlocks under the
        parallel fan-out. The device must always be passed explicitly."""
        retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        kwargs = counting_loaders["last_kwargs"]
        assert "model_kwargs" in kwargs, "device was not pinned — auto-selection is unsafe"
        assert kwargs["model_kwargs"] == {"device": retriever.settings.embedding_device}

    def test_default_device_is_cpu(self):
        """The conservative default; non-CPU devices are opt-in."""
        from honegumi_rag_assistant.app_config import Settings
        assert Settings().embedding_device == "cpu"

    def test_model_is_warmed_before_being_published(self, counting_loaders):
        """The warmup must happen while the lock is held, i.e. before load_local
        returns the store to be cached — otherwise parallel branches hit a cold
        model simultaneously."""
        retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        order = counting_loaders["order"]
        assert counting_loaders["warmups"] == 1, "model was never warmed up"
        assert order.index("construct") < order.index("embed:warmup"), \
            "warmup ran before the model was constructed"
        assert order.index("embed:warmup") < order.index("load_local"), \
            "warmup must complete before the store is published to the cache"

    def test_warmup_happens_once_not_per_retriever(self, counting_loaders):
        for _ in range(7):
            retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        assert counting_loaders["warmups"] == 1

    def test_changing_device_is_a_separate_cache_entry(self, counting_loaders, monkeypatch):
        monkeypatch.setattr(retriever.settings, "embedding_device", "cpu")
        a = retriever._load_vectorstore("/tmp/store", "m")
        monkeypatch.setattr(retriever.settings, "embedding_device", "mps")
        b = retriever._load_vectorstore("/tmp/store", "m")
        assert a is not b
        assert counting_loaders["load_local"] == 2

    def test_concurrent_callers_share_one_load(self, counting_loaders):
        """Mirrors the Send fan-out: parallel branches must not each load."""
        import threading

        results = []

        def worker():
            results.append(
                retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
            )

        threads = [threading.Thread(target=worker) for _ in range(7)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(results) == 7
        assert all(r is results[0] for r in results)
        assert counting_loaders["load_local"] == 1


class TestRetrieveSingleQuery:
    def test_no_vectorstore_configured_flags_only_first_retriever(self, monkeypatch):
        """The flag reducer ORs across branches; duplicates are avoidable noise."""
        monkeypatch.setattr(retriever.settings, "retrieval_vectorstore_path", "")

        first = retriever.retrieve_single_query("q", 0)
        second = retriever.retrieve_single_query("q", 1)

        assert first == {"contexts": [], "vectorstore_missing": True}
        assert second == {"contexts": [], "vectorstore_missing": None}

    def test_load_failure_degrades_to_empty_contexts(self, monkeypatch, tmp_path):
        """A broken store must not abort the run -- code generation continues."""
        monkeypatch.setattr(
            retriever.settings, "retrieval_vectorstore_path", str(tmp_path)
        )

        def boom(*args, **kwargs):
            raise RuntimeError("index is corrupt")

        monkeypatch.setattr(retriever, "_load_vectorstore", boom)

        result = retriever.retrieve_single_query("q", 0)
        assert result["contexts"] == []
        assert result["vectorstore_missing"] is True

    def test_successful_retrieval_tags_contexts_with_provenance(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setattr(
            retriever.settings, "retrieval_vectorstore_path", str(tmp_path)
        )
        monkeypatch.setattr(retriever.settings, "retrieval_top_k", 2)

        class FakeDoc:
            def __init__(self, content):
                self.page_content = content
                self.metadata = {"source": "ax-docs"}

        class FakeStore:
            def similarity_search(self, query, k):
                assert k == 2
                return [FakeDoc("chunk one"), FakeDoc("chunk two")]

        monkeypatch.setattr(
            retriever, "_load_vectorstore", lambda *a, **kw: FakeStore()
        )

        result = retriever.retrieve_single_query("how to add constraints", 3)

        assert len(result["contexts"]) == 2
        for ctx in result["contexts"]:
            # Provenance lets the Code Writer debug output attribute each chunk.
            assert ctx["query"] == "how to add constraints"
            assert ctx["query_index"] == 3
            assert ctx["metadata"] == {"source": "ax-docs"}
        assert result["contexts"][0]["text"] == "chunk one"


class TestLocalFilesOnly:
    """Opt-in flag that stops the loader contacting the HuggingFace Hub."""

    def test_absent_by_default(self, counting_loaders):
        """A machine that has never downloaded the model still needs the Hub."""
        retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        assert "local_files_only" not in counting_loaders["last_kwargs"]["model_kwargs"]

    def test_passed_through_when_enabled(self, counting_loaders, monkeypatch):
        monkeypatch.setattr(retriever.settings, "embedding_local_files_only", True)
        retriever._load_vectorstore("/tmp/store", "BAAI/bge-large-en-v1.5")
        assert counting_loaders["last_kwargs"]["model_kwargs"]["local_files_only"] is True


class TestWarmVectorstore:
    """Startup warmup: the first caller must not pay the 1.2 GB load."""

    def test_populates_the_cache(self, counting_loaders, monkeypatch, tmp_path):
        monkeypatch.setattr(retriever.settings, "retrieval_vectorstore_path", str(tmp_path))

        assert retriever.warm_vectorstore() is True
        assert counting_loaders["load_local"] == 1

        # A subsequent retrieval must be served from cache, not reload.
        retriever._load_vectorstore(str(tmp_path), retriever.settings.embedding_model)
        assert counting_loaders["load_local"] == 1

    def test_no_store_configured_is_not_an_error(self, monkeypatch):
        """A server with no vector store must still start."""
        monkeypatch.setattr(retriever.settings, "retrieval_vectorstore_path", "")
        assert retriever.warm_vectorstore() is False

    def test_load_failure_is_not_fatal(self, counting_loaders, monkeypatch, tmp_path):
        """Startup must survive a corrupt or missing index."""
        monkeypatch.setattr(retriever.settings, "retrieval_vectorstore_path", str(tmp_path))

        def boom(*args, **kwargs):
            raise RuntimeError("corrupt index")

        monkeypatch.setattr(retriever, "_load_vectorstore", boom)
        assert retriever.warm_vectorstore() is False
