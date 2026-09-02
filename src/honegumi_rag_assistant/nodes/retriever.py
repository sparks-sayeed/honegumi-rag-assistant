"""
Node: Retrieve relevant documentation contexts.

This node provides the code writer with additional grounding by
extracting snippets from the Ax documentation.  When configured with
``settings.retrieval_vectorstore_path``, it attempts to load a vector
store (for example a FAISS index) created from the documentation and
performs a similarity search using a query derived from the problem
description and optimisation parameters.  If the vector store or
dependencies are unavailable, it returns an empty list of contexts.

Each returned context should be a dictionary containing at least a
``text`` key with the content.  Additional metadata such as ``source``
and ``score`` may be included depending on the vector store
implementation.
"""

from __future__ import annotations

from typing import Dict, Any, List
import json
import threading
import time

from ..states import HonegumiRAGState
from ..app_config import settings
from ..timing_utils import time_node

# Optional dependencies.  LangChain is not always installed.
try:
    from langchain_community.vectorstores import FAISS  # type: ignore[import]
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore[import]
except ImportError:  # pragma: no cover - optional
    FAISS = None  # type: ignore
    HuggingFaceEmbeddings = None  # type: ignore


# Embeddings now run locally, which changes the cost model of loading them: the
# sentence-transformers weights are read from disk (~1.3 GB) and the FAISS index
# is deserialized on every use.  The retrieval planner fans out up to seven
# retrievers in parallel, so loading per call would repeat that work seven times
# per run.  Cache the loaded store, keyed by path and embedding model.
_STORE_CACHE: Dict[tuple, Any] = {}
_STORE_LOCK = threading.Lock()


def _load_vectorstore(path: str, model_name: str) -> Any:
    """Load and memoise the FAISS store for a given path and embedding model.

    Parameters
    ----------
    path : str
        Directory holding the serialized FAISS index.
    model_name : str
        sentence-transformers model to embed queries with.  This must match the
        model the store was built with; a mismatch yields meaningless
        neighbours rather than an error.

    Returns
    -------
    Any
        A loaded ``FAISS`` vector store, shared across parallel retrievers.
    """
    device = settings.embedding_device
    key = (path, model_name, device)
    cached = _STORE_CACHE.get(key)
    if cached is not None:
        return cached

    with _STORE_LOCK:
        # Re-check inside the lock: a concurrent retriever may have won the race.
        cached = _STORE_CACHE.get(key)
        if cached is not None:
            return cached

        # Only add local_files_only when it is on, so the default request to
        # sentence-transformers stays byte-identical to what it always was.
        model_kwargs: Dict[str, Any] = {"device": device}
        if settings.embedding_local_files_only:
            model_kwargs["local_files_only"] = True

        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )

        # Run one inference while still holding the lock, before this store is
        # visible to any other thread.  sentence-transformers initialises lazily
        # and compiles kernels on first use; on Apple's MPS backend, several
        # threads entering that compilation at once deadlock inside Metal -- no
        # exception, no output, just a pinned GPU until the process is killed.
        # Warming up serially here guarantees the parallel retrievers only ever
        # meet an already-initialised model, on whatever device is configured.
        embeddings.embed_query("warmup")

        store = FAISS.load_local(
            path,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        _STORE_CACHE[key] = store
        return store


def warm_vectorstore() -> bool:
    """Load the vector store now, so the first request does not pay for it.

    The store is otherwise loaded lazily inside the retrieval fan-out, which
    means whoever calls first waits for the ~1.2 GB embedding model and the
    FAISS deserialization -- several seconds warm, far worse on a cold machine.
    A long-lived server should absorb that at startup instead; a CLI run, which
    exits after one pipeline, gains nothing and should not call this.

    Returns
    -------
    bool
        True if the store is loaded and cached.  False when there is nothing to
        warm or the load failed.  Never raises: a server that cannot warm its
        cache must still start and degrade to no retrieval, exactly as the
        fan-out does.
    """
    if not settings.retrieval_vectorstore_path:
        print("No vector store configured - starting without retrieval.")
        return False

    if FAISS is None or HuggingFaceEmbeddings is None:
        print("Retrieval dependencies unavailable - starting without retrieval.")
        return False

    start = time.time()
    print(f"Warming vector store from {settings.retrieval_vectorstore_path} ...")
    try:
        _load_vectorstore(
            settings.retrieval_vectorstore_path,
            settings.embedding_model,
        )
    except Exception as exc:  # noqa: BLE001 - startup must not be fatal
        print(f"Vector store warmup failed ({exc}) - continuing without retrieval.")
        return False

    print(f"Vector store ready in {time.time() - start:.1f}s.")
    return True


def retrieve_single_query(query: str, query_index: int) -> Dict[str, Any]:
    """Retrieve contexts for a single query (used in parallel fan-out).
    
    This is a simpler version designed for parallel execution via Send API.
    Each parallel retriever handles one query independently.
    
    Parameters
    ----------
    query : str
        The retrieval query to execute
    query_index : int
        Index of this query (for debugging)
    
    Returns
    -------
    Dict[str, Any]
        Dictionary with 'contexts' key containing retrieved docs
    """
    start_time = time.time()
    
    if settings.debug:
        print(f"\n[PARALLEL RETRIEVER {query_index + 1}] Query: {query}")
    
    # If no vector store is configured, return empty with error marker
    if not settings.retrieval_vectorstore_path:
        if settings.debug:
            print(f"[PARALLEL RETRIEVER {query_index + 1}] No vector store configured\n")
        # Only set the flag on the first retriever to avoid duplicates
        return {"contexts": [], "vectorstore_missing": True if query_index == 0 else None}

    # Check dependencies
    if FAISS is None or HuggingFaceEmbeddings is None:
        if settings.debug:
            print(f"[PARALLEL RETRIEVER {query_index + 1}] Dependencies not available\n")
        # Only set the flag on the first retriever to avoid duplicates
        return {"contexts": [], "vectorstore_missing": True if query_index == 0 else None}

    try:
        # Shared across parallel retrievers; only the first call pays the
        # model-load and index-deserialization cost.
        vectorstore = _load_vectorstore(
            settings.retrieval_vectorstore_path,
            settings.embedding_model,
        )

        # Search using the query
        docs = vectorstore.similarity_search(query, k=settings.retrieval_top_k)
        
        # Convert documents to dictionaries
        contexts: List[Dict[str, Any]] = []
        for doc in docs:
            contexts.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "query": query,  # Track which query retrieved this
                "query_index": query_index,  # Track which parallel retriever this came from
            })
        
        elapsed_time = time.time() - start_time
        
        if settings.debug:
            print(f"[PARALLEL RETRIEVER {query_index + 1}] Retrieved {len(contexts)} contexts")
            print(f"[PARALLEL RETRIEVER {query_index + 1}] Time: {elapsed_time:.2f}s\n")
        
        return {"contexts": contexts}
        
    except Exception as exc:
        if settings.debug:
            print(f"[PARALLEL RETRIEVER {query_index + 1}] ERROR: {exc}\n")
            import traceback
            traceback.print_exc()
        # Mark as missing if we can't load the vector store (only on first retriever)
        return {"contexts": [], "vectorstore_missing": True if query_index == 0 else None}


class RetrieverAgent:
    """Retrieve relevant documentation snippets for the current problem.

    The retriever constructs a query by concatenating the natural
    language problem description with a JSON representation of the
    optimisation parameters.  It then executes a similarity search
    against the configured vector store to return the top ``k`` most
    relevant chunks.  If no vector store is configured or the required
    dependencies are missing the retriever returns an empty list.
    """

    @staticmethod
    @time_node("Retriever Agent")
    def retrieve_context(state: HonegumiRAGState) -> Dict[str, Any]:
        """Retrieve documentation contexts based on a specific question.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state containing ``retrieval_query``
            with a specific question from the Code Writer agent.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys:
            - ``contexts``: list of context snippets (empty if retrieval fails)
            - ``retrieval_count``: incremented counter
            - ``retrieval_query``: cleared (set to None)
        """
        # If no vector store is configured, return empty
        if not settings.retrieval_vectorstore_path:
            return {
                "contexts": state.get("contexts", []),
                "retrieval_count": state.get("retrieval_count", 0),
                "retrieval_query": None,
            }

        # Check that the necessary dependencies are present
        if FAISS is None or HuggingFaceEmbeddings is None:
            return {
                "contexts": state.get("contexts", []),
                "retrieval_count": state.get("retrieval_count", 0),
                "retrieval_query": None,
                "error": "Retrieval dependencies not installed; unable to load vector store."
            }

        # Get the specific question from the Code Writer
        retrieval_query = state.get("retrieval_query")
        if not retrieval_query:
            # No query provided, return current state
            return {
                "contexts": state.get("contexts", []),
                "retrieval_count": state.get("retrieval_count", 0),
                "retrieval_query": None,
            }

        retrieval_count = state.get("retrieval_count", 0)
        existing_contexts = state.get("contexts", [])
        
        if settings.debug:
            # DEBUG: Print retrieval query
            print("\n" + "="*80)
            print(f"DEBUG: RETRIEVER PROCESSING QUERY (Attempt {retrieval_count + 1}/3)")
            print("="*80)
            print(f"Query: {retrieval_query}")
            print(f"Existing contexts from state: {len(existing_contexts)}")
            print("="*80 + "\n")

        try:
            # Shared loader; the embedding model must match the one the store
            # was built with.
            vectorstore = _load_vectorstore(
                settings.retrieval_vectorstore_path,
                settings.embedding_model,
            )

            # Search using the specific question from Code Writer
            docs = vectorstore.similarity_search(retrieval_query, k=settings.retrieval_top_k)
            
            # Convert documents to dictionaries
            new_contexts: List[Dict[str, Any]] = []
            for doc in docs:
                new_contexts.append({
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                })
            
            # Append new contexts to existing ones (accumulate across retrieval loops)
            all_contexts = existing_contexts + new_contexts
            
            if settings.debug:
                # DEBUG: Print retrieval results
                print(f"Retrieved {len(new_contexts)} new context snippets")
                print(f"Total contexts accumulated: {len(all_contexts)}")
                print(f"Incrementing retrieval_count from {retrieval_count} to {retrieval_count + 1}\n")
            
            return {
                "contexts": all_contexts,
                "retrieval_count": retrieval_count + 1,
                "retrieval_query": None,  # Clear the query after use
            }
            
        except Exception as exc:
            # On error, return current state without incrementing counter
            if settings.debug:
                print(f"ERROR in retriever: {exc}")
                import traceback
                traceback.print_exc()
            return {
                "contexts": existing_contexts,
                "retrieval_count": retrieval_count,
                "retrieval_query": None,
                "error": f"Failed to retrieve contexts: {exc}",
            }