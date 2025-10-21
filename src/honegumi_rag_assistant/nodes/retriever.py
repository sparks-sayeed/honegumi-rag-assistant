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
import time

from ..states import HonegumiRAGState
from ..app_config import settings
from ..timing_utils import time_node

# Optional dependencies.  LangChain is not always installed.
try:
    from langchain_community.vectorstores import FAISS  # type: ignore[import]
    from langchain_openai import OpenAIEmbeddings  # type: ignore[import]
except ImportError:  # pragma: no cover - optional
    FAISS = None  # type: ignore
    OpenAIEmbeddings = None  # type: ignore


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
    if FAISS is None or OpenAIEmbeddings is None:
        if settings.debug:
            print(f"[PARALLEL RETRIEVER {query_index + 1}] Dependencies not available\n")
        # Only set the flag on the first retriever to avoid duplicates
        return {"contexts": [], "vectorstore_missing": True if query_index == 0 else None}

    try:
        # Load the vector store and embeddings
        embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-3-large"
        )
        vectorstore = FAISS.load_local(
            settings.retrieval_vectorstore_path,
            embeddings,
            allow_dangerous_deserialization=True
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
        if FAISS is None or OpenAIEmbeddings is None:
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
            # Load the vector store and embeddings
            # Use text-embedding-3-large to match the vector store creation
            embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-3-large"
            )
            vectorstore = FAISS.load_local(
                settings.retrieval_vectorstore_path,
                embeddings,
                allow_dangerous_deserialization=True
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