#!/usr/bin/env python
"""
Test the Ax documentation vector store.

This script loads the FAISS vector store and runs sample queries to verify
that retrieval is working correctly.

Usage:
    python scripts/test_vector_store.py
    python scripts/test_vector_store.py --vectorstore-path custom/path
"""

import argparse
import os
import sys
from pathlib import Path

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def test_vector_store(vectorstore_path: str, openai_api_key: str):
    """Test vector store with sample queries."""
    
    print("="*80)
    print("VECTOR STORE TEST")
    print("="*80)
    print(f"Vector store path: {vectorstore_path}")
    print("="*80 + "\n")
    
    # Load the vector store
    print("Loading vector store...")
    try:
        # Use text-embedding-3-large to match the vector store creation
        embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-large"
        )
        vectorstore = FAISS.load_local(vectorstore_path, embeddings, allow_dangerous_deserialization=True)
        print("  Vector store loaded successfully\n")
    except Exception as e:
        print(f"Error loading vector store: {e}")
        return False
    
    # Sample queries that should match Ax documentation
    test_queries = [
        "How do I define a search space in Ax?",
        "What is the difference between GP and SAASBO models?",
        "How do I set up a multi-objective optimization problem?",
        "How do I add parameter constraints?",
        "What is the Service API in Ax?",
    ]
    
    print("Running test queries...\n")
    
    for i, query in enumerate(test_queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 80)
        
        try:
            # Retrieve top 3 documents
            docs = vectorstore.similarity_search(query, k=3)
            
            if not docs:
                print("  Warning: No documents retrieved\n")
                continue
            
            print(f"  Retrieved {len(docs)} documents:\n")
            
            for j, doc in enumerate(docs, 1):
                content_preview = doc.page_content[:200].replace('\n', ' ')
                print(f"  [{j}] Source: {doc.metadata.get('source', 'unknown')}")
                print(f"      Title: {doc.metadata.get('title', 'unknown')}")
                print(f"      Preview: {content_preview}...")
                print()
            
        except Exception as e:
            print(f"Error running query: {e}\n")
            continue
        
        print("=" * 80 + "\n")
    
    print("All tests completed!")
    return True


def main():
    parser = argparse.ArgumentParser(description="Test Ax documentation vector store")
    parser.add_argument(
        "--vectorstore-path",
        default=None,
        help="Path to vector store directory (default: from AX_DOCS_VECTORSTORE_PATH env var)"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("LLM_API_KEY")
    if not openai_api_key:
        print("Error: LLM_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  PowerShell: $env:LLM_API_KEY = 'your-key'")
        print("  Bash: export LLM_API_KEY='your-key'")
        sys.exit(1)
    
    # Determine vector store path
    vectorstore_path = args.vectorstore_path or os.getenv("AX_DOCS_VECTORSTORE_PATH")
    
    if not vectorstore_path:
        print("Error: No vector store path specified")
        print("\nEither:")
        print("  1. Set AX_DOCS_VECTORSTORE_PATH environment variable")
        print("  2. Use --vectorstore-path argument")
        print("\nExample:")
        print("  $env:AX_DOCS_VECTORSTORE_PATH = 'data/processed/ax_docs_vectorstore'")
        print("  python scripts/test_vector_store.py")
        sys.exit(1)
    
    # Check if path exists
    if not Path(vectorstore_path).exists():
        print(f"Error: Vector store not found at {vectorstore_path}")
        print("\nCreate the vector store first:")
        print("  python scripts/build_vector_store.py")
        sys.exit(1)
    
    # Run tests
    success = test_vector_store(vectorstore_path, openai_api_key)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
