#!/usr/bin/env python
"""
Build a FAISS vector store from Ax documentation.

This script automates the process of:
1. Cloning Ax repository from GitHub
2. Extracting documentation from markdown files
3. Generating OpenAI embeddings
4. Building and saving a FAISS vector database

Usage:
    python scripts/build_vector_store.py
    python scripts/build_vector_store.py --output custom/path
    python scripts/build_vector_store.py --update  # Refresh existing store
"""

import argparse
import os
import sys
from pathlib import Path
import subprocess
import tempfile
import shutil
import json
from datetime import datetime
from typing import List, Dict, Any

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.docstore.document import Document


def clone_ax_repo(temp_dir: Path, ax_version: str = "0.4.3") -> Path:
    """
    Clone the Ax repository from GitHub at a specific version.
    
    Args:
        temp_dir: Temporary directory for cloning
        ax_version: Git tag/branch to clone (default: "0.4.3" to match honegumi)
        
    Returns:
        Path to cloned repository
    """
    print(f"Cloning Ax repository from GitHub (version: {ax_version})...")
    repo_url = "https://github.com/facebook/Ax.git"
    repo_path = temp_dir / "Ax"
    
    try:
        # Clone specific version/branch
        subprocess.run(
            ["git", "clone", "--depth", "1", "--branch", ax_version, repo_url, str(repo_path)],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"  Successfully cloned Ax v{ax_version} to {repo_path}")
        return repo_path
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr
        if "Remote branch" in error_msg or "not found" in error_msg.lower():
            print(f"\n✗ ERROR: Ax version '{ax_version}' not found!")
            print(f"\nAvailable versions can be checked at:")
            print(f"  https://github.com/facebook/Ax/tags")
            print(f"\nTry one of these common versions:")
            print(f"  - 0.4.3 (recommended for honegumi 0.4.3)")
            print(f"  - 0.4.0")
            print(f"  - main (latest, may have breaking changes)")
        else:
            print(f"Error cloning repository: {error_msg}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: git command not found. Please install git.")
        sys.exit(1)


def extract_docs_from_repo(repo_path: Path) -> List[Dict[str, str]]:
    """
    Extract documentation from cloned Ax repository.
    Includes both /docs markdown files and /tutorials notebooks.
    
    Args:
        repo_path: Path to cloned Ax repository
        
    Returns:
        List of dictionaries with 'url', 'title', and 'content'
    """
    print("\nExtracting documentation from repository...")
    documents = []
    
    # Extract from /docs folder (markdown files)
    docs_dir = repo_path / "docs"
    if docs_dir.exists():
        md_files = list(docs_dir.glob("**/*.md")) + list(docs_dir.glob("**/*.mdx"))
        print(f"  Found {len(md_files)} markdown documentation files in /docs")
        
        for md_file in md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Skip if too short
                if len(content) < 100:
                    continue
                
                rel_path = md_file.relative_to(docs_dir)
                title = rel_path.stem.replace('_', ' ').replace('-', ' ').title()
                url = f"https://github.com/facebook/Ax/tree/main/docs/{rel_path}"
                
                documents.append({
                    'url': url,
                    'title': f"Docs: {title}",
                    'content': content,
                    'file_path': str(rel_path),
                    'source_type': 'docs'
                })
                
                print(f"  Extracted: docs/{rel_path.name} ({len(content)} characters)")
                
            except Exception as e:
                print(f"  Warning: Could not read {md_file.name}: {e}")
                continue
    
    # Extract from /tutorials folder (notebooks + markdown)
    tutorials_dir = repo_path / "tutorials"
    if tutorials_dir.exists():
        # Get markdown files from tutorials
        tutorial_md_files = list(tutorials_dir.glob("**/*.md")) + list(tutorials_dir.glob("**/*.mdx"))
        # Get notebook files
        tutorial_nb_files = list(tutorials_dir.glob("**/*.ipynb"))
        
        print(f"  Found {len(tutorial_md_files)} markdown files and {len(tutorial_nb_files)} notebooks in /tutorials")
        
        # Process markdown files
        for md_file in tutorial_md_files:
            try:
                content = md_file.read_text(encoding='utf-8')
                
                if len(content) < 100:
                    continue
                
                rel_path = md_file.relative_to(tutorials_dir)
                title = rel_path.stem.replace('_', ' ').replace('-', ' ').title()
                url = f"https://github.com/facebook/Ax/tree/main/tutorials/{rel_path}"
                
                documents.append({
                    'url': url,
                    'title': f"Tutorial: {title}",
                    'content': content,
                    'file_path': str(rel_path),
                    'source_type': 'tutorial_md'
                })
                
                print(f"  Extracted: tutorials/{rel_path.name} ({len(content)} characters)")
                
            except Exception as e:
                print(f"  Warning: Could not read tutorial {md_file.name}: {e}")
                continue
        
        # Process notebook files - extract markdown cells and code cells
        for nb_file in tutorial_nb_files:
            try:
                import json
                with open(nb_file, 'r', encoding='utf-8') as f:
                    notebook = json.load(f)
                
                # Extract text from notebook cells
                content_parts = []
                for cell in notebook.get('cells', []):
                    cell_type = cell.get('cell_type', '')
                    source = cell.get('source', [])
                    
                    # Join source lines if it's a list
                    if isinstance(source, list):
                        source = ''.join(source)
                    
                    if cell_type == 'markdown' and source:
                        content_parts.append(source)
                    elif cell_type == 'code' and source:
                        # Include code cells too - they have useful examples
                        content_parts.append(f"```python\n{source}\n```")
                
                content = '\n\n'.join(content_parts)
                
                if len(content) < 100:
                    continue
                
                rel_path = nb_file.relative_to(tutorials_dir)
                title = rel_path.stem.replace('_', ' ').replace('-', ' ').title()
                url = f"https://github.com/facebook/Ax/tree/main/tutorials/{rel_path}"
                
                documents.append({
                    'url': url,
                    'title': f"Tutorial: {title}",
                    'content': content,
                    'file_path': str(rel_path),
                    'source_type': 'tutorial_notebook'
                })
                
                print(f"  Extracted: tutorials/{rel_path.name} ({len(content)} characters)")
                
            except Exception as e:
                print(f"  Warning: Could not read notebook {nb_file.name}: {e}")
                continue
    
    print(f"\nSuccessfully extracted {len(documents)} files total")
    return documents


def chunk_documents(documents: List[Dict[str, str]], 
                    chunk_size: int = 2000,
                    chunk_overlap: int = 400) -> List[Document]:
    """
    Split documents into chunks for retrieval.
    
    Uses larger chunks (2000 chars) to keep code examples and explanations together.
    Larger overlap (400 chars) ensures context continuity across chunks.
    
    Args:
        documents: List of dicts with 'content', 'title', 'url'
        chunk_size: Target size for each chunk in characters (default: 2000)
        chunk_overlap: Overlap between chunks to preserve context (default: 400)
        
    Returns:
        List of LangChain Document objects
    """
    print(f"\nChunking documents (chunk_size={chunk_size}, overlap={chunk_overlap})...")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""]
    )
    
    chunked_docs = []
    for doc in documents:
        chunks = text_splitter.split_text(doc['content'])
        for i, chunk in enumerate(chunks):
            chunked_docs.append(Document(
                page_content=chunk,
                metadata={
                    'source': doc['url'],
                    'title': doc['title'],
                    'chunk_index': i,
                    'total_chunks': len(chunks)
                }
            ))
    
    print(f"  Created {len(chunked_docs)} chunks from {len(documents)} documents")
    return chunked_docs


def build_faiss_index(documents: List[Document], 
                     openai_api_key: str,
                     embedding_model: str = "text-embedding-3-large") -> FAISS:
    """
    Build a FAISS vector store from documents.
    
    Uses text-embedding-3-large for better retrieval quality.
    This model has 3072 dimensions vs 1536 for the small model,
    providing more nuanced semantic understanding.
    
    Args:
        documents: List of LangChain Document objects
        openai_api_key: OpenAI API key for embeddings
        embedding_model: OpenAI embedding model (default: text-embedding-3-large)
        
    Returns:
        FAISS vector store
    """
    print(f"\nGenerating embeddings using {embedding_model}...")
    print(f"  This may take a few minutes for {len(documents)} chunks...")
    
    embeddings = OpenAIEmbeddings(
        openai_api_key=openai_api_key,
        model=embedding_model
    )
    
    # Build FAISS index - process all at once (LangChain handles batching internally)
    print(f"  Building FAISS index...")
    vectorstore = FAISS.from_documents(documents, embeddings)
    
    print("  FAISS index built successfully")
    return vectorstore


def main():
    parser = argparse.ArgumentParser(description="Build Ax documentation vector store")
    parser.add_argument(
        "--output",
        default="data/processed/ax_docs_vectorstore",
        help="Output directory for vector store (default: data/processed/ax_docs_vectorstore)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=2000,
        help="Size of text chunks (default: 2000)"
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=400,
        help="Overlap between chunks (default: 400)"
    )
    parser.add_argument(
        "--embedding-model",
        default="text-embedding-3-large",
        help="OpenAI embedding model (default: text-embedding-3-large)"
    )
    parser.add_argument(
        "--ax-version",
        default="0.4.3",
        help="Ax version/branch to clone (default: 0.4.3)"
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Force update even if vector store exists"
    )
    
    args = parser.parse_args()
    
    # Check for OpenAI API key
    openai_api_key = os.getenv("LLM_API_KEY")
    if not openai_api_key:
        print("❌ Error: LLM_API_KEY environment variable not set")
        print("\nSet your API key:")
        print("  PowerShell: $env:LLM_API_KEY = 'your-key'")
        print("  Bash: export LLM_API_KEY='your-key'")
        sys.exit(1)
    
    # Check if output already exists
    output_path = Path(args.output)
    if output_path.exists() and not args.update:
        print(f"Warning: Vector store already exists at {output_path}")
        print("  Use --update to rebuild it")
        response = input("  Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    print("="*80)
    print("AX DOCUMENTATION VECTOR STORE BUILDER")
    print("="*80)
    print(f"Ax version: {args.ax_version}")
    print(f"Output directory: {args.output}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"Chunk overlap: {args.chunk_overlap}")
    print(f"Embedding model: {args.embedding_model}")
    print("="*80)
    
    temp_dir = None
    try:
        # Step 1: Clone Ax repository
        temp_dir = Path(tempfile.mkdtemp())
        repo_path = clone_ax_repo(temp_dir, ax_version=args.ax_version)
        
        # Step 2: Extract documentation
        documents = extract_docs_from_repo(repo_path)
        
        if not documents:
            print("Error: No documentation files found.")
            sys.exit(1)
        
        # Step 3: Chunk documents
        chunked_docs = chunk_documents(
            documents,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )
        
        # Step 4: Build FAISS index
        vectorstore = build_faiss_index(
            chunked_docs,
            openai_api_key,
            embedding_model=args.embedding_model
        )
        
        # Step 5: Save to disk
        print(f"\nSaving vector store to {output_path}...")
        output_path.mkdir(parents=True, exist_ok=True)
        vectorstore.save_local(str(output_path))
        
        # Save metadata
        metadata = {
            'ax_version': args.ax_version,
            'build_date': datetime.now().isoformat(),
            'chunk_size': args.chunk_size,
            'chunk_overlap': args.chunk_overlap,
            'embedding_model': args.embedding_model,
            'total_documents': len(documents),
            'total_chunks': len(chunked_docs)
        }
        metadata_path = output_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  Vector store saved successfully!")
        print(f"\nStatistics:")
        print(f"  Ax version: {args.ax_version}")
        print(f"  Total documentation files: {len(documents)}")
        print(f"  Total chunks: {len(chunked_docs)}")
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in output_path.glob('**/*') if f.is_file())
        print(f"  Vector store size: {total_size / (1024*1024):.1f} MB")
        print(f"  Location: {output_path.absolute()}\n")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Clean up temporary directory
        if temp_dir and temp_dir.exists():
            print(f"\nCleaning up temporary files...")
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
