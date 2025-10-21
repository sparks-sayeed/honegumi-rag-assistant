"""
Configuration management for the Honegumi RAG Assistant.

This module defines a simple dataclass, :class:`Settings`, that holds
configuration values for the pipeline.  These values can be overridden
via environment variables or programmatically at runtime.  A global
instance, :data:`settings`, is created on import for convenience.

The following configuration options are supported:

``model_name``
    The name of the language model to use when invoking the OpenAI API.
    Defaults to ``"gpt-5"``.  Override via the ``OPENAI_MODEL_NAME``
    environment variable.

``openai_api_key``
    Your OpenAI API key.  The assistant will raise an exception if
    attempting to call the API when this value is empty.  Override via
    the ``OPENAI_API_KEY`` environment variable.

``retrieval_vectorstore_path``
    Path to a serialized vector store (e.g. FAISS index) containing the
    Ax documentation.  When supplied, the retriever will attempt to
    load this index and perform semantic searches.  Override via the
    ``AX_DOCS_VECTORSTORE_PATH`` environment variable.

``retrieval_top_k``
    The number of documents to return from the retriever.  Defaults to 5.
    Override via the ``RETRIEVAL_TOP_K`` environment variable.

``output_dir``
    Directory where the generated scripts and artefacts will be saved.
    Defaults to ``"./honegumi_rag_output"``.  Override via the
    ``OUTPUT_DIR`` environment variable.
"""

from dataclasses import dataclass
import os

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


@dataclass
class Settings:
    """Container for configuration values.

    Attributes
    ----------
    model_name : str
        Identifier of the OpenAI model to use for all LLM calls.  A
        sensible default of ``"gpt-5"`` is provided but can be
        overridden via the ``OPENAI_MODEL_NAME`` environment variable.
    code_writer_model : str
        Model to use specifically for the Code Writer agent. Defaults
        to model_name if not specified. Override via
        ``CODE_WRITER_MODEL`` environment variable.
    reviewer_model : str
        Model to use specifically for the Reviewer agent. Defaults
        to model_name if not specified. Override via
        ``REVIEWER_MODEL`` environment variable.
    retrieval_planner_model : str
        Model to use specifically for the Retrieval Planner agent. Defaults
        to model_name if not specified. Override via
        ``RETRIEVAL_PLANNER_MODEL`` environment variable.
    openai_api_key : str
        Secret API key used to authenticate with the OpenAI API.  You
        must set this in your environment or the pipeline will raise
        an exception when attempting to call the LLM.
    retrieval_vectorstore_path : str
        Optional path to a vector store used by the retriever to fetch
        Ax documentation.  If empty, no retrieval will be performed and
        the ``contexts`` field of the state will remain an empty list.
    retrieval_top_k : int
        Number of context snippets to return from the vector store.  A
        small number (5–10) keeps the LLM context manageable.
    output_dir : str
        Directory where the generated code and artefacts should be
        written by the :func:`run` function.  The directory will be
        created if it does not exist.
    debug : bool
        Whether to enable debug mode with verbose output showing all
        decisions, parameters, and intermediate steps. Set at runtime
        via CLI flag. Defaults to False.
    """

    model_name: str = os.getenv("OPENAI_MODEL_NAME", "gpt-5")
    code_writer_model: str = os.getenv("CODE_WRITER_MODEL", "gpt-5")
    reviewer_model: str = os.getenv("REVIEWER_MODEL", "gpt-4o")
    retrieval_planner_model: str = os.getenv("RETRIEVAL_PLANNER_MODEL", "gpt-5")
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    retrieval_vectorstore_path: str = os.getenv("AX_DOCS_VECTORSTORE_PATH", "")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    output_dir: str = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")
    debug: bool = False  # Set at runtime, not from environment
    stream_code: bool = False  # Set at runtime to enable streaming output
    
    def reload_from_env(self):
        """Reload settings from environment variables.
        
        Useful when environment variables are set after the module is imported,
        such as in Jupyter/Colab notebooks.
        """
        self.model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-5")
        self.code_writer_model = os.getenv("CODE_WRITER_MODEL", "gpt-5")
        self.reviewer_model = os.getenv("REVIEWER_MODEL", "gpt-4o")
        self.retrieval_planner_model = os.getenv("RETRIEVAL_PLANNER_MODEL", "gpt-5")
        self.openai_api_key = os.getenv("OPENAI_API_KEY", "")
        self.retrieval_vectorstore_path = os.getenv("AX_DOCS_VECTORSTORE_PATH", "")
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.output_dir = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")


# A singleton instance for easy access throughout the package
settings = Settings()