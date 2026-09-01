"""
Configuration management for the Honegumi RAG Assistant.

This module defines a simple dataclass, :class:`Settings`, that holds
configuration values for the pipeline.  These values can be overridden
via environment variables or programmatically at runtime.  A global
instance, :data:`settings`, is created on import for convenience.

The following configuration options are supported:

``model_name``
    The name of the Claude model to use for the Parameter Selector.
    Defaults to ``"claude-sonnet-5"``.  Override via the
    ``ANTHROPIC_MODEL_NAME`` environment variable.

``anthropic_api_key``
    Your Anthropic API key.  The assistant will raise an exception if
    attempting to call the API when this value is empty.  Override via
    the ``ANTHROPIC_API_KEY`` environment variable.

``retrieval_vectorstore_path``
    Path to a serialized vector store (e.g. FAISS index) containing the
    Ax documentation.  When supplied, the retriever will attempt to
    load this index and perform semantic searches.  Override via the
    ``AX_DOCS_VECTORSTORE_PATH`` environment variable.

``retrieval_top_k``
    The number of documents to return from the retriever.  Defaults to 5.
    Override via the ``RETRIEVAL_TOP_K`` environment variable.

``embedding_model``
    Name of the local sentence-transformers model used to embed both the
    documentation corpus and retrieval queries.  This must match the
    model the vector store was built with, or retrieval will return
    meaningless results.  Override via ``EMBEDDING_MODEL``.

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


# Default Claude model for every agent in the pipeline.  Claude model IDs are
# complete as written -- never append a date suffix.
DEFAULT_MODEL = "claude-sonnet-5"

# Default local embedding model.  bge-large-en-v1.5 scores 64.23 on the MTEB
# average, effectively matching OpenAI's text-embedding-3-large (64.6) while
# running locally at no cost.  Its 512-token maximum sequence length is why
# the vector store builder chunks at 1400 characters rather than 2000.
DEFAULT_EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"


@dataclass
class Settings:
    """Container for configuration values.

    Attributes
    ----------
    model_name : str
        Identifier of the Claude model used by the Parameter Selector
        (both extraction stages).  Overridable via ``ANTHROPIC_MODEL_NAME``.
    code_writer_model : str
        Model to use specifically for the Code Writer agent.  Override via
        the ``CODE_WRITER_MODEL`` environment variable.
    reviewer_model : str
        Model to use specifically for the Reviewer agent.  Override via
        the ``REVIEWER_MODEL`` environment variable.
    retrieval_planner_model : str
        Model to use specifically for the Retrieval Planner agent.  Override
        via the ``RETRIEVAL_PLANNER_MODEL`` environment variable.
    anthropic_api_key : str
        Secret API key used to authenticate with the Anthropic API.  You
        must set this in your environment or the pipeline will raise
        an exception when attempting to call the LLM.
    structured_max_tokens : int
        Output token ceiling for the structured-output agents (parameter
        extraction, retrieval planning, review).  Anthropic requires an
        explicit ``max_tokens`` on every request, and the ceiling covers
        thinking *and* visible output, so this leaves generous headroom
        above the few hundred tokens of JSON these agents actually emit.
    code_writer_max_tokens : int
        Output token ceiling for the Code Writer.  Generated scripts run
        around 2,000 tokens; the remaining budget is headroom for adaptive
        thinking.  Raise this if you increase ``code_writer_effort``.
    code_writer_effort : str
        Reasoning effort for the Code Writer: one of ``"low"``,
        ``"medium"``, ``"high"``, ``"xhigh"`` or ``"max"``.  Controls how
        deeply Claude reasons before rewriting the skeleton.  Override via
        ``CODE_WRITER_EFFORT``.
    retrieval_vectorstore_path : str
        Optional path to a vector store used by the retriever to fetch
        Ax documentation.  If empty, no retrieval will be performed and
        the ``contexts`` field of the state will remain an empty list.
    retrieval_top_k : int
        Number of context snippets to return from the vector store.  A
        small number (5-10) keeps the LLM context manageable.
    embedding_device : str
        Torch device for the local embedding model: ``"cpu"`` (default),
        ``"mps"`` or ``"cuda"``.  Defaults to CPU deliberately.  Letting
        sentence-transformers auto-select picks ``mps`` on Apple Silicon, and
        several retrievers entering Metal kernel compilation concurrently
        deadlock -- no error, just a pinned GPU.  The retriever warms the model
        up before releasing it to the parallel branches, which makes non-CPU
        devices safe, but CPU is the conservative default and is more than fast
        enough for query embedding.  Override via ``EMBEDDING_DEVICE``.
    embedding_model : str
        Local sentence-transformers model used for both indexing and
        querying.  Must match the model the store was built with.
    output_dir : str
        Directory where the generated code and artefacts should be
        written by the :func:`run` function.  The directory will be
        created if it does not exist.
    debug : bool
        Whether to enable debug mode with verbose output showing all
        decisions, parameters, and intermediate steps. Set at runtime
        via CLI flag. Defaults to False.
    stream_code : bool
        Whether the Code Writer should stream its output token by token.
        Set at runtime; enabled when the Reviewer is disabled.
    """

    model_name: str = os.getenv("ANTHROPIC_MODEL_NAME", DEFAULT_MODEL)
    code_writer_model: str = os.getenv("CODE_WRITER_MODEL", DEFAULT_MODEL)
    reviewer_model: str = os.getenv("REVIEWER_MODEL", DEFAULT_MODEL)
    retrieval_planner_model: str = os.getenv("RETRIEVAL_PLANNER_MODEL", DEFAULT_MODEL)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    structured_max_tokens: int = int(os.getenv("STRUCTURED_MAX_TOKENS", "8192"))
    code_writer_max_tokens: int = int(os.getenv("CODE_WRITER_MAX_TOKENS", "16000"))
    code_writer_effort: str = os.getenv("CODE_WRITER_EFFORT", "high")
    retrieval_vectorstore_path: str = os.getenv("AX_DOCS_VECTORSTORE_PATH", "")
    retrieval_top_k: int = int(os.getenv("RETRIEVAL_TOP_K", "5"))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")
    output_dir: str = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")
    debug: bool = False  # Set at runtime, not from environment
    stream_code: bool = False  # Set at runtime to enable streaming output

    def reload_from_env(self):
        """Reload settings from environment variables.

        Useful when environment variables are set after the module is imported,
        such as in Jupyter/Colab notebooks.
        """
        self.model_name = os.getenv("ANTHROPIC_MODEL_NAME", DEFAULT_MODEL)
        self.code_writer_model = os.getenv("CODE_WRITER_MODEL", DEFAULT_MODEL)
        self.reviewer_model = os.getenv("REVIEWER_MODEL", DEFAULT_MODEL)
        self.retrieval_planner_model = os.getenv("RETRIEVAL_PLANNER_MODEL", DEFAULT_MODEL)
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.structured_max_tokens = int(os.getenv("STRUCTURED_MAX_TOKENS", "8192"))
        self.code_writer_max_tokens = int(os.getenv("CODE_WRITER_MAX_TOKENS", "16000"))
        self.code_writer_effort = os.getenv("CODE_WRITER_EFFORT", "high")
        self.retrieval_vectorstore_path = os.getenv("AX_DOCS_VECTORSTORE_PATH", "")
        self.retrieval_top_k = int(os.getenv("RETRIEVAL_TOP_K", "5"))
        self.embedding_model = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        self.embedding_device = os.getenv("EMBEDDING_DEVICE", "cpu")
        self.output_dir = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")


# A singleton instance for easy access throughout the package
settings = Settings()
