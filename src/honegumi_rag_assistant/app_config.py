"""
Configuration management for the Honegumi RAG Assistant.

This module defines a simple dataclass, :class:`Settings`, that holds
configuration values for the pipeline.  These values can be overridden
via environment variables or programmatically at runtime.  A global
instance, :data:`settings`, is created on import for convenience.

The following configuration options are supported:

``model_name``
    The name of the Claude model to use for the Parameter Selector.
    Defaults to ``"claude-haiku-4-5"``.  Override via the
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

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Dict, Iterator
import os

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip


# Default Claude model for the structured-output agents (parameter selection,
# retrieval planning, review).  They emit a few hundred tokens of schema-
# constrained JSON, which Haiku handles at half Sonnet's price.  Claude model
# IDs are complete as written -- never append a date suffix.
DEFAULT_MODEL = "claude-haiku-4-5"

# The Code Writer stays on Sonnet.  It is the only agent that passes
# ``thinking={"type": "adaptive"}`` and ``output_config={"effort": ...}``, and
# both are 4.6+ features: Haiku 4.5 rejects them with a 400.  It is also the
# quality-critical agent -- a weaker model here shows up directly as stale-API
# errors in the generated script, which is what the sweep measures.
DEFAULT_CODE_WRITER_MODEL = "claude-sonnet-5"

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
    embedding_local_files_only : bool
        When True, load the embedding model strictly from the local cache and
        never contact the HuggingFace Hub.  Loading otherwise issues ~25 HTTP
        requests to check the cached weights are current, which is startup
        latency in a container and a hard failure on a host without egress.
        Off by default, because a machine that has never downloaded the model
        needs the Hub to fetch it.  Turn it on wherever the weights are baked
        into the image.  Override via ``EMBEDDING_LOCAL_FILES_ONLY``.
    mcp_host : str
        Interface the MCP server binds to.  Defaults to ``"127.0.0.1"``:
        loopback only, so a server started on a laptop is reachable from that
        laptop and nowhere else.  A container must override this to
        ``"0.0.0.0"`` -- inside one, ``127.0.0.1`` is the container's own
        loopback, so the port would be published and still refuse every
        connection.  Override via ``MCP_HOST``.
    mcp_port : int
        Port the MCP server listens on.  Override via ``MCP_PORT``.
    mcp_auth_token : str
        Shared secret callers must present as ``Authorization: Bearer <token>``.
        Empty disables the check, which is only safe on loopback: every call to
        this server spends *your* Anthropic budget, so an open public endpoint
        is an open wallet.  :func:`~honegumi_rag_assistant.mcp_server.main`
        refuses to start on a non-loopback interface without one.  Override via
        ``MCP_AUTH_TOKEN``.
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
    code_writer_model: str = os.getenv("CODE_WRITER_MODEL", DEFAULT_CODE_WRITER_MODEL)
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
    embedding_local_files_only: bool = (
        os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "").strip().lower()
        in {"1", "true", "yes", "on"}
    )
    mcp_host: str = os.getenv("MCP_HOST", "127.0.0.1")
    mcp_port: int = int(os.getenv("MCP_PORT", "8000"))
    mcp_auth_token: str = os.getenv("MCP_AUTH_TOKEN", "")
    output_dir: str = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")
    debug: bool = False  # Set at runtime, not from environment
    stream_code: bool = False  # Set at runtime to enable streaming output

    def reload_from_env(self):
        """Reload settings from environment variables.

        Useful when environment variables are set after the module is imported,
        such as in Jupyter/Colab notebooks.
        """
        self.model_name = os.getenv("ANTHROPIC_MODEL_NAME", DEFAULT_MODEL)
        self.code_writer_model = os.getenv("CODE_WRITER_MODEL", DEFAULT_CODE_WRITER_MODEL)
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
        self.embedding_local_files_only = (
            os.getenv("EMBEDDING_LOCAL_FILES_ONLY", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.mcp_host = os.getenv("MCP_HOST", "127.0.0.1")
        self.mcp_port = int(os.getenv("MCP_PORT", "8000"))
        self.mcp_auth_token = os.getenv("MCP_AUTH_TOKEN", "")
        self.output_dir = os.getenv("OUTPUT_DIR", "./honegumi_rag_output")


# Process-wide defaults, populated from the environment at import time.  The
# CLI entry points mutate this directly: they own the whole process and run one
# pipeline in it, so a global is the right scope for them.
_defaults = Settings()

# Per-run overrides.  A ContextVar is scoped to the current thread or async
# task rather than to the process, and LangGraph propagates the active context
# into the worker threads it spawns for parallel ``Send`` branches -- so an
# override set by one request is visible to every node of that request, and to
# no other request.
_run_overrides: ContextVar[Dict[str, Any]] = ContextVar("run_overrides", default={})


class _SettingsProxy:
    """Attribute access that prefers the active run's overrides.

    Reads resolve against :data:`_run_overrides` first and fall back to the
    process-wide :data:`_defaults`.  Writes always land on the defaults, which
    keeps ``settings.debug = True`` working for the CLI and keeps
    ``monkeypatch.setattr(settings, ...)`` working in the tests.
    """

    def __getattr__(self, name: str) -> Any:
        overrides = _run_overrides.get()
        if name in overrides:
            return overrides[name]
        return getattr(_defaults, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(_defaults, name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_defaults, name)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<settings overrides={_run_overrides.get()!r} defaults={_defaults!r}>"


# A singleton instance for easy access throughout the package
settings = _SettingsProxy()


@contextmanager
def override_settings(**values: Any) -> Iterator[None]:
    """Apply per-run settings visible only to the current request.

    Any concurrent caller -- another Gradio request, another MCP tool call --
    keeps its own values, because the overrides live in a
    :class:`~contextvars.ContextVar` rather than on the shared singleton.
    Nested scopes merge, with the inner values winning.

    Parameters
    ----------
    **values
        Setting names and the values they should take for the duration of the
        block.  Names are not validated against :class:`Settings`, so a typo
        silently does nothing -- pass keywords that match real fields.

    Yields
    ------
    None
        The block runs with the overrides applied; they are removed on exit,
        including when the block raises.

    Examples
    --------
    >>> with override_settings(debug=True, code_writer_model="claude-opus-5"):
    ...     run_from_text("maximise yield ...")   # doctest: +SKIP
    """
    token = _run_overrides.set({**_run_overrides.get(), **values})
    try:
        yield
    finally:
        _run_overrides.reset(token)
