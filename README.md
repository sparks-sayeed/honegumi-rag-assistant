[![Project generated with PyScaffold](https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold)](https://pyscaffold.org/)
[![Documentation Status](https://readthedocs.org/projects/honegumi-rag-assistant/badge/?version=stable)](https://honegumi-rag-assistant.readthedocs.io/en/stable/)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hasan-sayeed/honegumi_rag_assistant/blob/main/notebooks/honegumi_rag_colab_tutorial.ipynb)
[![GitHub issues](https://img.shields.io/github/issues/hasan-sayeed/honegumi_rag_assistant)](https://github.com/hasan-sayeed/honegumi_rag_assistant/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/hasan-sayeed/honegumi_rag_assistant)](https://github.com/hasan-sayeed/honegumi_rag_assistant/discussions)
[![Last Committed](https://img.shields.io/github/last-commit/hasan-sayeed/honegumi_rag_assistant)](https://github.com/hasan-sayeed/honegumi_rag_assistant/commits/main/)

# **Honegumi RAG Assistant**: Agentic Code Generation for Bayesian Optimization

![Honegumi RAG Assistant Pipeline](docs/_static/honegumi_rag_assistant_logo.JPG)

*Figure: Schematic of the Honegumi RAG Assistant agentic pipeline for generating Bayesian optimization code from natural language.*

*An intelligent AI assistant that converts natural language problem descriptions into ready-to-run Bayesian optimization code using Meta's [Ax Platform](https://ax.dev/)*

---

## Overview

**Honegumi RAG Assistant** is an advanced agentic AI system that automatically generates high-quality, executable Python code for Bayesian optimization experiments. Built on top of [**Honegumi**](https://honegumi.readthedocs.io/en/latest/), it uses **LangGraph** and **Anthropic Claude models** to orchestrate multiple specialized agents that collaborate to understand your optimization problem, retrieve relevant documentation, and generate production-ready code using the [**Ax Platform**](https://ax.dev/).

[**Honegumi**](https://honegumi.readthedocs.io/en/latest/) provides deterministic skeleton code generation based on problem parameters, and this RAG Assistant enhances it by retrieving relevant [Ax Platform](https://ax.dev/) documentation to help the LLM transform the skeleton into complete, domain-specific code tailored to your problem.

Simply describe your optimization problem in plain English, and the assistant produces complete, runnable code tailored to your specific requirements.

### Key Capabilities

- **Natural language to code**: Describe optimization problems conversationally
- **Intelligent RAG**: Parallel retrieval of relevant Ax documentation to supplement skeleton code
- **Built on Honegumi**: Leverages [Honegumi](https://honegumi.readthedocs.io/en/latest/) for deterministic skeleton generation
- **Multi-agent architecture**: Specialized agents for parameter extraction, retrieval planning, and code writing
- **Local embeddings**: Documentation retrieval runs entirely offline via sentence-transformers — no embedding API key, no per-token cost
- **Tunable reasoning**: Adaptive thinking with a configurable effort level to trade depth against token spend

---

## Key Features

### Multi-Agent Architecture
- **Intelligent Parameter Detection**: Two-stage extraction process with chain-of-thought reasoning for accurate grid parameter selection
- **Skeleton Generator**: Uses [Honegumi](https://honegumi.readthedocs.io/en/latest/) to create deterministic code templates
- **Retrieval Planner**: Intelligently generates retrieval queries based on problem complexity
- **Parallel Retrievers**: For efficient documentation retrieval - multiple queries executed concurrently to minimize latency
- **Code Writer**: Claude-powered code generation with adaptive thinking and streaming output
- **Reviewer** (optional): Quality assessment and revision requests (disabled by default for speed)

### Advanced Features
- **LangSmith Integration**: Full tracing support for debugging and monitoring

---

## 🛠️ Prerequisites

- **Conda** (Miniconda or Anaconda)
- **Python 3.11+**
- [**Anthropic API key**](https://console.anthropic.com/settings/keys)
- [**LangSmith API key**](https://docs.smith.langchain.com/administration/how_to_guides/organization_management/create_account_api_key) (optional)

---

## 📘 Google Colab Tutorial

To help you get started quickly, we've prepared an interactive Google Colab tutorial:

**[Google Colab Tutorial: Getting Started with Honegumi RAG Assistant](https://colab.research.google.com/github/hasan-sayeed/honegumi_rag_assistant/blob/main/notebooks/honegumi_rag_colab_tutorial.ipynb)**

In this tutorial, you'll learn how to:

- Install Honegumi RAG Assistant and all necessary dependencies on Colab
- Set up your API key using Colab Secrets
- Build a local vector store from Ax Platform documentation
- Describe your optimization problem and generate code
- View the generated code in your Google Drive

The tutorial runs entirely in Colab—no local setup required. All you need is access to your Google Drive and a valid Anthropic API key (LangSmith optional).

---

## Installation

### Quick Install via pip

1. Create & activate a conda environment
   ```bash
   conda create -n honegumi_rag python=3.11 -y
   conda activate honegumi_rag
   ```

2. Install via pip
   ```bash
   pip install honegumi-rag-assistant
   ```

3. Configure your API keys

   **Honegumi RAG Assistant** will automatically look for a file named `.env` in your current working directory (or any parent) and load any keys it finds.

   In the folder where you'll run the CLI (or in any ancestor), create a file called **`.env`** containing:

   ```bash
   ANTHROPIC_API_KEY=sk-ant-...
   LANGCHAIN_API_KEY=lsv2_...   # optional, for tracing
   ```

4. Build vector store (one-time setup)
   
   For best results with documentation retrieval, build the vector store:
   ```bash
   # Run the build script from the package
   python -m honegumi_rag_assistant.build_vector_store
   ```

   Embeddings are computed locally, so **no API key is needed for this step**.
   The first run downloads the embedding model (~1.3 GB) and embedding the
   corpus takes several minutes on CPU. Until it finishes, the assistant still
   works — it just generates code without documentation retrieval.

   **Note**: By default, this uses Ax v0.4.3 (matching honegumi). To use a different version:
   ```bash
   python -m honegumi_rag_assistant.build_vector_store --ax-version 0.4.0
   ```

   If you change the embedding model, you must rebuild the store: vector
   dimensions and semantics differ between models, and a mismatch between build
   time and query time yields meaningless results rather than an error.

5. Run the assistant
   ```bash
   honegumi-rag
   ```

### From Source: Clone & Run

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/hasan-sayeed/honegumi_rag_assistant.git
   cd honegumi_rag_assistant
   ```

2. **Create Conda Environment** (recommended):
   ```bash
   conda env create -f environment.yml
   conda activate honegumi_rag_assistant
   ```

3. **Install in editable mode**:
   ```bash
   pip install -e .
   ```

4. **Configure API Keys**:
   
   In the project root directory, create a file called **`.env`** containing:
   
   ```bash
   # Required: Anthropic API key for the Claude agents.
   # Embeddings run locally, so this is the only credential needed.
   ANTHROPIC_API_KEY=sk-ant-your-actual-api-key-here

   # Optional: LangSmith tracing (recommended for debugging)
   LANGCHAIN_API_KEY=your-langchain-api-key-here
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=Honegumi RAG Assistant

   # Path to FAISS vector store (set after building it)
   AX_DOCS_VECTORSTORE_PATH=data/processed/ax_docs_vectorstore
   RETRIEVAL_TOP_K=5
   ```

5. **Build Vector Store for RAG**:
   
   For best results with documentation retrieval, run:
   ```bash
   # Build vector store (one-time setup, defaults to Ax v0.4.3)
   python -m honegumi_rag_assistant.build_vector_store
   
   # Or specify a different Ax version
   python -m honegumi_rag_assistant.build_vector_store --ax-version 0.4.0
   ```
   
   The vector store will be saved to `data/processed/ax_docs_vectorstore/` and automatically loaded if present. Embeddings run locally via
   [`BAAI/bge-large-en-v1.5`](https://huggingface.co/BAAI/bge-large-en-v1.5) — no API key, no per-token cost. Build metadata (Ax version, embedding model, chunk settings) is recorded in `metadata.json` alongside the index.

6. **Verify Installation**:
   ```bash
   honegumi-rag --help
   # Or: python -m honegumi_rag_assistant --help
   ```

---

## Usage

Run the assistant:

```bash
honegumi-rag    # if you installed via pip
# Or: python -m honegumi_rag_assistant
```

The assistant will prompt you to describe your Bayesian optimization problem in natural language:

```
Your problem:
Optimize temperature (50-200°C) and pressure (1-10 bar) for maximum yield in a chemical reaction.
```

After typing your problem description, press **Enter**. The assistant will process your problem and generate code in real-time (streaming), displaying it as it's created.

**By default, code is only printed to the console (not saved)**. To save the generated script to a file, use `--output-dir`:

```bash
honegumi-rag --output-dir ./my_experiments
```

**Optional: Enable debug mode** to see detailed agent decisions:

```bash
honegumi-rag --debug
```

### Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Save generated script to specified directory (if omitted, code is only printed, not saved) | `None` (no save) |
| `--debug` | Enable debug mode with detailed logging, including a summary of the Code Writer's reasoning | `False` |
| `--review` | Enable Reviewer agent (slower, more accurate; disables streaming) | `False` |
| `--effort` | Reasoning effort for the Code Writer: `low`, `medium`, `high`, `xhigh`, `max` | `high` |
| `--param-selector-model` | Model for Parameter Selector | `claude-sonnet-5` |
| `--retrieval-planner-model` | Model for Retrieval Planner | `claude-sonnet-5` |
| `--code-writer-model` | Model for Code Writer agent | `claude-sonnet-5` |
| `--reviewer-model` | Model for Reviewer agent | `claude-sonnet-5` |


### Model Selection Guide

All agents default to **Claude Sonnet 5**, which gives the best balance of
capability and cost for this pipeline.

**Highest capability** — for hard, many-parameter or many-objective problems:
```bash
--code-writer-model claude-opus-5 --effort xhigh
```

**Budget** — the extraction agents do constrained classification against a fixed
schema, so they downgrade well; the Code Writer is where capability shows up:
```bash
--param-selector-model claude-haiku-4-5 \
--retrieval-planner-model claude-haiku-4-5 \
--code-writer-model claude-sonnet-5
```

### Reasoning Effort

The Code Writer runs with **adaptive thinking**: Claude decides per request how
much to reason before rewriting the skeleton. `--effort` sets the ceiling on that
reasoning, trading token spend against depth. `high` is the default; `medium`
costs less and is often sufficient for straightforward problems.

Note that on the Anthropic API, `max_tokens` caps thinking *plus* visible output.
If you raise `--effort` to `xhigh` or `max`, also raise `CODE_WRITER_MAX_TOKENS`
(default 16000) so long reasoning can't crowd out the generated script.

Rather than guessing, measure: the repo ships 20 problem statements with
ground-truth grid selections in `data/raw/problem_statements.yaml` and a
validation workflow that syntax-checks, imports and executes generated scripts.
Run a sweep at two effort levels and compare.

---

## Project Organization

```
├── AUTHORS.md              <- List of developers and maintainers
├── CHANGELOG.md            <- Changelog to keep track of new features and fixes
├── CONTRIBUTING.md         <- Guidelines for contributing to this project
├── LICENSE.txt             <- MIT License
├── README.md               <- This file
├── environment.yml         <- Conda environment specification
├── .env.example            <- Example environment variables (COPY TO .env)
│
├── configs/                <- Configuration files
│
├── data/
│   ├── raw/                <- Original, immutable data
│   └── processed/          <- Processed data (vector stores)
│       └── ax_docs_vectorstore/  <- FAISS vector store for Ax docs
│
├── src/
│   └── honegumi_rag_assistant/
│       ├── __init__.py
│       ├── __main__.py     <- CLI entry point
│       ├── orchestrator.py <- LangGraph pipeline orchestration
│       ├── app_config.py   <- Settings and configuration
│       ├── states.py       <- State definitions with custom reducers
│       ├── extractors.py   <- Pydantic schemas for structured extraction
│       ├── build_vector_store.py  <- Build FAISS vector store
│       └── nodes/          <- Agent implementations
│           ├── parameter_selector.py   <- Parameter extraction
│           ├── skeleton_generator.py   <- Honegumi skeleton generation
│           ├── retrieval_planner.py    <- Retrieval query generation
│           ├── retriever.py            <- Parallel FAISS retrieval
│           ├── code_writer.py          <- Claude code generation
│           └── reviewer.py             <- Code quality review
│
├── scripts/
│   ├── batch_process.py    <- Batch processing utility
│   └── test_vector_store.py <- Vector store testing
│
├── tests/                  <- Unit tests (pytest)
└── docs/                   <- Documentation
```

---

## Advanced Topics

### LangSmith Tracing

Enable comprehensive debugging:

1. Get API key from https://smith.langchain.com/
2. Add to `.env`:
   ```
   LANGCHAIN_API_KEY=your-key
   LANGCHAIN_TRACING_V2=true
   LANGCHAIN_PROJECT=Honegumi RAG Assistant
   ```
3. View all LLM calls, agent decisions, and timing in LangSmith dashboard

---

## Citation

If you use Honegumi RAG Assistant in your research, please cite:

```bibtex
@software{honegumi_rag_assistant2025,
  title = {Honegumi RAG Assistant: Agentic Code Generation for Bayesian Optimization},
  author = {Sayeed, Hasan Muhammad},
  year = {2025},
  url = {https://github.com/hasan-sayeed/honegumi_rag_assistant}
}
```

---

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**How to contribute**:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see [LICENSE.txt](LICENSE.txt) for details.

---

## Acknowledgments

- Built with [PyScaffold](https://pyscaffold.org/)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Skeleton generation by [Honegumi](https://honegumi.readthedocs.io/en/latest/)
- Uses Meta's [Ax Platform](https://ax.dev/) for Bayesian optimization
- Code generation by Anthropic's [Claude](https://www.anthropic.com/claude) models
- Local embeddings via [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)

---

## Feedback & Feature Requests

This project demonstrates a **proof of concept** of what's possible with agentic systems for Bayesian optimization code generation. While Honegumi RAG Assistant works out-of-the-box for many scenarios, your use case may involve more complex pipelines, custom constraints, multi-objective optimization, or specific modeling needs.

Have something bigger in mind? Want Honegumi RAG Assistant to handle advanced features, integrate with your workflow, or adapt to your domain?

**We'd love to hear from you!**

- Open a [GitHub issue](https://github.com/hasan-sayeed/honegumi_rag_assistant/issues)
- Start a [GitHub discussion](https://github.com/hasan-sayeed/honegumi_rag_assistant/discussions)
- Or reach out directly at hasan.sayeed@utah.edu

---

## Support

For questions, bug reports, or feature requests:
- **GitHub Issues**: https://github.com/hasan-sayeed/honegumi_rag_assistant/issues
- **Email**: hasan.sayeed@utah.edu

---

<!-- pyscaffold-notes -->

## Note

This project has been set up using [PyScaffold] 4.6 and the [dsproject extension] 0.7.2.

[conda]: https://docs.conda.io/
[pre-commit]: https://pre-commit.com/
[Jupyter]: https://jupyter.org/
[nbstripout]: https://github.com/kynan/nbstripout
[Google style]: http://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings
[PyScaffold]: https://pyscaffold.org/
[dsproject extension]: https://github.com/pyscaffold/pyscaffoldext-dsproject
