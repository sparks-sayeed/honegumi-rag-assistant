# Honegumi RAG Assistant

An **agentic workflow** for Bayesian optimization code generation, built with **LangGraph** and powered by Anthropic Claude models. Honegumi RAG Assistant codifies the end-to-end pipeline—parameter extraction, skeleton code generation, documentation retrieval, and code synthesis—into reusable nodes orchestrated as a LangGraph. LangSmith integration tracks and visualizes your graph executions. The result? Complete, ready-to-run Bayesian optimization code generated in seconds from natural language descriptions.

---

## 🚀 Why Honegumi RAG Assistant?

- **Agentic LangGraph design** lets you describe problems in natural language and get production-ready code
- **Fast iteration**: eliminate manual coding and focus on science
- **Built on Honegumi**: Leverages deterministic skeleton generation from [Honegumi](https://honegumi.readthedocs.io/en/latest/)
- **Intelligent RAG**: Retrieves relevant [Ax Platform](https://ax.dev/) documentation to enhance code generation
- **LangSmith-backed** for graph tracking, versioning, and observability
- Production-ready: versionable, testable, pip-installable

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
- Build a vector store from Ax Platform documentation
- Describe your optimization problem and generate code
- View the generated code in your Google Drive

The tutorial runs entirely in Colab—no local setup required. All you need is access to your Google Drive and a valid Anthropic API key (LangSmith optional).

---

## Installation via pip

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
   LANGCHAIN_API_KEY=lsv2_...
   ```

4. Build vector store (one-time setup)
   
   For best results with documentation retrieval, build the vector store:
   ```bash
   # Download the build script
   wget https://raw.githubusercontent.com/hasan-sayeed/honegumi_rag_assistant/main/scripts/build_vector_store.py
   
   # Run it
   python build_vector_store.py --output ./ax_docs_vectorstore
   
   # Set the path in your .env
   echo "AX_DOCS_VECTORSTORE_PATH=./ax_docs_vectorstore" >> .env
   ```

5. Run the assistant

   Now, start the `honegumi-rag` pipeline:
   ```bash
   honegumi-rag
   ```

   **Honegumi RAG Assistant** will:

   - Prompt you to describe your Bayesian optimization problem
   - Extract parameters (objectives, constraints, search space, etc.)
   - Generate a deterministic code skeleton using Honegumi
   - Retrieve relevant Ax Platform documentation
   - Generate complete, runnable Python code
   - Stream the code generation in real-time
   - Optionally save the generated code to a file

---

## Usage Examples

### Interactive Mode (Default)

```bash
honegumi-rag
```

You'll be prompted:
```
Please describe your Bayesian optimization problem.
(Press Enter when finished)

Your problem:
```

The assistant generates complete code and displays it in real-time.

### Save Generated Code

```bash
honegumi-rag --output-dir ./my_experiments
```

### Use Different Models

Customize which GPT models are used for each agent:

```bash
honegumi-rag \
  --code-writer-model claude-opus-5 \
  --param-selector-model claude-sonnet-5 \
  --retrieval-planner-model claude-sonnet-5
```

---

## Key Features

### Multi-Agent Architecture
- **Parameter Selector**: Extracts optimization parameters from natural language
- **Skeleton Generator**: Uses [Honegumi](https://honegumi.readthedocs.io/en/latest/) for deterministic code templates
- **Retrieval Planner**: Generates intelligent documentation queries
- **Parallel Retrievers**: Concurrent documentation retrieval for speed
- **Code Writer**: Claude-powered code generation with adaptive thinking and streaming
- **Reviewer** (optional): Quality assessment and refinement

### Advanced Features
- **LangSmith Integration**: Full tracing and debugging support
- **Streaming Output**: See code generation in real-time
- **Local Embeddings**: Documentation retrieval runs offline via sentence-transformers — no embedding API key or cost
- **Tunable Reasoning**: Configurable effort level to trade reasoning depth against token spend
- **Optional Save**: Print code or save to file—your choice

---

## Command Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--output-dir` | Save generated script to specified directory (if omitted, code is only printed, not saved) | `None` (no save) |
| `--debug` | Enable debug mode with detailed logging | `False` |
| `--review` | Enable Reviewer agent (slower, more accurate) | `False` |
| `--effort` | Code Writer reasoning effort (`low`..`max`) | `high` |
| `--param-selector-model` | Model for Parameter Selector | `claude-sonnet-5` |
| `--retrieval-planner-model` | Model for Retrieval Planner | `claude-sonnet-5` |
| `--code-writer-model` | Model for Code Writer agent | `claude-sonnet-5` |
| `--reviewer-model` | Model for Reviewer agent | `claude-sonnet-5` |

---

## Example Problems

### Chemical Process Optimization
```
Optimize temperature (100-300°C), pressure (1-5 bar), and catalyst concentration (0.1-1.0 M) 
to maximize conversion rate in a catalytic reaction.
```

### Materials Design
```
Optimize composition of a polymer blend: Component A (0-100%), Component B (0-100%), 
and curing temperature (80-150°C) to maximize tensile strength while minimizing cost.
```

### Machine Learning Hyperparameters
```
Optimize neural network hyperparameters: learning rate (1e-5 to 1e-1), 
batch size (16 to 256), and dropout rate (0.1 to 0.5) to maximize validation accuracy.
```

---

## Documentation & Support

- **Full Documentation**: [GitHub Repository](https://github.com/hasan-sayeed/honegumi_rag_assistant)
- **Google Colab Tutorial**: [Interactive Tutorial](https://colab.research.google.com/github/hasan-sayeed/honegumi_rag_assistant/blob/main/notebooks/honegumi_rag_colab_tutorial.ipynb)
- **Issues**: [GitHub Issues](https://github.com/hasan-sayeed/honegumi_rag_assistant/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hasan-sayeed/honegumi_rag_assistant/discussions)
- **Email**: hasan.sayeed@utah.edu

---

## Feedback & Feature Requests

This project demonstrates a **proof of concept** of what's possible with agentic systems for Bayesian optimization. While Honegumi RAG Assistant works out-of-the-box for many scenarios, your use case may involve more complex pipelines, custom constraints, or specific modeling needs.

Have something bigger in mind? Want Honegumi RAG Assistant to handle advanced features or integrate with your workflow?

**We'd love to hear from you!**

- Open a [GitHub issue](https://github.com/hasan-sayeed/honegumi_rag_assistant/issues)
- Start a [discussion](https://github.com/hasan-sayeed/honegumi_rag_assistant/discussions)
- Or reach out directly at hasan.sayeed.71.93@gmail.com

---

## License

MIT License - see [LICENSE.txt](https://github.com/hasan-sayeed/honegumi_rag_assistant/blob/main/LICENSE.txt) for details.

---

## Acknowledgments

- Built with [PyScaffold](https://pyscaffold.org/)
- Powered by [LangGraph](https://github.com/langchain-ai/langgraph) and [LangChain](https://github.com/langchain-ai/langchain)
- Skeleton generation by [Honegumi](https://honegumi.readthedocs.io/en/latest/)
- Uses Meta's [Ax Platform](https://ax.dev/) for Bayesian optimization
