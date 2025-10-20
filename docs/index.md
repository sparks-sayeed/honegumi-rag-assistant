# honegumi_rag_assistant

**AI-powered Bayesian optimization code generator using RAG with Ax Platform documentation**

Honegumi RAG Assistant is an intelligent code generation tool that helps researchers and practitioners quickly create Bayesian optimization code using the Ax Platform. It combines LangGraph-based agentic workflows with retrieval-augmented generation (RAG) to provide contextually accurate code based on official Ax documentation.

## Features

- 🤖 **Agentic Workflow**: Multi-step LangGraph pipeline for intelligent code generation
- 📚 **RAG-Powered**: Retrieves relevant context from Ax Platform documentation  
- ⚡ **Fast Generation**: Optimized vector search with FAISS
- 🎯 **Parameter Detection**: Automatically identifies optimization parameters from problem descriptions
- 🔄 **Self-Review**: Built-in code review and refinement capabilities
- 💻 **Multiple Interfaces**: CLI tool and programmatic API

## Quick Start

Install via pip:

```bash
pip install honegumi-rag-assistant
```

Run the CLI:

```bash
honegumi-rag
```

Or try it in [Google Colab](https://colab.research.google.com/github/hasan-sayeed/honegumi_rag_assistant/blob/main/notebooks/honegumi_rag_colab_tutorial.ipynb)!


## Contents

```{toctree}
:maxdepth: 2

Overview <readme>
Contributions & Help <contributing>
License <license>
Authors <authors>
Changelog <changelog>
Module Reference <api/modules>
```

## Indices and tables

* {ref}`genindex`
* {ref}`modindex`
* {ref}`search`
