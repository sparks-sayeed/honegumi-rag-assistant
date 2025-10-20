# Honegumi RAG Assistant

**Agentic Code Generation for Bayesian Optimization**

![Honegumi RAG Assistant Pipeline](_static/honegumi_rag_assistant_logo.JPG)

*An intelligent AI assistant that converts natural language problem descriptions into ready-to-run Bayesian optimization code using Meta's [Ax Platform](https://ax.dev/)*

## Overview

**Honegumi RAG Assistant** is an advanced agentic AI system that automatically generates high-quality, executable Python code for Bayesian optimization experiments. Built on top of [**Honegumi**](https://honegumi.readthedocs.io/en/latest/), it uses **LangGraph** and **OpenAI GPT models** to orchestrate multiple specialized agents that collaborate to understand your optimization problem, retrieve relevant documentation, and generate production-ready code using the [**Ax Platform**](https://ax.dev/).

Simply describe your optimization problem in plain English, and the assistant produces complete, runnable code tailored to your specific requirements.

## Key Capabilities

- **Natural language to code**: Describe optimization problems conversationally
- **Intelligent RAG**: Parallel retrieval of relevant Ax documentation to supplement skeleton code
- **Built on Honegumi**: Leverages [Honegumi](https://honegumi.readthedocs.io/en/latest/) for deterministic skeleton generation
- **Multi-agent architecture**: Specialized agents for parameter extraction, retrieval planning, and code writing
- **Flexible model selection**: Mix GPT-o1 and GPT-4o models for cost-performance optimization

## Quick Start

Install via pip:

```bash
pip install honegumi-rag-assistant
```

Run the CLI:

```bash
honegumi-rag
```

The assistant will prompt you:

```
Please describe your Bayesian optimization problem.
(Press Enter when finished)

Your problem:
```

Press Enter after describing your problem, and the code will be generated!

Or try it in [Google Colab](https://colab.research.google.com/github/hasan-sayeed/honegumi_rag_assistant/blob/main/notebooks/honegumi_rag_colab_tutorial.ipynb)!


## Documentation

```{toctree}
:maxdepth: 1

readme
contributing
changelog
api/modules
```
