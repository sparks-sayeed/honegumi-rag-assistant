# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.3] - 2025-10-20

### Added
- Moved `build_vector_store.py` from `scripts/` to `src/honegumi_rag_assistant/` for pip package accessibility
- Configured PyPI to use `README_PYPI.md` for concise package description
- ReadTheDocs configuration with Google-style docstring support
- Sphinx ReadTheDocs theme
- Napoleon extension configured for Google docstrings
- Comprehensive documentation structure

### Changed
- Updated `setup.cfg` to use `README_PYPI.md` as long_description
- Enhanced `docs/index.md` with project description and features
- Updated `docs/requirements.txt` with all package dependencies
- Changed documentation theme from Alabaster to ReadTheDocs

## [0.1.1] - 2025-10-20

### Added
- Google Colab tutorial notebook (`notebooks/honegumi_rag_colab_tutorial.ipynb`)
- `README_PYPI.md` for PyPI package page
- PyPI badges to `README.md` (Colab, Issues, Discussions, Last Commit)
- "Google Colab Tutorial" section in README.md
- "Feedback & Feature Requests" section in README.md
- `requirements.txt` with exact version pinning for reproducibility

### Changed
- Updated `setup.cfg` with all dependencies using exact versions (`==`)
- Added console script entry point: `honegumi-rag` command
- Enhanced README.md with comprehensive installation and usage instructions
- Colab notebook uses programmatic API only (CLI doesn't work interactively)

### Fixed
- Package structure to include `build_vector_store.py` in pip installation

## [0.1.0] - 2025-10-20

### Added
- Initial release of Honegumi RAG Assistant
- Multi-agent LangGraph pipeline for code generation
- RAG-based retrieval from Ax Platform documentation
- FAISS vector store for fast document retrieval
- Specialized agents: IssueScout, ParameterSelector, RetrievalPlanner, CodeWriter, Reviewer
- CLI interface with `honegumi-rag` command
- Programmatic API via `run_from_text()` and `run_from_dict()`
- Debug mode for detailed execution logging
- Optional code review and refinement step
- Support for custom output directories

### Changed
- Made file saving optional (only when `--output-dir` specified)
- Removed all emoji characters from output for cleaner UX
- Added visual spacing between user input and processing output
- Moved startup banners and timing to debug-only mode

### Technical
- Built on Honegumi for deterministic skeleton generation
- Uses OpenAI GPT models (configurable GPT-4o and GPT-o1)
- Python 3.11+ required
- Dependencies: langchain, langgraph, faiss-cpu, honegumi, openai
