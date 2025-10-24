# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Problem statement collection**: Created `data/raw/problem_statements.yaml` with initial problem statement for ceramic sintering optimization. Includes natural version (underspecified), corrected version (conversational with density units), Honegumi-specific grid selections (objective, model, task, constraints), links to relevant Honegumi tutorials (SOBO, Batch Fully Bayesian), and references to related Ax repository issues. Template for collecting 100 problem statements across different personas in physical sciences.

## [0.1.7] - 2025-10-20

### Added
- **Settings reload method**: Added `settings.reload_from_env()` method to allow reloading configuration from environment variables after module import. Fixes Colab/Jupyter notebook issue where settings were cached before environment variables were set.

### Changed
- **Colab tutorial improvements**: Updated Step 5 to explicitly call `settings.reload_from_env()` and print confirmation of vector store path. Ensures vector store is properly detected in notebooks.
- **Build script error message**: Updated orchestrator error message to use `python -m honegumi_rag_assistant.build_vector_store` instead of outdated `scripts/build_vector_store.py` path.

## [0.1.6] - 2025-10-20

### Fixed
- **Vector store Ax version pinning**: `src/honegumi_rag_assistant/build_vector_store.py` now defaults to cloning Ax v0.4.3 (matching honegumi dependency) instead of latest main branch. This prevents documentation version mismatch where LLM receives v1.x docs but generates v0.4.3 code. New `--ax-version` parameter allows specifying different versions when needed. Addresses issue raised in PR #3.
- Vector store missing condition now properly detected when path is configured but directory doesn't exist

### Changed
- **Build script location**: Moved `build_vector_store.py` from `scripts/` to `src/honegumi_rag_assistant/` to make it available as a module for pip-installed users. Usage: `python -m honegumi_rag_assistant.build_vector_store`
- **Retrieval UX improvements**: Simplified retrieval output to show only essential information. In non-debug mode, displays "Planning X parallel retrievals..." followed by success message with context count and timing, or clear error if vector store missing. Removes verbose query details and individual retriever timings from default output.
- **Vector store metadata**: Build script now saves `metadata.json` with build details including Ax version, build date, chunk configuration, and document counts for better provenance tracking.
- Removed verbose per-query retrieval details from non-debug output
- Cleaner success message: "✓ Retrieved X contexts in Y.XXs"

### Added
- Clear warning when vector store is not found: prompts user to build it
- Vector store path existence check before attempting retrieval

## [0.1.5] - 2025-10-20

### Changed
- Simplified ReadTheDocs navigation structure (removed unnecessary sections)
- Reduced documentation table of contents depth from 2 to 1 for cleaner appearance
- Removed License and Authors pages from main navigation

### Documentation
- Added interactive CLI usage example showing the prompt and user input flow
- Improved Quick Start section with clearer instructions
- Renamed "Contents" to "Documentation" for better clarity

## [0.1.4] - 2025-10-20

### Added
- Project logo in ReadTheDocs sidebar
- ReadTheDocs badge to README.md tracking stable version
- Enhanced ReadTheDocs theme options (version display, external link styling)

### Changed
- Updated project name from "honegumi_rag_assistant" to "Honegumi RAG Assistant" in documentation
- Replaced Features section with Key Capabilities from README in docs/index.md
- Added project overview and description to documentation landing page
- Updated README.md badge to track stable version instead of latest
- Improved documentation landing page with logo and better structure

### Documentation
- Set ReadTheDocs default version to `stable` for production use
- Configured logo to appear in documentation sidebar
- Enhanced theme configuration for better user experience

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
