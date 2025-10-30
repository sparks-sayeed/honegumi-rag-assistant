# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Expanded problem set to 30 diverse BO problems with full Honegumi tutorial coverage**: Added 27 new problems (IDs 4-30) covering diverse domains and complexity levels. Each problem includes 3 personas (experimentalist_basic, industrial_practitioner, research_scientist) providing 90 total persona-problem combinations. All 6 Honegumi tutorials are represented.
- **Problems 4-14**: Initial expansion including battery electrolyte, drug formulation, catalyst synthesis, alloy composition, bioreactor, concrete mix, drug discovery pipeline, MAX phase featurization, multi-task ceramic binder, acquisition function benchmarking, and RGB liquid color matching.
- **Problems 15-30**: Additional expansion including solar cells, food formulation, laser cutting, adhesive formulation, chemical reactor, LED packaging, water treatment, spray coating, ML hyperparameters, conductive ink, fermentation medium, composite curing, optical glass, solvent extraction, battery charging, and food packaging.
- **Complete Honegumi tutorial coverage**: All problems now reference relevant tutorials (SOBO, MOBO, Batch, Featurization, Multi-Task, Benchmarking).
- **Extended problem set with 2 additional personas**: Added `industrial_practitioner` and `research_scientist` personas to original 3 problems. Problem statements now use a nested structure where each problem ID contains multiple persona variations under a `personas` key, making it easier to track different communication styles for the same underlying optimization problem.
- **Industrial practitioner persona**: Practical, cost-conscious user with production constraints and informal language style. Applied to all 30 problems.
- **Research scientist persona**: Academic researcher with formal, hypothesis-driven approach and scientific language. Applied to all 30 problems.
- **Domain diversity**: 17 domains covered including materials_science (9), pharmaceutical (2), energy (2), chemical_engineering (2), computational (2), manufacturing (2), and 11 others.
- **Complexity balance**: Simple (12), intermediate (11), advanced (7) problems for varied difficulty levels.
- **Feature diversity**: 10 multi-objective problems, 8 composition constraints, 5 order constraints, 2 sum constraints.

### Changed
- **API key environment variable**: Updated all code to use `LLM_API_KEY` instead of `OPENAI_API_KEY` for consistency with repository secrets configuration. This affects:
  - `src/honegumi_rag_assistant/app_config.py` - Settings class now reads from `LLM_API_KEY`
  - `src/honegumi_rag_assistant/nodes/code_writer.py` - Error message updated
  - `src/honegumi_rag_assistant/extractors.py` - Error message updated
  - `src/honegumi_rag_assistant/build_vector_store.py` - Environment variable check updated
  - `scripts/test_vector_store.py` - Environment variable check updated
  - `scripts/run_rag_experiments.py` - Documentation comments updated

### Added
- **Two-stage parameter extraction with solution-format structure**: Implemented chain-of-thought reasoning for grid parameter selection. Stage 1 extracts explicit problem structure using `ProblemStructureExtractor` in the same format as test problem solutions (search_space, objective, budget, batch_size, noise_model, constraints). Stage 2 uses this structured representation to make grid selections via enhanced `ParameterExtractor`. This approach ensures consistency with validation expectations and improves accuracy through explicit reasoning.
- **Solution-format Pydantic models**: Added `SearchSpaceParameter`, `ObjectiveSpec`, `ConstraintSpec`, and `ProblemStructure` models that mirror the solution structure from test problems, ensuring perfect alignment between extraction and validation.
- **Enhanced debug output**: Parameter selector now shows both Stage 1 (problem structure in solution format) and Stage 2 (grid selections) with clear formatting for better transparency and debugging.
- **Detailed grid selection rules**: Enhanced prompt includes explicit lookup table mapping extracted structure to grid parameters with special emphasis on constraint type distinctions.
- **Problem statement collection**: Created `data/raw/problem_statements.yaml` with initial problem statement for ceramic sintering optimization. Includes natural version (underspecified), corrected version (conversational with density units), Honegumi-specific grid selections (objective, model, task, constraints), links to relevant Honegumi tutorials (SOBO, Batch Fully Bayesian), and references to related Ax repository issues. Template for collecting 100 problem statements across different personas in physical sciences.
- **RAG experiment infrastructure**: Created `data/raw/rag_assistant_runs.yaml` for tracking RAG assistant experiments with experiment IDs, prompts, results, and log correlations. Added `scripts/run_rag_experiments.py` to run experiments programmatically and capture intermediate grid selections, generated scripts, and terminal logs. Added `scripts/run_experiments_and_upload.sh` bash wrapper. Added `data/raw/README_RAG_EXPERIMENTS.md` documenting experiment structure and artifact locations. Updated to work with GitHub Actions environment secrets.

### Changed
- **ProblemStructure schema**: Now follows exact solution format with `search_space` (list of parameters), `objective` (list of objectives), `budget`, `batch_size`, `noise_model`, `constraints`, `historical_data_points`, and `model_preference` - matching test problem structure.
- **Constraint distinction**: Clear separation between `composition_constraint` (fractions sum to 1.0 for materials) and `sum_constraint` (general sum constraints) with explicit total value tracking via `total` field instead of `target_value`.
- **Parameter types**: Simplified to `continuous` and `categorical` (removed `integer`) to match solution format expectations.
- **ParameterExtractor prompt**: Includes formatted table showing grid selection logic with pre-computed boolean flags for each constraint type based on extracted structure, making the mapping explicit and deterministic.
- **Parameter selector workflow**: Two-stage extraction (structure → grid) with solution-format intermediate representation instead of direct natural language → grid mapping.
### Fixed
- Removed API key existence checks from experiment runner scripts - secrets are available in GitHub Actions runtime but not in Copilot agent sandbox.

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
