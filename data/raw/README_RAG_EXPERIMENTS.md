# RAG Assistant Experiment Results

This directory contains the results of running the Honegumi RAG Assistant on different problem statement versions.

## Structure

- `problem_statements.yaml` - Original problem statements with natural, corrected, and solution versions
- `rag_assistant_runs.yaml` - Experiment configuration and results

## Experiment IDs

Each experiment has a unique ID that correlates logs with results:

- `exp_001_natural` - Natural version (underspecified)
- `exp_002_corrected` - Corrected version (conversational with details)
- `exp_003_solution` - Solution-based version (explicit specifications)

## Running Experiments

To run the experiments:

```bash
# Ensure OPENAI_API_KEY is set
export OPENAI_API_KEY="your-key-here"

# Run the experiment script
python scripts/run_rag_experiments.py
```

## Artifacts

Experiment artifacts are stored in `/tmp/rag_experiments/` (gitignored):

- `logs/` - Terminal output logs for each experiment
- `scripts/` - Generated Python scripts from the RAG assistant

## Results Format

Each experiment in `rag_assistant_runs.yaml` contains:

- `experiment_id` - Unique identifier
- `problem_statement_id` - Links to problem_statements.yaml
- `version` - Which version of the prompt (natural/corrected/solution)
- `prompt` - The actual text sent to the RAG assistant
- `timestamp` - When the experiment was run
- `status` - Success/failure status
- `grid_selections` - Intermediate Honegumi grid selections (if captured)
- `generated_script_path` - Path to the generated script
- `log_path` - Path to the full terminal log
- `log_artifact_url` - GitHub Actions artifact URL (added post-run)

## Correlation

To find logs for a specific result:
1. Find the experiment_id in `rag_assistant_runs.yaml`
2. Check the `log_path` field for the log file location
3. Check the `generated_script_path` for the generated code
4. Use `log_artifact_url` to view in GitHub Actions artifacts
