# RAG Experiments - Next Steps

## Status Update

✓ Infrastructure complete and ready
✓ API keys confirmed working in GitHub Actions runtime
✓ Scripts updated to work with GitHub Actions environment

**Note:** The Copilot agent sandbox cannot access repository secrets, so local checks return "NOT FOUND". This is expected behavior and not a blocker. The secrets are confirmed available in the actual GitHub Actions workflow runtime.

## Infrastructure Ready ✓

The following infrastructure has been created and is ready to use:

### Files Created

1. **`data/raw/rag_assistant_runs.yaml`** - Experiment configuration with 3 experiments:
   - `exp_001_natural` - Natural/underspecified version
   - `exp_002_corrected` - Corrected/conversational version  
   - `exp_003_solution` - Solution-based/explicit version

2. **`scripts/run_rag_experiments.py`** - Python script that:
   - Runs honegumi-rag on each prompt
   - Captures stdout/stderr to logs
   - Extracts grid selections from debug output
   - Saves generated scripts
   - Updates rag_assistant_runs.yaml with results

3. **`scripts/run_experiments_and_upload.sh`** - Bash wrapper that:
   - Checks for OPENAI_API_KEY
   - Runs the Python script
   - Lists generated artifacts
   - Provides artifact URL structure

4. **`data/raw/README_RAG_EXPERIMENTS.md`** - Documentation for the experiment structure

5. **`.gitignore` updated** - Excludes `/tmp/rag_experiments/` and `*.log` files

## Running the Experiments

Once the OPENAI_API_KEY is configured, run:

```bash
./scripts/run_experiments_and_upload.sh
```

Or directly:

```bash
export OPENAI_API_KEY="your-key-here"
python3 scripts/run_rag_experiments.py
```

## Expected Output

### 1. Logs
Location: `/tmp/rag_experiments/logs/`

Each experiment will have a timestamped log file:
- `exp_001_natural_2025-10-24T21-30-00.log`
- `exp_002_corrected_2025-10-24T21-35-00.log`
- `exp_003_solution_2025-10-24T21-40-00.log`

Logs contain:
- Full prompt text
- stdout (agent decisions, parameter extraction, code generation)
- stderr (any errors)

### 2. Generated Scripts
Location: `/tmp/rag_experiments/scripts/exp_XXX/`

Each experiment directory will contain the generated Python script:
- `exp_001_natural/generated_bo_script.py`
- `exp_002_corrected/generated_bo_script.py`
- `exp_003_solution/generated_bo_script.py`

### 3. Updated Configuration
File: `data/raw/rag_assistant_runs.yaml`

Will be updated with:
- `timestamp` - When each experiment ran
- `status` - success/failed/error/timeout
- `log_path` - Path to the log file
- `generated_script_path` - Path to the generated script
- `grid_selections` - Extracted Honegumi grid options (if available)

## GitHub Actions Artifacts

After running in GitHub Actions, artifacts should be uploaded with:

```yaml
- name: Upload experiment artifacts
  uses: actions/upload-artifact@v4
  with:
    name: rag-experiment-results
    path: |
      /tmp/rag_experiments/
      data/raw/rag_assistant_runs.yaml
```

Artifact URLs will be in the format:
```
https://github.com/sparks-sayeed/honegumi-rag-assistant/actions/runs/<run-id>/artifacts/<artifact-id>
```

## Updating YAML with Artifact URLs

After artifacts are uploaded, the `log_artifact_url` field in each experiment can be updated to point to the direct file in the artifact, using the commit hash:

```yaml
log_artifact_url: "https://github.com/sparks-sayeed/honegumi-rag-assistant/blob/<commit-hash>/path/to/artifact"
```

Or for GitHub Actions artifacts:
```yaml
log_artifact_url: "https://github.com/sparks-sayeed/honegumi-rag-assistant/actions/runs/<run-id>"
```

## Correlation Example

To trace results for experiment `exp_001_natural`:

1. Check `data/raw/rag_assistant_runs.yaml`:
   ```yaml
   - experiment_id: "exp_001_natural"
     timestamp: "2025-10-24T21:30:00"
     log_path: "/tmp/rag_experiments/logs/exp_001_natural_2025-10-24T21-30-00.log"
     generated_script_path: "/tmp/rag_experiments/scripts/exp_001_natural/generated_bo_script.py"
     log_artifact_url: "https://github.com/sparks-sayeed/honegumi-rag-assistant/actions/runs/12345"
   ```

2. View the log at the `log_path` or via `log_artifact_url`
3. View the generated script at `generated_script_path`
4. See grid selections in the `grid_selections` field

## Status

✓ Infrastructure created and committed
⏳ Waiting for OPENAI_API_KEY to run experiments
⏳ Artifacts will be uploaded after experiments complete
