# Code Validation Workflow

This directory contains GitHub Actions workflows that automatically validate Python files in `data/processed/evaluation/code_files/`.

## Workflows

### Standard Workflow (`validate-code-files.yml`)
Sequential validation of all files in a single job. Suitable for small numbers of files (< 10).

### Parallel Workflow (`validate-code-files-parallel.yml`)
**Recommended for 10+ files.** Uses GitHub Actions matrix strategy to validate files in parallel:
- **list-files**: Discovers all Python files to validate
- **validate**: Runs validation in parallel (one job per file)
- **combine-results**: Aggregates all results into final report

## What It Does

The workflow runs three checks on each `.py` file:

1. **Syntax Check**: Validates Python syntax using `ast.parse()`
2. **Import Check**: Attempts to import the module to verify all dependencies are available
3. **Runtime Check**: Executes the file with a 5-minute timeout (timeouts are treated as success for long-running optimization scripts)

## Workflow Triggers

The workflow runs automatically on:
- Push to `data/processed/evaluation/code_files/`
- Pull requests modifying code files or the validation script
- Manual trigger via workflow_dispatch

## Output

Results are saved to:
- **Artifact**: `validation-report` (available in Actions tab)
- **File**: `data/processed/evaluation/validation_report.yaml` (committed to repo)

## Report Format

```yaml
validation_timestamp: 2025-10-31 04:02:44 UTC
summary:
  total_files: 2
  passed_files: 2
  failed_files: 0
results:
  - file: data/processed/evaluation/code_files/example.py
    syntax_check:
      passed: true
      error: null
    import_check:
      passed: true
      error: null
    runtime_check:
      passed: true
      error: null
      execution_time: "2.35s"
    overall_passed: true
```

## Running Locally

### Standard Mode (all files)
```bash
# Install dependencies
pip install -r requirements.txt
pip install -r requirements-validation.txt

# Run validation
python scripts/validate_code_files.py
```

### Single File Mode (for testing)
```bash
python scripts/validate_code_files.py --file filename.py --output-dir ./results
```

### List Files Mode
```bash
python scripts/validate_code_files.py --list-files
```

### Combine Partial Results
```bash
python scripts/validate_code_files.py --combine ./partial-results --output-dir ./final
```

The validation report will be generated at `data/processed/evaluation/validation_report.yaml`.
