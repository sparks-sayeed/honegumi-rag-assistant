# Code Validation Workflow

This directory contains a GitHub Actions workflow that automatically validates Python files in `data/processed/evaluation/code_files/`.

## What It Does

The workflow runs three checks on each `.py` file:

1. **Syntax Check**: Validates Python syntax using `ast.parse()`
2. **Import Check**: Attempts to import the module to verify all dependencies are available
3. **Runtime Check**: Executes the file with a 30-second timeout

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

```bash
# Install dependencies
pip install -r requirements.txt
pip install ax-platform==0.4.3 matplotlib pyyaml

# Run validation
python scripts/validate_code_files.py
```

The validation report will be generated at `data/processed/evaluation/validation_report.yaml`.
