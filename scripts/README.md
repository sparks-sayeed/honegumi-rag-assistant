# Scripts

This directory contains utility scripts for the Honegumi RAG Assistant.

## Available Scripts

### `batch_process.py`

Batch processing script for running the agentic pipeline on multiple problems from a CSV file.

**Purpose:**
- Process multiple optimization problems at once
- Collect comprehensive statistics on agentic behavior
- Generate results CSV with all parameters, decisions, and code

**Usage:**
```powershell
python scripts/batch_process.py --input problems.csv --output results.csv
```

**Arguments:**
- `--input`: Path to input CSV file (required)
- `--output`: Path to output CSV file (required)
- `--output-dir`: Directory for generated Python scripts (optional)
- `--problem-column`: Name of column with problems (default: "problems")

**Example:**
```powershell
python scripts/batch_process.py `
    --input data/raw/example_batch_problems.csv `
    --output results/batch_results.csv `
    --output-dir results/generated_codes
```

**See Also:** `BATCH_PROCESSING.md` for comprehensive documentation

### Future Scripts

This directory can contain additional utility scripts such as:
- `evaluate_results.py` - Analyze batch processing results
- `create_dataset.py` - Generate synthetic problem datasets
- `benchmark.py` - Benchmark the agentic system performance
- `visualize.py` - Create visualizations of agentic decision patterns

## Adding New Scripts

When adding new scripts:
1. Add clear docstring at the top explaining purpose
2. Use argparse for CLI arguments
3. Add entry in this README
4. Create separate documentation file if complex
5. Include example usage
