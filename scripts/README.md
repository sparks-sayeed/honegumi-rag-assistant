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

### `submit_to_kaggle.py`

Auto-submit an optimization result to an Acceleration Consortium Kaggle
competition (see `data/raw/kaggle_competitions.yaml`). These competitions are
non-hackable, black-box benchmarks: the objective is only reachable through a
provided kagglehub package that writes a `submission.csv`, which this script
uploads to the competition leaderboard.

**Prerequisites:**
- `pip install kaggle`
- Kaggle credentials. Either the modern access token (`KAGGLE_API_TOKEN`) or the
  legacy pair (`KAGGLE_USERNAME` + `KAGGLE_KEY`). `KAGGLE_API_KEY` is also
  accepted and routed automatically (treated as a legacy key if 32-hex,
  otherwise as an access token). Credentials in `~/.kaggle/kaggle.json` also work.
  Get an API token at https://www.kaggle.com/settings.
- You must accept the competition's rules once on its Kaggle web page (e.g.
  `https://www.kaggle.com/competitions/<slug>/rules`) before the API will accept
  a submission; there is no API endpoint to accept rules.

**Usage:**
```bash
# By competition slug
python scripts/submit_to_kaggle.py \
    --competition noisy-vanilla-optimization-2-d-branin-function \
    --file submission.csv \
    --message "Honegumi RAG Assistant run"

# Or by config id from data/raw/kaggle_competitions.yaml
python scripts/submit_to_kaggle.py --competition-id branin_noisy_2d --file submission.csv
```

**Arguments:**
- `--competition`: Full Kaggle competition slug
- `--competition-id`: Competition id from `kaggle_competitions.yaml` (alternative to `--competition`)
- `--file`: Path to the submission CSV (default: `submission.csv`)
- `--message`: Submission message

### `run_kaggle_branin.py`

Runs the Acceleration Consortium Branin competition end-to-end by actually
importing and calling the real competition package
(`amanichabouni/branin-package`) via kagglehub. It runs the competition-legal
12 campaigns x 40 evaluations against the hidden black-box objective and exports
a `submission.csv`. A uniform-random sampler is used as a baseline; drop in the
RAG-assistant-generated Ax code to replace `suggest` with Bayesian optimization.

**Prerequisites:** `pip install kagglehub`

**Usage:**
```bash
# Deterministic (vanilla) objective
python scripts/run_kaggle_branin.py --yes

# Noisy objective variant
python scripts/run_kaggle_branin.py --yes --method predict_noisy
```

**Note:** The package executes downloaded code, so `--yes` (or
`KAGGLEHUB_ALLOW_UNTRUSTED=1`) is required to bypass the interactive
confirmation prompt in non-interactive environments.

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
