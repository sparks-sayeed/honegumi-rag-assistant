#!/usr/bin/env python3
"""Evaluate Honegumi RAG Assistant outputs with the kaggle-benchmarks framework.

This is a proof-of-concept for the *code-correctness* evaluation layer discussed
in PR #17. It wraps the grid-selection check that `scripts/run_rag_experiments.py`
already performs by hand (comparing the RAG assistant's Honegumi grid choices
against the ground-truth `expected_grid_selections`) as a reusable
``kaggle_benchmarks`` task, so the whole problem set can be scored with aggregate
metrics and, on Kaggle, a leaderboard.

Two correctness dimensions are tracked per run:

* grid-selection accuracy - fraction of the 8 Honegumi grid dimensions that match
  the ground truth (returned as the task score);
* generated code runs without error - only checked when the generated script from
  ``run_rag_experiments.py`` is present on disk (recorded as an assertion).

Credentials: this offline evaluation reuses results already produced by
``run_rag_experiments.py`` and needs no extra credentials. Driving generation with
a live model (``kbench.llm.prompt(...)``) or publishing a Kaggle leaderboard
additionally requires Kaggle Model Proxy access or running inside a Kaggle
notebook (https://www.kaggle.com/benchmarks/tasks/new).

Install (kaggle-benchmarks is not on PyPI) and run from the repository root::

    pip install "kaggle_benchmarks @ git+https://github.com/Kaggle/kaggle-benchmarks.git"
    python scripts/kbench_honegumi_eval.py
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

import kaggle_benchmarks as kbench

# Honegumi grid dimensions shared by the expected and actual selections.
GRID_DIMENSIONS = [
    "objective",
    "model",
    "task",
    "categorical",
    "sum_constraint",
    "order_constraint",
    "linear_constraint",
    "composition_constraint",
]

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

problems = yaml.safe_load((DATA_DIR / "problem_statements.yaml").read_text())
expected_by_id = {
    problem["id"]: problem["expected_grid_selections"]
    for problem in problems["problem_statements"]
}

runs = yaml.safe_load((DATA_DIR / "rag_assistant_runs.yaml").read_text())
rows = []
for experiment in runs["experiments"]:
    actual = experiment.get("actual_grid_selections")
    expected = expected_by_id.get(experiment["problem_statement_id"])
    if not actual or not expected:
        continue
    rows.append(
        {
            "experiment_id": experiment["experiment_id"],
            "expected": {dim: expected.get(dim) for dim in GRID_DIMENSIONS},
            "actual": {dim: actual.get(dim) for dim in GRID_DIMENSIONS},
            "script_path": experiment.get("generated_script_path"),
        }
    )

evaluation_data = pd.DataFrame(rows)


@kbench.benchmark(
    name="honegumi_grid_selection",
    description="Honegumi RAG grid-selection accuracy vs. ground truth.",
)
def honegumi_grid_selection(llm, experiment_id, expected, actual, script_path) -> float:
    """Score one RAG run and, when available, check its generated code runs."""
    for dim in GRID_DIMENSIONS:
        kbench.assertions.assert_equal(
            expected[dim],
            actual[dim],
            expectation=f"{experiment_id}: {dim}",
        )

    script = Path(script_path) if script_path else None
    if script and script.exists():
        kbench.assertions.assert_raises_no_exceptions(
            lambda: subprocess.run(
                [sys.executable, str(script)],
                capture_output=True,
                check=True,
                timeout=600,
            ),
            expectation=f"{experiment_id}: generated code runs without error",
        )

    correct = sum(expected[dim] == actual[dim] for dim in GRID_DIMENSIONS)
    return correct / len(GRID_DIMENSIONS)


results = honegumi_grid_selection.evaluate(
    llm=[kbench.llm], evaluation_data=evaluation_data
)

summary = results.as_dataframe()
completed = results.completed_runs.as_dataframe()
print(summary[["experiment_id", "result"]].to_string(index=False))
print(f"\nRuns evaluated: {len(summary)} ({len(completed)} completed)")
print(f"Mean grid-selection accuracy: {completed['result'].mean():.4f}")
