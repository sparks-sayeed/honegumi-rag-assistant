#!/usr/bin/env python3
"""Run the Acceleration Consortium Branin Kaggle competition end-to-end.

This driver actually imports and calls the real competition package
(`amanichabouni/branin-package`) via kagglehub, runs the competition-legal
12 campaigns x 40 evaluations against the hidden black-box objective, and
exports a `submission.csv` ready for `scripts/submit_to_kaggle.py`.

The package executes downloaded code, so confirmation is required. Pass
``--yes`` (or set ``KAGGLEHUB_ALLOW_UNTRUSTED=1``) to bypass the interactive
prompt in non-interactive environments such as CI.

Two strategies are available via ``--strategy``:

* ``bo`` (default) — sample-efficient Bayesian optimization with Ax
  (``AxClient``), the same library the Honegumi RAG assistant generates code
  for. A fresh model is fit per campaign; Ax bootstraps with Sobol and then
  switches to a Gaussian-process surrogate that balances exploration and
  exploitation.
* ``random`` — a uniform-random sampler kept as a baseline for comparison.

The ``bo`` strategy follows this repository's Honegumi RAG framework. Running
the framework "manually" (the language model is the operator here): the
competition ``prompt`` in ``data/raw/kaggle_competitions.yaml`` maps to the
``expected_grid_selections`` recorded there (the parameter-selection step),
which this repo's ``SkeletonGenerator`` node feeds to the Honegumi package to
emit a reference Ax script (see ``--print-skeleton``). The BO loop below is the
code-writer adaptation of that skeleton: the analytical Branin is swapped for
the black-box package call and wrapped in the 12x40 campaign loop, with the
search space widened to the competition bounds (x2 in [0, 15]).

Usage:
    python scripts/run_kaggle_branin.py --yes
    python scripts/run_kaggle_branin.py --yes --strategy random
    python scripts/run_kaggle_branin.py --yes --method predict_noisy
    python scripts/run_kaggle_branin.py --print-skeleton  # no Kaggle needed
"""

import argparse
import os
from pathlib import Path

import numpy as np

PACKAGE_HANDLE = "amanichabouni/branin-package/versions/17"
BOUNDS = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}
COMPETITIONS_YAML = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "kaggle_competitions.yaml"
)


def honegumi_reference_skeleton(competition_id="branin_vanilla_2d"):
    """Generate the Honegumi reference Ax skeleton via this repo's RAG node.

    Acts out the framework's parameter-selection step manually: the competition
    ``expected_grid_selections`` (derived by the LLM operator from the prompt)
    are handed to the repository ``SkeletonGenerator`` node, which renders the
    reference Ax script with the Honegumi package.
    """
    import sys

    import yaml

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from honegumi_rag_assistant.nodes.skeleton_generator import SkeletonGenerator

    config = yaml.safe_load(COMPETITIONS_YAML.read_text())
    comp = next(
        c
        for c in config["kaggle_competitions"]
        if c["id"] == competition_id
    )
    grid = comp["expected_grid_selections"]
    bo_params = {
        "objective": grid["objective"],
        "model": grid["model"],
        "task": grid["task"],
        "categorical": grid["categorical"],
        "custom_gen": False,
        "sum_constraint": grid["sum_constraint"],
        "order_constraint": grid["order_constraint"],
        "linear_constraint": grid["linear_constraint"],
        "composition_constraint": grid["composition_constraint"],
        "custom_threshold": False,
        "existing_data": False,
        "synchrony": "Single",
        "visualize": False,
    }
    result = SkeletonGenerator.generate_skeleton({"bo_params": bo_params})
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result["skeleton_code"]


def suggest(rng):
    """Suggest the next point with a uniform-random baseline."""
    x1 = rng.uniform(*BOUNDS["x1"])
    x2 = rng.uniform(*BOUNDS["x2"])
    return float(x1), float(x2)


def make_ax_client(seed):
    """Create an Ax client set up to minimize the black-box objective."""
    from ax.service.ax_client import AxClient, ObjectiveProperties

    ax_client = AxClient(random_seed=seed, verbose_logging=False)
    ax_client.create_experiment(
        name="branin_vanilla_2d",
        parameters=[
            {"name": "x1", "type": "range", "bounds": list(BOUNDS["x1"])},
            {"name": "x2", "type": "range", "bounds": list(BOUNDS["x2"])},
        ],
        objectives={"value": ObjectiveProperties(minimize=True)},
    )
    return ax_client


def evaluate(campaign, method, x1, x2):
    """Call the requested black-box method on a campaign."""
    fn = getattr(campaign, method)
    if method == "predict_noisy":
        return fn(x1, x2, predictions=[])
    return fn(x1, x2)


def run_campaign_random(campaign, method, budget, rng):
    """Optimize a single campaign with the uniform-random baseline."""
    best = float("inf")
    for _ in range(budget):
        x1, x2 = suggest(rng)
        value = evaluate(campaign, method, x1, x2)
        best = min(best, float(value))
    return best


def run_campaign_bo(campaign, method, budget, seed):
    """Optimize a single campaign with Ax Bayesian optimization."""
    ax_client = make_ax_client(seed)
    best = float("inf")
    for _ in range(budget):
        params, trial_index = ax_client.get_next_trial()
        value = float(evaluate(campaign, method, params["x1"], params["x2"]))
        ax_client.complete_trial(trial_index, raw_data={"value": value})
        best = min(best, value)
    return best


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        default="bo",
        choices=["bo", "random"],
        help="Optimization strategy (default: bo).",
    )
    parser.add_argument(
        "--method",
        default="predict_single",
        choices=["predict_single", "predict_noisy"],
        help="Black-box method to call (default: predict_single).",
    )
    parser.add_argument(
        "--campaigns", type=int, default=12, help="Number of campaigns (max 12)."
    )
    parser.add_argument(
        "--budget", type=int, default=40, help="Evaluations per campaign (max 40)."
    )
    parser.add_argument(
        "--output", default="submission.csv", help="Path to export submission CSV."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Bypass the kagglehub untrusted-code confirmation prompt.",
    )
    parser.add_argument(
        "--print-skeleton",
        action="store_true",
        help=(
            "Print the Honegumi reference Ax skeleton for this benchmark "
            "(via this repo's RAG SkeletonGenerator node) and exit."
        ),
    )
    args = parser.parse_args()

    if args.print_skeleton:
        print(honegumi_reference_skeleton())
        return

    import kagglehub

    bypass = args.yes or os.environ.get("KAGGLEHUB_ALLOW_UNTRUSTED") == "1"
    package = kagglehub.package_import(PACKAGE_HANDLE, bypass_confirmation=bypass)

    rng = np.random.default_rng(args.seed)
    for campaign_index in range(args.campaigns):
        campaign = package.Model()
        if args.strategy == "bo":
            best = run_campaign_bo(
                campaign, args.method, args.budget, args.seed + campaign_index
            )
        else:
            best = run_campaign_random(campaign, args.method, args.budget, rng)
        print(f"  best value this campaign: {best:.6f}")

    package.Model.export_history(args.output)
    print(f"Exported submission to {args.output}")


if __name__ == "__main__":
    main()
