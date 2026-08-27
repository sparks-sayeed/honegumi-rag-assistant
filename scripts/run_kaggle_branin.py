#!/usr/bin/env python3
"""Run the Acceleration Consortium Branin Kaggle competition end-to-end.

This driver actually imports and calls the real competition package
(`amanichabouni/branin-package`) via kagglehub, runs the competition-legal
12 campaigns x 40 evaluations against the hidden black-box objective, and
exports a `submission.csv` ready for `scripts/submit_to_kaggle.py`.

The package executes downloaded code, so confirmation is required. Pass
``--yes`` (or set ``KAGGLEHUB_ALLOW_UNTRUSTED=1``) to bypass the interactive
prompt in non-interactive environments such as CI.

A Sobol/quasi-random sampler is used as a simple, sample-efficient baseline.
Drop in the RAG-assistant-generated Ax code to replace ``suggest`` with a
Bayesian optimization strategy.

Usage:
    python scripts/run_kaggle_branin.py --yes
    python scripts/run_kaggle_branin.py --yes --method predict_noisy
"""

import argparse
import os

import numpy as np

PACKAGE_HANDLE = "amanichabouni/branin-package/versions/17"
BOUNDS = {"x1": (-5.0, 10.0), "x2": (0.0, 15.0)}


def suggest(rng):
    """Suggest the next point (uniform baseline; replace with BO)."""
    x1 = rng.uniform(*BOUNDS["x1"])
    x2 = rng.uniform(*BOUNDS["x2"])
    return float(x1), float(x2)


def evaluate(campaign, method, x1, x2):
    """Call the requested black-box method on a campaign."""
    fn = getattr(campaign, method)
    if method == "predict_noisy":
        return fn(x1, x2, predictions=[])
    return fn(x1, x2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
    args = parser.parse_args()

    import kagglehub

    bypass = args.yes or os.environ.get("KAGGLEHUB_ALLOW_UNTRUSTED") == "1"
    package = kagglehub.package_import(PACKAGE_HANDLE, bypass_confirmation=bypass)

    rng = np.random.default_rng(args.seed)
    for _ in range(args.campaigns):
        campaign = package.Model()
        best = float("inf")
        for _ in range(args.budget):
            x1, x2 = suggest(rng)
            value = evaluate(campaign, args.method, x1, x2)
            best = min(best, float(value))
        print(f"  best value this campaign: {best:.6f}")

    package.Model.export_history(args.output)
    print(f"Exported submission to {args.output}")


if __name__ == "__main__":
    main()
