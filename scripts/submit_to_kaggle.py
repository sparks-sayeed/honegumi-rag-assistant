#!/usr/bin/env python3
"""Submit an optimization result to an Acceleration Consortium Kaggle competition.

The Kaggle optimization competitions (see ``data/raw/kaggle_competitions.yaml``)
are black-box benchmarks: the objective is only reachable through a provided
kagglehub package that writes a ``submission.csv``. This helper auto-submits that
CSV to the competition leaderboard via the Kaggle API.

Credentials are read from the environment (``KAGGLE_USERNAME`` and ``KAGGLE_KEY``)
or from ``~/.kaggle/kaggle.json``. See https://www.kaggle.com/docs/api.

Usage:
    python scripts/submit_to_kaggle.py \
        --competition noisy-vanilla-optimization-2-d-branin-function \
        --file submission.csv \
        --message "Honegumi RAG Assistant run"

You can also pass a competition ``id`` from kaggle_competitions.yaml via
``--competition-id`` instead of the full slug.
"""

import argparse
import sys
from pathlib import Path

import yaml

KAGGLE_COMPETITIONS = Path("data/raw/kaggle_competitions.yaml")


def resolve_slug(competition, competition_id):
    """Resolve a competition slug from an explicit slug or a config id."""
    if competition:
        return competition
    if not competition_id:
        raise ValueError("Provide either --competition (slug) or --competition-id.")
    if not KAGGLE_COMPETITIONS.exists():
        raise FileNotFoundError(f"Config not found: {KAGGLE_COMPETITIONS}")
    with open(KAGGLE_COMPETITIONS, "r") as f:
        config = yaml.safe_load(f)
    for comp in config.get("kaggle_competitions", []):
        if comp.get("id") == competition_id:
            return comp["slug"]
    raise ValueError(
        f"No competition with id '{competition_id}' in {KAGGLE_COMPETITIONS}"
    )


def submit(slug, file_path, message):
    """Submit a CSV file to a Kaggle competition."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:
        raise SystemExit(
            "The 'kaggle' package is required. Install it with: pip install kaggle"
        ) from exc

    api = KaggleApi()
    api.authenticate()
    return api.competition_submit(
        file_name=file_path, message=message, competition=slug
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition", help="Full Kaggle competition slug.")
    parser.add_argument(
        "--competition-id",
        help=(
            "Competition id from data/raw/kaggle_competitions.yaml "
            "(e.g. branin_noisy_2d)."
        ),
    )
    parser.add_argument(
        "--file",
        default="submission.csv",
        help="Path to submission.csv (default: submission.csv).",
    )
    parser.add_argument(
        "--message",
        default="Honegumi RAG Assistant submission",
        help="Submission message.",
    )
    args = parser.parse_args()

    slug = resolve_slug(args.competition, args.competition_id)

    file_path = Path(args.file)
    if not file_path.exists():
        raise SystemExit(f"Submission file not found: {file_path}")

    print(f"Submitting {file_path} to '{slug}'...")
    result = submit(slug, str(file_path), args.message)
    print(result)


if __name__ == "__main__":
    sys.exit(main())
