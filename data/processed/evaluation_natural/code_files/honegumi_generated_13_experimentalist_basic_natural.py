# Generated benchmark: Compare BO acquisition functions on Branin with Ax
# %pip install ax-platform==0.4.3 matplotlib

import math
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient, ObjectiveProperties

from botorch.acquisition.monte_carlo import (
    qExpectedImprovement,
    qNoisyExpectedImprovement,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP


# ----------------------------
# Branin function and evaluator
# ----------------------------

def branin_function(branin_x: float, branin_y: float) -> float:
    """Deterministic Branin function value at (branin_x, branin_y).

    Domain:
      - branin_x in [-5, 10]
      - branin_y in [0, 15]
    Global minima approx value: 0.397887
    """
    a = 1.0
    b = 5.1 / (4.0 * math.pi ** 2)
    c = 5.0 / math.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * math.pi)
    return a * (branin_y - b * branin_x ** 2 + c * branin_x - r) ** 2 + s * (1 - t) * math.cos(branin_x) + s


def evaluate_branin_with_noise(
    parameters: Dict[str, float],
    rng: np.random.Generator,
    noise_std: float,
) -> float:
    """Return a noisy observation of the Branin function.

    Returns a single float so Ax will treat noise as unknown and infer it.
    """
    x = float(parameters["branin_x"])
    y = float(parameters["branin_y"])
    y_true = branin_function(x, y)
    if noise_std > 0.0:
        y_obs = float(y_true + rng.normal(0.0, noise_std))
    else:
        y_obs = float(y_true)
    return y_obs


# ----------------------------
# Generation strategy per acquisition function
# ----------------------------

def make_generation_strategy(
    acqf_key: str,
    seed: int,
    n_init: int,
    n_bo: int,
) -> GenerationStrategy:
    """Create a GenerationStrategy using Sobol init and a specified BoTorch acquisition function."""
    acqf_registry = {
        # Noisy methods
        "qNEI": (qNoisyExpectedImprovement, {}),
        "qLogNEI": (qLogNoisyExpectedImprovement, {}),
        # Analytic-style objectives approximated via MC (noiseless assumption)
        "qEI": (qExpectedImprovement, {}),
        "qPI": (qProbabilityOfImprovement, {}),
        "qUCB": (qUpperConfidenceBound, {"beta": 0.2}),  # exploration parameter
    }
    if acqf_key not in acqf_registry:
        raise ValueError(f"Unknown acquisition key: {acqf_key}")

    botorch_acqf_class, acquisition_options = acqf_registry[acqf_key]

    gs = GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=n_init,
                min_trials_observed=n_init,
                max_parallelism=1,
                model_kwargs={"seed": seed},
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=n_bo,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": botorch_acqf_class,
                    "acquisition_options": acquisition_options,
                },
            ),
        ]
    )
    return gs


# ----------------------------
# Run a single optimization
# ----------------------------

def run_single_benchmark(
    acqf_key: str,
    seed: int,
    n_init: int,
    n_bo: int,
    noise_std: float,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Run one Ax optimization under a given acquisition function and seed."""
    gs = make_generation_strategy(acqf_key=acqf_key, seed=seed, n_init=n_init, n_bo=n_bo)
    ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=True)

    ax_client.create_experiment(
        name=f"branin_{acqf_key}_seed{seed}",
        parameters=[
            {"name": "branin_x", "type": "range", "bounds": [-5.0, 10.0]},
            {"name": "branin_y", "type": "range", "bounds": [0.0, 15.0]},
        ],
        objectives={"branin_value": ObjectiveProperties(minimize=True)},
    )

    rng = np.random.default_rng(seed)
    observations: List[float] = []

    total_trials = n_init + n_bo
    for _ in range(total_trials):
        params, trial_index = ax_client.get_next_trial()
        y_obs = evaluate_branin_with_noise(parameters=params, rng=rng, noise_std=noise_std)
        # Single-objective experiment allows passing a float directly.
        ax_client.complete_trial(trial_index=trial_index, raw_data=y_obs)
        observations.append(y_obs)

    # Compile per-trial traces
    trial_indices = np.arange(1, total_trials + 1)
    best_so_far = np.minimum.accumulate(observations)
    df = pd.DataFrame(
        {
            "trial_index": trial_indices,
            "observed_branin_value": observations,
            "best_branin_value_so_far": best_so_far,
            "acquisition": acqf_key,
            "seed": seed,
        }
    )

    # Extract best parameters and value
    best_params, metrics = ax_client.get_best_parameters()
    best_metrics = {k: float(v) if np.isscalar(v) else float(v[0]) for k, v in metrics.items()}
    # add parameters into a flat dict
    best_result = {
        "acquisition": acqf_key,
        "seed": seed,
        "best_branin_value": best_metrics.get("branin_value", float(best_so_far[-1])),
        "best_branin_x": float(best_params["branin_x"]),
        "best_branin_y": float(best_params["branin_y"]),
    }

    return df, best_result


# ----------------------------
# Main benchmark runner
# ----------------------------

def run_benchmark_suite(
    acquisitions: List[str],
    seeds: List[int],
    n_init: int,
    n_bo: int,
    noise_std: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run all acquisition methods across seeds and collect results."""
    all_traces: List[pd.DataFrame] = []
    best_rows: List[Dict[str, float]] = []

    for acqf in acquisitions:
        for s in seeds:
            trace_df, best_dict = run_single_benchmark(
                acqf_key=acqf, seed=s, n_init=n_init, n_bo=n_bo, noise_std=noise_std
            )
            all_traces.append(trace_df)
            best_rows.append(best_dict)

    traces = pd.concat(all_traces, axis=0, ignore_index=True)
    best_df = pd.DataFrame(best_rows)
    return traces, best_df


def plot_aggregate_convergence(traces: pd.DataFrame) -> None:
    """Plot mean best-so-far curves with +/- 1 std over seeds for each acquisition."""
    plt.figure(figsize=(7.5, 5.0), dpi=150)
    for acqf, dfg in traces.groupby("acquisition"):
        # Compute mean and std across seeds at each trial index
        pivot = dfg.pivot_table(
            index="trial_index", columns="seed", values="best_branin_value_so_far", aggfunc="first"
        ).sort_index()
        mean_curve = pivot.mean(axis=1).values
        std_curve = pivot.std(axis=1, ddof=0).values
        x = pivot.index.values
        plt.plot(x, mean_curve, lw=2, label=acqf)
        plt.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)

    # Global optimum reference line
    branin_global_min = 0.397887
    plt.axhline(branin_global_min, ls="--", c="k", lw=1.5, label="Branin global optimum")

    plt.xlabel("Trial number")
    plt.ylabel("Best objective value (min so far)")
    plt.title("Branin: Convergence across acquisition functions")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    warnings.filterwarnings("ignore")

    # Configuration
    ACQUISITIONS = ["qNEI", "qLogNEI", "qEI", "qPI", "qUCB"]
    SEEDS = [0, 1, 2]
    N_INIT = 5      # Sobol initial trials
    N_BO = 20       # Bayesian optimization trials
    NOISE_STD = 0.1 # observation noise std; set to 0.0 for noiseless

    traces, best_df = run_benchmark_suite(
        acquisitions=ACQUISITIONS,
        seeds=SEEDS,
        n_init=N_INIT,
        n_bo=N_BO,
        noise_std=NOISE_STD,
    )

    # Print summary of final best values averaged across seeds
    summary = (
        best_df.groupby("acquisition")["best_branin_value"]
        .agg(["mean", "std", "min", "max", "count"])
        .sort_values("mean")
    )
    print("\nFinal best Branin values across seeds by acquisition:\n")
    print(summary.to_string(float_format=lambda v: f"{v:0.6f}"))

    # Plot aggregate convergence
    plot_aggregate_convergence(traces)


if __name__ == "__main__":
    main()