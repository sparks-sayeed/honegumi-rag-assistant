import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.service.ax_client import AxClient, ObjectiveProperties

from botorch.models import SingleTaskGP
from botorch.acquisition import (
    qExpectedImprovement,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
)


# ----------------------------------------
# Branin benchmark proxy objective (minimize)
# Domain: branin_x1 in [-5, 10], branin_x2 in [0, 15]
# ----------------------------------------
def compute_branin_value(branin_x1: float, branin_x2: float) -> float:
    x1 = float(branin_x1)
    x2 = float(branin_x2)
    term1 = x2 - (5.1 / (4.0 * np.pi**2)) * x1**2 + (5.0 / np.pi) * x1 - 6.0
    term2 = 10.0 * (1.0 - 1.0 / (8.0 * np.pi)) * np.cos(x1)
    y = term1**2 + term2 + 10.0
    return float(y)


# ----------------------------------------
# Generation strategy builders per acquisition
# ----------------------------------------
def build_generation_strategy_for_benchmark(
    acquisition_id: str,
    init_trials: int,
    sobol_seed: int,
    ucb_beta: float = 2.0,
) -> GenerationStrategy:
    """
    acquisition_id in {"GPEI", "UCB", "PI", "SOBOL"}
    """
    if acquisition_id == "SOBOL":
        # Pure Sobol baseline
        return GenerationStrategy(
            steps=[
                GenerationStep(
                    model=Models.SOBOL,
                    num_trials=-1,
                    max_parallelism=1,
                    model_kwargs={"seed": sobol_seed},
                )
            ]
        )

    # Two-stage: Sobol init, then BO with selected acquisition
    acqf_class = {
        "GPEI": qExpectedImprovement,
        "UCB": qUpperConfidenceBound,
        "PI": qProbabilityOfImprovement,
    }[acquisition_id]

    acquisition_options = {}
    if acquisition_id == "UCB":
        acquisition_options["beta"] = ucb_beta

    return GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=init_trials,
                min_trials_observed=init_trials,
                max_parallelism=1,
                model_kwargs={"seed": sobol_seed},
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": acqf_class,
                    "acquisition_options": acquisition_options,
                },
            ),
        ]
    )


# ----------------------------------------
# Single run utility
# ----------------------------------------
def run_single_benchmark(
    acquisition_id: str,
    num_init: int,
    num_bo: int,
    seed: int,
    ucb_beta: float = 2.0,
) -> Tuple[List[float], Dict[str, float], float]:
    """
    Returns:
      - trace of best-so-far over trials (length num_init+num_bo)
      - best parameters dict
      - best objective value
    """
    # Reproducibility
    np.random.seed(seed)
    torch.manual_seed(seed)

    gs = build_generation_strategy_for_benchmark(
        acquisition_id=acquisition_id,
        init_trials=num_init,
        sobol_seed=seed,
        ucb_beta=ucb_beta,
    )
    ax_client = AxClient(generation_strategy=gs, random_seed=seed, verbose_logging=False)

    ax_client.create_experiment(
        name=f"branin_benchmark_{acquisition_id}_seed_{seed}",
        parameters=[
            {
                "name": "branin_x1",
                "type": "range",
                "bounds": [-5.0, 10.0],
                "value_type": "float",
            },
            {
                "name": "branin_x2",
                "type": "range",
                "bounds": [0.0, 15.0],
                "value_type": "float",
            },
        ],
        objectives={"branin_value": ObjectiveProperties(minimize=True)},
        is_test=True,
    )

    total_trials = num_init + num_bo
    best_trace: List[float] = []
    best_so_far: float = float("inf")

    for _ in range(total_trials):
        params, trial_index = ax_client.get_next_trial()

        x1 = float(params["branin_x1"])
        x2 = float(params["branin_x2"])
        value = compute_branin_value(x1, x2)

        # Noiseless proxy objective; SEM=0.0
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={"branin_value": (value, 0.0)},
        )

        best_so_far = min(best_so_far, value)
        best_trace.append(best_so_far)

    best_params, best_metrics = ax_client.get_best_parameters()
    best_val = float(best_metrics["branin_value"][0])

    return best_trace, best_params, best_val


# ----------------------------------------
# Experiment configuration
# ----------------------------------------
if __name__ == "__main__":
    # Compare multiple acquisition strategies
    acquisition_strategies = ["GPEI", "UCB", "PI", "SOBOL"]
    num_replicates = 10
    num_init_trials = 5
    num_bo_trials = 25
    ucb_beta_value = 2.0

    results: Dict[str, Dict[str, object]] = {}
    for acq in acquisition_strategies:
        traces = []
        best_vals = []
        best_params_list = []

        for r in range(num_replicates):
            seed = 1000 + 137 * r  # simple seed schedule
            trace, best_params, best_val = run_single_benchmark(
                acquisition_id=acq,
                num_init=num_init_trials,
                num_bo=num_bo_trials,
                seed=seed,
                ucb_beta=ucb_beta_value,
            )
            traces.append(trace)
            best_vals.append(best_val)
            best_params_list.append(best_params)

        traces_arr = np.array(traces)  # shape: [replicates, trials]
        mean_trace = traces_arr.mean(axis=0)
        std_trace = traces_arr.std(axis=0)

        results[acq] = {
            "traces": traces_arr,
            "mean_trace": mean_trace,
            "std_trace": std_trace,
            "best_vals": np.array(best_vals),
            "best_params": best_params_list,
        }

    # ----------------------------------------
    # Summary Table
    # ----------------------------------------
    summary_rows = []
    for acq in acquisition_strategies:
        best_vals = results[acq]["best_vals"]
        summary_rows.append(
            {
                "acquisition": acq,
                "final_best_mean": float(np.mean(best_vals)),
                "final_best_std": float(np.std(best_vals)),
                "replicates": int(num_replicates),
                "total_trials_per_run": int(num_init_trials + num_bo_trials),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values("final_best_mean")
    print("\n=== Branin benchmark: Final best value across replicates ===")
    print(summary_df.to_string(index=False))

    # ----------------------------------------
    # Visualization: incumbent traces averaged over replicates
    # ----------------------------------------
    plt.figure(figsize=(8, 5), dpi=150)
    x_axis = np.arange(1, num_init_trials + num_bo_trials + 1)
    color_map = {
        "GPEI": "#1f77b4",
        "UCB": "#ff7f0e",
        "PI": "#2ca02c",
        "SOBOL": "#7f7f7f",
    }

    for acq in acquisition_strategies:
        mean_trace = results[acq]["mean_trace"]
        std_trace = results[acq]["std_trace"]
        color = color_map.get(acq, None)
        plt.plot(x_axis, mean_trace, label=acq, color=color, linewidth=2)
        plt.fill_between(
            x_axis,
            mean_trace - std_trace,
            mean_trace + std_trace,
            color=color,
            alpha=0.15,
            linewidth=0,
        )

    plt.xlabel("Trial number")
    plt.ylabel("Best (incumbent) branin_value")
    plt.title("Branin benchmark: Mean incumbent over trials (±1 std)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()