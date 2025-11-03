# Generated benchmarking script for acquisition functions on Branin-Hoo
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import Surrogate

from botorch.models.gp_regression import SingleTaskGP
from botorch.acquisition.monte_carlo import qExpectedImprovement, qThompsonSampling
from botorch.acquisition.analytic import UpperConfidenceBound


# --------------------------
# Problem-specific utilities
# --------------------------
def branin_hoo(x: float, y: float) -> float:
    """
    Deterministic Branin-Hoo function (minimization).
    Domain: x in [-5, 10], y in [0, 15]
    Global minimum value ≈ 0.397887 at three locations.
    """
    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return float(a * (y - b * x**2 + c * x - r) ** 2 + s * (1 - t) * np.cos(x) + s)


# --------------------------
# Benchmark configuration
# --------------------------
# Total evaluations per run (including random initialization)
TOTAL_TRIALS = 40
# Random initialization (Sobol) trials before BO
N_INIT = 8
# Number of independent random seeds per acquisition function
NUM_REPLICATES = 10
# UCB exploration parameter (beta)
UCB_BETA = 0.20
# Base seed for reproducibility (per replicate a different Sobol seed is used)
BASE_SEED = 2025

PARAMETERS = [
    {"name": "x", "type": "range", "bounds": [-5.0, 10.0]},  # Branin x
    {"name": "y", "type": "range", "bounds": [0.0, 15.0]},  # Branin y
]
OBJECTIVE_NAME = "branin_value"

# Acquisition functions to compare
ACQF_SPECS = {
    "Expected Improvement": {
        "botorch_acqf_class": qExpectedImprovement,
        "acquisition_options": {},
    },
    "Upper Confidence Bound": {
        "botorch_acqf_class": UpperConfidenceBound,
        "acquisition_options": {"beta": UCB_BETA},
    },
    "Thompson Sampling": {
        "botorch_acqf_class": qThompsonSampling,
        "acquisition_options": {},
    },
}


# --------------------------
# Ax configuration builders
# --------------------------
def make_generation_strategy(
    acqf_spec: Dict, sobol_seed: int
) -> GenerationStrategy:
    """
    Build a generation strategy with:
    - Sobol initialization (N_INIT points)
    - GP with specified acquisition function for the remainder
    """
    return GenerationStrategy(
        steps=[
            GenerationStep(
                model=Models.SOBOL,
                num_trials=N_INIT,
                min_trials_observed=N_INIT,
                max_parallelism=1,
                model_kwargs={"seed": sobol_seed},
            ),
            GenerationStep(
                model=Models.BOTORCH_MODULAR,
                num_trials=-1,
                max_parallelism=1,
                model_kwargs={
                    "surrogate": Surrogate(SingleTaskGP),
                    "botorch_acqf_class": acqf_spec["botorch_acqf_class"],
                    "acquisition_options": acqf_spec.get("acquisition_options", {}),
                },
            ),
        ]
    )


def create_experiment(ax_client: AxClient, name: str) -> None:
    ax_client.create_experiment(
        name=name,
        parameters=PARAMETERS,
        objectives={OBJECTIVE_NAME: ObjectiveProperties(minimize=True)},
    )


# --------------------------
# Evaluation
# --------------------------
def evaluate_branin(parameters: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate the Branin-Hoo function and return objective dict.
    """
    x = float(parameters["x"])
    y = float(parameters["y"])
    value = branin_hoo(x, y)
    return {OBJECTIVE_NAME: (value, 0.0)}


# --------------------------
# Benchmark runner
# --------------------------
def run_single_benchmark(
    acqf_name: str, acqf_spec: Dict, replicate_id: int, total_trials: int
) -> List[float]:
    """
    Run one replicate for a given acquisition function.
    Returns:
        best_so_far: list of best incumbent objective values per trial index (0-based).
    """
    sobol_seed = BASE_SEED + replicate_id
    gs = make_generation_strategy(acqf_spec, sobol_seed=sobol_seed)
    ax_client = AxClient(generation_strategy=gs, enforce_sequential_optimization=True)
    exp_name = f"branin_{acqf_name.replace(' ', '_').lower()}_rep{replicate_id}"
    create_experiment(ax_client, name=exp_name)

    best_so_far: List[float] = []
    incumbent = np.inf

    for t in range(total_trials):
        parameters, trial_index = ax_client.get_next_trial()
        result = evaluate_branin(parameters)
        value = result[OBJECTIVE_NAME][0]
        incumbent = min(incumbent, value)
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
        best_so_far.append(incumbent)

    return best_so_far


def run_benchmark_suite() -> Dict[str, np.ndarray]:
    """
    Run all acquisition functions across NUM_REPLICATES.
    Returns:
        results: dict mapping acqf_name -> array shape (NUM_REPLICATES, TOTAL_TRIALS)
    """
    results: Dict[str, np.ndarray] = {}
    for acqf_name, acqf_spec in ACQF_SPECS.items():
        all_runs = []
        for r in range(NUM_REPLICATES):
            best_curve = run_single_benchmark(
                acqf_name=acqf_name,
                acqf_spec=acqf_spec,
                replicate_id=r,
                total_trials=TOTAL_TRIALS,
            )
            all_runs.append(best_curve)
        results[acqf_name] = np.array(all_runs, dtype=float)
    return results


# --------------------------
# Visualization and summary
# --------------------------
def plot_optimization_curves(results: Dict[str, np.ndarray]) -> None:
    """
    Plot mean best-so-far objective across replicates with 95% CI.
    """
    plt.figure(figsize=(8, 5), dpi=150)
    trial_idx = np.arange(1, TOTAL_TRIALS + 1)

    color_map = {
        "Expected Improvement": "#1f77b4",
        "Upper Confidence Bound": "#2ca02c",
        "Thompson Sampling": "#d62728",
    }

    for acqf_name, curves in results.items():
        mean_curve = curves.mean(axis=0)
        std_curve = curves.std(axis=0, ddof=1)
        sem_curve = std_curve / np.sqrt(curves.shape[0])
        ci95_low = mean_curve - 1.96 * sem_curve
        ci95_high = mean_curve + 1.96 * sem_curve

        color = color_map.get(acqf_name, None)
        plt.plot(trial_idx, mean_curve, label=acqf_name, color=color, lw=2)
        plt.fill_between(trial_idx, ci95_low, ci95_high, color=color, alpha=0.2)

    # Known Branin-Hoo global minimum
    branin_global_min = 0.397887
    plt.axhline(y=branin_global_min, color="k", ls="--", lw=1, label="Global minimum")

    plt.xlabel("Trial number")
    plt.ylabel("Best objective value (lower is better)")
    plt.title(f"Branin-Hoo: Best-so-far across {NUM_REPLICATES} replicates")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def summarize_final_results(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Summarize final best value per replicate, with mean/std.
    """
    rows = []
    for acqf_name, curves in results.items():
        final_vals = curves[:, -1]
        rows.append(
            {
                "Acquisition": acqf_name,
                "Mean final best": final_vals.mean(),
                "Std final best": final_vals.std(ddof=1),
                "Median final best": np.median(final_vals),
                "Min across reps": final_vals.min(),
                "Max across reps": final_vals.max(),
            }
        )
    df = pd.DataFrame(rows).sort_values(by="Mean final best")
    return df.reset_index(drop=True)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 0):
    """
    Simple bootstrap CI for difference of means: mean(a) - mean(b).
    """
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(a)
    m = len(b)
    for _ in range(n_boot):
        sa = a[rng.randint(0, n, size=n)]
        sb = b[rng.randint(0, m, size=m)]
        diffs.append(sa.mean() - sb.mean())
    diffs = np.array(diffs)
    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    return diffs.mean(), (ci_low, ci_high)


def pairwise_bootstrap_summary(results: Dict[str, np.ndarray]) -> pd.DataFrame:
    """
    Pairwise bootstrap mean differences (final best), for simple statistical comparison.
    Negative mean difference favors the row method (lower is better).
    """
    methods = list(results.keys())
    finals = {k: results[k][:, -1] for k in methods}
    data = []
    for i, mi in enumerate(methods):
        for j, mj in enumerate(methods):
            if i == j:
                data.append(
                    {
                        "Method_A": mi,
                        "Method_B": mj,
                        "MeanDiff_A_minus_B": 0.0,
                        "CI95_low": 0.0,
                        "CI95_high": 0.0,
                    }
                )
            else:
                mdiff, (lo, hi) = bootstrap_diff_ci(finals[mi], finals[mj], seed=BASE_SEED)
                data.append(
                    {
                        "Method_A": mi,
                        "Method_B": mj,
                        "MeanDiff_A_minus_B": mdiff,
                        "CI95_low": lo,
                        "CI95_high": hi,
                    }
                )
    df = pd.DataFrame(data)
    return df


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    np.random.seed(BASE_SEED)

    # Run benchmark
    benchmark_results = run_benchmark_suite()

    # Summaries
    summary_df = summarize_final_results(benchmark_results)
    print("\nFinal best value summary across replicates:")
    print(summary_df.to_string(index=False, float_format=lambda v: f"{v:0.6f}"))

    # Pairwise bootstrap comparisons
    pairwise_df = pairwise_bootstrap_summary(benchmark_results)
    print("\nPairwise bootstrap mean differences (A - B) on final best values (95% CI):")
    print(pairwise_df.to_string(index=False, float_format=lambda v: f"{v:0.6f}"))

    # Visualization
    plot_optimization_curves(benchmark_results)