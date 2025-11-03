# Generated for MAX phase composition-property optimization using Ax Platform
# %pip install ax-platform==0.4.3 matplotlib numpy pandas
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


random.seed(1234)
np.random.seed(1234)


# Domain: MAX phases M_{n+1}AX_n (n in {1, 2} corresponding to 211 and 312 phases).
# Objective: maximize electrical conductivity (S/m).
# This script:
# 1) Generates a synthetic library of MAX phase candidates with simple composition-derived descriptors;
# 2) Defines a hidden ground-truth function to simulate conductivity;
# 3) Pretends a subset is "historical" and attaches them to Ax;
# 4) Runs ~30 Bayesian optimization trials to select new candidates for measurement;
# 5) Plots optimization progress and reports the best candidate.


# Basic element properties needed for synthetic featurization (subset only)
ATOMIC_NUMBER = {
    "Ti": 22, "V": 23, "Nb": 41, "Cr": 24, "Zr": 40, "Hf": 72, "Ta": 73,
    "Al": 13, "Si": 14, "Ge": 32, "Sn": 50,
    "C": 6, "N": 7,
}
ELECTRONEGATIVITY = {
    "Ti": 1.54, "V": 1.63, "Nb": 1.60, "Cr": 1.66, "Zr": 1.33, "Hf": 1.30, "Ta": 1.50,
    "Al": 1.61, "Si": 1.90, "Ge": 2.01, "Sn": 1.96,
    "C": 2.55, "N": 3.04,
}
VALENCE_ELECTRONS = {
    "Ti": 4, "V": 5, "Nb": 5, "Cr": 6, "Zr": 4, "Hf": 4, "Ta": 5,
    "Al": 3, "Si": 4, "Ge": 4, "Sn": 4,
    "C": 4, "N": 5,
}


def format_max_formula(m: str, a: str, x: str, phase: str) -> str:
    # phase '211' -> M2AX, '312' -> M3AX2
    if phase == "211":
        return f"{m}2{a}{x}"
    else:
        return f"{m}3{a}{x}2"


def weighted_average_atomic_number(m: str, a: str, x: str, phase: str) -> float:
    if phase == "211":
        total = 4.0
        return (2 * ATOMIC_NUMBER[m] + 1 * ATOMIC_NUMBER[a] + 1 * ATOMIC_NUMBER[x]) / total
    else:
        total = 6.0
        return (3 * ATOMIC_NUMBER[m] + 1 * ATOMIC_NUMBER[a] + 2 * ATOMIC_NUMBER[x]) / total


def delta_en_ma(m: str, a: str) -> float:
    return abs(ELECTRONEGATIVITY[m] - ELECTRONEGATIVITY[a])


def generate_max_phase_library() -> pd.DataFrame:
    M_choices = ["Ti", "V", "Nb", "Cr", "Zr", "Hf", "Ta"]
    A_choices = ["Al", "Si", "Ge", "Sn"]
    X_choices = ["C", "N"]
    phase_types = ["211", "312"]  # 211: M2AX, 312: M3AX2

    rows = []
    for m in M_choices:
        for a in A_choices:
            for x in X_choices:
                for p in phase_types:
                    formula = format_max_formula(m, a, x, p)
                    avg_Z = weighted_average_atomic_number(m, a, x, p)
                    den_ma = delta_en_ma(m, a)
                    n_layers = 2 if p == "211" else 3
                    x_is_c = 1 if x == "C" else 0
                    rows.append(
                        {
                            "max_phase_id": formula,
                            "M_element": m,
                            "A_element": a,
                            "X_element": x,
                            "phase_type": p,  # '211' or '312'
                            "n_layers": n_layers,
                            "avg_atomic_number": avg_Z,
                            "delta_en_MA": den_ma,
                            "x_is_c": x_is_c,
                        }
                    )
    df = pd.DataFrame(rows)
    # Normalize delta EN for use in synthetic ground truth
    max_den = df["delta_en_MA"].max()
    df["delta_en_MA_norm"] = df["delta_en_MA"] / max_den if max_den > 0 else df["delta_en_MA"]
    # Simple density proxy
    df["density_proxy"] = df["avg_atomic_number"] * (1.0 + 0.05 * (df["n_layers"] - 2))
    return df


def synthetic_conductivity_function(row: pd.Series) -> float:
    # Synthetic "true" conductivity model (unknown to optimizer), returns S/m.
    # Shape values to roughly 5e6 -- 3e7 S/m range.
    base = 8.0e6

    # X dependence: carbides generally more conductive than nitrides in this toy model
    effect_x = 3.0e6 if row["x_is_c"] == 1 else -1.0e6

    # Layer/phase dependence
    effect_layers = 1.2e6 if row["phase_type"] == "312" else 0.0

    # M-A electronegativity mismatch (normalized): larger mismatch boosts metallicity in this toy model
    effect_delta_en = 2.4e6 * float(row["delta_en_MA_norm"])

    # Heavier (avg Z) modestly increases conductivity; scale around 40
    effect_avgZ = 1.8e6 * (float(row["avg_atomic_number"]) / 40.0)

    # A-element specific tuning
    a = row["A_element"]
    a_weight = {"Al": 1.0, "Si": 1.2, "Ge": 1.1, "Sn": 0.9}[a]
    effect_a = 6.0e5 * a_weight

    # Penalty for some M-elements to create structure
    m = row["M_element"]
    effect_m = -8.0e5 if m in {"Zr", "Hf"} else (2.0e5 if m in {"V", "Nb"} else 0.0)

    # Mild nonlinear interaction with delta EN
    den = float(row["delta_en_MA_norm"])
    effect_nonlinear = 1.0e6 * math.sin(3.0 * den + (0.7 if row["x_is_c"] == 1 else -0.3))

    # Density proxy, small positive effect
    effect_density = 2.0e5 * float(row["density_proxy"]) / 50.0

    cond = base + effect_x + effect_layers + effect_delta_en + effect_avgZ + effect_a + effect_m + effect_nonlinear + effect_density
    return max(cond, 1.0e5)


def build_synthetic_dataset() -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Generate full library and assign synthetic "true" conductivity
    library = generate_max_phase_library()
    library["true_electrical_conductivity_S_per_m"] = library.apply(synthetic_conductivity_function, axis=1)

    # Create a "historical" subset to seed the optimization
    all_ids = list(library["max_phase_id"].values)
    random.shuffle(all_ids)
    # Use around a third of library as historical; ensure at least 24 points
    n_hist = max(24, len(all_ids) // 3)
    historical_ids = set(all_ids[:n_hist])
    library["is_historical"] = library["max_phase_id"].apply(lambda cid: cid in historical_ids)

    truth_map = dict(zip(library["max_phase_id"], library["true_electrical_conductivity_S_per_m"]))
    return library, truth_map


def measure_conductivity_with_noise(true_value: float, rel_noise: float = 0.05) -> Tuple[float, float]:
    # Simulate a noisy measurement: Gaussian noise with relative std.
    sd = abs(rel_noise * true_value)
    measured = float(np.random.normal(loc=true_value, scale=sd))
    return measured, sd


def main():
    # Prepare synthetic data (replace with CSV loading + featurization as needed)
    library_df, truth_map = build_synthetic_dataset()

    candidate_ids: List[str] = list(library_df["max_phase_id"].values)
    historical_df = library_df[library_df["is_historical"]]
    non_historical_df = library_df[~library_df["is_historical"]]

    objective_name = "electrical_conductivity_S_per_m"
    parameter_name = "max_phase_id"
    experiment_name = "max_phase_electrical_conductivity_optimization"
    total_budget = 30  # Number of new experiments to run (beyond historical)

    ax_client = AxClient()

    ax_client.create_experiment(
        name=experiment_name,
        parameters=[
            {
                "name": parameter_name,
                "type": "choice",
                "values": sorted(candidate_ids),
                "value_type": "str",
                "is_ordered": False,
            }
        ],
        objectives={objective_name: ObjectiveProperties(minimize=False)},
    )

    # Attach historical measurements (seed data)
    for _, row in historical_df.iterrows():
        cid = row["max_phase_id"]
        true_y = truth_map[cid]
        measured, sd = measure_conductivity_with_noise(true_y, rel_noise=0.06)
        trial_index = ax_client.attach_trial(parameters={parameter_name: cid})
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={objective_name: (measured, sd)},
        )

    # Run optimization for the remaining budget, avoiding re-testing historical candidates
    n_trials = 0
    evaluated_ids = set(historical_df["max_phase_id"].values)

    while n_trials < total_budget:
        try:
            parameterization, trial_index = ax_client.get_next_trial()
        except Exception as e:
            # In rare cases Ax might require more data to proceed; break gracefully.
            print(f"Stopped early due to Ax exception: {e}")
            break

        cid = parameterization[parameter_name]

        # Skip if already evaluated due to any unforeseen duplication
        if cid in evaluated_ids:
            ax_client.log_trial_failure(trial_index=trial_index)
            continue

        # Simulate measurement from hidden ground-truth
        true_val = truth_map[cid]
        measured, sd = measure_conductivity_with_noise(true_val, rel_noise=0.06)
        ax_client.complete_trial(
            trial_index=trial_index,
            raw_data={objective_name: (measured, sd)},
        )
        evaluated_ids.add(cid)
        n_trials += 1

    best_parameters, best_metrics = ax_client.get_best_parameters()
    best_id = best_parameters.get(parameter_name)
    best_metric_val = best_metrics[objective_name]["mean"]
    print("Best candidate selected:")
    print(f"  MAX phase: {best_id}")
    print(f"  Estimated {objective_name}: {best_metric_val:.3e} (S/m)")

    # Plot optimization trace (best-so-far for maximization)
    df = ax_client.get_trials_data_frame()
    if objective_name in df.columns:
        y = df[objective_name]
        x = np.arange(len(y))
        best_so_far = np.maximum.accumulate(y)

        fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
        ax.scatter(x, y, ec="k", fc="none", label="Observed")
        ax.plot(x, best_so_far, color="#0033FF", lw=2, label="Best so far")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(objective_name + " (S/m)")
        ax.set_title("MAX Phase Conductivity Optimization")
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("Could not locate metric column for plotting; skipping visualization.")


if __name__ == "__main__":
    main()