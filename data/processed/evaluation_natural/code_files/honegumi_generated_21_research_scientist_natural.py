# Generated from a Honegumi skeleton and adapted to Multi-Objective Water Treatment Optimization
# %pip install ax-platform==0.4.3 matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple

from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain-specific objective names
REMOVAL_METRIC = "contaminant_removal_percent"        # Higher is better
COST_METRIC = "operational_cost_usd_per_m3"           # Lower is better


# Random generator for reproducible noisy evaluations
_rng = np.random.default_rng(2025)


def evaluate_water_treatment(
    alum_dose_mg_per_L: float,
    chlorine_dose_mg_per_L: float,
    polymer_dose_mg_per_L: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Simulated evaluation of a water treatment step with three dosages:
      - alum_dose_mg_per_L:     primary coagulant dose (0-50 mg/L)
      - chlorine_dose_mg_per_L: oxidant dose (0-8 mg/L)
      - polymer_dose_mg_per_L:  flocculant/coagulant aid dose (0-8 mg/L)

    Returns a dict mapping objective names to (mean, SEM):
      - contaminant_removal_percent: percentage removal (0-100), maximize
      - operational_cost_usd_per_m3: $/m^3, minimize

    Notes:
    - This is a plausible physics-inspired stub with noise to mimic measurement error.
    - Replace with actual plant measurements or process simulation as needed.
    """

    a = alum_dose_mg_per_L
    c = chlorine_dose_mg_per_L
    p = polymer_dose_mg_per_L

    # Contaminant removal model (saturating with diminishing returns + interactions)
    # Core saturation on combined effect of coagulant, oxidant, and polymer
    saturation_core = 100.0 * (1.0 - np.exp(-0.085 * a - 0.12 * c - 0.06 * p))

    # Synergy bonus between alum and polymer (common in coagulation-flocculation)
    synergy_bonus = 4.0 * (1.0 - np.exp(-0.006 * a * p))

    # Overdosing penalties (e.g., restabilization, floc shear, byproduct formation)
    penalty_alum = 0.35 * max(0.0, a - 30.0)
    penalty_chlorine = 0.7 * (max(0.0, c - 6.0) ** 1.2)
    penalty_polymer = 0.55 * (max(0.0, p - 6.0) ** 1.5)

    removal_mean = saturation_core + synergy_bonus - (penalty_alum + penalty_chlorine + penalty_polymer)
    removal_mean = float(np.clip(removal_mean, 0.0, 100.0))

    # Operational cost model ($/m^3)
    # Chemical cost is essentially linear in dose; add small nonlinear handling / residual treatment terms
    chem_cost = 0.0025 * a + 0.0018 * c + 0.0045 * p  # $ per m^3
    handling_cost = 0.0001 * (a + p) ** 2             # mixing/handling increases nonlinearly
    residuals_cost = 0.0008 * (max(0.0, c - 4.0) ** 2)  # e.g., DBP control costs at higher chlorine
    base_opex = 0.010  # fixed overhead per m^3

    cost_mean = float(max(0.0, chem_cost + handling_cost + residuals_cost + base_opex))

    # Inject realistic measurement noise and provide SEMs
    # Assume 3 replicate samples; adjust SEM accordingly if your experimental protocol differs.
    removal_sd = 1.6  # percent points
    cost_sd = 0.002   # dollars per m^3

    # Add noise to observed means
    removal_obs = float(np.clip(removal_mean + _rng.normal(0.0, removal_sd), 0.0, 100.0))
    cost_obs = float(max(0.0, cost_mean + _rng.normal(0.0, cost_sd)))

    # SEMs for 3 replicates
    n_rep = 3
    removal_sem = float(removal_sd / np.sqrt(n_rep))
    cost_sem = float(cost_sd / np.sqrt(n_rep))

    return {
        REMOVAL_METRIC: (removal_obs, removal_sem),
        COST_METRIC: (cost_obs, cost_sem),
    }


def is_nondominated_maximize_minimize(points: np.ndarray) -> np.ndarray:
    """
    Identify non-dominated points for 2D MOO with objectives:
      - points[:, 0]: maximize (e.g., contaminant removal)
      - points[:, 1]: minimize (e.g., operational cost)
    Returns a boolean mask of points on the Pareto frontier.
    """
    n = points.shape[0]
    is_pareto = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_pareto[i]:
            continue
        for j in range(n):
            if i == j:
                continue
            # Point j dominates i if:
            # removal_j >= removal_i AND cost_j <= cost_i with at least one strict
            if (points[j, 0] >= points[i, 0] and points[j, 1] <= points[i, 1]) and (
                points[j, 0] > points[i, 0] or points[j, 1] < points[i, 1]
            ):
                is_pareto[i] = False
                break
    return is_pareto


def main():
    # Set up Ax client and experiment
    ax_client = AxClient()
    ax_client.create_experiment(
        name="water_treatment_multi_objective",
        parameters=[
            {
                "name": "alum_dose_mg_per_L",
                "type": "range",
                "bounds": [0.0, 50.0],
                "value_type": "float",
            },
            {
                "name": "chlorine_dose_mg_per_L",
                "type": "range",
                "bounds": [0.0, 8.0],
                "value_type": "float",
            },
            {
                "name": "polymer_dose_mg_per_L",
                "type": "range",
                "bounds": [0.0, 8.0],
                "value_type": "float",
            },
        ],
        objectives={
            REMOVAL_METRIC: ObjectiveProperties(minimize=False),
            COST_METRIC: ObjectiveProperties(minimize=True),
        },
    )

    # Run a 32-trial study
    N_TRIALS = 32
    for _ in range(N_TRIALS):
        parameterization, trial_index = ax_client.get_next_trial()

        # Extract parameters
        alum = float(parameterization["alum_dose_mg_per_L"])
        chlorine = float(parameterization["chlorine_dose_mg_per_L"])
        polymer = float(parameterization["polymer_dose_mg_per_L"])

        # Evaluate and complete trial
        raw_data = evaluate_water_treatment(alum, chlorine, polymer)
        ax_client.complete_trial(trial_index=trial_index, raw_data=raw_data)

    # Retrieve results for plotting
    objectives = ax_client.objective_names
    df = ax_client.get_trials_data_frame()

    # Attempt to access metric columns directly by name
    # If your Ax version formats columns differently, adapt the selectors accordingly.
    if all(col in df.columns for col in objectives):
        removal_vals = df[objectives[0]].astype(float).values
        cost_vals = df[objectives[1]].astype(float).values
    else:
        # Fallback: try to pivot if the DataFrame is in long format
        if {"metric_name", "mean"}.issubset(df.columns):
            pivot = df.pivot_table(
                index="trial_index",
                columns="metric_name",
                values="mean",
                aggfunc="first",
            ).reset_index()
            removal_vals = pivot[REMOVAL_METRIC].astype(float).values
            cost_vals = pivot[COST_METRIC].astype(float).values
        else:
            raise RuntimeError("Unexpected Ax trials data frame format; cannot plot results.")

    results = np.column_stack([removal_vals, cost_vals])
    mask = np.isfinite(results).all(axis=1)
    results = results[mask]

    pareto_mask = is_nondominated_maximize_minimize(results)
    pareto_points = results[pareto_mask]
    pareto_sorted = pareto_points[np.argsort(pareto_points[:, 0])]

    # Plot observed points and empirical Pareto frontier
    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(results[:, 0], results[:, 1], facecolors="None", edgecolors="k", label="Observed")
    if len(pareto_sorted) > 0:
        ax.plot(
            pareto_sorted[:, 0],
            pareto_sorted[:, 1],
            color="#0033FF",
            lw=2,
            label="Pareto Front (observed)",
        )
    ax.set_xlabel(f"{REMOVAL_METRIC} (%)")
    ax.set_ylabel(f"{COST_METRIC} ($/m^3)")
    ax.legend()
    ax.set_title("Water Treatment: Removal vs. Operational Cost")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()