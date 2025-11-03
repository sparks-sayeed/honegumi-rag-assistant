# MAX phase conductivity optimization with Ax Platform
# %pip install ax-platform==0.4.3 matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties

np.random.seed(123)


# Objective metric name
objective_name = "electrical_conductivity"  # Units: S/m, maximize


# Domain-informed surrogate for electrical conductivity of MAX phases.
# If you have real measurements or a simulator, replace the logic in this function accordingly.
def measure_electrical_conductivity(
    M_element: str, A_element: str, X_anion: str, n_stoichiometry: str
):
    """
    Estimate electrical conductivity (S/m) for a MAX phase defined by:
      - M_element: early transition metal
      - A_element: A-group element
      - X_anion: C or N
      - n_stoichiometry: "1", "2", or "3" corresponding to Mn+1AXn

    This function provides an immediately executable, domain-inspired surrogate:
      1) Looks up a small table of known literature-inspired values (approximate).
      2) Otherwise computes a heuristic based on M/A/X/n contributions and mild synergies.
      3) Adds measurement noise and returns (mean, SEM) to match a noisy experimental setup.

    TODO: Replace this surrogate with your actual measurement pipeline:
      - run synthesis + measurement and return the measured conductivity,
      - or call your property prediction model trained on existing data,
      - or query a database of experimentally reported values.

    Returns:
      (mean, sem): tuple of floats. sem denotes the standard error of the mean.
    """
    # 1) Small lookup table for a few well-known MAX phases (values are illustrative)
    known_conductivities = {
        # (M, A, X, n) -> S/m
        ("Ti", "Al", "C", "2"): 3.8e6,  # Ti3AlC2
        ("Ti", "Si", "C", "2"): 4.5e6,  # Ti3SiC2
        ("Cr", "Al", "C", "1"): 2.0e6,  # Cr2AlC
        ("Nb", "Al", "C", "1"): 3.5e6,  # Nb2AlC
        ("Ta", "Al", "C", "1"): 3.0e6,  # Ta2AlC
        ("V", "Al", "C", "1"): 2.2e6,   # V2AlC
        ("Mo", "Ga", "C", "1"): 1.2e7,  # Mo2GaC (notable high conductivity)
        ("Ti", "Sn", "C", "2"): 3.0e6,  # Ti3SnC2 (approximate)
        ("Ti", "Al", "N", "1"): 5.0e6,  # Ti2AlN (approximate)
    }

    key = (M_element, A_element, X_anion, n_stoichiometry)
    if key in known_conductivities:
        base = known_conductivities[key]
    else:
        # 2) Heuristic model if not in lookup
        # Baseline scale is in the metallic regime for MAX phases
        base = 3.0e6

        M_factor = {
            "Ti": 1.00, "V": 1.05, "Cr": 0.95, "Nb": 1.10, "Mo": 1.15,
            "Ta": 1.12, "Zr": 0.98, "Hf": 1.00, "Sc": 0.85, "Y": 0.90
        }[M_element]

        A_factor = {
            "Al": 1.00, "Si": 1.03, "Ge": 1.02, "Ga": 1.01,
            "In": 1.00, "Sn": 0.98, "Pb": 0.95
        }[A_element]

        X_factor = {"C": 1.00, "N": 1.08}[X_anion]

        n = int(n_stoichiometry)
        if n == 1:
            n_factor = 1.00
        elif n == 2:
            n_factor = 1.04
        else:  # n == 3
            n_factor = 1.07

        # Mild synergy bonuses
        synergy = 1.0
        if X_anion == "N" and M_element in {"Mo", "Nb", "Ta"}:
            synergy *= 1.10
        if A_element in {"Si", "Ge"}:
            synergy *= 1.04
        if M_element in {"Ti", "V"} and A_element in {"Al", "Ga"}:
            synergy *= 1.02

        base *= M_factor * A_factor * X_factor * n_factor * synergy

    # 3) Add realistic measurement noise; report SEM to Ax
    rel_sd = 0.05  # 5% relative standard deviation for measurements
    noise = np.random.normal(loc=0.0, scale=rel_sd * base)
    measured = max(0.0, base + noise)
    sem = rel_sd * base  # SEM estimate; adjust if you average multiple replicates

    return measured, sem


# Initialize Ax client and create experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="maximize_max_phase_conductivity",
    parameters=[
        {
            "name": "M_element",
            "type": "choice",
            "values": ["Ti", "V", "Cr", "Nb", "Mo", "Ta", "Zr", "Hf", "Sc", "Y"],
            "is_ordered": False,
        },
        {
            "name": "A_element",
            "type": "choice",
            "values": ["Al", "Si", "Ge", "Ga", "In", "Sn", "Pb"],
            "is_ordered": False,
        },
        {
            "name": "X_anion",
            "type": "choice",
            "values": ["C", "N"],
            "is_ordered": False,
        },
        {
            "name": "n_stoichiometry",
            "type": "choice",
            "values": ["1", "2", "3"],  # corresponds to Mn+1AXn with n in {1,2,3}
            "is_ordered": True,
        },
    ],
    objectives={objective_name: ObjectiveProperties(minimize=False)},
)


# Optimization loop: budget of 30 synthesis-evaluation trials
N_TRIALS = 30
for i in range(N_TRIALS):
    parameters, trial_index = ax_client.get_next_trial()
    try:
        # Extract parameters
        M_el = parameters["M_element"]
        A_el = parameters["A_element"]
        X_an = parameters["X_anion"]
        n_val = parameters["n_stoichiometry"]

        # Evaluate (replace this with your lab or simulator call)
        mean, sem = measure_electrical_conductivity(M_el, A_el, X_an, n_val)

        # Log results back to Ax
        ax_client.complete_trial(trial_index=trial_index, raw_data=(mean, sem))
    except Exception as e:
        # In case of evaluation failure, mark the trial as failed so Ax won't propose it again
        ax_client.log_trial_failure(trial_index=trial_index)


# Retrieve best parameters (model-predicted best)
best_parameters, best_values = ax_client.get_best_parameters()
best_mean, best_cov = best_values

print("Best MAX phase parameters found:")
print(best_parameters)
print(f"Model-predicted mean conductivity: {best_mean[objective_name]:.3e} S/m")


# Plot optimization trace (observed values and best-so-far)
df = ax_client.get_trials_data_frame()
if objective_name in df.columns and len(df) > 0:
    y = df[objective_name].values.astype(float)
    best_so_far = np.maximum.accumulate(y)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(df.index, y, ec="k", fc="none", label="Observed")
    ax.plot(df.index, best_so_far, color="#0033FF", lw=2, label="Best to Trial")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(f"{objective_name} (S/m)")
    ax.set_title("MAX phase conductivity optimization")
    ax.legend()
    plt.tight_layout()
    plt.show()