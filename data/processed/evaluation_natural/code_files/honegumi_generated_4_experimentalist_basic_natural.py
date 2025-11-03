# Generated from Honegumi skeleton, adapted for electrolyte formulation optimization
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Domain-specific objective/metric name
objective_name = "ionic_conductivity_mS_per_cm"


def evaluate_electrolyte_conductivity(
    salt1_fraction: float,
    salt2_fraction: float,
    salt3_fraction: float,
    salt4_fraction: float,
    noise_std: float = 0.1,
    rng: np.random.Generator = None,
) -> float:
    """
    Evaluate ionic conductivity (mS/cm) for a 4-salt electrolyte formulation.

    IMPORTANT: Replace this stub with actual experimental measurement.
    In a real workflow:
      - Prepare electrolyte with given salt fractions
      - Measure ionic conductivity (e.g., via EIS) at a fixed temperature (e.g., 25 C)
      - Return the measured conductivity in mS/cm

    Parameters
    ----------
    salt1_fraction, salt2_fraction, salt3_fraction, salt4_fraction
        Fractions of each salt in the mixture; should be nonnegative and sum to 1.0.
    noise_std
        Standard deviation of measurement noise to simulate experimental uncertainty.
    rng
        Optional numpy random number generator for reproducibility.

    Returns
    -------
    float
        Simulated ionic conductivity in mS/cm.
    """

    # Ensure fractions sum to 1.0 (numerical robustness) and are nonnegative
    fractions = np.array(
        [salt1_fraction, salt2_fraction, salt3_fraction, salt4_fraction], dtype=float
    )
    fractions = np.clip(fractions, 0.0, None)
    total = fractions.sum()
    if total <= 0.0:
        # Degenerate case; no salts -> no conductivity
        return 0.0
    fractions = fractions / total

    # Intrinsic conductivities (mS/cm) for each salt when used alone (hypothetical values)
    # These are placeholders; replace with values/models appropriate to your chemistry.
    intrinsic = np.array([7.5, 10.5, 6.0, 12.0], dtype=float)

    # Base contribution: weighted by fractions
    base_conductivity = float(np.dot(fractions, intrinsic))

    # Pairwise interaction (synergy/antagonism) terms (hypothetical coefficients)
    # Positive beta_ij means synergy, negative means antagonism.
    f1, f2, f3, f4 = fractions
    beta12 = 2.0
    beta13 = 0.5
    beta14 = 1.2
    beta23 = -0.4
    beta24 = 2.3
    beta34 = 0.3
    interaction = (
        beta12 * f1 * f2
        + beta13 * f1 * f3
        + beta14 * f1 * f4
        + beta23 * f2 * f3
        + beta24 * f2 * f4
        + beta34 * f3 * f4
    )

    # Mild curvature favoring balanced mixtures (optional, hypothetical)
    # Peaks around evenly mixed formulations; coefficient tunes strength.
    balance_bonus = 0.8 * (4.0 * np.prod(fractions + 1e-12) ** 0.25)

    # Aggregate conductivity model
    conductivity = base_conductivity + interaction + balance_bonus

    # Add measurement noise if requested
    if rng is None:
        rng = np.random.default_rng()
    noisy_conductivity = float(conductivity + rng.normal(0.0, noise_std))

    # Ensure non-negative result
    return max(0.0, noisy_conductivity)


# Reparameterized compositional constraint:
# Optimize over salt1..salt3 in [0, 1] with sum <= 1.0; derive salt4 as the remainder.
composition_total = 1.0

ax_client = AxClient()

ax_client.create_experiment(
    name="battery_electrolyte_formulation_optimization",
    parameters=[
        {"name": "salt1_fraction", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "salt2_fraction", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "salt3_fraction", "type": "range", "bounds": [0.0, 1.0]},
    ],
    objectives={
        objective_name: ObjectiveProperties(minimize=False),
    },
    parameter_constraints=[
        "salt1_fraction + salt2_fraction + salt3_fraction <= 1.0",
    ],
)

# Optimization budget
num_trials = 40
rng = np.random.default_rng(2025)

for _ in range(num_trials):
    parameterization, trial_index = ax_client.get_next_trial()

    # Extract reparameterized fractions
    s1 = float(parameterization["salt1_fraction"])
    s2 = float(parameterization["salt2_fraction"])
    s3 = float(parameterization["salt3_fraction"])
    remainder = composition_total - (s1 + s2 + s3)
    s4 = max(0.0, remainder)  # ensure nonnegative remainder

    # Evaluate ionic conductivity (mS/cm)
    conductivity_value = evaluate_electrolyte_conductivity(
        salt1_fraction=s1,
        salt2_fraction=s2,
        salt3_fraction=s3,
        salt4_fraction=s4,
        noise_std=0.1,  # simulate noisy measurements
        rng=rng,
    )

    # Single-objective; can pass float directly
    ax_client.complete_trial(trial_index=trial_index, raw_data=conductivity_value)

# Retrieve best result
best_parameters, best_values = ax_client.get_best_parameters()
best_conductivity = best_values[objective_name]["value"]
best_s1 = best_parameters["salt1_fraction"]
best_s2 = best_parameters["salt2_fraction"]
best_s3 = best_parameters["salt3_fraction"]
best_s4 = composition_total - (best_s1 + best_s2 + best_s3)
best_s4 = max(0.0, best_s4)

print("Best electrolyte formulation found:")
print(f"  salt1_fraction = {best_s1:.4f}")
print(f"  salt2_fraction = {best_s2:.4f}")
print(f"  salt3_fraction = {best_s3:.4f}")
print(f"  salt4_fraction = {best_s4:.4f}  (derived to satisfy sum=1.0)")
print(f"Estimated {objective_name} = {best_conductivity:.4f}")

# Plot optimization history
objective_names = ax_client.objective_names
df = ax_client.get_trials_data_frame()

# Filter to trials with observed objective
df_plot = df.dropna(subset=objective_names)

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.scatter(
    df_plot.index.values,
    df_plot[objective_names[0]].values,
    ec="k",
    fc="none",
    label="Observed",
)
ax.plot(
    df_plot.index.values,
    np.maximum.accumulate(df_plot[objective_names[0]].values),
    color="#0033FF",
    lw=2,
    label="Best-so-far",
)
ax.set_xlabel("Trial Number")
ax.set_ylabel(f"{objective_names[0]} (mS/cm)")
ax.set_title("Electrolyte Ionic Conductivity Optimization")
ax.legend()
plt.tight_layout()
plt.show()