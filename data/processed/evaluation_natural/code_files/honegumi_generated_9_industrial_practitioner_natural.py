# High-strength concrete mix optimization with Ax (Bayesian Optimization)
# %pip install ax-platform==0.4.3 matplotlib

import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Domain-specific objective name
objective_name = "compressive_strength_mpa"

# Total fraction for mix composition: cement + water + aggregate == 1.0
TOTAL_FRACTION = 1.0

# Random generator for simulating experimental noise (set a fixed seed for reproducibility)
_rng = np.random.default_rng(42)


def simulate_compressive_strength_mpa(cement_fraction: float, water_fraction: float, aggregate_fraction: float) -> float:
    """
    Simulated compressive strength (MPa) for a concrete mixture defined by fractions
    of cement, water, and aggregate that sum to 1.0.

    This is a physics-inspired surrogate based on Abrams' law and typical mix behavior:
    - Strength decreases as water/cement ratio increases.
    - Sufficient cement paste content improves strength.
    - Excessive aggregate reduces paste continuity; there is a broad optimum around ~0.6 aggregate.

    The function also includes mild random measurement noise to reflect experimental uncertainty.

    Parameters are fractions by mass or volume that sum to 1.0.
    """

    # Guard against invalid edge cases
    cement = max(cement_fraction, 0.0)
    water = max(water_fraction, 0.0)
    aggregate = max(aggregate_fraction, 0.0)

    # Handle degenerate case (no cement -> no binding, very low strength)
    if cement <= 1e-8:
        base_strength = 2.0  # MPa, essentially no binder
    else:
        # Water-cement ratio
        w_c = water / cement

        # Abrams-like inverse relationship; values tuned for plausible high-strength ranges.
        # Produces ~60-100 MPa in realistic w/c ranges (0.25 - 0.6)
        base_strength = 80.0 / (0.5 + w_c)

    # Cement fraction effect: more cement (within reason) increases paste continuity.
    # Center around 0.18 with a gentle slope. Clamp to avoid extreme multipliers.
    cement_factor = 1.0 + 1.0 * (cement - 0.18)
    cement_factor = float(np.clip(cement_factor, 0.8, 1.2))

    # Aggregate effect: broad optimum near ~0.6, penalize extremes quadratically.
    aggregate_factor = 1.0 - 1.2 * (aggregate - 0.6) ** 2
    aggregate_factor = float(np.clip(aggregate_factor, 0.7, 1.05))

    # Combine factors
    strength = base_strength * cement_factor * aggregate_factor

    # Simulated measurement/process noise (MPa)
    noise = _rng.normal(loc=0.0, scale=2.0)
    measured_strength = max(strength + noise, 0.0)

    return float(measured_strength)


def evaluate_concrete_mix(parameterization: dict) -> dict:
    """
    Evaluate the compressive strength for a proposed mix.

    Ax will propose cement_fraction and water_fraction. We enforce the composition
    constraint by computing aggregate_fraction = 1 - (cement + water).

    Returns a dict mapping objective metric name to (mean, SEM). We pass SEM=None
    to indicate unknown noise level (Ax will infer it).
    """
    cement_fraction = float(parameterization["cement_fraction"])
    water_fraction = float(parameterization["water_fraction"])
    aggregate_fraction = float(TOTAL_FRACTION - (cement_fraction + water_fraction))

    strength_mpa = simulate_compressive_strength_mpa(
        cement_fraction=cement_fraction,
        water_fraction=water_fraction,
        aggregate_fraction=aggregate_fraction,
    )

    return {objective_name: (strength_mpa, None)}  # Unknown noise (SEM=None)


# Initialize Ax client and define experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="high_strength_concrete_mix_optimization",
    parameters=[
        # We optimize over two parameters and compute the third from the composition constraint.
        {"name": "cement_fraction", "type": "range", "bounds": [0.0, TOTAL_FRACTION], "value_type": "float"},
        {"name": "water_fraction", "type": "range", "bounds": [0.0, TOTAL_FRACTION], "value_type": "float"},
    ],
    objectives={objective_name: ObjectiveProperties(minimize=False)},
    # Linear constraints to reflect domain knowledge:
    # 1) cement + water <= 1.0  (aggregate >= 0)
    # 2) water <= cement        (cement >= water)
    # 3) 2*cement + water <= 1.0   (aggregate >= cement)
    # 4) cement + 2*water <= 1.0   (aggregate >= water)
    parameter_constraints=[
        "cement_fraction + water_fraction <= 1.0",
        "water_fraction <= cement_fraction",
        "2.0*cement_fraction + water_fraction <= 1.0",
        "cement_fraction + 2.0*water_fraction <= 1.0",
    ],
)

# Run optimization with a budget of 20 trials
NUM_TRIALS = 20
for _ in range(NUM_TRIALS):
    params, trial_index = ax_client.get_next_trial()
    res = evaluate_concrete_mix(params)
    ax_client.complete_trial(trial_index=trial_index, raw_data=res)

# Retrieve best parameters according to Ax's model
best_parameters, best_metrics = ax_client.get_best_parameters()
best_cement = float(best_parameters["cement_fraction"])
best_water = float(best_parameters["water_fraction"])
best_aggregate = float(TOTAL_FRACTION - (best_cement + best_water))

# Compute strength at best parameters using our simulator (without additional noise for reporting)
reported_strength = simulate_compressive_strength_mpa(best_cement, best_water, best_aggregate)

print("Best mix found:")
print(f"  cement_fraction    = {best_cement:.4f}")
print(f"  water_fraction     = {best_water:.4f}")
print(f"  aggregate_fraction = {best_aggregate:.4f}")
print(f"  estimated_strength = {reported_strength:.2f} MPa")

# Plot results: observed objective by trial and running best-so-far (maximize)
try:
    objective_names = ax_client.objective_names
    df = ax_client.get_trials_data_frame()

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
    ax.scatter(df.index, df[objective_names], ec="k", fc="none", label="Observed")
    ax.plot(
        df.index,
        np.maximum.accumulate(df[objective_names]),
        color="#0033FF",
        lw=2,
        label="Best to Trial",
    )
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(objective_names[0])
    ax.set_title("High-strength concrete mix optimization")
    ax.legend()
    plt.tight_layout()
    plt.show()
except Exception as e:
    # Fallback: simple print if plotting fails due to environment or dataframe format changes
    print("Plotting skipped due to error:", repr(e))