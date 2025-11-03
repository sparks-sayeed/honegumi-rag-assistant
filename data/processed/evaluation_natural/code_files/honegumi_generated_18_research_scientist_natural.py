# Multi-objective optimization for adhesive formulation with composition constraint
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


# Domain-specific objective names
MECH_OBJ = "mechanical_performance"  # maximize (e.g., tensile shear strength index)
TIME_OBJ = "processing_time"         # minimize (e.g., total processing time in minutes)


def evaluate_adhesive_formulation(parameters: dict) -> dict:
    """
    Evaluate an adhesive formulation with four components under a composition constraint.
    We reparameterize the equality constraint A + B + C + D = 1.0 by optimizing A, B, C
    and computing D = 1.0 - (A + B + C). All components are fractions in [0, 1].

    This function simulates:
    - mechanical_performance (higher is better) as a function of component fractions with
      a crosslinking optimum and interaction terms.
    - processing_time (lower is better) as a function of solids content and cure behavior.

    Noise model: True (returns known SEM values to Ax).
    Replace this simulation with actual lab measurements or a high-fidelity simulator.
    """
    # Extract primary fractions and compute the dependent fraction to satisfy the composition constraint.
    A = float(parameters["component_A"])  # base polymer fraction
    B = float(parameters["component_B"])  # crosslinker fraction
    C = float(parameters["component_C"])  # plasticizer fraction
    D = 1.0 - (A + B + C)                 # solvent fraction (computed)

    # Numerical safety: clip D into [0, 1] to handle tiny floating rounding; Ax enforces A+B+C <= 1.0.
    D = float(np.clip(D, 0.0, 1.0))

    # Mechanical performance model (unitless index scaled ~0-100):
    # - Strong baseline from base polymer A (sublinear return)
    # - Crosslinker B has an optimum near ~0.18 with Gaussian peak
    # - Plasticizer C and solvent D reduce performance
    # - Positive synergy between A and B, mild penalty for high D with A (dilution)
    gaussian_peak = 38.0 * np.exp(-((B - 0.18) / 0.08) ** 2)
    baseline = 30.0 * (A ** 0.6)
    synergy = 22.0 * A * B
    dilution_penalty = 12.0 * D + 6.0 * C
    mech_perf = baseline + gaussian_peak + synergy - dilution_penalty
    mech_perf = float(np.clip(mech_perf, 0.0, None))

    # Processing time model (minutes):
    # - Increases with solids content and viscosity contributors (A, B), slightly with C
    # - Decreases with solvent D
    # - Additional cure-time penalty when B exceeds ~0.20
    base_time = 25.0
    solids_term = 55.0 * A + 35.0 * B + 10.0 * C
    solvent_effect = -60.0 * D
    cure_penalty = 28.0 * max(0.0, B - 0.20)
    proc_time = base_time + solids_term + solvent_effect + cure_penalty
    proc_time = float(np.clip(proc_time, 5.0, None))

    # Add observational noise; provide SEM to Ax (can be empirically estimated).
    rng = np.random.default_rng()
    noise_sd_mech = 1.5
    noise_sd_time = 0.6
    mech_obs = mech_perf + rng.normal(0.0, noise_sd_mech)
    time_obs = proc_time + rng.normal(0.0, noise_sd_time)

    # Return dict of metrics to (mean, SEM)
    return {
        MECH_OBJ: (float(mech_obs), float(noise_sd_mech)),
        TIME_OBJ: (float(time_obs), float(noise_sd_time)),
    }


# Total fraction for compositional constraint A + B + C + D == 1.0
TOTAL_FRACTION = 1.0

# Create Ax client and experiment
ax_client = AxClient()
ax_client.create_experiment(
    name="adhesive_formulation_moo",
    parameters=[
        # Reparameterization: optimize the first 3 components; the 4th is inferred.
        {"name": "component_A", "type": "range", "bounds": [0.0, 1.0]},  # fraction
        {"name": "component_B", "type": "range", "bounds": [0.0, 1.0]},  # fraction
        {"name": "component_C", "type": "range", "bounds": [0.0, 1.0]},  # fraction
        # component_D will be computed as 1.0 - (A + B + C)
    ],
    objectives={
        MECH_OBJ: ObjectiveProperties(minimize=False),  # maximize mechanical performance
        TIME_OBJ: ObjectiveProperties(minimize=True),   # minimize processing time
    },
    # Linear reparameterized constraint to ensure component_D >= 0.0
    parameter_constraints=[f"component_A + component_B + component_C <= {TOTAL_FRACTION}"],
)


# Run optimization loop
N_TRIALS = 40
for _ in range(N_TRIALS):
    params, trial_index = ax_client.get_next_trial()
    results = evaluate_adhesive_formulation(params)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)

# Fetch results
df = ax_client.get_trials_data_frame()

# Compute empirical Pareto frontier (non-dominated set) from observed data
def compute_pareto_mask(performance: np.ndarray, time_cost: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating which points are Pareto-optimal.
    Maximize performance, minimize time_cost.
    """
    n = performance.shape[0]
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        # A point j dominates i if perf_j >= perf_i and time_j <= time_i with at least one strict.
        dominates = (performance >= performance[i]) & (time_cost <= time_cost[i]) & (
            (performance > performance[i]) | (time_cost < time_cost[i])
        )
        # Exclude self when checking dominance
        dominates[i] = False
        if np.any(dominates):
            mask[i] = False
    return mask


# Extract metric columns
if MECH_OBJ not in df.columns or TIME_OBJ not in df.columns:
    raise RuntimeError("Expected metric columns not found in trials data frame.")

perf_vals = df[MECH_OBJ].to_numpy(dtype=float)
time_vals = df[TIME_OBJ].to_numpy(dtype=float)

pareto_mask = compute_pareto_mask(perf_vals, time_vals)
pareto_df = df.loc[pareto_mask].copy()
pareto_df.sort_values(by=MECH_OBJ, inplace=True)  # sort for plotting a Pareto curve

# Plot observed points and empirical Pareto frontier
fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
ax.scatter(
    df[MECH_OBJ],
    df[TIME_OBJ],
    fc="None",
    ec="tab:gray",
    label="Observed",
)
ax.plot(
    pareto_df[MECH_OBJ],
    pareto_df[TIME_OBJ],
    color="#0033FF",
    lw=2,
    marker="o",
    ms=4,
    label="Empirical Pareto Front",
)

ax.set_xlabel("Mechanical performance (higher is better)")
ax.set_ylabel("Processing time (min, lower is better)")
ax.set_title("Adhesive formulation: Pareto trade-off")
ax.grid(True, alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()