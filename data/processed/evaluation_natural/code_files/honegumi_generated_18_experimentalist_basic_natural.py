# Generated for optimizing a 3-resin + hardener adhesive formulation using Ax (Bayesian MOO)
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient, ObjectiveProperties


# Reproducibility for the synthetic evaluator below
RNG = np.random.RandomState(42)


def evaluate_adhesive_formulation(parameterization: dict) -> dict:
    """
    Evaluate an adhesive formulation defined by resin fractions and an implied hardener fraction.

    Parameters:
      parameterization: dict with keys:
        - 'resin_A_fraction'
        - 'resin_B_fraction'
        - 'resin_C_fraction'
      The hardener fraction is computed as: 1 - (A + B + C), and clipped to [0, 1].

    Returns:
      dict mapping objective names to (mean, sem) tuples:
        - 'bond_strength_mpa' (maximize)
        - 'cure_time_minutes' (minimize)

    Notes:
      This function provides a physics-inspired synthetic model with noise, to allow
      running the example end-to-end. Replace with actual experimental measurement
      code to use in production.
    """
    A = float(parameterization["resin_A_fraction"])
    B = float(parameterization["resin_B_fraction"])
    C = float(parameterization["resin_C_fraction"])
    # Composition constraint: fractions sum to 1. Compute hardener as the remainder.
    H = max(0.0, 1.0 - (A + B + C))
    # Slight numerical safety
    H = min(1.0, H)

    # Synthetic domain logic (plausible trends):
    # - Resin A: boosts bond strength strongly but slows cure.
    # - Resin B: moderate boost to strength with moderate cure time.
    # - Resin C: reduces cure time but can weaken strength at high levels.
    # - Hardener: there is an optimal stoichiometric window; deviation hurts strength and increases cure time.

    # Hardener stoichiometric "optimum" depends weakly on resin mix
    H_opt = 0.25 + 0.10 * A - 0.05 * C
    H_opt = np.clip(H_opt, 0.05, 0.5)
    sigma_h = 0.07  # width of the optimal window

    # Bond strength model (MPa)
    base_strength = 18.0 + 28.0 * A + 18.0 * B + 8.0 * C
    synergy_AB = 18.0 * np.sqrt(max(0.0, A * B))
    hardener_fit = 22.0 * np.exp(-((H - H_opt) / sigma_h) ** 2) * (0.6 + 0.4 * A)
    penalty_C = -16.0 * (C ** 1.5)
    penalty_total_solids_extremes = -6.0 * ((A + B + C) - 0.8) ** 2  # penalize away from ~0.8 resins
    s_mpa = base_strength + synergy_AB + hardener_fit + penalty_C + penalty_total_solids_extremes

    # Cure time model (minutes): lower is better
    # Faster with more hardener and more resin C; slower with more resin A
    t_min = 85.0 - 120.0 * H - 35.0 * C + 28.0 * A - 10.0 * (B - 0.3)
    # Penalize deviation from stoichiometry (slower out of the window)
    t_min += 45.0 * ((H - H_opt) / 0.09) ** 2

    # Clamp to reasonable bounds
    s_mpa = float(np.clip(s_mpa, 2.0, 70.0))
    t_min = float(np.clip(t_min, 3.0, 180.0))

    # Add measurement noise (5-8% relative), and provide SEM
    strength_noise_sd = max(0.5, 0.06 * s_mpa)
    time_noise_sd = max(0.5, 0.08 * t_min)
    s_obs = float(s_mpa + RNG.normal(0.0, strength_noise_sd))
    t_obs = float(t_min + RNG.normal(0.0, time_noise_sd))

    # Clip observed values again to remain in feasible ranges
    s_obs = float(np.clip(s_obs, 2.0, 80.0))
    t_obs = float(np.clip(t_obs, 2.0, 240.0))

    # Return mean and SEM (set SEM to noise SD as a simple estimate)
    return {
        "bond_strength_mpa": (s_obs, strength_noise_sd),
        "cure_time_minutes": (t_obs, time_noise_sd),
    }


# Create Ax client and experiment
ax_client = AxClient()

ax_client.create_experiment(
    name="adhesive_formulation_optimization",
    parameters=[
        {
            "name": "resin_A_fraction",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "resin_B_fraction",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "resin_C_fraction",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
    ],
    # Reparameterized compositional constraint: A + B + C <= 1, hardener is 1 - (A+B+C)
    parameter_constraints=[
        "resin_A_fraction + resin_B_fraction + resin_C_fraction <= 1.0",
    ],
    objectives={
        "bond_strength_mpa": ObjectiveProperties(minimize=False),  # maximize
        "cure_time_minutes": ObjectiveProperties(minimize=True),   # minimize
    },
)


# Run sequential optimization (budget: ~40)
NUM_TRIALS = 40
for i in range(NUM_TRIALS):
    parameterization, trial_index = ax_client.get_next_trial()
    results = evaluate_adhesive_formulation(parameterization)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)

# Fetch observed data
experiment = ax_client.experiment
data = experiment.fetch_data()
df_metrics = data.df  # columns: ['arm_name', 'metric_name', 'mean', 'sem', 'trial_index']

# Pivot to get metrics per trial
pivot = (
    df_metrics.pivot_table(
        index=["trial_index", "arm_name"], columns="metric_name", values="mean", aggfunc="last"
    )
    .reset_index()
    .rename_axis(None, axis=1)
)

# Compute Pareto set (bond_strength_mpa: maximize, cure_time_minutes: minimize)
def is_dominated(i_row, others):
    s_i = i_row["bond_strength_mpa"]
    t_i = i_row["cure_time_minutes"]
    for _, r in others.iterrows():
        s_j = r["bond_strength_mpa"]
        t_j = r["cure_time_minutes"]
        if (s_j >= s_i and t_j <= t_i) and (s_j > s_i or t_j < t_i):
            return True
    return False


pareto_mask = []
for idx, row in pivot.iterrows():
    others = pivot.drop(index=idx)
    pareto_mask.append(not is_dominated(row, others))
pivot["is_pareto"] = pareto_mask

# Plot observed points and highlight Pareto front
fig, ax = plt.subplots(figsize=(7.0, 5.0), dpi=140)
ax.scatter(
    pivot["bond_strength_mpa"],
    pivot["cure_time_minutes"],
    s=30,
    alpha=0.65,
    edgecolor="k",
    facecolor="#AAAAAA",
    label="Observed",
)

pareto_pts = pivot[pivot["is_pareto"]].copy()
# Sort Pareto points by cure time ascending (for a visually coherent connection)
pareto_pts = pareto_pts.sort_values("cure_time_minutes")
ax.scatter(
    pareto_pts["bond_strength_mpa"],
    pareto_pts["cure_time_minutes"],
    s=55,
    edgecolor="k",
    facecolor="#1f77b4",
    label="Pareto-optimal",
)
ax.plot(
    pareto_pts["bond_strength_mpa"],
    pareto_pts["cure_time_minutes"],
    color="#1f77b4",
    lw=1.8,
)

ax.set_xlabel("Bond strength (MPa)")
ax.set_ylabel("Cure time (minutes)")
ax.set_title("Adhesive formulation: Observations and Pareto frontier")
ax.legend(loc="best")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()