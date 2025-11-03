# Generated from Honegumi skeleton and adapted to steel alloy design with Ax (https://ax.dev)
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Domain-specific metric names
strength_metric = "tensile_strength_mpa"
corrosion_metric = "pitting_resistance_index"


def evaluate_steel_alloy(cr: float, ni: float, mo: float, fe: float) -> dict:
    """
    Evaluate a steel alloy candidate with fractions of Cr, Ni, Mo, Fe that sum to 1.0.

    This is a realistic synthetic model intended to mimic trends seen in stainless steels:
    - Tensile strength generally increases with Mo, Cr, and Ni additions (solid solution strengthening),
      with diminishing returns and penalties for excessive total alloying (embrittlement/cost).
    - Corrosion resistance (pitting) improves with Cr and Mo and is supported by Ni; includes synergy between Cr and Mo.

    Returns a dict mapping metric names to (mean, SEM), modeling experimental noise.
    """
    # Guard against tiny floating inaccuracies in Fe
    fe = max(0.0, fe)

    # Total alloying additions (non-Fe)
    total_alloy = cr + ni + mo

    # Tensile strength model (MPa)
    # Base strength for Fe-rich baseline
    base_strength = 420.0
    # Contributions from elements (diminishing returns via sqrt)
    add_strength = (
        900.0 * np.sqrt(max(mo, 0.0))  # Mo has strong effect
        + 350.0 * np.sqrt(max(cr, 0.0))
        + 180.0 * np.sqrt(max(ni, 0.0))
    )
    # Penalty for excessive alloying content (brittleness/cost/processability)
    penalty_strength = 0.0
    if total_alloy > 0.40:
        penalty_strength = 400.0 * (total_alloy - 0.40)

    tensile_strength = base_strength + add_strength - penalty_strength

    # Corrosion (pitting) resistance index model (dimensionless index, larger is better)
    # PRE-like combination with synergy
    corrosion_base = 5.0
    corrosion_linear = 22.0 * cr + 55.0 * mo + 12.0 * ni
    corrosion_synergy = 18.0 * cr * mo
    # Low-alloy penalty: if Cr is very low, penalize
    low_cr_penalty = 0.0
    if cr < 0.10:
        low_cr_penalty = 30.0 * (0.10 - cr)

    pitting_index = corrosion_base + corrosion_linear + corrosion_synergy - low_cr_penalty

    # Add observation noise to simulate test variability (SEM provided)
    rng = np.random.default_rng()
    strength_noise_sd = 8.0
    corrosion_noise_sd = 1.0

    tensile_strength_noisy = float(tensile_strength + rng.normal(0.0, strength_noise_sd))
    pitting_index_noisy = float(pitting_index + rng.normal(0.0, corrosion_noise_sd))

    return {
        strength_metric: (tensile_strength_noisy, strength_noise_sd),
        corrosion_metric: (pitting_index_noisy, corrosion_noise_sd),
    }


# Compositional total (mass fractions)
total_fraction = 1.0

# Practical maximums (mass fraction) for alloying additions
CR_MAX = 0.30  # ~30 wt% upper bound for Cr
NI_MAX = 0.25  # ~25 wt% upper bound for Ni
MO_MAX = 0.08  # ~8 wt% upper bound for Mo

# Initialize Ax client
ax_client = AxClient()

# Create experiment: optimize Cr, Ni, Mo; Fe is the balance (computed, not optimized)
ax_client.create_experiment(
    name="steel_alloy_strength_corrosion_moo",
    parameters=[
        {"name": "Cr", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "Ni", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "Mo", "type": "range", "bounds": [0.0, 1.0]},
    ],
    objectives={
        strength_metric: ObjectiveProperties(minimize=False),
        corrosion_metric: ObjectiveProperties(minimize=False),
    },
    # Constraints:
    # - Compositional: Cr + Ni + Mo <= 1.0 (Fe is balance >= 0)
    # - Upper bounds for Cr, Ni, Mo per practical processing/spec limits
    parameter_constraints=[
        f"Cr + Ni + Mo <= {total_fraction}",
        f"Cr <= {CR_MAX}",
        f"Ni <= {NI_MAX}",
        f"Mo <= {MO_MAX}",
    ],
)

# Budget: 40 trials
N_TRIALS = 40

for i in range(N_TRIALS):
    params, trial_index = ax_client.get_next_trial()

    # Extract alloying additions
    cr = float(params["Cr"])
    ni = float(params["Ni"])
    mo = float(params["Mo"])
    # Compute Fe as balance to enforce compositional sum to 1.0
    fe = total_fraction - (cr + ni + mo)

    results = evaluate_steel_alloy(cr, ni, mo, fe)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)

# Retrieve Pareto-optimal parameterizations
pareto_params = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)

# Plot observed outcomes and Pareto front
objectives = ax_client.objective_names
df = ax_client.get_trials_data_frame()

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
pareto = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)
pareto_data = [p[1][0] for p in pareto.values()]
pareto_df = pd.DataFrame(pareto_data).sort_values(objectives[0])

ax.scatter(df[objectives[0]], df[objectives[1]], fc="None", ec="k", label="Observed")
if not pareto_df.empty:
    ax.plot(
        pareto_df[objectives[0]],
        pareto_df[objectives[1]],
        color="#0033FF",
        lw=2,
        label="Pareto Front",
    )
ax.set_xlabel(objectives[0])
ax.set_ylabel(objectives[1])
ax.set_title("Steel Alloy Design: Strength vs. Corrosion Resistance")

ax.legend()
plt.tight_layout()
plt.show()