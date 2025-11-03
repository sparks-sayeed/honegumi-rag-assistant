# Generated from Honegumi skeleton and adapted to multi-objective lead optimization
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


# Objective names
AFFINITY = "target_affinity"           # higher is better
SOLUBILITY = "aqueous_solubility"      # higher is better
TOXICITY = "toxicity"                  # lower is better


def evaluate_molecule(
    molecular_weight: float,
    logP: float,
    hydrogen_bond_donors: float,
    hydrogen_bond_acceptors: float,
):
    """
    Pseudo-QSAR surrogate used as an executable stand-in for real measurements.

    Replace this function with your actual scoring pipelines:
    - Docking / binding free-energy prediction (affinity)
    - Solubility prediction (e.g., ESOL-like models)
    - Toxicity prediction (e.g., toxicity classification / LD50 regression)
    The function must return a dict: metric_name -> (mean_value, standard_error)

    Design heuristics encoded below:
    - Target affinity: increases with moderate logP and moderate MW; too many HBD/HBA penalize.
    - Aqueous solubility: decreases with logP and MW; increases with HBD/HBA.
    - Toxicity: increases with higher logP, higher MW, and excessive HBD/HBA.
    """
    # Affinity components
    affinity_logp_term = 6.0 * np.exp(-0.5 * ((logP - 2.2) / 0.9) ** 2)
    affinity_mw_term = 3.0 / (1.0 + np.exp(-(molecular_weight - 280.0) / 60.0))
    affinity_penalty_hbd = 0.4 * max(0.0, hydrogen_bond_donors - 2.0)
    affinity_penalty_hba = 0.25 * max(0.0, hydrogen_bond_acceptors - 4.0)
    affinity = affinity_logp_term + affinity_mw_term - affinity_penalty_hbd - affinity_penalty_hba
    affinity = float(np.clip(affinity, 0.0, 10.0))

    # Solubility components
    sol_logp_term = 6.0 * np.exp(-0.5 * ((logP + 0.2) / 1.3) ** 2)
    sol_mw_term = 2.0 * np.exp(-0.5 * ((molecular_weight - 220.0) / 80.0) ** 2)
    sol_hbond_term = 0.4 * hydrogen_bond_donors + 0.2 * hydrogen_bond_acceptors
    solubility = sol_logp_term + sol_mw_term + sol_hbond_term
    solubility = float(np.clip(solubility, 0.0, 12.0))

    # Toxicity components (lower is better)
    tox_logp = 0.4 * max(0.0, logP - 1.5) + 0.3 * max(0.0, logP - 3.0) ** 1.2
    tox_mw = 0.0025 * max(0.0, molecular_weight - 200.0)
    tox_hbond = 0.06 * max(0.0, hydrogen_bond_donors - 1.0) + 0.04 * max(0.0, hydrogen_bond_acceptors - 3.0)
    toxicity = tox_logp + tox_mw + tox_hbond
    toxicity = float(np.clip(toxicity, 0.0, 10.0))

    # Add measurement noise; return standard errors to indicate noise model is present.
    rng = np.random.default_rng()
    sd_aff, sd_sol, sd_tox = 0.3, 0.3, 0.1
    affinity_obs = float(affinity + rng.normal(0.0, sd_aff))
    solubility_obs = float(solubility + rng.normal(0.0, sd_sol))
    toxicity_obs = float(max(0.0, toxicity + rng.normal(0.0, sd_tox)))

    return {
        AFFINITY: (affinity_obs, sd_aff),
        SOLUBILITY: (solubility_obs, sd_sol),
        TOXICITY: (toxicity_obs, sd_tox),
    }


# Seed for reproducibility of the synthetic evaluator
np.random.seed(42)

ax_client = AxClient()

# Create the experiment in "Lipinski space" with drug-likeness constraints.
ax_client.create_experiment(
    name="lead_compound_multiobjective_optimization",
    parameters=[
        # Bounds chosen to target drug-like small molecules within Lipinski space.
        {"name": "molecular_weight", "type": "range", "bounds": [150.0, 500.0], "value_type": "float"},
        {"name": "logP", "type": "range", "bounds": [-1.0, 5.0], "value_type": "float"},
        {"name": "hydrogen_bond_donors", "type": "range", "bounds": [0.0, 5.0], "value_type": "float"},
        {"name": "hydrogen_bond_acceptors", "type": "range", "bounds": [0.0, 10.0], "value_type": "float"},
    ],
    objectives={
        AFFINITY: ObjectiveProperties(minimize=False),   # Maximize target affinity
        SOLUBILITY: ObjectiveProperties(minimize=False), # Maximize aqueous solubility
        TOXICITY: ObjectiveProperties(minimize=True),    # Minimize toxicity
    },
    # Lipinski rule of five upper-bound constraints (redundant with bounds, but explicit as linear constraints).
    parameter_constraints=[
        "molecular_weight <= 500.0",
        "logP <= 5.0",
        "hydrogen_bond_donors <= 5.0",
        "hydrogen_bond_acceptors <= 10.0",
    ],
)

# Optimization loop
num_trials = 40
for _ in range(num_trials):
    parameters, trial_index = ax_client.get_next_trial()
    try:
        mw = parameters["molecular_weight"]
        lp = parameters["logP"]
        hbd = parameters["hydrogen_bond_donors"]
        hba = parameters["hydrogen_bond_acceptors"]
        results = evaluate_molecule(mw, lp, hbd, hba)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)
    except Exception:
        ax_client.log_trial_failure(trial_index=trial_index)

# Retrieve Pareto-optimal candidates using observed data
pareto = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)

# Convert Pareto metrics to a DataFrame for plotting
pareto_rows = []
for _, (params, metrics) in pareto.items():
    pareto_rows.append(
        {
            AFFINITY: metrics[AFFINITY][0],
            SOLUBILITY: metrics[SOLUBILITY][0],
            TOXICITY: metrics[TOXICITY][0],
            **params,
        }
    )
pareto_df = pd.DataFrame(pareto_rows) if len(pareto) > 0 else pd.DataFrame(columns=[AFFINITY, SOLUBILITY, TOXICITY])

# Collate all observed trials
df = ax_client.get_trials_data_frame()

# Basic pairwise visualization of the 3-objective trade-offs with Pareto highlights
fig, axes = plt.subplots(1, 3, figsize=(16, 4), dpi=150)

# 1) Affinity vs Solubility (both maximize)
axes[0].scatter(df[AFFINITY], df[SOLUBILITY], fc="None", ec="k", label="Observed")
if not pareto_df.empty:
    axes[0].scatter(pareto_df[AFFINITY], pareto_df[SOLUBILITY], marker="*", s=120, c="#d62728", ec="k", label="Pareto")
axes[0].set_xlabel("Target Affinity (higher is better)")
axes[0].set_ylabel("Aqueous Solubility (higher is better)")
axes[0].legend()

# 2) Affinity vs Toxicity (toxicity minimize)
axes[1].scatter(df[AFFINITY], df[TOXICITY], fc="None", ec="k", label="Observed")
if not pareto_df.empty:
    axes[1].scatter(pareto_df[AFFINITY], pareto_df[TOXICITY], marker="*", s=120, c="#d62728", ec="k", label="Pareto")
axes[1].set_xlabel("Target Affinity (higher is better)")
axes[1].set_ylabel("Toxicity (lower is better)")
axes[1].legend()

# 3) Solubility vs Toxicity (toxicity minimize)
axes[2].scatter(df[SOLUBILITY], df[TOXICITY], fc="None", ec="k", label="Observed")
if not pareto_df.empty:
    axes[2].scatter(pareto_df[SOLUBILITY], pareto_df[TOXICITY], marker="*", s=120, c="#d62728", ec="k", label="Pareto")
axes[2].set_xlabel("Aqueous Solubility (higher is better)")
axes[2].set_ylabel("Toxicity (lower is better)")
axes[2].legend()

plt.tight_layout()
plt.show()