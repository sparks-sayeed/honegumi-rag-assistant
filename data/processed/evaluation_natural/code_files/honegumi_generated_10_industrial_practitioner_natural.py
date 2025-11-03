# Generated for multi-objective drug lead optimization using Ax Platform
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Objective metric names
TARGET_BINDING = "target_binding"  # Higher is better (dimensionless binding score)
SOLUBILITY = "solubility"          # Higher is better (mg/mL)
TOXICITY = "toxicity"              # Lower is better (dimensionless toxicity score)


_rng = np.random.default_rng(42)


def evaluate_molecule(molecular_weight: float, logP: float) -> dict:
    """Simulate evaluation of a small-molecule candidate.

    This is a realistic in-silico proxy for real assays, capturing typical structure-property trends:
    - Binding affinity generally benefits from moderate MW and moderate-to-high lipophilicity (logP),
      but degrades when MW is too low/high or logP is extreme.
    - Solubility decreases with higher logP and higher MW.
    - Toxicity risk increases with higher logP and higher MW.

    Noise is added to emulate experimental measurement noise and biological variability.
    Replace this with actual lab or simulation results when available.

    Args:
        molecular_weight: Molecular weight in Daltons (0–500 per bounds).
        logP: Octanol/water partition coefficient (0–5 per bounds).

    Returns:
        Dict of metric_name -> observed value (floats). Ax will infer noise.
    """
    # Target binding: product of MW "sweet spot" and logP effect, scaled to ~[0, 1.2]
    mw_opt = 380.0
    mw_width = 90.0
    mw_effect = np.exp(-0.5 * ((molecular_weight - mw_opt) / mw_width) ** 2)  # 0..1
    logp_effect = 1.0 / (1.0 + np.exp(-1.2 * (logP - 2.5)))  # 0..1
    binding = 1.2 * mw_effect * logp_effect
    binding += _rng.normal(0.0, 0.05)  # additive noise
    binding = float(np.clip(binding, 0.0, None))

    # Solubility (mg/mL): exponential decay with MW and logP, realistic 0–150+ mg/mL range
    # log(Solubility) ~ 5.0 - 1.0*logP - 0.0035*MW (natural log)
    log_sol = 5.0 - 1.0 * logP - 0.0035 * molecular_weight
    sol = float(np.exp(log_sol))
    # multiplicative noise (log-normal)
    sol *= float(np.exp(_rng.normal(0.0, 0.12)))
    sol = float(np.clip(sol, 0.0, 200.0))

    # Toxicity risk score: logistic increase with MW and logP, roughly in [0, 1]
    tox_mw = 1.0 / (1.0 + np.exp(-(molecular_weight - 350.0) / 60.0))
    tox_lip = 1.0 / (1.0 + np.exp(-(logP - 2.8) / 0.6))
    tox = 0.55 * tox_mw + 0.45 * tox_lip
    tox += _rng.normal(0.0, 0.03)
    tox = float(np.clip(tox, 0.0, 1.2))

    return {
        TARGET_BINDING: binding,
        SOLUBILITY: sol,
        TOXICITY: tox,
    }


# Initialize Ax client
ax_client = AxClient()

# Define experiment: drug-like design within MW and logP ranges; optimize binding↑, solubility↑, toxicity↓
ax_client.create_experiment(
    name="drug_lead_optimization",
    parameters=[
        {"name": "molecular_weight", "type": "range", "bounds": [0.0, 500.0]},
        {"name": "logP", "type": "range", "bounds": [0.0, 5.0]},
    ],
    objectives={
        TARGET_BINDING: ObjectiveProperties(minimize=False),
        SOLUBILITY: ObjectiveProperties(minimize=False),
        TOXICITY: ObjectiveProperties(minimize=True),
    },
)

# Run optimization loop for up to 50 experimental candidates
N_TRIALS = 50
for _ in range(N_TRIALS):
    parameters, trial_index = ax_client.get_next_trial()

    mw = float(parameters["molecular_weight"])
    lp = float(parameters["logP"])

    results = evaluate_molecule(mw, lp)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)

# Retrieve Pareto-optimal set (observed)
pareto_map = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)

# Build a DataFrame of all trials
df = ax_client.get_trials_data_frame()
objectives = list(ax_client.objective_names)  # [TARGET_BINDING, SOLUBILITY, TOXICITY]

# Mark Pareto points
pareto_arm_names = set(pareto_map.keys())
if "arm_name" in df.columns:
    df["is_pareto"] = df["arm_name"].isin(pareto_arm_names)
else:
    # Fallback: if arm_name not present, mark all as non-Pareto to avoid errors
    df["is_pareto"] = False

# Plot pairwise objective tradeoffs with Pareto highlights
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150)
pairs = [
    (TARGET_BINDING, SOLUBILITY),
    (TARGET_BINDING, TOXICITY),
    (SOLUBILITY, TOXICITY),
]
titles = [
    "Binding vs Solubility (maximize both)",
    "Binding vs Toxicity (maximize, minimize)",
    "Solubility vs Toxicity (maximize, minimize)",
]

for ax, (x_metric, y_metric), title in zip(axes, pairs, titles):
    if x_metric in df.columns and y_metric in df.columns:
        ax.scatter(
            df[x_metric], df[y_metric], c="#999999", alpha=0.7, label="Observed"
        )
        if "is_pareto" in df.columns and df["is_pareto"].any():
            pareto_df = df[df["is_pareto"]]
            ax.scatter(
                pareto_df[x_metric],
                pareto_df[y_metric],
                c="#0033FF",
                alpha=0.9,
                label="Pareto-optimal",
            )
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        ax.set_title(title)
        ax.legend()

plt.tight_layout()
plt.show()