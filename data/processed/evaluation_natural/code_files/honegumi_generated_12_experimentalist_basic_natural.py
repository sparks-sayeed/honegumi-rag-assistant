# Generated from Honegumi skeleton and adapted for ceramic binder formulation optimization
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# ============================
# Problem-specific definitions
# ============================

# Objective metric names: maximize green strength for each ceramic system
METRIC_ALUMINA = "green_strength_alumina_mpa"
METRIC_ZIRCONIA = "green_strength_zirconia_mpa"
METRIC_SIC = "green_strength_silicon_carbide_mpa"

# Trial budget (approx. 40 per ceramic type)
TOTAL_TRIALS = 120

rng = np.random.default_rng(seed=42)


def _strength_response(
    x: Dict[str, float],
    mu: Dict[str, float],
    sigma: Dict[str, float],
    scale: float = 30.0,
    baseline: float = 2.0,
    noise_sd: float = 0.8,
) -> Tuple[float, float]:
    """
    Generic smooth response surface around a ceramic-specific optimum (mu),
    with dimension-wise sensitivities (sigma). Returns (mean, sem).
    """
    # Compute a radial basis response around the optimum
    dist2 = 0.0
    for k in x:
        # Avoid zero sigma
        s = max(sigma.get(k, 1.0), 1e-6)
        dist2 += ((x[k] - mu.get(k, x[k])) / s) ** 2

    response = baseline + scale * np.exp(-0.5 * dist2)

    # Heuristic penalties for unrealistic regions seen in binder systems
    binder = x["binder_wt_percent"]
    plast = x["plasticizer_to_binder_ratio"]
    disp = x["dispersant_wt_percent"]
    polar = x["polar_solvent_fraction"]
    mix = x["mixing_time_minutes"]
    mw = x["binder_mw_kDa"]

    penalty = 1.0
    if binder > 6.5:
        penalty *= 0.88  # too much binder leads to brittleness / poor packing
    if plast > 0.7:
        penalty *= 0.9  # over-plasticized bodies lose cohesive strength
    if disp > 1.0:
        penalty *= 0.92  # over-dispersion can destabilize slurry networks
    if mix < 20:
        penalty *= 0.9  # insufficient mixing
    if polar < 0.05 or polar > 0.95:
        penalty *= 0.95  # very extreme solvent blends often problematic
    if mw < 30:
        penalty *= 0.9  # very low MW binders may not form strong networks

    mean_strength = max(0.0, response * penalty)

    # Add observation noise and provide SEM representative of measurement uncertainty
    observed = mean_strength + rng.normal(0.0, noise_sd)
    sem = noise_sd  # if known; otherwise set to None

    return float(max(0.0, observed)), float(sem)


def evaluate_binder_formulation(parameters: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate the binder formulation for green strength on alumina, zirconia, and silicon carbide.
    Replace this surrogate with your experimental measurements when integrating with the lab.

    Inputs (parameters dict):
      - binder_wt_percent: Binder mass fraction relative to ceramic powder (wt%), [0.5, 8.0]
      - plasticizer_to_binder_ratio: Mass ratio of plasticizer to binder, [0.0, 1.0]
      - dispersant_wt_percent: Dispersant mass fraction relative to powder (wt%), [0.0, 1.5]
      - polar_solvent_fraction: Fraction of polar solvent in blend (e.g., IPA/EtOH) [0.0, 1.0]
      - mixing_time_minutes: Total high-shear mixing time in minutes, [10, 120]
      - binder_mw_kDa: Binder molecular weight in kDa, [25, 200]

    Return:
      Dict of metric name -> (mean, sem)
    """
    # Extract and normalize parameters
    x = {
        "binder_wt_percent": float(parameters["binder_wt_percent"]),
        "plasticizer_to_binder_ratio": float(parameters["plasticizer_to_binder_ratio"]),
        "dispersant_wt_percent": float(parameters["dispersant_wt_percent"]),
        "polar_solvent_fraction": float(parameters["polar_solvent_fraction"]),
        "mixing_time_minutes": float(parameters["mixing_time_minutes"]),
        "binder_mw_kDa": float(parameters["binder_mw_kDa"]),
    }

    # System-specific best-response locations (mu) and tolerances (sigma)
    # These encode that the same binder must perform across different powder chemistries.
    mu_alumina = {
        "binder_wt_percent": 3.5,
        "plasticizer_to_binder_ratio": 0.25,
        "dispersant_wt_percent": 0.50,
        "polar_solvent_fraction": 0.60,
        "mixing_time_minutes": 60.0,
        "binder_mw_kDa": 80.0,
    }
    mu_zirconia = {
        "binder_wt_percent": 3.0,
        "plasticizer_to_binder_ratio": 0.35,
        "dispersant_wt_percent": 0.60,
        "polar_solvent_fraction": 0.50,
        "mixing_time_minutes": 75.0,
        "binder_mw_kDa": 100.0,
    }
    mu_sic = {
        "binder_wt_percent": 4.0,
        "plasticizer_to_binder_ratio": 0.20,
        "dispersant_wt_percent": 0.40,
        "polar_solvent_fraction": 0.70,
        "mixing_time_minutes": 90.0,
        "binder_mw_kDa": 120.0,
    }
    sigma_common = {
        "binder_wt_percent": 1.0,
        "plasticizer_to_binder_ratio": 0.15,
        "dispersant_wt_percent": 0.30,
        "polar_solvent_fraction": 0.20,
        "mixing_time_minutes": 20.0,
        "binder_mw_kDa": 40.0,
    }

    alumina_val, alumina_sem = _strength_response(x, mu_alumina, sigma_common, scale=28.0, baseline=3.0, noise_sd=0.7)
    zirconia_val, zirconia_sem = _strength_response(x, mu_zirconia, sigma_common, scale=32.0, baseline=2.5, noise_sd=0.9)
    sic_val, sic_sem = _strength_response(x, mu_sic, sigma_common, scale=30.0, baseline=2.0, noise_sd=0.8)

    return {
        METRIC_ALUMINA: (alumina_val, alumina_sem),
        METRIC_ZIRCONIA: (zirconia_val, zirconia_sem),
        METRIC_SIC: (sic_val, sic_sem),
    }


# ============================
# Set up Ax optimization
# ============================

ax_client = AxClient()

ax_client.create_experiment(
    name="ceramic_binder_formulation_moo",
    parameters=[
        {
            "name": "binder_wt_percent",
            "type": "range",
            "bounds": [0.5, 8.0],
            "value_type": "float",
        },
        {
            "name": "plasticizer_to_binder_ratio",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "dispersant_wt_percent",
            "type": "range",
            "bounds": [0.0, 1.5],
            "value_type": "float",
        },
        {
            "name": "polar_solvent_fraction",
            "type": "range",
            "bounds": [0.0, 1.0],
            "value_type": "float",
        },
        {
            "name": "mixing_time_minutes",
            "type": "range",
            "bounds": [10.0, 120.0],
            "value_type": "float",
        },
        {
            "name": "binder_mw_kDa",
            "type": "range",
            "bounds": [25.0, 200.0],
            "value_type": "float",
        },
    ],
    objectives={
        METRIC_ALUMINA: ObjectiveProperties(minimize=False),
        METRIC_ZIRCONIA: ObjectiveProperties(minimize=False),
        METRIC_SIC: ObjectiveProperties(minimize=False),
    },
)


# ============================
# Optimization loop
# ============================

for i in range(TOTAL_TRIALS):
    parameterization, trial_index = ax_client.get_next_trial()
    results = evaluate_binder_formulation(parameterization)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)


# ============================
# Pareto analysis and plotting
# ============================

def pareto_mask_maximize(Y: np.ndarray) -> np.ndarray:
    """
    Compute non-dominated (Pareto-optimal) mask for maximizing all objectives.
    Y: (n_samples, n_objectives)
    Returns boolean array of shape (n_samples,), True if non-dominated.
    """
    n = Y.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Any point that is dominated by point i should be marked inefficient
        is_dominated = np.all(Y[i] >= Y[is_efficient], axis=1) & np.any(Y[i] > Y[is_efficient], axis=1)
        # Shift indices to the global mask
        idx = np.where(is_efficient)[0]
        is_efficient[idx[is_dominated]] = False
    return is_efficient


# Extract observed data
df = ax_client.get_trials_data_frame()
metrics = [METRIC_ALUMINA, METRIC_ZIRCONIA, METRIC_SIC]
df = df.dropna(subset=metrics).reset_index(drop=True)

if len(df) > 0:
    Y = df[metrics].values.astype(float)
    pareto_mask = pareto_mask_maximize(Y)

    # Print top Pareto-optimal formulations
    print("\nPareto-optimal binder formulations (observed):")
    pareto_df = df.loc[pareto_mask].copy()
    # Keep parameter columns
    param_cols = ["binder_wt_percent", "plasticizer_to_binder_ratio", "dispersant_wt_percent",
                  "polar_solvent_fraction", "mixing_time_minutes", "binder_mw_kDa"]
    cols_to_show = param_cols + metrics
    print(pareto_df[cols_to_show].sort_values(by=metrics, ascending=False).head(10).to_string(index=False))

    # Pairwise scatter plots of the three green strength metrics
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), dpi=150)
    pairs = [
        (METRIC_ALUMINA, METRIC_ZIRCONIA),
        (METRIC_ALUMINA, METRIC_SIC),
        (METRIC_ZIRCONIA, METRIC_SIC),
    ]
    for ax, (m1, m2) in zip(axes, pairs):
        ax.scatter(df[m1], df[m2], fc="None", ec="#555555", label="Observed", alpha=0.6)
        ax.scatter(df.loc[pareto_mask, m1], df.loc[pareto_mask, m2], c="#0033FF", label="Pareto", s=28)
        ax.set_xlabel(m1.replace("_", " "))
        ax.set_ylabel(m2.replace("_", " "))
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Pairwise green strength trade-offs (MPa)")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.show()
else:
    print("No data to plot.")