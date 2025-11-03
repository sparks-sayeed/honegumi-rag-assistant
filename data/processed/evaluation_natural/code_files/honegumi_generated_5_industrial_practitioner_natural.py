# Generated and adapted for pharmaceutical formulation optimization using Ax Platform
# %pip install ax-platform==0.4.3 matplotlib

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List

from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain-specific objective names
BIOAVAILABILITY = "bioavailability"
SHELF_STABILITY = "shelf_stability"

# Parameter names (4 explicit excipients; 5th is derived to enforce sum-to-1.0)
MCC = "mcc_fraction"                    # microcrystalline cellulose (filler)
HPMC = "hpmc_fraction"                  # hydroxypropyl methylcellulose (binder/matrix former)
CCS = "ccs_fraction"                    # croscarmellose sodium (disintegrant)
SLS = "sls_fraction"                    # sodium lauryl sulfate (surfactant)
PH = "pH"
STORAGE_TEMP = "storage_temperature_c"  # storage temperature in Celsius

# Bounds for pH and storage temperature (domain-reasonable)
PH_LOWER, PH_UPPER = 2.0, 9.0
TEMP_LOWER_C, TEMP_UPPER_C = 2.0, 40.0

# Experimental budget and batch size
N_TRIALS = 50
BATCH_SIZE = 1

# Noise level (SEM) for measurements
MEASUREMENT_SEM = 2.5


def simulate_formulation_metrics(
    mcc: float,
    hpmc: float,
    ccs: float,
    sls: float,
    mannitol: float,
    ph: float,
    temp_c: float,
    rng: np.random.Generator,
) -> Dict[str, Tuple[float, float]]:
    """
    Synthetic but domain-inspired simulator for bioavailability and shelf stability.

    Notes:
    - Bioavailability increases with disintegrant (CCS) and surfactant (SLS),
      is penalized by excess binder (HPMC), weakly boosted by mannitol and slightly by MCC,
      and is best near an intestinal pH (~6.5).
    - Shelf stability degrades at higher storage temperatures, is improved by HPMC and MCC,
      and is worsened by SLS and very acidic or very basic pH.
    - All scores are scaled to [0, 100] and clipped. Random Gaussian noise added.
    """
    # Sanity clamp the derived component in case of tiny numerical drift
    mannitol = max(0.0, min(1.0, mannitol))

    # pH preference for bioavailability around 6.5 (Gaussian-like)
    ph_effect_bio = math.exp(-0.5 * ((ph - 6.5) / 1.5) ** 2)

    # Surfactant and disintegrant promote dissolution; binder penalizes if excessive
    # Filler components (MCC, mannitol) have smaller effects on bioavailability
    raw_bio = (
        0.40 * ccs
        + 0.50 * sls
        - 0.20 * hpmc
        + 0.10 * mannitol
        + 0.05 * mcc
        + 0.30 * ph_effect_bio
    )
    bioavailability = 100.0 * np.clip(raw_bio, 0.0, 1.0)

    # Stability:
    # Temperature effect: lower temperature better (sigmoid centered near 25C)
    stability_temp = 1.0 / (1.0 + math.exp(0.25 * (temp_c - 25.0)))
    # pH stability prefers near neutral (7.0)
    stability_ph = math.exp(-0.5 * ((ph - 7.0) / 2.0) ** 2)
    # Excipient effects (plausible assumptions)
    excipient_stability = 0.40 * hpmc + 0.10 * mcc - 0.20 * sls - 0.10 * mannitol + 0.0 * ccs

    raw_stability = 0.50 * stability_temp + 0.30 * stability_ph + 0.20 * excipient_stability
    shelf_stability = 100.0 * np.clip(raw_stability, 0.0, 1.0)

    # Add Gaussian noise
    bioavailability_obs = float(np.clip(bioavailability + rng.normal(0.0, MEASUREMENT_SEM), 0.0, 100.0))
    shelf_stability_obs = float(np.clip(shelf_stability + rng.normal(0.0, MEASUREMENT_SEM), 0.0, 100.0))

    return {
        BIOAVAILABILITY: (bioavailability_obs, MEASUREMENT_SEM),
        SHELF_STABILITY: (shelf_stability_obs, MEASUREMENT_SEM),
    }


def evaluate_formulation(parameterization: Dict[str, float], rng: np.random.Generator) -> Dict[str, Tuple[float, float]]:
    """
    Wraps the simulator using Ax parameterization dict. Computes the 5th excipient (mannitol)
    to satisfy the composition constraint: mcc + hpmc + ccs + sls + mannitol == 1.0.
    """
    mcc_val = float(parameterization[MCC])
    hpmc_val = float(parameterization[HPMC])
    ccs_val = float(parameterization[CCS])
    sls_val = float(parameterization[SLS])
    ph_val = float(parameterization[PH])
    temp_val = float(parameterization[STORAGE_TEMP])

    # Derived 5th excipient fraction to satisfy sum-to-1.0
    mannitol_val = 1.0 - (mcc_val + hpmc_val + ccs_val + sls_val)

    # If constraint is violated due to numerical issues, clip and degrade metrics heavily
    if mannitol_val < -1e-6:
        # Return very poor outcomes but still provide metric values
        poor_value = 0.0
        return {
            BIOAVAILABILITY: (poor_value, MEASUREMENT_SEM),
            SHELF_STABILITY: (poor_value, MEASUREMENT_SEM),
        }

    return simulate_formulation_metrics(
        mcc=mcc_val,
        hpmc=hpmc_val,
        ccs=ccs_val,
        sls=sls_val,
        mannitol=mannitol_val,
        ph=ph_val,
        temp_c=temp_val,
        rng=rng,
    )


def compute_empirical_pareto_frontier(points: np.ndarray) -> np.ndarray:
    """
    Compute indices of Pareto-efficient points for a 2D array of points (maximize both metrics).
    points: array of shape (n, 2) where columns correspond to [bioavailability, shelf_stability]
    Returns boolean mask of length n indicating Pareto-efficient points.
    """
    n = points.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # Any point that is strictly dominated by point i is not efficient
        dominates = (points[i, 0] >= points[:, 0]) & (points[i, 1] >= points[:, 1])
        strictly_better = (points[i, 0] > points[:, 0]) | (points[i, 1] > points[:, 1])
        is_efficient = is_efficient & ~(dominates & strictly_better & (np.arange(n) != i))
    # Remove points dominated by any other point
    for i in range(n):
        if not is_efficient[i]:
            continue
        others = np.arange(n) != i
        dominated_by_others = (points[others, 0] >= points[i, 0]) & (points[others, 1] >= points[i, 1]) & (
            (points[others, 0] > points[i, 0]) | (points[others, 1] > points[i, 1])
        )
        if dominated_by_others.any():
            is_efficient[i] = False
    return is_efficient


def main():
    rng = np.random.default_rng(42)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="pharma_formulation_moo",
        parameters=[
            {"name": MCC, "type": "range", "bounds": [0.0, 1.0]},
            {"name": HPMC, "type": "range", "bounds": [0.0, 1.0]},
            {"name": CCS, "type": "range", "bounds": [0.0, 1.0]},
            {"name": SLS, "type": "range", "bounds": [0.0, 1.0]},
            {"name": PH, "type": "range", "bounds": [PH_LOWER, PH_UPPER]},
            {"name": STORAGE_TEMP, "type": "range", "bounds": [TEMP_LOWER_C, TEMP_UPPER_C]},
        ],
        # Both objectives are to be maximized; let Ax infer thresholds
        objectives={
            BIOAVAILABILITY: ObjectiveProperties(minimize=False),
            SHELF_STABILITY: ObjectiveProperties(minimize=False),
        },
        # Reparameterized compositional constraint: mcc + hpmc + ccs + sls <= 1.0
        # The 5th excipient (mannitol) is computed as 1 - (mcc + hpmc + ccs + sls)
        parameter_constraints=[f"{MCC} + {HPMC} + {CCS} + {SLS} <= 1.0"],
    )

    # Optimization loop
    for _ in range(N_TRIALS):
        for _ in range(BATCH_SIZE):
            params, trial_index = ax_client.get_next_trial()
            results = evaluate_formulation(params, rng)
            ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    # Collect observed data
    df = ax_client.get_trials_data_frame()
    # Pivot to wide format for metrics
    df_wide = df.pivot_table(index="arm_name", columns="metric_name", values="mean", aggfunc="mean").reset_index()
    if BIOAVAILABILITY not in df_wide.columns or SHELF_STABILITY not in df_wide.columns:
        print("No data collected for one or more objectives.")
        return

    # Plot observed points and empirical Pareto frontier
    x = df_wide[BIOAVAILABILITY].to_numpy()
    y = df_wide[SHELF_STABILITY].to_numpy()
    pts = np.vstack([x, y]).T
    pareto_mask = compute_empirical_pareto_frontier(pts)
    pareto_pts = pts[pareto_mask]
    # Sort Pareto front by bioavailability for plotting
    pareto_pts = pareto_pts[np.argsort(pareto_pts[:, 0])]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=150)
    ax.scatter(x, y, fc="None", ec="k", label="Observed")
    if len(pareto_pts) >= 2:
        ax.plot(pareto_pts[:, 0], pareto_pts[:, 1], color="#0033FF", lw=2, label="Pareto Front (observed)")
    elif len(pareto_pts) == 1:
        ax.scatter(pareto_pts[:, 0], pareto_pts[:, 1], color="#0033FF", label="Pareto Point")
    ax.set_xlabel("Bioavailability (a.u.)")
    ax.set_ylabel("Shelf Stability (a.u.)")
    ax.set_title("Formulation Optimization: Bioavailability vs. Shelf Stability")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print Pareto-optimal parameterizations with metrics
    if len(pareto_pts) > 0:
        print("Empirical Pareto-optimal arms and outcomes:")
        pareto_arm_names = df_wide.loc[pareto_mask, "arm_name"].tolist()
        for arm_name, (bio, stab) in zip(pareto_arm_names, pareto_pts):
            arm_params = ax_client.experiment.arms_by_name[arm_name].parameters
            # Compute derived mannitol fraction for reporting
            mannitol_frac = 1.0 - (
                arm_params[MCC] + arm_params[HPMC] + arm_params[CCS] + arm_params[SLS]
            )
            print(
                f"- {arm_name}: "
                f"mcc={arm_params[MCC]:.3f}, hpmc={arm_params[HPMC]:.3f}, ccs={arm_params[CCS]:.3f}, "
                f"sls={arm_params[SLS]:.3f}, mannitol={mannitol_frac:.3f}, "
                f"pH={arm_params[PH]:.2f}, T={arm_params[STORAGE_TEMP]:.1f}C | "
                f"bio={bio:.1f}, stability={stab:.1f}"
            )


if __name__ == "__main__":
    main()