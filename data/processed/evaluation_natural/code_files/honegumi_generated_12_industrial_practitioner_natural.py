# Generated for optimizing a universal ceramic binder formulation with Ax Platform
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from typing import Dict, Tuple
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt
from itertools import combinations


# Objective metric names (maximize all)
OBJ_ALUMINA = "green_strength_alumina_mpa"
OBJ_ZIRCONIA = "green_strength_zirconia_mpa"
OBJ_SIC = "green_strength_sic_mpa"


def _gauss(x: float, mu: float, sigma: float) -> float:
    """Bounded Gaussian-shaped contribution between 0 and 1."""
    return float(np.exp(-0.5 * ((x - mu) / sigma) ** 2))


def _saturating(x: float, scale: float) -> float:
    """Saturating contribution between 0 and 1; rises with x then saturates."""
    return float(1.0 - np.exp(-max(x, 0.0) / max(scale, 1e-8)))


def _penalty_over(x: float, threshold: float, strength: float) -> float:
    """Soft penalty when x exceeds threshold."""
    excess = max(0.0, x - threshold)
    return float(-strength * excess**2)


def evaluate_binder_formulation(
    binder_polymer_wt_pct: float,
    plasticizer_wt_pct: float,
    dispersant_wt_pct: float,
    solids_loading_vol_frac: float,
    slurry_pH: float,
    mixing_time_min: float,
    drying_temperature_c: float,
    drying_time_hr: float,
    rng: np.random.Generator,
) -> Dict[str, float]:
    """
    Evaluate green strength for alumina, zirconia, and SiC using a realistic surrogate model.

    This function simulates how formulation and process variables affect green strength.
    Replace this surrogate with actual lab measurements (e.g., call out to LIMS/DB)
    and return the measured strengths in MPa for each material.

    Parameters expected ranges:
      - binder_polymer_wt_pct: 0.0 - 10.0 (wt% relative to solids)
      - plasticizer_wt_pct:    0.0 - 5.0
      - dispersant_wt_pct:     0.0 - 2.0
      - solids_loading_vol_frac: 0.40 - 0.65 (vol fraction)
      - slurry_pH:             4.0 - 11.0
      - mixing_time_min:       10 - 120
      - drying_temperature_c:  25 - 80
      - drying_time_hr:        1 - 24

    Returns:
      Dict with keys OBJ_ALUMINA, OBJ_ZIRCONIA, OBJ_SIC and green strength values (MPa).
    """

    # Sum of organics; excessive organics tend to weaken green strength
    organics_total = binder_polymer_wt_pct + plasticizer_wt_pct + dispersant_wt_pct

    # Common contributions
    mix_contrib = _saturating(mixing_time_min, scale=45.0)  # diminishing returns by ~45 min
    dryT_contrib = _gauss(drying_temperature_c, mu=60.0, sigma=12.0)  # film formation
    dryt_contrib = _saturating(drying_time_hr, scale=8.0)  # reaches plateau ~8 h

    # Small generic penalty if organics too high
    penalty_common = _penalty_over(organics_total, threshold=10.0, strength=0.01)

    # Material-specific preferences (approximate, domain-inspired)
    # Each term contributes 0..1; combine with weights, scale to MPa.
    def mat_score(
        binder_mu: float,
        binder_sigma: float,
        plast_mu: float,
        plast_sigma: float,
        disp_mu: float,
        disp_sigma: float,
        solids_mu: float,
        solids_sigma: float,
        ph_mu: float,
        ph_sigma: float,
        base_strength: float,
    ) -> float:
        binder_c = _gauss(binder_polymer_wt_pct, binder_mu, binder_sigma)
        plast_c = _gauss(plasticizer_wt_pct, plast_mu, plast_sigma)
        disp_c = _gauss(dispersant_wt_pct, disp_mu, disp_sigma)
        solids_c = _gauss(solids_loading_vol_frac, solids_mu, solids_sigma)
        ph_c = _gauss(slurry_pH, ph_mu, ph_sigma)

        # Interaction: solids loading synergizes with adequate binder and mixing
        synergy = 0.2 * solids_c * binder_c * mix_contrib

        # Combine with weights (sum of weights about 1.0 plus minor terms)
        combined = (
            0.25 * binder_c
            + 0.10 * plast_c
            + 0.10 * disp_c
            + 0.25 * solids_c
            + 0.10 * ph_c
            + 0.10 * mix_contrib
            + 0.08 * dryT_contrib
            + 0.07 * dryt_contrib
            + synergy
            + penalty_common
        )
        combined = max(combined, 0.0)
        # Convert to MPa with base scale
        return base_strength * combined

    # Target base strength scales (typical green strengths)
    # Alumina often robust; zirconia slightly lower; SiC can be trickier.
    alumina_strength = mat_score(
        binder_mu=4.0, binder_sigma=1.4,
        plast_mu=1.0, plast_sigma=0.5,
        disp_mu=0.5, disp_sigma=0.25,
        solids_mu=0.58, solids_sigma=0.035,
        ph_mu=9.0, ph_sigma=1.0,
        base_strength=32.0,
    )
    zirconia_strength = mat_score(
        binder_mu=3.5, binder_sigma=1.3,
        plast_mu=0.9, plast_sigma=0.45,
        disp_mu=0.7, disp_sigma=0.3,
        solids_mu=0.60, solids_sigma=0.03,
        ph_mu=7.5, ph_sigma=1.1,
        base_strength=28.0,
    )
    sic_strength = mat_score(
        binder_mu=3.0, binder_sigma=1.2,
        plast_mu=0.8, plast_sigma=0.4,
        disp_mu=0.4, disp_sigma=0.25,
        solids_mu=0.55, solids_sigma=0.03,
        ph_mu=8.5, ph_sigma=1.2,
        base_strength=26.0,
    )

    # Add small heteroscedastic noise to emulate measurement noise
    def noisy(v: float) -> float:
        sigma = 0.6 + 0.03 * v  # slightly increases with strength
        return float(max(0.0, v + rng.normal(0.0, sigma)))

    return {
        OBJ_ALUMINA: noisy(alumina_strength),
        OBJ_ZIRCONIA: noisy(zirconia_strength),
        OBJ_SIC: noisy(sic_strength),
    }


def main() -> None:
    # Reproducibility
    rng = np.random.default_rng(seed=12345)

    ax_client = AxClient()
    ax_client.create_experiment(
        name="universal_binder_green_strength_optimization",
        parameters=[
            # Formulation parameters (continuous ranges; adjust to your system)
            {"name": "binder_polymer_wt_pct", "type": "range", "bounds": [0.0, 10.0]},  # e.g., PVA/PVB wt%
            {"name": "plasticizer_wt_pct", "type": "range", "bounds": [0.0, 5.0]},     # e.g., PEG/DBP wt%
            {"name": "dispersant_wt_pct", "type": "range", "bounds": [0.0, 2.0]},      # e.g., PAA wt%
            {"name": "solids_loading_vol_frac", "type": "range", "bounds": [0.40, 0.65]},
            {"name": "slurry_pH", "type": "range", "bounds": [4.0, 11.0]},
            # Process parameters
            {"name": "mixing_time_min", "type": "range", "bounds": [10.0, 120.0]},
            {"name": "drying_temperature_c", "type": "range", "bounds": [25.0, 80.0]},
            {"name": "drying_time_hr", "type": "range", "bounds": [1.0, 24.0]},
        ],
        objectives={
            OBJ_ALUMINA: ObjectiveProperties(minimize=False),
            OBJ_ZIRCONIA: ObjectiveProperties(minimize=False),
            OBJ_SIC: ObjectiveProperties(minimize=False),
        },
        overwrite_existing_experiment=True,
    )

    # Run optimization (budget: 120 trials)
    n_trials = 120
    for _ in range(n_trials):
        parameterization, trial_index = ax_client.get_next_trial()

        # Extract parameters
        binder_polymer_wt_pct = float(parameterization["binder_polymer_wt_pct"])
        plasticizer_wt_pct = float(parameterization["plasticizer_wt_pct"])
        dispersant_wt_pct = float(parameterization["dispersant_wt_pct"])
        solids_loading_vol_frac = float(parameterization["solids_loading_vol_frac"])
        slurry_pH = float(parameterization["slurry_pH"])
        mixing_time_min = float(parameterization["mixing_time_min"])
        drying_temperature_c = float(parameterization["drying_temperature_c"])
        drying_time_hr = float(parameterization["drying_time_hr"])

        # Evaluate
        results = evaluate_binder_formulation(
            binder_polymer_wt_pct=binder_polymer_wt_pct,
            plasticizer_wt_pct=plasticizer_wt_pct,
            dispersant_wt_pct=dispersant_wt_pct,
            solids_loading_vol_frac=solids_loading_vol_frac,
            slurry_pH=slurry_pH,
            mixing_time_min=mixing_time_min,
            drying_temperature_c=drying_temperature_c,
            drying_time_hr=drying_time_hr,
            rng=rng,
        )

        ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    # Retrieve Pareto-optimal set
    pareto = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)

    # Summarize
    objectives = ax_client.objective_names
    print(f"Completed {n_trials} trials.")
    print(f"Found {len(pareto)} Pareto-optimal formulations.")
    # Print top 5 Pareto solutions by simple hypervolume proxy (sum of normalized objectives)
    try:
        df = ax_client.get_trials_data_frame()
        if objectives[0] not in df.columns:
            df = (
                df.pivot(index="trial_index", columns="metric_name", values="mean")
                .reset_index()
            )
        # normalize by column max for proxy
        norm = df[objectives] / df[objectives].max()
        df["hv_proxy"] = norm.sum(axis=1)
        # Extract Pareto trial indices
        pareto_trial_indices = list(pareto.keys())
        pareto_df = df[df["trial_index"].isin(pareto_trial_indices)].copy()
        top5 = pareto_df.sort_values("hv_proxy", ascending=False).head(5)
        print("Top Pareto candidates (hv_proxy and metrics):")
        print(top5[["trial_index", "hv_proxy"] + objectives])
    except Exception as e:
        print(f"Summary creation issue: {e}")

    # Visualization: pairwise objective scatter and Pareto set overlay
    try:
        df = ax_client.get_trials_data_frame()
        if objectives[0] not in df.columns:
            df = (
                df.pivot(index="trial_index", columns="metric_name", values="mean")
                .reset_index()
            )

        pareto_data = [{obj: p[1][0][obj] for obj in objectives} for p in pareto.values()]
        pareto_df = pd.DataFrame(pareto_data)

        obj_pairs = list(combinations(objectives, 2))
        n_pairs = len(obj_pairs)
        fig, axes = plt.subplots(1, n_pairs, figsize=(5 * n_pairs, 4), dpi=140)
        if n_pairs == 1:
            axes = [axes]
        for ax, (o1, o2) in zip(axes, obj_pairs):
            ax.scatter(df[o1], df[o2], fc="None", ec="k", alpha=0.6, label="Observed")
            if not pareto_df.empty:
                ax.scatter(
                    pareto_df[o1],
                    pareto_df[o2],
                    color="#0033FF",
                    s=40,
                    label="Pareto-optimal",
                )
            ax.set_xlabel(o1)
            ax.set_ylabel(o2)
            ax.grid(True, ls="--", alpha=0.3)
            ax.legend()
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"Visualization issue: {e}")


if __name__ == "__main__":
    main()