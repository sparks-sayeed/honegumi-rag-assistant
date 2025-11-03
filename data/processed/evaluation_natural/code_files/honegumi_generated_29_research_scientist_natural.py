# Multi-objective battery charging optimization with Ax (35-cell study)
# %pip install ax-platform==0.4.3 matplotlib pandas numpy

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Dict, Tuple

from ax.service.ax_client import AxClient
from ax.service.utils.instantiation import ObjectiveProperties


# Domain-specific parameter and objective names
# Parameters (continuous):
# - cc_c_rate: Constant-Current charge rate [C], e.g., 0.25C to 3.0C
# - cv_voltage_limit_v: Constant-Voltage limit [V], e.g., 4.05V to 4.35V
# - cv_cutoff_c_rate: CV termination current threshold [C], e.g., 0.02C to 0.25C
#
# Objectives:
# - charging_speed_c_rate (maximize): Effective charging speed in "C", where 1C = 60 minutes to full charge.
# - degradation_index (minimize): Unitless degradation index (higher is worse), proxy for long-term capacity fade.


# Synthetic but physics-informed evaluation model for CC-CV charging tradeoffs.
# This is a realistic stub that captures common qualitative trends:
# - Faster charging (higher C-rate, higher voltage) increases "charging_speed_c_rate".
# - Higher current, higher voltage, and longer time at high voltage increase "degradation_index".
# If you have real experimental code, replace this function body with actual measurements or a simulator.
def evaluate_charging_protocol(parameters: Dict[str, float]) -> Dict[str, Tuple[float, float]]:
    """
    Evaluate a CC-CV charging protocol.

    Parameters
    ----------
    parameters : dict
        Dictionary with keys:
            - "cc_c_rate": float, in [0.25, 3.0]
            - "cv_voltage_limit_v": float, in [4.05, 4.35]
            - "cv_cutoff_c_rate": float, in [0.02, 0.25]

    Returns
    -------
    dict
        {
            "charging_speed_c_rate": (mean_value, sem or None),
            "degradation_index": (mean_value, sem or None)
        }

    Notes
    -----
    - Replace this function with actual lab execution and measurements if available.
    - Currently returns noisy observations with unknown SEM (SEM=None) so Ax will infer noise.
    """
    cc_c_rate = float(parameters["cc_c_rate"])
    cv_voltage_limit_v = float(parameters["cv_voltage_limit_v"])
    cv_cutoff_c_rate = float(parameters["cv_cutoff_c_rate"])

    # Clamp to safe bounds to avoid numerical edge cases
    cc_c_rate = float(np.clip(cc_c_rate, 0.25, 3.0))
    cv_voltage_limit_v = float(np.clip(cv_voltage_limit_v, 4.05, 4.35))
    cv_cutoff_c_rate = float(np.clip(cv_cutoff_c_rate, 0.02, 0.25))

    # Model the fraction of capacity delivered during CC stage (alpha).
    # Higher voltage allows longer CC phase (generally faster overall).
    alpha = 0.60 + 1.20 * (cv_voltage_limit_v - 4.20)  # base around 4.20 V
    alpha = float(np.clip(alpha, 0.30, 0.95))

    # Approximate CC time (minutes): proportional to charged fraction / C-rate
    cc_time_min = 60.0 * alpha / cc_c_rate

    # Approximate CV time (minutes):
    # Longer if the cutoff current is very low (stricter termination), shorter for higher cutoff.
    # Also depends on how much capacity remains (1 - alpha) and on the ratio cc/cutoff.
    ratio = cc_c_rate / (cv_cutoff_c_rate + 1e-6)
    # Normalize the log factor to keep values in a comparable range
    norm_log = math.log(1.0 + 3.0 / 0.02)
    cv_time_min = 60.0 * (1.0 - alpha) * (math.log(1.0 + ratio) / norm_log)

    # Total charge time
    total_time_min = cc_time_min + cv_time_min

    # Effective charging speed in "C" units (1C = 60 minutes)
    charging_speed_c_rate = 60.0 / total_time_min

    # Degradation index (unitless): grows with current, high voltage, and stricter CV (longer time at high V).
    delta_v = max(cv_voltage_limit_v - 4.20, 0.0)
    degradation_index = (
        0.30 * (cc_c_rate ** 2) +          # resistive heating/current stress
        120.0 * (delta_v ** 2) +           # high-voltage stress (SEI growth / oxidation)
        0.05 / cv_cutoff_c_rate            # longer CV stress for lower cutoff threshold
    )

    # Add observation noise; SEM unknown so we return None and let Ax infer noise
    rng = np.random.default_rng()
    charging_speed_noise = rng.normal(0.0, 0.03)  # ~3% C noise
    degradation_noise = rng.normal(0.0, 0.08)     # small noise in degradation index

    charging_speed_obs = float(max(0.05, charging_speed_c_rate + charging_speed_noise))
    degradation_obs = float(max(0.01, degradation_index + degradation_noise))

    return {
        "charging_speed_c_rate": (charging_speed_obs, None),  # SEM unknown
        "degradation_index": (degradation_obs, None),         # SEM unknown
    }


def compute_pareto_front(observed_wide: pd.DataFrame,
                         maximize_cols,
                         minimize_cols):
    """
    Compute non-dominated set indices for a two-objective case.

    Parameters
    ----------
    observed_wide : pd.DataFrame
        Wide dataframe indexed by trial_index with metric columns as means.
    maximize_cols : list[str]
        Names of metrics to maximize.
    minimize_cols : list[str]
        Names of metrics to minimize.

    Returns
    -------
    pd.Index
        Index of observed_wide corresponding to non-dominated points.
    """
    # Convert to array with direction normalization:
    # For maximization metrics keep as is; for minimization, negate to convert into maximization.
    df = observed_wide.copy()
    for col in minimize_cols:
        df[col + "__for_dom"] = -df[col]
    for col in maximize_cols:
        df[col + "__for_dom"] = df[col]

    dom_cols = [c for c in df.columns if c.endswith("__for_dom")]
    vals = df[dom_cols].to_numpy()
    n = vals.shape[0]
    is_dominated = np.zeros(n, dtype=bool)
    for i in range(n):
        if is_dominated[i]:
            continue
        vi = vals[i]
        # A point j dominates i if all objectives >= and at least one >
        dominates_i = np.all(vals >= vi, axis=1) & np.any(vals > vi, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            is_dominated[i] = True
    nondom_idx = observed_wide.index[~is_dominated]
    return nondom_idx


def main():
    # Initialize Ax client for multi-objective optimization
    ax_client = AxClient()

    # Create experiment with 3 protocol parameters and 2 competing objectives
    ax_client.create_experiment(
        name="battery_cc_cv_moo",
        parameters=[
            {
                "name": "cc_c_rate",
                "type": "range",
                "bounds": [0.25, 3.00],
                "value_type": "float",
            },
            {
                "name": "cv_voltage_limit_v",
                "type": "range",
                "bounds": [4.05, 4.35],
                "value_type": "float",
            },
            {
                "name": "cv_cutoff_c_rate",
                "type": "range",
                "bounds": [0.02, 0.25],
                "value_type": "float",
            },
        ],
        objectives={
            # Maximize charging speed
            "charging_speed_c_rate": ObjectiveProperties(minimize=False),
            # Minimize degradation
            "degradation_index": ObjectiveProperties(minimize=True),
        },
    )

    # Run 35 trials (35-cell study)
    N_TRIALS = 35
    for _ in range(N_TRIALS):
        parameterization, trial_index = ax_client.get_next_trial()
        results = evaluate_charging_protocol(parameterization)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    # Retrieve all trial data
    df_long = ax_client.get_trials_data_frame()
    # Pivot to wide format: trial_index x metric -> mean
    observed = (
        df_long.pivot_table(
            index="trial_index",
            columns="metric_name",
            values="mean",
            aggfunc="mean",
        )
        .sort_index()
    )

    # Compute Pareto frontier from observed data
    metric_maximize = ["charging_speed_c_rate"]
    metric_minimize = ["degradation_index"]
    pareto_idx = compute_pareto_front(
        observed_wide=observed,
        maximize_cols=metric_maximize,
        minimize_cols=metric_minimize,
    )
    pareto_points = observed.loc[pareto_idx].copy()

    # Sort Pareto points by the minimizing metric for a nice front line
    pareto_points = pareto_points.sort_values(by="degradation_index", ascending=True)

    # Plot observed outcomes and Pareto front
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    ax.scatter(
        observed["degradation_index"],
        observed["charging_speed_c_rate"],
        facecolors="none",
        edgecolors="k",
        label="Observed",
    )
    ax.plot(
        pareto_points["degradation_index"],
        pareto_points["charging_speed_c_rate"],
        color="#1f77b4",
        linewidth=2.0,
        label="Pareto Front (observed)",
    )
    ax.set_xlabel("Degradation Index (lower is better)")
    ax.set_ylabel("Charging Speed (C-rate, higher is better)")
    ax.set_title("Battery CC-CV Charging: Observed Trade-off and Pareto Front")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Display Pareto-optimal parameterizations with metrics
    # Collect parameters for each trial
    trial_params = {}
    for t_index, trial in ax_client.experiment.trials.items():
        # Each trial has a single arm in this setup
        arm = trial.arm
        trial_params[t_index] = arm.parameters

    pareto_table = []
    for t_idx in pareto_points.index:
        params = trial_params.get(t_idx, {})
        pareto_table.append(
            {
                "trial_index": t_idx,
                "cc_c_rate": params.get("cc_c_rate"),
                "cv_voltage_limit_v": params.get("cv_voltage_limit_v"),
                "cv_cutoff_c_rate": params.get("cv_cutoff_c_rate"),
                "degradation_index": observed.loc[t_idx, "degradation_index"],
                "charging_speed_c_rate": observed.loc[t_idx, "charging_speed_c_rate"],
            }
        )
    pareto_df = pd.DataFrame(pareto_table).sort_values(
        by=["degradation_index", "charging_speed_c_rate"], ascending=[True, False]
    )
    pd.set_option("display.precision", 4)
    print("\nPareto-optimal protocols (observed):")
    print(pareto_df.reset_index(drop=True))


if __name__ == "__main__":
    main()