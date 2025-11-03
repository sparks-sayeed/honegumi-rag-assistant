# Generated for fermentation optimization with Ax Platform
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain-specific evaluation: protein production from a fermentation run
def evaluate_fermentation_run(temperature_celsius: float, ph: float, mixing_speed_rpm: float) -> dict:
    """
    Evaluate a single fermentation batch configuration.

    This function simulates protein production (e.g., g/L) as a function of:
      - temperature_celsius in °C
      - ph (unitless)
      - mixing_speed_rpm in rpm

    Replace this simulated model with actual measurement logic when integrating with lab execution:
      - Run batch with provided settings (temperature, pH, mixing speed)
      - After fermentation, quantify protein production (e.g., HPLC, Bradford, ELISA)
      - Return {"protein_production": (measured_value, standard_error)}

    Returns:
        dict: {"protein_production": (mean, sem)}
    """
    # Simulated "true" response surface: unimodal with interactions
    # Peak near (34°C, pH 6.8, 350 rpm)
    t_opt, ph_opt, rpm_opt = 34.0, 6.8, 350.0
    scale_t, scale_ph, scale_rpm = 3.0, 0.6, 120.0  # sensitivity (widths)

    # Base response: 3D Gaussian dome
    z_base = 8.5 * np.exp(
        -((temperature_celsius - t_opt) / scale_t) ** 2
        - ((ph - ph_opt) / scale_ph) ** 2
        - ((mixing_speed_rpm - rpm_opt) / scale_rpm) ** 2
    )

    # Mild interaction ripples (e.g., shear or oxygen transfer effects)
    z_ripple = (
        0.35 * np.sin((mixing_speed_rpm - rpm_opt) / 70.0)
        + 0.25 * np.cos((temperature_celsius - t_opt) / 2.0)
        - 0.2 * np.abs(ph - 7.0)
    )

    # Aggregate "true" production (g/L), clipped to non-negative
    true_production = max(0.0, z_base + z_ripple)

    # Heteroscedastic measurement/process noise: more variance at high shear
    noise_std = 0.30 + 0.0006 * (mixing_speed_rpm - rpm_opt) ** 2 / 350.0
    rng = np.random.default_rng(2025)
    observed_production = float(np.clip(true_production + rng.normal(0.0, noise_std), 0.0, None))
    sem_estimate = float(noise_std)  # Replace with empirical SEM if available

    return {"protein_production": (observed_production, sem_estimate)}


if __name__ == "__main__":
    # Configure Ax for a single-objective, noisy optimization
    ax_client = AxClient()

    # Safe operating ranges (adjust to your process constraints if needed):
    # - temperature_celsius: typical mesophilic fermentation temp range
    # - ph: common range for many microbial processes
    # - mixing_speed_rpm: benchtop bioreactor typical range
    ax_client.create_experiment(
        name="fermentation_max_protein_production",
        parameters=[
            {
                "name": "temperature_celsius",
                "type": "range",
                "bounds": [25.0, 40.0],
            },
            {
                "name": "ph",
                "type": "range",
                "bounds": [5.5, 7.5],
            },
            {
                "name": "mixing_speed_rpm",
                "type": "range",
                "bounds": [100.0, 600.0],
            },
        ],
        objectives={
            "protein_production": ObjectiveProperties(minimize=False),
        },
        # Tracking a noisy objective; SEM provided via evaluation function
        tracking_metrics=None,
        choose_generation_strategy=True,
    )

    # Optimization budget: 30 trials
    n_trials = 30
    for _ in range(n_trials):
        parameterization, trial_index = ax_client.get_next_trial()

        # Extract domain parameters
        temperature_celsius = float(parameterization["temperature_celsius"])
        ph = float(parameterization["ph"])
        mixing_speed_rpm = float(parameterization["mixing_speed_rpm"])

        # Evaluate the batch (replace with real experimental run + measurement)
        results = evaluate_fermentation_run(
            temperature_celsius=temperature_celsius,
            ph=ph,
            mixing_speed_rpm=mixing_speed_rpm,
        )

        # Report results to Ax (provide mean and SEM for the noisy objective)
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    # Best found parameters and metrics
    best_parameters, best_metrics = ax_client.get_best_parameters()
    print("Best parameters found:")
    for k, v in best_parameters.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")
    print("\nBest observed protein production:")
    metric = best_metrics["protein_production"]
    mean_val = metric["mean"] if isinstance(metric, dict) and "mean" in metric else metric
    sem_val = metric.get("sem", None) if isinstance(metric, dict) else None
    if sem_val is not None:
        print(f"  mean = {mean_val:.4f}, SEM = {sem_val:.4f}")
    else:
        print(f"  mean = {mean_val:.4f}")

    # Visualization of objective over trials
    objectives = ax_client.objective_names
    df = ax_client.get_trials_data_frame()

    # Ensure we have a clean series for plotting
    y_series = df[objectives[0]] if objectives and objectives[0] in df.columns else None
    if y_series is not None and len(y_series) > 0:
        x_vals = np.arange(len(y_series))
        best_so_far = np.maximum.accumulate(y_series.values.astype(float))

        fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
        ax.scatter(x_vals, y_series.values, ec="k", fc="none", label="Observed")
        ax.plot(x_vals, best_so_far, color="#0033FF", lw=2, label="Best to Trial")
        ax.set_xlabel("Trial Number")
        ax.set_ylabel(objectives[0])
        ax.set_title("Fermentation Optimization: Protein Production")
        ax.legend()
        plt.tight_layout()
        plt.show()
    else:
        print("No data available for plotting.")