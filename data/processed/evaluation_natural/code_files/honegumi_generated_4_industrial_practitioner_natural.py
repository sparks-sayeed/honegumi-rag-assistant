# Generated for optimizing battery electrolyte conductivity with Ax (https://ax.dev/)
# If needed, install dependencies:
# %pip install ax-platform==0.4.3 matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain-specific objective name
conductivity_metric_name = "conductivity"  # Units: S/cm

# Reproducibility
RNG = np.random.default_rng(seed=2025)


def measure_electrolyte_conductivity(
    salt_1_concentration: float,
    salt_2_concentration: float,
    salt_3_concentration: float,
    salt_4_concentration: float,
    replicate_measurements: int = 1,
) -> tuple[float, float]:
    """
    Simulated evaluation of electrolyte ionic conductivity (S/cm) for a 4-salt mixture.

    This stub models:
      - Base contribution: sum of per-salt molar conductivity coefficients times concentration.
      - Mobility reduction at higher ionic strength.
      - Pairwise interaction synergies or penalties between salts.
      - Measurement noise and SEM (to inform Ax that data are noisy).

    Replace with actual lab measurement or high-fidelity simulation as needed:
      - Run mixing / temperature equilibration protocol.
      - Measure conductivity (e.g., electrochemical impedance spectroscopy).
      - Aggregate replicate measurements and compute SEM.

    Returns:
      (mean_conductivity_S_per_cm, sem_S_per_cm)
    """

    c1 = float(salt_1_concentration)
    c2 = float(salt_2_concentration)
    c3 = float(salt_3_concentration)
    c4 = float(salt_4_concentration)

    # Ensure concentrations are non-negative
    c1 = max(0.0, c1)
    c2 = max(0.0, c2)
    c3 = max(0.0, c3)
    c4 = max(0.0, c4)

    total_c = c1 + c2 + c3 + c4  # M

    # Base per-salt contributions (approximate scalings to keep values in 0–0.02 S/cm range)
    # These represent per-Molar contributions under ideal dilution.
    lambda0 = np.array([0.0095, 0.0075, 0.0110, 0.0080])  # S/cm per M
    base = lambda0[0] * c1 + lambda0[1] * c2 + lambda0[2] * c3 + lambda0[3] * c4

    # Mobility reduction with increasing ionic strength (heuristic)
    # Mobility factor peaks at low-to-moderate concentration and decreases toward 2 M.
    mobility_factor = 1.0 - 0.25 * total_c - 0.18 * (total_c**2)
    mobility_factor = float(np.clip(mobility_factor, 0.45, 1.0))

    # Synergy / interaction terms (heuristic):
    # Slight synergy between salt_1 and salt_3, mild penalty for salt_2 with salt_4, small positive for salt_1 with salt_4
    synergy = (
        0.0032 * c1 * c3
        - 0.0010 * c2 * c4
        + 0.0008 * c1 * c4
    )

    # Compute "true" conductivity before noise
    kappa = base * mobility_factor + synergy

    # Clip to nonnegative (physical)
    kappa = max(0.0, kappa)

    # Simulate replicate measurements with Gaussian noise (5% relative noise + floor)
    rel_noise = 0.05
    noise_floor = 2.5e-4  # 0.25 mS/cm floor
    sigma = max(noise_floor, rel_noise * kappa)

    if replicate_measurements < 1:
        replicate_measurements = 1

    measurements = kappa + RNG.normal(loc=0.0, scale=sigma, size=replicate_measurements)
    measured_mean = float(np.mean(measurements))
    measured_sem = float(np.std(measurements, ddof=1) / np.sqrt(replicate_measurements)) if replicate_measurements > 1 else float(sigma)

    # Ensure nonnegative after noise averaging
    measured_mean = max(0.0, measured_mean)

    return measured_mean, measured_sem


# Initialize Ax client
ax_client = AxClient()

# Define the experiment: 4 salt concentrations with a total concentration cap of 2.0 M.
ax_client.create_experiment(
    name="electrolyte_conductivity_optimization",
    parameters=[
        {
            "name": "salt_1_concentration",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "float",
        },
        {
            "name": "salt_2_concentration",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "float",
        },
        {
            "name": "salt_3_concentration",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "float",
        },
        {
            "name": "salt_4_concentration",
            "type": "range",
            "bounds": [0.0, 2.0],
            "value_type": "float",
        },
    ],
    objectives={
        conductivity_metric_name: ObjectiveProperties(minimize=False),
    },
    parameter_constraints=[
        # Total salt concentration must be <= 2.0 M to avoid excessive viscosity
        "salt_1_concentration + salt_2_concentration + salt_3_concentration + salt_4_concentration <= 2.0"
    ],
)

# Optimization budget (number of trials)
total_trials = 40

for i in range(total_trials):
    parameters, trial_index = ax_client.get_next_trial()

    # Extract concentrations
    c1 = parameters["salt_1_concentration"]
    c2 = parameters["salt_2_concentration"]
    c3 = parameters["salt_3_concentration"]
    c4 = parameters["salt_4_concentration"]

    # Evaluate (replace this with actual lab/simulation measurement integration)
    mean_cond, sem_cond = measure_electrolyte_conductivity(c1, c2, c3, c4)

    # Report back to Ax; include SEM to inform the noise model
    ax_client.complete_trial(
        trial_index=trial_index,
        raw_data={conductivity_metric_name: (mean_cond, sem_cond)},
    )

# Retrieve best parameters and predicted metric values
best_parameters, best_values = ax_client.get_best_parameters()
print("Best formulation (Ax predicted best):")
print(best_parameters)
print("Predicted metric statistics (mean, covariance):")
print(best_values)

# Plot optimization trace (Observed best so far)
objectives = ax_client.objective_names
df = ax_client.get_trials_data_frame()

if len(objectives) == 1 and objectives[0] in df.columns:
    metric_series = df[objectives[0]]

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.scatter(df.index, metric_series, ec="k", fc="none", label="Observed")
    ax.plot(df.index, metric_series.cummax(), color="#0033FF", lw=2, label="Best to Trial")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel(f"{objectives[0]} (S/cm)")
    ax.set_title("Electrolyte Conductivity Optimization Trace")
    ax.grid(True, alpha=0.2)
    ax.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No objective data available to plot.")