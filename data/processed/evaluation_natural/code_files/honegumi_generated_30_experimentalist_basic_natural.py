# Optimizing packaging film barrier performance with Ax (Meta's Bayesian optimization)
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Metric name for the objective to minimize
oxygen_permeability_metric = "oxygen_permeability_cc_m2_day_atm"


def measure_oxygen_permeability(thickness_microns: float, coating_weight_g_m2: float, rng: np.random.Generator | None = None) -> float:
    """Simulate measuring oxygen permeability of a packaging film.

    Domain assumptions for the simulator:
    - Thicker base film reduces oxygen transmission approximately inversely with thickness.
    - Barrier coating reduces permeability approximately exponentially with coating weight (diminishing returns).
    - Excessive coating can introduce defects (microcracks) that increase permeability (quadratic penalty beyond a threshold).
    - Extremely thin films are prone to pinholes and defects (sharp penalty when too thin).
    - Measurements are noisy (additive noise).

    Units:
    - thickness_microns: micrometers [µm]
    - coating_weight_g_m2: grams per square meter [g/m^2]
    - return oxygen_permeability_cc_m2_day_atm: cubic centimeters per square meter per day per atmosphere [cc/(m^2·day·atm)]

    TODO (replace this simulator with actual measurement or a validated physics-based model):
    - Integrate with your lab instrument's API or database
    - Return the measured oxygen permeability for the given parameters

    """
    if rng is None:
        rng = np.random.default_rng()

    # Base permeability model for an uncoated reference film
    ref_thickness_um = 30.0
    base_otr_at_ref = 120.0  # cc/(m^2·day·atm) at 30 µm uncoated (example)
    otr_from_thickness = base_otr_at_ref * (ref_thickness_um / thickness_microns)

    # Coating barrier effect (diminishing returns)
    # Larger k => stronger barrier per g/m^2
    k_barrier = 0.9
    barrier_factor = np.exp(-k_barrier * coating_weight_g_m2)

    # Penalty for excessive coating weight (defects beyond ~3.5 g/m^2)
    excessive_coating_threshold = 3.5
    defect_penalty_coating = 1.0
    if coating_weight_g_m2 > excessive_coating_threshold:
        defect_penalty_coating += 0.15 * (coating_weight_g_m2 - excessive_coating_threshold) ** 2

    # Penalty for very thin films (pinholes and handling defects < ~15 µm)
    thin_threshold = 15.0
    # Smooth logistic penalty: ~1 when thick, grows rapidly when thinner than threshold
    thin_sharpness = 1.2
    defect_penalty_thin = 1.0 + 2.0 / (1.0 + np.exp((thickness_microns - thin_threshold) / thin_sharpness))

    # Combine effects
    otr_true = otr_from_thickness * barrier_factor * defect_penalty_coating * defect_penalty_thin

    # Additive measurement noise (heteroskedastic-ish)
    noise_sd = 1.5 + 0.02 * otr_true
    measured_otr = max(0.0005, otr_true + rng.normal(0.0, noise_sd))

    return float(measured_otr)


# Initialize Ax client
ax_client = AxClient()

# Define the experiment: parameters and objective
ax_client.create_experiment(
    parameters=[
        {
            "name": "thickness_microns",
            "type": "range",
            "bounds": [12.0, 80.0],  # typical packaging films ~12–80 µm
        },
        {
            "name": "coating_weight_g_m2",
            "type": "range",
            "bounds": [0.0, 5.0],  # typical barrier coating weights ~0–5 g/m^2
        },
    ],
    objectives={
        oxygen_permeability_metric: ObjectiveProperties(minimize=True),
    },
)

# Optimization budget: 24 trials (no prior data)
rng = np.random.default_rng(42)
num_trials = 24
for _ in range(num_trials):
    parameterization, trial_index = ax_client.get_next_trial()

    # Extract parameters
    thickness_um = float(parameterization["thickness_microns"])
    coating_g_m2 = float(parameterization["coating_weight_g_m2"])

    # Evaluate experiment (replace with real measurement if available)
    oxygen_permeability_value = measure_oxygen_permeability(thickness_um, coating_g_m2, rng=rng)

    # Report result to Ax; pass a float (Ax will infer noise model)
    ax_client.complete_trial(trial_index=trial_index, raw_data=oxygen_permeability_value)

# Find and print the best parameters according to the model
best_parameters, best_metrics = ax_client.get_best_parameters()
best_value = best_metrics[oxygen_permeability_metric]["mean"]
print("Best parameters found:")
print(f"  thickness_microns       = {best_parameters['thickness_microns']:.3f} µm")
print(f"  coating_weight_g_m2     = {best_parameters['coating_weight_g_m2']:.3f} g/m^2")
print(f"Estimated best oxygen permeability: {best_value:.3f} cc/(m^2·day·atm)")

# Plot results over trials
df = ax_client.get_trials_data_frame()
objective_name = ax_client.objective_names[0]
df = df.sort_index()

fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
ax.scatter(df.index, df[objective_name], ec="k", fc="none", label="Observed")
ax.plot(
    df.index,
    np.minimum.accumulate(df[objective_name].values),
    color="#0033FF",
    lw=2,
    label="Best to Trial",
)
ax.set_xlabel("Trial Number")
ax.set_ylabel("Oxygen Permeability [cc/(m^2·day·atm)]")
ax.set_title("Optimization of Packaging Film Oxygen Barrier")
ax.legend()
plt.tight_layout()
plt.show()