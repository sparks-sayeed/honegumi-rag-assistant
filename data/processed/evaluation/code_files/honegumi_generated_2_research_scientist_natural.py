# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# Domain: Optimize anti-corrosion coating formulation to minimize corrosion damage.
# Variables:
#   - resin_fraction, inhibitor_fraction, insulator_fraction, stabilizer_fraction (sum to 1.0)
#   - thickness_micrometers
# Constraints:
#   - resin_fraction >= inhibitor_fraction
#   - resin_fraction >= insulator_fraction
#   - resin_fraction + inhibitor_fraction + insulator_fraction <= 1.0 (stabilizer is the remainder)
# Objective: Minimize corrosion_damage (arbitrary units; lower is better).
# Batch size: 6 (parallel experiments)
# Noise model: True (we simulate measurement noise)


RNG = np.random.default_rng(42)
objective_name = "corrosion_damage"
composition_total = 1.0


def evaluate_coating_corrosion_damage(
    resin_fraction: float,
    inhibitor_fraction: float,
    insulator_fraction: float,
    thickness_micrometers: float,
    add_noise: bool = True,
) -> float:
    """
    Simulated evaluation of corrosion damage for a given coating formulation.

    This stub models plausible relationships:
      - Inhibitor fraction reduces corrosion chemically (saturating returns).
      - Resin ensures film continuity; insufficient resin increases damage.
      - Insulator reduces ionic transport (barrier effect, saturating).
      - Thickness improves barrier up to a point (saturating), too thick may crack.
      - Stabilizer is neutral to mildly diluting at high levels.

    Returns:
      corrosion_damage: float, lower is better.

    TODO: Replace with actual experimental measurement collection:
      - Dispense formulation based on fractions and thickness
      - Cure/apply coating and run corrosion test
      - Measure corrosion damage (e.g., area loss, mass loss, EIS degradation)
    """
    # Compute dependent stabilizer fraction from composition closure
    stabilizer_fraction = composition_total - (
        resin_fraction + inhibitor_fraction + insulator_fraction
    )
    # Numerical safeguard
    stabilizer_fraction = max(0.0, stabilizer_fraction)

    # Base scale of damage (arbitrary units)
    base_damage = 100.0

    # Film continuity from resin (higher resin improves film integrity)
    # Logistic centered around 0.35 resin with moderate steepness
    resin_continuity = 1.0 / (1.0 + np.exp(-(resin_fraction - 0.35) / 0.06))

    # Ionic transport reduction from insulator (saturating exponential)
    ionic_reduction = 1.0 - np.exp(-4.0 * max(0.0, insulator_fraction))

    # Thickness effect (saturating exponential with diminishing returns)
    thickness_effect = 1.0 - np.exp(-0.03 * max(0.0, thickness_micrometers))

    # Combine barrier effects
    barrier_effect = 0.5 * resin_continuity + 0.5 * ionic_reduction
    barrier_effect *= thickness_effect
    barrier_effect = np.clip(barrier_effect, 0.0, 1.0)

    # Inhibitor chemical protection (saturating)
    inhibitor_effect = 1.0 - np.exp(-6.0 * max(0.0, inhibitor_fraction))
    inhibitor_effect = np.clip(inhibitor_effect, 0.0, 1.0)

    # Core damage model: multiplicative reduction by inhibitor and barrier
    damage = base_damage * (1.0 - 0.55 * inhibitor_effect) * (1.0 - 0.60 * barrier_effect)

    # Mild penalty if stabilizer is very high (dilution of active/binder)
    if stabilizer_fraction > 0.40:
        damage *= 1.0 + 0.10 * (stabilizer_fraction - 0.40) / 0.60

    # Penalty for excessive thickness (risk of cracking/delamination)
    if thickness_micrometers > 160.0:
        damage *= 1.0 + 0.06 * ((thickness_micrometers - 160.0) / 60.0) ** 2

    # Encourage resin dominance (aligned with prior belief); small penalty if barely dominant
    resin_margin = resin_fraction - max(inhibitor_fraction, insulator_fraction)
    if 0.0 <= resin_margin < 0.05:
        damage *= 1.03

    # Add measurement/process noise (heteroscedastic-like scaling)
    if add_noise:
        noise_sigma = 2.5 + 0.02 * damage
        damage = float(damage + RNG.normal(0.0, noise_sigma))

    # Ensure non-negative damage
    damage = float(max(0.0, damage))
    return damage


# Initialize Ax client
ax_client = AxClient()

# Define experiment
ax_client.create_experiment(
    name="anti_corrosion_coating_optimization",
    parameters=[
        # Three free composition variables; stabilizer is computed to close the sum to 1.0
        {"name": "resin_fraction", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "inhibitor_fraction", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "insulator_fraction", "type": "range", "bounds": [0.0, 1.0]},
        # Coating thickness in micrometers (typical lab range; adjust as needed)
        {"name": "thickness_micrometers", "type": "range", "bounds": [5.0, 200.0]},
    ],
    objectives={
        objective_name: ObjectiveProperties(minimize=True),
    },
    parameter_constraints=[
        # Composition closure (stabilizer_fraction = 1 - resin - inhibitor - insulator >= 0)
        "resin_fraction + inhibitor_fraction + insulator_fraction <= 1.0",
        # Order constraints from prior knowledge
        "resin_fraction >= inhibitor_fraction",
        "resin_fraction >= insulator_fraction",
    ],
    overwrite_existing_experiment=True,
)

# Parallel batch size
batch_size = 6

# Optimization loop: multiple parallel batches
n_batches = 20
for _ in range(n_batches):
    parameterizations, optimization_complete = ax_client.get_next_trials(batch_size)
    for trial_index, params in list(parameterizations.items()):
        resin_fraction = params["resin_fraction"]
        inhibitor_fraction = params["inhibitor_fraction"]
        insulator_fraction = params["insulator_fraction"]
        thickness_micrometers = params["thickness_micrometers"]

        # Evaluate objective
        corrosion_damage = evaluate_coating_corrosion_damage(
            resin_fraction=resin_fraction,
            inhibitor_fraction=inhibitor_fraction,
            insulator_fraction=insulator_fraction,
            thickness_micrometers=thickness_micrometers,
            add_noise=True,
        )
        # Report result for the single objective
        ax_client.complete_trial(trial_index=trial_index, raw_data=corrosion_damage)

    if optimization_complete:
        break

# Retrieve best found parameters
best_parameters, best_values = ax_client.get_best_parameters()
best_corrosion_damage = best_values[objective_name]["mean"]

# Compute derived stabilizer for reporting
best_stabilizer_fraction = composition_total - (
    best_parameters["resin_fraction"]
    + best_parameters["inhibitor_fraction"]
    + best_parameters["insulator_fraction"]
)
best_stabilizer_fraction = max(0.0, best_stabilizer_fraction)

print("Best formulation found:")
print(
    f"  resin_fraction={best_parameters['resin_fraction']:.4f}, "
    f"inhibitor_fraction={best_parameters['inhibitor_fraction']:.4f}, "
    f"insulator_fraction={best_parameters['insulator_fraction']:.4f}, "
    f"stabilizer_fraction={best_stabilizer_fraction:.4f}, "
    f"thickness_micrometers={best_parameters['thickness_micrometers']:.2f}"
)
print(f"Estimated corrosion_damage (mean): {best_corrosion_damage:.3f}")

# Plot optimization progress
df = ax_client.get_trials_data_frame()
obj_name = list(ax_client.objective_names)[0]

# Align x-axis by batch number
df = df.reset_index(drop=True)
df["batch_index"] = df.index // batch_size

y_values = df[obj_name].astype(float).values
x_values = df["batch_index"].values
cum_best = np.minimum.accumulate(y_values)

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
ax.scatter(x_values, y_values, edgecolors="k", facecolors="none", label="Observed")
ax.plot(x_values, cum_best, color="#0033FF", lw=2, label="Best-so-far")
ax.set_xlabel("Batch Index")
ax.set_ylabel(obj_name)
ax.set_title("Anti-corrosion Coating Optimization Progress")
ax.legend()
plt.tight_layout()
plt.show()