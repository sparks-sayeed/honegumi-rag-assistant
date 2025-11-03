# Color matching with red/yellow/blue mixtures using Bayesian optimization (Ax Platform)
# %pip install ax-platform==0.4.3 matplotlib
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Tuple

from ax.service.ax_client import AxClient, ObjectiveProperties
import matplotlib.pyplot as plt


# -----------------------------
# Problem setup and simulation
# -----------------------------
# We simulate an 8-channel color sensor as measuring transmittance at 8 wavelengths.
# The transmittance spectra of red, yellow, and blue liquids are modeled with Gaussian
# profiles to be physically plausible. Mixtures follow a Beer-Lambert-like rule:
#     T_mix = T_red^f_red * T_yellow^f_yellow * T_blue^f_blue
# where f_* are volume fractions summing to 1.

RNG = np.random.default_rng(12345)

num_channels: int = 8
wavelengths_nm: np.ndarray = np.linspace(440.0, 720.0, num_channels)  # sensor channels (nm)

def gaussian_spectrum(
    wl_nm: np.ndarray, center_nm: float, width_nm: float, amplitude: float = 0.9, baseline: float = 0.05
) -> np.ndarray:
    """Return a Gaussian-like transmittance spectrum in [baseline, baseline+amplitude]."""
    return baseline + amplitude * np.exp(-0.5 * ((wl_nm - center_nm) / width_nm) ** 2)

# Basis transmittance spectra for the three primary liquids
transmittance_red: np.ndarray = gaussian_spectrum(wavelengths_nm, center_nm=650.0, width_nm=80.0)
transmittance_yellow: np.ndarray = gaussian_spectrum(wavelengths_nm, center_nm=580.0, width_nm=70.0)
transmittance_blue: np.ndarray = gaussian_spectrum(wavelengths_nm, center_nm=460.0, width_nm=60.0)

def mix_transmittance(
    red_fraction: float, yellow_fraction: float, blue_fraction: float
) -> np.ndarray:
    """Compute mixture transmittance spectrum given fractions summing to 1."""
    # Ensure fractions are within [0,1] and sum to 1 numerically
    red = float(np.clip(red_fraction, 0.0, 1.0))
    yellow = float(np.clip(yellow_fraction, 0.0, 1.0))
    blue = float(np.clip(blue_fraction, 0.0, 1.0))
    total = red + yellow + blue
    if total <= 0:
        # Degenerate, return near-opaque (low transmittance)
        return np.full_like(transmittance_red, 0.01)
    # Normalize to enforce sum to 1 if small numeric deviations exist
    red, yellow, blue = red / total, yellow / total, blue / total

    # Beer-Lambert style multiplicative mixing in transmittance domain
    t_mix = (
        (transmittance_red ** red)
        * (transmittance_yellow ** yellow)
        * (transmittance_blue ** blue)
    )
    return np.clip(t_mix, 0.0, 1.0)

def spectrum_mse(a: np.ndarray, b: np.ndarray) -> float:
    """Mean squared error between two spectra."""
    return float(np.mean((a - b) ** 2))

# Define a hidden target color by choosing target fractions, then computing its spectrum.
# Replace these with your actual target fractions or directly with measured target_spectrum if available.
hidden_target_red = 0.25
hidden_target_yellow = 0.55
hidden_target_blue = 1.0 - (hidden_target_red + hidden_target_yellow)

target_spectrum: np.ndarray = mix_transmittance(hidden_target_red, hidden_target_yellow, hidden_target_blue)

def measure_color_difference(
    red_fraction: float,
    yellow_fraction: float,
    blue_fraction: float,
    sensor_noise_std: float = 0.003,
) -> float:
    """Simulate an experimental measurement on the 8-channel sensor and return color difference.

    In a real setup, replace the simulated measurement with:
      - Pump volumes to achieve the requested fractions
      - Read the 8-channel sensor for the mixture
      - Compute the difference between the measured spectrum and the target spectrum

    Args:
        red_fraction: Volume fraction of red liquid (0..1)
        yellow_fraction: Volume fraction of yellow liquid (0..1)
        blue_fraction: Volume fraction of blue liquid (0..1)
        sensor_noise_std: Per-channel Gaussian noise std to model measurement noise

    Returns:
        color_difference: Mean squared error between mixture and target spectra (to minimize)
    """
    # Simulate the sensor reading for the mixed sample
    true_mix = mix_transmittance(red_fraction, yellow_fraction, blue_fraction)
    measured_mix = true_mix + RNG.normal(loc=0.0, scale=sensor_noise_std, size=true_mix.shape)
    measured_mix = np.clip(measured_mix, 0.0, 1.0)

    # Compute difference to target spectrum
    return spectrum_mse(measured_mix, target_spectrum)


# -----------------------------
# Ax experiment configuration
# -----------------------------
objective_name = "color_difference"
composition_total = 1.0  # red + yellow + blue must sum to 1.0

ax_client = AxClient()

# We reparameterize the composition constraint by optimizing over red and yellow,
# while enforcing red + yellow <= 1. Then blue = 1 - (red + yellow) at evaluation.
ax_client.create_experiment(
    name="rgb_liquid_color_matching",
    parameters=[
        {"name": "red_fraction", "type": "range", "bounds": [0.0, composition_total]},
        {"name": "yellow_fraction", "type": "range", "bounds": [0.0, composition_total]},
    ],
    objectives={
        objective_name: ObjectiveProperties(minimize=True),
    },
    parameter_constraints=[
        f"red_fraction + yellow_fraction <= {composition_total}",
    ],
)

# -----------------------------
# Optimization loop
# -----------------------------
total_trials: int = 30  # budget
for i in range(total_trials):
    parameterization, trial_index = ax_client.get_next_trial()

    # Extract decision variables
    red_fraction = float(parameterization["red_fraction"])
    yellow_fraction = float(parameterization["yellow_fraction"])
    blue_fraction = composition_total - (red_fraction + yellow_fraction)  # enforce composition

    try:
        result = measure_color_difference(
            red_fraction=red_fraction,
            yellow_fraction=yellow_fraction,
            blue_fraction=blue_fraction,
            sensor_noise_std=0.003,  # tune to reflect your sensor's noise
        )
        ax_client.complete_trial(trial_index=trial_index, raw_data=result)
    except Exception as e:
        ax_client.log_trial_failure(trial_index=trial_index)
        raise e

best_parameters, metrics = ax_client.get_best_parameters()

# Compute full composition including blue for the best parameters
best_red = float(best_parameters["red_fraction"])
best_yellow = float(best_parameters["yellow_fraction"])
best_blue = composition_total - (best_red + best_yellow)

print("Best mixture fractions found (sum to 1.0):")
print(f"  red_fraction   = {best_red:.4f}")
print(f"  yellow_fraction= {best_yellow:.4f}")
print(f"  blue_fraction  = {best_blue:.4f}")
print(f"Best observed {objective_name}: {metrics[objective_name]:.6f}")

# -----------------------------
# Plot results
# -----------------------------
df = ax_client.get_trials_data_frame()
if objective_name in df.columns:
    y = df[objective_name]
else:
    # Fallback for potential column naming differences
    y = df.filter(like=objective_name).iloc[:, 0]

fig, ax = plt.subplots(figsize=(6, 4), dpi=150)
ax.scatter(df.index, y, ec="k", fc="none", label="Observed")
ax.plot(df.index, np.minimum.accumulate(y), color="#0033FF", lw=2, label="Best so far")
ax.set_xlabel("Trial Number")
ax.set_ylabel(objective_name)
ax.set_title("Color matching optimization progress")
ax.legend()
plt.tight_layout()
plt.show()