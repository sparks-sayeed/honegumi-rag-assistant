# Generated for multi-task optimization of ceramic binder systems using Ax (https://ax.dev)
# %pip install ax-platform==0.4.3 matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from ax.core.observation import ObservationFeatures


# Domain-specific objective name
objective_metric_name = "green_strength_mpa"

# Tasks (ceramic matrices)
CERAMIC_MATRICES = ["alumina", "zirconia", "silicon_carbide"]

rng = np.random.default_rng(2025)


def measure_green_strength(parameters: dict) -> dict:
    """
    Simulated multi-task evaluation for green strength (MPa) of ceramic binder systems.
    This stub encodes plausible physics-inspired structure with task-specific biases.
    Replace this function with actual experimental measurements or simulation calls.

    Inputs (in `parameters` dict):
      - polymer_binder_wt_pct: float in [1.0, 8.0]
      - plasticizer_wt_pct: float in [0.0, 4.0]
      - dispersant_wt_pct: float in [0.0, 1.0]
      - solids_loading_pct: float in [45.0, 65.0]
      - mixing_time_min: float in [10.0, 60.0]
      - drying_temperature_c: float in [25.0, 80.0]
      - drying_time_hr: float in [0.5, 8.0]
      - ceramic_matrix: str in {"alumina", "zirconia", "silicon_carbide"}

    Returns:
      { "green_strength_mpa": (mean, sem) }
    """
    # Extract
    b = float(parameters["polymer_binder_wt_pct"])
    p = float(parameters["plasticizer_wt_pct"])
    d = float(parameters["dispersant_wt_pct"])
    sl = float(parameters["solids_loading_pct"])
    mix = float(parameters["mixing_time_min"])
    t_dry = float(parameters["drying_temperature_c"])
    t_hold = float(parameters["drying_time_hr"])
    matrix = parameters["ceramic_matrix"]

    # Task-specific optima and scaling to encode related, but not identical, response surfaces
    # Base optima for each matrix (shared formulation space, small shifts by matrix)
    matrix_mod = {
        "alumina": {
            "binder_opt": 4.0,
            "solids_opt": 58.0,
            "mix_opt": 35.0,
            "temp_opt": 55.0,
            "time_opt": 3.0,
            "base": 28.0,
        },
        "zirconia": {
            "binder_opt": 4.5,
            "solids_opt": 59.0,
            "mix_opt": 38.0,
            "temp_opt": 52.0,
            "time_opt": 3.5,
            "base": 30.0,
        },
        "silicon_carbide": {
            "binder_opt": 3.6,
            "solids_opt": 57.0,
            "mix_opt": 33.0,
            "temp_opt": 58.0,
            "time_opt": 2.7,
            "base": 26.0,
        },
    }
    mm = matrix_mod[matrix]

    # Helper: smooth peaked response around an optimum with tolerance
    def gaussian_peak(x, mu, width):
        return np.exp(-0.5 * ((x - mu) / (width + 1e-8)) ** 2)

    # Binder effect: too little => weak bridges; too much => residual porosity / brittleness
    binder_term = 8.0 * gaussian_peak(b, mm["binder_opt"], width=1.2)

    # Plasticizer: small amount improves flexibility/packing, too much reduces strength
    plasticizer_term = 3.0 * gaussian_peak(p, 1.2, width=0.6) - 1.0 * max(0.0, p - 2.0)

    # Dispersant: improves packing to a point, excessive can cause defects or retard binder cohesion
    dispersant_term = 2.0 * gaussian_peak(d, 0.35, width=0.2) - 1.0 * max(0.0, d - 0.6)

    # Solids loading: higher packing improves contact area up to rheology limit
    solids_term = 10.0 * gaussian_peak(sl, mm["solids_opt"], width=2.5)

    # Mixing time: under-mixed has defects, over-mixed can break down binder network
    mixing_term = 4.0 * gaussian_peak(mix, mm["mix_opt"], width=8.0) - 0.02 * max(0.0, mix - (mm["mix_opt"] + 10))

    # Drying profile: moderate temp/time removes solvent without cracking; too hot/long => defects
    temp_term = 5.0 * gaussian_peak(t_dry, mm["temp_opt"], width=8.0)
    time_term = 3.0 * gaussian_peak(t_hold, mm["time_opt"], width=1.2)

    # Mild interactions: binder x solids synergy and plasticizer aiding mixing dispersion
    interaction_term = 0.04 * b * (sl - 50.0) + 0.01 * p * (mix - 20.0)

    # Aggregate with matrix-specific base strength level
    strength_mean = (
        mm["base"]
        + binder_term
        + plasticizer_term
        + dispersant_term
        + solids_term
        + mixing_term
        + temp_term
        + time_term
        + interaction_term
    )

    # Penalize infeasible-ish regimes (simulate defects):
    # Extremely low solids or excessive binder induce microcracking on drying.
    penalty = 0.0
    if sl < 48.0:
        penalty -= (48.0 - sl) * 0.7
    if b > 7.0:
        penalty -= (b - 7.0) * 0.9
    if t_dry > 70.0 and t_hold > 5.0:
        penalty -= (t_dry - 70.0) * 0.3 + (t_hold - 5.0) * 0.8

    strength_mean += penalty

    # Add structured task correlation (shared latent function + small matrix bias already included)
    # and experimental noise
    noise_sd = 1.2  # MPa, typical experimental measurement noise
    observed = float(strength_mean + rng.normal(0.0, noise_sd))
    sem = noise_sd  # Report SEM; if averaging repeats, reduce accordingly

    # Do not allow negative strength in simulation
    observed = max(0.1, observed)

    return {objective_metric_name: (observed, sem)}


# Configure generation strategy: Sobol initialization then Multi-task GP
num_init_total = 18  # 6 per task for 3 tasks, rule of thumb ~2x params per task
gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=num_init_total,
            max_parallelism=12,
            model_kwargs={"seed": 1234},
            model_gen_kwargs={"deduplicate": True},
        ),
        GenerationStep(
            model=Models.ST_MTGP,
            num_trials=-1,  # Use MTGP for the remainder
            max_parallelism=6,
        ),
    ]
)

ax_client = AxClient(generation_strategy=gs)

# Create experiment: shared formulation space, multi-task via ceramic_matrix
ax_client.create_experiment(
    name="ceramic_binder_green_strength_mtbo",
    parameters=[
        {"name": "polymer_binder_wt_pct", "type": "range", "bounds": [1.0, 8.0]},
        {"name": "plasticizer_wt_pct", "type": "range", "bounds": [0.0, 4.0]},
        {"name": "dispersant_wt_pct", "type": "range", "bounds": [0.0, 1.0]},
        {"name": "solids_loading_pct", "type": "range", "bounds": [45.0, 65.0]},
        {"name": "mixing_time_min", "type": "range", "bounds": [10.0, 60.0]},
        {"name": "drying_temperature_c", "type": "range", "bounds": [25.0, 80.0]},
        {"name": "drying_time_hr", "type": "range", "bounds": [0.5, 8.0]},
        {
            "name": "ceramic_matrix",
            "type": "choice",
            "values": CERAMIC_MATRICES,
            "is_task": True,
            "target_value": "alumina",  # primary target task for selection; others assist via transfer
        },
    ],
    objectives={
        objective_metric_name: ObjectiveProperties(minimize=False),
    },
)

# Plan: 40 trials per task, total 120
total_trials = 120
tasks = CERAMIC_MATRICES  # round-robin to evenly allocate 40 per task

for i in range(total_trials):
    current_task = tasks[i % len(tasks)]

    parameterization, trial_index = ax_client.get_next_trial(
        fixed_features=ObservationFeatures({"ceramic_matrix": current_task})
    )

    # Evaluate green strength for the suggested parameters
    results = measure_green_strength(parameterization)
    ax_client.complete_trial(trial_index=trial_index, raw_data=results)

# Summary: best parameters per task (by observed mean)
df = ax_client.get_trials_data_frame()
# Ensure we only consider completed trials with the objective present
df = df.dropna(subset=["mean"])
df_tasks = {}
for m in CERAMIC_MATRICES:
    df_m = df[df["ceramic_matrix"] == m]
    if len(df_m) == 0:
        continue
    idx_best = df_m["mean"].idxmax()
    best_row = df_m.loc[idx_best]
    df_tasks[m] = best_row

print("Best observed parameters per ceramic matrix (by measured green strength):")
for m, row in df_tasks.items():
    print(f"\nMatrix: {m}")
    print(f"  Best observed green strength (MPa): {row['mean']:.2f}  ± {row.get('sem', np.nan):.2f}")
    # Extract parameters for this arm
    arm_name = row["arm_name"]
    arm_params = ax_client.experiment.arms_by_name[arm_name].parameters
    for k, v in arm_params.items():
        if k == "ceramic_matrix":
            continue
        print(f"    {k}: {v}")

# Plot optimization traces per task
fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=150, sharey=True)
axes = axes.flatten()
for ax, m in zip(axes, CERAMIC_MATRICES):
    df_m = df[df["ceramic_matrix"] == m].sort_values("trial_index")
    if len(df_m) == 0:
        ax.set_title(f"{m}: no data")
        continue
    y = df_m["mean"].to_numpy()
    ax.scatter(range(len(y)), y, ec="k", fc="none", label="Observed")
    ax.plot(np.maximum.accumulate(y), color="#0033FF", lw=2, label="Best to Trial")
    ax.set_title(f"{m}")
    ax.set_xlabel("Trial Number (per task order)")
    ax.set_ylabel("Green strength (MPa)")
    ax.grid(True, alpha=0.3)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=2)
fig.suptitle("Optimization traces by ceramic matrix")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.show()