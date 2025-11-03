import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain-specific objective names
CONDUCTIVITY_METRIC = "electrical_conductivity_s_per_m"
PRINTABILITY_METRIC = "printability_index"

# Total composition must sum to 1.0
TOTAL_FRACTION = 1.0

rng = np.random.default_rng(seed=42)


def evaluate_conductive_ink(
    conductive_filler_fraction: float,
    polymer_binder_fraction: float,
    solvent_fraction: float,
    rheology_modifier_fraction: float,
) -> dict:
    """
    Simulated evaluation of a conductive ink formulation with four components:
      - conductive_filler_fraction
      - polymer_binder_fraction
      - solvent_fraction
      - rheology_modifier_fraction
    Fractions must sum to 1.0.

    Returns a dict with:
      - electrical_conductivity_s_per_m: (mean, sem)
      - printability_index: (mean, sem)

    Notes:
      - Replace this simulation with actual lab measurements when available.
      - The simulation is based on plausible percolation and printability heuristics.
    """
    # Numerical safety
    conductive_filler_fraction = max(0.0, float(conductive_filler_fraction))
    polymer_binder_fraction = max(0.0, float(polymer_binder_fraction))
    solvent_fraction = max(0.0, float(solvent_fraction))
    rheology_modifier_fraction = max(0.0, float(rheology_modifier_fraction))

    # Conductivity model (percolation-like on solids)
    solids = max(1e-8, 1.0 - solvent_fraction)
    phi = conductive_filler_fraction / solids  # conductive filler fraction in the dried film
    phi_c = 0.18  # percolation threshold (typical for particulate systems)
    exponent_t = 2.0
    sigma0 = 5.0e6  # S/m, effective upper bound for a packed film (lower than bulk metal)

    if phi <= phi_c or solids <= 0.05:
        base_conductivity = 0.0
    else:
        base_conductivity = sigma0 * ((phi - phi_c) / (1.0 - phi_c)) ** exponent_t

    # Connectivity penalty from polymer and rheology modifier (insulating phases)
    connectivity_penalty = np.exp(-2.0 * (polymer_binder_fraction + rheology_modifier_fraction) / max(1e-8, solids))
    conductivity = base_conductivity * connectivity_penalty

    # Additional smooth penalties:
    # - Excessively high filler in wet ink may cause flake bridges / gaps on drying
    high_filler_penalty = 1.0 / (1.0 + np.exp(20.0 * (conductive_filler_fraction - 0.62)))
    conductivity *= high_filler_penalty

    # Printability model (dimensionless index ~ [0,100])
    # Preference for certain regions:
    # - Solids content around ~0.6 is often printable
    # - Binder ~0.15 for adhesion/film formation
    # - Rheology ~0.05 for line edge definition
    # - Solvent ~0.35 for open time and wetting
    def gaussian_pref(x, mu, sigma):
        return float(np.exp(-0.5 * ((x - mu) / max(1e-8, sigma)) ** 2))

    solids_pref = gaussian_pref(solids, 0.60, 0.15)
    binder_pref = gaussian_pref(polymer_binder_fraction, 0.15, 0.07)
    rheology_pref = gaussian_pref(rheology_modifier_fraction, 0.05, 0.03)
    solvent_pref = gaussian_pref(solvent_fraction, 0.35, 0.10)

    # Penalize filler extremes for nozzle clogging and poor flow
    filler_low_penalty = 1.0 / (1.0 + np.exp(-30.0 * (conductive_filler_fraction - 0.08)))
    filler_high_penalty = 1.0 / (1.0 + np.exp(30.0 * (conductive_filler_fraction - 0.52)))
    filler_penalty = filler_low_penalty * filler_high_penalty

    # Aggregate printability score
    printability_score = 100.0 * solids_pref * binder_pref * rheology_pref * solvent_pref * filler_penalty

    # Add stochastic experimental noise and report SEM (standard error of mean)
    cond_sem = 0.08 * max(conductivity, 1e4)  # relative uncertainty with a floor
    printable_sem = 3.0  # points on 0-100 scale

    conductivity_meas = max(0.0, rng.normal(loc=conductivity, scale=cond_sem))
    printability_meas = float(np.clip(rng.normal(loc=printability_score, scale=printable_sem), 0.0, 100.0))

    return {
        CONDUCTIVITY_METRIC: (conductivity_meas, cond_sem),
        PRINTABILITY_METRIC: (printability_meas, printable_sem),
    }


def main():
    ax_client = AxClient()

    # Reparameterization for composition constraint:
    # We optimize first three components directly; the fourth is computed as the remainder.
    # conductive_filler + polymer_binder + solvent + rheology_modifier == 1.0
    # => rheology_modifier = 1.0 - (conductive_filler + polymer_binder + solvent)
    ax_client.create_experiment(
        name="conductive_ink_formulation",
        parameters=[
            {
                "name": "conductive_filler_fraction",
                "type": "range",
                "bounds": [0.0, TOTAL_FRACTION],
            },
            {
                "name": "polymer_binder_fraction",
                "type": "range",
                "bounds": [0.0, TOTAL_FRACTION],
            },
            {
                "name": "solvent_fraction",
                "type": "range",
                "bounds": [0.0, TOTAL_FRACTION],
            },
        ],
        objectives={
            CONDUCTIVITY_METRIC: ObjectiveProperties(minimize=False),
            PRINTABILITY_METRIC: ObjectiveProperties(minimize=False),
        },
        # Linear inequality ensures the computed remainder is >= 0.
        parameter_constraints=[
            f"conductive_filler_fraction + polymer_binder_fraction + solvent_fraction <= {TOTAL_FRACTION}"
        ],
    )

    NUM_TRIALS = 45

    for _ in range(NUM_TRIALS):
        parameters, trial_index = ax_client.get_next_trial()

        filler = parameters["conductive_filler_fraction"]
        binder = parameters["polymer_binder_fraction"]
        solvent = parameters["solvent_fraction"]
        rheology = TOTAL_FRACTION - (filler + binder + solvent)

        results = evaluate_conductive_ink(
            conductive_filler_fraction=filler,
            polymer_binder_fraction=binder,
            solvent_fraction=solvent,
            rheology_modifier_fraction=rheology,
        )

        ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    # Retrieve Pareto-optimal results (based on observed data)
    pareto = ax_client.get_pareto_optimal_parameters(use_model_predictions=False)

    # Construct a DataFrame with Pareto formulations (including the computed remainder)
    pareto_rows = []
    for params, (means, cov) in pareto.items():
        filler = params["conductive_filler_fraction"]
        binder = params["polymer_binder_fraction"]
        solvent = params["solvent_fraction"]
        rheology = TOTAL_FRACTION - (filler + binder + solvent)
        row = {
            "conductive_filler_fraction": filler,
            "polymer_binder_fraction": binder,
            "solvent_fraction": solvent,
            "rheology_modifier_fraction": rheology,
            CONDUCTIVITY_METRIC: means[CONDUCTIVITY_METRIC],
            PRINTABILITY_METRIC: means[PRINTABILITY_METRIC],
        }
        pareto_rows.append(row)
    pareto_df = pd.DataFrame(pareto_rows).sort_values(CONDUCTIVITY_METRIC, ascending=False)
    print("Top 5 Pareto-optimal formulations (by conductivity):")
    print(pareto_df.head(5).to_string(index=False))

    # Plot all observations and Pareto front
    df = ax_client.get_trials_data_frame()
    objectives = ax_client.objective_names

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    # Observed points
    try:
        ax.scatter(df[objectives[0]], df[objectives[1]], fc="None", ec="k", label="Observed")
    except Exception:
        # Fallback if column names are not directly accessible (robustness for different Ax versions)
        obs = df.pivot_table(index="trial_index", columns="metric_name", values="mean")
        ax.scatter(obs[objectives[0]], obs[objectives[1]], fc="None", ec="k", label="Observed")

    # Pareto frontier (observed)
    pareto_means = [means for _, (means, _) in pareto.items()]
    if len(pareto_means) > 0:
        p_df = pd.DataFrame(pareto_means)
        p_df = p_df.sort_values(objectives[0], ascending=True)
        ax.plot(p_df[objectives[0]], p_df[objectives[1]], color="#0033FF", lw=2, label="Pareto Front")

    ax.set_xlabel(objectives[0])
    ax.set_ylabel(objectives[1])
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()