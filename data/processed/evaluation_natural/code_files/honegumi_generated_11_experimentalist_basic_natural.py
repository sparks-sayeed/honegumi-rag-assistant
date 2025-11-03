# Bayesian optimization for discovering high-conductivity MAX-phase materials
# %pip install ax-platform==0.4.3 matplotlib
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ax.service.ax_client import AxClient, ObjectiveProperties


# Domain: MAX phases (M_{n+1}AX_n) with M = early transition metal, A = A-group element, X = C or N, n in {1,2,3,4}
# Goal: Maximize electrical conductivity (S/m)
# Budget: 30 sequential trials, noisy measurements

# Candidate sets (customize as needed for your lab/materials availability)
M_CATIONS = ["Ti", "V", "Cr", "Nb", "Ta", "Mo"]
A_ELEMENTS = ["Al", "Si", "Ge", "Ga", "In", "Sn"]
X_ANIONS = ["C", "N"]
N_CHOICES = [1, 2, 3, 4]

# Minimal element property table for featurization (Pauling EN, atomic number, covalent radius pm)
# Values approximate; extend as needed
ELEMENT_PROPS = {
    # M-sublattice (early transition metals)
    "Ti": {"Z": 22, "EN": 1.54, "R": 147},
    "V": {"Z": 23, "EN": 1.63, "R": 134},
    "Cr": {"Z": 24, "EN": 1.66, "R": 128},
    "Nb": {"Z": 41, "EN": 1.60, "R": 146},
    "Ta": {"Z": 73, "EN": 1.50, "R": 146},
    "Mo": {"Z": 42, "EN": 2.16, "R": 139},
    # A-sublattice (A-group elements)
    "Al": {"Z": 13, "EN": 1.61, "R": 118},
    "Si": {"Z": 14, "EN": 1.90, "R": 111},
    "Ge": {"Z": 32, "EN": 2.01, "R": 122},
    "Ga": {"Z": 31, "EN": 1.81, "R": 122},
    "In": {"Z": 49, "EN": 1.78, "R": 142},
    "Sn": {"Z": 50, "EN": 1.96, "R": 139},
    # X-sublattice (C or N)
    "C": {"Z": 6, "EN": 2.55, "R": 77},
    "N": {"Z": 7, "EN": 3.04, "R": 75},
}

# Known high-conductivity references (S/m) to anchor the synthetic ground truth
# Values are representative order-of-magnitude for metallic MAX phases; adjust if you have measured data
KNOWN_SIGMA = {
    ("Ti", "Si", "C", 2): 4.8e6,   # Ti3SiC2 (n=2)
    ("Ti", "Al", "C", 1): 2.5e6,   # Ti2AlC (n=1)
    ("Nb", "Al", "C", 2): 4.0e6,   # Nb3AlC2
    ("V", "Al", "C", 2): 3.6e6,    # V3AlC2
    ("Cr", "Ge", "C", 2): 3.3e6,   # Cr3GeC2
    ("Mo", "Si", "C", 2): 4.2e6,   # Mo3SiC2
    ("Ta", "Al", "C", 2): 4.3e6,   # Ta3AlC2
}

# Optional CSV with historical data (columns: M_element,A_element,X_element,n,electrical_conductivity)
# If present, these values will override the synthetic "ground truth" for matching compositions.
HISTORICAL_CSV_PATH = "max_phase_database.csv"

# Reproducibility for noise
_rng = np.random.default_rng(seed=12345)
random.seed(12345)


def _stoichiometric_counts(n: int):
    # For M_{n+1}AX_n
    m_count = n + 1
    a_count = 1
    x_count = n
    total = m_count + a_count + x_count
    return m_count, a_count, x_count, total


def _featurize_max_phase(M_element: str, A_element: str, X_element: str, n: int) -> np.ndarray:
    # Build simple physically-motivated features
    m = ELEMENT_PROPS[M_element]
    a = ELEMENT_PROPS[A_element]
    x = ELEMENT_PROPS[X_element]

    m_count, a_count, x_count, tot = _stoichiometric_counts(n)

    # Base elemental properties
    feats = {
        "n": float(n),
        "M_Z": float(m["Z"]),
        "A_Z": float(a["Z"]),
        "X_Z": float(x["Z"]),
        "M_EN": float(m["EN"]),
        "A_EN": float(a["EN"]),
        "X_EN": float(x["EN"]),
        "M_R": float(m["R"]),
        "A_R": float(a["R"]),
        "X_R": float(x["R"]),
        # Stoichiometric ratios
        "M_ratio": m_count / tot,
        "A_ratio": a_count / tot,
        "X_ratio": x_count / tot,
        # Pairwise EN differences
        "dEN_MA": abs(m["EN"] - a["EN"]),
        "dEN_MX": abs(m["EN"] - x["EN"]),
        "dEN_AX": abs(a["EN"] - x["EN"]),
        # Weighted averages
        "Z_avg": (m["Z"] * m_count + a["Z"] * a_count + x["Z"] * x_count) / tot,
        "EN_avg": (m["EN"] * m_count + a["EN"] * a_count + x["EN"] * x_count) / tot,
        "R_avg": (m["R"] * m_count + a["R"] * a_count + x["R"] * x_count) / tot,
        # Binary indicator for carbide vs nitride
        "is_carbide": 1.0 if X_element == "C" else 0.0,
    }

    return np.array(list(feats.values()), dtype=float)


def _synthetic_ground_truth_sigma(M_element: str, A_element: str, X_element: str, n: int) -> float:
    # If in known references, return that deterministic value
    if (M_element, A_element, X_element, n) in KNOWN_SIGMA:
        base_sigma = KNOWN_SIGMA[(M_element, A_element, X_element, n)]
    else:
        # Build a smooth synthetic mapping from features to sigma (S/m)
        # Encourages heavier M, carbide systems, moderate EN mismatch, and n around 2–3
        m = ELEMENT_PROPS[M_element]
        a = ELEMENT_PROPS[A_element]
        x = ELEMENT_PROPS[X_element]

        # Normalize some quantities
        m_z_norm = (m["Z"] - 20) / 60.0  # ~0 to ~1 in our set
        en_mx = abs(m["EN"] - x["EN"])
        en_mx_term = 1.0 - np.tanh(0.6 * (en_mx - 0.8))  # favor smaller EN mismatch but not too small
        n_preference = 1.0 - (abs(n - 2.5) / 2.5)  # peak near n≈2–3

        carbide_bonus = 0.7 if X_element == "C" else 0.35
        a_penalty = 0.15 * np.tanh(max(a["EN"] - 1.9, 0))  # slight penalty for very electronegative A

        base_sigma = (
            1.2e6
            + 1.6e6 * m_z_norm
            + 0.9e6 * en_mx_term
            + 0.8e6 * n_preference
            + 0.6e6 * carbide_bonus
            - 0.25e6 * a_penalty
        )

    # Add modest heteroscedastic noise (to simulate experimental noise)
    noise_scale = max(0.03 * base_sigma, 3e4)  # at least 3e4 S/m noise
    sigma_noisy = float(base_sigma + _rng.normal(0.0, noise_scale))
    # Physical lower bound: non-negative
    return max(sigma_noisy, 1e4)


def _load_historical_database(path: str) -> pd.DataFrame | None:
    if not os.path.isfile(path):
        return None
    try:
        df = pd.read_csv(path)
        required_cols = {"M_element", "A_element", "X_element", "n", "electrical_conductivity"}
        if not required_cols.issubset(set(df.columns)):
            return None
        # Coerce column types
        df = df.copy()
        df["M_element"] = df["M_element"].astype(str)
        df["A_element"] = df["A_element"].astype(str)
        df["X_element"] = df["X_element"].astype(str)
        df["n"] = df["n"].astype(int)
        df["electrical_conductivity"] = df["electrical_conductivity"].astype(float)
        return df
    except Exception:
        return None


HIST_DF = _load_historical_database(HISTORICAL_CSV_PATH)


def evaluate_max_conductivity(M_element: str, A_element: str, X_element: str, n: int) -> float:
    """
    Evaluate electrical conductivity (S/m) for MAX-phase M_{n+1}AX_n.

    If a historical database row exists for this composition, return its measured conductivity.
    Otherwise, return a synthetic 'ground truth' based on physically-inspired features to enable
    an immediately runnable demo. Replace this logic with actual measurement collection in practice.
    """
    # Check stoichiometry validity: parameters are constructed to always follow M_{n+1}AX_n
    if M_element not in M_CATIONS or A_element not in A_ELEMENTS or X_element not in X_ANIONS or int(n) not in N_CHOICES:
        # Invalid definition—shouldn't happen given the search space.
        return 1e4

    # Use historical measurement if available
    if HIST_DF is not None:
        row = HIST_DF[
            (HIST_DF["M_element"] == M_element)
            & (HIST_DF["A_element"] == A_element)
            & (HIST_DF["X_element"] == X_element)
            & (HIST_DF["n"] == int(n))
        ]
        if len(row) > 0:
            # Use the first match or average if multiple entries
            return float(row["electrical_conductivity"].mean())

    # Fall back to synthetic ground truth for immediate executability
    return _synthetic_ground_truth_sigma(M_element, A_element, X_element, int(n))


# Set up Ax optimization
ax_client = AxClient()

ax_client.create_experiment(
    name="discover_max_phase_high_conductivity",
    parameters=[
        {
            "name": "M_element",
            "type": "choice",
            "is_ordered": False,
            "values": M_CATIONS,
        },
        {
            "name": "A_element",
            "type": "choice",
            "is_ordered": False,
            "values": A_ELEMENTS,
        },
        {
            "name": "X_element",
            "type": "choice",
            "is_ordered": False,
            "values": X_ANIONS,
        },
        {
            "name": "n",
            "type": "choice",
            "is_ordered": True,
            "values": N_CHOICES,
            "value_type": "int",
        },
    ],
    objectives={
        "electrical_conductivity": ObjectiveProperties(minimize=False),
    },
    # Stoichiometry is implicitly captured by parameterization of n in M_{n+1}AX_n.
    # No linear parameter constraints are necessary here.
)

# Run optimization loop
N_TRIALS = 30  # Budget

for _ in range(N_TRIALS):
    parameters, trial_index = ax_client.get_next_trial()

    try:
        M_el = parameters["M_element"]
        A_el = parameters["A_element"]
        X_el = parameters["X_element"]
        n_val = int(parameters["n"])

        sigma = evaluate_max_conductivity(M_el, A_el, X_el, n_val)

        # Provide a single mean; Ax will infer noise from data (no SEM supplied)
        ax_client.complete_trial(trial_index=trial_index, raw_data=sigma)
    except Exception:
        # Ensure failed trials are not re-proposed
        ax_client.log_trial_failure(trial_index=trial_index)

# Retrieve best suggestion
best_parameters, best_metrics = ax_client.get_best_parameters()

print("Best parameters found:")
print(best_parameters)
print("Model-predicted metrics at best parameters:")
print(best_metrics)

# Plot optimization trace (maximize)
objective_name = ax_client.objective_names[0]
df = ax_client.get_trials_data_frame()

fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
ax.scatter(df.index, df[objective_name], ec="k", fc="none", label="Observed")
ax.plot(
    df.index,
    np.maximum.accumulate(df[objective_name]),
    color="#0072B2",
    lw=2,
    label="Best so far",
)
ax.set_xlabel("Trial Number")
ax.set_ylabel(f"{objective_name} (S/m)")
ax.set_title("Optimization Trace: MAX-phase Electrical Conductivity")
ax.grid(True, ls="--", alpha=0.3)
ax.legend()
plt.tight_layout()
plt.show()