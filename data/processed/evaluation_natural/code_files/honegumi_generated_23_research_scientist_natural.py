# Generated from a Honegumi skeleton, adapted to neural network hyperparameter optimization
# Requirements:
#   pip install ax-platform==0.4.3 matplotlib numpy pandas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Any
from ax.service.ax_client import AxClient, ObjectiveProperties


OBJECTIVE_NAME = "validation_accuracy"


def evaluate_neural_network_generalization(
    learning_rate: float,
    weight_decay: float,
    dropout_rate: float,
    rng: np.random.RandomState,
) -> Dict[str, Tuple[float, float]]:
    """
    Simulated evaluation of a neural network training run.

    This function mimics how validation accuracy depends on three common hyperparameters:
    - learning_rate: step size for optimizer (log-sensitive)
    - weight_decay: L2 regularization strength (log-sensitive)
    - dropout_rate: probability of dropout during training

    The "true" relationship below is a smooth synthetic surface with a single broad optimum
    and realistic noise to model run-to-run variability. Replace this function with actual
    training code to measure validation accuracy on your dataset.

    Returns:
      dict mapping metric name to (mean, sem) as required by Ax to model observation noise.
    """

    # "Best" region for this synthetic task
    opt_log_lr = np.log10(2e-3)      # around 2e-3
    opt_log_wd = np.log10(3e-4)      # around 3e-4
    opt_do = 0.20                    # around 0.2

    # Convert to log-space where appropriate
    log_lr = np.log10(learning_rate)
    log_wd = np.log10(weight_decay)

    # Quadratic penalties around the optimum in each dimension
    penalty_lr = (log_lr - opt_log_lr) ** 2
    penalty_wd = (log_wd - opt_log_wd) ** 2
    penalty_do = (dropout_rate - opt_do) ** 2

    # Base accuracy and penalties; coefficients set to yield a realistic landscape
    base = 0.92
    acc = base - 0.05 * penalty_lr - 0.035 * penalty_wd - 0.045 * penalty_do

    # Mild interaction effects to make the surface non-separable
    if learning_rate > 5e-3 and dropout_rate < 0.1:
        acc -= 0.01
    if (learning_rate < 5e-4) and (weight_decay > 2e-3):
        acc -= 0.008
    acc -= 0.003 * np.cos(3.0 * dropout_rate * np.pi)  # smooth periodicity with dropout

    # Add observation noise to model stochastic training outcomes (noisy objective)
    # More dropout reduces variance slightly; larger LR increases variance
    noise_sd = 0.004 + 0.003 * (np.log10(learning_rate) - opt_log_lr) ** 2 + 0.002 * (0.25 - min(dropout_rate, 0.5))
    noise_sd = max(0.003, float(noise_sd))
    observed = float(acc + rng.normal(loc=0.0, scale=noise_sd))

    # Clip to [0, 1] valid accuracy range
    observed = float(np.clip(observed, 0.0, 1.0))

    # Report mean with an estimated standard error; Ax uses SEM to fit a noise model.
    return {OBJECTIVE_NAME: (observed, noise_sd)}


def main() -> None:
    rng = np.random.RandomState(2025)

    ax_client = AxClient(random_seed=2025)

    # Define the search space for three hyperparameters:
    # - learning_rate: continuous, log-scaled
    # - weight_decay: continuous, log-scaled
    # - dropout_rate: continuous, linear scale
    ax_client.create_experiment(
        name="nn_hyperparameter_optimization",
        parameters=[
            {
                "name": "learning_rate",
                "type": "range",
                "bounds": [1e-5, 1e-1],
                "log_scale": True,
            },
            {
                "name": "weight_decay",
                "type": "range",
                "bounds": [1e-6, 1e-2],
                "log_scale": True,
            },
            {
                "name": "dropout_rate",
                "type": "range",
                "bounds": [0.0, 0.7],
            },
        ],
        objectives={
            OBJECTIVE_NAME: ObjectiveProperties(minimize=False),
        },
        # No additional outcome constraints; single task; default GP model with noise
    )

    num_trials = 40  # Budget: 40 trials
    observed_values = []  # Track observed validation accuracy for plotting
    trial_indices = []

    # Sequential (batch size = 1) evaluation loop
    for _ in range(num_trials):
        parameterization, trial_index = ax_client.get_next_trial()

        # Extract parameters
        lr = float(parameterization["learning_rate"])
        wd = float(parameterization["weight_decay"])
        do = float(parameterization["dropout_rate"])

        # Evaluate the training run (replace this with real training + validation)
        results = evaluate_neural_network_generalization(
            learning_rate=lr,
            weight_decay=wd,
            dropout_rate=do,
            rng=rng,
        )

        # Record the observed mean accuracy for visualization
        observed_values.append(results[OBJECTIVE_NAME][0])
        trial_indices.append(trial_index)

        # Report the result (mean, sem) to Ax
        ax_client.complete_trial(trial_index=trial_index, raw_data=results)

    best_parameters, best_values = ax_client.get_best_parameters()

    print("Best hyperparameters found:")
    for k, v in best_parameters.items():
        print(f"  {k}: {v}")

    best_mean = best_values[OBJECTIVE_NAME]["mean"] if isinstance(best_values[OBJECTIVE_NAME], dict) else best_values[OBJECTIVE_NAME]
    print(f"Best estimated {OBJECTIVE_NAME}: {best_mean:.4f}")

    # Visualization: observed validation accuracy per trial and best-so-far curve
    observed_values = np.array(observed_values, dtype=float)
    best_so_far = np.maximum.accumulate(observed_values)

    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    ax.scatter(np.arange(1, num_trials + 1), observed_values, ec="k", fc="none", label="Observed accuracy")
    ax.plot(np.arange(1, num_trials + 1), best_so_far, color="#0033FF", lw=2, label="Best so far")
    ax.set_xlabel("Trial Number")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Neural Network Hyperparameter Optimization (Ax)")
    ax.set_ylim(0.0, 1.0)
    ax.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()