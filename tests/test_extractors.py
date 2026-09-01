"""Tests for the Pydantic schemas that constrain Claude's structured output.

The validators here are the last line of defence against a parameter set that
Honegumi would reject, and against a problem structure so empty that every
downstream stage is working from nothing.
"""

import pytest
from pydantic import ValidationError

from honegumi_rag_assistant.extractors import (
    ObjectiveSpec,
    OptimizationParameters,
    ProblemStructure,
    SearchSpaceParameter,
)


def _valid_params(**overrides):
    """A minimal valid grid selection, with optional field overrides."""
    params = {
        "objective": "Single",
        "model": "Default",
        "task": "Single",
        "existing_data": False,
        "sum_constraint": False,
        "order_constraint": False,
        "linear_constraint": False,
        "composition_constraint": False,
        "categorical": False,
        "custom_threshold": False,
        "synchrony": "Single",
        "visualize": True,
    }
    params.update(overrides)
    return params


class TestOptimizationParameters:
    def test_minimal_valid_selection(self):
        params = OptimizationParameters(**_valid_params())
        assert params.objective == "Single"

    def test_custom_threshold_requires_multi_objective(self):
        """Mirrors Honegumi's is_incompatible: thresholds are multi-objective only."""
        with pytest.raises(ValidationError, match="custom_threshold"):
            OptimizationParameters(**_valid_params(custom_threshold=True))

    def test_custom_threshold_allowed_with_multi_objective(self):
        params = OptimizationParameters(
            **_valid_params(objective="Multi", custom_threshold=True)
        )
        assert params.custom_threshold is True

    def test_rejects_unknown_objective_value(self):
        with pytest.raises(ValidationError):
            OptimizationParameters(**_valid_params(objective="Triple"))

    def test_rejects_unknown_model_value(self):
        with pytest.raises(ValidationError):
            OptimizationParameters(**_valid_params(model="Bayesian-ish"))

    @pytest.mark.parametrize("synchrony", ["Batch", "Single"])
    def test_accepts_both_synchrony_values(self, synchrony):
        assert OptimizationParameters(**_valid_params(synchrony=synchrony)).synchrony == synchrony


class TestProblemStructure:
    def test_requires_at_least_one_objective(self):
        with pytest.raises(ValidationError, match="No objectives found"):
            ProblemStructure(search_space=[], objective=[])

    def test_accepts_structure_with_objective_only(self):
        """An empty search space is tolerated here; the node retries instead."""
        structure = ProblemStructure(
            search_space=[],
            objective=[ObjectiveSpec(name="density", goal="maximize")],
        )
        assert structure.search_space == []
        assert len(structure.objective) == 1

    def test_full_structure_round_trips(self):
        structure = ProblemStructure(
            search_space=[
                SearchSpaceParameter(
                    name="temperature", type="continuous", bounds=[800, 1200], units="C"
                ),
                SearchSpaceParameter(
                    name="atmosphere", type="categorical", categories=["air", "argon"]
                ),
            ],
            objective=[ObjectiveSpec(name="density", goal="maximize", units="g/cm3")],
            budget=25,
            noise_model=True,
        )
        dumped = structure.model_dump()
        assert len(dumped["search_space"]) == 2
        assert dumped["budget"] == 25
        assert dumped["constraints"] == []

    def test_noise_model_defaults_true(self):
        structure = ProblemStructure(
            search_space=[], objective=[ObjectiveSpec(name="yield", goal="maximize")]
        )
        assert structure.noise_model is True

    def test_rejects_invalid_parameter_type(self):
        with pytest.raises(ValidationError):
            SearchSpaceParameter(name="n_layers", type="integer")

    @pytest.mark.parametrize("goal", ["maximize", "minimize"])
    def test_objective_goals(self, goal):
        assert ObjectiveSpec(name="cost", goal=goal).goal == goal
