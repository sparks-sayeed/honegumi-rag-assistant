"""
Node: Parameter selection agent.

This node is responsible for translating the user’s natural language
problem description into a structured set of Bayesian optimisation
parameters.  It delegates the heavy lifting to the
:class:`~honegumi_rag_assistant.extractors.ParameterExtractor`, which
invokes a language model via the OpenAI function calling interface.

The output of this node is merged into the global state under the
``bo_params`` key.  If an error occurs during extraction, the error
message is added to the ``error`` field of the state.
"""

from __future__ import annotations

from typing import Dict, Any

from ..states import HonegumiRAGState
from ..extractors import ParameterExtractor
from ..app_config import settings
from ..timing_utils import time_node


class ParameterSelector:
    """A LangGraph node that selects optimisation parameters via LLM.

    This node accepts the current pipeline state and extracts the
    ``problem`` field to construct a prompt.  It then calls the
    :class:`ParameterExtractor` to obtain a dictionary of optimisation
    parameters.  Any returned errors are propagated into the
    ``error`` field of the state.
    """

    @staticmethod
    @time_node("Parameter Selector")
    def select_parameters(state: HonegumiRAGState) -> Dict[str, Any]:
        """Select optimisation parameters for the given problem.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state containing at least a ``problem`` key.

        Returns
        -------
        Dict[str, Any]
            A dictionary with updated keys for the pipeline state.  At
            minimum this will include a ``bo_params`` entry and may
            contain an ``error`` entry if something goes wrong.
        """
        problem_description = state.get("problem", "").strip()
        if not problem_description:
            return {"error": "No problem description provided to parameter selector."}

        if not settings.debug:
            print("Analyzing problem and selecting optimization parameters...")
        
        result = ParameterExtractor.invoke(problem_description)
        
        # Debug: Print extracted parameters immediately
        if settings.debug:
            print("\n" + "="*80)
            print("DEBUG: EXTRACTED PARAMETERS")
            print("="*80)
            if "error" in result and result["error"]:
                print(f"Error: {result['error']}")
            elif result.get("bo_params"):
                for key, value in result["bo_params"].items():
                    print(f"  {key}: {value}")
            else:
                print("Error: No parameters extracted (bo_params is None)")
            print("="*80 + "\n")
        
        return result