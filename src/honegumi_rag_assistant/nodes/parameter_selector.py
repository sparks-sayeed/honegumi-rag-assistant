"""
Node: Parameter selection agent.

This node is responsible for translating the user's natural language
problem description into a structured set of Bayesian optimisation
parameters.  

This node implements a two-stage extraction process:
1. Extract problem structure (parameters, objectives, constraints)
2. Select grid parameters based on the extracted structure

The heavy lifting is delegated to :class:`~honegumi_rag_assistant.extractors.ProblemStructureExtractor`
and :class:`~honegumi_rag_assistant.extractors.ParameterExtractor`, which invoke
language models via the OpenAI function calling interface.

The output of this node is merged into the global state under the
``bo_params`` key.  If an error occurs during extraction, the error
message is added to the ``error`` field of the state.
"""

from __future__ import annotations

from typing import Dict, Any

from ..states import HonegumiRAGState
from ..extractors import ProblemStructureExtractor, ParameterExtractor
from ..app_config import settings
from ..timing_utils import time_node


class ParameterSelector:
    """A LangGraph node that selects optimisation parameters via LLM.

    This node accepts the current pipeline state and extracts the
    ``problem`` field to construct a prompt.  It uses a two-stage process:
    
    1. Extract problem structure (parameters, objectives, constraints)
    2. Select grid parameters based on extracted structure
    
    This approach improves accuracy by forcing explicit reasoning about
    problem elements before making grid selections.
    
    Any returned errors are propagated into the ``error`` field of the state.
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
            print("Analyzing problem structure...")
        
        # Stage 1: Extract problem structure with retry logic
        max_retries = 2
        problem_structure = None
        
        for attempt in range(max_retries):
            structure_result = ProblemStructureExtractor.invoke(problem_description)
            
            if "error" in structure_result and structure_result["error"]:
                if attempt == max_retries - 1:
                    return {"bo_params": None, "error": structure_result["error"]}
                continue
            
            problem_structure = structure_result.get("problem_structure")
            
            # Validate that we got meaningful structure
            if problem_structure:
                num_params = len(problem_structure.get('search_space', []))
                num_objectives = len(problem_structure.get('objective', []))
                
                # Check if extraction is reasonable (at least has objectives)
                if num_objectives > 0:
                    break  # Good extraction, proceed
                elif attempt < max_retries - 1:
                    if settings.debug:
                        print(f"\n⚠️ Stage 1 extraction incomplete (attempt {attempt + 1}/{max_retries}): {num_objectives} objectives, {num_params} parameters. Retrying...")
                    continue
        
        # If we still have empty/invalid structure after retries, proceed with warning
        if problem_structure:
            num_params = len(problem_structure.get('search_space', []))
            num_objectives = len(problem_structure.get('objective', []))
            if num_objectives == 0 and settings.debug:
                print("\n⚠️ WARNING: Stage 1 extraction may be incomplete. Proceeding with Stage 2...")
        
        # Debug: Print extracted structure
        if settings.debug:
            print("\n" + "="*80)
            print("DEBUG: STAGE 1 - PROBLEM STRUCTURE (Solution Format)")
            print("="*80)
            if problem_structure:
                print(f"\nSEARCH SPACE ({len(problem_structure.get('search_space', []))} parameters):")
                for p in problem_structure.get('search_space', []):
                    bounds_info = f"{p.get('bounds', p.get('categories', 'N/A'))}"
                    units = f" ({p.get('units')})" if p.get('units') else ""
                    print(f"  - {p['name']} [{p['type']}]: {bounds_info}{units}")
                
                print(f"\nOBJECTIVES ({len(problem_structure.get('objective', []))}):")
                for o in problem_structure.get('objective', []):
                    threshold_info = f", threshold: {o['threshold']}" if o.get('threshold') else ""
                    units = f" ({o.get('units')})" if o.get('units') else ""
                    print(f"  - {o['name']} ({o['goal']}){threshold_info}{units}")
                
                print(f"\nCONSTRAINTS ({len(problem_structure.get('constraints', []))}):")
                if problem_structure.get('constraints'):
                    for c in problem_structure.get('constraints', []):
                        total_info = f" (total: {c.get('total')})" if c.get('total') is not None else ""
                        print(f"  - {c['type']}{total_info}: {c['description']}")
                        print(f"    Parameters: {', '.join(c['parameters'])}")
                else:
                    print("  (none)")
                
                print(f"\nEXPERIMENTAL SETUP:")
                print(f"  - Budget: {problem_structure.get('budget', 'Not specified')}")
                print(f"  - Batch size: {problem_structure.get('batch_size', 'Sequential (1)')}")
                print(f"  - Noise model: {problem_structure.get('noise_model', True)}")
                print(f"  - Historical data points: {problem_structure.get('historical_data_points', 0)}")
                print(f"  - Model preference: {problem_structure.get('model_preference', 'Default')}")
            else:
                print("Error: No problem structure extracted")
            print("="*80 + "\n")
        
        if not settings.debug:
            print("Selecting optimization grid parameters...")
        
        # Stage 2: Select grid parameters based on structure
        result = ParameterExtractor.invoke(problem_description, problem_structure)
        
        # Debug: Print extracted parameters immediately
        if settings.debug:
            print("\n" + "="*80)
            print("DEBUG: STAGE 2 - GRID PARAMETERS")
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
