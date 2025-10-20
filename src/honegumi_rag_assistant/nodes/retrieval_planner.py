"""
Node: Retrieval Planner Agent.

This agent analyzes the problem description and Honegumi skeleton to determine
what additional information is needed from the Ax documentation. It can generate
up to 7 specific queries that will be executed in parallel via fan-out subgraphs.

The planner acts as an intelligent information needs analyzer that decides:
1. Does the skeleton provide enough structure, or do we need more details?
2. What specific aspects of Ax Platform need clarification?
3. How to formulate focused queries for maximum relevance?
"""

from __future__ import annotations

from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from ..states import HonegumiRAGState
from ..app_config import settings
from ..timing_utils import time_node


class RetrievalQuery(BaseModel):
    """A single focused query for the vector database."""
    
    query: str = Field(
        description=(
            "A SHORT, focused question about Ax Platform (under 20 words). "
            "Examples: 'How to define parameter constraints?', "
            "'Multi-objective optimization with custom thresholds?', "
            "'Setting up SAASBO acquisition function?'"
        )
    )


class RetrievalPlan(BaseModel):
    """Decision about what to retrieve from Ax documentation."""
    
    action: Literal["skip_retrieval", "retrieve"] = Field(
        description=(
            "Choose 'skip_retrieval' if the Honegumi skeleton provides enough "
            "structure and no additional Ax documentation is needed. "
            "Choose 'retrieve' if specific Ax Platform details are unclear."
        )
    )
    
    queries: List[RetrievalQuery] = Field(
        default_factory=list,
        max_length=7,
        description=(
            "If action='retrieve', provide 1-7 specific questions. "
            "Each query should target a different aspect of the implementation. "
            "Keep queries SHORT and focused for better semantic search results."
        )
    )


class RetrievalPlannerAgent:
    """Plan retrieval strategy by analyzing information needs."""

    @staticmethod
    @time_node("Retrieval Planner")
    def plan_retrieval(state: HonegumiRAGState) -> Dict[str, Any]:
        """Analyze problem and skeleton to determine retrieval needs.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state containing ``problem``, ``bo_params``,
            and ``skeleton_code``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with:
            - ``retrieval_queries``: List[str] of queries to execute in parallel
            - Empty list if no retrieval needed
        """
        problem = state.get("problem", "")
        bo_params = state.get("bo_params", {})
        skeleton = state.get("skeleton_code", "")
        
        if settings.debug:
            print(f"\n[RETRIEVAL PLANNER START]")
            print(f"Problem length: {len(problem)} chars")
            print(f"Skeleton length: {len(skeleton)} chars")
            print(f"Parameters: {list(bo_params.keys())}")
        
        if not settings.openai_api_key:
            if settings.debug:
                print("[RETRIEVAL PLANNER] No API key, skipping retrieval\n")
            return {"retrieval_queries": []}
        
        # Build analysis prompt
        param_str = "\n".join([f"{k}: {v}" for k, v in bo_params.items()])
        
        planning_prompt = f"""You are an expert at identifying information gaps in Bayesian optimization code.

**YOUR TASK:**
Analyze the problem description and Honegumi skeleton to determine if additional 
information from Ax Platform documentation is needed.

**PROBLEM DESCRIPTION:**
{problem}

**OPTIMIZATION PARAMETERS:**
{param_str}

**HONEGUMI SKELETON (first 1500 chars):**
{skeleton[:1500]}...

**ANALYSIS CHECKLIST:**
Consider if you need clarification on:
1. Objective function implementation (how to compute metrics?)
2. Parameter constraints syntax (sum constraints, order constraints, etc.)
3. Multi-objective optimization details (thresholds, reference points?)
4. Custom acquisition functions or generation strategies?
5. Data initialization from existing trials?
6. Advanced features (multi-task, SAASBO, custom models?)
7. Specific Ax API usage patterns?

**DECISION GUIDELINES:**
- If the skeleton structure is clear and the problem is straightforward → skip_retrieval
- If specific Ax implementation details are unclear → retrieve with focused queries
- Generate 1-7 queries, each targeting a DIFFERENT aspect
- Keep each query SHORT (under 20 words) for better semantic search

**REMEMBER:**
- The skeleton already has the basic Ax structure (imports, AxClient setup, trial loop)
- Only retrieve if you need SPECIFIC implementation details
- Focus queries on what's actually unclear, not what's already in the skeleton
"""

        try:
            if not settings.debug:
                print("Planning retrieval strategy...")
            
            llm = ChatOpenAI(
                model=settings.retrieval_planner_model,
                api_key=settings.openai_api_key,
            )
            
            structured_llm = llm.with_structured_output(
                RetrievalPlan,
                method="function_calling",
                include_raw=False,
            )
            
            decision: RetrievalPlan = structured_llm.invoke([
                {"role": "system", "content": "You are an expert at analyzing information needs for Bayesian optimization code."},
                {"role": "user", "content": planning_prompt}
            ])
            
            # Extract queries
            query_strings = [q.query for q in decision.queries] if decision.action == "retrieve" else []
            
            # Debug output
            if settings.debug:
                print("\n" + "="*80)
                print("DEBUG: RETRIEVAL PLAN")
                print("="*80)
                print(f"Action: {decision.action}")
                print(f"Number of queries: {len(query_strings)}")
                for i, q in enumerate(query_strings, 1):
                    print(f"  Query {i}: {q}")
                print("="*80 + "\n")
            else:
                if decision.action == "retrieve" and query_strings:
                    print(f"Planning {len(query_strings)} parallel retrieval{'s' if len(query_strings) > 1 else ''}...")
                    for i, q in enumerate(query_strings, 1):
                        print(f"   {i}. {q}")
                elif decision.action == "skip_retrieval":
                    print("Skeleton is sufficient, skipping retrieval")
            
            return {"retrieval_queries": query_strings}
            
        except Exception as exc:
            if settings.debug:
                print(f"\n[ERROR] Retrieval planning failed: {exc}\n")
                import traceback
                traceback.print_exc()
            
            # Fallback: no retrieval
            return {"retrieval_queries": []}
