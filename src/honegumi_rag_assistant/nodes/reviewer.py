"""
Node: Reviewer agent.

The reviewer acts as a quality control gate that can either approve the
generated code or send it back for revision. It uses LLM-based code review
to check correctness, completeness, and adherence to Ax Platform best practices.

The reviewer can send code back for revision up to 2 times with specific
feedback. After the maximum number of revisions, it will approve the code
regardless of quality to prevent infinite loops.
"""

from __future__ import annotations

from typing import Dict, Any, Literal
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI

from ..states import HonegumiRAGState
from ..app_config import settings
from ..timing_utils import time_node


class ReviewDecision(BaseModel):
    """Decision made by the Reviewer Agent."""
    
    action: Literal["approve", "revise"] = Field(
        description=(
            "Choose 'approve' if the code is correct, complete, and ready to use. "
            "Choose 'revise' if the code has issues that need to be fixed by the Code Writer."
        )
    )
    
    feedback: str | None = Field(
        default=None,
        description=(
            "If action is 'revise', provide specific, actionable feedback about what "
            "needs to be fixed. Be precise about the issues (e.g., 'Missing parameter bounds', "
            "'Objective function returns wrong type'). If action is 'approve', set to None."
        )
    )


class ReviewerAgent:
    """Perform LLM-based review and decide whether to approve or request revision."""

    @staticmethod
    @time_node("Reviewer Agent")
    def review_code(state: HonegumiRAGState) -> Dict[str, Any]:
        """Review the candidate code and decide to approve or revise.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state containing ``candidate_code``,
            ``review_count``, and other context.

        Returns
        -------
        Dict[str, Any]
            Either:
            - {"final_code": str} if approved
            - {"critique_report": List[str], "review_count": int} if revision needed
        """
        code = state.get("candidate_code") or ""
        review_count = state.get("review_count", 0)
        problem = state.get("problem", "")
        bo_params = state.get("bo_params", {})
        skeleton = state.get("skeleton_code", "")
        
        # Basic sanity check
        if not code.strip():
            return {"final_code": "# Error: Generated code is empty\npass"}
        
        # If we've already revised twice, approve regardless of quality
        if review_count >= 2:
            return {
                "final_code": code,
                "critique_report": [f"Approved after {review_count} revision attempts (max reached)."],
            }
        
        if not settings.openai_api_key:
            # Without API key, do simple checks and approve
            lower = code.lower()
            if "notimplementederror" in lower or "todo" in lower:
                if review_count < 2:
                    return {
                        "critique_report": ["Code contains placeholders (TODO/NotImplementedError). Remove them."],
                        "review_count": review_count + 1,
                    }
            return {"final_code": code}
        
        # Build review prompt
        param_str = "\n".join([f"{k}: {v}" for k, v in bo_params.items()])
        
        if not settings.debug:
            print("Reviewing the generated code for errors...")
        
        review_prompt = f"""You are an expert code reviewer specializing in Bayesian optimization with the Ax Platform.

**YOUR TASK:**
Review the generated Python script for correctness, completeness, and adherence to best practices.

**PROBLEM DESCRIPTION:**
{problem}

**OPTIMIZATION PARAMETERS:**
{param_str}

**HONEGUMI SKELETON (expected structure):**
{skeleton[:800]}... (truncated)

**GENERATED CODE TO REVIEW:**
{code}

**REVIEW CRITERIA:**
1. **Correctness**: Does the code correctly implement the problem requirements?
2. **Completeness**: Are all necessary components filled in (objective function, parameters, constraints)?
3. **Ax API Usage**: Does it use Ax Platform APIs correctly?
4. **Structure Preservation**: Does it maintain the Honegumi skeleton structure?
5. **No Placeholders**: No TODO, NotImplementedError, or placeholder comments?
6. **Executability**: Would this code run without errors?

**YOUR DECISION:**
- Choose 'approve' if the code meets all criteria and is ready to use
- Choose 'revise' if there are issues that need fixing (be specific about what's wrong)

Keep in mind: This is review attempt {review_count + 1}/3. Be thorough but fair.
"""

        try:
            # Use LangChain's ChatOpenAI for LangSmith tracing
            llm = ChatOpenAI(
                model=settings.reviewer_model,
                api_key=settings.openai_api_key,
            )
            
            # Use structured output with Pydantic model
            structured_llm = llm.with_structured_output(
                ReviewDecision,
                method="function_calling",
                include_raw=False,
            )
            
            decision: ReviewDecision = structured_llm.invoke([
                {"role": "system", "content": "You are an expert code reviewer specializing in Bayesian optimization and the Ax Platform."},
                {"role": "user", "content": review_prompt}
            ])
            
            if settings.debug:
                # DEBUG: Print Reviewer's decision
                print("\n" + "="*80)
                print(f"DEBUG: REVIEWER DECISION (Review Attempt {review_count + 1}/3)")
                print("="*80)
                print(f"Action: {decision.action}")
                if decision.action == "revise":
                    print(f"Feedback: {decision.feedback or '(No specific feedback provided)'}")
                    print(f"Review count after this: {review_count + 1}/2")
                else:
                    print("Status: Code approved!")
                print("="*80 + "\n")
            
            if decision.action == "approve":
                if not settings.debug:
                    print("Code approved!\n")
                return {
                    "final_code": code,
                    "critique_report": [f"Code approved by reviewer on attempt {review_count + 1}."],
                }
            else:
                # Send back for revision with retrieval_count reset to 0
                # This allows Code Writer to retrieve again with fresh context
                feedback = decision.feedback or "Code needs improvement (no specific feedback provided)."
                if settings.debug:
                    print(f"Sending code back to Code Writer for revision...\n")
                    print(f"DEBUG: Resetting retrieval_count to 0 (contexts preserved)\n")
                    print(f"DEBUG: Reviewer returning critique_report, review_count, retrieval_count\n")
                    print(f"DEBUG: NOT returning contexts key - LangGraph should preserve existing\n")
                else:
                    print("Requesting code revision...")
                return {
                    "critique_report": [feedback],
                    "review_count": review_count + 1,
                    "retrieval_count": 0,  # Reset counter but keep contexts
                }
                
        except Exception as exc:
            # On error, approve the code to avoid blocking pipeline
            return {
                "final_code": code,
                "critique_report": [f"Review error: {exc}. Approving by default."],
            }