"""
Node: Code writer agent.

This node generates Python code from the Honegumi skeleton and any
retrieved documentation contexts. In the new architecture, retrieval
is performed upfront by the Retrieval Planner, so the Code Writer
focuses solely on code generation.

The agent receives the problem description, optimization parameters,
skeleton code, and pre-retrieved documentation contexts, then generates
the final Python script.
"""

from __future__ import annotations

from typing import Dict, Any, List

from langchain_openai import ChatOpenAI

from ..app_config import settings
from ..states import HonegumiRAGState
from ..timing_utils import time_node


class CodeWriterAgent:
    """Code writer that generates Ax Platform code from skeleton and contexts.

    In the new architecture, retrieval is handled upfront by the Retrieval
    Planner agent with parallel execution. The Code Writer receives all
    necessary contexts and focuses on generating high-quality code.
    """

    @staticmethod
    @time_node("Code Writer Agent")
    def write_code(state: HonegumiRAGState) -> Dict[str, Any]:
        """Generate executable Python code from skeleton and contexts.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state with keys ``problem``,
            ``bo_params``, ``skeleton_code``, and ``contexts``.

        Returns
        -------
        Dict[str, Any]
            Dictionary with either:
            - "final_code" if streaming is enabled (no review)
            - "candidate_code" if review is enabled
        """
        problem = state.get("problem", "")
        bo_params = state.get("bo_params", {})
        skeleton = state.get("skeleton_code", "") or ""
        contexts = state.get("contexts", [])
        review_feedback = state.get("critique_report", [])
        
        if settings.debug:
            print(f"\n[CODE WRITER START] contexts: {len(contexts)}")
            print(f"Received {len(contexts)} documentation contexts from retrievers")
        
        # Debug: Show context summary
        if len(contexts) > 0 and settings.debug:
            context_by_query = {}
            for ctx in contexts:
                query_idx = ctx.get("query_index", "unknown") if isinstance(ctx, dict) else "unknown"
                if query_idx not in context_by_query:
                    context_by_query[query_idx] = 0
                context_by_query[query_idx] += 1
            
            print("[CODE WRITER] Context breakdown by retriever:")
            for idx in sorted(context_by_query.keys()):
                print(f"  Retriever {idx + 1 if isinstance(idx, int) else idx}: {context_by_query[idx]} contexts")
            print()
        
        if not settings.openai_api_key:
            raise RuntimeError("LLM_API_KEY is not set in environment or settings.")
        
        # Generate the code
        return CodeWriterAgent._generate_code(problem, bo_params, skeleton, contexts, review_feedback)
    @staticmethod
    def _generate_code(
        problem: str,
        bo_params: Dict[str, Any],
        skeleton: str,
        contexts: List[Dict[str, Any]],
        review_feedback: List[str]
    ) -> Dict[str, Any]:
        """Generate the final Python code.
        
        Parameters
        ----------
        problem : str
            Problem description
        bo_params : Dict[str, Any]
            Bayesian optimization parameters
        skeleton : str
            Honegumi skeleton code
        contexts : List[Dict[str, Any]]
            Retrieved documentation contexts
        review_feedback : List[str]
            Review feedback from previous iterations
            
        Returns
        -------
        Dict[str, Any]
            Dictionary with final_code (if streaming) or candidate_code (if review enabled)
        """
        if settings.debug:
            print(f"\n[DEBUG] _generate_code called, stream_code={settings.stream_code}\n")
        
        param_str = "\n".join([f"{k}: {v}" for k, v in bo_params.items()])
        
        context_strs: List[str] = []
        for ctx in contexts:
            text = ctx.get("text") if isinstance(ctx, dict) else str(ctx)
            if text:
                context_strs.append(text)
        contexts_block = "\n\n".join(context_strs) if context_strs else "(No Ax documentation retrieved)"
        
        # Convert review_feedback from list to string
        feedback_str = "\n".join(review_feedback) if review_feedback else "(No feedback yet)"
        
        code_gen_prompt = f"""You are an expert at adapting Bayesian optimization templates to solve specific real-world problems.

**CONTEXT:**
- **Ax Platform**: Meta's Bayesian optimization framework with state-of-the-art algorithms (Gaussian processes, EHVI, SAASBO, etc.)
- **Honegumi**: A template generator that creates Ax Platform code SKELETONS with PLACEHOLDER names and DUMMY evaluation functions

**YOUR TASK:**
Transform the generic Honegumi skeleton into a complete, executable solution for the user's specific problem.

**PROBLEM DESCRIPTION:**
{problem}

**EXTRACTED OPTIMIZATION CONFIGURATION:**
{param_str}

These parameters were extracted from the problem and determine key decisions:
- **objective**: Single or Multi-objective optimization
- **model**: Default (standard GP), Custom (user-defined), or Fully Bayesian (MCMC)
- **task**: Single-task or Multi-task optimization
- **existing_data**: Whether to initialize with historical data
- **sum_constraint**: Whether variables must sum to a specific value
- **order_constraint**: Whether variables must follow an ordering (e.g., x1 <= x2)
- **linear_constraint**: Whether a linear combination inequality applies

**HONEGUMI SKELETON (TEMPLATE TO ADAPT):**
{skeleton}

**RETRIEVED AX DOCUMENTATION:**
{contexts_block}

**REVIEW FEEDBACK (IF ANY):**
{feedback_str}

**STEP-BY-STEP TRANSFORMATION INSTRUCTIONS:**

1. **ANALYZE THE PROBLEM DOMAIN**
   - Identify what real-world system is being optimized
   - List ALL objectives the user wants to optimize (could be 1-10+)
   - List ALL parameters/variables the user wants to tune (could be 1-20+)
   - Note any constraints mentioned (budgets, orderings, physical limits)

2. **REPLACE ALL PLACEHOLDER NAMES**
   The skeleton uses generic names like "branin", "x1", "x2", "task_A". Replace EVERY instance with domain-specific names:
   - Objective names: Use descriptive metric names (e.g., "yield", "cost", "quality_score")
   - Parameter names: Use meaningful variable names (e.g., "temperature_celsius", "pressure_bar", "catalyst_concentration")
   - Task names (if multi-task): Use actual task identifiers (e.g., "batch_A", "reactor_1", "patient_cohort_young")
   
   Example transformation:
   ```
   # Skeleton (WRONG):
   def branin(x1, x2):
       return {{"branin": (x2 - 5.1*x1**2/(4*np.pi**2) + 5*x1/np.pi - 6)**2}}
   
   # Problem-specific (CORRECT):
   def evaluate_chemical_reaction(temperature, pressure):
       # TODO: Replace with actual experimental measurement
       # For now, simulate based on physical model or return placeholder
       yield_percent = ...  # Actual computation or stub
       cost_dollars = ...   # Actual computation or stub
       return {{"yield": yield_percent, "cost": cost_dollars}}
   ```

3. **SCALE TO MATCH PROBLEM REQUIREMENTS**
   **CRITICAL**: The skeleton's counts are just EXAMPLES. Adapt to the actual problem:
   
   - **Objectives**: If problem has 5 objectives but skeleton shows 2, ADD 3 more
     * Update ObjectiveProperties in create_experiment() for ALL objectives
     * Ensure evaluation function returns dict with ALL objective names as keys
     * Example: `ObjectiveProperties(minimize=False, threshold=100)` for each objective
   
   - **Parameters**: If problem has 8 parameters but skeleton shows 3, ADD 5 more
     * Add parameter definitions: `ax_client.add_parameter(name=..., type="range", bounds=[min, max])`
     * Update evaluation function signature to accept ALL parameters
     * Use appropriate types: "range" for continuous, "choice" for categorical
   
   - **Constraints**: Match the configuration flags
     * If sum_constraint=True: Add `ax_client.add_parameter_constraint(["x1", "x2"], bound=total)`
     * If order_constraint=True: Add `ax_client.add_order_constraint(["x1", "x2"])`
     * If linear_constraint=True: Add linear constraint with appropriate coefficients

4. **IMPLEMENT THE EVALUATION FUNCTION**
   This is THE MOST IMPORTANT part - the skeleton has a dummy function you MUST replace:
   
   **If the user describes HOW to compute objectives:**
   - Implement their exact logic (formulas, API calls, simulations, etc.)
   
   **If computation details are NOT specified (common case):**
   - Create a realistic STUB that returns the correct data structure
   - Add clear TODO comments explaining what data/computation is needed
   - Provide example return values with correct types
   
   Example stub structure:
   ```python
   def evaluate_experiment(param1, param2, param3):
       \"\"\"Evaluate the experiment with given parameters.
       
       TODO: Replace this stub with actual evaluation logic.
       This might involve:
       - Running a physical experiment and measuring outcomes
       - Calling a simulation API
       - Querying a database of experimental results
       - Computing from a mathematical model
       \"\"\"
       
       # Placeholder return - replace with actual measurements
       objective1_value = 0.0  # TODO: Measure/compute actual value
       objective2_value = 0.0  # TODO: Measure/compute actual value
       
       return {{
           "objective1_name": objective1_value,
           "objective2_name": objective2_value,
       }}
   ```
   
   **CRITICAL**: Return value MUST be a dict with ALL objective names as keys

5. **CONFIGURE BASED ON EXTRACTED PARAMETERS**
   Use the extracted configuration to set up the optimization correctly:
   
   - **objective=="Multi"**: 
     * Use multiple ObjectiveProperties in create_experiment
     * Set minimize= and threshold= appropriately for each
     * Consider using EHVI acquisition function (check docs)
   
   - **model=="Fully Bayesian"**:
     * Use SAASBO acquisition function
     * May need to specify model in generation_strategy
   
   - **task=="Multi"**:
     * Add task parameter as a ChoiceParameter
     * Evaluation function should handle task-specific logic
   
   - **existing_data==True**:
     * Add code to attach trials from CSV/database before optimization loop
     * Use ax_client.attach_trial() for each historical data point

6. **ENSURE PRODUCTION QUALITY**
   - **All imports present**: numpy, pandas, ax.service.ax_client, etc.
   - **No TODO stubs in critical logic**: Skeleton structure should be complete
   - **Descriptive comments**: Explain the problem domain and what objectives measure
   - **Proper error handling**: Wrap evaluation in try/except if needed
   - **Type hints where helpful**: Makes code more maintainable
   - **Follow Python conventions**: PEP 8 style, clear naming

7. **SELF-VALIDATION CHECKLIST**
   Before returning the code, verify:
   - [ ] All placeholder names replaced with domain-specific names
   - [ ] Number of objectives matches problem description
   - [ ] Number of parameters matches problem description
   - [ ] Constraints match the extracted configuration flags
   - [ ] Evaluation function returns dict with correct objective names
   - [ ] All imports are present
   - [ ] Code is immediately executable (even if evaluation is a stub)
   - [ ] Comments explain the domain and any stubs/TODOs

**CRITICAL REMINDERS:**
- The skeleton is a TEMPLATE - adapt everything to the specific problem
- ALL generic names must be replaced (no "branin", "x1", "x2" in final code)
- Scale the code to match actual problem requirements (objectives, parameters, constraints)
- Evaluation function is the heart of the code - make it problem-specific
- Use the extracted parameters (objective, model, task, constraints) to configure correctly
- The retrieved Ax documentation shows you the correct API syntax
- Code must be immediately runnable - no broken imports or undefined functions

**OUTPUT FORMAT:**
Write ONLY the complete Python script. No markdown fences, no explanations.
Just the raw Python code, ready to execute.
"""

        try:
            # Use LangChain's ChatOpenAI for LangSmith tracing
            llm = ChatOpenAI(
                model=settings.code_writer_model,
                api_key=settings.openai_api_key,
            )
            
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are an expert at transforming generic Bayesian optimization templates into "
                        "problem-specific, executable solutions. You excel at understanding domain requirements "
                        "and adapting placeholder code to real-world problems. You are meticulous about replacing "
                        "ALL generic names with domain-appropriate terminology and implementing actual evaluation logic."
                    )
                },
                {"role": "user", "content": code_gen_prompt},
            ]
            
            if settings.debug:
                # DEBUG: Print code generation start
                print("\n" + "="*80)
                print("DEBUG: CODE WRITER GENERATING CODE")
                print("="*80)
                print(f"Contexts available: {len(contexts)}")
                print(f"Has review feedback: {'Yes' if feedback_str.strip() and feedback_str != '(No feedback yet)' else 'No'}")
                print("Calling LLM API to generate code...")
                print("="*80 + "\n")
            
            # Stream or invoke based on settings
            if settings.stream_code:
                try:
                    # Try to stream the response in real-time
                    print("\n" + "="*80)
                    print("GENERATED CODE (streaming...)")
                    print("="*80 + "\n")
                    
                    candidate_code = ""
                    for chunk in llm.stream(messages):
                        content = chunk.content
                        if content:
                            print(content, end="", flush=True)
                            candidate_code += content
                    
                    print("\n\n" + "="*80 + "\n")
                    
                    # Strip whitespace from streamed code
                    candidate_code = candidate_code.strip()
                except Exception as stream_error:
                    # Fallback to non-streaming if streaming fails
                    print(f"\nStreaming failed (organization not verified?), using non-streaming mode...\n")
                    if settings.debug:
                        print(f"[DEBUG] Streaming error: {stream_error}\n")
                    response = llm.invoke(messages)
                    candidate_code: str = response.content.strip()
            else:
                # Non-streaming (original behavior)
                response = llm.invoke(messages)
                candidate_code: str = response.content.strip()
            
            if settings.debug and not settings.stream_code:
                print("\n" + "="*80)
                print("DEBUG: GENERATED CODE (Before Review)")
                print("="*80)
                print(candidate_code[:1000] + "..." if len(candidate_code) > 1000 else candidate_code)
                print("="*80 + "\n")
            
        except Exception as exc:
            candidate_code = (
                f"# Failed to generate code: {exc}\n"
                f"{skeleton}\n"
            )
            return {
                "candidate_code": candidate_code,
                "critique_report": [f"Code generation error: {exc}"],
                "confidence": 0.0,
            }

        # Self-check: ensure skeleton structure is preserved
        if skeleton:
            first_lines = [line for line in skeleton.splitlines() if line.strip()]
            if first_lines and first_lines[0] not in candidate_code:
                candidate_code = skeleton + "\n" + candidate_code

        confidence = 1.0 if candidate_code.strip() and "# Failed" not in candidate_code else 0.0
        
        # If streaming mode (no review), set final_code directly
        if settings.stream_code:
            if settings.debug:
                print(f"\n[DEBUG] Setting final_code (length: {len(candidate_code)} chars)\n")
            return {
                "final_code": candidate_code,
                "candidate_code": candidate_code,
                "critique_report": ["Code generated successfully (no review)."],
                "confidence": confidence,
            }
        
        return {
            "candidate_code": candidate_code,
            "critique_report": ["Code generated successfully by agentic writer."],
            "confidence": confidence,
        }
