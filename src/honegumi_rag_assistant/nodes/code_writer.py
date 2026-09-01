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

from langchain_anthropic import ChatAnthropic

from ..app_config import settings
from ..states import HonegumiRAGState
from ..timing_utils import time_node


# ---------------------------------------------------------------------------
# Content-block helpers
#
# With adaptive thinking enabled, a message's ``content`` is no longer a plain
# string -- it is a list of blocks mixing Claude's reasoning with its visible
# output.  Reasoning must never reach ``candidate_code``: it would be written
# straight into the generated .py file and fail every syntax check.
#
# These helpers accept every shape we may be handed: a bare string, raw
# Anthropic blocks (``{"type": "thinking", "thinking": ...}``), and LangChain's
# normalised blocks (``{"type": "reasoning", "reasoning": ...}`` or a
# ``non_standard`` wrapper around a raw block).
# ---------------------------------------------------------------------------


def _block_text(block: Any) -> str:
    """Return the visible text carried by a single content block."""
    if isinstance(block, str):
        return block
    if not isinstance(block, dict):
        return ""
    if block.get("type") == "text":
        return block.get("text") or ""
    if block.get("type") == "non_standard":
        return _block_text(block.get("value"))
    return ""


def _block_reasoning(block: Any) -> str:
    """Return the reasoning/thinking text carried by a single content block."""
    if not isinstance(block, dict):
        return ""
    btype = block.get("type")
    if btype == "thinking":
        return block.get("thinking") or ""
    if btype == "reasoning":
        value = block.get("reasoning")
        if isinstance(value, dict):
            return value.get("text") or ""
        return value or ""
    if btype == "non_standard":
        return _block_reasoning(block.get("value"))
    # redacted_thinking carries no readable text.
    return ""


def extract_text(content: Any) -> str:
    """Concatenate the visible text from a message or chunk ``content``.

    Parameters
    ----------
    content : Any
        A ``.content`` value: either a plain string or a list of content blocks.

    Returns
    -------
    str
        Only the visible text.  Reasoning blocks are excluded, which is what
        keeps Claude's thinking out of the generated script.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(_block_text(block) for block in content)
    return ""


def extract_reasoning(content: Any) -> str:
    """Concatenate the reasoning text from a message or chunk ``content``.

    Returns an empty string when thinking is disabled, when ``display`` is
    ``"omitted"``, or when the content carries no reasoning blocks.
    """
    if isinstance(content, list):
        return "".join(_block_reasoning(block) for block in content)
    return ""


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
        problem_structure = state.get("problem_structure", {})
        skeleton = state.get("skeleton_code", "") or ""
        contexts = state.get("contexts", [])
        review_feedback = state.get("critique_report", [])
        
        if settings.debug:
            print(f"\n[CODE WRITER START] contexts: {len(contexts)}")
            print(f"Received {len(contexts)} documentation contexts from retrievers")
            print(f"Problem structure available: {'Yes' if problem_structure else 'No'}")
        
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
        
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment or settings.")
        
        # Generate the code
        return CodeWriterAgent._generate_code(problem, bo_params, problem_structure, skeleton, contexts, review_feedback)
    @staticmethod
    def _generate_code(
        problem: str,
        bo_params: Dict[str, Any],
        problem_structure: Dict[str, Any],
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
            Bayesian optimization grid parameters
        problem_structure : Dict[str, Any]
            Stage 1 extracted problem structure (search space, objectives, constraints)
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
        
        # Format problem structure for the prompt
        structure_str = ""
        if problem_structure:
            structure_str = "**PROBLEM STRUCTURE (Stage 1 Analysis):**\n\n"
            
            # Search space
            if "search_space" in problem_structure:
                params = problem_structure['search_space']
                structure_str += f"Parameters to optimize ({len(params) if isinstance(params, list) else 'N/A'}):\n"
                if isinstance(params, list):
                    for param in params:
                        structure_str += f"  - {param}\n"
                else:
                    structure_str += f"  {params}\n"
            
            # Objectives
            if "objective" in problem_structure:
                objectives = problem_structure['objective']
                if isinstance(objectives, list):
                    structure_str += f"\nObjectives to optimize ({len(objectives)}):\n"
                    for obj in objectives:
                        structure_str += f"  - {obj}\n"
                else:
                    structure_str += f"\nObjective: {objectives}\n"
            
            # Constraints
            if "constraints" in problem_structure:
                constraints = problem_structure['constraints']
                if constraints:
                    structure_str += f"\nConstraints ({len(constraints) if isinstance(constraints, list) else 'N/A'}):\n"
                    if isinstance(constraints, list):
                        for const in constraints:
                            structure_str += f"  - {const}\n"
                    else:
                        structure_str += f"  {constraints}\n"
                else:
                    structure_str += "\nConstraints: None\n"
            
            # Experimental setup
            setup_items = []
            if "budget" in problem_structure and problem_structure["budget"]:
                setup_items.append(f"Budget: {problem_structure['budget']} trials")
            if "batch_size" in problem_structure and problem_structure["batch_size"]:
                setup_items.append(f"Batch size: {problem_structure['batch_size']}")
            if "noise_model" in problem_structure:
                setup_items.append(f"Noise model: {problem_structure['noise_model']}")
            if "historical_data_points" in problem_structure and problem_structure["historical_data_points"]:
                setup_items.append(f"Historical data points: {problem_structure['historical_data_points']}")
            
            if setup_items:
                structure_str += "\nExperimental setup:\n"
                for item in setup_items:
                    structure_str += f"  - {item}\n"
            
            structure_str += "\n"
        
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

**ORIGINAL PROBLEM DESCRIPTION:**
{problem}

{structure_str}
Note: The problem structure above was automatically extracted from the user's problem description 
to identify the search space, objectives, constraints, and experimental setup. Use this structured 
information to accurately adapt the skeleton code.

**EXTRACTED GRID CONFIGURATION (Stage 2):**
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

**HARD RULES ABOUT THE SKELETON'S API SURFACE:**

The skeleton was generated by Honegumi against a pinned Ax version, and its API
calls are known-correct for that version. Your job is to fill in the *problem* --
names, bounds, objectives, constraints, evaluation logic -- not to improve the
*plumbing*.

- **Never add an argument the skeleton left out.** If the skeleton contains
  `model_kwargs={{}}`, an empty argument list, or simply omits a keyword, that
  omission is deliberate and correct. Leave it exactly as it is. Do not fill it
  with hyperparameters, tuning values, or options you recall from elsewhere:
  Ax's API changed across releases, and an argument that was valid in a
  different version raises `ValueError` here rather than being ignored.
- **Do not reconfigure the model or acquisition machinery.** If the skeleton
  already constructs a `GenerationStrategy`, `GenerationStep`, or selects a
  model class, reproduce it verbatim.
- **For a call the skeleton already makes, its argument list is complete.** Fill
  in the values that describe the problem; do not append further keyword
  arguments to that call.
- **If the problem genuinely requires an Ax call the skeleton does not contain**
  (attaching historical trials, for instance), write it -- but use the simplest
  form the retrieved documentation supports, and prefer the required arguments
  over optional tuning ones you are less certain about. A working call with
  defaults beats a richer call that raises.
- **This rule constrains API surface only -- nothing else.** Adding the
  parameters, objectives and constraints the problem requires is expected (see
  the scaling instructions below), and so is everything that makes the script
  usable: printing the best parameters found, reporting quantities you derived
  from them, docstrings, comments, axis labels and plot titles. Leaving those out
  makes the script worse, not safer. The skeleton sets the *structure*; it is not
  a ceiling on *usefulness*.

**Why this matters:** an argument that is merely unnecessary is harmless in most
libraries, but Ax validates keyword arguments strictly and raises `ValueError`
on ones it does not recognise. Arguments that were correct in a different Ax
release are therefore not a harmless extra -- they are a crash. When you are
unsure whether an argument exists in this version, omitting it is always the
safer choice.

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
     * The skeleton already builds the generation strategy for this. Reproduce it
       verbatim, including any empty option dict.
     * Surrogate and acquisition hyperparameters are the single most common place
       where remembered API details turn out to be stale. If the skeleton does not
       show a setting, the correct value is "not set" -- not a value you recall
       from another version or another framework.
   
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
   - [ ] The script reports its result: prints the best parameters found, plus any
         values you derived from them (a dependent composition fraction, say), so
         the user actually sees the answer rather than it being computed and dropped

**CRITICAL REMINDERS:**
- The skeleton is a TEMPLATE - adapt everything to the specific problem
- ALL generic names must be replaced (no "branin", "x1", "x2" in final code)
- Scale the code to match actual problem requirements (objectives, parameters, constraints)
- Evaluation function is the heart of the code - make it problem-specific
- Use the extracted parameters (objective, model, task, constraints) to configure correctly
- The retrieved Ax documentation shows you the correct API syntax
- Don't add keyword arguments to calls the skeleton already makes; if it leaves an
  option dict empty, leave it empty
- Code must be immediately runnable - no broken imports or undefined functions

**OUTPUT FORMAT:**
Write ONLY the complete Python script. No markdown fences, no explanations.
Just the raw Python code, ready to execute.
"""

        try:
            # Use LangChain's ChatAnthropic for LangSmith tracing.
            #
            # max_tokens is mandatory on the Anthropic API and caps thinking
            # *plus* visible output, so it must comfortably exceed the ~2k
            # tokens a generated script needs.  display="summarized" asks for a
            # readable digest of the reasoning; it costs nothing extra (thinking
            # is billed identically under every display setting) and we simply
            # choose whether to print it based on debug mode.
            llm = ChatAnthropic(
                model=settings.code_writer_model,
                api_key=settings.anthropic_api_key,
                max_tokens=settings.code_writer_max_tokens,
                thinking={"type": "adaptive", "display": "summarized"},
                output_config={"effort": settings.code_writer_effort},
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
                    candidate_code = CodeWriterAgent._stream_code(llm, messages)
                except Exception as stream_error:
                    # Fallback to non-streaming if streaming fails
                    print("\nStreaming failed, falling back to non-streaming mode...\n")
                    if settings.debug:
                        print(f"[DEBUG] Streaming error: {stream_error}\n")
                    response = llm.invoke(messages)
                    candidate_code = extract_text(response.content).strip()
            else:
                # Non-streaming: pull only the visible text out of the response,
                # leaving any thinking blocks behind.
                response = llm.invoke(messages)
                candidate_code = extract_text(response.content).strip()
                if settings.debug:
                    reasoning = extract_reasoning(response.content)
                    if reasoning:
                        print("\n" + "=" * 80)
                        print("DEBUG: CODE WRITER REASONING (summarized)")
                        print("=" * 80)
                        print(reasoning)
                        print("=" * 80 + "\n")
            
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

    @staticmethod
    def _stream_code(llm: Any, messages: List[Dict[str, str]]) -> str:
        """Stream the generated script, separating reasoning from code.

        Claude reasons before it writes, so the first blocks off the wire carry
        thinking rather than code.  In debug mode that reasoning is printed as it
        arrives; otherwise a single ``Generating code...`` line stands in for it,
        so the terminal is never silent while the model thinks.

        Parameters
        ----------
        llm : Any
            A configured :class:`~langchain_anthropic.ChatAnthropic` instance.
        messages : List[Dict[str, str]]
            The system and user messages to send.

        Returns
        -------
        str
            The generated script -- visible text only, with reasoning excluded.
        """
        code_parts: List[str] = []
        reasoning_open = False   # a reasoning header is currently open (debug only)
        code_open = False        # the code banner has been printed
        waiting_printed = False  # the placeholder line has been printed

        if not settings.debug:
            print("Generating code...", end="", flush=True)
            waiting_printed = True

        for chunk in llm.stream(messages):
            reasoning = extract_reasoning(chunk.content)
            if reasoning:
                if settings.debug:
                    if not reasoning_open:
                        print("\n" + "=" * 80)
                        print("DEBUG: CODE WRITER REASONING (summarized)")
                        print("=" * 80)
                        reasoning_open = True
                    print(reasoning, end="", flush=True)
                # Outside debug mode reasoning is intentionally swallowed --
                # the placeholder line already tells the user work is happening.

            text = extract_text(chunk.content)
            if text:
                if not code_open:
                    # First visible token: close whatever placeholder is on
                    # screen, then open the code banner.
                    if reasoning_open:
                        print("\n" + "=" * 80)
                        reasoning_open = False
                    elif waiting_printed:
                        print()
                    print("\n" + "=" * 80)
                    print("GENERATED CODE (streaming...)")
                    print("=" * 80 + "\n")
                    code_open = True
                print(text, end="", flush=True)
                code_parts.append(text)

        if reasoning_open:
            print("\n" + "=" * 80)
        if code_open:
            print("\n\n" + "=" * 80 + "\n")
        elif waiting_printed:
            # Model returned no visible text at all.
            print()

        return "".join(code_parts).strip()
