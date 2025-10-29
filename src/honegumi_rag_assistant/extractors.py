"""
Lightweight wrappers around the OpenAI function calling API used by the
Honegumi RAG Assistant.

This module defines extractor classes that use structured output via
Pydantic models and LangChain's function calling with validation for
robust parameter extraction. The :class:`ParameterExtractor` exposes
an :meth:`invoke` method that accepts a problem description and returns
a validated dictionary containing the selected Bayesian optimisation
parameters.

The structured output approach uses LangChain's with_structured_output()
which provides automatic validation, retry on validation failures, and
type coercion through Pydantic schemas.
"""

from __future__ import annotations

from typing import Dict, Any, Literal, List, Optional

from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI

from .app_config import settings


class SearchSpaceParameter(BaseModel):
    """Represents a single parameter in the search space."""
    name: str = Field(description="The parameter name (e.g., 'temperature', 'resin_fraction')")
    type: Literal["continuous", "categorical"] = Field(
        description="The type of parameter: continuous (real-valued) or categorical (discrete choices)"
    )
    bounds: Optional[List[float]] = Field(
        default=None,
        description="For continuous parameters: [lower_bound, upper_bound]. Leave None for categorical."
    )
    categories: Optional[List[str]] = Field(
        default=None,
        description="For categorical parameters: list of possible values. Leave None for continuous."
    )
    units: Optional[str] = Field(
        default=None,
        description="Units of measurement if applicable (e.g., '°C', 'hours', 'fraction', 'mm')"
    )


class ObjectiveSpec(BaseModel):
    """Represents an optimization objective."""
    name: str = Field(description="The objective name (e.g., 'density', 'strength', 'cost')")
    goal: Literal["maximize", "minimize"] = Field(
        description="Whether to maximize or minimize this objective"
    )
    threshold: Optional[float] = Field(
        default=None,
        description="Minimum acceptable value (for maximize) or maximum acceptable value (for minimize). Only for multi-objective."
    )
    units: Optional[str] = Field(
        default=None,
        description="Units of measurement if applicable"
    )


class ConstraintSpec(BaseModel):
    """Represents an optimization constraint."""
    type: Literal["sum", "order", "linear", "composition"] = Field(
        description=(
            "Type of constraint:\n"
            "- sum: parameters sum to a specific value (e.g., x1 + x2 + x3 <= 100)\n"
            "- order: one parameter must be >= another (e.g., x1 >= x2)\n"
            "- linear: linear combination inequality (e.g., 0.2*x1 + x2 <= 0.5)\n"
            "- composition: fractions sum to 1 (special case for material compositions, e.g., monomer fractions)"
        )
    )
    parameters: List[str] = Field(
        description="List of parameter names involved in this constraint"
    )
    description: str = Field(
        description="Human-readable description of the constraint (e.g., 'resin_fraction >= inhibitor_fraction')"
    )
    total: Optional[float] = Field(
        default=None,
        description="Target value for sum/composition constraints (e.g., 1.0 for composition, 100 for sum)"
    )


class ProblemStructure(BaseModel):
    """Structured representation of the optimization problem following the solution format.
    
    This intermediate representation extracts the key elements of the optimization problem
    in the same structure as the 'solution' field in test problems. This ensures consistency
    and makes validation straightforward.
    
    Structure matches:
      solution:
        search_space: [...]
        objective: {...} or [{...}, ...]
        budget: int
        batch_size: int (optional)
        noise_model: bool
        constraints: [...]
    """
    
    search_space: List[SearchSpaceParameter] = Field(
        description="List of all parameters to optimize over (search space definition)"
    )
    
    objective: List[ObjectiveSpec] = Field(
        description=(
            "List of objectives to optimize. "
            "Single-objective: list with 1 objective. "
            "Multi-objective: list with 2+ objectives."
        )
    )
    
    budget: Optional[int] = Field(
        default=None,
        description="Total number of experiments/trials planned"
    )
    
    batch_size: Optional[int] = Field(
        default=None,
        description="Number of experiments to run in parallel (omit or set to 1 for sequential)"
    )
    
    noise_model: bool = Field(
        default=True,
        description="Whether measurements are expected to have noise/variability"
    )
    
    constraints: List[ConstraintSpec] = Field(
        default_factory=list,
        description="List of constraints on parameters (empty list if no constraints)"
    )
    
    historical_data_points: Optional[int] = Field(
        default=None,
        description="Number of historical/existing data points to incorporate (None if no existing data)"
    )
    
    model_preference: Optional[Literal["Default", "Fully Bayesian", "Custom"]] = Field(
        default=None,
        description="User's explicit model preference if stated (None means use Default)"
    )
    
    @model_validator(mode='after')
    def validate_minimum_structure(self) -> 'ProblemStructure':
        """Validate that the problem structure has minimum required elements.
        
        A valid optimization problem must have:
        - At least 1 objective (what are we optimizing?)
        - Ideally at least 1 parameter (what are we optimizing over?)
        
        This validator ensures the LLM extracted meaningful structure
        and fails fast if extraction was incomplete.
        
        Raises
        ------
        ValueError
            If the structure is missing critical elements.
        """
        if not self.objective or len(self.objective) == 0:
            raise ValueError(
                "Problem structure extraction failed: No objectives found. "
                "An optimization problem must have at least one objective to optimize. "
                "Please ensure the problem description clearly states what should be maximized or minimized."
            )
        
        # Warning for missing parameters (not a hard error since some descriptions might be very high-level)
        if not self.search_space or len(self.search_space) == 0:
            # Don't fail here - the LLM might infer parameters in Stage 2
            # But this indicates the extraction might be incomplete
            pass
        
        return self


class OptimizationParameters(BaseModel):
    """Pydantic model for Bayesian optimization parameters.
    
    This model defines the schema for optimization parameters that will
    be extracted from the user's problem description using structured
    output. All fields are required and will be validated automatically.
    """
    
    objective: Literal["Single", "Multi"] = Field(
        description=(
            "Choose between single and multi-objective optimization based on your project needs. "
            "Single objective optimization targets one primary goal (e.g. maximize the strength of a material), "
            "while multi-objective optimization considers several objectives simultaneously "
            "(e.g. maximize the strength of a material while minimizing synthesis cost). "
            "Select the option that best aligns with your optimization goals and problem complexity."
        )
    )
    
    model: Literal["Default", "Custom", "Fully Bayesian"] = Field(
        description=(
            "Choose between three surrogate model implementations: Default uses a standard Gaussian process (GP), "
            "Custom enables user-defined acquisition functions and hyperparameters, and Fully Bayesian implements "
            "MCMC estimation of GP parameters. The Default option provides a robust baseline performance, "
            "Custom allows advanced users to tailor the optimization process (avoid 'Custom' unless the user explicitly requests it), "
            "while Fully Bayesian (avoid 'Fully Bayesian' unless the user explicitly requests it) offers deeper uncertainty exploration at higher computational cost. "
            "Consider your optimization needs and computational resources when selecting this option."
        )
    )
    
    task: Literal["Single", "Multi"] = Field(
        description=(
            "Choose between single and multi-task optimization based on your experimental setup. "
            "Single-task optimization focuses on one specific task, while multi-task optimization leverages data "
            "from multiple related tasks simultaneously (e.g. optimizing similar manufacturing processes across "
            "different production sites). Multi-task optimization can improve efficiency by sharing information "
            "between tasks but requires related task structures. Consider whether your tasks share underlying "
            "similarities when making this selection."
        )
    )
    
    existing_data: bool = Field(
        description=(
            "Choose whether to fit the surrogate model to previous data before starting the optimization process. "
            "Including historical data may give your model a better starting place and potentially speed up convergence. "
            "Conversely, excluding existing data means starting the optimization from scratch, which might be preferred "
            "in scenarios where historical data could introduce bias or noise into the optimization process. "
            "Consider the relevance and reliability of your existing data when making your selection."
        )
    )
    
    sum_constraint: bool = Field(
        description=(
            "Choose whether to apply a sum constraint over two or more optimization variables "
            "(e.g. ensuring total allocation remains within available budget). This constraint focuses generated "
            "optimization trials on feasible candidates at the cost of flexibility. Consider whether such a constraint "
            "reflects the reality of variable interactions when selecting this option."
        )
    )
    
    order_constraint: bool = Field(
        description=(
            "Choose whether to implement an order constraint over two or more optimization variables "
            "(e.g. ensuring certain tasks precede others or x1 <= x2). This constraint focuses generated optimization "
            "trials on variable combinations that follow a specific order. Excluding the constraint offers flexibility "
            "in variable arrangements but may neglect important task sequencing or value inequality considerations. "
            "Consider whether such a constraint reflects the reality of variable interactions when selecting this option."
        )
    )
    
    linear_constraint: bool = Field(
        description=(
            "Choose whether to implement a linear constraint over two or more optimization variables such that "
            "the linear combination of parameter values adheres to an inequality (e.g. 0.2*x1 + x2 < 0.1). "
            "This constraint focuses generated optimization trials on variable combinations that follow an enforced "
            "rule at the cost of flexibility. Consider whether such a constraint reflects the reality of variable "
            "interactions when selecting this option."
        )
    )
    
    composition_constraint: bool = Field(
        description=(
            "Choose whether to include a composition constraint over two or more optimization variables such that "
            "their sum does not exceed a specified total (e.g. ensuring the mole fractions of elements in a composition "
            "sum to one). This constraint is particularly relevant to fabrication-related tasks where the quantities of "
            "components must sum to a total. Consider whether such a constraint reflects the reality of variable "
            "interactions when selecting this option."
        )
    )
    
    categorical: bool = Field(
        description=(
            "Choose whether to include a categorical variable in the optimization process (e.g. dark or milk chocolate "
            "chips in a cookie recipe). Including categorical variables allows choice parameters and their interaction "
            "with continuous variables to be optimized. Note that adding categorical variables can create discontinuities "
            "in the search space that are difficult to optimize over. Consider the value of adding categorical variables "
            "to the optimization task when selecting this option."
        )
    )
    
    custom_threshold: bool = Field(
        description=(
            "Choose whether to apply custom thresholds to objectives in a multi-objective optimization problem "
            "(e.g. a minimum acceptable strength requirement for a material). Setting a threshold on an objective "
            "guides the optimization algorithm to prioritize solutions that meet or exceed these criteria. "
            "Excluding thresholds enables greater exploration of the design space, but may produce sub-optimal solutions. "
            "Consider whether threshold values reflect the reality or expectations of your optimization task when "
            "selecting this option."
        )
    )
    
    synchrony: Literal["Batch", "Single"] = Field(
        description=(
            "Choose whether to perform single or batch evaluations for your Bayesian optimization campaign. "
            "Single evaluations analyze one candidate solution at a time, offering precise control and adaptability "
            "after each trial at the expense of more compute time. Batch evaluations, however, process several solutions "
            "in parallel, significantly reducing the number of optimization cycles but potentially diluting the specificity "
            "of adjustments. Batch evaluation is helpful in scenarios where it is advantageous to test several solutions "
            "simultaneously. Consider the nature of your evaluation tool when selecting between the two options."
        )
    )
    
    visualize: bool = Field(
        description=(
            "Choose whether to include visualization tools for tracking optimization progress. The default visualizations "
            "display key performance metrics like optimization traces and model uncertainty (e.g. objective value convergence "
            "over time). Including visualizations helps monitor optimization progress and identify potential issues, "
            "but may add minor computational overhead. Consider whether real-time performance tracking would benefit "
            "your optimization workflow when selecting this option."
        )
    )
    
    @model_validator(mode='after')
    def validate_parameter_combinations(self) -> 'OptimizationParameters':
        """Validate that parameter combinations are compatible.
        
        Based on Honegumi's is_incompatible function, certain parameter
        combinations are invalid:
        
        1. custom_threshold=True requires objective="Multi"
           (Custom thresholds only apply to multi-objective optimization)
           
        2. model="Fully Bayesian" requires custom_gen (not exposed in this model,
           so we document this limitation)
        
        Raises
        ------
        ValueError
            If an incompatible parameter combination is detected.
        """
        # Check: custom_threshold requires multi-objective
        if self.custom_threshold and self.objective == "Single":
            raise ValueError(
                "Invalid parameter combination: custom_threshold=True can only be used with "
                "objective='Multi'. Custom thresholds are used to specify minimum acceptable "
                "values for each objective in multi-objective optimization. For single-objective "
                "optimization, set custom_threshold=False."
            )
        
        # Note: The Honegumi framework also requires model="Fully Bayesian" to use
        # a custom generator, but since custom_gen is not exposed in this parameter
        # extraction model, we cannot validate this constraint here. The skeleton
        # generator will handle this validation when generating the Honegumi template.
        
        return self


class ProblemStructureExtractor:
    """Extract structured problem representation from natural language.
    
    This is the first stage of a two-stage extraction process. It extracts
    the explicit problem structure (parameters, objectives, constraints)
    before making grid selections. This intermediate representation helps
    improve accuracy by forcing explicit reasoning about problem elements.
    """

    @classmethod
    def invoke(cls, prompt: str) -> Dict[str, Any]:
        """Extract problem structure from natural language description.

        Parameters
        ----------
        prompt : str
            The user's problem description in natural language.

        Returns
        -------
        Dict[str, Any]
            A dictionary with a 'problem_structure' key containing the
            extracted structure, or an 'error' key if extraction failed.
        """
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Provide an API key via the environment or settings."
            )

        try:
            llm = ChatOpenAI(
                model=settings.model_name,
                api_key=settings.openai_api_key,
            )
            
            structured_llm = llm.with_structured_output(
                ProblemStructure,
                method="function_calling",
                include_raw=False,
            )
            
            # Enhanced prompt that encourages thorough extraction
            enhanced_prompt = f"""Analyze the following optimization problem and extract its structure in the standard solution format.

CRITICAL REQUIREMENTS:
1. OBJECTIVE(S): You MUST identify at least one objective (what to maximize or minimize)
   - Look for words like: maximize, minimize, optimize, improve, reduce, increase, best, highest, lowest
   - Examples: "maximize yield", "minimize cost", "optimize performance"
   
2. SEARCH SPACE: Identify all parameters being optimized
   - Look for: ranges, variables, factors, conditions, parameters
   - Examples: "temperature (50-200°C)", "pressure (1-10 bar)", "composition fractions"

3. CONSTRAINTS: Identify constraints on parameters
   - composition: Material fractions that must sum to 1.0 (e.g., monomer_a + monomer_b + monomer_c = 1)
   - sum: General sum constraints (e.g., x1 + x2 <= 100, budget allocation)
   - order: Ordering constraints (e.g., x1 >= x2, "resin should be higher than inhibitor")
   - linear: Linear combination inequalities (e.g., 0.2*x1 + x2 <= 0.5)

4. EXPERIMENTAL SETUP:
   - Batch size: Number of parallel experiments (look for: "parallel", "at a time", "batch")
   - Budget: Total number of trials/experiments (look for: "budget", "number of experiments", "trials")
   - Noise model: Whether measurements have variability (default: true unless stated otherwise)
   - Historical data: Existing data points mentioned (look for: "historical", "previous", "existing data")
   - Model preference: Default, Fully Bayesian, or Custom (only if explicitly mentioned)

Problem description:
{prompt}

Extract the complete problem structure. You MUST identify at least one objective."""
            
            result: ProblemStructure = structured_llm.invoke(enhanced_prompt)
            
            return {"problem_structure": result.model_dump()}
            
        except Exception as exc:
            return {"problem_structure": None, "error": f"Problem structure extraction failed: {exc}"}


class ParameterExtractor:
    """Extract optimisation parameters using structured output.

    This class uses LangChain's ChatOpenAI with structured output to
    reliably extract Bayesian optimization parameters from a natural
    language problem description. 
    
    This is the second stage of a two-stage extraction process. It can
    optionally accept a ProblemStructure from the first stage to improve
    grid selection accuracy by reasoning over explicit problem elements.
    
    The structured output approach provides automatic validation via 
    Pydantic, retry logic on validation failures, and type coercion, 
    making it more robust than manual JSON parsing.
    
    Note: LangChain's with_structured_output() internally uses validation
    and retry mechanisms similar to TrustCall when method='function_calling'
    is used (the default).
    """

    @classmethod
    def invoke(cls, prompt: str, problem_structure: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Invoke the LLM with a problem description and parse the result.

        Parameters
        ----------
        prompt : str
            The user's problem description. This text should explain the
            optimisation task in natural language.
        problem_structure : Optional[Dict[str, Any]], optional
            The extracted problem structure from ProblemStructureExtractor.
            If provided, this will be included in the prompt to improve
            grid selection accuracy.

        Returns
        -------
        Dict[str, Any]
            A dictionary with a single key, ``bo_params``, containing the
            parsed and validated optimisation parameters as a dict. If the
            call fails, ``bo_params`` will be ``None`` and an ``error``
            key will contain the error message.
        """
        if not settings.openai_api_key:
            raise RuntimeError(
                "LLM_API_KEY is not set. Provide an API key via the environment or settings."
            )

        try:
            # Create a ChatOpenAI instance
            llm = ChatOpenAI(
                model=settings.model_name,
                api_key=settings.openai_api_key,
            )
            
            # Enable structured output with the Pydantic model
            # with_structured_output uses function calling by default, which includes:
            # - Automatic validation against the Pydantic schema
            # - Retry with error messages if validation fails
            # - Type coercion for compatible types
            # Set include_raw=False to get just the parsed model (not the raw response)
            structured_llm = llm.with_structured_output(
                OptimizationParameters,
                method="function_calling",  # Explicit: use function calling (default)
                include_raw=False,  # Return only the validated Pydantic model
            )
            
            # Build the prompt, optionally including problem structure
            if problem_structure:
                # Extract key information from the solution-format structure
                num_params = len(problem_structure.get('search_space', []))
                num_objectives = len(problem_structure.get('objective', []))
                num_constraints = len(problem_structure.get('constraints', []))
                
                param_names = [p['name'] for p in problem_structure.get('search_space', [])]
                param_types = [p['type'] for p in problem_structure.get('search_space', [])]
                has_categorical = 'categorical' in param_types
                
                objectives_info = [
                    f"{o['name']} ({o['goal']})" + (f", threshold: {o['threshold']}" if o.get('threshold') else "")
                    for o in problem_structure.get('objective', [])
                ]
                
                constraints_info = []
                has_sum_constraint = False
                has_order_constraint = False
                has_linear_constraint = False
                has_composition_constraint = False
                
                for c in problem_structure.get('constraints', []):
                    constraint_type = c['type']
                    if constraint_type == 'sum':
                        has_sum_constraint = True
                    elif constraint_type == 'order':
                        has_order_constraint = True
                    elif constraint_type == 'linear':
                        has_linear_constraint = True
                    elif constraint_type == 'composition':
                        has_composition_constraint = True
                    
                    constraints_info.append(f"{c['type']}: {c['description']}")
                
                batch_size = problem_structure.get('batch_size')
                is_batch = batch_size is not None and batch_size > 1
                model_pref = problem_structure.get('model_preference', 'Default')
                historical_data = problem_structure.get('historical_data_points')
                has_existing = historical_data is not None and historical_data > 0
                
                enhanced_prompt = f"""Based on the following problem description and its extracted structure, select the appropriate grid parameters for Honegumi optimization.

Problem Description:
{prompt}

Extracted Problem Structure (Solution Format):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SEARCH SPACE ({num_params} parameters):
{chr(10).join(f"  • {p['name']} ({p['type']}): {p.get('bounds', p.get('categories', 'N/A'))}" for p in problem_structure.get('search_space', []))}

OBJECTIVES ({num_objectives}):
{chr(10).join(f"  • {info}" for info in objectives_info)}

CONSTRAINTS ({num_constraints}):
{chr(10).join(f"  • {info}" for info in constraints_info) if constraints_info else '  (none)'}

EXPERIMENTAL SETUP:
  • Budget: {problem_structure.get('budget', 'Not specified')}
  • Batch size: {batch_size if batch_size else 'Sequential (1)'}
  • Noise model: {problem_structure.get('noise_model', True)}
  • Historical data points: {problem_structure.get('historical_data_points', 0)}
  • Model preference: {model_pref or 'Default'}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GRID SELECTION RULES (apply based on extracted structure above):
┌─────────────────────────┬──────────────────────────────────────────────────┐
│ Grid Parameter          │ Selection Logic                                  │
├─────────────────────────┼──────────────────────────────────────────────────┤
│ objective               │ "Single" if 1 objective, "Multi" if 2+           │
│ model                   │ Use model_preference ({model_pref or 'Default'})                    │
│ task                    │ "Single" (Multi only for multi-task learning)    │
│ existing_data           │ {has_existing} (historical_data_points > 0)                │
│ categorical             │ {has_categorical} (any categorical parameters)              │
│ sum_constraint          │ {has_sum_constraint} (constraint type = "sum")                │
│ order_constraint        │ {has_order_constraint} (constraint type = "order")              │
│ linear_constraint       │ {has_linear_constraint} (constraint type = "linear")            │
│ composition_constraint  │ {has_composition_constraint} (constraint type = "composition")      │
│ custom_threshold        │ True if any objective has threshold (Multi only) │
│ synchrony               │ "Batch" if batch_size > 1, else "Single"         │
│ visualize               │ True (default for tracking progress)             │
└─────────────────────────┴──────────────────────────────────────────────────┘

CRITICAL: Constraint Type Distinction
• composition_constraint: ONLY for material fractions summing to 1.0 
  Example: monomer_a + monomer_b + monomer_c = 1 (materials composition)
  
• sum_constraint: General sum constraints NOT equal to 1.0
  Example: x1 + x2 <= 100 (budget), allocation sums, etc.

These are MUTUALLY EXCLUSIVE - check the constraint type and total value!

Select the grid parameters following the rules above."""
                final_prompt = enhanced_prompt
            else:
                final_prompt = prompt
            
            # Invoke the LLM with the problem description
            # If the LLM returns invalid data, it will automatically retry with
            # the validation error message to get corrected output
            result: OptimizationParameters = structured_llm.invoke(final_prompt)
            
            # Convert the Pydantic model to a dict
            bo_params = result.model_dump()
            
            return {"bo_params": bo_params}
            
        except Exception as exc:
            # Surface the exception in the error field
            return {"bo_params": None, "error": f"LLM invocation failed: {exc}"}

