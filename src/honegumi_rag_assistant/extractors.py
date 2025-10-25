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

from typing import Dict, Any, Literal

from pydantic import BaseModel, Field, model_validator
from langchain_openai import ChatOpenAI

from .app_config import settings


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


class ParameterExtractor:
    """Extract optimisation parameters using structured output.

    This class uses LangChain's ChatOpenAI with structured output to
    reliably extract Bayesian optimization parameters from a natural
    language problem description. The structured output approach provides
    automatic validation via Pydantic, retry logic on validation failures,
    and type coercion, making it more robust than manual JSON parsing.
    
    Note: LangChain's with_structured_output() internally uses validation
    and retry mechanisms similar to TrustCall when method='function_calling'
    is used (the default).
    """

    @classmethod
    def invoke(cls, prompt: str) -> Dict[str, Any]:
        """Invoke the LLM with a problem description and parse the result.

        Parameters
        ----------
        prompt : str
            The user's problem description. This text should explain the
            optimisation task in natural language.

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
            
            # Invoke the LLM with the problem description
            # If the LLM returns invalid data, it will automatically retry with
            # the validation error message to get corrected output
            result: OptimizationParameters = structured_llm.invoke(prompt)
            
            # Convert the Pydantic model to a dict
            bo_params = result.model_dump()
            
            return {"bo_params": bo_params}
            
        except Exception as exc:
            # Surface the exception in the error field
            return {"bo_params": None, "error": f"LLM invocation failed: {exc}"}
