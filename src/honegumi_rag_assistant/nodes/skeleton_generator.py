"""
Node: Generate a deterministic code skeleton using Honegumi.

This node converts the optimisation parameters selected by the
:class:`ParameterSelector` into a baseline Python script using the
Honegumi template engine.  The skeleton contains the correct Ax API
calls and experiment loop structure but does not implement any
problem‑specific logic.  Once generated, the skeleton and any
associated metadata are inserted into the pipeline state.
"""

from __future__ import annotations

from typing import Dict, Any
import hashlib
import json

from ..states import HonegumiRAGState
from ..app_config import settings
from ..timing_utils import time_node
import honegumi
from honegumi.core._honegumi import Honegumi
from honegumi.ax.utils import constants as cst
from honegumi.ax._ax import option_rows


class SkeletonGenerator:
    """Generate the Honegumi skeleton code for the selected parameters.

    Given a set of optimisation parameters (``bo_params``), this node
    instantiates the Honegumi template engine and renders the
    corresponding Ax script.
    """

    @staticmethod
    @time_node("Skeleton Generator")
    def generate_skeleton(state: HonegumiRAGState) -> Dict[str, Any]:
        """Generate a skeleton code file from the provided parameters.

        Parameters
        ----------
        state : HonegumiRAGState
            The current pipeline state containing at least a ``bo_params`` key.

        Returns
        -------
        Dict[str, Any]
            A dictionary with keys ``skeleton_code`` and ``template_metadata``.
            If an error occurs, an ``error`` key will also be set.
        """
        bo_params = state.get("bo_params")
        if not bo_params:
            return {"error": "Optimisation parameters are missing; cannot generate skeleton."}

        if not settings.debug:
            print("Generating code skeleton using Honegumi...")
        
        try:
            # Determine the location of the Honegumi templates using the package
            # itself.  These attributes are available when honegumi is
            # installed as a Python package.
            script_template_dir = honegumi.ax.__path__[0]  # type: ignore[attr-defined]
            core_template_dir = honegumi.core.__path__[0]  # type: ignore[attr-defined]
            script_template_name = "main.py.jinja"
            core_template_name = "honegumi.html.jinja"
            hg = Honegumi(
                cst,
                option_rows,
                script_template_dir=script_template_dir,
                core_template_dir=core_template_dir,
                script_template_name=script_template_name,
                core_template_name=core_template_name,
            )

            options_model = hg.OptionsModel(**bo_params)  # type: ignore[arg-type]
            skeleton_code: str = hg.generate(options_model)

            # Debug: Print skeleton immediately after generation
            if settings.debug:
                print("\n" + "="*80)
                print("DEBUG: GENERATED SKELETON CODE")
                print("="*80)
                print(skeleton_code[:1000] + "..." if len(skeleton_code) > 1000 else skeleton_code)
                print("="*80 + "\n")

            # Create a simple metadata payload.  We compute a hash of the
            # bo_params to allow caching and identify the template used.
            params_str = json.dumps(bo_params, sort_keys=True)
            options_hash = hashlib.sha256(params_str.encode()).hexdigest()
            metadata = {
                "template_name": script_template_name,
                "options_hash": options_hash,
            }
            return {
                "skeleton_code": skeleton_code,
                "template_metadata": metadata,
            }
        except Exception as exc:
            # Propagate the error
            return {
                "skeleton_code": None,
                "template_metadata": {},
                "error": f"Honegumi generation failed: {exc}",
            }