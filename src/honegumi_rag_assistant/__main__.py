"""
Command‑line interface for the Honegumi RAG Assistant.

This module defines an entrypoint that can be invoked with
``python -m honegumi_rag_assistant``.  It prompts the user to enter
their optimization problem description, then runs the full agentic
pipeline defined in :mod:`honegumi_rag_assistant.orchestrator`.
Upon completion, the generated Python script is written to the
configured output directory and the script contents are printed
to standard output.
"""

from __future__ import annotations

import argparse
import sys

# Load environment variables from .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, skip

from .orchestrator import run_from_text
from .app_config import settings


def main(argv: list[str] | None = None) -> int:
    """Parse command‑line arguments and run the pipeline interactively.

    Parameters
    ----------
    argv : list of str, optional
        List of arguments to parse.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit status code: ``0`` on success, non‑zero on error.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Run the Honegumi RAG Assistant to generate Bayesian optimisation "
            "code from a problem description."
        )
    )
    
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Directory where the generated script will be saved.  Defaults to settings.output_dir.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode to show detailed decisions, parameters, and intermediate outputs.",
    )
    parser.add_argument(
        "--review",
        action="store_true",
        help="Enable the code review step for quality checking. Disabled by default for faster execution.",
    )
    parser.add_argument(
        "--code-writer-model",
        default="gpt-5",
        help="OpenAI model to use for the Code Writer agent. Default: gpt-5",
    )
    parser.add_argument(
        "--reviewer-model",
        default="gpt-4o",
        help="OpenAI model to use for the Reviewer agent. Default: gpt-4o",
    )
    parser.add_argument(
        "--retrieval-planner-model",
        default="gpt-5",
        help="OpenAI model to use for the Retrieval Planner agent. Default: gpt-5",
    )
    parser.add_argument(
        "--param-selector-model",
        default="gpt-5",
        help="OpenAI model to use for the Parameter Selector and Retrieval Planner. Default: gpt-5",
    )

    args = parser.parse_args(argv)

    try:
        # Apply model selections from CLI arguments
        settings.code_writer_model = args.code_writer_model
        settings.reviewer_model = args.reviewer_model
        settings.retrieval_planner_model = args.retrieval_planner_model
        settings.model_name = args.param_selector_model
        settings.stream_code = not args.review  # Enable streaming when review is disabled
        
        # Interactive prompt for problem description
        print("="*80)
        print("Honegumi RAG Assistant - Interactive Mode")
        print("="*80)
        print("\nPlease describe your Bayesian optimization problem.")
        print("(Press Enter when finished)\n")
        print("Example: Optimize temperature (50-200°C) and pressure (1-10 bar)")
        print("         for maximum yield in a chemical reaction.\n")
        
        # Read problem from user input
        print("Your problem:")
        problem = input().strip()
        
        if not problem:
            print("Error: No problem description provided.", file=sys.stderr)
            return 1
        
        if args.debug:
            print("\n" + "="*80)
            print("Starting optimization code generation...")
            print("="*80)
            print(f"Parameter Selector: {args.param_selector_model}")
            print(f"Retrieval Planner: {args.retrieval_planner_model}")
            print(f"Code Writer: {args.code_writer_model}")
            if args.review:
                print(f"Reviewer: {args.reviewer_model}")
            else:
                print(f"Streaming: Enabled (review disabled for speed)")
            print("="*80 + "\n")
        
        # Run the pipeline
        code = run_from_text(problem, args.output_dir, debug=args.debug, enable_review=args.review)
        
        # In debug mode, print the final code with a header
        # In normal mode, code is already streamed so don't print it again
        if args.debug:
            print("\n" + "="*80)
            print("GENERATED CODE")
            print("="*80)
            print(code)
        
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())