"""
Batch processing script for Honegumi RAG Assistant.

This script processes multiple optimization problems from a CSV file,
runs the agentic pipeline for each, and saves all results to a new CSV
with detailed debug information.

Usage:
    python scripts/batch_process.py --input problems.csv --output results.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path to import honegumi_rag_assistant
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from honegumi_rag_assistant.orchestrator import run_from_text_with_state


def process_batch(
    input_csv: str,
    output_csv: str,
    output_dir: str | None = None,
    problem_column: str = "problems",
) -> None:
    """Process a batch of problems from a CSV file.

    Parameters
    ----------
    input_csv : str
        Path to input CSV file with a column containing problem descriptions.
    output_csv : str
        Path to output CSV file where results will be saved.
    output_dir : str or None, optional
        Directory where generated Python scripts will be saved.
    problem_column : str, optional
        Name of the column containing problem descriptions. Default: "problems".
    """
    # Read input CSV
    print(f"📖 Reading problems from: {input_csv}")
    try:
        df = pd.read_csv(input_csv)
    except Exception as exc:
        print(f"❌ Error reading CSV: {exc}", file=sys.stderr)
        return
    
    if problem_column not in df.columns:
        print(f"❌ Column '{problem_column}' not found in CSV.", file=sys.stderr)
        print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
        return
    
    print(f"✅ Found {len(df)} problems to process\n")
    
    # Prepare results storage
    results: List[Dict[str, Any]] = []
    
    # Process each problem
    for idx, row in df.iterrows():
        problem = row[problem_column]
        run_id = f"batch_{idx}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        print(f"{'='*80}")
        print(f"Processing problem {idx + 1}/{len(df)}")
        print(f"{'='*80}")
        print(f"Problem: {problem[:100]}..." if len(problem) > 100 else f"Problem: {problem}")
        print()
        
        # Run the pipeline
        try:
            result = run_from_text_with_state(
                problem=problem,
                output_dir=output_dir,
                run_id=run_id,
            )
            
            # Flatten the result for CSV storage
            flat_result = {
                "index": idx,
                "problem": problem,
                "run_id": run_id,
                "success": result["error"] is None,
                "error": result["error"] or "",
                
                # Parameters (as JSON string)
                "objective": result["bo_params"].get("objective", ""),
                "model": result["bo_params"].get("model", ""),
                "constraint": result["bo_params"].get("constraint", ""),
                "parallelism": result["bo_params"].get("parallelism", ""),
                "custom_gen": result["bo_params"].get("custom_gen", ""),
                "custom_threshold": result["bo_params"].get("custom_threshold", ""),
                "winsorize": result["bo_params"].get("winsorize", ""),
                "standardize_Y": result["bo_params"].get("standardize_Y", ""),
                "log_Y": result["bo_params"].get("log_Y", ""),
                "no_Y_transform": result["bo_params"].get("no_Y_transform", ""),
                "infer_noise": result["bo_params"].get("infer_noise", ""),
                "early_stop": result["bo_params"].get("early_stop", ""),
                
                # Loop statistics
                "retrieval_count": result["retrieval_count"],
                "review_count": result["review_count"],
                
                # Queries and feedback (as strings, truncated for CSV)
                "retrieval_queries": "; ".join(result["retrieval_queries"]),
                "critique_reports": "; ".join(result["critique_reports"]),
                
                # Context count
                "num_contexts": len(result["contexts"]),
                
                # Code lengths
                "skeleton_length": len(result["skeleton_code"]),
                "final_code_length": len(result["final_code"]),
                
                # Full outputs (for reference)
                "skeleton_code": result["skeleton_code"],
                "final_code": result["final_code"],
            }
            
            results.append(flat_result)
            
            # Print summary
            if result["error"]:
                print(f"❌ Failed: {result['error']}")
            else:
                print(f"✅ Success!")
                print(f"   - Retrievals: {result['retrieval_count']}/3")
                print(f"   - Reviews: {result['review_count']}/2")
                print(f"   - Contexts: {len(result['contexts'])}")
                print(f"   - Code length: {len(result['final_code'])} chars")
            print()
            
        except Exception as exc:
            print(f"❌ Unexpected error: {exc}")
            print()
            
            # Add error result
            results.append({
                "index": idx,
                "problem": problem,
                "run_id": run_id,
                "success": False,
                "error": str(exc),
                "objective": "",
                "model": "",
                "constraint": "",
                "parallelism": "",
                "custom_gen": "",
                "custom_threshold": "",
                "winsorize": "",
                "standardize_Y": "",
                "log_Y": "",
                "no_Y_transform": "",
                "infer_noise": "",
                "early_stop": "",
                "retrieval_count": 0,
                "review_count": 0,
                "retrieval_queries": "",
                "critique_reports": "",
                "num_contexts": 0,
                "skeleton_length": 0,
                "final_code_length": 0,
                "skeleton_code": "",
                "final_code": "",
            })
    
    # Save results to CSV
    print(f"{'='*80}")
    print(f"💾 Saving results to: {output_csv}")
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)
    print(f"✅ Batch processing complete!")
    print(f"{'='*80}\n")
    
    # Print summary statistics
    success_count = sum(1 for r in results if r["success"])
    print("📊 Summary:")
    print(f"   Total problems: {len(results)}")
    print(f"   Successful: {success_count}")
    print(f"   Failed: {len(results) - success_count}")
    print(f"   Success rate: {success_count/len(results)*100:.1f}%")
    print()
    
    # Print retrieval/review statistics
    if success_count > 0:
        avg_retrievals = sum(r["retrieval_count"] for r in results if r["success"]) / success_count
        avg_reviews = sum(r["review_count"] for r in results if r["success"]) / success_count
        print(f"   Average retrievals: {avg_retrievals:.2f}/3")
        print(f"   Average reviews: {avg_reviews:.2f}/2")


def main(argv: list[str] | None = None) -> int:
    """Main entry point for batch processing."""
    parser = argparse.ArgumentParser(
        description="Batch process multiple optimization problems from a CSV file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input CSV file with problems.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file for results.",
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help="Directory where generated Python scripts will be saved.",
    )
    parser.add_argument(
        "--problem-column",
        default="problems",
        help="Name of the column containing problem descriptions (default: 'problems').",
    )
    
    args = parser.parse_args(argv)
    
    try:
        process_batch(
            input_csv=args.input,
            output_csv=args.output,
            output_dir=args.output_dir,
            problem_column=args.problem_column,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
