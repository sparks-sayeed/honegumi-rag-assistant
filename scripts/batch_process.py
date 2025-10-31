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
import json
from pathlib import Path
import pandas as pd
from datetime import datetime
from typing import Dict, Any, List

# Add parent directory to path to import honegumi_rag_assistant
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from honegumi_rag_assistant.orchestrator import run_from_text_with_state


def compare_parameters(expected: dict, generated: dict) -> tuple[int, int, list[str]]:
    """Compare expected and generated BO parameters.
    
    Parameters
    ----------
    expected : dict
        Expected grid selections from the input CSV.
    generated : dict
        Generated BO parameters from the pipeline.
    
    Returns
    -------
    tuple[int, int, list[str]]
        (num_correct, num_incorrect, list of incorrect keys)
    """
    # Common keys to compare
    common_keys = set(expected.keys()) & set(generated.keys())
    
    correct_count = 0
    incorrect_keys = []
    
    for key in common_keys:
        expected_val = expected[key]
        generated_val = generated[key]
        
        # Compare values (handle different types)
        if expected_val == generated_val:
            correct_count += 1
        else:
            incorrect_keys.append(key)
    
    incorrect_count = len(incorrect_keys)
    
    return correct_count, incorrect_count, incorrect_keys


def process_batch(
    input_csv: str,
    output_csv: str,
    scripts_dir: str | None = None,
    problem_column: str = "problem",
) -> None:
    """Process a batch of problems from a CSV file.

    Parameters
    ----------
    input_csv : str
        Path to input CSV file with a column containing problem descriptions.
    output_csv : str
        Path to output CSV file where results will be saved.
    scripts_dir : str or None, optional
        Directory where generated Python scripts will be saved.
    problem_column : str, optional
        Name of the column containing problem descriptions. Default: "problem".
    """
    # Read input CSV
    print(f"📖 Reading problems from: {input_csv}")
    try:
        # Try UTF-8 first, then fall back to other common encodings
        try:
            df = pd.read_csv(input_csv, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(input_csv, encoding='latin-1')
            except UnicodeDecodeError:
                df = pd.read_csv(input_csv, encoding='cp1252')
    except Exception as exc:
        print(f"❌ Error reading CSV: {exc}", file=sys.stderr)
        return
    
    if problem_column not in df.columns:
        print(f"❌ Column '{problem_column}' not found in CSV.", file=sys.stderr)
        print(f"Available columns: {', '.join(df.columns)}", file=sys.stderr)
        return
    
    print(f"✅ Found {len(df)} problems to process\n")
    
    # Check for problem_id column
    if 'problem_id' not in df.columns:
        print("⚠️  Warning: 'problem_id' column not found. Using index as ID.", file=sys.stderr)
    
    # Prepare results storage
    results: List[Dict[str, Any]] = []
    
    # Process each problem
    for idx, row in df.iterrows():
        problem = row[problem_column]
        problem_id = row.get('problem_id', f"problem_{idx}")
        
        print(f"{'='*80}")
        print(f"Processing problem {idx + 1}/{len(df)}: {problem_id}")
        print(f"{'='*80}")
        print(f"Problem: {problem[:100]}..." if len(problem) > 100 else f"Problem: {problem}")
        print()
        
        # Run the pipeline
        try:
            result = run_from_text_with_state(
                problem=problem,
                output_dir=scripts_dir,
                run_id=problem_id,
            )
            
            # Start with all input columns
            flat_result = row.to_dict()
            
            # Add result columns
            flat_result.update({
                "success": result["error"] is None,
                "error": result["error"] or "",
                
                # Stage 1: Problem Structure (as formatted JSON string)
                "problem_structure": json.dumps(result.get("problem_structure", {}), ensure_ascii=False),
                
                # Stage 2: Grid Parameters (as formatted JSON string)
                "bo_params": json.dumps(result.get("bo_params", {}), ensure_ascii=False),
                
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
                
                # Full outputs
                "skeleton_code": result["skeleton_code"],
                "final_code": result["final_code"],
                "final_code_filename": f"RAG_generated_{problem_id}.py" if result["final_code"] else "",
            })
            
            # Compare expected_grid_selections with bo_params if available
            if 'expected_grid_selections' in row:
                try:
                    expected = json.loads(row['expected_grid_selections']) if isinstance(row['expected_grid_selections'], str) else row['expected_grid_selections']
                    generated = result.get("bo_params", {})
                    
                    correct, incorrect, incorrect_keys = compare_parameters(expected, generated)
                    
                    flat_result.update({
                        "params_correct": correct,
                        "params_incorrect": incorrect,
                        "params_incorrect_keys": json.dumps(incorrect_keys),
                    })
                except Exception as e:
                    print(f"⚠️  Warning: Could not compare parameters: {e}")
                    flat_result.update({
                        "params_correct": 0,
                        "params_incorrect": 0,
                        "params_incorrect_keys": "[]",
                    })
            
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
                
                # Print parameter comparison if available
                if 'expected_grid_selections' in row and 'params_correct' in flat_result:
                    total_params = flat_result['params_correct'] + flat_result['params_incorrect']
                    print(f"   - Parameters: {flat_result['params_correct']}/{total_params} correct")
                    if flat_result['params_incorrect'] > 0:
                        print(f"     Incorrect: {flat_result['params_incorrect_keys']}")
            print()
            
        except Exception as exc:
            print(f"❌ Unexpected error: {exc}")
            print()
            
            # Start with all input columns
            flat_result = row.to_dict()
            
            # Add error result columns
            flat_result.update({
                "success": False,
                "error": str(exc),
                "problem_structure": "{}",
                "bo_params": "{}",
                "retrieval_count": 0,
                "review_count": 0,
                "retrieval_queries": "",
                "critique_reports": "",
                "num_contexts": 0,
                "skeleton_length": 0,
                "final_code_length": 0,
                "skeleton_code": "",
                "final_code": "",
                "final_code_filename": "",
                "params_correct": 0,
                "params_incorrect": 0,
                "params_incorrect_keys": "[]",
            })
            
            results.append(flat_result)
    
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
        "--scripts-dir",
        required=False,
        help="Directory where generated Python scripts will be saved.",
    )
    parser.add_argument(
        "--problem-column",
        default="problem",
        help="Name of the column containing problem descriptions (default: 'problem').",
    )
    
    args = parser.parse_args(argv)
    
    try:
        process_batch(
            input_csv=args.input,
            output_csv=args.output,
            scripts_dir=args.scripts_dir,
            problem_column=args.problem_column,
        )
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
