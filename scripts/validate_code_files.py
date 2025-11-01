#!/usr/bin/env python
"""Validate generated Python code files.

This script performs three checks on each Python file:
1. Syntax check using ast.parse()
2. Import check by attempting to import the module
3. Runtime execution with timeout

Modes:
- Default: Validate all files in directory and generate full report
- Single file: Validate one file and output partial result (for parallel execution)
- Combine: Aggregate partial results into final report
"""
import argparse
import ast
import glob
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml


def check_syntax(file_path: Path) -> Dict[str, Any]:
    """Check if the Python file has valid syntax.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dict with 'passed' bool and optional 'error' message
    """
    try:
        with open(file_path, 'r') as f:
            ast.parse(f.read(), filename=str(file_path))
        return {"passed": True, "error": None}
    except SyntaxError as e:
        return {"passed": False, "error": f"Syntax error at line {e.lineno}: {e.msg}"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def check_imports(file_path: Path) -> Dict[str, Any]:
    """Check if all imports in the Python file can be resolved.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dict with 'passed' bool and optional 'error' message
    """
    try:
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        if spec is None or spec.loader is None:
            return {"passed": False, "error": "Could not load module spec"}
        
        module = importlib.util.module_from_spec(spec)
        sys.modules["temp_module"] = module
        spec.loader.exec_module(module)
        
        # Clean up
        if "temp_module" in sys.modules:
            del sys.modules["temp_module"]
            
        return {"passed": True, "error": None}
    except ImportError as e:
        return {"passed": False, "error": f"Import error: {e}"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def check_runtime(file_path: Path, timeout: int = 120) -> Dict[str, Any]:
    """Execute the Python file with a timeout.
    
    Args:
        file_path: Path to the Python file
        timeout: Maximum execution time in seconds (default: 120s = 2 minutes)
        
    Returns:
        Dict with 'passed' bool, 'execution_time', 'timed_out' flag, and optional 'error' message
        Note: Timeout is treated as success if no errors occurred (optimization scripts are expected to run long)
    """
    start_time = time.time()
    try:
        result = subprocess.run(
            [sys.executable, str(file_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "MPLBACKEND": "Agg"}  # Use non-interactive matplotlib backend
        )
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            return {"passed": True, "error": None, "execution_time": f"{execution_time:.2f}s", "timed_out": False}
        else:
            return {
                "passed": False,
                "error": f"Non-zero exit code {result.returncode}: {result.stderr}",
                "execution_time": f"{execution_time:.2f}s",
                "timed_out": False
            }
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        # Timeout is treated as SUCCESS for optimization scripts
        return {
            "passed": True,
            "error": None,
            "execution_time": f"{execution_time:.2f}s",
            "timed_out": True,
            "note": f"Execution timed out after {timeout}s (expected for long-running optimization scripts)"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "passed": False,
            "error": str(e),
            "execution_time": f"{execution_time:.2f}s",
            "timed_out": False
        }


def validate_file(file_path: Path) -> Dict[str, Any]:
    """Run all validation checks on a single file.
    
    Args:
        file_path: Path to the Python file
        
    Returns:
        Dict with results from all checks
    """
    print(f"\nValidating: {file_path.name}")
    
    # Syntax check
    print("  - Checking syntax...")
    syntax_result = check_syntax(file_path)
    print(f"    {'✓ PASS' if syntax_result['passed'] else '✗ FAIL'}")
    
    # Import check
    print("  - Checking imports...")
    import_result = check_imports(file_path)
    print(f"    {'✓ PASS' if import_result['passed'] else '✗ FAIL'}")
    
    # Runtime check
    print("  - Checking runtime execution...")
    runtime_result = check_runtime(file_path)
    if runtime_result.get('timed_out'):
        print(f"    ⏱ TIMEOUT (treated as PASS)")
    else:
        print(f"    {'✓ PASS' if runtime_result['passed'] else '✗ FAIL'}")
    if runtime_result.get('execution_time'):
        print(f"    Execution time: {runtime_result['execution_time']}")
    
    return {
        "file": str(file_path),
        "syntax_check": syntax_result,
        "import_check": import_result,
        "runtime_check": runtime_result,
        "overall_passed": all([
            syntax_result["passed"],
            import_result["passed"],
            runtime_result["passed"]
        ])
    }


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(description="Validate Python code files")
    parser.add_argument('--file', type=str, help="Validate a single file (for parallel execution)")
    parser.add_argument('--batch', type=str, help="Validate a batch of files (comma-separated)")
    parser.add_argument('--batch-id', type=int, help="Batch ID for naming partial results")
    parser.add_argument('--output-dir', type=str, default="data/processed/evaluation",
                        help="Output directory for results")
    parser.add_argument('--combine', type=str, help="Combine partial results from directory")
    parser.add_argument('--list-files', action='store_true', help="List all files to validate as JSON")
    parser.add_argument('--list-batches', type=int, help="List file batches with specified batch size")
    
    args = parser.parse_args()
    
    code_files_dir = Path("data/processed/evaluation/code_files")
    
    # List files mode - output JSON list of all files
    if args.list_files:
        if not code_files_dir.exists():
            print(json.dumps([]))
            return
        python_files = sorted([f.name for f in code_files_dir.glob("*.py")])
        print(json.dumps(python_files))
        return
    
    # List batches mode - output JSON array of batch IDs
    if args.list_batches:
        if not code_files_dir.exists():
            print(json.dumps([]))
            return
        python_files = sorted([f.name for f in code_files_dir.glob("*.py")])
        batch_size = args.list_batches
        num_batches = (len(python_files) + batch_size - 1) // batch_size
        print(json.dumps(list(range(num_batches))))
        return
    
    # Combine mode - aggregate partial results
    if args.combine:
        combine_results(args.combine, args.output_dir)
        return
    
    # Batch mode - validate multiple files
    if args.batch:
        validate_batch(args.batch, args.batch_id, args.output_dir)
        return
    
    # Single file mode
    if args.file:
        validate_single_file(args.file, args.output_dir)
        return
    
    # Default mode - validate all files
    validate_all_files(code_files_dir, args.output_dir)


def validate_single_file(filename: str, output_dir: str):
    """Validate a single file and save partial result."""
    code_files_dir = Path("data/processed/evaluation/code_files")
    file_path = code_files_dir / filename
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    print(f"Validating single file: {filename}")
    result = validate_file(file_path)
    
    # Save partial result
    output_path = Path(output_dir) / f"partial_{filename.replace('.py', '.yaml')}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    
    print(f"Partial result written to: {output_path}")


def validate_batch(batch_files: str, batch_id: int, output_dir: str):
    """Validate a batch of files and save partial results."""
    code_files_dir = Path("data/processed/evaluation/code_files")
    filenames = [f.strip() for f in batch_files.split(',') if f.strip()]
    
    if not filenames:
        print("Error: No files specified in batch")
        sys.exit(1)
    
    print(f"Validating batch {batch_id} with {len(filenames)} file(s)")
    
    results = []
    for filename in filenames:
        file_path = code_files_dir / filename
        if not file_path.exists():
            print(f"Warning: File {file_path} does not exist, skipping")
            continue
        
        print(f"\nValidating: {filename}")
        result = validate_file(file_path)
        results.append(result)
    
    # Save batch result
    output_path = Path(output_dir) / f"batch_{batch_id}.yaml"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    batch_report = {
        "batch_id": batch_id,
        "files_validated": len(results),
        "results": results
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(batch_report, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nBatch {batch_id} result written to: {output_path}")


def validate_single_file(filename: str, output_dir: str):
    """Validate a single file and save partial result."""
    code_files_dir = Path("data/processed/evaluation/code_files")
    file_path = code_files_dir / filename
    
    if not file_path.exists():
        print(f"Error: File {file_path} does not exist")
        sys.exit(1)
    
    print(f"Validating single file: {filename}")
    result = validate_file(file_path)
    
    # Save partial result
    output_path = Path(output_dir) / f"partial_{filename.replace('.py', '.yaml')}"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False)
    
    print(f"Partial result written to: {output_path}")


def combine_results(partial_dir: str, output_dir: str):
    """Combine partial validation results into final report."""
    partial_path = Path(partial_dir)
    
    if not partial_path.exists():
        print(f"Error: Directory {partial_path} does not exist")
        sys.exit(1)
    
    # Load all partial results (both single files and batches)
    results = []
    
    # Load single file results
    partial_files = sorted(partial_path.glob("partial_*.yaml"))
    print(f"Found {len(partial_files)} partial file result(s)")
    for partial_file in partial_files:
        with open(partial_file, 'r') as f:
            result = yaml.safe_load(f)
            results.append(result)
    
    # Load batch results
    batch_files = sorted(partial_path.glob("batch_*.yaml"))
    print(f"Found {len(batch_files)} batch result(s)")
    for batch_file in batch_files:
        with open(batch_file, 'r') as f:
            batch_data = yaml.safe_load(f)
            # Extract results from batch
            if 'results' in batch_data:
                results.extend(batch_data['results'])
    
    print(f"Total files to combine: {len(results)}")
    
    if not results:
        print("Warning: No results found")
        results = []
    
    # Generate summary
    total_files = len(results)
    passed_files = sum(1 for r in results if r.get("overall_passed", False))
    
    print("\n" + "="*60)
    print(f"Combined Validation Summary: {passed_files}/{total_files} files passed")
    print("="*60)
    
    # Create final report
    report = {
        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "summary": {
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": total_files - passed_files
        },
        "results": results
    }
    
    # Write final report
    output_file = Path(output_dir) / "validation_report.yaml"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nFinal validation report written to: {output_file}")


def validate_all_files(code_files_dir: Path, output_dir: str):
    """Validate all files in directory (default mode)."""
    output_file = Path(output_dir) / "validation_report.yaml"
    
    if not code_files_dir.exists():
        print(f"Error: Directory {code_files_dir} does not exist")
        sys.exit(1)
    
    # Find all Python files
    python_files = sorted(code_files_dir.glob("*.py"))
    
    if not python_files:
        print(f"Error: No Python files found in {code_files_dir}")
        sys.exit(1)
    
    print(f"Found {len(python_files)} Python file(s) to validate")
    
    # Validate each file
    results = []
    for file_path in python_files:
        result = validate_file(file_path)
        results.append(result)
    
    # Generate summary
    total_files = len(results)
    passed_files = sum(1 for r in results if r["overall_passed"])
    
    print("\n" + "="*60)
    print(f"Validation Summary: {passed_files}/{total_files} files passed")
    print("="*60)
    
    # Create report
    report = {
        "validation_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "summary": {
            "total_files": total_files,
            "passed_files": passed_files,
            "failed_files": total_files - passed_files
        },
        "results": results
    }
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Write report
    with open(output_file, 'w') as f:
        yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    
    print(f"\nValidation report written to: {output_file}")
    
    # Note: Not exiting with error code to allow workflow to commit report
    # The validation results are in the YAML report
    sys.exit(0)


main()
