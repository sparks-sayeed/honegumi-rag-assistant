#!/usr/bin/env python
"""Validate generated Python code files.

This script performs three checks on each Python file:
1. Syntax check using py_compile
2. Import check by attempting to import the module
3. Runtime execution with timeout
"""
import ast
import importlib.util
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import yaml


def check_syntax(file_path: Path) -> Dict[str, any]:
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


def check_imports(file_path: Path) -> Dict[str, any]:
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


def check_runtime(file_path: Path, timeout: int = 30) -> Dict[str, any]:
    """Execute the Python file with a timeout.
    
    Args:
        file_path: Path to the Python file
        timeout: Maximum execution time in seconds
        
    Returns:
        Dict with 'passed' bool, 'execution_time', and optional 'error' message
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
            return {"passed": True, "error": None, "execution_time": f"{execution_time:.2f}s"}
        else:
            return {
                "passed": False,
                "error": f"Non-zero exit code {result.returncode}: {result.stderr}",
                "execution_time": f"{execution_time:.2f}s"
            }
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "passed": False,
            "error": f"Execution timeout after {timeout}s",
            "execution_time": f"{execution_time:.2f}s"
        }
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "passed": False,
            "error": str(e),
            "execution_time": f"{execution_time:.2f}s"
        }


def validate_file(file_path: Path) -> Dict[str, any]:
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
    print(f"    {'✓ PASS' if runtime_result['passed'] else '✗ FAIL'}")
    if runtime_result.get('execution_time'):
        print(f"    Execution time: {runtime_result['execution_time']}")
    
    return {
        "file": str(file_path.relative_to(Path.cwd())),
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
    code_files_dir = Path("data/processed/evaluation/code_files")
    output_file = Path("data/processed/evaluation/validation_report.yaml")
    
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
    
    # Exit with error code if any validation failed
    if passed_files < total_files:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
