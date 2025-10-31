#!/usr/bin/env python
"""Validate generated code files by running them in isolated subprocesses with timeouts."""

import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Any
import traceback

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not installed. Install it with: pip install pyyaml")
    yaml = None


def find_generated_files(base_path: Path) -> List[Path]:
    """Find all generated Python files to validate."""
    code_files_dir = base_path / "data" / "processed" / "evaluation" / "code_files"
    if not code_files_dir.exists():
        return []
    
    # Find all .py files in the code_files directory
    return sorted(code_files_dir.glob("*.py"))


def run_file_in_subprocess(
    file_path: Path,
    timeout: int,
    validation_mode: str = "1"
) -> Dict[str, Any]:
    """
    Run a Python file in an isolated subprocess with timeout.
    
    Args:
        file_path: Path to the Python file to execute
        timeout: Timeout in seconds for the subprocess
        validation_mode: Value for VALIDATION_MODE environment variable
    
    Returns:
        Dict with status, duration, stdout, stderr, and error information
    """
    result = {
        "file": str(file_path.name),
        "status": "unknown",
        "duration": 0.0,
        "stdout": "",
        "stderr": "",
        "exit_code": None,
        "error_trace": None,
    }
    
    start_time = time.time()
    
    try:
        # Set up environment with VALIDATION_MODE
        env = os.environ.copy()
        env["VALIDATION_MODE"] = validation_mode
        
        # Run the file in a subprocess
        proc = subprocess.run(
            [sys.executable, str(file_path)],
            timeout=timeout,
            capture_output=True,
            text=True,
            env=env,
        )
        
        duration = time.time() - start_time
        result["duration"] = round(duration, 2)
        result["stdout"] = proc.stdout
        result["stderr"] = proc.stderr
        result["exit_code"] = proc.returncode
        
        if proc.returncode == 0:
            result["status"] = "passed"
        else:
            result["status"] = "failed"
            result["error_trace"] = f"Process exited with code {proc.returncode}"
            
    except subprocess.TimeoutExpired as e:
        duration = time.time() - start_time
        result["duration"] = round(duration, 2)
        result["status"] = "timeout"
        result["error_trace"] = f"Execution timed out after {timeout}s"
            
    except Exception as e:
        duration = time.time() - start_time
        result["duration"] = round(duration, 2)
        result["status"] = "error"
        result["error_trace"] = traceback.format_exc()
        result["stderr"] = str(e)
    
    return result


def generate_validation_report(results: List[Dict[str, Any]], output_path: Path):
    """Generate a YAML validation report with per-file diagnostics."""
    passed = sum(1 for r in results if r["status"] == "passed")
    total = len(results)
    
    report = {
        "summary": {
            "total_files": total,
            "passed": passed,
            "failed": total - passed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        },
        "results": results,
    }
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write the report
    if yaml:
        with open(output_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback to simple text format if yaml not available
        with open(output_path, "w") as f:
            f.write(f"# Validation Report\n")
            f.write(f"Total files: {total}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {total - passed}\n\n")
            for r in results:
                f.write(f"## {r['file']}\n")
                f.write(f"Status: {r['status']}\n")
                f.write(f"Duration: {r['duration']}s\n")
                if r.get('error_trace'):
                    f.write(f"Error: {r['error_trace']}\n")
                f.write("\n")
    
    return report


def main():
    """Main validation logic."""
    # Get the repository root
    repo_root = Path(__file__).parent.parent
    
    # Get timeout from environment or use default
    per_file_timeout = int(os.environ.get("PER_FILE_TIMEOUT", "30"))
    validation_mode = os.environ.get("VALIDATION_MODE", "1")
    
    print(f"Validation configuration:")
    print(f"  Per-file timeout: {per_file_timeout}s")
    print(f"  Validation mode: {validation_mode}")
    print()
    
    # Find files to validate
    files = find_generated_files(repo_root)
    
    if not files:
        print("No generated code files found to validate.")
        return 0
    
    print(f"Found {len(files)} file(s) to validate:")
    for f in files:
        print(f"  - {f.name}")
    print()
    
    # Run validation on each file
    results = []
    for file_path in files:
        print(f"Validating {file_path.name}...")
        result = run_file_in_subprocess(
            file_path,
            timeout=per_file_timeout,
            validation_mode=validation_mode
        )
        results.append(result)
        
        # Print result
        status_symbol = "✓" if result["status"] == "passed" else "✗"
        print(f"  {status_symbol} Status: {result['status']} (Duration: {result['duration']}s)")
        
        if result["status"] != "passed":
            if result.get("error_trace"):
                print(f"    Error: {result['error_trace']}")
            if result.get("stderr"):
                # Print first few lines of stderr
                stderr_lines = result["stderr"].split("\n")[:5]
                for line in stderr_lines:
                    if line.strip():
                        print(f"    {line}")
        print()
    
    # Generate report
    output_path = repo_root / "data" / "processed" / "evaluation" / "validation_report.yaml"
    report = generate_validation_report(results, output_path)
    
    print(f"Validation complete!")
    print(f"Summary: {report['summary']['passed']}/{report['summary']['total_files']} files passed")
    print(f"Report saved to: {output_path}")
    
    # Return non-zero exit code if any validations failed
    if report['summary']['failed'] > 0:
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
