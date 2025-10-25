#!/usr/bin/env python3
"""
Script to run Honegumi RAG Assistant experiments and capture results.
This script runs the RAG assistant on different problem statement versions,
captures intermediate grid selections, final scripts, and logs.
"""

import os
import sys
import yaml
import subprocess
from datetime import datetime
from pathlib import Path
import json

def load_experiments_config():
    """Load experiments configuration from YAML."""
    config_path = Path("data/raw/rag_assistant_runs.yaml")
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def save_experiments_config(config):
    """Save updated experiments configuration to YAML."""
    config_path = Path("data/raw/rag_assistant_runs.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

def run_rag_assistant(prompt, experiment_id, output_dir):
    """
    Run the RAG assistant with a given prompt and capture all output.
    
    Args:
        prompt: The problem description to send to the RAG assistant
        experiment_id: Unique identifier for this experiment
        output_dir: Directory to save generated scripts
    
    Returns:
        dict: Results including status, script path, and log path
    """
    timestamp = datetime.now().isoformat()
    log_file = f"/tmp/rag_experiments/logs/{experiment_id}_{timestamp.replace(':', '-')}.log"
    script_dir = f"/tmp/rag_experiments/scripts/{experiment_id}"
    
    # Create directories
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(script_dir, exist_ok=True)
    
    # Prepare command
    cmd = [
        "honegumi-rag",
        "--output-dir", script_dir,
        "--debug"
    ]
    
    print(f"\n{'='*80}")
    print(f"Running experiment: {experiment_id}")
    print(f"Timestamp: {timestamp}")
    print(f"Prompt: {prompt[:100]}...")
    print(f"{'='*80}\n")
    
    result = {
        "status": "running",
        "timestamp": timestamp,
        "log_path": log_file,
        "script_dir": script_dir,
        "stdout": "",
        "stderr": "",
        "return_code": None
    }
    
    try:
        # Run the command and capture output
        with open(log_file, 'w') as log:
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Send the prompt to stdin
            stdout, stderr = process.communicate(input=prompt, timeout=300)
            
            # Save output to log file
            log.write(f"=== EXPERIMENT: {experiment_id} ===\n")
            log.write(f"Timestamp: {timestamp}\n")
            log.write(f"Prompt:\n{prompt}\n")
            log.write(f"\n{'='*80}\n")
            log.write(f"STDOUT:\n{stdout}\n")
            log.write(f"\n{'='*80}\n")
            log.write(f"STDERR:\n{stderr}\n")
            
            result["stdout"] = stdout
            result["stderr"] = stderr
            result["return_code"] = process.returncode
            result["status"] = "success" if process.returncode == 0 else "failed"
            
            # Find generated script
            script_files = list(Path(script_dir).glob("*.py"))
            if script_files:
                result["generated_script"] = str(script_files[0])
            
            print(f"✓ Experiment {experiment_id} completed with status: {result['status']}")
            print(f"  Log saved to: {log_file}")
            if script_files:
                print(f"  Script saved to: {script_files[0]}")
            
    except subprocess.TimeoutExpired:
        result["status"] = "timeout"
        result["stderr"] = "Process timed out after 300 seconds"
        print(f"✗ Experiment {experiment_id} timed out")
        
    except Exception as e:
        result["status"] = "error"
        result["stderr"] = str(e)
        print(f"✗ Experiment {experiment_id} failed with error: {e}")
    
    return result

def extract_grid_selections_from_log(log_content):
    """
    Extract grid selections from log content if available.
    This looks for parameter extraction or skeleton generation output.
    """
    # This is a placeholder - actual implementation would parse the debug output
    # to extract the grid selections made by the parameter selector
    grid_selections = {
        "note": "Grid selections would be extracted from debug output",
        "parsed": False
    }
    
    # Try to find parameter extraction in the log
    if "Parameter Selector" in log_content or "Skeleton Generator" in log_content:
        grid_selections["parsed"] = True
        # Add actual parsing logic here
    
    return grid_selections

def main():
    """Main execution function."""
    print("Honegumi RAG Assistant Experiment Runner")
    print("="*80)
    
    # Note: LLM_API_KEY and COPILOT_MCP_FIRECRAWL_API_KEY are expected to be
    # available in the GitHub Actions runtime environment as repository secrets.
    # The honegumi-rag tool will access them directly from os.environ.
    
    print("Starting experiments (API keys expected from GitHub Actions environment)")
    
    # Load experiments
    config = load_experiments_config()
    experiments = config.get("experiments", [])
    
    print(f"\nFound {len(experiments)} experiments to run\n")
    
    # Run each experiment
    for i, exp in enumerate(experiments, 1):
        exp_id = exp["experiment_id"]
        prompt = exp["prompt"]
        
        print(f"\n[{i}/{len(experiments)}] Processing {exp_id}...")
        
        # Run the assistant
        result = run_rag_assistant(prompt, exp_id, "/tmp/rag_experiments/scripts")
        
        # Update experiment with results
        exp["timestamp"] = result["timestamp"]
        exp["status"] = result["status"]
        exp["log_path"] = result["log_path"]
        
        if result.get("generated_script"):
            exp["generated_script_path"] = result["generated_script"]
        
        # Try to extract grid selections from log
        if os.path.exists(result["log_path"]):
            with open(result["log_path"], 'r') as f:
                log_content = f.read()
                exp["grid_selections"] = extract_grid_selections_from_log(log_content)
        
        # Save progress after each experiment
        save_experiments_config(config)
        print(f"  Updated configuration saved")
    
    print(f"\n{'='*80}")
    print("All experiments completed!")
    print(f"{'='*80}")
    print(f"\nResults summary:")
    for exp in experiments:
        print(f"  {exp['experiment_id']}: {exp['status']}")
    
    print(f"\nLogs and scripts saved to: /tmp/rag_experiments/")
    print(f"Configuration updated in: data/raw/rag_assistant_runs.yaml")

if __name__ == "__main__":
    main()
