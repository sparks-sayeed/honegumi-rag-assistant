#!/bin/bash
# Script to run RAG experiments and upload artifacts to GitHub Actions
# This should be run after OPENAI_API_KEY is configured

set -e

echo "================================================"
echo "Honegumi RAG Assistant Experiment Runner"
echo "================================================"

# Check for API key
if [ -z "$OPENAI_API_KEY" ]; then
    echo "ERROR: OPENAI_API_KEY not set!"
    echo "Please configure the API key as a repository secret."
    exit 1
fi

echo "✓ API key configured"
echo ""

# Run the experiments
echo "Running experiments..."
python3 scripts/run_rag_experiments.py

# Check if experiments completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Experiments completed successfully"
    echo ""
    
    # List generated files
    echo "Generated artifacts:"
    echo "-------------------"
    ls -lh /tmp/rag_experiments/logs/
    echo ""
    ls -lh /tmp/rag_experiments/scripts/
    
    # Get the current commit hash (short form - 7 characters)
    COMMIT_HASH=$(git rev-parse --short=7 HEAD)
    echo ""
    echo "Commit hash: $COMMIT_HASH"
    
    # Update the YAML with artifact URLs
    # Note: The actual artifact URLs will be added by GitHub Actions
    # This is just a placeholder showing the structure
    echo ""
    echo "Experiment results have been saved to:"
    echo "  - data/raw/rag_assistant_runs.yaml"
    echo "  - /tmp/rag_experiments/ (temporary, will be uploaded as artifacts)"
    echo ""
    echo "Artifact URLs will be available at:"
    echo "  https://github.com/sparks-sayeed/honegumi-rag-assistant/actions"
    echo ""
    echo "Direct links to logs will use the commit hash: $COMMIT_HASH"
    
else
    echo ""
    echo "✗ Experiments failed"
    exit 1
fi
