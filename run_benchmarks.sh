#!/bin/bash
# Unified Benchmark Runner Shell Script
# This is a convenient wrapper around run_all_benchmarks.py

set -e

# Default values
MODEL=""
MODEL_NAME=""
QUERY_TYPE=""
TASKS="all"
TRANSLATION_PAIRS="all"

# Color codes for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to display usage
usage() {
    cat << EOF
Usage: $0 [--model <backend>] [--model_name <model>] --query_type <query_type> [options]

Arguments:
    --model             Backend name (open_ai, cohere, hf_llm, mt0) OR model name for auto-detection
    --model_name        Actual model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct, gpt-4o)
    --query_type        Query type (zero-shot, zero-shot-si, few-shot, few-shot-si) [REQUIRED]

Note: You must provide at least one of --model or --model_name:
    - Both --model and --model_name: Explicit control (RECOMMENDED)
    - Only --model: If backend name, uses default model; if model name, auto-detects backend
    - Only --model_name: Auto-detects backend from model name

Optional Arguments:
    --tasks             Comma-separated tasks (default: all)
                        Options: simplification, summarization, headline, translation
    --translation_pairs Comma-separated pairs (default: all)
                        Options: en_si, ta_si, pi_si
    -h, --help         Show this help message

Examples:
    # Explicit: Specify both backend and model name (RECOMMENDED)
    $0 --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
    $0 --model open_ai --model_name gpt-4o --query_type few-shot
    $0 --model cohere --model_name command-r --query_type zero-shot-si

    # Auto-detect backend from model name
    $0 --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
    $0 --model_name gpt-4o --query_type few-shot

    # Use backend with default model
    $0 --model cohere --query_type zero-shot

    # Run specific tasks
    $0 --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type few-shot --tasks simplification,summarization

    # Run only English-Sinhala translation
    $0 --model open_ai --model_name gpt-4o --query_type zero-shot-si --tasks translation --translation_pairs en_si

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --model_name)
            MODEL_NAME="$2"
            shift 2
            ;;
        --query_type)
            QUERY_TYPE="$2"
            shift 2
            ;;
        --tasks)
            TASKS="$2"
            shift 2
            ;;
        --translation_pairs)
            TRANSLATION_PAIRS="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            ;;
    esac
done

# Validate required arguments
if [ -z "$QUERY_TYPE" ]; then
    echo -e "${RED}Error: --query_type is required${NC}"
    usage
fi

if [ -z "$MODEL" ] && [ -z "$MODEL_NAME" ]; then
    echo -e "${RED}Error: At least one of --model or --model_name must be provided${NC}"
    usage
fi

# Run the Python script
echo -e "${GREEN}Starting benchmark runner...${NC}"

# Build the command with optional arguments
CMD="python run_all_benchmarks.py"
if [ -n "$MODEL" ]; then
    CMD="$CMD --model \"$MODEL\""
fi
if [ -n "$MODEL_NAME" ]; then
    CMD="$CMD --model_name \"$MODEL_NAME\""
fi
CMD="$CMD --query_type \"$QUERY_TYPE\""
CMD="$CMD --tasks \"$TASKS\""
CMD="$CMD --translation_pairs \"$TRANSLATION_PAIRS\""

# Execute the command
eval $CMD

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}All benchmarks completed successfully!${NC}"
else
    echo -e "${RED}Some benchmarks failed. Check the output above for details.${NC}"
fi

exit $exit_code
