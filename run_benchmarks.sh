#!/bin/bash
# Unified Benchmark Runner Shell Script
# This is a convenient wrapper around run_all_benchmarks.py

set -e

# Default values
MODEL=""
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
Usage: $0 --model <model_name> --query_type <query_type> [options]

Required Arguments:
    --model             Model backend (open_ai, cohere, hf_llm, mt0)
    --query_type        Query type (zero-shot, zero-shot-si, few-shot, few-shot-si)

Optional Arguments:
    --tasks             Comma-separated tasks (default: all)
                        Options: simplification, summarization, headline, translation
    --translation_pairs Comma-separated pairs (default: all)
                        Options: en_si, ta_si, pi_si
    -h, --help         Show this help message

Examples:
    # Run all benchmarks with Cohere model
    $0 --model cohere --query_type zero-shot

    # Run specific tasks with OpenAI
    $0 --model open_ai --query_type few-shot --tasks simplification,summarization

    # Run only English-Sinhala translation
    $0 --model hf_llm --query_type zero-shot-si --tasks translation --translation_pairs en_si

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
if [ -z "$MODEL" ] || [ -z "$QUERY_TYPE" ]; then
    echo -e "${RED}Error: --model and --query_type are required${NC}"
    usage
fi

# Run the Python script
echo -e "${GREEN}Starting benchmark runner...${NC}"
python run_all_benchmarks.py \
    --model "$MODEL" \
    --query_type "$QUERY_TYPE" \
    --tasks "$TASKS" \
    --translation_pairs "$TRANSLATION_PAIRS"

exit_code=$?

if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}All benchmarks completed successfully!${NC}"
else
    echo -e "${RED}Some benchmarks failed. Check the output above for details.${NC}"
fi

exit $exit_code
