#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# Headline prompts carry up to 2500 chars of article (plus 3 few-shot articles
# at 500 chars each), so batches are smaller than in text simplification.
# max_new_tokens is 256 rather than the 128 used for Gemma/Llama: under greedy
# decoding a larger budget cannot change the extracted headline, it only
# protects against a residual <think> block eating the whole budget and
# leaving an empty prediction (which would score ROUGE 0 and look like a
# model failure). Watch the "Empty predictions" line in each log.
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) bs=2 ;;
        *9B*)        bs=4 ;;
        *)           bs=8 ;;
    esac

    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m evaluate_headline_generation_qwen \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 256 \
            --test_size 1000
    done
done