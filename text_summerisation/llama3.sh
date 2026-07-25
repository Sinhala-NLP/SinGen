#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40:3
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# Summarisation is a LONG-input task (full XL-Sum articles are passed
# untruncated), so batches are smaller than the translation/headline jobs:
# 70B checkpoints at 2, everything else at 4.
#
# Decoding is greedy (no --do_sample) for ROUGE reproducibility. max_new_tokens
# is 512 to give summaries room; Llama has no thinking mode, so the only risk is
# truncating a long summary.
#
# test_size is left at the default (0 = full test set = 500 instances).
#
# CONTEXT-LENGTH WARNING: Meta-Llama-3-8B-Instruct and Meta-Llama-3-70B-Instruct
# have only an 8k window. Long XL-Sum articles can overflow it (the 3.1/3.2/3.3
# checkpoints are 128k and fine). If those two error or produce empty preds,
# cap input length for them specifically rather than changing it globally, so
# the 128k models stay comparable to the other families.
#
# Note: all Llama-3 checkpoints are gated -- HF_TOKEN must be set above and the
# Llama-3 license accepted on the HF account.

for model in \
    meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-3.1-8B-Instruct \
    meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.3-70B-Instruct; do

    case "$model" in
        *70B*) bs=2 ;;
        *)     bs=4 ;;
    esac

    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m llama3_summarisation \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 512
    done
done