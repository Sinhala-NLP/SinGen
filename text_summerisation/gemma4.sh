#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
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

# Summarisation uses full XL-Sum Sinhala test split (500 = Table 1), so no
# --test_size is passed. batch_size lowered to 4 vs. the simplification job:
# XL-Sum articles are long (~1000 input tokens) + 512 new tokens, so packing
# 8 into a batch for the 31B model risks OOM. Raise for gemma-4-12B-it if headroom allows.
for model in google/gemma-4-31B-it google/gemma-4-12B-it; do
    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m gemma4 \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size 4 \
            --max_new_tokens 512
    done
done