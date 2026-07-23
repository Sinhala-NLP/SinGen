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

# Headline generation prompts carry up to 2500 chars of article (plus 3
# few-shot articles at 500 chars each), so batches are smaller than in MT.
for model in \
    meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-3.1-8B-Instruct \
    meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.3-70B-Instruct; do

    case "$model" in
        *70B*) bs=2 ;;
        *8B*)  bs=4 ;;
        *)     bs=8 ;;
    esac

    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m llama_headline_generation \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 128 \
            --test_size 1000
    done
done