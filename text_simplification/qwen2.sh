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

for model in \
    Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct \
    Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct Qwen/Qwen2.5-72B-Instruct \
    Qwen/Qwen2-0.5B-Instruct Qwen/Qwen2-1.5B-Instruct Qwen/Qwen2-7B-Instruct \
    Qwen/Qwen2-57B-A14B-Instruct Qwen/Qwen2-72B-Instruct; do
    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m qwen2 \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size 8 \
            --max_new_tokens 128
    done
done



