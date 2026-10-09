#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-3

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# Qwen2.5-72B in bf16 is ~145GB, so it needs two H200s.
# 4 array tasks = one per query type.
model=Qwen/Qwen2.5-72B-Instruct
qts=(zero-shot zero-shot-si few-shot few-shot-si)
qt=${qts[$SLURM_ARRAY_TASK_ID]}
bs=16

if [ -f "outputs/transliteration/${model#*/}/$qt/translit_summary.txt" ]; then
    echo "=== $model | $qt already done, skipping ==="
    exit 0
fi

echo "=== $model | $qt | batch_size=$bs ==="
python -m qwen \
    --model_id "$model" \
    --query_type "$qt" \
    --batch_size "$bs" \
    --max_new_tokens 512