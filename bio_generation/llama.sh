#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-15


source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# 16 array tasks = 4 models x 4 query types (model = id / 4, query type = id % 4).
# 8B and smaller fit on one H200.
models=(
    meta-llama/Meta-Llama-3-8B-Instruct
    meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Llama-3.2-1B-Instruct
    meta-llama/Llama-3.2-3B-Instruct
)
qts=(zero-shot zero-shot-si few-shot few-shot-si)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
qt=${qts[$((SLURM_ARRAY_TASK_ID % 4))]}

# Linearised infoboxes reach ~5k chars and few-shot prompts stack three of them.
bs=16

if [ -f "outputs/biography_generation/${model#*/}/$qt/rouge_summary.txt" ]; then
    echo "=== $model | $qt already done, skipping ==="
    exit 0
fi

echo "=== $model | $qt | batch_size=$bs ==="
python -m llama \
    --model_id "$model" \
    --query_type "$qt" \
    --batch_size "$bs" \
    --max_new_tokens 512