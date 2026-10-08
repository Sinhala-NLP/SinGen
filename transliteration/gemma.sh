#!/bin/bash
#SBATCH -p gpu-short
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --array=0-15
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# 16 array tasks = 4 models x 4 query types (task id -> model = id / 4, query type = id % 4)
models=(google/gemma-4-31B-it google/gemma-4-12B-it google/gemma-3-27b-it google/gemma-3-12b-it)
qts=(zero-shot zero-shot-si few-shot few-shot-si)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
qt=${qts[$((SLURM_ARRAY_TASK_ID % 4))]}

# Linearised infoboxes reach ~5k chars and few-shot prompts stack three of them.
declare -A BATCH=(
    ["google/gemma-4-31B-it"]=8
    ["google/gemma-4-12B-it"]=16
    ["google/gemma-3-27b-it"]=8
    ["google/gemma-3-12b-it"]=16
)
bs=${BATCH[$model]}

echo "=== $model | $qt | batch_size=$bs ==="
python -m gemma \
    --model_id "$model" \
    --query_type "$qt" \
    --batch_size "$bs" \
    --max_new_tokens 256