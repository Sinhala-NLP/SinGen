#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --array=0-67

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# 68 array tasks = 17 models x 4 query types (model = id / 4, query type = id % 4).
# Qwen2.5-72B does not fit on one H200 and runs from qwen_transliteration_72b.sh.
models=(
    Qwen/Qwen2-0.5B-Instruct
    Qwen/Qwen2-1.5B-Instruct
    Qwen/Qwen2-7B-Instruct
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B
    Qwen/Qwen3.5-35B-A3B
    Qwen/Qwen3.6-27B
    Qwen/Qwen3.6-35B-A3B
)
qts=(zero-shot zero-shot-si few-shot few-shot-si)
model=${models[$((SLURM_ARRAY_TASK_ID / 4))]}
qt=${qts[$((SLURM_ARRAY_TASK_ID % 4))]}

# Short single-sentence prompts, so batches can be large.
declare -A BATCH=(
    ["Qwen/Qwen2.5-32B-Instruct"]=32
    ["Qwen/Qwen3.5-27B"]=32
    ["Qwen/Qwen3.5-35B-A3B"]=32
    ["Qwen/Qwen3.6-27B"]=32
    ["Qwen/Qwen3.6-35B-A3B"]=32
)
bs=${BATCH[$model]:-64}

if [ -f "outputs/transliteration/${model#*/}/$qt/translit_summary.txt" ]; then
    echo "=== $model | $qt already done, skipping ==="
    exit 0
fi

echo "=== $model | $qt | batch_size=$bs ==="
python -m qwen_transliteration \
    --model_id "$model" \
    --query_type "$qt" \
    --batch_size "$bs" \
    --max_new_tokens 512