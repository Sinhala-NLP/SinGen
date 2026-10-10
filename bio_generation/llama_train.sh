#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --array=0-13
#SBATCH --requeue


# Usage:  sbatch llama_finetune_biogen.sh
# One array task = one (model, fine-tune language): 7 models x 2 languages = 14.

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=          # needed: Llama checkpoints are gated
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    meta-llama/Llama-3.3-70B-Instruct meta-llama/Llama-3.1-70B-Instruct
    meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Llama-3.1-8B-Instruct
    meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-3.2-3B-Instruct
    meta-llama/Llama-3.2-1B-Instruct
)
LANGS=(en si)

model=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
lang=${LANGS[$((SLURM_ARRAY_TASK_ID % 2))]}

out=outputs/biography_generation_finetuned/${model##*/}/$lang/rouge_summary.txt

if [ -f "$out" ]; then
    echo "=== $model | $lang already done, skipping ==="
    exit 0
fi

echo "=== $model | $lang ==="
python -m llama_train \
    --model_id "$model" \
    --prompt_lang "$lang" \
    --eval_batch_size 8