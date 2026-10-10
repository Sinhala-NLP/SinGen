#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --array=0-35
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

# Usage:  sbatch qwen_finetune_biogen.sh
# One array task = one (model, fine-tune language): 18 models x 2 languages = 36.

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    Qwen/Qwen2-0.5B-Instruct Qwen/Qwen2-1.5B-Instruct Qwen/Qwen2-7B-Instruct
    Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct Qwen/Qwen2.5-14B-Instruct Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen2.5-72B-Instruct
    Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B
    Qwen/Qwen3.6-27B Qwen/Qwen3.6-35B-A3B
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
python -m qwen_train \
    --model_id "$model" \
    --prompt_lang "$lang" \
    --eval_batch_size 8