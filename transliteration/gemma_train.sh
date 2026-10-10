#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:2
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

# Usage:  sbatch gemma_finetune_translit.sh
# One array task = one (model, fine-tune language): 5 models x 2 languages = 10.

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

MODELS=(
    google/gemma-4-31B-it google/gemma-4-12B-it
    google/gemma-3-27b-it google/gemma-3-12b-it google/gemma-3-4b-it
)
LANGS=(en si)

model=${MODELS[$((SLURM_ARRAY_TASK_ID / 2))]}
lang=${LANGS[$((SLURM_ARRAY_TASK_ID % 2))]}

out=outputs/transliteration_finetuned/${model##*/}/$lang/translit_summary.txt

if [ -f "$out" ]; then
    echo "=== $model | $lang already done, skipping ==="
    exit 0
fi

echo "=== $model | $lang ==="
python -m gemma_train \
    --model_id "$model" \
    --prompt_lang "$lang" \
    --eval_batch_size 32