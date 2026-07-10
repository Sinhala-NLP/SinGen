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

# Keep pip/tmp off the home quota (consistent with the other jobs)
export PIP_CACHE_DIR=/scratch/hpc/37/ranasint/pip_cache
export TMPDIR=/scratch/hpc/37/ranasint/tmp

# Each run: LoRA instruction-fine-tune on the ~800 SiTSE train instances, then
# evaluate on the 200 held-out test instances. Loop over model sizes x prompt
# language (English / Sinhala instruction). device_map="auto" spreads the large
# checkpoints over both H200s, so launch with plain `python` (no torchrun).
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do
    for lang in en si; do
        python -m train_simplification_qwen \
            --model_id "$model" \
            --prompt_lang "$lang" \
            --num_train_epochs 3 \
            --train_batch_size 4 \
            --grad_accum 4 \
            --learning_rate 2e-4 \
            --eval_batch_size 8 \
            --max_new_tokens 512
    done
done