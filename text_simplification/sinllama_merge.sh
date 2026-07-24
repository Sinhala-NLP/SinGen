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

MODELS=/scratch/hpc/37/ranasint/models

# slerp-a03: the merge method | cv-baseline: Chat Vector | SinLlama_v01-merged: no-merge control
for run in "slerp-a03:${MODELS}/SinLlama-slerp-a03" \
           "cv-baseline:${MODELS}/SinLlama-cv-baseline" \
           "sinllama-nomerge:${MODELS}/SinLlama_v01-merged"; do
  tag="${run%%:*}"
  path="${run#*:}"
  for LANG in en si; do
    echo "==================================================================="
    echo " ${tag} | text simplification | prompt_lang=${LANG}"
    echo "==================================================================="
    python sinllama_merge.py \
      --model_path "${path}" \
      --run_tag "${tag}" \
      --prompt_lang "${LANG}" \
      --num_train_epochs 3 \
      --train_batch_size 4 \
      --grad_accum 4 \
      --learning_rate 2e-4 \
      --max_seq_len 1024 \
      --eval_batch_size 8 \
      --max_new_tokens 256
  done
done

echo "All merged-checkpoint text-simplification runs finished."