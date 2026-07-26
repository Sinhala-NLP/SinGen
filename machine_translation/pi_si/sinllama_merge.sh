#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
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
  for qt in zero-shot zero-shot-si few-shot few-shot-si; do
    echo "==================================================================="
    echo " ${tag} | Pali->Si translation (tail-1000) | query_type=${qt}"
    echo "==================================================================="
    python -m sinllama_merge \
      --model_id "${path}" \
      --run_tag "${tag}" \
      --query_type "${qt}" \
      --batch_size 8 \
      --max_new_tokens 200 \
      --test_size 1000
  done
done

echo "All merged-checkpoint zero/few-shot Pali->Si translation runs finished."