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

for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | text simplification | prompt_lang=${LANG}"
  echo "==================================================================="
  python train_simplification_sinllama.py \
    --prompt_lang "${LANG}" \
    --num_train_epochs 3 \
    --train_batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --max_seq_len 1024 \
    --eval_batch_size 8 \
    --max_new_tokens 256
done

echo "All SinLlama text-simplification runs finished."


