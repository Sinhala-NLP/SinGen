#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
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
# Needs a WRITE-scoped token: it gates both Meta-Llama-3-8B (read) and the
# push to sinhala-nlp (write). A read-only token trains fine and then fails
# at the very last step, after ~a day of compute.
export HF_TOKEN=

# Headline articles are far longer than the SiTSE sentences, so max_seq_len
# goes 1024 -> 1536 and the per-device batch drops 4 -> 2 with grad_accum
# 4 -> 8, keeping the effective batch at 16 as in the simplification recipe.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | headline generation | prompt_lang=${LANG}"
  echo "==================================================================="
  python train_headline_sinllama.py \
    --prompt_lang "${LANG}" \
    --num_train_epochs 3 \
    --train_batch_size 2 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --max_seq_len 1536 \
    --eval_batch_size 4 \
    --max_new_tokens 128 \
    --test_size 1000 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-NSINA-Headlines-${LANG}"
done

echo "All SinLlama headline-generation runs finished."