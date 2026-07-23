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
# Needs a WRITE-scoped token: it gates both Meta-Llama-3-8B (read) and the
# push to sinhala-nlp (write). A read-only token trains fine and then fails
# at the very last step.
export HF_TOKEN=

# FLORES-200 ships no train split, so this fine-tunes on `dev` (997 pairs) and
# evaluates on `devtest` (1012) -- disjoint, but ~1k pairs is a very small
# supervised MT signal and gives only ~187 optimizer steps at 3 epochs.
# To train a real supervised baseline, point --train_tsv at a larger En-Si
# parallel corpus with columns english/sinhala; --train_size then subsamples it.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | En=>Si translation | prompt_lang=${LANG}"
  echo "==================================================================="
  python train_translation_sinllama.py \
    --prompt_lang "${LANG}" \
    --num_train_epochs 3 \
    --train_batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --max_seq_len 512 \
    --train_size 10000 \
    --eval_batch_size 8 \
    --max_new_tokens 200 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-FLORES200-En2Si-${LANG}"
done

echo "All SinLlama En=>Si translation runs finished."