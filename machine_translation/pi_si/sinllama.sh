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

# Test set is the last 1000 rows, matching the prompting script; training uses
# every pair before it. Leading verse numbers are stripped from both sides.
#
# The log prints an exact train/test overlap count before training. The Pali
# canon repeats stock passages verbatim across suttas, so check that number
# before trusting the BLEU: add --drop_train_overlap to remove exact matches.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | Pi=>Si translation | prompt_lang=${LANG}"
  echo "==================================================================="
  python -m sinllama \
    --prompt_lang "${LANG}" \
    --test_size 1000 \
    --num_train_epochs 3 \
    --train_batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --max_seq_len 768 \
    --eval_batch_size 8 \
    --max_new_tokens 200 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-PaliSinhala-Pi2Si-${LANG}"
done

echo "All SinLlama Pi=>Si translation runs finished."