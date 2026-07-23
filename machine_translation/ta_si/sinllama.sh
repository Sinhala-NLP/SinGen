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
# Needs a WRITE-scoped token: it gates both Meta-Llama-3-8B (read) and the
# push to sinhala-nlp (write). A read-only token trains fine and then fails
# at the very last step.
export HF_TOKEN=

# Test set is the last 1000 rows of ta_si.tsv, matching the prompting scripts;
# training uses a 10k random subsample of everything before it.
# max_seq_len is 768 rather than 512: SinLlama's vocabulary extension covers
# Sinhala only, so Tamil source text falls back to the base Llama-3 vocabulary
# where fertility is high. Check the sequence-length line in the log.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | Ta=>Si translation | prompt_lang=${LANG}"
  echo "==================================================================="
  python train_ta_si_translation_sinllama.py \
    --prompt_lang "${LANG}" \
    --tsv_file ta_si.tsv \
    --test_size 1000 \
    --train_size 10000 \
    --num_train_epochs 3 \
    --train_batch_size 4 \
    --grad_accum 4 \
    --learning_rate 2e-4 \
    --max_seq_len 768 \
    --eval_batch_size 8 \
    --max_new_tokens 200 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-TamSiPara-Ta2Si-${LANG}"
done

echo "All SinLlama Ta=>Si translation runs finished."