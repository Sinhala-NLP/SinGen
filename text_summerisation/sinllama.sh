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

# Summarisation has the longest inputs in SinGen, so max_seq_len is 2048 (vs
# 1536 for headlines, 768 for Ta=>Si) and the per-device batch drops to 2 with
# grad_accum 8, keeping the effective batch at 16. With max_new_tokens 512 the
# prompt budget is 1536 tokens; check the sequence-length and truncation lines
# in the log and raise --max_seq_len if too many articles are being cut.
# Drop to --train_batch_size 1 --grad_accum 16 if this OOMs.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | text summarisation | prompt_lang=${LANG}"
  echo "==================================================================="
  python -m sinllama \
    --prompt_lang "${LANG}" \
    --num_train_epochs 3 \
    --train_batch_size 2 \
    --grad_accum 8 \
    --learning_rate 2e-4 \
    --max_seq_len 2048 \
    --eval_batch_size 4 \
    --max_new_tokens 512 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-XLSum-${LANG}"
done

echo "All SinLlama summarisation runs finished."