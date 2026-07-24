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

# Settings derived from the measured length distribution of this corpus:
#
#   max_seq_len 8192   - the Llama-3 context ceiling. Skips ~0.2% of the train
#                        pool (targets that alone exceed the budget) and
#                        middle-truncates the prompt on ~0.7%.
#   max_new_tokens 3000 - covers p99 of the test-tail references (2980 tokens).
#                        At the old 200, 65% of references were cut off and the
#                        BLEU brevity penalty dominated the score.
#   batch 1 x accum 16  - at 8192 tokens the 139k-row softmax makes logits the
#                        dominant memory term; batch 1 also means zero padding.
#
# Eval is the slow part: 1000 instances at up to 3000 new tokens each. Prompts
# are length-sorted before batching to limit wasted decode steps, but budget
# several hours per language. Run one language per job if the 48h window looks
# tight.
#
# NOTE: the tail(1000) test set is far longer than the training body (median
# 368 vs 47 target tokens) -- it is not distributionally matched. This is
# recorded in bleu_summary.txt and belongs in the paper.
for LANG in en si; do
  echo "==================================================================="
  echo " SinLlama_v01 | Pi=>Si translation | prompt_lang=${LANG}"
  echo "==================================================================="
  python train_pali_translation_sinllama.py \
    --prompt_lang "${LANG}" \
    --test_size 1000 \
    --num_train_epochs 3 \
    --train_batch_size 1 \
    --grad_accum 16 \
    --learning_rate 2e-4 \
    --max_seq_len 8192 \
    --min_prompt_tokens 128 \
    --eval_batch_size 4 \
    --max_new_tokens 3000 \
    --push_to_hub \
    --hub_repo "sinhala-nlp/SinLlama-PaliSinhala-Pi2Si-${LANG}"
done

echo "All SinLlama Pi=>Si translation runs finished."