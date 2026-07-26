#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:4
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
export PYTHONIOENCODING=utf-8

# FLORES-200 is fetched as a tarball straight from dl.fbaipublicfiles.com
# (datasets>=4.0 no longer runs the HF loading script). Put it on scratch so the
# eight runs below share one download instead of re-fetching per run.
export FLORES200_CACHE=/scratch/hpc/37/ranasint/flores200

# Needs a WRITE-scoped token: it gates the Gemma checkpoints (read) and the
# push to sinhala-nlp (write). A read-only token trains fine and then fails
# at the very last step -- check_hub_token() in the script verifies the scope
# before the model is loaded.
export HF_TOKEN=

# No TSV: FLORES-200 ships dev (997) / devtest (1012) and no train split, so
# training runs on dev and evaluation on devtest. The splits are
# document-disjoint, so there is no sibling-sentence leakage -- but 997 pairs is
# a very thin supervised signal (~186 optimizer steps at 3 epochs). To train a
# real supervised baseline, point --train_tsv at a larger En-Si parallel corpus
# with columns english/sinhala and add --drop_train_overlap: mined corpora
# (CCMatrix, NLLB, OPUS bundles) routinely contain FLORES sentences.
# TRAIN_TSV=en_si_parallel.tsv

# Effective batch is held at 16 everywhere so one learning rate is valid
# across all four sizes; only the micro-batch changes to fit memory.
for MODEL in google/gemma-4-31B-it google/gemma-4-12B-it google/gemma-3-27b-it google/gemma-3-12b-it; do

  TAG=$(basename "$MODEL")
  case "$MODEL" in
    *31B*|*27b*) TRAIN_BS=2; GRAD_ACCUM=8;  EVAL_BS=8  ;;
    *)           TRAIN_BS=4; GRAD_ACCUM=4;  EVAL_BS=16 ;;
  esac

  for LANG in en si; do
    echo "==================================================================="
    echo " ${TAG} | En=>Si translation | prompt_lang=${LANG}"
    echo "==================================================================="
    python gemma_train.py \
      --model_name "${MODEL}" \
      --prompt_lang "${LANG}" \
      --num_train_epochs 3 \
      --train_batch_size "${TRAIN_BS}" \
      --grad_accum "${GRAD_ACCUM}" \
      --learning_rate 2e-4 \
      --max_seq_len 512 \
      --eval_batch_size "${EVAL_BS}" \
      --max_new_tokens 200 \
      --push_to_hub \
      --hub_repo "sinhala-nlp/${TAG}-FLORES200-En2Si-${LANG}"
  done
done

echo "All Gemma En=>Si translation runs finished."