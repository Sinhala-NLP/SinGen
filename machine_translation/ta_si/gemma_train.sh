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

# Needs a WRITE-scoped token: it gates the Gemma checkpoints (read) and the
# push to sinhala-nlp (write). A read-only token trains fine and then fails
# at the very last step -- the script checks this before loading the model.
export HF_TOKEN=

# TamSiPara TSV with columns Tamil / Sinhala. Test is the last 1000 rows,
# matching the prompting scripts; everything before is the training pool.
TSV_FILE=ta_si.tsv

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
    echo " ${TAG} | Ta=>Si translation | prompt_lang=${LANG}"
    echo "==================================================================="
    python train_translation_gemma.py \
      --model_name "${MODEL}" \
      --prompt_lang "${LANG}" \
      --tsv_file "${TSV_FILE}" \
      --test_size 1000 \
      --drop_train_overlap \
      --num_train_epochs 3 \
      --train_batch_size "${TRAIN_BS}" \
      --grad_accum "${GRAD_ACCUM}" \
      --learning_rate 2e-4 \
      --max_seq_len 768 \
      --eval_batch_size "${EVAL_BS}" \
      --max_new_tokens 200 \
      --push_to_hub \
      --hub_repo "sinhala-nlp/${TAG}-TamSiPara-Ta2Si-${LANG}"
  done
done

echo "All Gemma Ta=>Si translation runs finished."