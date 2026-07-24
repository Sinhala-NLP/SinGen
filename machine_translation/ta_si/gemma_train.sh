#!/bin/bash
#SBATCH --job-name=gemma_tasi_ft
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=logs/gemma_tasi_ft_%j.out
#SBATCH --error=logs/gemma_tasi_ft_%j.err

set -euo pipefail

mkdir -p logs

# --------------------------------------------------------------------------- #
# Environment
# --------------------------------------------------------------------------- #
source ~/.bashrc
conda activate /storage/hpc/37/ranasint/conda_envs/llm_exp

export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_HUB_CACHE=/scratch/hpc/37/ranasint/hf_cache/hub
export TMPDIR=/scratch/hpc/37/ranasint/tmp
export PIP_CACHE_DIR=/scratch/hpc/37/ranasint/pip_cache
mkdir -p "$HF_HOME" "$TMPDIR" "$PIP_CACHE_DIR"

# Gemma checkpoints are gated: accept the licence on the model pages first.
export HF_TOKEN=

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# --------------------------------------------------------------------------- #
# Run matrix
# --------------------------------------------------------------------------- #
# One consolidated job -- no queue contention between checkpoints.
# Fields: model | train_bs | grad_accum | eval_bs | max_seq_len
# Effective batch size is held at 16 across all sizes so the LR schedule is
# comparable; only the micro-batch changes to stay inside memory.
MODELS=(
  "google/gemma-4-31B-it|1|16|4|768"
  "google/gemma-4-12B-it|2|8|8|768"
  "google/gemma-3-27b-it|1|16|4|768"
  "google/gemma-3-12b-it|2|8|8|768"
)

PROMPT_LANGS=(en si)

TSV_FILE=${TSV_FILE:-ta_si.tsv}
TEST_SIZE=${TEST_SIZE:-1000}
EPOCHS=${EPOCHS:-3}
LR=${LR:-2e-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-200}
EXTRA_ARGS=${EXTRA_ARGS:-}          # e.g. EXTRA_ARGS="--load_in_4bit --push_to_hub"

nvidia-smi
python -c "import torch, transformers, peft; \
print('torch', torch.__version__, '| transformers', transformers.__version__, \
'| peft', peft.__version__, '| gpus', torch.cuda.device_count())"

# --------------------------------------------------------------------------- #
# Main loop
# --------------------------------------------------------------------------- #
for entry in "${MODELS[@]}"; do
  IFS='|' read -r MODEL TRAIN_BS GRAD_ACCUM EVAL_BS MAX_SEQ_LEN <<< "$entry"

  for LANG in "${PROMPT_LANGS[@]}"; do
    echo ""
    echo "==========================================================="
    echo " Ta=>Si fine-tune | $MODEL | prompt_lang=$LANG"
    echo " micro-bs=$TRAIN_BS accum=$GRAD_ACCUM eval-bs=$EVAL_BS seq=$MAX_SEQ_LEN"
    echo " started $(date)"
    echo "==========================================================="

    # `python`, not torchrun: device_map="auto" already shards the model
    # across both H200s. torchrun would spawn one full copy per rank.
    python gemma_train.py \
      --model_name "$MODEL" \
      --prompt_lang "$LANG" \
      --tsv_file "$TSV_FILE" \
      --test_size "$TEST_SIZE" \
      --drop_train_overlap \
      --num_train_epochs "$EPOCHS" \
      --train_batch_size "$TRAIN_BS" \
      --grad_accum "$GRAD_ACCUM" \
      --learning_rate "$LR" \
      --max_seq_len "$MAX_SEQ_LEN" \
      --eval_batch_size "$EVAL_BS" \
      --max_new_tokens "$MAX_NEW_TOKENS" \
      --attn_impl eager \
      --save_adapter \
      $EXTRA_ARGS \
      || { echo "FAILED: $MODEL / $LANG -- continuing with the next run"; continue; }

    echo "Finished $MODEL / $LANG at $(date)"
  done
done

echo ""
echo "All runs complete at $(date)"
echo "Summaries:"
find outputs/tamil_sinhala_translation_finetuned -name bleu_summary.txt | sort