#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40:1
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --signal=B:USR1@180
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
# guards against the hub-download stall that killed the SinGemma CPT run
export HF_HUB_DOWNLOAD_TIMEOUT=60

# Which model to build: bert, roberta, or electra. Defaults to bert; override
# per submission without editing the file, e.g.
#   sbatch --export=ALL,MODEL_TYPE=roberta pretrain_sinhala_lm.sh
#   sbatch --export=ALL,MODEL_TYPE=electra pretrain_sinhala_lm.sh
#   sbatch --export=ALL,MODEL_TYPE=bert,MODEL_SIZE=large pretrain_sinhala_lm.sh
MODEL_TYPE="${MODEL_TYPE:-bert}"
MODEL_SIZE="${MODEL_SIZE:-base}"     # base or large

# The tokeniser (64k vocab) is identical across sizes, so it is keyed on type
# only; outputs are keyed on type + size so base and large never collide.
TOK_DIR=/scratch/hpc/37/ranasint/sinhala_lms/tok_${MODEL_TYPE}
OUT_DIR=/scratch/hpc/37/ranasint/sinhala_lms/${MODEL_TYPE}_${MODEL_SIZE}

# per-device batch / grad-accum for an L40 (48 GB), seq 512, bf16, gradient
# checkpointing on; effective batch held at 256 across all configs.
case "${MODEL_TYPE}_${MODEL_SIZE}" in
    electra_base)  bs=32; ga=8  ;;
    electra_large) bs=8;  ga=32 ;;
    *_base)        bs=64; ga=4  ;;   # bert / roberta base
    *_large)       bs=16; ga=16 ;;   # bert / roberta large
esac

# WordPiece (bert/electra) packs ~3x fewer tokens per epoch than byte-level BPE
# (roberta), so give them 3 epochs to see a comparable token budget. Warmup is a
# ratio of total steps (set in the .py, default 0.06), so it scales with the run.
case "$MODEL_TYPE" in
    roberta) epochs=1 ;;
    *)       epochs=3 ;;
esac

# large ELECTRA needs its discriminator dims passed explicitly (the .py defaults
# are base); base ELECTRA and all MLM sizes are handled by their own flags.
electra_size_args=()
if [ "$MODEL_SIZE" = "large" ]; then
    electra_size_args=(--num_hidden_layers 24 --disc_hidden_size 1024 \
                       --disc_num_heads 16 --disc_intermediate_size 4096)
fi

# --- self-requeue on wall-time -------------------------------------------
# SLURM sends USR1 180s before the time limit (see --signal above). We then
# submit the next job for this model and forward USR1 to the trainer, which
# writes a fresh checkpoint and stops cleanly -- so the next job resumes with
# ~no lost work. SELF must point at this script; submit from the directory
# holding it and the .py files.
SELF="$SLURM_SUBMIT_DIR/pretrain_sinhala_lm.sh"
TRAIN_PID=""
_requeued=0
requeue() {
    if [ "$_requeued" -eq 0 ]; then
        _requeued=1
        echo "=== wall-time approaching: resubmitting ${MODEL_TYPE}_${MODEL_SIZE} ==="
        sbatch --export=ALL,MODEL_TYPE="$MODEL_TYPE",MODEL_SIZE="$MODEL_SIZE" "$SELF"
    fi
    [ -n "$TRAIN_PID" ] && kill -USR1 "$TRAIN_PID" 2>/dev/null
}
trap requeue USR1
# -------------------------------------------------------------------------

# Train the 64k tokeniser once. tokenizer.json is the skip-if-done sentinel: on
# every resubmission (pretraining needs several to clear the 24h wall-time
# limit) this step is skipped and we go straight to the auto-resuming trainer.
if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
    echo "=== training $MODEL_TYPE tokeniser -> $TOK_DIR ==="
    python train_tokenizer.py --model_type "$MODEL_TYPE" --output_dir "$TOK_DIR"
fi

# Fail fast: if the tokeniser step didn't produce a tokeniser, stop here rather
# than fall through to pretraining, which would fail obscurely trying to load an
# empty dir as a Hub repo id.
if [ ! -f "$TOK_DIR/tokenizer.json" ]; then
    echo "ERROR: no tokenizer.json in $TOK_DIR -- tokeniser step failed; aborting." >&2
    exit 1
fi

# Pretrain. Both trainers resume from the last checkpoint in OUT_DIR if present.
echo "=== pretraining ${MODEL_TYPE}_${MODEL_SIZE} (bs=$bs ga=$ga ep=$epochs) -> $OUT_DIR ==="
if [ "$MODEL_TYPE" = "electra" ]; then
    python pretrain_electra.py \
        --tokenizer_dir "$TOK_DIR" \
        --output_dir "$OUT_DIR" \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size "$bs" \
        --gradient_accumulation_steps "$ga" \
        --num_proc "$SLURM_CPUS_PER_TASK" \
        "${electra_size_args[@]}" &
else
    python pretrain_mlm.py \
        --model_type "$MODEL_TYPE" \
        --model_size "$MODEL_SIZE" \
        --tokenizer_dir "$TOK_DIR" \
        --output_dir "$OUT_DIR" \
        --num_train_epochs "$epochs" \
        --per_device_train_batch_size "$bs" \
        --gradient_accumulation_steps "$ga" \
        --num_proc "$SLURM_CPUS_PER_TASK" &
fi
TRAIN_PID=$!

# `wait` returns as soon as the requeue trap fires, so loop until the trainer
# has actually finished its graceful checkpoint save and exited.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done