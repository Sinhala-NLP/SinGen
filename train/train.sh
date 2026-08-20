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
MODEL_TYPE="${MODEL_TYPE:-bert}"

TOK_DIR=/scratch/hpc/37/ranasint/sinhala_lms/tok_${MODEL_TYPE}
OUT_DIR=/scratch/hpc/37/ranasint/sinhala_lms/${MODEL_TYPE}

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
        echo "=== wall-time approaching: resubmitting $MODEL_TYPE ==="
        sbatch --export=ALL,MODEL_TYPE="$MODEL_TYPE" "$SELF"
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

# Pretrain. Both trainers resume from the last checkpoint in OUT_DIR if present.
echo "=== pretraining $MODEL_TYPE -> $OUT_DIR ==="
if [ "$MODEL_TYPE" = "electra" ]; then
    python pretrain_electra.py \
        --tokenizer_dir "$TOK_DIR" \
        --output_dir "$OUT_DIR" \
        --num_proc "$SLURM_CPUS_PER_TASK" &
else
    python pretrain_mlm.py \
        --model_type "$MODEL_TYPE" \
        --tokenizer_dir "$TOK_DIR" \
        --output_dir "$OUT_DIR" \
        --num_proc "$SLURM_CPUS_PER_TASK" &
fi
TRAIN_PID=$!

# `wait` returns as soon as the requeue trap fires, so loop until the trainer
# has actually finished its graceful checkpoint save and exited.
while kill -0 "$TRAIN_PID" 2>/dev/null; do
    wait "$TRAIN_PID"
done