#!/bin/bash
#SBATCH -p gpu-long
#SBATCH --nodes=1
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=gemma4_cpt_si_ext
#SBATCH --output=cpt_gemma4_ext_%j.out

# --------------------------------------------------------------------------- #
# Tokenizer-EXTENSION continual pretraining of gemma-4-31B-it on
# sinhala-7m-corpus (bf16 LoRA + trained new embedding rows).
#
# PREREQUISITES (run these BEFORE submitting):
#   1. build_extended_tokenizer.py  -> produces $TOKENIZER_DIR (check fertility!)
#   2. a 50-step SMOKE TEST of this exact script (set --max_steps 50) to confirm
#      the embed-scan is clean and reloaded generation isn't garbled.
#      Only after that passes should you submit the full 7-day run below.
#
# Single node, 2x H200 assumed (matches your device_map="auto" setup). If your
# H200s are on two separate nodes, this launcher needs --nodes=2 + srun/torchrun
# multi-node rendezvous instead.
# --------------------------------------------------------------------------- #

source /etc/profile
module add anaconda3/2023.09          # required: puts conda's `activate` on PATH
module add cuda/12.0                  # match the CUDA your llm_exp PyTorch was built against
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

export HF_HOME=/scratch/hpc/37/ranasint/hf
export HF_DATASETS_CACHE=/scratch/hpc/37/ranasint/hf/datasets
export PIP_CACHE_DIR=/scratch/hpc/37/ranasint/pip
export TMPDIR=/scratch/hpc/37/ranasint/tmp
mkdir -p "$TMPDIR"

# gated model: set your token (or `huggingface-cli login`) before submitting
export HF_TOKEN=

export HF_DATASETS_STREAMING=1

TOKENIZER_DIR=/scratch/hpc/37/ranasint/gemma4_si_tokenizer
OUTPUT_DIR=/scratch/hpc/37/ranasint/gemma4_cpt_si_ext
mkdir -p "$OUTPUT_DIR"

# --- build the extended tokenizer if it doesn't exist yet ------------------ #
# Guarded so it runs ONCE. On a requeue/resume the tokenizer already exists and
# is reused, so training always resizes to the SAME vocab and existing
# checkpoints stay valid -- rebuilding mid-run would silently corrupt resume.
# Built into a temp dir + atomic mv so a killed build can't leave a half-written
# tokenizer that passes the existence check. Runs with plain `python` (single
# process) -- must NOT run under torchrun, which would race two SP trainings.
if [ ! -f "$TOKENIZER_DIR/new_tokens.txt" ]; then
    echo "[prereq] extended tokenizer not found -- building it now (CPU)..."
    rm -rf "${TOKENIZER_DIR}.tmp"
    python build_extended_tokenizer.py \
        --model google/gemma-4-31B-it \
        --dataset sinhala-nlp/sinhala-7m-corpus \
        --out "${TOKENIZER_DIR}.tmp" \
        --train_sample 300000 \
        --new_tokens 15000 || { echo "tokenizer build failed"; exit 1; }
    mv "${TOKENIZER_DIR}.tmp" "$TOKENIZER_DIR"
    echo "[prereq] tokenizer built at $TOKENIZER_DIR"
else
    echo "[prereq] reusing existing tokenizer at $TOKENIZER_DIR"
fi

# effective batch = per_device(2) * gpus(2) * grad_accum(8) = 32 seqs
#   -> 32 * 2048 = 65,536 tokens/step; 30000 steps ~= 2.0B tokens.
# NOTE: with the extended tokenizer, fewer tokens/doc means each step covers
# more DOCUMENTS -- set max_steps from your post-extension token estimate.
#
# For a smoke test, change to:  --max_steps 50 --save_steps 50
#
# Add --freeze_old_rows to train ONLY the new Sinhala embedding rows.
torchrun --standalone --nproc_per_node=2 train_gemma.py \
    --model google/gemma-4-31B-it \
    --tokenizer_dir "$TOKENIZER_DIR" \
    --dataset sinhala-nlp/sinhala-7m-corpus \
    --output_dir "$OUTPUT_DIR" \
    --block_size 2048 \
    --per_device_batch 2 \
    --grad_accum 8 \
    --max_steps 30000 \
    --lr 1e-4 \
    --warmup_steps 500 \
    --save_steps 500 \
    --lora_r 32 \
    --lora_alpha 64