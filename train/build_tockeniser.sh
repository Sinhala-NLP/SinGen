#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --job-name=build_si_tok
#SBATCH --output=build_si_tok_%j.out

# --------------------------------------------------------------------------- #
# CPU-only prerequisite: build the Sinhala-extended gemma-4 tokenizer.
# Run this FIRST, then read the fertility number in the log to decide whether
# the extension is worth it. Once it finishes, run_cpt_gemma4_extended.sh will
# find the tokenizer and skip its own build step.
#
# No GPU needed -- SentencePiece training is CPU (multi-threaded, hence the 16
# cores). Confirm 'serial' is your cluster's CPU queue and allows this many
# cpus-per-task; if it's capped, switch -p to 'parallel'.
# --------------------------------------------------------------------------- #

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

export HF_HOME=/scratch/hpc/37/ranasint/hf
export HF_DATASETS_CACHE=/scratch/hpc/37/ranasint/hf/datasets
export PIP_CACHE_DIR=/scratch/hpc/37/ranasint/pip
export TMPDIR=/scratch/hpc/37/ranasint/tmp
mkdir -p "$TMPDIR"

# gated model: the tokenizer download needs the licence-accepted token too
export HF_TOKEN=

export HF_DATASETS_STREAMING=1
pip install sentencepiece

TOKENIZER_DIR=/scratch/hpc/37/ranasint/gemma4_si_tokenizer

# needs sentencepiece in the env: pip install sentencepiece --break-system-packages
python -c "import sentencepiece" 2>/dev/null || {
    echo "ERROR: sentencepiece not installed in this env."; exit 1; }

if [ -f "$TOKENIZER_DIR/new_tokens.txt" ]; then
    echo "Tokenizer already exists at $TOKENIZER_DIR -- nothing to do."
    exit 0
fi

# build into a temp dir + atomic mv so an interrupted build can't leave a
# half-written dir that later jobs mistake for a finished tokenizer.
rm -rf "${TOKENIZER_DIR}.tmp"
python build_extended_tokenizer.py \
    --model google/gemma-4-31B-it \
    --dataset sinhala-nlp/sinhala-7m-corpus \
    --out "${TOKENIZER_DIR}.tmp" \
    --train_sample 300000 \
    --sp_vocab 32000 \
    --new_tokens 15000 || { echo "tokenizer build failed"; exit 1; }
mv "${TOKENIZER_DIR}.tmp" "$TOKENIZER_DIR"

echo "============================================================"
echo "Tokenizer built at $TOKENIZER_DIR"
echo ">>> Check the 'Fertility reduction' line above BEFORE submitting"
echo "    the GPU run. If it's small (<10%), reconsider the extension."
echo "============================================================"