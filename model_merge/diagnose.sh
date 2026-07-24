#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --job-name=sinllama_diag
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#
# One L40S is enough: an 8B model in bf16 is ~16.5 GB and only one is resident
# at a time. Unlike the merge job, this one actually uses the GPU.

# ---------------------------------------------------------------------------
# Environment. No `set -u` until after /etc/profile and conda have run --
# lang.sh reads $LC_ALL and conda's activate touches $PS1, both unset.
# ---------------------------------------------------------------------------
source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

set -euo pipefail

export LC_ALL=${LC_ALL:-en_US.UTF-8}
export LANG=${LANG:-en_US.UTF-8}
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT=diagnose.py
MODELS=/scratch/hpc/37/ranasint/models
OUT=/scratch/hpc/37/ranasint/singen/merge_diagnostics.csv
mkdir -p "$(dirname "${OUT}")"

# Optional: score on FLORES-200 Sinhala devtest instead of the eight built-in
# sentences. Eight is enough to RANK checkpoints; it is not enough for a table
# in the paper. Extract one sentence per line to this path.
SI_FILE=/scratch/hpc/37/ranasint/data/flores200_devtest.sin_Sinh.txt
EN_FILE=/scratch/hpc/37/ranasint/data/flores200_devtest.eng_Latn.txt

TEXT_ARGS=()
if [ -r "${SI_FILE}" ]; then
    TEXT_ARGS+=(--sinhala_file "${SI_FILE}")
    echo "Using Sinhala text: ${SI_FILE} ($(wc -l < "${SI_FILE}") lines)"
else
    echo "NOTE: ${SI_FILE} not found -- falling back to the 8 built-in Sinhala"
    echo "      sentences. Fine for ranking, not for the paper."
fi
if [ -r "${EN_FILE}" ]; then
    TEXT_ARGS+=(--english_file "${EN_FILE}")
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------------------
# Run
#
# The two --reference models are the anchors that make the table readable:
#
#   SinLlama_v01-merged            Sinhala ceiling. Nothing should beat it by
#                                  much on bpc_si; a merge that is far worse
#                                  has traded away the adaptation.
#   Meta-Llama-3-8B-Instruct       English and instruction-following ceiling.
#                                  Compare bpc_en against it to quantify
#                                  catastrophic forgetting.
#
# A merge is working if bpc_si sits near SinLlama, bpc_en near Instruct, and
# stops_si is 1. Note that slerp-2x2ls leaves 253 of 289 tensors at pure
# Instruct, so if it "wins" on anything, that is the trivial result of barely
# merging rather than evidence for the method.
# ---------------------------------------------------------------------------
python "${SCRIPT}" \
    --models "${MODELS}/SinLlama-*" \
    --reference "${MODELS}/SinLlama_v01-merged" \
                meta-llama/Meta-Llama-3-8B-Instruct \
    --dtype bfloat16 \
    --device cuda \
    --max_new_tokens 96 \
    --out "${OUT}" \
    ${TEXT_ARGS[@]+"${TEXT_ARGS[@]}"}

echo
echo "==================================================================="
echo " Diagnostics written to ${OUT}"
echo "==================================================================="
column -s, -t < "${OUT}" | cut -c1-160 || cat "${OUT}"