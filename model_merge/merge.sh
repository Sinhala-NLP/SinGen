#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --job-name=sinllama_merge
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#
# The GPU is requested only because the astro partition requires a GRES.
# Merging is pure CPU tensor arithmetic and this script never calls .cuda(),
# so the card will sit idle. Drop the --gres line if the partition allows it.

# ---------------------------------------------------------------------------
# Environment
#
# IMPORTANT: no `set -u` until after /etc/profile and conda have run.
# /etc/profile.d/lang.sh reads $LC_ALL to decide whether to set a locale, and
# conda's activate touches $PS1 -- under `set -u` an unset variable is fatal,
# so strict mode here kills the job before it starts.
# ---------------------------------------------------------------------------
source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

# Safe from here on.
set -euo pipefail

# A C/POSIX locale can make Python's default encoding ASCII, which turns any
# echo of a Sinhala string into a UnicodeEncodeError hours into the job.
export LC_ALL=${LC_ALL:-en_US.UTF-8}
export LANG=${LANG:-en_US.UTF-8}

export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export TMPDIR=/scratch/hpc/37/ranasint/tmp
export HF_TOKEN=

# The merge is embarrassingly parallel across each tensor's elements and torch
# will happily use every core we were allocated.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT=/storage/hpc/37/ranasint/singen/merge_sinllama.py
MODELS=/scratch/hpc/37/ranasint/models
MERGED_TARGET=${MODELS}/SinLlama_v01-merged      # PEFT-merged SinLlama, cached
mkdir -p "${MODELS}"

BASE=meta-llama/Meta-Llama-3-8B
CHAT=meta-llama/Meta-Llama-3-8B-Instruct
ADAPTER=polyglots/SinLlama_v01

# Set PUSH=1 (sbatch --export=ALL,PUSH=1) to upload each merged model.
PUSH=${PUSH:-0}
HUB_ORG=sinhala-nlp

# Every output is a full bf16 8B checkpoint with the 139,336-token vocabulary,
# i.e. ~16.5 GB each. The sweep writes 10 of them plus the cached target:
# budget ~180 GB of scratch before launching.
echo "Free space on ${MODELS}:"
df -h "${MODELS}" | tail -1

# ---------------------------------------------------------------------------
# Step 0: materialise SinLlama once
#
# Base Llama-3-8B -> resize embeddings to 139,336 -> attach adapter ->
# merge_and_unload. Every later merge reads this instead of redoing the PEFT
# merge. The log line to watch is the new-token embedding std: if it warns the
# rows look untrained, the adapter did not carry embed_tokens/lm_head and
# nothing downstream is valid -- stop there.
# ---------------------------------------------------------------------------
if [ ! -f "${MERGED_TARGET}/config.json" ]; then
    echo "==================================================================="
    echo " Step 0 | materialising ${ADAPTER} onto ${BASE}"
    echo "==================================================================="
    python "${SCRIPT}" \
        --method slerp --alpha 0.0 \
        --target "${ADAPTER}" \
        --base "${BASE}" \
        --chat "${CHAT}" \
        --save_merged_target "${MERGED_TARGET}" \
        --no_copy_special_tokens --extra_copy_ids \
        --no_copy_chat_template \
        --out "${MODELS}/_scratch_alpha0" \
        --smoke_test
    rm -rf "${MODELS}/_scratch_alpha0"
else
    echo "Reusing cached merged target at ${MERGED_TARGET}"
fi

# ---------------------------------------------------------------------------
# Step 1: the sweep
#
# Entries are "<run_name>|<extra args>". Runs whose output directory already
# has a config.json are skipped, so the job is safe to requeue after a timeout.
#
#  * cv-baseline -- dare_linear at density 1.0 is exactly Chat Vector:
#                   theta_pre + tau_sin + tau_chat, no pruning. This is the
#                   number the other methods have to beat, since you have the
#                   base model and CV is the natural method for that setting.
#  * slerp   -- ElChat's Merge step verbatim; alpha weights the chat model.
#  * ties    -- density 0.2 is the TIES paper's k=20%. Note SinLlama's deltas
#               are low-rank LoRA products, which spread small values densely,
#               so aggressive trimming may cut signal rather than noise. If
#               d02 loses to d05, that is why.
#  * dare_*  -- density is the KEEP rate, so 0.5 means drop rate p=0.5.
#  * 2x2ls   -- ablation reproducing the released ElChat pipeline, which only
#               touches layers 0,1,L-2,L-1. SinLlama's LoRA hit all 32 layers,
#               so this should lose to --layers all.
# ---------------------------------------------------------------------------
RUNS=(
  "cv-baseline|--method dare_linear --density 1.0 --lam 1.0"
  "slerp-a02|--method slerp --alpha 0.2"
  "slerp-a03|--method slerp --alpha 0.3"
  "slerp-a05|--method slerp --alpha 0.5"
  "linear-a03|--method linear --alpha 0.3"
  "ties-d02|--method ties --density 0.2 --lam 1.0"
  "ties-d05|--method ties --density 0.5 --lam 1.0"
  "dareties-d05|--method dare_ties --density 0.5 --lam 1.0 --seed 42"
  "dareties-d01|--method dare_ties --density 0.1 --lam 1.0 --seed 42"
  "slerp-2x2ls|--method slerp --layers 2x2ls"
)

for entry in "${RUNS[@]}"; do
    name="${entry%%|*}"
    extra="${entry#*|}"
    out="${MODELS}/SinLlama-${name}"

    if [ -f "${out}/config.json" ]; then
        echo "-- skipping ${name}: ${out} already exists"
        continue
    fi

    echo "==================================================================="
    echo " ${name} | ${extra}"
    echo "==================================================================="
    start=$SECONDS

    # shellcheck disable=SC2086
    python "${SCRIPT}" \
        --target "${MERGED_TARGET}" \
        --base "${BASE}" \
        --chat "${CHAT}" \
        --dtype bfloat16 \
        --out "${out}" \
        --smoke_test \
        ${extra}

    echo "-- ${name} done in $((SECONDS - start))s"

    if [ "${PUSH}" = "1" ]; then
        huggingface-cli upload "${HUB_ORG}/SinLlama-${name}" "${out}" . \
            --repo-type model --private
    fi
done

echo "All SinLlama merge runs finished."