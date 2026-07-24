#!/bin/bash
#SBATCH -p astro
#SBATCH --job-name=sinllama_merge
#SBATCH --mem=140G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
# No --gres here on purpose: merging is pure CPU tensor arithmetic. The script
# never calls .cuda(), so a GPU would sit idle for the whole 24h. If the astro
# partition refuses jobs without a GRES, uncomment the next line.
##SBATCH --gres=gpu:nvidia_l40s:1

set -euo pipefail

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export TMPDIR=/scratch/hpc/37/ranasint/tmp
# Meta-Llama-3-8B and -Instruct are gated: a READ-scoped token is enough unless
# PUSH=1 below, in which case it needs WRITE for the sinhala-nlp org.
export HF_TOKEN=

# The merge is embarrassingly parallel across the tensor's elements and torch
# will happily use every core we were allocated.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-32}

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT=sinllama_merge.py
MODELS=/scratch/hpc/37/ranasint/models
MERGED_TARGET=${MODELS}/SinLlama_v01-merged      # PEFT-merged SinLlama, cached
mkdir -p "${MODELS}"

BASE=meta-llama/Meta-Llama-3-8B
CHAT=meta-llama/Meta-Llama-3-8B-Instruct
ADAPTER=polyglots/SinLlama_v01

# Set PUSH=1 to upload each merged model to the hub after it is written.
PUSH=${PUSH:-0}
HUB_ORG=sinhala-nlp

# Every output is a full bf16 8B checkpoint with the 139,336-token vocabulary,
# i.e. ~16.5 GB each. The sweep below writes 9 of them plus the cached target:
# budget ~165 GB of scratch before launching.
echo "Free space on ${MODELS}:"
df -h "${MODELS}" | tail -1

# ---------------------------------------------------------------------------
# Step 0: materialise SinLlama once
#
# Base Llama-3-8B -> resize embeddings to 139,336 -> attach adapter ->
# merge_and_unload. Every later merge reads this instead of redoing the PEFT
# merge, which saves ~10 min a run. The log line to watch for is the
# new-token embedding std: if it warns that the rows look untrained, the
# adapter did not carry embed_tokens/lm_head and nothing downstream is valid.
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
# Each entry is "<run_name>|<extra args>". Runs whose output directory already
# has a config.json are skipped, so the job is safe to requeue after a timeout.
#
#  * slerp  -- ElChat's Merge step verbatim. alpha is the weight on the chat
#              model; 0.3 is what the paper used for its outermost layers.
#  * ties   -- density 0.2 is the TIES paper's k=20%. lambda 1.0.
#  * dare_* -- density is the KEEP rate, so 0.5 means drop rate p=0.5. DARE
#              tolerates larger p on larger models but degrades once delta
#              magnitudes get big, and continual-pretraining deltas are bigger
#              than the SFT deltas Yu et al. tested. 0.1 is the stress case.
#  * 2x2ls  -- ablation reproducing the released ElChat pipeline, which only
#              touches layers 0,1,L-2,L-1. SinLlama's LoRA hit all 32 layers,
#              so this should underperform --layers all. Worth a table row.
# ---------------------------------------------------------------------------
RUNS=(
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