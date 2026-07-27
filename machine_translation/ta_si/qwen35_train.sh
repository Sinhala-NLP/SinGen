#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:3
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --requeue
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Needs a WRITE-scoped token for the push to sinhala-nlp.
export HF_TOKEN=
if [ -z "$HF_TOKEN" ]; then
    echo "ERROR: HF_TOKEN is empty but every run below passes --push_to_hub." >&2
    exit 1
fi

# The TSV path is relative, so this job must be submitted from the directory
# holding ta_si.tsv. Checked here rather than after a multi-GB model download.
TSV=ta_si.tsv
if [ ! -f "$TSV" ]; then
    echo "ERROR: $TSV not found in $(pwd) -- submit from the directory containing it." >&2
    exit 1
fi

# ---------------------------------------------------------------------------
# Tamil => Sinhala translation, LoRA instruction fine-tuning on TamSiPara.
#
# The test set is the trailing 1000 rows, identical to the prompting scripts.
# Training uses the rest of the corpus.
#
# CROSS-FAMILY COMPARABILITY: these two values must match the existing Gemma
# Ta->Si fine-tuning script (train_translation_gemma.py). The Gemma runs came
# first, so they are the fixed point -- change these to whatever it used before
# submitting, not afterwards.
EPOCHS=1
LR=2e-4

# Effective batch size is held at 16 for every checkpoint (tbs x ga), so only
# the memory split changes with model size, not the optimisation.
#
# --max_seq_len 1024 rather than the En->Si 512: Tamil and Sinhala are both
# non-Latin scripts at 3 UTF-8 bytes per character, so a sentence pair costs
# roughly twice the tokens. Watch the "Prompt-truncated examples" line -- it
# should read 0 for both training and eval.
#
# WARNING on hardware: 2 x L40S is 96 GB total. The 27B (~54 GB of bf16 weights)
# is workable but slow over PCIe; the 35B-A3B (~70 GB) may OOM. If `astro` can
# give you the H200 NVL pair, use it.
# ---------------------------------------------------------------------------
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) tbs=1; ga=16; ebs=2 ;;
        *9B*)        tbs=1; ga=16; ebs=4 ;;
        *4B*)        tbs=2; ga=8;  ebs=4 ;;
        *)           tbs=4; ga=4;  ebs=8 ;;
    esac

    for LANG in en si; do
        tag=$(basename "$model")
        summary="outputs/tamil_sinhala_translation_finetuned/${tag}/${LANG}/bleu_summary.txt"
        if [ -f "$summary" ]; then
            echo "skip ${tag}/${LANG} (already complete)"
            continue
        fi

        echo "==================================================================="
        echo " ${model} | Ta=>Si translation | prompt_lang=${LANG}"
        echo "==================================================================="
        python -m qwen35_ta_si_train \
            --model_id "$model" \
            --prompt_lang "${LANG}" \
            --tsv_file "$TSV" \
            --test_size 1000 \
            --num_train_epochs "$EPOCHS" \
            --train_batch_size "$tbs" \
            --grad_accum "$ga" \
            --learning_rate "$LR" \
            --max_seq_len 1024 \
            --eval_batch_size "$ebs" \
            --max_new_tokens 200 \
            --push_to_hub \
            --hub_repo "sinhala-nlp/${tag}-TamSiPara-Ta2Si-${LANG}"
    done
done

echo "All Qwen Ta=>Si translation fine-tuning runs finished."