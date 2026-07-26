#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40:4
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

# ---------------------------------------------------------------------------
# NSINA-Headlines headline generation, LoRA instruction fine-tuning.
#
# Unlike the FLORES En=>Si runs, the input here is a full news article: ~2500
# characters of Sinhala is well over a thousand tokens on the Qwen BPE, so
# --max_seq_len 2560 (not 512) and the article is budgeted from the LEAD, since
# news is inverted-pyramid. Watch the "Article-truncated examples" percentage in
# the log -- if it is high for a given tokenizer, raise --max_seq_len.
#
# NSINA has plenty of training data, so this is 1 epoch over an 8k subsample
# rather than 3 epochs over everything (500 optimizer steps at bs 1 x ga 16).
# Fine-tuned models are trained AND evaluated zero-shot, so the comparable
# prompting rows are `zero-shot` (en) and `zero-shot-si` (si).
#
# WARNING on hardware: 2 x L40S is 96 GB total. At 2560-token sequences the 27B
# (~54 GB of bf16 weights) is tight and the 35B-A3B (~70 GB) will very likely
# OOM, with no NVLink to hide the pipeline cost. If `astro` can give you the
# H200 NVL pair, use it; otherwise expect to drop the last two checkpoints.
# ---------------------------------------------------------------------------
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) tbs=1; ga=16; ebs=1 ;;
        *9B*)        tbs=1; ga=16; ebs=2 ;;
        *4B*)        tbs=2; ga=8;  ebs=4 ;;
        *)           tbs=4; ga=4;  ebs=8 ;;
    esac

    for LANG in en si; do
        tag=$(basename "$model")
        summary="outputs/headline_generation_finetuned/${tag}/${LANG}/rouge_summary.txt"
        if [ -f "$summary" ]; then
            echo "skip ${tag}/${LANG} (already complete)"
            continue
        fi

        echo "==================================================================="
        echo " ${model} | headline generation | prompt_lang=${LANG}"
        echo "==================================================================="
        python -m qwen35_train \
            --model_id "$model" \
            --prompt_lang "${LANG}" \
            --train_size 8000 \
            --test_size 1000 \
            --num_train_epochs 1 \
            --train_batch_size "$tbs" \
            --grad_accum "$ga" \
            --learning_rate 2e-4 \
            --max_seq_len 2560 \
            --eval_batch_size "$ebs" \
            --max_new_tokens 128 \
            --push_to_hub \
            --hub_repo "sinhala-nlp/${tag}-NSINA-Headlines-${LANG}"
    done
done

echo "All Qwen headline generation fine-tuning runs finished."