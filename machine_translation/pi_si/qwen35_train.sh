#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:1
#SBATCH --mem=100G
#SBATCH --time=48:00:00
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
# Pali => Sinhala translation, LoRA instruction fine-tuning.
#
# The test set is the trailing 1000 rows, identical to the prompting scripts.
# The corpus is in canonical order, so that tail is much LONGER than the
# training pool (median target ~370 vs ~50 whitespace tokens). Consequences:
#   * --max_seq_len 3072 and --max_new_tokens 512, not the FLORES values.
#     Overlong training targets are dropped, not cut; the count is logged.
#   * The fine-tuned model inherits a short-output prior from training and will
#     under-generate on the long test passages. The script prints a hyp/ref
#     length ratio after evaluation -- read the BLEU numbers next to it, since a
#     ratio well under 1.0 means the brevity penalty is doing the work.
#   * --length_match_train is available as an ablation (second loop below,
#     commented out) that resamples the training set toward the test length
#     distribution. It oversamples the few long training passages, so treat it
#     as a diagnostic row rather than the headline number.
#
# WARNING on hardware: 2 x L40S is 96 GB total. At 3072-token sequences the 27B
# (~54 GB of bf16 weights) is tight and the 35B-A3B (~70 GB) will very likely
# OOM. If `astro` can give you the H200 NVL pair, use it.
# ---------------------------------------------------------------------------
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) tbs=1; ga=16; ebs=1 ;;
        *9B*)        tbs=1; ga=16; ebs=2 ;;
        *4B*)        tbs=2; ga=8;  ebs=2 ;;
        *)           tbs=2; ga=8;  ebs=4 ;;
    esac

    for LANG in en si; do
        tag=$(basename "$model")
        summary="outputs/pali_sinhala_translation_finetuned/${tag}/${LANG}/bleu_summary.txt"
        if [ -f "$summary" ]; then
            echo "skip ${tag}/${LANG} (already complete)"
            continue
        fi

        echo "==================================================================="
        echo " ${model} | Pali=>Si translation | prompt_lang=${LANG}"
        echo "==================================================================="
        python -m qwen35_train \
            --model_id "$model" \
            --prompt_lang "${LANG}" \
            --test_size 1000 \
            --train_size 10000 \
            --num_train_epochs 1 \
            --train_batch_size "$tbs" \
            --grad_accum "$ga" \
            --learning_rate 2e-4 \
            --max_seq_len 3072 \
            --eval_batch_size "$ebs" \
            --max_new_tokens 512 \
            --push_to_hub \
            --hub_repo "sinhala-nlp/${tag}-PaliSinhala-Pali2Si-${LANG}"
    done
done

# --- Optional length-matched ablation (one model, both prompt languages) -----
# Writes to outputs/pali_sinhala_translation_finetuned/<tag>/<lang>-lenmatch/
# and does NOT push, so it cannot overwrite the headline adapter on the Hub.
#
# for LANG in en si; do
#     python -m qwen35_pali_sinhala_train \
#         --model_id Qwen/Qwen3.5-9B \
#         --prompt_lang "${LANG}" \
#         --train_size 10000 \
#         --length_match_train \
#         --num_train_epochs 1 \
#         --train_batch_size 1 --grad_accum 16 \
#         --max_seq_len 3072 --eval_batch_size 2 --max_new_tokens 512
# done

echo "All Qwen Pali=>Si translation fine-tuning runs finished."