#!/bin/bash
#SBATCH -p gpu-medium
#SBATCH --gres=gpu:nvidia_h200_nvl:2
#SBATCH --mem=100G
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
# Needs a WRITE-scoped token for the push to sinhala-nlp.
export HF_TOKEN=

# FLORES-200 ships no train split: this fine-tunes on `dev` (997 pairs) and
# evaluates on `devtest` (1012). The splits are document-disjoint. ~187
# optimizer steps at 3 epochs is a thin signal -- point --train_tsv at a larger
# En-Si parallel corpus (columns english/sinhala) for a real supervised baseline.
#
# LoRA targets are selected at runtime: vision-tower linears are excluded, and
# MoE checkpoints (35B-A3B) fall back to attention-only automatically.
# 27B and 35B-A3B need both GPUs; the smaller checkpoints use one.
for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) tbs=1; ga=16; ebs=2 ;;
        *9B*)        tbs=2; ga=8;  ebs=4 ;;
        *)           tbs=4; ga=4;  ebs=8 ;;
    esac

    for LANG in en si; do
        echo "==================================================================="
        echo " ${model} | En=>Si translation | prompt_lang=${LANG}"
        echo "==================================================================="
        python train_translation_qwen.py \
            --model_id "$model" \
            --prompt_lang "${LANG}" \
            --num_train_epochs 3 \
            --train_batch_size "$tbs" \
            --grad_accum "$ga" \
            --learning_rate 2e-4 \
            --max_seq_len 512 \
            --eval_batch_size "$ebs" \
            --max_new_tokens 200 \
            --push_to_hub \
            --hub_repo "sinhala-nlp/$(basename $model)-FLORES200-En2Si-${LANG}"
    done
done

echo "All Qwen En=>Si translation fine-tuning runs finished."