#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:3
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
export HF_TOKEN=

# Pali->Sinhala prompts are MUCH longer than the Ta->Si ones: a single Pali
# passage can run to several hundred tokens, and the few-shot variants stack
# three demonstration pairs on top (each capped at 500 chars per side by
# FEWSHOT_PREVIEW_CHARS). Batch sizes are therefore roughly halved relative to
# the Ta->Si job at every model size.
#
# Decoding is greedy (no --do_sample): BLEU must be reproducible across runs and
# comparable to the Gemma and Llama Pali->Si numbers, which were also greedy.
#
# max_new_tokens is 512 rather than 256. A translated Pali passage genuinely
# needs more room than a single sentence, and the extra budget also guards
# against a residual <think> block eating the whole generation and leaving an
# empty prediction (which scores BLEU 0 and looks like a model failure). Watch
# the "Empty predictions" line in each log.

for model in Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
             Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B; do

    case "$model" in
        *27B*|*35B*) bs=2 ;;
        *9B*)        bs=4 ;;
        *)           bs=8 ;;
    esac

    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        echo "=== $model | $qt | batch_size=$bs ==="
        python -m qwen35_pali_sinhala_translation \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 512
    done
done