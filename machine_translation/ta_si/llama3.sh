#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40:3
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=

# Tamil->Sinhala translation prompts are short (single source sentence + 3 short
# few-shot pairs), so smaller Llamas can take a larger batch than the flat bs=8
# used for the (longer-input) simplification job. Only the 70B checkpoints are
# held at 8.
#
# Decoding is greedy (no --do_sample) for BLEU reproducibility and comparability
# with the Gemma/Qwen Ta->Si numbers. max_new_tokens is 256: Llama has no
# thinking mode so truncation is the only risk, but Sinhala tokenizes densely,
# so a longer source sentence can need well over the 128 used elsewhere.
#
# Note: all Llama-3 checkpoints are gated -- HF_TOKEN must be set above and the
# Llama-3 license accepted on the HF account.

for model in \
    meta-llama/Llama-3.2-1B-Instruct meta-llama/Llama-3.2-3B-Instruct \
    meta-llama/Meta-Llama-3-8B-Instruct meta-llama/Llama-3.1-8B-Instruct \
    meta-llama/Meta-Llama-3-70B-Instruct meta-llama/Llama-3.1-70B-Instruct meta-llama/Llama-3.3-70B-Instruct; do

    case "$model" in
        *70B*) bs=8 ;;
        *)     bs=16 ;;
    esac

    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        python -m llama3 \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 256
    done
done