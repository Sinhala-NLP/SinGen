#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40:3
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

# Pali->Sinhala translation. Test set is the last 1000 rows (fixed in the .py);
# the dev pool (the rest) supplies few-shot examples.
#
# Batch casing follows the Tamil->Sinhala job (70B at 8, smaller at 16). Pali
# passages can run longer than single FLORES sentences, though, so if you see
# OOM under few-shot, drop each tier by one step.
#
# Decoding is greedy (no --do_sample) for BLEU reproducibility. max_new_tokens
# is 256; Llama has no thinking mode, so truncation of a long passage is the
# only risk.
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
        python -m llama3_pali_sinhala_translation \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 256
    done
done