#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:nvidia_l40s:3
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

# Pali passages are long and few-shot prompts stack three of them, so the larger
# checkpoints get a smaller batch to keep headroom on 2x H200.
declare -A BATCH=(
    ["google/gemma-4-31B-it"]=4
    ["google/gemma-4-12B-it"]=8
    ["google/gemma-3-27b-it"]=4
    ["google/gemma-3-12b-it"]=8
)

for model in google/gemma-4-31B-it google/gemma-4-12B-it google/gemma-3-27b-it google/gemma-3-12b-it; do
    bs=${BATCH[$model]:-8}
    for qt in zero-shot zero-shot-si few-shot few-shot-si; do
        echo "=== $model | $qt | batch_size=$bs ==="
        python -m gemma_pali_sinhala_translation \
            --model_id "$model" \
            --query_type "$qt" \
            --batch_size "$bs" \
            --max_new_tokens 512
    done
done