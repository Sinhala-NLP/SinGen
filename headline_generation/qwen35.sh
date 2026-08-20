#!/bin/bash
#SBATCH -p astro
#SBATCH --array=0-23%5
#SBATCH --gres=gpu:nvidia_l40s:3
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=32
#SBATCH --output=log/output_%A_%a.log
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/llm_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export PYTHONIOENCODING=utf-8
export HF_HUB_DOWNLOAD_TIMEOUT=60
export TOKENIZERS_PARALLELISM=false

# 6 models x 4 query types = 24 combos -> array indices 0..23.
# %5 caps concurrency at 5 tasks = 5 x 3 = 15 L40S GPUs (your astro quota).
models=(Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B \
        Qwen/Qwen3.5-9B Qwen/Qwen3.5-27B Qwen/Qwen3.5-35B-A3B)
query_types=(zero-shot zero-shot-si few-shot few-shot-si)

n_qt=${#query_types[@]}
i=$SLURM_ARRAY_TASK_ID
model="${models[$(( i / n_qt ))]}"
qt="${query_types[$(( i % n_qt ))]}"

echo "Task $i -> model=$model  query_type=$qt on $(hostname)"

# Skip-if-done guard (sentinel = the merged summary), so a resubmit of the same
# array picks up only the combos that didn't finish within the 24h wall time.
summary="outputs/headline_generation/${model##*/}/${qt}/rouge_summary.txt"
if [ -f "$summary" ]; then
    echo "already done: $model $qt -- skipping"
    exit 0
fi

# max_new_tokens is 256 rather than the 128 used for Gemma/Llama: under greedy
# decoding a larger budget cannot change the extracted headline, it only
# protects against a residual <think> block eating the whole budget and leaving
# an empty prediction. Watch the "Empty predictions" line in each shard log.
case "$model" in
    *27B*|*35B*)
        # 27B (~54GB) and 35B-A3B (~70GB) do not fit on one 48GB L40S, so they
        # span all 3 GPUs via device_map="auto" -- no data split for these.
        python -m qwen35 \
            --model_id "$model" --query_type "$qt" \
            --batch_size 2 --max_new_tokens 256 --test_size 1000 \
            --num_shards 1 --shard_id 0
        ;;
    *)
        # Small models fit on one L40S, so run 3 data shards at once, one per GPU.
        case "$model" in *9B*) bs=4 ;; *) bs=8 ;; esac
        CUDA_VISIBLE_DEVICES=0 python -m qwen35 \
            --model_id "$model" --query_type "$qt" \
            --batch_size "$bs" --max_new_tokens 256 --test_size 1000 \
            --num_shards 3 --shard_id 0 &
        CUDA_VISIBLE_DEVICES=1 python -m qwen35 \
            --model_id "$model" --query_type "$qt" \
            --batch_size "$bs" --max_new_tokens 256 --test_size 1000 \
            --num_shards 3 --shard_id 1 &
        CUDA_VISIBLE_DEVICES=2 python -m qwen35 \
            --model_id "$model" --query_type "$qt" \
            --batch_size "$bs" --max_new_tokens 256 --test_size 1000 \
            --num_shards 3 --shard_id 2 &
        wait
        ;;
esac

# Fold this combo's shard(s) into one canonical file set + ROUGE (same script,
# --merge mode: no model is loaded).
python -m qwen35 --merge --model_id "$model" --query_type "$qt" --test_size 1000