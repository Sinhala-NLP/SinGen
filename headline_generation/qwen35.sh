#!/bin/bash
#SBATCH -p astro
#SBATCH --array=0-27%5
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

# Real Qwen3 (text-only causal LMs). Dropped: 235B-A22B (too big for L40S) and
# all FP8/GPTQ/AWQ/GGUF/MLX/Base variants. qwen35.py auto-detects text vs
# multimodal, so the SAME script handles these. Qwen3 thinks by default; the
# script disables thinking (enable_thinking=False) and strips residual <think>.
#
# 7 models x 4 query types = 28 combos -> array indices 0..27.
# %5 caps concurrency at 5 tasks = 5 x 3 = 15 L40S GPUs (your astro quota).
models=(Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
        Qwen/Qwen3-8B Qwen/Qwen3-14B Qwen/Qwen3-32B Qwen/Qwen3-30B-A3B)
query_types=(zero-shot zero-shot-si few-shot few-shot-si)

n_qt=${#query_types[@]}
i=$SLURM_ARRAY_TASK_ID
model="${models[$(( i / n_qt ))]}"
qt="${query_types[$(( i % n_qt ))]}"

echo "Task $i -> model=$model  query_type=$qt on $(hostname)"

# Skip-if-done guard (sentinel = the merged summary).
summary="outputs/headline_generation/${model##*/}/${qt}/rouge_summary.txt"
if [ -f "$summary" ]; then
    echo "already done: $model $qt -- skipping"
    exit 0
fi

case "$model" in
    *32B*|*30B-A3B*)
        # 32B (~66GB) and 30B-A3B (~62GB total params) do not fit one 48GB L40S,
        # so they span all 3 GPUs via device_map="auto" -- no data split.
        python -m qwen35 \
            --model_id "$model" --query_type "$qt" \
            --batch_size 2 --max_new_tokens 256 --test_size 1000 \
            --num_shards 1 --shard_id 0
        ;;
    *)
        # <=14B fit on one L40S -> 3 data shards at once, one per GPU.
        case "$model" in *14B*) bs=4 ;; *8B*) bs=6 ;; *) bs=8 ;; esac
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

# Fold this combo's shard(s) into one canonical file set + ROUGE.
python -m qwen35 --merge --model_id "$model" --query_type "$qt" --test_size 1000