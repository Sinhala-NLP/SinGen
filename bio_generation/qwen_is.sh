
#!/bin/bash
#SBATCH --job-name=qwen_biography
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --array=0-67%4
#SBATCH --output=logs/qwen_%A_%a.out
#SBATCH --error=logs/qwen_%A_%a.err


# ==========================================
# Conda environment
# ==========================================

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$SCRATCHDIR/conda-envs/llm_exp"

# ==========================================
# Hugging Face configuration
# ==========================================

export HF_HOME="$SCRATCHDIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"

export HF_TOKEN=

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# ==========================================
# Runtime configuration
# ==========================================

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# Prevent unnecessary tokenizer warnings
export HF_HUB_DISABLE_TELEMETRY=1

# Go to project directory
cd "$SLURM_SUBMIT_DIR"

# ==========================================
# Models (17 models)
# ==========================================

models=(
    Qwen/Qwen2-0.5B-Instruct
    Qwen/Qwen2-1.5B-Instruct
    Qwen/Qwen2-7B-Instruct
    Qwen/Qwen2.5-0.5B-Instruct
    Qwen/Qwen2.5-1.5B-Instruct
    Qwen/Qwen2.5-3B-Instruct
    Qwen/Qwen2.5-7B-Instruct
    Qwen/Qwen2.5-14B-Instruct
    Qwen/Qwen2.5-32B-Instruct
    Qwen/Qwen3.5-0.8B
    Qwen/Qwen3.5-2B
    Qwen/Qwen3.5-4B
    Qwen/Qwen3.5-9B
    Qwen/Qwen3.5-27B
    Qwen/Qwen3.5-35B-A3B
    Qwen/Qwen3.6-27B
    Qwen/Qwen3.6-35B-A3B
)

# ==========================================
# Query types (4)
# ==========================================

qts=(
    zero-shot
    zero-shot-si
    few-shot
    few-shot-si
)

# ==========================================
# Array task mapping
# ==========================================

model_idx=$((SLURM_ARRAY_TASK_ID / 4))
qt_idx=$((SLURM_ARRAY_TASK_ID % 4))

model=${models[$model_idx]}
qt=${qts[$qt_idx]}

# ==========================================
# Model-specific batch sizes
# ==========================================

declare -A BATCH=(
    ["Qwen/Qwen2-7B-Instruct"]=16
    ["Qwen/Qwen2.5-7B-Instruct"]=16
    ["Qwen/Qwen2.5-14B-Instruct"]=16
    ["Qwen/Qwen3.5-9B"]=16
    ["Qwen/Qwen2.5-32B-Instruct"]=8
    ["Qwen/Qwen3.5-27B"]=8
    ["Qwen/Qwen3.5-35B-A3B"]=8
    ["Qwen/Qwen3.6-27B"]=8
    ["Qwen/Qwen3.6-35B-A3B"]=8
)

bs=${BATCH[$model]:-32}

# ==========================================
# Skip completed experiments
# ==========================================

output_dir="outputs/biography_generation/${model#*/}/$qt"

if [ -f "$output_dir/rouge_summary.txt" ]; then
    echo "Already completed: $model | $qt"
    exit 0
fi

# ==========================================
# Job information
# ==========================================

echo "=========================================="
echo "Isambard-AI Qwen Biography Generation"
echo "=========================================="
echo "Job ID:           $SLURM_JOB_ID"
echo "Array task:       $SLURM_ARRAY_TASK_ID"
echo "Node:             $(hostname)"
echo "Model:            $model"
echo "Query type:       $qt"
echo "Batch size:       $bs"
echo "Conda env:        $CONDA_PREFIX"
echo "HuggingFace home: $HF_HOME"
echo "=========================================="

# ==========================================
# GPU verification
# ==========================================

nvidia-smi

python -c "
import torch
print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError('CUDA is not available')
print('GPU:', torch.cuda.get_device_name(0))
"

# ==========================================
# Run experiment
# ==========================================

echo "Starting experiment: $model | $qt"

python -m qwen \
    --model_id "$model" \
    --query_type "$qt" \
    --batch_size "$bs" \
    --max_new_tokens 512

echo "=========================================="
echo "Finished: $model | $qt"
echo "=========================================="
