
#!/bin/bash
#SBATCH --job-name=gemma_biogen_ft
#SBATCH --partition=workq
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --requeue
#SBATCH --output=logs/gemma_ft_%A_%a.out
#SBATCH --error=logs/gemma_ft_%A_%a.err


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

export HF_TOKEN="hf_YOUR_TOKEN_HERE"

mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# ==========================================
# Runtime configuration
# ==========================================

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd "$SLURM_SUBMIT_DIR"

# ==========================================
# Models (5)
# ==========================================

MODELS=(
    google/gemma-4-31B-it
    google/gemma-4-12B-it
    google/gemma-3-27b-it
    google/gemma-3-12b-it
    google/gemma-3-4b-it
)

# ==========================================
# Fine-tuning languages
# ==========================================

LANGS=(en si)

# ==========================================
# Array task mapping
# ==========================================

model_idx=$((SLURM_ARRAY_TASK_ID / 2))
lang_idx=$((SLURM_ARRAY_TASK_ID % 2))

model=${MODELS[$model_idx]}
lang=${LANGS[$lang_idx]}

# ==========================================
# Skip completed experiments
# ==========================================

out="outputs/biography_generation_finetuned/${model##*/}/$lang/rouge_summary.txt"

if [ -f "$out" ]; then
    echo "Already completed: $model | $lang"
    exit 0
fi

# ==========================================
# Job information
# ==========================================

echo "=========================================="
echo "Isambard-AI Gemma BioGen Fine-tuning"
echo "=========================================="
echo "Job ID:          $SLURM_JOB_ID"
echo "Array task:      $SLURM_ARRAY_TASK_ID"
echo "Node:            $(hostname)"
echo "Model:           $model"
echo "Language:        $lang"
echo "Conda env:       $CONDA_PREFIX"
echo "HF_HOME:         $HF_HOME"
echo "Visible GPUs:    ${CUDA_VISIBLE_DEVICES:-unset}"
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
print('GPU count:', torch.cuda.device_count())

assert torch.cuda.is_available(), 'CUDA unavailable'
assert torch.cuda.device_count() >= 2, 'Two GPUs required'

for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f'GPU {i}: {props.name}')
    print(f'Memory: {props.total_memory / 1024**3:.1f} GiB')
"

# ==========================================
# Run Gemma fine-tuning
# ==========================================

echo "Starting fine-tuning: $model | $lang"

python -m gemma_train \
    --model_id "$model" \
    --prompt_lang "$lang" \
    --eval_batch_size 8

echo "=========================================="
echo "Finished: $model | $lang"
echo "=========================================="
