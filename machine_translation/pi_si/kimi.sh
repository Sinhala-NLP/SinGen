#!/bin/bash
#SBATCH --job-name=kimi_pali_si_mxfp4
#SBATCH --partition=workq
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=kimi_pali_%j.out
#SBATCH --error=kimi_pali_%j.err

# EXPERIMENTAL multi-node native-MXFP4 deployment for Isambard-AI GH200.
# Kimi-K3's vLLM recipe supports H100; TP8/PP4 across 8 x 4-GPU nodes
# is a proposed topology, NOT a validated Kimi-K3/Isambard configuration.
# Requires a working aarch64 Kimi-K3-capable vLLM + Ray environment,
# correct Slingshot networking/NCCL settings, and enough scratch space.

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate "$SCRATCHDIR/conda-envs/llm_exp"
export HF_HOME="$SCRATCHDIR/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8
export VLLM_ENGINE_READY_TIMEOUT_S=7200
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
cd "$SLURM_SUBMIT_DIR"

MODEL_ID=moonshotai/Kimi-K3
MODEL_NAME=moonshotai/Kimi-K3
QUERY_TYPES=(zero-shot zero-shot-si few-shot few-shot-si)

command -v ray >/dev/null || { echo 'ERROR: ray unavailable in environment' >&2; exit 1; }
command -v vllm >/dev/null || { echo 'ERROR: vllm unavailable in environment' >&2; exit 1; }
python - <<'PY'
import json
from huggingface_hub import hf_hub_download
cfg=json.load(open(hf_hub_download('moonshotai/Kimi-K3', 'config.json')))
txt=cfg.get('text_config', cfg)
q=txt.get('quantization_config') or cfg.get('quantization_config') or {}
print('Native checkpoint quantisation configuration:', q, flush=True)
if 'mxfp4' not in str(q).lower() and 'fp4' not in str(q).lower():
    raise SystemExit('Expected original native MXFP4 checkpoint; verify config before running')
print('Using original native MXFP4: NO additional quantisation requested.')
PY

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#NODES[@]} == 8 ]] || { echo 'Expected 8 allocated nodes' >&2; exit 1; }
HEAD="${NODES[0]}"
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD" hostname -I | awk '{print $1}')
[[ -n "$HEAD_IP" ]] || { echo 'Cannot resolve head node IP' >&2; exit 1; }
PORT=6379
export RAY_ADDRESS="$HEAD_IP:$PORT"
export VLLM_HOST_IP="$HEAD_IP"
RAY_PIDS=()
SERVE_PID=''
cleanup() {
    if [[ -n "$SERVE_PID" ]]; then kill "$SERVE_PID" 2>/dev/null || true; fi
    for PID in "${RAY_PIDS[@]}"; do kill "$PID" 2>/dev/null || true; done
}
trap cleanup EXIT

srun --overlap --nodes=1 --ntasks=1 --nodelist="$HEAD" \
    ray start --head --node-ip-address="$HEAD_IP" --port="$PORT" \
    --num-gpus=4 --block > "ray_head_${SLURM_JOB_ID}.log" 2>&1 &
RAY_PIDS+=($!)
sleep 15
for NODE in "${NODES[@]:1}"; do
    srun --overlap --nodes=1 --ntasks=1 --nodelist="$NODE" \
        ray start --address="$RAY_ADDRESS" --num-gpus=4 --block \
        > "ray_${NODE}_${SLURM_JOB_ID}.log" 2>&1 &
    RAY_PIDS+=($!)
done

# Wait for every node/GPU to be registered (not just a fixed sleep).
READY=0
for ((i=0; i<90; i++)); do
    if ray status --address="$RAY_ADDRESS" > "ray_status_${SLURM_JOB_ID}.log" 2>&1; then
        if python - "$RAY_ADDRESS" <<'PY'
import sys, ray
ray.init(address=sys.argv[1], logging_level='ERROR')
resources=ray.cluster_resources()
print('Cluster resources:', resources)
assert resources.get('GPU',0)>=32
PY
        then READY=1; break; fi
    fi
    sleep 10
done
[[ "$READY" == 1 ]] || { echo 'Ray cluster did not register 32 GPUs' >&2; exit 1; }

# No --quantization flag: vLLM reads the checkpoint's native MXFP4 metadata.
# --dtype bfloat16 applies to non-quantised tensors; it does NOT convert MXFP4.
# TP8 + PP4 is unvalidated on Isambard; adjust based on confirmed vLLM support.
vllm serve "$MODEL_ID" \
    --served-model-name "$MODEL_NAME" \
    --trust-remote-code \
    --dtype bfloat16 \
    --distributed-executor-backend ray \
    --tensor-parallel-size 8 \
    --pipeline-parallel-size 4 \
    --language-model-only \
    --reasoning-parser kimi_k3 \
    --max-model-len 8192 \
    --max-num-seqs 4 \
    --gpu-memory-utilization 0.90 \
    --host 127.0.0.1 --port 8000 \
    > "vllm_${SLURM_JOB_ID}.log" 2>&1 &
SERVE_PID=$!

SERVER_READY=0
for ((i=0; i<720; i++)); do
    kill -0 "$SERVE_PID" 2>/dev/null || { echo 'vLLM exited; inspect vllm log' >&2; exit 1; }
    if curl -fsS http://127.0.0.1:8000/v1/models >/dev/null; then SERVER_READY=1; break; fi
    sleep 10
done
[[ "$SERVER_READY" == 1 ]] || { echo 'vLLM readiness timeout' >&2; exit 1; }

for QUERY in "${QUERY_TYPES[@]}"; do
    python kimi.py \
        --model_id "$MODEL_NAME" \
        --query_type "$QUERY" \
        --base_url http://127.0.0.1:8000/v1 \
        --test_size 1000 \
        --max_new_tokens 2048 \
        --reasoning_effort low
 done
