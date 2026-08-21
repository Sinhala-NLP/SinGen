#!/bin/bash
#SBATCH -p serial                     # a CPU partition -- set to your cluster's CPU queue
#SBATCH --job-name=sinhala-tokenizer
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk
#SBATCH -o logs/tokenizer-%j.out
#SBATCH -e logs/tokenizer-%j.err

source /etc/profile
module add anaconda3/2023.09
source activate /storage/hpc/37/ranasint/conda_envs/llm_exp

# set -u only after conda activation (activation scripts reference unbound vars)
set -uo pipefail

export HF_HOME=/scratch/hpc/37/ranasint/hf_cache
export HF_TOKEN=
export HF_HUB_DISABLE_XET=1            # plain resumable downloader (avoids the Xet writer crash)
export HF_HUB_DOWNLOAD_TIMEOUT=60
export PYTHONIOENCODING=utf-8

mkdir -p logs

# Retrain the WordPiece tokenisers with the strip-accents fix. RoBERTa uses
# byte-level BPE and was never affected, so it's omitted; add it to the list if
# you want a fresh one anyway.
for model_type in bert electra; do
    tok_dir=/scratch/hpc/37/ranasint/sinhala_lms/tok_${model_type}

    echo "[$(date)] TRAIN $model_type -> $tok_dir"
    rm -rf "$tok_dir"                 # force a clean rebuild over the buggy vocab

    if python train_tokenizer.py --model_type "$model_type" --output_dir "$tok_dir"; then
        echo "[$(date)] DONE  $model_type"
    else
        echo "[$(date)] FAIL  $model_type" >&2
    fi
done

# Sanity-check the fix: hal kirima must survive and vowels stay composed.
echo "[$(date)] verifying tok_bert round-trip:"
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('/scratch/hpc/37/ranasint/sinhala_lms/tok_bert')
s = 'අද ඉතා හොඳ දවසක්.'
print(' tokens:', t.tokenize(s))
print(' decode:', t.decode(t(s, add_special_tokens=False)['input_ids']))
"

echo "[$(date)] Tokeniser job finished."