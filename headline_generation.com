#!/bin/bash
#SBATCH --partition=a5000-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home

pip install flash_attn --no-build-isolation

huggingface-cli login --token

python -m headline_generation.hf_llm --model_id='meta-llama/Llama-3.1-8B-Instruct' --query_type='zero-shot-si'