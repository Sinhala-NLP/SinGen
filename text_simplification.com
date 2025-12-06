#!/bin/bash
#SBATCH --partition=a5000-6h
#SBATCH --gres=gpu:nvidia_rtx_a5000:3
#SBATCH --mem=80G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

export HF_HOME=/mnt/nfs/homes/ranasint/hf_home

huggingface-cli login --token

python -m text_simplification.hf_llm --model_id Qwen/Qwen3-30B-A3B-Instruct-2507 --query_type='zero-shot'
python -m text_simplification.hf_llm --model_id Qwen/Qwen3-30B-A3B-Instruct-2507 --query_type='zero-shot-si'
python -m text_simplification.hf_llm --model_id Qwen/Qwen3-30B-A3B-Instruct-2507 --query_type='few-shot'
python -m text_simplification.hf_llm --model_id Qwen/Qwen3-30B-A3B-Instruct-2507 --query_type='few-shot-si'

python -m text_simplification.hf_llm --model_id meta-llama/Llama-3.1-8B-Instruct --query_type='zero-shot'
python -m text_simplification.hf_llm --model_id meta-llama/Llama-3.1-8B-Instruct --query_type='zero-shot-si'
python -m text_simplification.hf_llm --model_id meta-llama/Llama-3.1-8B-Instruct --query_type='few-shot'
python -m text_simplification.hf_llm --model_id meta-llama/Llama-3.1-8B-Instruct --query_type='few-shot-si'

