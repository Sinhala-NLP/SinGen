#!/bin/bash
#SBATCH --partition=cpu-6h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

conda init
conda activate /mnt/nfs/homes/ranasint/anaconda3/envs/llm_exp
export HF_HOME=/mnt/nfs/homes/ranasint/hf_home

huggingface-cli login --token

python -m text_simplification.cohere --query_type='zero-shot'