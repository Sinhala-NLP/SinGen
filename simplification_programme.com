#!/bin/bash
#SBATCH -p astro
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk

source /etc/profile
module add anaconda3/2023.09
module add cuda/12.0

source activate /storage/hpc/37/ranasint/conda_envs/sts_exp
export HF_HOME=/scratch/hpc/37/ranasint/hf_cache

python -m text_simplification.hf_llm --query_type='zero-shot'