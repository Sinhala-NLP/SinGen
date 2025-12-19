#!/bin/bash -l
#SBATCH --job-name=headline_generation
#SBATCH --account=dn-rana1
#SBATCH --partition=pvc9
#SBATCH -n 1   # Number of tasks (usually number of MPI ranks)
#SBATCH -c 24  # Number of cores per task
#SBATCH --gres=gpu:1 # Number of requested GPUs per node
#SBATCH --time=06:00:00              # total run time limit (HH:MM:SS)

module purge
module load rhel9/default-dawn
module load intelpython-conda

conda activate ~/rds/conda_envs/llm_exp

export HF_HOME=/home/dn-rana1/rds/hf_home


huggingface-cli login --token

python -m headline_generation.hf_llm --model_id='meta-llama/Llama-3.3-70B-Instruct' --query_type='zero-shot-si'