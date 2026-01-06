#!/bin/bash -l
#SBATCH --job-name=headline_generation-gpu
#SBATCH --account=AIRR-P39-DAWN-GPU
#SBATCH --partition=pvc9
#SBATCH --nodes=1               # number of nodes
#SBATCH --gres=gpu:1 # Number of requested GPUs per node
#SBATCH --time=06:00:00              # total run time limit (HH:MM:SS)

module purge
module load rhel8/default-dawn
module load intelpython-conda/2025.0

conda activate /home/dn-rana1/rds/conda_envs/llm_exp

export HF_HOME=//home/dn-rana1/rds/rds-airr-p39-JpwWyPZa2Oc/hf_home/
export HF_TOKEN=""


python -m headline_generation.hf_llm --model_id='meta-llama/Llama-3.3-70B-Instruct' --query_type='zero-shot-si'
