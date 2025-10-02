#!/bin/bash
#SBATCH --partition=cpu-48h
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=t.ranasinghe@lancaster.ac.uk


conda init
conda activate /mnt/nfs/homes/ranasint/anaconda3/envs/llm_exp
export HF_HOME=/mnt/nfs/homes/ranasint/hf_home


python -m text_simplification.cohere --query_type='zero-shot'
python -m text_simplification.cohere --query_type='zero-shot-si'
python -m text_simplification.cohere --query_type='few-shot'
python -m text_simplification.cohere --query_type='few-shot-si'

python -m machine_translation.en_si.cohere --query_type='zero-shot'
python -m machine_translation.en_si.cohere --query_type='zero-shot-si'
python -m machine_translation.en_si.cohere --query_type='few-shot'
python -m machine_translation.en_si.cohere --query_type='few-shot-si'

python -m machine_translation.ta_si.cohere --query_type='zero-shot'
python -m machine_translation.ta_si.cohere --query_type='zero-shot-si'
python -m machine_translation.ta_si.cohere --query_type='few-shot'
python -m machine_translation.ta_si.cohere --query_type='few-shot-si'

python -m machine_translation.pi_si.cohere --query_type='zero-shot'
python -m machine_translation.pi_si.cohere --query_type='zero-shot-si'
python -m machine_translation.pi_si.cohere --query_type='few-shot'
python -m machine_translation.pi_si.cohere --query_type='few-shot-si'

python -m headline_generation.cohere --query_type='zero-shot'
python -m headline_generation.cohere --query_type='zero-shot-si'
python -m headline_generation.cohere --query_type='few-shot'
python -m headline_generation.cohere --query_type='few-shot-si'

python -m text_summerisation.cohere --query_type='zero-shot'
python -m text_summerisation.cohere --query_type='zero-shot-si'
python -m text_summerisation.cohere --query_type='few-shot'
python -m text_summerisation.cohere --query_type='few-shot-si'

