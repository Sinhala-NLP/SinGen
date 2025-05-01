import argparse
import logging
import os
import os
import re


import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline, set_seed

set_seed(777)

model_id = "meta-llama/Llama-3.1-8B-Instruct"

OUTPUT_FOLDER = os.path.join("outputs", model_id.split('/')[-1])
if not os.path.exists(OUTPUT_FOLDER): os.makedirs(OUTPUT_FOLDER)

print(model_id)

pipe_lm = pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.bfloat16},
    device_map="auto",
    do_sample=False,
    top_p=1.0,
)

def format_chat(row):
    # task_desc = "Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations."
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    if QUERY_TYPE == "zero-shot":
        return [
                {"role": "user",
                 # "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
                "content": f"{task_desc} {action_desc} S1: {row['Complex']}"}]


def query(pipe, inputs):
    """
    :param pipe: text-generation pipeline
    :param model_folder_path: list of messages
    :return: list
    """
    assistant_outputs = []

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    for out in tqdm(pipe(
            inputs,
            max_new_tokens=200,
            # pad_token_id=pipe.model.config.eos_token_id,
            eos_token_id=terminators,
            pad_token_id=pipe.tokenizer.eos_token_id

    )):
        assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())

    return assistant_outputs

def extract_score(response):
    try:
        simplified_sentence = re.findall(r'Simplified text:\s*(.*)', response)
    except IndexError:
        simplified_sentence = ""

    return simplified_sentence

def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))

    df = full.tail(400)

    df.loc[:, 'chat'] = df.apply(format_chat, axis=1)

    # generate responses
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # extract predictions
    df['preds'] = df.apply(lambda row: extract_score(row['responses']), axis=1)

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), header=True, index=False,
              encoding='utf-8')

    return df['preds'].tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False, help='Type of query')

    args = parser.parse_args()
    QUERY_TYPE = args.query_type
    # print(f"query type: {QUERY_TYPE}")

    predict()