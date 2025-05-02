import argparse
import os
import re

import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

# Set seed for reproducibility
set_seed(777)

# Model checkpoint and output folder
checkpoint = "bigscience/mt0-xxl"
OUTPUT_FOLDER = os.path.join("outputs", checkpoint.split('/')[-1])
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForSeq2SeqLM.from_pretrained(
    checkpoint,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
# Print model name
print(checkpoint)

# Query type is set later through argparse
QUERY_TYPE = "zero-shot"  # default

def format_chat(row):
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} S: {row['Complex']}"
        }

def query(model, tokenizer, inputs):
    outputs = []

    for inp in tqdm(inputs):
        input_text = inp['content']
        encoded = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
        generated_ids = model.generate(encoded, max_new_tokens=200)
        decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        outputs.append(decoded.strip())

    return outputs

def extract_score(response):
    matches = re.findall(r'Simplified text:\s*(.*)', response)
    return matches[0] if matches else ""

def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    df = full.tail(4)

    df['chat'] = df.apply(format_chat, axis=1)

    # generate responses
    responses = query(model, tokenizer, df['chat'].tolist())
    df['responses'] = responses

    # extract predictions
    df['preds'] = df['responses'].apply(extract_score)

    # save results
    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), header=True, index=False, encoding='utf-8')

    return df['preds'].tolist()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False, help='Type of query')
    args = parser.parse_args()
    QUERY_TYPE = args.query_type

    predict()
