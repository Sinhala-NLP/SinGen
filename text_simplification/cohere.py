import argparse
import os
import re

import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
import cohere

# Set up Cohere client
COHERE_API_KEY = "<<your-api-key>>"  # Replace with your actual key
co = cohere.ClientV2(COHERE_API_KEY)

model_id = "command-a-03-2025"  # or "command-a-03-2025" as per your need

OUTPUT_FOLDER = os.path.join("outputs", "text_simplification", model_id)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def format_chat(row):
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} S: {row['Complex']}"
        }


def query_cohere(client, model, messages):
    outputs = []
    for msg in tqdm(messages):
        response = client.chat(
            model=model,
            messages=[msg],
            temperature=0.3,
        )
        outputs.append(response.text.strip())
    return outputs


def extract_score(response):
    matches = re.findall(r'Simplified text:\s*(.*)', response)
    return matches[0] if matches else ""


def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    df = full.tail(4)

    df['chat'] = df.apply(format_chat, axis=1)

    responses = query_cohere(co, model_id, df['chat'].tolist())
    df['responses'] = responses
    df['preds'] = df['responses'].apply(extract_score)

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), header=True, index=False, encoding='utf-8')
    return df['preds'].tolist()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False, help='Type of query')
    args = parser.parse_args()
    QUERY_TYPE = args.query_type

    predict()
