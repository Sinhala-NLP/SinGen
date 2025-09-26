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

model_id = "meta-llama/Llama-3.3-70B-Instruct"

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

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න (3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න."
    action_desc_si = "‘Simplified text:’ යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return [
                {"role": "user",
                 # "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
                "content": f"{task_desc} {action_desc} S: {row['Complex']}"}]

    if QUERY_TYPE == "zero-shot-si":
        return [
                {"role": "user",
                 # "content": f"Determine the semantic textual similarity between the following two sentences (S1, S2). The score should be ranging from 0.0 to 5.0, and can be a decimal. Return the score only following the prefix 'Score:' without any other text or explanations. S1: {row['sentence1']} S2: {row['sentence2']}"}]
                "content": f"{task_desc_si} {action_desc_si} S: {row['Complex']}"}]


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

def tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    if pd.isna(text) or text is None:
        return []
    # Simple tokenization - you might want to use a proper Sinhala tokenizer
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens



def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))

    df = full.tail(400)

    df.loc[:, 'chat'] = df.apply(format_chat, axis=1)

    # generate responses
    print("Generating predictions...")
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # extract predictions
    print("Extracting simplified text...")
    df['preds'] = df.apply(lambda row: extract_score(row['responses']), axis=1)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Calculate SARI scores
    sari_results = evaluate_sari_scores(df)

    # Save results with SARI scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_sari.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with SARI scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "sari_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"SARI Score Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Mean SARI Score: {sari_results['mean_sari']:.4f}\n")
        f.write(f"Standard Deviation: {sari_results['std_sari']:.4f}\n")
        f.write(f"Median SARI Score: {sari_results['median_sari']:.4f}\n")
        f.write(f"Min SARI Score: {sari_results['min_sari']:.4f}\n")
        f.write(f"Max SARI Score: {sari_results['max_sari']:.4f}\n")

    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), sari_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False, help='Type of query')

    args = parser.parse_args()
    QUERY_TYPE = args.query_type
    print(f"Query type: {QUERY_TYPE}")

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification", model_id.split('/')[-1], QUERY_TYPE)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    predictions, sari_results = predict()