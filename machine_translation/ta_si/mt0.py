import argparse
import os
import re
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed

# Set seed for reproducibility
set_seed(777)

# Model checkpoint
checkpoint = "bigscience/mt0-xxl"

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
    task_desc = "You are an expert translator specializing in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} T: {row['Tamil']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} T: {row['Tamil']}"
        }


def query(model, tokenizer, inputs):
    outputs = []

    for inp in tqdm(inputs, desc="Generating translations"):
        try:
            input_text = inp['content']
            encoded = tokenizer.encode(input_text, return_tensors="pt").to("cuda")
            generated_ids = model.generate(encoded, max_new_tokens=200)
            decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            outputs.append(decoded.strip())
        except Exception as e:
            print(f"Error with input: {e}")
            outputs.append("")

    return outputs


def extract_translation(response):
    if not isinstance(response, str):
        print("Non-string response:", response)
        return ""
    matches = re.findall(r'Translation:\s*(.*)', response, re.DOTALL)
    return matches[0].strip() if matches else response.strip()


# BLEU Score Evaluation Functions
def tokenize(text):
    """Simple tokenization by splitting on whitespace"""
    if pd.isna(text) or text is None:
        return []
    text = str(text).strip()
    tokens = text.split()
    return tokens


def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
    """
    Calculate individual BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4) and overall BLEU
    Returns a dictionary with individual scores and the overall BLEU score
    """
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not pred_tokens or not ref_tokens:
        return {
            'bleu_1': 0.0,
            'bleu_2': 0.0,
            'bleu_3': 0.0,
            'bleu_4': 0.0,
            'bleu_overall': 0.0
        }

    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

    # Calculate n-gram precisions
    precisions = []
    individual_bleu_scores = {}

    for n in range(1, max_n + 1):
        ref_ngrams = {}
        pred_ngrams = {}

        # Count reference n-grams
        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

        # Count prediction n-grams
        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i + n])
            pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1

        # Calculate clipped counts
        clipped_count = 0
        total_count = sum(pred_ngrams.values())

        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                clipped_count += min(count, ref_ngrams[ngram])

        # Calculate precision for this n-gram order
        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)

        # Calculate individual BLEU-n score (with brevity penalty)
        individual_bleu_scores[f'bleu_{n}'] = (bp * precision * 100) if precision > 0 else 0.0

    # Calculate overall BLEU score (geometric mean of all precisions)
    if min(precisions) > 0:
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    individual_bleu_scores['bleu_overall'] = bp * geo_mean * 100

    return individual_bleu_scores


def evaluate_bleu_scores(df):
    """
    Calculate individual and overall BLEU scores for the translations dataframe
    """
    print("\nCalculating BLEU scores...")

    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    bleu_overall_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing BLEU"):
        reference = row['Sinhala']
        prediction = row['preds']

        if pd.isna(reference) or pd.isna(prediction) or not str(reference).strip() or not str(prediction).strip():
            bleu_1_scores.append(0.0)
            bleu_2_scores.append(0.0)
            bleu_3_scores.append(0.0)
            bleu_4_scores.append(0.0)
            bleu_overall_scores.append(0.0)
            continue

        scores = calculate_bleu_score_individual(str(reference), str(prediction))
        bleu_1_scores.append(scores['bleu_1'])
        bleu_2_scores.append(scores['bleu_2'])
        bleu_3_scores.append(scores['bleu_3'])
        bleu_4_scores.append(scores['bleu_4'])
        bleu_overall_scores.append(scores['bleu_overall'])

    df['bleu_1'] = bleu_1_scores
    df['bleu_2'] = bleu_2_scores
    df['bleu_3'] = bleu_3_scores
    df['bleu_4'] = bleu_4_scores
    df['bleu_overall'] = bleu_overall_scores

    # Calculate statistics for each BLEU variant
    print(f"\n" + "=" * 70)
    print(f"BLEU Score Evaluation Results:")
    print(f"=" * 70)

    for score_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
        scores = df[score_type].tolist()
        print(f"\n{score_type.upper().replace('_', '-')}:")
        print(f"  Mean:   {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std:    {np.std(scores):.4f}")
        print(f"  Min:    {np.min(scores):.4f}")
        print(f"  Max:    {np.max(scores):.4f}")

    print(f"=" * 70)

    return {
        'bleu_1': {
            'mean': np.mean(bleu_1_scores),
            'median': np.median(bleu_1_scores),
            'std': np.std(bleu_1_scores),
            'min': np.min(bleu_1_scores),
            'max': np.max(bleu_1_scores),
            'scores': bleu_1_scores
        },
        'bleu_2': {
            'mean': np.mean(bleu_2_scores),
            'median': np.median(bleu_2_scores),
            'std': np.std(bleu_2_scores),
            'min': np.min(bleu_2_scores),
            'max': np.max(bleu_2_scores),
            'scores': bleu_2_scores
        },
        'bleu_3': {
            'mean': np.mean(bleu_3_scores),
            'median': np.median(bleu_3_scores),
            'std': np.std(bleu_3_scores),
            'min': np.min(bleu_3_scores),
            'max': np.max(bleu_3_scores),
            'scores': bleu_3_scores
        },
        'bleu_4': {
            'mean': np.mean(bleu_4_scores),
            'median': np.median(bleu_4_scores),
            'std': np.std(bleu_4_scores),
            'min': np.min(bleu_4_scores),
            'max': np.max(bleu_4_scores),
            'scores': bleu_4_scores
        },
        'bleu_overall': {
            'mean': np.mean(bleu_overall_scores),
            'median': np.median(bleu_overall_scores),
            'std': np.std(bleu_overall_scores),
            'min': np.min(bleu_overall_scores),
            'max': np.max(bleu_overall_scores),
            'scores': bleu_overall_scores
        }
    }


def predict(tsv_file_path):
    """Main prediction function"""
    # Load the TSV file
    print(f"Loading data from {tsv_file_path}...")
    full_df = pd.read_csv(tsv_file_path, sep='\t', encoding='utf-8')

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    # Use last 1000 samples for testing
    test_size = min(1000, len(full_df))
    df = full_df.tail(test_size).copy()
    print(f"Using last {test_size} samples for testing")

    df['chat'] = df.apply(format_chat, axis=1)

    # Generate responses
    print("Generating translations...")
    responses = query(model, tokenizer, df['chat'].tolist())
    df['responses'] = responses

    # Extract translations
    print("Extracting translations...")
    df['preds'] = df['responses'].apply(extract_translation)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Evaluate with BLEU score
    print("Evaluating translations with BLEU score...")
    bleu_results = evaluate_bleu_scores(df)

    # Save results with BLEU scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_bleu.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with BLEU scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "bleu_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score Evaluation Results\n")
        f.write(f"Model: {checkpoint}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"=" * 70 + "\n\n")

        for score_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
            f.write(f"{score_type.upper().replace('_', '-')}:\n")
            f.write(f"  Mean:   {bleu_results[score_type]['mean']:.4f}\n")
            f.write(f"  Median: {bleu_results[score_type]['median']:.4f}\n")
            f.write(f"  Std:    {bleu_results[score_type]['std']:.4f}\n")
            f.write(f"  Min:    {bleu_results[score_type]['min']:.4f}\n")
            f.write(f"  Max:    {bleu_results[score_type]['max']:.4f}\n\n")

    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), bleu_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False, help='Type of query')
    args = parser.parse_args()
    QUERY_TYPE = args.query_type
    print(f"Query type: {QUERY_TYPE}")

    TSV_FILE = os.path.join("machine_translation", "ta_si", "ta_si.tsv")
    print(f"TSV file: {TSV_FILE}")

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "tamil_sinhala_translation", checkpoint.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict(TSV_FILE)