import argparse
import os
import re
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
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
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න (3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න."
    action_desc_si = "'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} S: {row['Complex']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} S: {row['Complex']}"
        }


def query(model, tokenizer, inputs):
    outputs = []

    for inp in tqdm(inputs, desc="Generating predictions"):
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


def extract_score(response):
    if not isinstance(response, str):
        print("Non-string response:", response)
        return ""
    matches = re.findall(r'Simplified text:\s*(.*)', response)
    return matches[0] if matches else response.strip()


# SARI Evaluation Functions
def tokenize(text):
    """Simple tokenization by splitting on whitespace and punctuation"""
    if pd.isna(text) or text is None:
        return []
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_ngram_f1_multi_ref(source_tokens: List[str],
                                 target_tokens_list: List[List[str]],
                                 prediction_tokens: List[str],
                                 n: int) -> Tuple[float, float, float]:
    """
    Calculate n-gram F1 scores for KEEP, DELETE, and ADD operations with multiple references
    """

    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return set()
        return set([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])

    source_ngrams = get_ngrams(source_tokens, n)
    pred_ngrams = get_ngrams(prediction_tokens, n)

    # For multiple references, we need to consider all possible target ngrams
    all_target_ngrams = set()
    for target_tokens in target_tokens_list:
        target_ngrams = get_ngrams(target_tokens, n)
        all_target_ngrams.update(target_ngrams)

    # KEEP: n-grams that should be kept from source
    keep_target = set()
    for target_tokens in target_tokens_list:
        target_ngrams = get_ngrams(target_tokens, n)
        keep_target.update(source_ngrams & target_ngrams)

    keep_pred = source_ngrams & pred_ngrams

    # DELETE: n-grams that should be deleted from source
    delete_target = source_ngrams - all_target_ngrams
    delete_pred = source_ngrams - pred_ngrams

    # ADD: n-grams that should be added
    add_target = all_target_ngrams - source_ngrams
    add_pred = pred_ngrams - source_ngrams

    def f1_score(tp, fp, fn):
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0
        else:
            return 2 * precision * recall / (precision + recall)

    keep_f1 = f1_score(len(keep_target & keep_pred),
                       len(keep_pred - keep_target),
                       len(keep_target - keep_pred))

    delete_f1 = f1_score(len(delete_target & delete_pred),
                         len(delete_pred - delete_target),
                         len(delete_target - delete_pred))

    add_f1 = f1_score(len(add_target & add_pred),
                      len(add_pred - add_target),
                      len(add_target - add_pred))

    return keep_f1, delete_f1, add_f1


def calculate_ngram_f1_single_ref(source_tokens: List[str],
                                  target_tokens: List[str],
                                  prediction_tokens: List[str],
                                  n: int) -> Tuple[float, float, float]:
    """
    Calculate n-gram F1 scores for KEEP, DELETE, and ADD operations with single reference
    """

    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return set()
        return set([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])

    source_ngrams = get_ngrams(source_tokens, n)
    target_ngrams = get_ngrams(target_tokens, n)
    pred_ngrams = get_ngrams(prediction_tokens, n)

    keep_target = source_ngrams & target_ngrams
    keep_pred = source_ngrams & pred_ngrams

    delete_target = source_ngrams - target_ngrams
    delete_pred = source_ngrams - pred_ngrams

    add_target = target_ngrams - source_ngrams
    add_pred = pred_ngrams - source_ngrams

    def f1_score(tp, fp, fn):
        if tp + fp == 0:
            precision = 0.0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0.0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            return 0.0
        else:
            return 2 * precision * recall / (precision + recall)

    keep_f1 = f1_score(len(keep_target & keep_pred),
                       len(keep_pred - keep_target),
                       len(keep_target - keep_pred))

    delete_f1 = f1_score(len(delete_target & delete_pred),
                         len(delete_pred - delete_target),
                         len(delete_target - delete_pred))

    add_f1 = f1_score(len(add_target & add_pred),
                      len(add_pred - add_target),
                      len(add_target - add_pred))

    return keep_f1, delete_f1, add_f1


def calculate_sari_score_multi_ref(source: str, targets: List[str], prediction: str) -> float:
    """
    Calculate SARI score for a single example with multiple references
    """
    source_tokens = tokenize(source)
    target_tokens_list = [tokenize(target) for target in targets if pd.notna(target) and target.strip()]
    pred_tokens = tokenize(prediction)

    if not target_tokens_list:
        return 0.0

    f1_scores = []
    for n in range(1, 5):
        keep_f1, delete_f1, add_f1 = calculate_ngram_f1_multi_ref(source_tokens, target_tokens_list, pred_tokens, n)
        f1_scores.append((keep_f1, delete_f1, add_f1))

    avg_keep = np.mean([f1[0] for f1 in f1_scores])
    avg_delete = np.mean([f1[1] for f1 in f1_scores])
    avg_add = np.mean([f1[2] for f1 in f1_scores])

    sari = (avg_keep + avg_delete + avg_add) / 3
    return sari * 100


def calculate_sari_score_single_ref(source: str, target: str, prediction: str) -> float:
    """
    Calculate SARI score for a single example with single reference
    """
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    pred_tokens = tokenize(prediction)

    f1_scores = []
    for n in range(1, 5):
        keep_f1, delete_f1, add_f1 = calculate_ngram_f1_single_ref(source_tokens, target_tokens, pred_tokens, n)
        f1_scores.append((keep_f1, delete_f1, add_f1))

    avg_keep = np.mean([f1[0] for f1 in f1_scores])
    avg_delete = np.mean([f1[1] for f1 in f1_scores])
    avg_add = np.mean([f1[2] for f1 in f1_scores])

    sari = (avg_keep + avg_delete + avg_add) / 3
    return sari * 100


def evaluate_sari_scores_multi_ref(df):
    """
    Calculate SARI scores for the predictions dataframe with multiple references
    """
    print("\nCalculating SARI scores with multiple references...")

    simplification_cols = []
    for col in ['Simplification 1', 'Simplification 2', 'Simplification 3']:
        if col in df.columns:
            simplification_cols.append(col)

    if not simplification_cols:
        print("No 'Simplification X' columns found.")
        return None

    print(f"Using columns: {simplification_cols}")

    sari_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI with multiple refs"):
        source = row['Complex']

        targets = []
        for col in simplification_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                targets.append(str(row[col]))

        prediction = row['preds']

        if len(targets) > 1:
            sari = calculate_sari_score_multi_ref(source, targets, prediction)
        elif len(targets) == 1:
            sari = calculate_sari_score_single_ref(source, targets[0], prediction)
        else:
            sari = 0.0

        sari_scores.append(sari)

    df['sari_score'] = sari_scores

    mean_sari = np.mean(sari_scores)
    std_sari = np.std(sari_scores)
    median_sari = np.median(sari_scores)

    print(f"\n" + "=" * 60)
    print(f"SARI Score Evaluation Results (Multiple References):")
    print(f"=" * 60)
    print(f"Reference columns used: {', '.join(simplification_cols)}")
    print(f"Mean SARI Score: {mean_sari:.4f}")
    print(f"Standard Deviation: {std_sari:.4f}")
    print(f"Median SARI Score: {median_sari:.4f}")
    print(f"Min SARI Score: {min(sari_scores):.4f}")
    print(f"Max SARI Score: {max(sari_scores):.4f}")
    print(f"=" * 60)

    return {
        'mean_sari': mean_sari,
        'std_sari': std_sari,
        'median_sari': median_sari,
        'min_sari': min(sari_scores),
        'max_sari': max(sari_scores),
        'scores': sari_scores,
        'reference_columns': simplification_cols
    }


def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    df = full.tail(200).copy()

    df['chat'] = df.apply(format_chat, axis=1)

    # Generate responses
    print("Generating predictions...")
    responses = query(model, tokenizer, df['chat'].tolist())
    df['responses'] = responses

    # Extract predictions
    print("Extracting simplified text...")
    df['preds'] = df['responses'].apply(extract_score)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Calculate SARI scores with multiple references
    print("Using multi-reference SARI evaluation with all available simplification columns.")
    sari_results = evaluate_sari_scores_multi_ref(df)

    if sari_results is None:
        print("Error: Could not evaluate SARI scores.")
        return df['preds'].tolist(), None

    # Save results with SARI scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_sari_multi_ref.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with SARI scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "sari_summary_multi_ref.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"SARI Score Evaluation Results (Multiple References)\n")
        f.write(f"Model: {checkpoint}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Reference columns: {', '.join(sari_results['reference_columns'])}\n")
        f.write(f"=" * 60 + "\n")
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
    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification", checkpoint.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, sari_results = predict()