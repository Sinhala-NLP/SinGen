import argparse
import os
import re
from typing import List, Tuple
import random

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


def get_few_shot_examples_for_instance(train_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance
    Each test instance will get different randomly selected examples
    """
    # Use instance-specific seed for randomization
    if seed is not None:
        random.seed(seed + instance_idx)

    # Randomly sample few-shot examples for this specific instance
    few_shot_indices = random.sample(range(len(train_df)), min(num_examples, len(train_df)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = train_df.iloc[idx]

        # Skip if news content or headline is missing
        if pd.notna(row['News Content']) and pd.notna(row['Headline']) and \
                str(row['News Content']).strip() and str(row['Headline']).strip():
            example = {
                'content': str(row['News Content']),
                'headline': str(row['Headline'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat_few_shot(row, few_shot_examples):
    """
    Format chat with few-shot examples for headline generation
    """
    task_desc = "You are an expert in Sinhala journalism. Generate a concise and informative headline for the following Sinhala news article. The headline should capture the main point of the article in a brief, engaging manner."
    action_desc = "Return only the headline following the prefix 'Headline:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල පුවත්පත් කලාවේ ප්‍රවීණයෙකු වන්න. පහත සිංහල පුවත් ලිපිය සඳහා සංක්ෂිප්ත හා තොරතුරුදායක සිරස්තලයක් ජනනය කරන්න. සිරස්තලය කෙටි, ආකර්ෂණීය ආකාරයෙන් ලිපියේ ප්‍රධාන කරුණ ග්‍රහණය කර ගත යුතුය."
    action_desc_si = "'Headline:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිරස්තලය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string
    examples_str = ""
    for i, example in enumerate(few_shot_examples, 1):
        # Truncate content if too long for context
        content_preview = example['content'][:500] + "..." if len(example['content']) > 500 else example['content']
        examples_str += f"\nExample {i}:\n"
        examples_str += f"News Content: {content_preview}\n"
        examples_str += f"Headline: {example['headline']}\n"

    if QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow generate a headline for this news article:\nNews Content: {row['News Content']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පුවත් ලිපිය සඳහා සිරස්තලයක් ජනනය කරන්න:\nNews Content: {row['News Content']}"
        return {
            "role": "user",
            "content": prompt
        }


def format_chat(row):
    """
    Original format_chat function for zero-shot headline generation
    """
    task_desc = "You are an expert in Sinhala journalism. Generate a concise and informative headline for the following Sinhala news article. The headline should capture the main point of the article in a brief, engaging manner."
    action_desc = "Return only the headline following the prefix 'Headline:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල පුවත්පත් කලාවේ ප්‍රවීණයෙකු වන්න. පහත සිංහල පුවත් ලිපිය සඳහා සංක්ෂිප්ත හා තොරතුරුදායක සිරස්තලයක් ජනනය කරන්න. සිරස්තලය කෙටි, ආකර්ෂණීය ආකාරයෙන් ලිපියේ ප්‍රධාන කරුණ ග්‍රහණය කර ගත යුතුය."
    action_desc_si = "'Headline:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිරස්තලය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} News Content: {row['News Content']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} News Content: {row['News Content']}"
        }


def query(model, tokenizer, inputs):
    outputs = []

    for inp in tqdm(inputs, desc="Generating headlines"):
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


def extract_headline(response):
    if not isinstance(response, str):
        print("Non-string response:", response)
        return ""
    matches = re.findall(r'Headline:\s*(.*)', response, re.DOTALL)
    return matches[0].strip() if matches else response.strip()


# ROUGE Evaluation Functions
def tokenize(text):
    """Simple tokenization by splitting on whitespace"""
    if pd.isna(text) or text is None:
        return []
    text = str(text).lower()
    tokens = re.findall(r'\b\w+\b', text)
    return tokens


def calculate_rouge_n(reference_tokens, prediction_tokens, n=1):
    """
    Calculate ROUGE-N score
    """

    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return []
        return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    ref_ngrams = get_ngrams(reference_tokens, n)
    pred_ngrams = get_ngrams(prediction_tokens, n)

    if len(ref_ngrams) == 0:
        return 0.0, 0.0, 0.0

    ref_ngram_counts = {}
    for ngram in ref_ngrams:
        ref_ngram_counts[ngram] = ref_ngram_counts.get(ngram, 0) + 1

    pred_ngram_counts = {}
    for ngram in pred_ngrams:
        pred_ngram_counts[ngram] = pred_ngram_counts.get(ngram, 0) + 1

    overlapping_ngrams = 0
    for ngram, count in pred_ngram_counts.items():
        if ngram in ref_ngram_counts:
            overlapping_ngrams += min(count, ref_ngram_counts[ngram])

    if len(pred_ngrams) == 0:
        precision = 0.0
    else:
        precision = overlapping_ngrams / len(pred_ngrams)

    recall = overlapping_ngrams / len(ref_ngrams)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def calculate_rouge_l(reference_tokens, prediction_tokens):
    """
    Calculate ROUGE-L score using Longest Common Subsequence
    """

    def lcs_length(x, y):
        m, n = len(x), len(y)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        return dp[m][n]

    if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
        return 0.0, 0.0, 0.0

    lcs_len = lcs_length(reference_tokens, prediction_tokens)

    precision = lcs_len / len(prediction_tokens) if len(prediction_tokens) > 0 else 0.0
    recall = lcs_len / len(reference_tokens) if len(reference_tokens) > 0 else 0.0

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return precision, recall, f1


def evaluate_rouge_scores(df):
    """
    Calculate ROUGE scores for the predictions dataframe
    """
    print("\nCalculating ROUGE scores...")

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing ROUGE"):
        reference = str(row['Headline'])
        prediction = str(row['preds'])

        ref_tokens = tokenize(reference)
        pred_tokens = tokenize(prediction)

        # ROUGE-1
        _, _, rouge1_f1 = calculate_rouge_n(ref_tokens, pred_tokens, n=1)
        rouge1_scores.append(rouge1_f1)

        # ROUGE-2
        _, _, rouge2_f1 = calculate_rouge_n(ref_tokens, pred_tokens, n=2)
        rouge2_scores.append(rouge2_f1)

        # ROUGE-L
        _, _, rougeL_f1 = calculate_rouge_l(ref_tokens, pred_tokens)
        rougeL_scores.append(rougeL_f1)

    df['rouge1_score'] = rouge1_scores
    df['rouge2_score'] = rouge2_scores
    df['rougeL_score'] = rougeL_scores

    # Calculate overall statistics
    results = {
        'rouge1': {
            'mean': np.mean(rouge1_scores),
            'std': np.std(rouge1_scores),
            'median': np.median(rouge1_scores),
            'min': np.min(rouge1_scores),
            'max': np.max(rouge1_scores)
        },
        'rouge2': {
            'mean': np.mean(rouge2_scores),
            'std': np.std(rouge2_scores),
            'median': np.median(rouge2_scores),
            'min': np.min(rouge2_scores),
            'max': np.max(rouge2_scores)
        },
        'rougeL': {
            'mean': np.mean(rougeL_scores),
            'std': np.std(rougeL_scores),
            'median': np.median(rougeL_scores),
            'min': np.min(rougeL_scores),
            'max': np.max(rougeL_scores)
        }
    }

    print(f"\n" + "=" * 60)
    print(f"ROUGE Score Evaluation Results:")
    print(f"=" * 60)
    print(f"ROUGE-1:")
    print(f"  Mean: {results['rouge1']['mean']:.4f}")
    print(f"  Std:  {results['rouge1']['std']:.4f}")
    print(f"ROUGE-2:")
    print(f"  Mean: {results['rouge2']['mean']:.4f}")
    print(f"  Std:  {results['rouge2']['std']:.4f}")
    print(f"ROUGE-L:")
    print(f"  Mean: {results['rougeL']['mean']:.4f}")
    print(f"  Std:  {results['rougeL']['std']:.4f}")
    print(f"=" * 60)

    return results


def predict():
    # Load the dataset
    print("Loading NSINA-Headlines dataset...")
    ds = load_dataset("sinhala-nlp/NSINA-Headlines")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Columns: {test_df.columns.tolist()}")

    # Filter out rows with missing News Content or Headline
    test_df = test_df[test_df['News Content'].notna() & test_df['Headline'].notna()].copy()
    train_df = train_df[train_df['News Content'].notna() & train_df['Headline'].notna()].copy()

    print(f"After filtering - Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Use first 1000 test samples
    test_size = min(1000, len(test_df))
    df = test_df.head(test_size).copy()
    print(f"Using {len(df)} test samples")

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Available training examples: {len(train_df)}")

        # Apply few-shot formatting with dynamic example selection per instance
        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            # Get unique few-shot examples for this specific test instance
            few_shot_examples = get_few_shot_examples_for_instance(
                train_df,
                instance_idx=idx,
                num_examples=3,
                seed=777  # Base seed for reproducibility (matching set_seed)
            )

            # Format the chat with these examples
            chat_message = format_chat_few_shot(row, few_shot_examples)
            chat_messages.append(chat_message)

        df['chat'] = chat_messages
        print(f"Each test instance has been assigned unique few-shot examples")
    else:
        # Use zero-shot formatting
        df['chat'] = df.apply(format_chat, axis=1)

    # Generate responses
    print("Generating headlines...")
    responses = query(model, tokenizer, df['chat'].tolist())
    df['responses'] = responses

    # Extract headlines
    print("Extracting headlines...")
    df['preds'] = df['responses'].apply(extract_headline)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Evaluate with ROUGE
    print("Evaluating with ROUGE metrics...")
    rouge_results = evaluate_rouge_scores(df)

    # Save results with ROUGE scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_rouge.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with ROUGE scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "rouge_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"ROUGE Score Evaluation Results\n")
        f.write(f"Model: {checkpoint}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset: NSINA-Headlines\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write(f"Few-shot approach: Dynamic (unique examples per test instance)\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"ROUGE-1:\n")
        f.write(f"  Mean: {rouge_results['rouge1']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rouge1']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rouge1']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rouge1']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rouge1']['max']:.4f}\n")
        f.write(f"\nROUGE-2:\n")
        f.write(f"  Mean: {rouge_results['rouge2']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rouge2']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rouge2']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rouge2']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rouge2']['max']:.4f}\n")
        f.write(f"\nROUGE-L:\n")
        f.write(f"  Mean: {rouge_results['rougeL']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rougeL']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rougeL']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rougeL']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rougeL']['max']:.4f}\n")

    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), rouge_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='Type of query: zero-shot, zero-shot-si, few-shot, few-shot-si')
    args = parser.parse_args()
    QUERY_TYPE = args.query_type
    print(f"Query type: {QUERY_TYPE}")

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "headline_generation", checkpoint.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, rouge_results = predict()