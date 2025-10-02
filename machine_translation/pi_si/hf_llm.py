import argparse
import os
import re
from typing import List
import random

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline, set_seed
from datasets import load_dataset

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


def load_and_prepare_pali_sinhala_dataset():
    """
    Loads Pali-Sinhala dataset from HuggingFace.
    Returns full dataframe with cleaned text.
    """
    print("Loading Pali-Sinhala dataset from HuggingFace...")

    # Load the dataset
    ds = load_dataset("sinhala-nlp/pali-sinhala")
    full_df = ds['train'].to_pandas()

    # Remove leading numbers from both columns
    print("Cleaning dataset (removing leading numbers)...")
    full_df['pali_text'] = full_df['pali_text'].str.replace(r'^\d+\s+', '', regex=True)
    full_df['sinhala_text'] = full_df['sinhala_text'].str.replace(r'^\d+\s+', '', regex=True)

    print(f"Total dataset size: {len(full_df)}")

    return full_df


def split_dataset(full_df, test_size=1000):
    """
    Split dataset into dev (for few-shot examples) and test sets.
    Uses last test_size samples for testing, rest for dev/few-shot examples.
    """
    test_size = min(test_size, len(full_df))

    # Use last samples for testing
    test_df = full_df.tail(test_size).copy()

    # Use the rest for dev/few-shot examples
    dev_df = full_df.head(len(full_df) - test_size).copy()

    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")

    return dev_df, test_df


def get_few_shot_examples_for_instance(dev_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance from dev set
    Each test instance will get different randomly selected examples
    """
    # Use instance-specific seed for randomization
    if seed is not None:
        random.seed(seed + instance_idx)

    # Get available examples from dev set
    available_indices = list(dev_df.index)

    # Randomly sample few-shot examples for this specific instance
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = dev_df.loc[idx]

        # Check if both Pali and Sinhala are available
        if pd.notna(row['pali_text']) and pd.notna(row['sinhala_text']) and \
                str(row['pali_text']).strip() and str(row['sinhala_text']).strip():
            example = {
                'pali': str(row['pali_text']),
                'sinhala': str(row['sinhala_text'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in Pali to Sinhala translation. Translate the following Pali text (P) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ පාලි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාලි පාඨය (P) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string if provided
    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"P: {example['pali']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} P: {row['pali_text']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} P: {row['pali_text']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this text:\nP: {row['pali_text']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පාඨය පරිවර්තනය කරන්න:\nP: {row['pali_text']}"
        return [{"role": "user", "content": prompt}]

    else:
        # Default fallback
        return [{"role": "user", "content": f"{task_desc} {action_desc} P: {row['pali_text']}"}]


def query(pipe, inputs):
    """
    :param pipe: text-generation pipeline
    :param inputs: list of messages
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
            eos_token_id=terminators,
            pad_token_id=pipe.tokenizer.eos_token_id
    )):
        assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())

    return assistant_outputs


def extract_translation(response):
    """Extract translation from model response"""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    try:
        # Look for the "Translation:" pattern
        matches = re.findall(r'Translation:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: try without strict matching
        if "translation:" in response.lower():
            parts = response.lower().split("translation:")
            if len(parts) > 1:
                return parts[1].strip()

        # Last resort: return the response as is
        return response.strip()
    except Exception as e:
        print(f"Error extracting translation: {e}")
        return ""


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
        reference = row['sinhala_text']
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


def predict(dev_df, test_df):
    """Main prediction function"""
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Columns: {test_df.columns.tolist()}")

    # Use the test set for testing
    df = test_df.copy()

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Dev set available for few-shot examples: {len(dev_df)}")
        print(f"Test instances: {len(df)}")

        # Apply few-shot formatting with dynamic example selection per instance
        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            # Get unique few-shot examples for this specific test instance from dev set
            few_shot_examples = get_few_shot_examples_for_instance(
                dev_df,
                instance_idx=idx,
                num_examples=3,
                seed=42  # Base seed for reproducibility
            )

            # Format the chat with these examples
            chat_message = format_chat(row, few_shot_examples)
            chat_messages.append(chat_message)

        df['chat'] = chat_messages
        print(f"Each test instance has been assigned unique few-shot examples")
    else:
        # Use zero-shot formatting
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    # Generate responses
    print("Generating translations...")
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # Extract predictions
    print("Extracting translations...")
    df['preds'] = df.apply(lambda row: extract_translation(row['responses']), axis=1)

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
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset: sinhala-nlp/pali-sinhala\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write(f"Few-shot approach: Dynamic (unique examples per test instance from dev set)\n")
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
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='Type of query: zero-shot, zero-shot-si, few-shot, few-shot-si')

    args = parser.parse_args()

    QUERY_TYPE = args.query_type

    print(f"Query type: {QUERY_TYPE}")

    # Load and split dataset (test size fixed at 1000)
    full_df = load_and_prepare_pali_sinhala_dataset()
    dev_df, test_df = split_dataset(full_df, test_size=1000)

    if dev_df is None or test_df is None or len(test_df) == 0:
        print("Error: Could not load or split dataset properly")
        exit(1)

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "pali_sinhala_translation", model_id.split('/')[-1], QUERY_TYPE)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    predictions, bleu_results = predict(dev_df, test_df)