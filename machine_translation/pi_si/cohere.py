import argparse
import os
import re
from typing import List
import random

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cohere
from datasets import load_dataset


def get_few_shot_examples_for_instance(full_df, test_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance
    Each test instance will get different randomly selected examples
    """
    # Get indices of test instances
    test_indices = set(test_df.index)

    # Get available examples (excluding test instances)
    available_indices = [i for i in full_df.index if i not in test_indices]

    # Use instance-specific seed for randomization
    if seed is not None:
        random.seed(seed + instance_idx)

    # Randomly sample few-shot examples for this specific instance
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = full_df.loc[idx]

        # Check if both Pali and Sinhala are available
        if pd.notna(row['pali_text']) and pd.notna(row['sinhala_text']) and \
                str(row['pali_text']).strip() and str(row['sinhala_text']).strip():
            example = {
                'pali': str(row['pali_text']),
                'sinhala': str(row['sinhala_text'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat_few_shot(row, few_shot_examples):
    """
    Format chat with few-shot examples for translation
    """
    task_desc = "You are an expert translator specializing in Pali to Sinhala translation. Translate the following Pali text (P) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ පාලි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාලි පාඨය (P) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string
    examples_str = ""
    for i, example in enumerate(few_shot_examples, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"P: {example['pali']}\n"
        examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this text:\nP: {row['pali_text']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පාඨය පරිවර්තනය කරන්න:\nP: {row['pali_text']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} P: {row['pali_text']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} P: {row['pali_text']}"
        }


def format_chat(row):
    """
    Original format_chat function for zero-shot translation
    """
    task_desc = "You are an expert translator specialising in Pali to Sinhala translation. Translate the following Pali text (P) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ පාලි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාලි පාඨය (P) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} P: {row['pali_text']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} P: {row['pali_text']}"
        }


def query_cohere(client, model, messages):
    outputs = []
    for msg in tqdm(messages, desc="Generating translations"):
        try:
            response = client.chat(
                model=model,
                messages=[msg],
                temperature=0.3,
            )

            # content is a list of content items; extract all `text` fields and join them
            content_items = response.message.content
            text_parts = [c.text for c in content_items if c.type == "text"]
            full_text = " ".join(text_parts).strip()

            outputs.append(full_text)
        except Exception as e:
            print(f"Error with message: {e}")
            outputs.append("")
    return outputs


def extract_translation(response):
    """Extract translation from model response"""
    if not isinstance(response, str):
        print("Non-string response:", response)
        return ""
    matches = re.findall(r'Translation:\s*(.*)', response)
    return matches[0] if matches else response.strip()


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


def load_and_prepare_dataset():
    """Load dataset from Hugging Face and prepare it"""
    print("Loading dataset from Hugging Face...")
    ds = load_dataset("sinhala-nlp/pali-sinhala")
    df = ds['train'].to_pandas()

    # Remove leading numbers from both columns
    print("Cleaning dataset (removing leading numbers)...")
    df['pali_text'] = df['pali_text'].str.replace(r'^\d+\s+', '', regex=True)
    df['sinhala_text'] = df['sinhala_text'].str.replace(r'^\d+\s+', '', regex=True)

    return df


def predict():
    """Main prediction function"""
    # Load and prepare the dataset
    full_df = load_and_prepare_dataset()

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    # Use last 1000 samples for testing
    test_size = min(1000, len(full_df))
    df = full_df.tail(test_size).copy()
    print(f"Using last {test_size} samples for testing")

    # Get the rest of the instances (excluding the tail)
    rest_of_instances = full_df.head(len(full_df) - test_size)

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Total dataset size: {len(full_df)}")
        print(f"Test instances: {len(df)}")
        print(f"Available for few-shot examples: {len(rest_of_instances)}")

        # Apply few-shot formatting with dynamic example selection per instance
        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            # Get unique few-shot examples for this specific test instance
            few_shot_examples = get_few_shot_examples_for_instance(
                rest_of_instances,
                df,
                instance_idx=idx,
                num_examples=3,
                seed=42  # Base seed for reproducibility
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
    print("Generating translations...")
    responses = query_cohere(co, model_id, df['chat'].tolist())
    df['responses'] = responses

    # Extract predictions
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
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset: sinhala-nlp/pali-sinhala\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write(f"Few-shot approach: Dynamic (unique examples per test instance)\n")
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
    print(f"Dataset: sinhala-nlp/pali-sinhala (from Hugging Face)")

    # Set up Cohere client
    COHERE_API_KEY = "<<your-api-key>>"  # Replace with your actual key
    co = cohere.ClientV2(COHERE_API_KEY)

    model_id = "command-a-03-2025"

    OUTPUT_FOLDER = os.path.join("outputs", "pali_sinhala_translation", model_id, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict()