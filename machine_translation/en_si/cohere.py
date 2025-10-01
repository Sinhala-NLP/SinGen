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


def download_and_load_flores_en_si():
    """
    Downloads and loads FLORES-200 English-Sinhala dataset.
    Returns dev and devtest DataFrames.
    """
    print("Loading FLORES-200 dataset for English-Sinhala...")

    # Load the dataset - FLORES-200 from Muennighoff
    # Use hyphenated format for language pairs
    dataset = load_dataset("Muennighoff/flores200", "eng_Latn-sin_Sinh")

    # Process dev and devtest splits
    splits = {}
    for split_name in ['dev', 'devtest']:
        if split_name in dataset:
            print(f"\nProcessing {split_name} split...")
            split_data = dataset[split_name]

            # Convert to pandas DataFrame
            # FLORES-200 has 'sentence_eng_Latn' and 'sentence_sin_Sinh' columns
            df = pd.DataFrame({
                'english': split_data['sentence_eng_Latn'],
                'sinhala': split_data['sentence_sin_Sinh']
            })

            splits[split_name] = df
            print(f"{split_name.capitalize()} split size: {len(df)}")
        else:
            print(f"Warning: {split_name} split not found in dataset")

    return splits.get('dev'), splits.get('devtest')


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

        # Check if both English and Sinhala are available
        if pd.notna(row['english']) and pd.notna(row['sinhala']) and \
                str(row['english']).strip() and str(row['sinhala']).strip():
            example = {
                'english': str(row['english']),
                'sinhala': str(row['sinhala'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat_few_shot(row, few_shot_examples):
    """
    Format chat with few-shot examples for translation
    """
    task_desc = "You are an expert translator specializing in English to Sinhala translation. Translate the following English sentence (E) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ ඉංග්‍රීසි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත ඉංග්‍රීසි වාක්‍යය (E) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string
    examples_str = ""
    for i, example in enumerate(few_shot_examples, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"E: {example['english']}\n"
        examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nE: {row['english']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nE: {row['english']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} E: {row['english']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} E: {row['english']}"
        }


def format_chat(row):
    """
    Original format_chat function for zero-shot translation
    """
    task_desc = "You are an expert translator specialising in English to Sinhala translation. Translate the following English sentence (E) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ ඉංග්‍රීසි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත ඉංග්‍රීසි වාක්‍යය (E) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    if QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} E: {row['english']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} E: {row['english']}"
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
        reference = row['sinhala']
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


def predict(dev_df, devtest_df):
    """Main prediction function"""
    print(f"Dev set size: {len(dev_df)}")
    print(f"Devtest set size: {len(devtest_df)}")
    print(f"Columns: {devtest_df.columns.tolist()}")

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Dev set available for few-shot examples: {len(dev_df)}")
        print(f"Test instances (devtest): {len(devtest_df)}")

        # Apply few-shot formatting with dynamic example selection per instance
        chat_messages = []
        for idx, (test_idx, row) in enumerate(
                tqdm(devtest_df.iterrows(), total=len(devtest_df), desc="Preparing few-shot prompts")):
            # Get unique few-shot examples for this specific test instance from dev set
            few_shot_examples = get_few_shot_examples_for_instance(
                dev_df,
                instance_idx=idx,
                num_examples=3,
                seed=42  # Base seed for reproducibility
            )

            # Format the chat with these examples
            chat_message = format_chat_few_shot(row, few_shot_examples)
            chat_messages.append(chat_message)

        devtest_df['chat'] = chat_messages
        print(f"Each test instance has been assigned unique few-shot examples")
    else:
        # Use zero-shot formatting
        devtest_df['chat'] = devtest_df.apply(format_chat, axis=1)

    # Generate responses
    print("Generating translations...")
    responses = query_cohere(co, model_id, devtest_df['chat'].tolist())
    devtest_df['responses'] = responses

    # Extract predictions
    print("Extracting translations...")
    devtest_df['preds'] = devtest_df['responses'].apply(extract_translation)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    devtest_df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Evaluate with BLEU score
    print("Evaluating translations with BLEU score...")
    bleu_results = evaluate_bleu_scores(devtest_df)

    # Save results with BLEU scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_bleu.csv")
    devtest_df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with BLEU scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "bleu_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset: FLORES-200 English-Sinhala (devtest split)\n")
        f.write(f"Dataset Size: {len(devtest_df)} samples\n")
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

    return devtest_df['preds'].tolist(), bleu_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='Type of query: zero-shot, zero-shot-si, few-shot, few-shot-si')

    args = parser.parse_args()

    QUERY_TYPE = args.query_type

    print(f"Query type: {QUERY_TYPE}")

    # Load datasets from HuggingFace
    dev_df, devtest_df = download_and_load_flores_en_si()

    if dev_df is None or devtest_df is None:
        print("Error: Could not load required dataset splits")
        exit(1)

    # Set up Cohere client
    COHERE_API_KEY = "<<your-api-key>>"  # Replace with your actual key
    co = cohere.ClientV2(COHERE_API_KEY)

    model_id = "command-a-03-2025"

    OUTPUT_FOLDER = os.path.join("outputs", "english_sinhala_translation", model_id, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict(dev_df, devtest_df)