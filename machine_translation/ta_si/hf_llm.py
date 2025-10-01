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

        # Check if both Tamil and Sinhala are available
        if pd.notna(row['Tamil']) and pd.notna(row['Sinhala']) and \
                str(row['Tamil']).strip() and str(row['Sinhala']).strip():
            example = {
                'tamil': str(row['Tamil']),
                'sinhala': str(row['Sinhala'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string if provided
    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"T: {example['tamil']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} T: {row['Tamil']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} T: {row['Tamil']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nT: {row['Tamil']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nT: {row['Tamil']}"
        return [{"role": "user", "content": prompt}]

    else:
        # Default fallback
        return [{"role": "user", "content": f"{task_desc} {action_desc} T: {row['Tamil']}"}]


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
    TSV_FILE = os.path.join("machine_translation", "ta_si", "ta_si.tsv")

    print(f"Query type: {QUERY_TYPE}")
    print(f"TSV file: {TSV_FILE}")

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "tamil_sinhala_translation", model_id.split('/')[-1], QUERY_TYPE)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    predictions, bleu_results = predict(TSV_FILE)