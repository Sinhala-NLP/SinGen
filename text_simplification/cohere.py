import argparse
import os
import re
from typing import List, Tuple
import random

import numpy as np
import pandas as pd
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
import cohere


def get_few_shot_examples(full_df, test_df, num_examples=3):
    """
    Get random few-shot examples from the dataset excluding test instances
    """
    # Get indices of test instances
    test_indices = set(test_df.index)

    # Get available examples (excluding test instances)
    available_indices = [i for i in full_df.index if i not in test_indices]

    # Randomly sample few-shot examples
    random.seed(42)  # For reproducibility
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = full_df.loc[idx]
        example = {
            'complex': row['Complex'],
            'simple': row['Simple']
        }
        few_shot_examples.append(example)

    return few_shot_examples


def format_chat_few_shot(row, few_shot_examples):
    """
    Format chat with few-shot examples
    """
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න (3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න."
    action_desc_si = "'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string
    examples_str = ""
    for i, example in enumerate(few_shot_examples, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"S: {example['complex']}\n"
        examples_str += f"Simplified text: {example['simple']}\n"

    if QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow simplify this sentence:\nS: {row['Complex']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය සරල කරන්න:\nS: {row['Complex']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} S: {row['Complex']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} S: {row['Complex']}"
        }


def format_chat(row):
    """
    Original format_chat function for backward compatibility
    """
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


def query_cohere(client, model, messages):
    outputs = []
    for msg in tqdm(messages, desc="Generating predictions"):
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
    # Simple tokenization - you might want to use a proper Sinhala tokenizer
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

    # KEEP: n-grams that should be kept from source (present in source AND in at least one target)
    keep_target = set()
    for target_tokens in target_tokens_list:
        target_ngrams = get_ngrams(target_tokens, n)
        keep_target.update(source_ngrams & target_ngrams)

    keep_pred = source_ngrams & pred_ngrams

    # DELETE: n-grams that should be deleted from source (present in source but NOT in any target)
    delete_target = source_ngrams - all_target_ngrams
    delete_pred = source_ngrams - pred_ngrams

    # ADD: n-grams that should be added (not in source but present in at least one target)
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

    # Calculate F1 scores
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

    # If no valid targets, return 0
    if not target_tokens_list:
        return 0.0

    # Calculate F1 scores for unigrams, bigrams, trigrams, and 4-grams
    f1_scores = []
    for n in range(1, 5):
        keep_f1, delete_f1, add_f1 = calculate_ngram_f1_multi_ref(source_tokens, target_tokens_list, pred_tokens, n)
        f1_scores.append((keep_f1, delete_f1, add_f1))

    # Average across n-gram orders
    avg_keep = np.mean([f1[0] for f1 in f1_scores])
    avg_delete = np.mean([f1[1] for f1 in f1_scores])
    avg_add = np.mean([f1[2] for f1 in f1_scores])

    # SARI score is the average of KEEP, DELETE, and ADD F1 scores
    sari = (avg_keep + avg_delete + avg_add) / 3
    return sari * 100  # Convert to percentage


def calculate_ngram_f1(source_tokens: List[str],
                       target_tokens: List[str],
                       prediction_tokens: List[str],
                       n: int) -> Tuple[float, float, float]:
    """
    Calculate n-gram F1 scores for KEEP, DELETE, and ADD operations (single reference)
    """

    def get_ngrams(tokens, n):
        if len(tokens) < n:
            return set()
        return set([tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)])

    source_ngrams = get_ngrams(source_tokens, n)
    target_ngrams = get_ngrams(target_tokens, n)
    pred_ngrams = get_ngrams(prediction_tokens, n)

    # KEEP: n-grams that should be kept from source
    keep_target = source_ngrams & target_ngrams
    keep_pred = source_ngrams & pred_ngrams

    # DELETE: n-grams that should be deleted from source
    delete_target = source_ngrams - target_ngrams
    delete_pred = source_ngrams - pred_ngrams

    # ADD: n-grams that should be added (not in source)
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

    # Calculate F1 scores
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


def calculate_sari_score(source: str, target: str, prediction: str) -> float:
    """
    Calculate SARI score for a single example (single reference)
    """
    source_tokens = tokenize(source)
    target_tokens = tokenize(target)
    pred_tokens = tokenize(prediction)

    # Calculate F1 scores for unigrams, bigrams, trigrams, and 4-grams
    f1_scores = []
    for n in range(1, 5):
        keep_f1, delete_f1, add_f1 = calculate_ngram_f1(source_tokens, target_tokens, pred_tokens, n)
        f1_scores.append((keep_f1, delete_f1, add_f1))

    # Average across n-gram orders
    avg_keep = np.mean([f1[0] for f1 in f1_scores])
    avg_delete = np.mean([f1[1] for f1 in f1_scores])
    avg_add = np.mean([f1[2] for f1 in f1_scores])

    # SARI score is the average of KEEP, DELETE, and ADD F1 scores
    sari = (avg_keep + avg_delete + avg_add) / 3
    return sari * 100  # Convert to percentage


def evaluate_sari_scores_multi_ref(df):
    """
    Calculate SARI scores for the predictions dataframe with multiple references
    """
    print("\nCalculating SARI scores with multiple references...")

    # Check which simplification columns exist
    simplification_cols = []
    for col in ['Simplification 1', 'Simplification 2', 'Simplification 3']:
        if col in df.columns:
            simplification_cols.append(col)

    if not simplification_cols:
        # Fallback to original single reference if no Simplification columns found
        print("No 'Simplification X' columns found. Using 'Simple' column as single reference.")
        simplification_cols = ['Simple']

    print(f"Using columns: {simplification_cols}")

    sari_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI with multiple refs"):
        source = row['Complex']

        # Get all available simplifications for this row
        targets = []
        for col in simplification_cols:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                targets.append(str(row[col]))

        prediction = row['preds']

        if len(targets) > 1:
            sari = calculate_sari_score_multi_ref(source, targets, prediction)
        elif len(targets) == 1:
            sari = calculate_sari_score(source, targets[0], prediction)
        else:
            sari = 0.0

        sari_scores.append(sari)

    df['sari_score'] = sari_scores

    # Calculate overall statistics
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


def evaluate_sari_scores(df):
    """
    Calculate SARI scores for the predictions dataframe (single reference)
    """
    print("\nCalculating SARI scores...")

    sari_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI"):
        source = row['Complex']
        target = row['Simple']
        prediction = row['preds']

        sari = calculate_sari_score(source, target, prediction)
        sari_scores.append(sari)

    df['sari_score'] = sari_scores

    # Calculate overall statistics
    mean_sari = np.mean(sari_scores)
    std_sari = np.std(sari_scores)
    median_sari = np.median(sari_scores)

    print(f"\n" + "=" * 50)
    print(f"SARI Score Evaluation Results:")
    print(f"=" * 50)
    print(f"Mean SARI Score: {mean_sari:.4f}")
    print(f"Standard Deviation: {std_sari:.4f}")
    print(f"Median SARI Score: {median_sari:.4f}")
    print(f"Min SARI Score: {min(sari_scores):.4f}")
    print(f"Max SARI Score: {max(sari_scores):.4f}")
    print(f"=" * 50)

    return {
        'mean_sari': mean_sari,
        'std_sari': std_sari,
        'median_sari': median_sari,
        'min_sari': min(sari_scores),
        'max_sari': max(sari_scores),
        'scores': sari_scores
    }


def predict():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    df = full.tail(200)

    # Get the rest of the instances (excluding the tail 200)
    rest_of_instances = full.head(len(full) - 200)

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting few-shot examples...")
        print(f"Total dataset size: {len(full)}")
        print(f"Test instances (tail 200): {len(df)}")
        print(f"Available for few-shot examples: {len(rest_of_instances)}")
        few_shot_examples = get_few_shot_examples(rest_of_instances, df, num_examples=3)
        print(f"Selected {len(few_shot_examples)} few-shot examples")

        # Print the examples for verification
        print("\nFew-shot examples:")
        for i, example in enumerate(few_shot_examples, 1):
            print(f"Example {i}:")
            print(f"  Complex: {example['complex'][:100]}...")
            print(f"  Simple: {example['simple'][:100]}...")
            print()

        # Apply few-shot formatting
        df['chat'] = df.apply(lambda row: format_chat_few_shot(row, few_shot_examples), axis=1)
    else:
        # Use original zero-shot formatting
        df['chat'] = df.apply(format_chat, axis=1)

    # Generate responses
    print("Generating predictions...")
    responses = query_cohere(co, model_id, df['chat'].tolist())
    df['responses'] = responses

    # Extract predictions
    print("Extracting simplified text...")
    df['preds'] = df['responses'].apply(extract_score)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Check if we have multiple references and use appropriate evaluation
    has_multiple_refs = any(col in df.columns for col in ['Simplification 1', 'Simplification 2', 'Simplification 3'])

    if has_multiple_refs:
        print("Multiple reference columns detected. Using multi-reference SARI evaluation.")
        sari_results = evaluate_sari_scores_multi_ref(df)
        results_filename = "predictions_with_sari_multi_ref.csv"
        summary_filename = "sari_summary_multi_ref.txt"
    else:
        print("Single reference detected. Using single-reference SARI evaluation.")
        sari_results = evaluate_sari_scores(df)
        results_filename = "predictions_with_sari.csv"
        summary_filename = "sari_summary.txt"

    # Save results with SARI scores
    results_file = os.path.join(OUTPUT_FOLDER, results_filename)
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with SARI scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, summary_filename)
    with open(summary_file, 'w', encoding='utf-8') as f:
        if has_multiple_refs:
            f.write(f"SARI Score Evaluation Results (Multiple References)\n")
            f.write(f"Reference columns: {', '.join(sari_results.get('reference_columns', []))}\n")
        else:
            f.write(f"SARI Score Evaluation Results\n")

        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
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
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='Type of query: zero-shot, zero-shot-si, few-shot, few-shot-si')
    args = parser.parse_args()
    QUERY_TYPE = args.query_type
    print(f"Query type: {QUERY_TYPE}")

    # Set up Cohere client
    COHERE_API_KEY = "<<your-api-key>>"  # Replace with your actual key
    co = cohere.ClientV2(COHERE_API_KEY)

    model_id = "command-a-03-2025"  # or "command-a-03-2025" as per your need

    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification", model_id, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, sari_results = predict()