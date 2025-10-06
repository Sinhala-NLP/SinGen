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
    test_indices = set(test_df.index)
    available_indices = [i for i in full_df.index if i not in test_indices]

    # Use instance-specific seed for randomization
    if seed is not None:
        random.seed(seed + instance_idx)

    simplification_columns = ['Simplification 1', 'Simplification 2', 'Simplification 3']
    few_shot_examples = []

    # Shuffle available indices for this specific instance
    shuffled_indices = available_indices.copy()
    random.shuffle(shuffled_indices)

    for idx in shuffled_indices:
        if len(few_shot_examples) >= num_examples:
            break

        row = full_df.loc[idx]
        available_simplifications = []

        for col in simplification_columns:
            if col in row and pd.notna(row[col]) and str(row[col]).strip():
                available_simplifications.append(str(row[col]))

        if available_simplifications:
            selected_simple = random.choice(available_simplifications)
            few_shot_examples.append({
                'complex': row['Complex'],
                'simple': selected_simple
            })

    return few_shot_examples


def format_chat(row, few_shot_examples=None):
    task_desc = "Imagine you are an expert in Sinhala language. Please provide a simplified version of the following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, and replacing complex words with commonly used simple words."
    action_desc = "Return the simplified text only following the prefix 'Simplified text:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න (3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න."
    action_desc_si = "'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string if provided
    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"S: {example['complex']}\n"
            examples_str += f"Simplified text: {example['simple']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} S: {row['Complex']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} S: {row['Complex']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow simplify this sentence:\nS: {row['Complex']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය සරල කරන්න:\nS: {row['Complex']}"
        return [{"role": "user", "content": prompt}]

    else:
        # Default fallback
        return [{"role": "user", "content": f"{task_desc} {action_desc} S: {row['Complex']}"}]


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


def extract_score(response):
    """
    Extract simplified text from model response
    """
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    try:
        # Look for the "Simplified text:" pattern
        matches = re.findall(r'Simplified text:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

        # Fallback: try without strict matching
        if "simplified text:" in response.lower():
            parts = response.lower().split("simplified text:")
            if len(parts) > 1:
                return parts[1].strip()

        # Last resort: return the response as is
        return response.strip()
    except Exception as e:
        print(f"Error extracting text: {e}")
        return ""


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

    # Get the rest of the instances (excluding the tail 200)
    rest_of_instances = full.head(len(full) - 200)

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Total dataset size: {len(full)}")
        print(f"Test instances (tail 200): {len(df)}")
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
    print("Generating predictions...")
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # Extract predictions
    print("Extracting simplified text...")
    df['preds'] = df.apply(lambda row: extract_score(row['responses']), axis=1)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Calculate SARI scores with multiple references
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
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Few-shot approach: Dynamic (unique examples per test instance)\n")
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
    OUTPUT_FOLDER = os.path.join("outputs", "text_summarisation", model_id.split('/')[-1], QUERY_TYPE)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    predictions, sari_results = predict()