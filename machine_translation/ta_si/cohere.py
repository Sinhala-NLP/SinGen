import argparse
import os
import re
from typing import List
import random

import numpy as np
import pandas as pd
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

        # Check if both Tamil and Sinhala are available
        if pd.notna(row['Tamil']) and pd.notna(row['Sinhala']) and \
                str(row['Tamil']).strip() and str(row['Sinhala']).strip():
            example = {
                'tamil': str(row['Tamil']),
                'sinhala': str(row['Sinhala'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat_few_shot(row, few_shot_examples):
    """
    Format chat with few-shot examples for translation
    """
    task_desc = "You are an expert translator specializing in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු වන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string
    examples_str = ""
    for i, example in enumerate(few_shot_examples, 1):
        examples_str += f"\nExample {i}:\n"
        examples_str += f"T: {example['tamil']}\n"
        examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nT: {row['Tamil']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nT: {row['Tamil']}"
        return {
            "role": "user",
            "content": prompt
        }
    elif QUERY_TYPE == "zero-shot":
        return {
            "role": "user",
            "content": f"{task_desc} {action_desc} T: {row['Tamil']}"
        }
    elif QUERY_TYPE == "zero-shot-si":
        return {
            "role": "user",
            "content": f"{task_desc_si} {action_desc_si} T: {row['Tamil']}"
        }


def format_chat(row):
    """
    Original format_chat function for zero-shot translation
    """
    task_desc = "You are an expert translator specialising in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු වන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
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


def calculate_bleu_score(reference: str, prediction: str, max_n: int = 4) -> float:
    """
    Calculate BLEU score for a single translation
    Simplified implementation focusing on n-gram precision
    """
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not pred_tokens or not ref_tokens:
        return 0.0

    # Calculate brevity penalty
    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

    # Calculate n-gram precisions
    precisions = []
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

    # Calculate geometric mean of precisions
    if min(precisions) > 0:
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    # BLEU score
    bleu = bp * geo_mean
    return bleu * 100  # Convert to percentage


def evaluate_bleu_scores(df):
    """
    Calculate BLEU scores for the translations dataframe
    """
    print("\nCalculating BLEU scores...")

    bleu_scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing BLEU"):
        reference = row['Sinhala']
        prediction = row['preds']

        if pd.isna(reference) or pd.isna(prediction) or not str(reference).strip() or not str(prediction).strip():
            bleu_scores.append(0.0)
            continue

        bleu = calculate_bleu_score(str(reference), str(prediction))
        bleu_scores.append(bleu)

    df['bleu_score'] = bleu_scores

    # Calculate overall statistics
    mean_bleu = np.mean(bleu_scores)
    std_bleu = np.std(bleu_scores)
    median_bleu = np.median(bleu_scores)

    print(f"\n" + "=" * 60)
    print(f"BLEU Score Evaluation Results:")
    print(f"=" * 60)
    print(f"Mean BLEU Score: {mean_bleu:.4f}")
    print(f"Standard Deviation: {std_bleu:.4f}")
    print(f"Median BLEU Score: {median_bleu:.4f}")
    print(f"Min BLEU Score: {min(bleu_scores):.4f}")
    print(f"Max BLEU Score: {max(bleu_scores):.4f}")
    print(f"=" * 60)

    return {
        'mean_bleu': mean_bleu,
        'std_bleu': std_bleu,
        'median_bleu': median_bleu,
        'min_bleu': min(bleu_scores),
        'max_bleu': max(bleu_scores),
        'scores': bleu_scores
    }


def predict(tsv_file_path):
    """Main prediction function"""
    # Load the TSV file
    print(f"Loading data from {tsv_file_path}...")
    full_df = pd.read_csv(tsv_file_path, sep='\t', encoding='utf-8')

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    # Use last 200 samples for testing
    test_size = min(1000, len(full_df))
    df = full_df.tail(test_size).copy()
    print(f"Using last {test_size} samples for testing")

    # Get the rest of the instances (excluding the tail)
    rest_of_instances = full_df.head(len(full_df) - test_size)

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting few-shot examples...")
        print(f"Available for few-shot examples: {len(rest_of_instances)}")
        few_shot_examples = get_few_shot_examples(rest_of_instances, df, num_examples=3)
        print(f"Selected {len(few_shot_examples)} few-shot examples")

        # Print the examples for verification
        print("\nFew-shot examples:")
        for i, example in enumerate(few_shot_examples, 1):
            print(f"Example {i}:")
            print(f"  Tamil: {example['tamil'][:100]}...")
            print(f"  Sinhala: {example['sinhala'][:100]}...")
            print()

        # Apply few-shot formatting
        df['chat'] = df.apply(lambda row: format_chat_few_shot(row, few_shot_examples), axis=1)
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
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Mean BLEU Score: {bleu_results['mean_bleu']:.4f}\n")
        f.write(f"Standard Deviation: {bleu_results['std_bleu']:.4f}\n")
        f.write(f"Median BLEU Score: {bleu_results['median_bleu']:.4f}\n")
        f.write(f"Min BLEU Score: {bleu_results['min_bleu']:.4f}\n")
        f.write(f"Max BLEU Score: {bleu_results['max_bleu']:.4f}\n")

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

    # Set up Cohere client
    COHERE_API_KEY = "<<your-api-key>>"  # Replace with your actual key
    co = cohere.ClientV2(COHERE_API_KEY)

    model_id = "command-a-03-2025"

    OUTPUT_FOLDER = os.path.join("outputs", "tamil_sinhala_translation", model_id, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict(TSV_FILE)