"""Pali→Sinhala prompting through a multi-node vLLM OpenAI-compatible server.
Uses identical dataset, prompts and BLEU implementation to the supplied Gemma script.
No model weights are loaded by this client.
"""
import argparse
import json
import os
import re
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import load_dataset
from tqdm.auto import tqdm
from openai import OpenAI

FEWSHOT_PREVIEW_CHARS = 500
QUERY_TYPE = "zero-shot"


def load_and_prepare_pali_sinhala_dataset():
    """
    Loads Pali-Sinhala dataset from HuggingFace and strips leading verse numbers
    from both columns. Returns the full dataframe.
    """
    print("Loading Pali-Sinhala dataset from HuggingFace...")
    ds = load_dataset("sinhala-nlp/pali-sinhala")
    full_df = ds['train'].to_pandas()

    print("Cleaning dataset (removing leading numbers)...")
    full_df['pali_text'] = full_df['pali_text'].str.replace(r'^\d+\s+', '', regex=True)
    full_df['sinhala_text'] = full_df['sinhala_text'].str.replace(r'^\d+\s+', '', regex=True)

    print(f"Total dataset size: {len(full_df)}")
    return full_df

def split_dataset(full_df, test_size=1000):
    """
    Last `test_size` rows are the test set; the rest form the dev/few-shot pool.
    Identical split to the Llama script so families stay comparable.
    """
    test_size = min(test_size, len(full_df))
    test_df = full_df.tail(test_size).copy()
    dev_df = full_df.head(len(full_df) - test_size).copy()
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    return dev_df, test_df

def get_few_shot_examples_for_instance(dev_df, instance_idx, num_examples=3, seed=None):
    """
    Random few-shot examples for a specific test instance, drawn from the dev set.
    Each test instance gets a different set (instance-specific seed).
    """
    if seed is not None:
        random.seed(seed + instance_idx)

    available_indices = list(dev_df.index)
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = dev_df.loc[idx]
        if pd.notna(row['pali_text']) and pd.notna(row['sinhala_text']) and \
                str(row['pali_text']).strip() and str(row['sinhala_text']).strip():
            few_shot_examples.append({
                'pali': str(row['pali_text']),
                'sinhala': str(row['sinhala_text'])
            })

    return few_shot_examples

def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in Pali to Sinhala translation. Translate the following Pali text (P) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ පාලි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාලි පාඨය (P) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            pali = example['pali']
            sinhala = example['sinhala']
            pali_preview = pali[:FEWSHOT_PREVIEW_CHARS] + "..." if len(pali) > FEWSHOT_PREVIEW_CHARS else pali
            sinhala_preview = sinhala[:FEWSHOT_PREVIEW_CHARS] + "..." if len(sinhala) > FEWSHOT_PREVIEW_CHARS else sinhala
            examples_str += f"\nExample {i}:\n"
            examples_str += f"P: {pali_preview}\n"
            examples_str += f"Translation: {sinhala_preview}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} P: {row['pali_text']}"

    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} P: {row['pali_text']}"

    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this text:\nP: {row['pali_text']}"

    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පාඨය පරිවර්තනය කරන්න:\nP: {row['pali_text']}"

    else:
        content = f"{task_desc} {action_desc} P: {row['pali_text']}"

    return [{"role": "user", "content": content}]

def extract_translation(response):
    """Extract translation from model response."""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    try:
        matches = re.findall(r'Translation:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

        if "translation:" in response.lower():
            parts = response.lower().split("translation:")
            if len(parts) > 1:
                return parts[1].strip()

        return response.strip()
    except Exception as e:
        print(f"Error extracting translation: {e}")
        return ""

def tokenize(text):
    """Simple tokenization by splitting on whitespace."""
    if pd.isna(text) or text is None:
        return []
    return str(text).strip().split()

def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
    """Calculate individual BLEU scores (BLEU-1..4) and overall BLEU."""
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not pred_tokens or not ref_tokens:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu_overall': 0.0}

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

    precisions = []
    individual_bleu_scores = {}

    for n in range(1, max_n + 1):
        ref_ngrams = {}
        pred_ngrams = {}

        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i + n])
            pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1

        clipped_count = 0
        total_count = sum(pred_ngrams.values())

        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                clipped_count += min(count, ref_ngrams[ngram])

        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)
        individual_bleu_scores[f'bleu_{n}'] = (bp * precision * 100) if precision > 0 else 0.0

    if min(precisions) > 0:
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    individual_bleu_scores['bleu_overall'] = bp * geo_mean * 100
    return individual_bleu_scores

def evaluate_bleu_scores(df):
    """Calculate individual and overall BLEU scores for the translations dataframe."""
    print("\nCalculating BLEU scores...")

    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, bleu_overall_scores = [], [], [], [], []

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

    print("\n" + "=" * 70)
    print("BLEU Score Evaluation Results:")
    print("=" * 70)

    for score_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
        scores = df[score_type].tolist()
        print(f"\n{score_type.upper().replace('_', '-')}:")
        print(f"  Mean:   {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std:    {np.std(scores):.4f}")
        print(f"  Min:    {np.min(scores):.4f}")
        print(f"  Max:    {np.max(scores):.4f}")

    print("=" * 70)

    def stats(scores):
        return {
            'mean': np.mean(scores), 'median': np.median(scores), 'std': np.std(scores),
            'min': np.min(scores), 'max': np.max(scores), 'scores': scores
        }

    return {
        'bleu_1': stats(bleu_1_scores),
        'bleu_2': stats(bleu_2_scores),
        'bleu_3': stats(bleu_3_scores),
        'bleu_4': stats(bleu_4_scores),
        'bleu_overall': stats(bleu_overall_scores),
    }

def run(args):
    global QUERY_TYPE
    QUERY_TYPE = args.query_type
    output_folder = Path(args.output_dir) / args.model_id.split('/')[-1] / QUERY_TYPE
    output_folder.mkdir(parents=True, exist_ok=True)
    summary_path = output_folder / 'bleu_summary.txt'
    if summary_path.exists() and not args.overwrite:
        print('Already completed:', summary_path)
        return

    client = OpenAI(base_url=args.base_url, api_key='EMPTY', timeout=1800, max_retries=2)
    df = load_and_prepare_pali_sinhala_dataset()
    dev_df, test_df = split_dataset(df, test_size=args.test_size)
    if QUERY_TYPE in ('few-shot', 'few-shot-si'):
        chats = [format_chat(row, get_few_shot_examples_for_instance(dev_df, idx, 3, seed=42))
                 for idx, (_, row) in enumerate(test_df.iterrows())]
    else:
        chats = [format_chat(row) for _, row in test_df.iterrows()]
    result = test_df.copy()
    result['chat'] = chats
    result['responses'] = ''
    result['preds'] = ''
    result['finish_reason'] = ''
    result['reasoning_tokens'] = 0
    pred_path = output_folder / 'predictions.csv'
    if pred_path.exists() and args.resume and not args.overwrite:
        previous = pd.read_csv(pred_path, keep_default_na=False)
        if len(previous) == len(result):
            for col in ('responses', 'preds', 'finish_reason', 'reasoning_tokens'):
                
                if col in previous:
                    result[col] = previous[col].fillna('').tolist()
    for i, messages in enumerate(tqdm(chats, desc='Generating')):
        if str(result.iloc[i]['responses']).strip():
            continue
        error = None
        for attempt in range(args.retries):
            try:
                completion = client.chat.completions.create(
                    model=args.model_id,
                    messages=messages,
                    max_tokens=args.max_new_tokens,
                    temperature=0.0,
                    reasoning_effort=args.reasoning_effort,
                )
                choice = completion.choices[0]
                answer = choice.message.content or ''
                # Save finish reason so output truncation can be audited.
                result.iat[i, result.columns.get_loc('responses')] = answer
                result.iat[i, result.columns.get_loc('preds')] = extract_translation(answer)
                result.iat[i, result.columns.get_loc('finish_reason')] = choice.finish_reason or ''
                usage = getattr(completion, 'usage', None)
                details = getattr(usage, 'completion_tokens_details', None)
                result.iat[i, result.columns.get_loc('reasoning_tokens')] = (getattr(details, 'reasoning_tokens', 0) or 0)
                error = None
                break
            except Exception as exc:
                error = exc
                print(f'Example {i}: retry {attempt+1}/{args.retries}: {exc}', flush=True)
                time.sleep(min(10 * (attempt + 1), 60))
        if error is not None:
            result.to_csv(pred_path, index=False, encoding='utf-8')
            raise RuntimeError(f'Example {i} failed; checkpoint saved') from error
        if (i + 1) % args.save_every == 0:
            result.to_csv(pred_path, index=False, encoding='utf-8')
    result.to_csv(pred_path, index=False, encoding='utf-8')
    stats = evaluate_bleu_scores(result)
    result.to_csv(output_folder / 'predictions_with_bleu.csv', index=False, encoding='utf-8')
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write(f'Model: {args.model_id}\nQuery Type: {QUERY_TYPE}\n')
        f.write(f'Dataset: sinhala-nlp/pali-sinhala (last {args.test_size} rows)\n')
        f.write(f'Max New Tokens (including reasoning): {args.max_new_tokens}\nDecoding: temperature=0\nReasoning effort: {args.reasoning_effort}\n')
        f.write('Native Kimi-K3 MXFP4 checkpoint, no additional quantisation\n')
        f.write(f'Finish reason length: {(result.finish_reason == "length").sum()}\n')
        f.write('Few-shot: 3 instance-specific examples from dev; seed=42; preview=500 chars\n')
        f.write(f'Empty predictions: {(result.preds.str.len() == 0).sum()}\n')
        for metric, vals in stats.items():
            f.write(f'{metric}: mean={vals["mean"]:.4f}, median={vals["median"]:.4f}, std={vals["std"]:.4f}\n')
    print('Finished:', summary_path)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model_id', default='moonshotai/Kimi-K3')
    p.add_argument('--query_type', choices=['zero-shot','zero-shot-si','few-shot','few-shot-si'], required=True)
    p.add_argument('--base_url', default='http://127.0.0.1:8000/v1')
    p.add_argument('--test_size', type=int, default=1000)
    p.add_argument('--max_new_tokens', type=int, default=2048)
    p.add_argument('--reasoning_effort', choices=['low','high','max'], default='low')
    p.add_argument('--output_dir', default='outputs/pali_sinhala_translation')
    p.add_argument('--save_every', type=int, default=10)
    p.add_argument('--retries', type=int, default=5)
    p.add_argument('--resume', action='store_true', default=True)
    p.add_argument('--overwrite', action='store_true')
    run(p.parse_args())
