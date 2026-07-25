import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

set_seed(777)

# Few-shot example passages are truncated to this many characters to bound prompt
# length (Pali passages can be long, and three stacked examples otherwise risk
# overflowing the 8k-context Meta-Llama-3 checkpoints). Only the demonstration
# examples are trimmed; the actual test instance is always passed in full.
FEWSHOT_PREVIEW_CHARS = 500


# --------------------------------------------------------------------------- #
# Data loading + split (from the original Pali->Sinhala script)
# --------------------------------------------------------------------------- #
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
    Identical split to the original script so families stay comparable.
    """
    test_size = min(test_size, len(full_df))
    test_df = full_df.tail(test_size).copy()
    dev_df = full_df.head(len(full_df) - test_size).copy()
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    return dev_df, test_df


# --------------------------------------------------------------------------- #
# Few-shot selection (from the original Pali->Sinhala script; drawn from dev)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Prompting (IDENTICAL wording to the original Pali->Sinhala script for BLEU
# comparability — do not reword without updating every model script)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Output post-processing (Llama has no thinking mode)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Generation (text-only causal LM, Llama-3 <|eot_id|> terminator)
# --------------------------------------------------------------------------- #
def build_terminators(tok):
    terminators = []
    if tok.eos_token_id is not None:
        terminators.append(tok.eos_token_id)
    # Llama 3 ends a turn with <|eot_id|>, which differs from the default EOS.
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tok.unk_token_id and eot not in terminators:
        terminators.append(eot)
    return terminators or None


def generate(model, tokenizer, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    terminators = build_terminators(tokenizer)
    sample_kwargs = dict(temperature=0.6, top_p=0.9) if do_sample else {}  # Llama defaults

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating translations"):
        batch = list_of_messages[start:start + batch_size]
        inputs = tokenizer.apply_chat_template(
            batch, add_generation_prompt=True, tokenize=True,
            padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id, **sample_kwargs)
        outputs.extend(tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# BLEU (whitespace tokenization — required for Sinhala; identical to the other
# translation scripts so numbers are directly comparable across families)
# --------------------------------------------------------------------------- #
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


# --------------------------------------------------------------------------- #
# Prediction driver
# --------------------------------------------------------------------------- #
def predict(model, tokenizer, model_id, dev_df, test_df,
            max_new_tokens=200, batch_size=8, do_sample=False):
    print(f"Dev set size: {len(dev_df)}")
    print(f"Test set size: {len(test_df)}")
    print(f"Columns: {test_df.columns.tolist()}")

    df = test_df.copy()

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Dev set available for few-shot examples: {len(dev_df)}")
        print(f"Test instances: {len(df)}")

        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            few_shot_examples = get_few_shot_examples_for_instance(
                dev_df, instance_idx=idx, num_examples=3, seed=42
            )
            chat_messages.append(format_chat(row, few_shot_examples))

        df['chat'] = chat_messages
        print("Each test instance has been assigned unique few-shot examples")
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    print("Generating translations...")
    responses = generate(model, tokenizer, df['chat'].tolist(),
                         batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=do_sample)
    df['responses'] = responses

    print("Extracting translations...")
    df['preds'] = df.apply(lambda row: extract_translation(row['responses']), axis=1)

    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"Empty predictions: {n_empty}/{len(df)}")

    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    print("Evaluating translations with BLEU score...")
    bleu_results = evaluate_bleu_scores(df)

    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_bleu.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with BLEU scores saved to: {results_file}")

    summary_file = os.path.join(OUTPUT_FOLDER, "bleu_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: sinhala-nlp/pali-sinhala (last 1000 rows as test)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.6,p=0.9)' if do_sample else 'greedy'}\n")
        f.write(f"Empty predictions: {n_empty}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance from dev set)\n")
        f.write("=" * 70 + "\n\n")

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
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.3-70B-Instruct', required=False,
                        help='HF model id (any Llama-3 instruct checkpoint)')
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='Number of prompts decoded per generation call')
    parser.add_argument('--max_new_tokens', type=int, default=200, required=False,
                        help='Max new tokens to generate per instance')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Llama sampling (t=0.6,p=0.9) instead of greedy. '
                             'Leave off for BLEU reproducibility.')

    args = parser.parse_args()

    MODEL_ID = args.model_id
    QUERY_TYPE = args.query_type
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens
    DO_SAMPLE = args.do_sample

    print(f"Model: {MODEL_ID}")
    print(f"Query type: {QUERY_TYPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Decoding: {'sampling' if DO_SAMPLE else 'greedy'}")

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    # Load + split dataset (test size fixed at 1000, matching the original script)
    full_df = load_and_prepare_pali_sinhala_dataset()
    dev_df, test_df = split_dataset(full_df, test_size=1000)

    if dev_df is None or test_df is None or len(test_df) == 0:
        print("Error: Could not load or split dataset properly")
        raise SystemExit(1)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    OUTPUT_FOLDER = os.path.join("outputs", "pali_sinhala_translation", MODEL_ID.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict(
        model, tokenizer, MODEL_ID, dev_df, test_df,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, do_sample=DO_SAMPLE,
    )