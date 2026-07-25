import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoProcessor, set_seed

# Qwen3.5/3.6 are multimodal; their cards load them with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers.
try:
    from transformers import AutoModelForMultimodalLM as _AutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoModel

set_seed(777)


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def log_gpu_memory():
    if torch.cuda.is_available():
        print("\n" + "=" * 60 + "\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
            resv = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"  GPU {i}: allocated {alloc:.2f} GB | reserved {resv:.2f} GB")
        print("=" * 60 + "\n")


# --------------------------------------------------------------------------- #
# Few-shot selection (identical task logic to the Gemma Ta->Si script)
# --------------------------------------------------------------------------- #
def get_few_shot_examples_for_instance(full_df, test_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance.
    Each test instance gets a different randomly selected set, drawn from
    `full_df` (the non-test pool) and never from the test instances.
    """
    test_indices = set(test_df.index)
    available_indices = [i for i in full_df.index if i not in test_indices]

    if seed is not None:
        random.seed(seed + instance_idx)

    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = full_df.loc[idx]
        if pd.notna(row['Tamil']) and pd.notna(row['Sinhala']) and \
                str(row['Tamil']).strip() and str(row['Sinhala']).strip():
            few_shot_examples.append({
                'tamil': str(row['Tamil']),
                'sinhala': str(row['Sinhala'])
            })

    return few_shot_examples


# --------------------------------------------------------------------------- #
# Prompting (IDENTICAL wording to the Gemma Ta->Si script for BLEU
# comparability — do not reword without updating every model script)
# --------------------------------------------------------------------------- #
def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"T: {example['tamil']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} T: {row['Tamil']}"

    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} T: {row['Tamil']}"

    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nT: {row['Tamil']}"

    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nT: {row['Tamil']}"

    else:
        content = f"{task_desc} {action_desc} T: {row['Tamil']}"

    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Output post-processing (Qwen thinking-aware)
# --------------------------------------------------------------------------- #
# Qwen3.5/3.6 think BY DEFAULT, emitting <think>...</think> before the answer.
# We disable thinking at generation time AND strip any residual block here.
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_translation(response):
    """Extract translation from model response (thinking stripped first)."""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    text = strip_thinking(response)

    try:
        matches = re.findall(r'Translation:\s*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

        if "translation:" in text.lower():
            parts = text.lower().split("translation:")
            if len(parts) > 1:
                return parts[1].strip()

        return text.strip()
    except Exception as e:
        print(f"Error extracting translation: {e}")
        return ""


# --------------------------------------------------------------------------- #
# Generation (batched, left-padded, Qwen thinking disabled)
# --------------------------------------------------------------------------- #
def generate(model, processor, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Qwen-recommended non-thinking sampling params (used only if do_sample=True).
    # For BLEU we default to greedy for reproducibility.
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating translations"):
        batch = list_of_messages[start:start + batch_size]
        try:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True,
                enable_thinking=False)          # Qwen honours this in its template
        except TypeError:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=tok.pad_token_id,
                                 **sample_kwargs)
        outputs.extend(tok.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# BLEU (whitespace tokenization — required for Sinhala; identical to the
# Gemma Ta->Si script so numbers are directly comparable across families)
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
# Prediction driver (TSV loading + tail-1000 split — identical to Gemma Ta->Si)
# --------------------------------------------------------------------------- #
def predict(model, processor, model_id, tsv_file_path,
            max_new_tokens=200, batch_size=8, do_sample=False):
    print(f"Loading data from {tsv_file_path}...")
    full_df = pd.read_csv(tsv_file_path, sep='\t', encoding='utf-8')

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    # Use last 1000 samples for testing (identical split to the Gemma script so
    # the two families evaluate on the same test set).
    test_size = min(1000, len(full_df))
    df = full_df.tail(test_size).copy()
    print(f"Using last {test_size} samples for testing")

    # Pool for few-shot examples (everything except the test tail)
    rest_of_instances = full_df.head(len(full_df) - test_size)

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Total dataset size: {len(full_df)}")
        print(f"Test instances: {len(df)}")
        print(f"Available for few-shot examples: {len(rest_of_instances)}")

        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            few_shot_examples = get_few_shot_examples_for_instance(
                rest_of_instances, df, instance_idx=idx, num_examples=3, seed=42
            )
            chat_messages.append(format_chat(row, few_shot_examples))

        df['chat'] = chat_messages
        print("Each test instance has been assigned unique few-shot examples")
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    log_gpu_memory()
    print("Generating translations...")
    responses = generate(model, processor, df['chat'].tolist(),
                         batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=do_sample)
    df['responses'] = responses
    log_gpu_memory()

    print("Extracting translations...")
    df['preds'] = df.apply(lambda row: extract_translation(row['responses']), axis=1)

    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for truncated/looping <think> if high")

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
        f.write("Dataset: TamSiPara Tamil-Sinhala (last 1000 rows as test)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Empty predictions: {n_empty}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance)\n")
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
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3.5-35B-A3B', required=False,
                        help='HF model id (any Qwen3.5/3.6 multimodal checkpoint)')
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='Number of prompts decoded per generation call')
    parser.add_argument('--max_new_tokens', type=int, default=200, required=False,
                        help='Max new tokens to generate per instance')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy. '
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

    TSV_FILE = os.path.join("ta_si.tsv")
    print(f"TSV file: {TSV_FILE}")

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = _AutoModel.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    OUTPUT_FOLDER = os.path.join("outputs", "tamil_sinhala_translation", MODEL_ID.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    log_gpu_memory()
    predictions, bleu_results = predict(
        model, processor, MODEL_ID, TSV_FILE,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, do_sample=DO_SAMPLE,
    )
    log_gpu_memory()