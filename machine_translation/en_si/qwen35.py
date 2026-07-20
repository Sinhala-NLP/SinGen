import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
)
import tarfile
import urllib.request

# Qwen3.5/3.6 are multimodal; their cards load with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers.
try:
    from transformers import AutoModelForMultimodalLM as _AutoMultimodal
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoMultimodal

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
set_seed(777)


# ---------------------------------------------------------------------------
# Checkpoint routing (Qwen)
# ---------------------------------------------------------------------------
# Qwen3.5 / Qwen3.6 (multimodal, vision_config present)
#     -> AutoProcessor + AutoModelForMultimodalLM
# Qwen3 / Qwen2.5 / Qwen2 (text-only)
#     -> AutoTokenizer   + AutoModelForCausalLM
#
# Multimodal vs. text-only is decided at runtime by the presence of a
# `vision_config` on the checkpoint config (same convention as the other
# SINGEN loaders). <think>...</think> stripping applies to both branches.
# ---------------------------------------------------------------------------

def detect_checkpoint_type(model_id):
    config = AutoConfig.from_pretrained(model_id)
    is_multimodal = getattr(config, "vision_config", None) is not None
    return config, is_multimodal


def load_model(model_id):
    config, is_multimodal = detect_checkpoint_type(model_id)
    common_kwargs = dict(dtype="auto", device_map="auto")

    if is_multimodal:
        print("[loader] Multimodal Qwen checkpoint detected (vision_config present) "
              "-> AutoProcessor + AutoModelForMultimodalLM")
        proc = AutoProcessor.from_pretrained(model_id)
        model = _AutoMultimodal.from_pretrained(model_id, **common_kwargs)
        is_processor = True
    else:
        print("[loader] Text-only Qwen checkpoint detected "
              "-> AutoTokenizer + AutoModelForCausalLM")
        proc = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, **common_kwargs)
        is_processor = False

    model.eval()
    return model, proc, is_processor


def get_tokenizer(proc, is_processor):
    """Return the underlying tokenizer whether proc is a processor (wraps a
    .tokenizer) or a bare tokenizer."""
    if is_processor and hasattr(proc, "tokenizer"):
        return proc.tokenizer
    return proc


# ---------------------------------------------------------------------------
# Qwen thinking handling
# ---------------------------------------------------------------------------
# Qwen3.5/3.6 think BY DEFAULT, emitting <think>...</think> before the answer.
# We disable thinking at generation time AND strip any residual block here.
_QWEN_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r"</?think>", re.IGNORECASE)


def strip_thinking(text):
    if not isinstance(text, str):
        return ""
    text = _QWEN_THINK.sub("", text)
    text = _STRAY_THINK.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading (identical to the Gemma MT script)
# ---------------------------------------------------------------------------

def _ensure_flores200(cache_dir=None):
    """Download + extract the FLORES-200 archive once; return the path to the
    extracted `flores200_dataset` dir. Replaces the removed HF loading script
    (datasets>=4.0 no longer runs flores200.py)."""
    cache_dir = cache_dir or os.environ.get(
        "FLORES200_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "flores200"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    extracted = os.path.join(cache_dir, "flores200_dataset")

    # Treat as present only if the files we actually need exist.
    sentinel = os.path.join(extracted, "devtest", "sin_Sinh.devtest")
    if not os.path.exists(sentinel):
        tarball = os.path.join(cache_dir, "flores200_dataset.tar.gz")
        if not os.path.exists(tarball):
            print(f"Downloading FLORES-200 from {FLORES_URL} ...")
            urllib.request.urlretrieve(FLORES_URL, tarball)
        print(f"Extracting {tarball} ...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")  # filter= is py>=3.12
    return extracted


def _read_lines(path):
    # Explicit utf-8 is important for Sinhala.
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def download_and_load_flores_en_si():
    """
    Loads FLORES-200 English-Sinhala dev/devtest directly from the official
    archive (no HF dataset script). Returns dev and devtest DataFrames with
    columns 'english' and 'sinhala'.
    """
    print("Loading FLORES-200 dataset for English-Sinhala...")
    root = _ensure_flores200()

    splits = {}
    for split_name in ["dev", "devtest"]:
        eng = _read_lines(os.path.join(root, split_name, f"eng_Latn.{split_name}"))
        sin = _read_lines(os.path.join(root, split_name, f"sin_Sinh.{split_name}"))
        assert len(eng) == len(sin), (
            f"{split_name}: eng/sin line-count mismatch ({len(eng)} vs {len(sin)})"
        )
        df = pd.DataFrame({"english": eng, "sinhala": sin})
        splits[split_name] = df
        print(f"{split_name.capitalize()} split size: {len(df)}")

    return splits.get("dev"), splits.get("devtest")


def get_few_shot_examples_for_instance(dev_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance from dev set.
    Each test instance gets a different randomly selected set.
    """
    if seed is not None:
        random.seed(seed + instance_idx)

    available_indices = list(dev_df.index)
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = dev_df.loc[idx]
        if pd.notna(row['english']) and pd.notna(row['sinhala']) and \
                str(row['english']).strip() and str(row['sinhala']).strip():
            few_shot_examples.append({
                'english': str(row['english']),
                'sinhala': str(row['sinhala'])
            })

    return few_shot_examples


# ---------------------------------------------------------------------------
# Prompting (identical wording to the Gemma MT script for BLEU comparability)
# ---------------------------------------------------------------------------

def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in English to Sinhala translation. Translate the following English sentence (E) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ ඉංග්‍රීසි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත ඉංග්‍රීසි වාක්‍යය (E) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"E: {example['english']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} E: {row['english']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} E: {row['english']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nE: {row['english']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nE: {row['english']}"
        return [{"role": "user", "content": prompt}]

    else:
        return [{"role": "user", "content": f"{task_desc} {action_desc} E: {row['english']}"}]


def to_processor_messages(messages):
    """Processor chat templates (multimodal Qwen) expect content as a list of
    typed parts. Wrap plain-string content in a single text part."""
    converted = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": m["role"], "content": content})
    return converted


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def query(model, proc, is_processor, messages_list, max_new_tokens=200,
          batch_size=8, do_sample=False):
    """
    Runs decoding in batches. Returns list of decoded strings with any thinking
    blocks stripped. Uses left padding so new tokens start at the same offset
    for every sequence in a batch.
    """
    tok = get_tokenizer(proc, is_processor)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Qwen-recommended non-thinking sampling params (used only if do_sample=True).
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    assistant_outputs = []

    for start in tqdm(range(0, len(messages_list), batch_size), desc="Generating"):
        batch = messages_list[start:start + batch_size]
        if is_processor:
            batch = [to_processor_messages(m) for m in batch]

        template_kwargs = dict(
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )

        # Disable thinking on Qwen; older templates don't accept the kwarg.
        try:
            inputs = proc.apply_chat_template(batch, enable_thinking=False, **template_kwargs)
        except TypeError:
            inputs = proc.apply_chat_template(batch, **template_kwargs)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_id,
                **sample_kwargs,
            )

        new_tokens = generated[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        assistant_outputs.extend(strip_thinking(t) for t in decoded)

    return assistant_outputs


def extract_translation(response):
    """Extract translation from model response."""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    try:
        response = strip_thinking(response)
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


# ---------------------------------------------------------------------------
# BLEU (whitespace tokenization — required for Sinhala; identical to MT script)
# ---------------------------------------------------------------------------

def tokenize(text):
    """Simple tokenization by splitting on whitespace."""
    if pd.isna(text) or text is None:
        return []
    return str(text).strip().split()


def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
    """
    Calculate individual BLEU scores (BLEU-1..4) and overall BLEU.
    """
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


# ---------------------------------------------------------------------------
# Prediction driver
# ---------------------------------------------------------------------------

def predict(model, proc, is_processor, dev_df, devtest_df,
            max_new_tokens=200, batch_size=8, do_sample=False):
    print(f"Dev set size: {len(dev_df)}")
    print(f"Devtest set size: {len(devtest_df)}")
    print(f"Columns: {devtest_df.columns.tolist()}")

    df = devtest_df.copy()

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Dev set available for few-shot examples: {len(dev_df)}")
        print(f"Test instances (devtest): {len(df)}")

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
    responses = query(model, proc, is_processor, df['chat'].tolist(),
                      max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=do_sample)
    df['responses'] = responses

    print("Extracting translations...")
    df['preds'] = df.apply(lambda row: extract_translation(row['responses']), axis=1)

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
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: FLORES-200 English-Sinhala (devtest split)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
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
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3.5-27B', required=False,
                        help='HF model id (any Qwen checkpoint; loader auto-detects multimodal vs text-only)')
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='Number of prompts decoded per generation call')
    parser.add_argument('--max_new_tokens', type=int, default=200, required=False,
                        help='Max new tokens to generate per instance')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy. '
                             'Qwen warns greedy can cause repetition loops.')

    args = parser.parse_args()

    MODEL_ID = args.model_id
    QUERY_TYPE = args.query_type
    BATCH_SIZE = args.batch_size
    MAX_NEW_TOKENS = args.max_new_tokens

    print(f"Model: {MODEL_ID}")
    print(f"Query type: {QUERY_TYPE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max new tokens: {MAX_NEW_TOKENS}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    model, proc, is_processor = load_model(MODEL_ID)

    dev_df, devtest_df = download_and_load_flores_en_si()
    if dev_df is None or devtest_df is None:
        print("Error: Could not load required dataset splits")
        exit(1)

    OUTPUT_FOLDER = os.path.join("outputs", "english_sinhala_translation", MODEL_ID.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predictions, bleu_results = predict(
        model, proc, is_processor, dev_df, devtest_df,
        max_new_tokens=MAX_NEW_TOKENS, batch_size=BATCH_SIZE, do_sample=args.do_sample,
    )