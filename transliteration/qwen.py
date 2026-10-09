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
from transformers import AutoConfig, AutoProcessor, AutoTokenizer, AutoModelForCausalLM, set_seed

# Qwen3.5/3.6 are multimodal; their cards load them with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers.
try:
    from transformers import AutoModelForMultimodalLM as _AutoMultimodalModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoMultimodalModel

# Shared transliteration metrics (CER, WER, chrF on NFC-normalised text).
# Same module as the Gemma script, so scores are comparable across families.
from translit_metric import score_corpus, METRICS

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
# Checkpoint routing
# --------------------------------------------------------------------------- #
# Qwen3.5 / Qwen3.6 (vision_config present) -> AutoProcessor + AutoModelForMultimodalLM
# Qwen2 / Qwen2.5 (text-only)                -> AutoTokenizer + AutoModelForCausalLM
# --------------------------------------------------------------------------- #
def _from_pretrained(cls, model_id):
    """`torch_dtype` was renamed to `dtype` in transformers 5.x; try the new
    name first and fall back for older installs."""
    try:
        return cls.from_pretrained(model_id, dtype="auto", device_map="auto")
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype="auto", device_map="auto")


def load_model(model_id):
    config = AutoConfig.from_pretrained(model_id)
    is_multimodal = getattr(config, "vision_config", None) is not None

    if is_multimodal:
        print("[loader] Multimodal Qwen checkpoint (vision_config present) "
              "-> AutoProcessor + AutoModelForMultimodalLM")
        proc = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(_AutoMultimodalModel, model_id)
    else:
        print("[loader] Text-only Qwen checkpoint -> AutoTokenizer + AutoModelForCausalLM")
        proc = AutoTokenizer.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForCausalLM, model_id)

    model.eval()

    if hasattr(model, "hf_device_map"):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    return model, proc


# ---------------------------------------------------------------------------
# Data loading (official train/test splits from the HF dataset)
# ---------------------------------------------------------------------------

def load_translit_dataset():
    """
    Loads sinhala-nlp/translit (Dakshina Sinhala romanisations).

    The splits are loaded one CSV at a time: test.csv has an extra `id` column,
    so a single load_dataset("sinhala-nlp/translit") call fails with
    DatasetGenerationCastError (mismatched columns across splits).
    """
    print("Loading transliteration dataset from HuggingFace...")
    train_df = load_dataset("sinhala-nlp/translit", data_files="train.csv", split="train").to_pandas()
    test_df = load_dataset("sinhala-nlp/translit", data_files="test.csv", split="train").to_pandas()

    train_df = train_df.dropna(subset=['romanized', 'sinhala']).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    print(f"Train set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    return train_df, test_df


# ---------------------------------------------------------------------------
# Few-shot selection (drawn from train; never from test)
# ---------------------------------------------------------------------------

def get_few_shot_examples_for_instance(train_df, instance_idx, num_examples=3, seed=None):
    """
    Random few-shot examples for a specific test instance, drawn from the train set.
    Each test instance gets a different set (instance-specific seed).
    """
    if seed is not None:
        random.seed(seed + instance_idx)

    available_indices = list(train_df.index)
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = train_df.loc[idx]
        if str(row['romanized']).strip() and str(row['sinhala']).strip():
            few_shot_examples.append({
                'romanized': str(row['romanized']),
                'sinhala': str(row['sinhala'])
            })

    return few_shot_examples


# ---------------------------------------------------------------------------
# Prompting (IDENTICAL wording to the Gemma transliteration script — do not
# reword without updating every model script)
# ---------------------------------------------------------------------------

def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert in the Sinhala language. The following text (R) is Sinhala written in the Latin script (romanised Sinhala). Convert it into the Sinhala script, preserving the original words and meaning."
    action_desc = "Return only the Sinhala script text following the prefix 'Transliteration:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාඨය (R) ඉංග්‍රීසි අකුරින් ලියන ලද සිංහල පාඨයකි. මුල් වචන සහ අර්ථය ආරක්ෂා කරමින් එය සිංහල අකුරින් ලියන්න."
    action_desc_si = "'Transliteration:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල අකුරින් ලියූ පාඨය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"R: {example['romanized']}\n"
            examples_str += f"Transliteration: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} R: {row['romanized']}"

    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} R: {row['romanized']}"

    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow transliterate this text:\nR: {row['romanized']}"

    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මෙම පාඨය සිංහල අකුරින් ලියන්න:\nR: {row['romanized']}"

    else:
        content = f"{task_desc} {action_desc} R: {row['romanized']}"

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


def extract_transliteration(response):
    """Extract the Sinhala text after the 'Transliteration:' prefix (first line only)."""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    text = strip_thinking(response)

    try:
        matches = re.findall(r'Transliteration:\s*(.*?)(?:\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if matches and matches[0].strip():
            return matches[0].strip()

        lines = [line.strip() for line in text.split("\n") if line.strip()]
        return lines[0] if lines else ""
    except Exception as e:
        print(f"Error extracting transliteration: {e}")
        return ""


# --------------------------------------------------------------------------- #
# Generation (batched, left-padded, Qwen thinking disabled)
# --------------------------------------------------------------------------- #
def generate(model, proc, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Qwen-recommended non-thinking sampling params (used only if do_sample=True).
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    template_kwargs = dict(add_generation_prompt=True, tokenize=True, padding=True,
                           return_tensors="pt", return_dict=True)

    outputs = []
    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating transliterations"):
        batch = list_of_messages[start:start + batch_size]
        try:
            inputs = proc.apply_chat_template(batch, enable_thinking=False, **template_kwargs)
        except TypeError:
            inputs = proc.apply_chat_template(batch, **template_kwargs)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=tok.pad_token_id,
                                 **sample_kwargs)
        decoded = tok.batch_decode(gen[:, input_len:], skip_special_tokens=True)
        outputs.extend(strip_thinking(t) for t in decoded)
    return outputs


# ---------------------------------------------------------------------------
# Prediction driver
# ---------------------------------------------------------------------------

def predict(model, proc, model_id, train_df, test_df,
            max_new_tokens=512, batch_size=16, do_sample=False):
    df = test_df.copy()

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        chat_messages = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            few_shot_examples = get_few_shot_examples_for_instance(
                train_df, instance_idx=idx, num_examples=3, seed=42
            )
            chat_messages.append(format_chat(row, few_shot_examples))
        df['chat'] = chat_messages
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    log_gpu_memory()
    print("Generating transliterations...")
    df['responses'] = generate(model, proc, df['chat'].tolist(),
                               batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=do_sample)
    log_gpu_memory()

    print("Extracting transliterations...")
    df['preds'] = df['responses'].apply(extract_transliteration)

    n_empty = int((df['preds'].str.len() == 0).sum())
    n_missing_prefix = int((~df['responses'].str.contains('Transliteration:', case=False, na=False)).sum())
    print(f"Empty predictions: {n_empty}/{len(df)}")
    print(f"Responses without 'Transliteration:' prefix: {n_missing_prefix}/{len(df)}")

    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.drop(columns=['chat']).to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    print("Evaluating with CER / WER / chrF...")
    results = score_corpus(df['sinhala'].astype(str).tolist(), df['preds'].astype(str).tolist())
    for m in METRICS:
        df[m] = results[m]['scores']

    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_scores.csv")
    df.drop(columns=['chat']).to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with scores saved to: {results_file}")

    summary_file = os.path.join(OUTPUT_FOLDER, "translit_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("Transliteration Evaluation Results (NFC; CER/WER lower is better, chrF higher is better)\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: sinhala-nlp/translit (official test split)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Empty predictions: {n_empty}/{len(df)}\n")
        f.write(f"Responses without 'Transliteration:' prefix: {n_missing_prefix}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance from train set)\n")
        f.write("=" * 70 + "\n\n")

        for m in METRICS:
            f.write(f"{m.upper()}:\n")
            f.write(f"  Corpus: {results[m]['corpus']:.4f}\n")
            for stat in ['mean', 'median', 'std', 'min', 'max']:
                f.write(f"  {stat.capitalize() + ':':8}{results[m][stat]:.4f}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    for m in METRICS:
        print(f"{m.upper()} (corpus): {results[m]['corpus']:.4f}")
    print("=" * 70)
    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen2.5-7B-Instruct', required=False,
                        help='HF model id (Qwen2/2.5 text-only or Qwen3.5/3.6 multimodal; loader auto-detects)')
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=16, required=False,
                        help='Number of prompts decoded per generation call')
    parser.add_argument('--max_new_tokens', type=int, default=512, required=False,
                        help='Max new tokens to generate per instance')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy.')

    args = parser.parse_args()

    MODEL_ID = args.model_id
    QUERY_TYPE = args.query_type

    print(f"Model: {MODEL_ID}")
    print(f"Query type: {QUERY_TYPE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    train_df, test_df = load_translit_dataset()

    model, proc = load_model(MODEL_ID)

    OUTPUT_FOLDER = os.path.join("outputs", "transliteration", MODEL_ID.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predict(
        model, proc, MODEL_ID, train_df, test_df,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, do_sample=args.do_sample,
    )