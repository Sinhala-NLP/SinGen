import argparse
import os
import re
import random

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    set_seed,
)

# Sinhala-safe ROUGE (whitespace-tokenized). Same module as the headline and
# summarisation tasks, so all ROUGE-L columns in the paper are comparable.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

# Few-shot demonstration infoboxes are truncated to this many characters to bound
# prompt length (linearised infoboxes reach ~5k chars, and three stacked examples
# otherwise blow up the prompt). Same value as the Pali->Si scripts. Only the
# demonstration sources are trimmed; demonstration targets and the actual test
# infobox are always passed in full.
FEWSHOT_PREVIEW_CHARS = 500


# ---------------------------------------------------------------------------
# Checkpoint routing (identical to the other Gemma scripts)
# ---------------------------------------------------------------------------
# Gemma-4                     -> AutoProcessor + AutoModelForCausalLM
# Gemma-3 multimodal (vision) -> AutoProcessor + AutoModelForImageTextToText
# Gemma-3 text-only           -> AutoTokenizer + AutoModelForCausalLM
# ---------------------------------------------------------------------------

def detect_checkpoint_type(model_id):
    config = AutoConfig.from_pretrained(model_id)
    model_type = (getattr(config, "model_type", "") or "").lower()

    is_gemma4 = "gemma4" in model_type or re.search(r"gemma[-_]?4", model_id.lower()) is not None

    vision_config = getattr(config, "vision_config", None)
    is_multimodal = vision_config is not None

    return config, is_gemma4, is_multimodal


def _from_pretrained(cls, model_id):
    """`torch_dtype` was renamed to `dtype` and removed in transformers 5.x;
    try the new name first and fall back for older installs."""
    try:
        return cls.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="auto")


def load_model(model_id):
    config, is_gemma4, is_multimodal = detect_checkpoint_type(model_id)

    if is_gemma4:
        print("[loader] Gemma-4 checkpoint detected -> AutoProcessor + AutoModelForCausalLM")
        proc = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForCausalLM, model_id)
        is_processor = True
    elif is_multimodal:
        print("[loader] Multimodal Gemma-3 checkpoint detected (vision_config present) "
              "-> AutoProcessor + AutoModelForImageTextToText")
        proc = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForImageTextToText, model_id)
        is_processor = True
    else:
        print("[loader] Text-only Gemma-3 checkpoint detected -> AutoTokenizer + AutoModelForCausalLM")
        proc = AutoTokenizer.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForCausalLM, model_id)
        is_processor = False

    model.eval()

    if hasattr(model, "hf_device_map"):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    return model, proc, is_processor, is_gemma4


def get_tokenizer(proc, is_processor):
    if is_processor and hasattr(proc, "tokenizer"):
        return proc.tokenizer
    return proc


def build_terminators(proc, is_processor):
    """Gemma turns end with <end_of_turn>; include it alongside eos."""
    tok = get_tokenizer(proc, is_processor)
    terminators = [tok.eos_token_id]
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if eot is not None and eot != tok.unk_token_id and eot not in terminators:
        terminators.append(eot)
    return terminators


# ---------------------------------------------------------------------------
# Reasoning / thinking-tag stripping
# ---------------------------------------------------------------------------

_THINK_PATTERNS = [
    re.compile(r"<\|channel\|>\s*thought.*?(?=<\|channel\|>|<\|message\|>|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>.*?<\|message\|>", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL),
]


def strip_reasoning(text, is_gemma4):
    if not isinstance(text, str):
        return ""
    if is_gemma4:
        for pat in _THINK_PATTERNS:
            text = pat.sub("", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Data loading (official train/test splits from the HF dataset)
# ---------------------------------------------------------------------------

def load_biogen_dataset():
    """
    Loads sinhala-nlp/BioGen. `source` is the linearised infobox
    (<TAG> key </TAG> value ...), `target` is the opening sentence of the
    Sinhala Wikipedia article.
    """
    print("Loading BioGen dataset from HuggingFace...")
    ds = load_dataset("sinhala-nlp/BioGen")
    train_df = ds['train'].to_pandas().reset_index(drop=True)
    test_df = ds['test'].to_pandas().reset_index(drop=True)
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
        if pd.notna(row['source']) and pd.notna(row['target']) and \
                str(row['source']).strip() and str(row['target']).strip():
            few_shot_examples.append({
                'source': str(row['source']),
                'target': str(row['target'])
            })

    return few_shot_examples


# ---------------------------------------------------------------------------
# Prompting (keep wording identical across model families)
# ---------------------------------------------------------------------------

def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert in the Sinhala language. Using the following Wikipedia infobox (I) about a person, write the opening sentence of their Sinhala Wikipedia biography."
    action_desc = "Return only the Sinhala sentence following the prefix 'Biography:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පුද්ගලයෙකු පිළිබඳ පහත විකිපීඩියා තොරතුරු කොටුව (I) භාවිත කරමින්, එම පුද්ගලයාගේ සිංහල විකිපීඩියා චරිතාපදානයේ ආරම්භක වාක්‍යය ලියන්න."
    action_desc_si = "'Biography:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            source = example['source']
            source_preview = source[:FEWSHOT_PREVIEW_CHARS] + "..." if len(source) > FEWSHOT_PREVIEW_CHARS else source
            examples_str += f"\nExample {i}:\n"
            examples_str += f"I: {source_preview}\n"
            examples_str += f"Biography: {example['target']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} I: {row['source']}"

    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} I: {row['source']}"

    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow write the opening sentence for this infobox:\nI: {row['source']}"

    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මෙම තොරතුරු කොටුව සඳහා ආරම්භක වාක්‍යය ලියන්න:\nI: {row['source']}"

    else:
        content = f"{task_desc} {action_desc} I: {row['source']}"

    return [{"role": "user", "content": content}]


def to_processor_messages(messages):
    """Processor chat templates (multimodal Gemma-3 / Gemma-4) expect content as a
    list of typed parts. Wrap plain-string content in a single text part."""
    converted = []
    for m in messages:
        content = m["content"]
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        converted.append({"role": m["role"], "content": content})
    return converted


# ---------------------------------------------------------------------------
# Generation (batched, left-padded)
# ---------------------------------------------------------------------------

def query(model, proc, is_processor, is_gemma4, messages_list,
          max_new_tokens=256, batch_size=8, do_sample=False):
    tok = get_tokenizer(proc, is_processor)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    terminators = build_terminators(proc, is_processor)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    sample_kwargs = dict(temperature=1.0, top_p=0.95, top_k=64) if do_sample else {}

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

        if is_gemma4:
            try:
                inputs = proc.apply_chat_template(batch, enable_thinking=False, **template_kwargs)
            except TypeError:
                inputs = proc.apply_chat_template(batch, **template_kwargs)
        else:
            inputs = proc.apply_chat_template(batch, **template_kwargs)

        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                eos_token_id=terminators,
                pad_token_id=pad_id,
                **sample_kwargs,
            )

        new_tokens = generated[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        assistant_outputs.extend(strip_reasoning(t, is_gemma4) for t in decoded)

    return assistant_outputs


def extract_biography(response):
    """Extract the generated sentence after the 'Biography:' prefix (first line only)."""
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    try:
        matches = re.findall(r'Biography:\s*(.*?)(?:\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if matches and matches[0].strip():
            return matches[0].strip()

        lines = [line.strip() for line in response.split("\n") if line.strip()]
        return lines[0] if lines else ""
    except Exception as e:
        print(f"Error extracting biography: {e}")
        return ""


# ---------------------------------------------------------------------------
# Prediction driver
# ---------------------------------------------------------------------------

def predict(model, proc, is_processor, is_gemma4, model_id, train_df, test_df,
            max_new_tokens=256, batch_size=8, do_sample=False):
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

    print("Generating biographies...")
    responses = query(model, proc, is_processor, is_gemma4, df['chat'].tolist(),
                      max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=do_sample)
    df['responses'] = responses

    print("Extracting biographies...")
    df['preds'] = df['responses'].apply(extract_biography)

    n_empty = int((df['preds'].str.len() == 0).sum())
    n_missing_prefix = int((~df['responses'].str.contains('Biography:', case=False, na=False)).sum())
    print(f"Empty predictions: {n_empty}/{len(df)}")
    print(f"Responses without 'Biography:' prefix: {n_missing_prefix}/{len(df)}")

    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.drop(columns=['chat']).to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    print("Evaluating with ROUGE...")
    rouge_results = score_corpus(df['target'].astype(str).tolist(), df['preds'].astype(str).tolist())
    for t in ROUGE_TYPES:
        df[t] = rouge_results[t]['scores']

    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_rouge.csv")
    df.drop(columns=['chat']).to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with ROUGE scores saved to: {results_file}")

    summary_file = os.path.join(OUTPUT_FOLDER, "rouge_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("ROUGE Evaluation Results (whitespace-tokenized, F1 x100)\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: sinhala-nlp/BioGen (official test split)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=1.0,p=0.95,k=64)' if do_sample else 'greedy'}\n")
        f.write(f"Empty predictions: {n_empty}/{len(df)}\n")
        f.write(f"Responses without 'Biography:' prefix: {n_missing_prefix}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance from train set)\n")
            f.write(f"Few-shot infobox truncation: {FEWSHOT_PREVIEW_CHARS} chars\n")
        f.write("=" * 70 + "\n\n")

        for t in ROUGE_TYPES:
            f.write(f"{t.upper()}:\n")
            for stat in ['mean', 'median', 'std', 'min', 'max']:
                f.write(f"  {stat.capitalize() + ':':8}{rouge_results[t][stat]:.4f}\n")
            f.write("\n")

    print("\n" + "=" * 70)
    for t in ROUGE_TYPES:
        print(f"{t.upper()} mean: {rouge_results[t]['mean']:.4f}")
    print("=" * 70)
    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), rouge_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='google/gemma-3-27b-it', required=False,
                        help='HF model id (Gemma-4 or any Gemma-3 checkpoint; loader auto-detects type)')
    parser.add_argument('--query_type', type=str, default='zero-shot', required=False,
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8, required=False,
                        help='Number of prompts decoded per generation call')
    parser.add_argument('--max_new_tokens', type=int, default=256, required=False,
                        help='Max new tokens to generate per instance')
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Gemma sampling (t=1.0,p=0.95,k=64) instead of greedy.')

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

    train_df, test_df = load_biogen_dataset()

    model, proc, is_processor, is_gemma4 = load_model(MODEL_ID)

    OUTPUT_FOLDER = os.path.join("outputs", "biography_generation", MODEL_ID.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predict(
        model, proc, is_processor, is_gemma4, MODEL_ID, train_df, test_df,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, do_sample=args.do_sample,
    )