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

# Sinhala-safe ROUGE (whitespace-tokenized). Same module as the headline task,
# so summarisation and headline ROUGE-L are directly comparable.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

SYSTEM_MSG = "You are an expert in Sinhala language summarization."


# --------------------------------------------------------------------------- #
# Few-shot + prompting (from the original summarisation script)
# --------------------------------------------------------------------------- #
def get_few_shot_examples_for_instance(train_df, instance_idx, num_examples=3, seed=None):
    if seed is not None:
        random.seed(seed + instance_idx)
    idxs = random.sample(range(len(train_df)), min(num_examples, len(train_df)))
    out = []
    for idx in idxs:
        row = train_df.iloc[idx]
        if (pd.notna(row['text']) and pd.notna(row['summary'])
                and str(row['text']).strip() and str(row['summary']).strip()):
            out.append({'text': str(row['text']), 'summary': str(row['summary'])})
    return out


def build_prompt(row, few_shot_examples=None):
    task_desc = ("Imagine you are an expert in Sinhala language. Please provide a concise summary of the "
                 "following Sinhala news article. The summary should capture the main ideas and key points "
                 "while being significantly shorter than the original text.")
    action_desc = "Return the summary only following the prefix 'Summary:' without any other text or explanations."
    task_desc_si = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත සිංහල පුවත් ලිපියේ සංක්ෂිප්ත සාරාංශයක් ලබා දෙන්න. "
                    "සාරාංශය මුල් පාඨයට වඩා බෙහෙවින් කෙටි විය යුතු අතර ප්‍රධාන අදහස් සහ ප්‍රධාන කරුණු අන්තර්ගත විය යුතුය.")
    action_desc_si = ("'Summary:' යන ප්‍රත්‍යයයයෙන් පසුව පමණක් සාරාංශය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් "
                      "එක් නොකරන්න.")

    examples_str = ""
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples, 1):
            preview = ex['text'][:500] + "..." if len(ex['text']) > 500 else ex['text']
            examples_str += f"\nExample {i}:\nText: {preview}\nSummary: {ex['summary']}\n"

    if QUERY_TYPE == "zero-shot":
        return f"{task_desc} {action_desc} Text: {row['text']}"
    elif QUERY_TYPE == "zero-shot-si":
        return f"{task_desc_si} {action_desc_si} Text: {row['text']}"
    elif QUERY_TYPE == "few-shot":
        return f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow summarize this text:\nText: {row['text']}"
    elif QUERY_TYPE == "few-shot-si":
        return f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පාඨය සාරාංශ කරන්න:\nText: {row['text']}"
    return f"{task_desc} {action_desc} Text: {row['text']}"


def format_chat(row, few_shot_examples=None):
    return [{"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": build_prompt(row, few_shot_examples)}]


# --------------------------------------------------------------------------- #
# Output post-processing (Qwen3 thinking-aware)
# --------------------------------------------------------------------------- #
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_summary(response: str):
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)
    m = re.search(r'Summary:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip().split('\n\n')[0].strip()   # summaries may span lines; take first block
        if cand:
            return cand, True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


# --------------------------------------------------------------------------- #
# Generation (text-only causal LM + Qwen <think> handling)
# --------------------------------------------------------------------------- #
def generate(model, tokenizer, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating summaries"):
        batch = list_of_messages[start:start + batch_size]
        try:
            inputs = tokenizer.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True, enable_thinking=False)
        except TypeError:
            inputs = tokenizer.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=tokenizer.pad_token_id, **sample_kwargs)
        outputs.extend(tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def predict(model, tokenizer, model_id, batch_size, max_new_tokens, do_sample, test_size):
    print("Loading XL-Sum Sinhala dataset...")
    ds = load_dataset("csebuetnlp/xlsum", "sinhala", trust_remote_code=True)
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    df = test_df.copy()
    if test_size and test_size > 0:
        df = df.head(test_size).copy()   # NOTE: Table 1 lists |Test|=500 for summarisation
    print(f"Using {len(df)} test samples")

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        msgs = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            fs = get_few_shot_examples_for_instance(train_df, instance_idx=idx, num_examples=3, seed=42)
            msgs.append(format_chat(row, fs))
        df['chat'] = msgs
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    responses = generate(model, tokenizer, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_summary(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Summary:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for truncated/looping <think> if high")

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), index=False, encoding='utf-8')

    print("Evaluating with ROUGE (Sinhala-safe, whitespace tokenized)...")
    rouge = score_corpus(df['summary'].tolist(), df['preds'].tolist())
    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions_with_rouge.csv"), index=False, encoding='utf-8')

    print("\n" + "=" * 60 + "\nROUGE F1 (x100)\n" + "=" * 60)
    for t in ROUGE_TYPES:
        print(f"{t:8s} mean={rouge[t]['mean']:.4f}  median={rouge[t]['median']:.4f}  std={rouge[t]['std']:.4f}")
    print("=" * 60)

    with open(os.path.join(OUTPUT_FOLDER, "rouge_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nDataset: XL-Sum (sinhala)\n")
        f.write(f"Samples: {len(df)}\nDecoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write("Metric: ROUGE F1 x100, whitespace-tokenized (Sinhala-safe)\n" + "=" * 60 + "\n")
        for t in ROUGE_TYPES:
            r = rouge[t]
            f.write(f"{t}:\n  Mean: {r['mean']:.4f}\n  Std: {r['std']:.4f}\n  Median: {r['median']:.4f}\n"
                    f"  Min: {r['min']:.4f}\n  Max: {r['max']:.4f}\n")

    return df['preds'].tolist(), rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3-32B')
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--test_size', type=int, default=0, help='0 = full test set; else cap (Table 1 uses 500)')
    parser.add_argument('--do_sample', action='store_true', help='Qwen sampling (t=0.7,p=0.8,k=20) instead of greedy')
    args = parser.parse_args()

    QUERY_TYPE = args.query_type
    model_id = args.model_id
    print(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nDecoding: {'sampling' if args.do_sample else 'greedy'}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()

    OUTPUT_FOLDER = os.path.join("outputs", "text_summarisation", model_id.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predict(model, tokenizer, model_id, args.batch_size, args.max_new_tokens, args.do_sample, args.test_size)