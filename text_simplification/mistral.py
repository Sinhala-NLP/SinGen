import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    Mistral3ForConditionalGeneration,
    MistralCommonBackend,
    set_seed,
)

# Validated SARI (whitespace-tokenized, Sinhala-safe). Reused unchanged.
from sari_metric import sari_sentence

set_seed(777)


# --------------------------------------------------------------------------- #
# Prompting (identical to the Gemma / Qwen scripts)
# --------------------------------------------------------------------------- #
def get_few_shot_examples_for_instance(full_df, test_df, instance_idx, num_examples=3, seed=None):
    test_indices = set(test_df.index)
    available_indices = [i for i in full_df.index if i not in test_indices]
    if seed is not None:
        random.seed(seed + instance_idx)
    cols = ['Simplification 1', 'Simplification 2', 'Simplification 3']
    out = []
    shuffled = available_indices.copy()
    random.shuffle(shuffled)
    for idx in shuffled:
        if len(out) >= num_examples:
            break
        row = full_df.loc[idx]
        sims = [str(row[c]) for c in cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
        if sims:
            out.append({'complex': row['Complex'], 'simple': random.choice(sims)})
    return out


def format_chat(row, few_shot_examples=None):
    task_desc = ("Imagine you are an expert in Sinhala language. Please provide a simplified version of the "
                 "following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main "
                 "idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, "
                 "and replacing complex words with commonly used simple words.")
    action_desc = ("Return the simplified text only following the prefix 'Simplified text:' without any other "
                   "text or explanations.")
    task_desc_si = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. "
                    "ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න "
                    "(3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න.")
    action_desc_si = ("'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                      "විස්තරයක් එක් නොකරන්න.")

    examples_str = ""
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\nS: {ex['complex']}\nSimplified text: {ex['simple']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} S: {row['Complex']}"
    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} S: {row['Complex']}"
    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow simplify this sentence:\nS: {row['Complex']}"
    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය සරල කරන්න:\nS: {row['Complex']}"
    else:
        content = f"{task_desc} {action_desc} S: {row['Complex']}"
    # mistral-common expects plain-string content for text-only turns.
    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Output post-processing (Mistral-aware)
# --------------------------------------------------------------------------- #
# Ministral-3-Instruct is NOT a reasoning model, so it should not emit thinking
# blocks. This strip is defensive only: Mistral's *reasoning* checkpoints
# (Magistral family) use [THINK]...[/THINK] markers rather than Qwen's <think>.
# If you later benchmark a Magistral checkpoint, this will already handle it.
_MISTRAL_THINK = re.compile(r'\[THINK\].*?\[/THINK\]', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'\[/?THINK\]', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _MISTRAL_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_simplified(response: str):
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)
    m = re.search(r'Simplified text:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip().split('\n\n')[0].strip()
        cand = cand.splitlines()[0].strip() if cand else cand
        if cand:
            return cand, True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(model, tokenizer, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []

    # Left-padding is required for decoder-only batched generation.
    tokenizer.padding_side = "left"
    if getattr(tokenizer, "pad_token_id", None) is None and getattr(tokenizer, "eos_token", None) is not None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    # Neutral sampling params. Unlike Qwen, Mistral does not warn about greedy
    # repetition loops, so greedy (do_sample=False) is the reproducible default.
    # Tune these to the Ministral-3 card if you enable sampling.
    sample_kwargs = dict(temperature=0.7, top_p=0.95) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating"):
        batch = list_of_messages[start:start + batch_size]
        # NOTE: batched padding via MistralCommonBackend.apply_chat_template needs
        # transformers >= 5.7 (arg-validation was softened there). On older builds
        # this can raise; if so, upgrade transformers or run with --batch_size 1.
        inputs = tokenizer.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            padding=True,
            return_tensors="pt",
            return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=pad_id,
                                 **sample_kwargs)
        outputs.extend(tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def predict(model, tokenizer, model_id, batch_size, max_new_tokens, do_sample):
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    df = full.tail(200).copy()   # confirm this matches the official SiTSE split
    rest = full.head(len(full) - 200)

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        msgs = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            fs = get_few_shot_examples_for_instance(rest, df, instance_idx=idx, num_examples=3, seed=42)
            msgs.append(format_chat(row, fs))
        df['chat'] = msgs
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    responses = generate(model, tokenizer, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_simplified(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Simplified text:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for truncated output if high")

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), index=False, encoding='utf-8')

    ref_cols = [c for c in ['Simplification 1', 'Simplification 2', 'Simplification 3'] if c in df.columns]
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI"):
        refs = [str(row[c]) for c in ref_cols if pd.notna(row[c]) and str(row[c]).strip()]
        scores.append(sari_sentence(str(row['Complex']), str(row['preds']), refs)[0] if refs else 0.0)
    df['sari_score'] = scores
    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions_with_sari.csv"), index=False, encoding='utf-8')

    mean_sari = float(np.mean(scores))
    print("\n" + "=" * 60)
    print(f"SARI mean={mean_sari:.4f}  median={np.median(scores):.4f}  std={np.std(scores):.4f}")
    print("=" * 60)

    with open(os.path.join(OUTPUT_FOLDER, "sari_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nSamples: {len(df)}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.95)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Mean SARI: {mean_sari:.4f}\nMedian: {np.median(scores):.4f}\nStd: {np.std(scores):.4f}\n")

    return df['preds'].tolist(), mean_sari


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='mistralai/Ministral-3-14B-Instruct-2512')
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true',
                        help='Use sampling (t=0.7,p=0.95) instead of greedy decoding.')
    args = parser.parse_args()

    QUERY_TYPE = args.query_type
    model_id = args.model_id
    print(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nDecoding: {'sampling' if args.do_sample else 'greedy'}")

    # MistralCommonBackend wraps mistral-common. AutoTokenizer.from_pretrained
    # would also resolve to this backend automatically; the explicit class here
    # mirrors the Ministral-3 model card.
    tokenizer = MistralCommonBackend.from_pretrained(model_id)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        model_id, dtype="auto", device_map="auto")
    model.eval()

    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification", model_id.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    predict(model, tokenizer, model_id, args.batch_size, args.max_new_tokens, args.do_sample)