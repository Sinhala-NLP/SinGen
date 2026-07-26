import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

from xlsum_loader import load_xlsum_sinhala

# Sinhala-safe ROUGE (whitespace-tokenized). Same module as the headline and the
# Qwen summarisation runs, so ROUGE-L is directly comparable across the board.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

SYSTEM_MSG = "You are an expert in Sinhala language summarization."


# --------------------------------------------------------------------------- #
# What differs from the Qwen summarisation script
# --------------------------------------------------------------------------- #
#   * AutoTokenizer + AutoModelForCausalLM (merged SinLlama checkpoints are
#     text-only LlamaForCausalLM, not multimodal).
#   * <|eot_id|> added to the generation stop set.
#   * Llama special-token cleanup in post-processing.
#   * Context guard: Llama-3-8B is 8192 tokens vs 32k+ on the Qwen checkpoints.
#     XL-Sum articles (+ three few-shot previews) overrun that on a real slice
#     of the test set, which manifests as garbage/empty preds rather than an
#     error. Articles are truncated at token level to fit and the count is
#     reported so the paper can state it.
# Prompt wording and the four query types are byte-identical to the prompting
# eval. Zero/few-shot ONLY -- no fine-tuning.


# --------------------------------------------------------------------------- #
# Few-shot + prompting (identical wording to the Qwen/Gemma summarisation runs)
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
# Context fitting
# --------------------------------------------------------------------------- #
# NOTE: apply_chat_template(tokenize=True) returns a BatchEncoding on some
# transformers versions rather than a flat id list, so length is measured via
# tokenize=False + a separate tokenizer call (the same fix used in the LoRA
# training scripts).
def n_prompt_tokens(tok, messages) -> int:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fit_prompt(tok, row, few_shot_examples, budget):
    """Build the chat, truncating the article body if the prompt exceeds budget.

    Returns (messages, was_truncated). Few-shot examples are left intact so the
    demonstration set stays identical to the other model families; only the
    article under test is shortened.
    """
    msgs = format_chat(row, few_shot_examples)
    n = n_prompt_tokens(tok, msgs)
    if n <= budget:
        return msgs, False

    row = row.copy()
    art_ids = tok(str(row['text']), add_special_tokens=False)["input_ids"]
    keep = max(64, len(art_ids) - (n - budget) - 32)
    for _ in range(4):
        row['text'] = tok.decode(art_ids[:keep], skip_special_tokens=True)
        msgs = format_chat(row, few_shot_examples)
        n = n_prompt_tokens(tok, msgs)
        if n <= budget:
            break
        keep = max(64, keep - (n - budget) - 64)
    return msgs, True


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
# Llama-3 has no reasoning channel; the <think> strip is a harmless no-op kept
# for symmetry with the Qwen path. The Llama tag strip is the part that matters.
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)
_LLAMA_TAGS = re.compile(r'<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    text = _LLAMA_TAGS.sub('', text)
    return text.strip()


def extract_summary(response: str):
    """Return (prediction, matched_marker). Never dumps the whole raw response."""
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
# Generation (AutoTokenizer, batched, left-padded)
# --------------------------------------------------------------------------- #
def build_terminators(tok):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return sorted(ids)


def generate(model, tok, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = build_terminators(tok)
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating summaries"):
        batch = list_of_messages[start:start + batch_size]

        inputs = tok.apply_chat_template(
            batch, add_generation_prompt=True, tokenize=True,
            padding=True, return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids or None,
                **sample_kwargs,
            )
        outputs.extend(tok.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def predict(model, tok, model_id, batch_size, max_new_tokens, do_sample, test_size, budget):
    print("Loading XL-Sum Sinhala dataset (direct archive read)...")
    train_df, _, test_df = load_xlsum_sinhala()
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")

    df = test_df.copy()
    if test_size and test_size > 0:
        # The official Sinhala test split is exactly 500, so --test_size 500 is a
        # no-op rather than a subsample.
        df = df.head(test_size).copy()
    print(f"Using {len(df)} test samples")

    msgs, truncated = [], []
    use_fs = QUERY_TYPE in ["few-shot", "few-shot-si"]
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing prompts")):
        fs = get_few_shot_examples_for_instance(train_df, instance_idx=idx, num_examples=3, seed=42) if use_fs else None
        m, was_trunc = fit_prompt(tok, row, fs, budget)
        msgs.append(m)
        truncated.append(was_trunc)
    df['chat'] = msgs
    df['article_truncated'] = truncated

    n_trunc = int(sum(truncated))
    print(f"Articles truncated to fit {budget}-token prompt budget: {n_trunc}/{len(df)}")

    responses = generate(model, tok, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_summary(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Summary:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

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
        f.write(f"Prompt budget: {budget} tokens  Articles truncated: {n_trunc}/{len(df)}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write("Metric: ROUGE F1 x100, whitespace-tokenized (Sinhala-safe)\n" + "=" * 60 + "\n")
        for t in ROUGE_TYPES:
            r = rouge[t]
            f.write(f"{t}:\n  Mean: {r['mean']:.4f}\n  Std: {r['std']:.4f}\n  Median: {r['median']:.4f}\n"
                    f"  Min: {r['min']:.4f}\n  Max: {r['max']:.4f}\n")

    return df['preds'].tolist(), rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True,
                        help='Path to a merged checkpoint directory.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Output folder name. Defaults to basename of --model_id.')
    parser.add_argument('--output_root', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base dir for outputs/. Defaults to the script's own directory.")
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--test_size', type=int, default=0,
                        help='0 = full test set (500 for Sinhala); else cap')
    parser.add_argument('--max_prompt_tokens', type=int, default=0,
                        help='0 = derive from model config (max_position_embeddings - max_new_tokens - 64)')
    parser.add_argument('--do_sample', action='store_true',
                        help='sampling (t=0.7,p=0.8,k=20) instead of greedy')
    args = parser.parse_args()

    QUERY_TYPE = args.query_type
    model_id = args.model_id
    tag = args.run_tag or os.path.basename(model_id.rstrip('/'))
    print(f"Model: {model_id}\nRun tag: {tag}\nQuery type: {QUERY_TYPE}\n"
          f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{model_id} has no chat_template; expected a merged checkpoint "
                         f"produced with --copy_chat_template.")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()

    ctx = int(getattr(model.config, "max_position_embeddings", 8192) or 8192)
    budget = args.max_prompt_tokens or max(512, ctx - args.max_new_tokens - 64)
    print(f"Model context: {ctx}  Prompt budget: {budget}  Vocab: {len(tok)}")

    OUTPUT_FOLDER = os.path.join(args.output_root, "outputs",
                                 "text_summarisation_zeroshot", tag, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    predict(model, tok, model_id, args.batch_size, args.max_new_tokens,
            args.do_sample, args.test_size, budget)