import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, PeftModel, TaskType

set_seed(777)


# --------------------------------------------------------------------------- #
# SinLlama specifics
# --------------------------------------------------------------------------- #
# SinLlama_v01 is a *base* (not instruct-tuned) continual-pretraining LoRA on
# Meta-Llama-3-8B with a Sinhala-extended tokenizer (vocab 128256 -> 139336).
# It has NO chat template, so we can't reuse the apply_chat_template path used
# by the prompting scripts. The card documents Alpaca-style prompts.
#
# Measured Pali-side fertility on this corpus is 3.72 tokens/word (p95 4.91),
# so the extended Sinhala vocabulary does not cover Pali orthography well even
# where the script is shared. That is a major reason sequences run long here.
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"        # gated -> needs HF_TOKEN
SINLLAMA_ADAPTER = "polyglots/SinLlama_v01"
SINLLAMA_TOKENIZER = "polyglots/Extended-Sinhala-LLaMA"

# Meta-Llama-3-8B has an 8192-token context window; nothing above this is
# usable, so it is both the default and the ceiling for --max_seq_len.
MODEL_CONTEXT_LIMIT = 8192


# --------------------------------------------------------------------------- #
# Prompt text  (identical instruction wording to the prompting eval script so
# scores stay comparable; only the *wrapper* changes chat-template -> Alpaca)
# --------------------------------------------------------------------------- #
TASK_DESC_EN = ("You are an expert translator specializing in Pali to Sinhala translation. Translate the "
                "following Pali text (P) into Sinhala accurately while preserving the meaning and context.")
ACTION_DESC_EN = ("Return only the Sinhala translation following the prefix 'Translation:' without any other "
                  "text or explanations.")
TASK_DESC_SI = ("ඔබ පාලි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත පාලි පාඨය (P) අර්ථය සහ "
                "සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න.")
ACTION_DESC_SI = ("'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                  "විස්තරයක් එක් නොකරන්න.")

TARGET_PREFIX = "Translation:"

ALPACA_HEADER = ("Below is an instruction that describes a task, paired with an input that provides further "
                 "context. Write a response that appropriately completes the request.")


def build_prompt(pali: str, prompt_lang: str) -> str:
    """Alpaca-style prompt ending at '### Response:\n' (generation starts here)."""
    if prompt_lang == "si":
        instr = f"{TASK_DESC_SI} {ACTION_DESC_SI}"
    else:
        instr = f"{TASK_DESC_EN} {ACTION_DESC_EN}"
    return (f"{ALPACA_HEADER}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\nP: {pali}\n\n"
            f"### Response:\n")


# --------------------------------------------------------------------------- #
# Data loading (same cleaning + tail-1000 test split as the prompting script)
# --------------------------------------------------------------------------- #
def load_splits(test_size, train_size):
    print("Loading Pali-Sinhala dataset from HuggingFace...")
    full_df = load_dataset("sinhala-nlp/pali-sinhala")['train'].to_pandas()

    # Remove leading verse/section numbers from both columns (as in the
    # prompting script) -- otherwise the model learns to emit them.
    print("Cleaning dataset (removing leading numbers)...")
    full_df['pali_text'] = full_df['pali_text'].str.replace(r'^\d+\s+', '', regex=True)
    full_df['sinhala_text'] = full_df['sinhala_text'].str.replace(r'^\d+\s+', '', regex=True)

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    # Test = last N rows, train = everything before, matching the prompting script.
    #
    # CAVEAT worth stating in the paper: this corpus is ordered by canonical
    # text and the alignment granularity is not uniform. The tail is far longer
    # than the body -- measured medians are 580 source / 368 target tokens in
    # the test tail against 47 / 47 in the train pool. The held-out set is
    # therefore not a sample of the same distribution the model trains on.
    test_size = min(test_size, len(full_df))
    test_df = full_df.tail(test_size).copy()
    train_df = full_df.head(len(full_df) - test_size).copy()
    print(f"Test (tail): {len(test_df)}  Train pool (head): {len(train_df)}")

    for df_ in (train_df, test_df):
        df_.dropna(subset=['pali_text', 'sinhala_text'], inplace=True)
    train_df = train_df[train_df['pali_text'].astype(str).str.strip().astype(bool)
                        & train_df['sinhala_text'].astype(str).str.strip().astype(bool)].copy()
    print(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")

    # Leakage check. The Pali canon is heavily formulaic, with stock passages
    # repeated verbatim across suttas, so a positional head/tail split can put
    # identical or near-identical segments on both sides. Exact duplicates are
    # counted below; near-duplicates are NOT caught by this check.
    overlap = set(test_df['pali_text'].astype(str)) & set(train_df['pali_text'].astype(str))
    if overlap:
        print(f"WARNING: {len(overlap)}/{len(test_df)} test Pali segments appear verbatim in the "
              f"train pool. Use --drop_train_overlap to remove them from training.")
    else:
        print("No exact Pali-side overlap between train pool and test tail "
              "(near-duplicates are not detected by this check).")

    if train_size and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=777).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_df)} pairs (seed=777)")
    elif train_size:
        print(f"Train pool has {len(train_df)} pairs; --train_size {train_size} is a no-op")
    else:
        print(f"Training on the full pool: {len(train_df)} pairs")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), overlap


# --------------------------------------------------------------------------- #
# Length control
# --------------------------------------------------------------------------- #
# Two separate mechanisms, for two separate failure modes:
#
#   SKIP  - if the target alone leaves less than --min_prompt_tokens of budget,
#           the example is dropped. Truncating the target instead would teach
#           the model to emit cut-off translations, and there would be no
#           prompt left to condition on. (This is the case that previously
#           raised ValueError and killed the job during dataset construction.)
#
#   TRUNCATE - if the target fits but the prompt does not, tokens are removed
#           from the MIDDLE of the prompt, preserving the instruction header
#           and the '### Response:\n' marker.
def truncate_prompt_ids(prompt_ids: List[int], budget: int):
    if budget <= 0:
        raise ValueError("Token budget for the prompt is non-positive; raise --max_seq_len.")
    if len(prompt_ids) <= budget:
        return prompt_ids, False
    head = budget // 2
    tail = budget - head
    return prompt_ids[:head] + prompt_ids[-tail:], True


# --------------------------------------------------------------------------- #
# Training example builder  (BOS on prompt, EOS after target; mask the prompt)
# --------------------------------------------------------------------------- #
def build_training_example(tok, prompt_text: str, sinhala: str, max_len: int):
    prompt_ids = tok(prompt_text, add_special_tokens=True)["input_ids"]
    target_text = f"{TARGET_PREFIX} {sinhala}"
    # append EOS so the base model learns to STOP (base LMs otherwise ramble)
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    prompt_ids, truncated = truncate_prompt_ids(prompt_ids, max_len - len(target_ids))

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}, truncated


def build_train_dataset(tok, train_df, prompt_lang, max_len, min_prompt_tokens):
    examples, n_trunc, n_skip = [], 0, 0
    seq_lens, tgt_lens, src_fert = [], [], []

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        pali, sinhala = str(row['pali_text']), str(row['sinhala_text'])

        # Measure the target first: it decides whether the row is usable at all.
        n_tgt = len(tok(f"{TARGET_PREFIX} {sinhala}", add_special_tokens=False)["input_ids"]) + 1
        tgt_lens.append(n_tgt)
        if max_len - n_tgt < min_prompt_tokens:
            n_skip += 1
            continue

        ex, truncated = build_training_example(tok, build_prompt(pali, prompt_lang), sinhala, max_len)
        n_trunc += int(truncated)
        seq_lens.append(len(ex["input_ids"]))
        words = max(len(pali.split()), 1)
        src_fert.append(len(tok(pali, add_special_tokens=False)["input_ids"]) / words)
        examples.append(ex)

    print(f"Target length: p50={np.percentile(tgt_lens, 50):.0f} p95={np.percentile(tgt_lens, 95):.0f} "
          f"p99={np.percentile(tgt_lens, 99):.0f} max={np.max(tgt_lens)}")
    print(f"Sequence length: mean={np.mean(seq_lens):.0f} p95={np.percentile(seq_lens, 95):.0f} "
          f"max={np.max(seq_lens)}")
    print(f"Pali-side fertility (tokens per word): mean={np.mean(src_fert):.2f} "
          f"p95={np.percentile(src_fert, 95):.2f}")
    print(f"SKIPPED (target alone exceeds max_seq_len={max_len} minus {min_prompt_tokens}): "
          f"{n_skip}/{len(train_df)} ({100 * n_skip / max(len(train_df), 1):.2f}%)")
    print(f"Prompt-truncated (middle removed, target intact): {n_trunc}/{len(examples)}")
    if n_skip / max(len(train_df), 1) > 0.05:
        print("NOTE: more than 5% of the corpus was dropped. State this in the paper -- "
              "it changes what the model was trained on.")
    return Dataset.from_list(examples), n_skip


# --------------------------------------------------------------------------- #
# Data collator (right-padding for training)
# --------------------------------------------------------------------------- #
@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids, labels, attn = [], [], []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            labels.append(f["labels"] + [-100] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }


# --------------------------------------------------------------------------- #
# Output post-processing  (Alpaca-aware; cut at the next section header)
# --------------------------------------------------------------------------- #
def extract_translation(response: str):
    if not isinstance(response, str):
        return "", False
    text = response.strip()
    m = re.search(r'Translation:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        # References here are multi-sentence passages, not one-liners, so keep
        # the whole block up to the next section rather than the first line.
        cand = re.split(r'###', m.group(1))[0].strip()
        if cand:
            return cand, True
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith('#') and 'Translation' not in ln:
            return ln, False
    return "", False


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; length-sorted batching)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens, do_sample) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same
    truncation rule as training.

    Prompts are sorted by length before batching and restored to the original
    order afterwards. A batch keeps generating until every sequence in it hits
    EOS, so pairing a 200-token reference with a 3000-token one spends the
    whole budget on the short one. Sorting cuts eval wall-clock substantially
    at these lengths without changing the greedy outputs.
    """
    pad_id = tok.pad_token_id
    sample_kwargs = dict(temperature=0.7, top_p=0.9, top_k=50) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=pad_id, **sample_kwargs)

    order = sorted(range(len(prompt_id_lists)), key=lambda i: len(prompt_id_lists[i]))
    results: List[Any] = [None] * len(prompt_id_lists)

    for start in tqdm(range(0, len(order), batch_size), desc="Generating translations"):
        idxs = order[start:start + batch_size]
        batch = [prompt_id_lists[i] for i in idxs]
        width = max(len(ids) for ids in batch)
        input_ids, attn = [], []
        for ids in batch:                                  # left padding
            pad = width - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))

        enc = {"input_ids": torch.tensor(input_ids, dtype=torch.long).to(model.device),
               "attention_mask": torch.tensor(attn, dtype=torch.long).to(model.device)}

        with torch.no_grad():
            try:
                gen = model.generate(**enc, tokenizer=tok, stop_strings=["###"], **gen_common)
            except TypeError:
                gen = model.generate(**enc, **gen_common)
        decoded = tok.batch_decode(gen[:, width:], skip_special_tokens=True)
        for i, text in zip(idxs, decoded):
            results[i] = text

    return results


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #
# Copied verbatim from the prompting MT scripts. This is a *sentence-level*
# BLEU averaged over instances, with no smoothing: if any n-gram order has zero
# matches the geometric mean is 0. It is NOT corpus BLEU and is not comparable
# to published figures. Kept unchanged so the fine-tuned run is comparable to
# the prompting runs; corpus sacreBLEU is reported alongside.
def tokenize(text):
    if pd.isna(text) or text is None:
        return []
    return str(text).strip().split()


def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
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
        ref_ngrams, pred_ngrams = {}, {}
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
    print("\nCalculating BLEU scores...")
    cols = {k: [] for k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']}

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing BLEU"):
        reference, prediction = row['sinhala_text'], row['preds']
        if (pd.isna(reference) or pd.isna(prediction)
                or not str(reference).strip() or not str(prediction).strip()):
            for k in cols:
                cols[k].append(0.0)
            continue
        scores = calculate_bleu_score_individual(str(reference), str(prediction))
        for k in cols:
            cols[k].append(scores[k])

    for k, v in cols.items():
        df[k] = v

    print("\n" + "=" * 70)
    print("BLEU Score Evaluation Results (sentence-level mean):")
    print("=" * 70)
    for k in cols:
        s = cols[k]
        print(f"\n{k.upper().replace('_', '-')}:")
        print(f"  Mean:   {np.mean(s):.4f}\n  Median: {np.median(s):.4f}\n  Std:    {np.std(s):.4f}")
        print(f"  Min:    {np.min(s):.4f}\n  Max:    {np.max(s):.4f}")
    print("=" * 70)

    def stats(s):
        return {'mean': np.mean(s), 'median': np.median(s), 'std': np.std(s),
                'min': np.min(s), 'max': np.max(s), 'scores': s}

    return {k: stats(v) for k, v in cols.items()}


def corpus_sacrebleu(refs: List[str], preds: List[str]):
    """Standard corpus-level BLEU, reported alongside the sentence-level mean.
    `tokenize='none'` with whitespace-split text keeps it consistent with the
    Sinhala-safe tokenization used everywhere else in SinGen."""
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed -- skipping corpus BLEU.")
        return None
    pairs = [(str(r), str(p)) for r, p in zip(refs, preds)
             if str(r).strip() and str(p).strip()]
    if not pairs:
        return None
    r, p = zip(*pairs)
    return sacrebleu.corpus_bleu(list(p), [list(r)], tokenize="none").score


# --------------------------------------------------------------------------- #
# Evaluation on the held-out test tail
# --------------------------------------------------------------------------- #
def evaluate(model, tok, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, max_seq_len, do_sample, n_overlap):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    budget = max_seq_len - max_new_tokens
    prompt_ids, n_trunc = [], 0
    for pali in df['pali_text']:
        ids = tok(build_prompt(str(pali), prompt_lang), add_special_tokens=True)["input_ids"]
        ids, truncated = truncate_prompt_ids(ids, budget)
        n_trunc += int(truncated)
        prompt_ids.append(ids)
    print(f"Prompt budget: {budget} tokens "
          f"(max_seq_len {max_seq_len} - max_new_tokens {max_new_tokens})")
    print(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}")

    # How many references cannot fit in the generation budget? Those produce
    # cut-off predictions and get hammered by BLEU's brevity penalty regardless
    # of translation quality.
    ref_lens = [len(tok(f"{TARGET_PREFIX} {s}", add_special_tokens=False)["input_ids"])
                for s in df['sinhala_text'].astype(str)]
    n_over = int(sum(1 for l in ref_lens if l > max_new_tokens))
    print(f"References longer than max_new_tokens: {n_over}/{len(df)} "
          f"(p95={np.percentile(ref_lens, 95):.0f} p99={np.percentile(ref_lens, 99):.0f})")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_translation(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Translation:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    bleu_results = evaluate_bleu_scores(df)
    corpus = corpus_sacrebleu(df['sinhala_text'].tolist(), df['preds'].tolist())
    if corpus is not None:
        print(f"\nCorpus sacreBLEU (tokenize=none): {corpus:.4f}")

    df.to_csv(os.path.join(output_folder, "predictions_with_bleu.csv"), index=False, encoding='utf-8')

    with open(os.path.join(output_folder, "bleu_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, Alpaca format)\n")
        f.write("Dataset: sinhala-nlp/pali-sinhala (last rows as test)\n")
        f.write(f"Prompt language: {prompt_lang}\nDataset Size: {len(df)} samples\n")
        f.write(f"Max Seq Len: {max_seq_len}  Max New Tokens: {max_new_tokens}  "
                f"Prompt budget: {budget}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.9,k=50)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}\n")
        f.write(f"References exceeding the generation budget: {n_over}/{len(df)}\n")
        f.write(f"Exact train/test Pali-side overlap detected: {n_overlap}\n")
        f.write("NOTE: the test tail is not distributionally matched to the train pool "
                "(median lengths differ by roughly an order of magnitude).\n")
        if corpus is not None:
            f.write(f"Corpus sacreBLEU (tokenize=none): {corpus:.4f}\n")
        f.write("Primary metric below is the sentence-level BLEU mean used by the "
                "prompting scripts (not corpus BLEU).\n")
        f.write("=" * 70 + "\n\n")
        for k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
            f.write(f"{k.upper().replace('_', '-')}:\n")
            for stat in ['mean', 'median', 'std', 'min', 'max']:
                f.write(f"  {stat.capitalize():7s} {bleu_results[k][stat]:.4f}\n")
            f.write("\n")

    return bleu_results, corpus, n_over


# --------------------------------------------------------------------------- #
# Model loading: base -> resize vocab -> load SinLlama adapter -> merge
# --------------------------------------------------------------------------- #
def load_sinllama(base_id, adapter_id, tokenizer_id, dtype, device_map):
    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForCausalLM.from_pretrained(base_id, dtype=dtype, device_map=device_map)

    # SinLlama extends the vocab (128256 -> 139336). Resize BEFORE loading the
    # adapter so the trained Sinhala embedding rows (saved as modules_to_save on
    # the adapter) line up and overwrite the freshly-initialised rows.
    if base.get_input_embeddings().weight.shape[0] != len(tok):
        print(f"Resizing embeddings: {base.get_input_embeddings().weight.shape[0]} -> {len(tok)}")
        base.resize_token_embeddings(len(tok))

    print(f"Loading SinLlama adapter: {adapter_id}")
    model = PeftModel.from_pretrained(base, adapter_id)
    model = model.merge_and_unload()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


# --------------------------------------------------------------------------- #
# Hugging Face Hub upload
# --------------------------------------------------------------------------- #
def write_model_card(path, repo_id, base_id, adapter_id, prompt_lang, bleu, corpus,
                     args, n_train, n_test, n_skip, n_over):
    sent = bleu['bleu_overall']['mean'] if bleu else None
    card = f"""---
license: llama3
language:
- si
- pi
base_model: {adapter_id}
library_name: peft
tags:
- sinhala
- pali
- translation
- machine-translation
- singen
- lora
datasets:
- sinhala-nlp/pali-sinhala
metrics:
- bleu
---

# {repo_id.split('/')[-1]}

Built with Meta Llama 3.

A LoRA adapter for **Pali to Sinhala machine translation**, trained on
[sinhala-nlp/pali-sinhala](https://huggingface.co/datasets/sinhala-nlp/pali-sinhala) as part
of the SinGen Sinhala text generation benchmark.

The task adapter is stacked on top of [{adapter_id}](https://huggingface.co/{adapter_id}),
a Sinhala continual-pretraining adapter over [{base_id}](https://huggingface.co/{base_id})
with an extended Sinhala tokenizer.

## Usage

The base model must be loaded with the extended tokenizer and resized embeddings, the
SinLlama adapter merged in, and only then this task adapter applied:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{SINLLAMA_TOKENIZER}")
base = AutoModelForCausalLM.from_pretrained("{base_id}", dtype="auto", device_map="auto")
base.resize_token_embeddings(len(tok))
model = PeftModel.from_pretrained(base, "{adapter_id}").merge_and_unload()
model = PeftModel.from_pretrained(model, "{repo_id}")
```

Prompts follow the Alpaca format used during training, ending at `### Response:` with the
model continuing from the `{TARGET_PREFIX}` prefix. Leading verse/section numbers were
stripped from both sides of the corpus before training.

## Training

| | |
|---|---|
| Training pairs used | {n_train} |
| Pairs dropped (target exceeded the sequence budget) | {n_skip} |
| Instruction language | `{prompt_lang}` |
| Epochs | {args.num_train_epochs} |
| Effective batch size | {args.train_batch_size * args.grad_accum} |
| Learning rate | {args.learning_rate} |
| Max sequence length | {args.max_seq_len} |
| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |
| Target modules | all-linear |

## Evaluation

Held-out tail of the corpus ({n_test} segments), whitespace-tokenized (sacreBLEU's default
13a tokenizer splits Sinhala conjuncts and vowel signs):

| Metric | Score |
|---|---|
| Corpus sacreBLEU | {f'{corpus:.2f}' if corpus is not None else 'n/a'} |
| Sentence-level BLEU mean | {f'{sent:.2f}' if sent is not None else 'n/a'} |

### Caveats

- The split is positional (last {n_test} rows). The corpus is ordered by canonical text and
  aligned at uneven granularity, so the test tail contains far longer segments than the
  training body: it is not a sample of the same distribution.
- {n_over} of {n_test} references exceed the generation budget of {args.max_new_tokens}
  tokens; those predictions are cut off and penalised by BLEU's brevity term.
- The Pali canon repeats stock passages verbatim, so near-duplicate overlap between splits
  is possible even where exact overlap is zero.

## Licence

Derived from Meta Llama 3 and governed by the
[Meta Llama 3 Community License](https://llama.meta.com/llama3/license/).
"""
    with open(os.path.join(path, "README.md"), "w", encoding="utf-8") as f:
        f.write(card)


def push_adapter(adapter_dir, repo_id, private=False):
    from huggingface_hub import HfApi, create_repo

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set -- skipping upload. Export a WRITE-scoped token to push.")
        return
    print(f"Uploading adapter to https://huggingface.co/{repo_id} ...")
    create_repo(repo_id, token=token, private=private, exist_ok=True, repo_type="model")
    HfApi().upload_folder(folder_path=adapter_dir, repo_id=repo_id, token=token,
                          repo_type="model", commit_message="Add SinLlama Pali-Sinhala translation LoRA")
    print(f"Uploaded: https://huggingface.co/{repo_id}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default=BASE_MODEL)
    parser.add_argument('--adapter', type=str, default=SINLLAMA_ADAPTER)
    parser.add_argument('--tokenizer', type=str, default=SINLLAMA_TOKENIZER)
    parser.add_argument('--prompt_lang', type=str, default='en', choices=['en', 'si'],
                        help="Instruction language used for BOTH training and evaluation.")
    # data
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Held-out tail used as the test set (matches the prompting script).')
    parser.add_argument('--train_size', type=int, default=0,
                        help='Optional random subsample of the train pool (seed=777). '
                             '0 (default) uses every available pair.')
    parser.add_argument('--drop_train_overlap', action='store_true',
                        help='Remove train pairs whose Pali side appears verbatim in the test tail.')
    parser.add_argument('--min_prompt_tokens', type=int, default=128,
                        help='Drop a training row if the target leaves less than this much '
                             'prompt budget. Truncating the target instead would teach the '
                             'model to emit cut-off translations.')
    # training
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--train_batch_size', type=int, default=1,
                        help='1 by default: at max_seq_len 8192 the 139k-row softmax makes '
                             'logits the dominant memory term, and batch 1 means no padding.')
    parser.add_argument('--grad_accum', type=int, default=16,
                        help='Keeps the effective batch at 16, matching the other SinGen runs.')
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=MODEL_CONTEXT_LIMIT,
                        help=f"Prompt + target budget. Ceiling is the {MODEL_CONTEXT_LIMIT}-token "
                             f"Llama-3 context window.")
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=3000,
                        help='Covers p99 of the test-tail references (2980 tokens). At the '
                             'previous default of 200, 65 percent of references were truncated.')
    parser.add_argument('--do_sample', action='store_true',
                        help='Sampling instead of greedy at eval time.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/SinLlama-PaliSinhala-Pi2Si-<prompt_lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    if args.max_seq_len > MODEL_CONTEXT_LIMIT:
        raise SystemExit(f"--max_seq_len {args.max_seq_len} exceeds the model's "
                         f"{MODEL_CONTEXT_LIMIT}-token context window.")
    if args.max_new_tokens >= args.max_seq_len - args.min_prompt_tokens:
        raise SystemExit(f"--max_new_tokens {args.max_new_tokens} leaves under "
                         f"{args.min_prompt_tokens} tokens for the prompt at "
                         f"--max_seq_len {args.max_seq_len}.")

    model_tag = args.adapter.split('/')[-1]   # -> SinLlama_v01
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/SinLlama-PaliSinhala-Pi2Si-{prompt_lang}"
    print(f"Model: {args.adapter} (base {args.base_model})")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (Alpaca format)")
    print("Task: Pali to Sinhala translation")
    print(f"max_seq_len={args.max_seq_len}  max_new_tokens={args.max_new_tokens}  "
          f"prompt budget={args.max_seq_len - args.max_new_tokens}")

    OUTPUT_FOLDER = os.path.join("outputs", "pali_sinhala_translation_finetuned", model_tag, prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ data
    # Loaded before the model so a dataset failure surfaces in seconds, not
    # after a 16GB checkpoint download.
    train_df, test_df, overlap = load_splits(args.test_size, args.train_size)
    if overlap and args.drop_train_overlap:
        before = len(train_df)
        train_df = train_df[~train_df['pali_text'].astype(str).isin(overlap)].reset_index(drop=True)
        print(f"Dropped {before - len(train_df)} overlapping train pairs")

    # ------------------------------------------------------------------ load
    model, tok = load_sinllama(args.base_model, args.adapter, args.tokenizer,
                               torch.bfloat16, "auto")
    model.config.use_cache = False   # required with gradient checkpointing

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    train_ds, n_skip = build_train_dataset(tok, train_df, prompt_lang,
                                           args.max_seq_len, args.min_prompt_tokens)
    print(f"Train examples: {len(train_ds)}  Test: {len(test_df)}")

    steps = int(len(train_ds) * args.num_train_epochs /
                (args.train_batch_size * args.grad_accum))
    print(f"Approximate optimizer steps: {steps}")

    collator = CausalCollator(pad_token_id=tok.pad_token_id)

    training_args = TrainingArguments(
        output_dir=os.path.join(OUTPUT_FOLDER, "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=777,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        data_collator=collator,
    )

    # ------------------------------------------------------------------ train
    trainer.train()

    adapter_dir = os.path.join(OUTPUT_FOLDER, "lora_adapter")
    if args.save_adapter or args.push_to_hub:
        model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        print(f"Saved task LoRA adapter to {adapter_dir}")

    # ------------------------------------------------------------------ eval
    bleu, corpus, n_over = evaluate(model, tok, test_df, args.adapter, prompt_lang, OUTPUT_FOLDER,
                                    args.eval_batch_size, args.max_new_tokens, args.max_seq_len,
                                    args.do_sample, len(overlap))

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.base_model, args.adapter,
                         prompt_lang, bleu, corpus, args, len(train_ds), len(test_df),
                         n_skip, n_over)
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)