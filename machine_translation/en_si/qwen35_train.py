import argparse
import os
import re
import tarfile
import urllib.request
from dataclasses import dataclass
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from tqdm.auto import tqdm
from transformers import (
    AutoConfig,
    AutoProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

# Qwen3.5/3.6 are multimodal; their cards load them with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers.
try:
    from transformers import AutoModelForMultimodalLM as _AutoMultimodal
except ImportError:
    try:
        from transformers import AutoModelForImageTextToText as _AutoMultimodal
    except ImportError:
        _AutoMultimodal = None

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
set_seed(777)


# --------------------------------------------------------------------------- #
# Data loading (identical to the prompting MT scripts)
# --------------------------------------------------------------------------- #
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
    """dev (997) and devtest (1012) DataFrames with columns 'english'/'sinhala'.
    The splits are document-disjoint (FLORES divides 842 articles, not
    individual sentences), so training on dev and testing on devtest does not
    leak sibling sentences from the same article."""
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


def load_train_pairs(train_tsv, dev_df, train_size):
    """FLORES-200 ships no training split. By default we fine-tune on `dev`
    (997 pairs) and evaluate on `devtest` (1012) -- disjoint, so no leakage,
    but ~1k pairs is a very small supervised MT signal. --train_tsv points at
    a larger external En-Si parallel corpus (columns english/sinhala) if you
    want a supervised baseline in the usual sense."""
    if train_tsv:
        sep = "," if train_tsv.endswith(".csv") else "\t"
        train_df = pd.read_csv(train_tsv, sep=sep)
        missing = {"english", "sinhala"} - set(train_df.columns)
        if missing:
            raise ValueError(f"{train_tsv} is missing column(s): {sorted(missing)}")
        print(f"Loaded external training corpus: {train_tsv} ({len(train_df)} pairs)")
    else:
        train_df = dev_df.copy()
        print(f"No --train_tsv given: fine-tuning on the FLORES-200 dev split "
              f"({len(train_df)} pairs), evaluating on devtest")

    train_df = train_df[train_df['english'].notna() & train_df['sinhala'].notna()].copy()
    train_df = train_df[train_df['english'].astype(str).str.strip().astype(bool)
                        & train_df['sinhala'].astype(str).str.strip().astype(bool)].copy()

    if train_size and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=777).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_df)} pairs (seed=777)")
    elif train_size:
        print(f"Train set has {len(train_df)} pairs; --train_size {train_size} is a no-op")
    return train_df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Prompt text  (identical instruction wording to the prompting eval scripts so
# scores stay comparable). Unlike SinLlama, Qwen instruct checkpoints HAVE a
# chat template, so training uses the same chat wrapper the prompting runs use
# -- no Alpaca format here.
# --------------------------------------------------------------------------- #
TASK_DESC_EN = ("You are an expert translator specializing in English to Sinhala translation. Translate the "
                "following English sentence (E) into Sinhala accurately while preserving the meaning and context.")
ACTION_DESC_EN = ("Return only the Sinhala translation following the prefix 'Translation:' without any other "
                  "text or explanations.")
TASK_DESC_SI = ("ඔබ ඉංග්‍රීසි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත ඉංග්‍රීසි වාක්‍යය (E) අර්ථය සහ "
                "සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න.")
ACTION_DESC_SI = ("'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                  "විස්තරයක් එක් නොකරන්න.")

TARGET_PREFIX = "Translation:"


def build_user_content(english: str, prompt_lang: str) -> str:
    if prompt_lang == "si":
        return f"{TASK_DESC_SI} {ACTION_DESC_SI} E: {english}"
    return f"{TASK_DESC_EN} {ACTION_DESC_EN} E: {english}"


def build_prompt_ids(tok, english: str, prompt_lang: str, enable_thinking: bool) -> List[int]:
    """Render the chat template to TEXT, then tokenize separately.

    apply_chat_template(tokenize=True) returns a BatchEncoding dict rather than
    a flat list of ids; indexing it silently yields the wrong thing and the
    labels end up misaligned with no visible error. tokenize=False avoids that
    entirely. add_special_tokens=False because the template already emits them.
    """
    messages = [{"role": "user", "content": build_user_content(english, prompt_lang)}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        # Older templates don't accept enable_thinking.
        text = tok.apply_chat_template(messages, **kwargs)
    return tok(text, add_special_tokens=False)["input_ids"]


def turn_end_id(tok):
    """Qwen turns end with <|im_end|>, which may differ from the default EOS."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id and im_end >= 0:
        return im_end
    return tok.eos_token_id


# --------------------------------------------------------------------------- #
# Length control
# --------------------------------------------------------------------------- #
# FLORES sentences are short, so this rarely fires -- but a naive tail
# truncation would cut off the target (and the generation prompt), leaving
# labels that are all -100, i.e. a silently dead training example. Keep the
# target intact and remove tokens from the MIDDLE of the prompt, which
# preserves both the system/instruction head and the assistant-turn header.
def truncate_prompt_ids(prompt_ids: List[int], budget: int):
    if budget <= 0:
        raise ValueError("Token budget for the prompt is non-positive; raise --max_seq_len.")
    if len(prompt_ids) <= budget:
        return prompt_ids, False
    head = budget // 2
    tail = budget - head
    return prompt_ids[:head] + prompt_ids[-tail:], True


# --------------------------------------------------------------------------- #
# Training example builder  (mask the prompt; end the turn with <|im_end|>)
# --------------------------------------------------------------------------- #
def build_training_example(tok, prompt_ids: List[int], sinhala: str, max_len: int, end_id: int):
    target_text = f"{TARGET_PREFIX} {sinhala}"
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [end_id]

    prompt_ids, truncated = truncate_prompt_ids(prompt_ids, max_len - len(target_ids))

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}, truncated


def build_train_dataset(tok, train_df, prompt_lang, max_len, enable_thinking):
    end_id = turn_end_id(tok)
    examples, n_trunc, seq_lens = [], 0, []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        prompt_ids = build_prompt_ids(tok, str(row['english']), prompt_lang, enable_thinking)
        ex, truncated = build_training_example(tok, prompt_ids, str(row['sinhala']), max_len, end_id)
        n_trunc += int(truncated)
        seq_lens.append(len(ex["input_ids"]))
        examples.append(ex)
    print(f"Sequence length: mean={np.mean(seq_lens):.0f} p95={np.percentile(seq_lens, 95):.0f} "
          f"max={np.max(seq_lens)}")
    print(f"Prompt-truncated examples: {n_trunc}/{len(examples)} "
          f"-- raise --max_seq_len if this is non-zero")
    return Dataset.from_list(examples)


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
# Output post-processing (Qwen thinking-aware)
# --------------------------------------------------------------------------- #
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_translation(response: str):
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)
    m = re.search(r'Translation:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip().split('\n\n')[0].strip()
        cand = cand.split('\n')[0].strip() if cand else cand
        if cand:
            return cand, True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[0] if lines else ""), False


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens, do_sample) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same
    chat rendering and truncation rule as training."""
    outputs = []
    pad_id = tok.pad_token_id
    end_id = turn_end_id(tok)
    terminators = list({tok.eos_token_id, end_id} - {None})
    # Qwen-recommended non-thinking sampling params (used only if do_sample)
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      eos_token_id=terminators, pad_token_id=pad_id, **sample_kwargs)

    for start in tqdm(range(0, len(prompt_id_lists), batch_size), desc="Generating translations"):
        batch = prompt_id_lists[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids, attn = [], []
        for ids in batch:                                  # left padding
            pad = width - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))

        enc = {"input_ids": torch.tensor(input_ids, dtype=torch.long).to(model.device),
               "attention_mask": torch.tensor(attn, dtype=torch.long).to(model.device)}

        with torch.no_grad():
            gen = model.generate(**enc, **gen_common)
        outputs.extend(tok.batch_decode(gen[:, width:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# BLEU
# --------------------------------------------------------------------------- #
# Copied verbatim from the prompting MT scripts. This is a *sentence-level*
# BLEU averaged over instances, with no smoothing: if any n-gram order has zero
# matches the geometric mean is 0, so a single sentence with no 4-gram match
# scores 0 overall. It is NOT corpus BLEU and is not comparable to published
# FLORES numbers. Kept unchanged so the fine-tuned run is comparable to the
# prompting runs; corpus sacreBLEU is reported alongside.
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
        reference, prediction = row['sinhala'], row['preds']
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
    Sinhala-safe tokenization used everywhere else in SinGen; sacreBLEU's
    default 13a tokenizer mangles Sinhala the same way \\b\\w+\\b does."""
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
# Evaluation on the held-out devtest split
# --------------------------------------------------------------------------- #
def evaluate(model, tok, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, max_seq_len, do_sample, enable_thinking):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    budget = max_seq_len - max_new_tokens
    prompt_ids, n_trunc = [], 0
    for eng in df['english']:
        ids = build_prompt_ids(tok, str(eng), prompt_lang, enable_thinking)
        ids, truncated = truncate_prompt_ids(ids, budget)
        n_trunc += int(truncated)
        prompt_ids.append(ids)
    if n_trunc:
        print(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_translation(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Translation:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for residual <think> if high")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    bleu_results = evaluate_bleu_scores(df)
    corpus = corpus_sacrebleu(df['sinhala'].tolist(), df['preds'].tolist())
    if corpus is not None:
        print(f"\nCorpus sacreBLEU (tokenize=none): {corpus:.4f}")

    df.to_csv(os.path.join(output_folder, "predictions_with_bleu.csv"), index=False, encoding='utf-8')

    with open(os.path.join(output_folder, "bleu_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, chat template)\n")
        f.write("Dataset: FLORES-200 English-Sinhala (devtest split)\n")
        f.write(f"Prompt language: {prompt_lang}\nDataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\nBatch Size: {batch_size}\n")
        f.write(f"Thinking enabled: {enable_thinking}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
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

    return bleu_results, corpus


# --------------------------------------------------------------------------- #
# Model loading + LoRA target selection
# --------------------------------------------------------------------------- #
def load_qwen(model_id, dtype, device_map):
    """Mirror the prompting scripts' loader so the fine-tuned model is the same
    object that was evaluated zero/few-shot: multimodal checkpoints go through
    AutoProcessor, text-only ones through AutoTokenizer."""
    config = AutoConfig.from_pretrained(model_id)
    is_multimodal = getattr(config, "vision_config", None) is not None

    if is_multimodal and _AutoMultimodal is not None:
        print("[loader] Multimodal Qwen checkpoint -> AutoProcessor + multimodal class")
        proc = AutoProcessor.from_pretrained(model_id)
        tok = proc.tokenizer if hasattr(proc, "tokenizer") else proc
        model = _AutoMultimodal.from_pretrained(model_id, dtype=dtype, device_map=device_map)
    else:
        print("[loader] Text-only Qwen checkpoint -> AutoTokenizer + AutoModelForCausalLM")
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, device_map=device_map)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, config


_VISION_HINTS = ("vision", "visual", "image", "patch_embed", "mm_projector", "merger")
_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def select_lora_targets(model, config, attention_only):
    """Build an explicit target-module list instead of using "all-linear".

    Two reasons "all-linear" is wrong for these checkpoints:
      1. On multimodal Qwen it attaches adapters to the vision tower, which
         never sees a gradient on a text-only task -- wasted parameters and
         memory.
      2. On MoE checkpoints (e.g. 35B-A3B) it attaches an adapter to every
         expert MLP. Each adapter then only receives gradient on the tokens
         routed to its expert, so rarely-routed experts barely train while
         memory use balloons. Attention-only is the standard choice there.
    """
    n_experts = getattr(config, "num_experts", None) or getattr(config, "num_local_experts", None)
    text_cfg = getattr(config, "text_config", None)
    if n_experts is None and text_cfg is not None:
        n_experts = getattr(text_cfg, "num_experts", None) or getattr(text_cfg, "num_local_experts", None)
    is_moe = bool(n_experts and n_experts > 1)

    if is_moe and not attention_only:
        print(f"[lora] MoE checkpoint detected ({n_experts} experts) -> restricting LoRA to "
              f"attention projections. Override with --all_linear.")
        attention_only = True

    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(h in name.lower() for h in _VISION_HINTS):
            continue
        if name.endswith("lm_head"):
            continue
        if attention_only and not name.endswith(_ATTN_SUFFIXES):
            continue
        names.append(name)

    if not names:
        raise RuntimeError("No LoRA target modules found; inspect model.named_modules().")
    suffixes = sorted({n.split('.')[-1] for n in names})
    print(f"[lora] {len(names)} target modules, suffixes: {suffixes}")
    return names


# --------------------------------------------------------------------------- #
# Hugging Face Hub upload
# --------------------------------------------------------------------------- #
def write_model_card(path, repo_id, model_id, prompt_lang, bleu, corpus, args, n_train):
    sent = bleu['bleu_overall']['mean'] if bleu else None
    card = f"""---
license: apache-2.0
language:
- si
- en
base_model: {model_id}
library_name: peft
tags:
- sinhala
- translation
- machine-translation
- singen
- lora
datasets:
- facebook/flores
metrics:
- bleu
---

# {repo_id.split('/')[-1]}

A LoRA adapter for **English to Sinhala machine translation**, fine-tuned from
[{model_id}](https://huggingface.co/{model_id}) as part of the SinGen Sinhala text
generation benchmark.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

tok = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained("{model_id}", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "{repo_id}")
```

Prompts use the base model's chat template with thinking disabled, and the assistant
response begins with the `{TARGET_PREFIX}` prefix.

## Training

| | |
|---|---|
| Training pairs | {n_train} |
| Instruction language | `{prompt_lang}` |
| Epochs | {args.num_train_epochs} |
| Effective batch size | {args.train_batch_size * args.grad_accum} |
| Learning rate | {args.learning_rate} |
| Max sequence length | {args.max_seq_len} |
| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |
| Thinking during training | {args.enable_thinking} |

## Evaluation

FLORES-200 English-Sinhala `devtest` (1012 sentences), whitespace-tokenized (sacreBLEU's
default 13a tokenizer splits Sinhala conjuncts and vowel signs):

| Metric | Score |
|---|---|
| Corpus sacreBLEU | {f'{corpus:.2f}' if corpus is not None else 'n/a'} |
| Sentence-level BLEU mean | {f'{sent:.2f}' if sent is not None else 'n/a'} |

## Licence

Inherits the licence of the base model; verify on the base model card before
redistributing.
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
                          repo_type="model", commit_message="Add Qwen En-Si translation LoRA")
    print(f"Uploaded: https://huggingface.co/{repo_id}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3.5-9B')
    parser.add_argument('--prompt_lang', type=str, default='en', choices=['en', 'si'],
                        help="Instruction language used for BOTH training and evaluation.")
    # data
    parser.add_argument('--train_tsv', type=str, default=None,
                        help='Optional external En-Si parallel corpus (columns english/sinhala). '
                             'Default: the FLORES-200 dev split (997 pairs).')
    parser.add_argument('--train_size', type=int, default=0,
                        help='Optional random subsample of the training pairs (seed=777). '
                             '0 (default) uses every pair.')
    # training
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=512)
    parser.add_argument('--enable_thinking', action='store_true',
                        help='Train and evaluate with the thinking channel on (default: off).')
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--attention_only', action='store_true',
                        help='Attach LoRA to attention projections only (forced on for MoE).')
    parser.add_argument('--all_linear', action='store_true',
                        help='Override the MoE guard and target every non-vision linear layer.')
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--do_sample', action='store_true',
                        help='Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/<model>-FLORES200-En2Si-<prompt_lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    model_tag = args.model_id.split('/')[-1]
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/{model_tag}-FLORES200-En2Si-{prompt_lang}"
    print(f"Model: {args.model_id}")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (chat template)")
    print(f"Thinking: {'enabled' if args.enable_thinking else 'disabled'}")
    print("Task: English to Sinhala translation (FLORES-200)")

    OUTPUT_FOLDER = os.path.join("outputs", "english_sinhala_translation_finetuned", model_tag, prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ data
    dev_df, devtest_df = download_and_load_flores_en_si()
    if dev_df is None or devtest_df is None:
        print("Error: Could not load required dataset splits")
        raise SystemExit(1)
    train_df = load_train_pairs(args.train_tsv, dev_df, args.train_size)

    # ------------------------------------------------------------------ load
    model, tok, config = load_qwen(args.model_id, torch.bfloat16, "auto")
    model.config.use_cache = False   # required with gradient checkpointing

    targets = select_lora_targets(model, config, args.attention_only and not args.all_linear)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    train_ds = build_train_dataset(tok, train_df, prompt_lang, args.max_seq_len, args.enable_thinking)
    print(f"Train examples: {len(train_ds)}  Test (devtest): {len(devtest_df)}")

    steps = int(len(train_ds) * args.num_train_epochs /
                (args.train_batch_size * args.grad_accum))
    print(f"Approximate optimizer steps: {steps}")
    if steps < 100:
        print("WARNING: fewer than 100 optimizer steps. Consider more epochs or "
              "a larger parallel corpus via --train_tsv.")

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
    bleu, corpus = evaluate(model, tok, devtest_df, args.model_id, prompt_lang, OUTPUT_FOLDER,
                            args.eval_batch_size, args.max_new_tokens, args.max_seq_len,
                            args.do_sample, args.enable_thinking)

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.model_id, prompt_lang,
                         bleu, corpus, args, len(train_ds))
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)