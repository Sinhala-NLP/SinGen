import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
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

# Sinhala-safe ROUGE (whitespace-tokenized). The stock rouge_score tokenizer
# strips non-ASCII and would zero out every Sinhala score.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

MAX_CONTENT_LENGTH = 2500   # characters; identical to the prompting headline scripts
RESERVE_TOKENS = 8          # slack for merge effects at the article/template boundary


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def char_trim(text: str, max_length: int = MAX_CONTENT_LENGTH) -> str:
    """Same character-level trim the prompting scripts apply, so the fine-tuned
    run sees articles cut at the same place."""
    text = str(text)
    return text[:max_length] + "..." if len(text) > max_length else text


def load_nsina(train_size: int, test_size: int, dedupe: bool):
    print("Loading NSINA-Headlines dataset...")
    ds = load_dataset("sinhala-nlp/NSINA-Headlines")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()
    print(f"Raw splits - Train: {len(train_df)}, Test: {len(test_df)}")

    for df in (train_df, test_df):
        df.drop(df.index[df['News Content'].isna() | df['Headline'].isna()], inplace=True)
    train_df = train_df[train_df['News Content'].astype(str).str.strip().astype(bool)
                        & train_df['Headline'].astype(str).str.strip().astype(bool)].copy()
    test_df = test_df[test_df['News Content'].astype(str).str.strip().astype(bool)
                      & test_df['Headline'].astype(str).str.strip().astype(bool)].copy()
    print(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")

    # The prompting scripts take test_df.head(test_size) -- NOT a random sample.
    # Mirrored exactly, otherwise the fine-tuned numbers sit on a different
    # subset from the zero/few-shot rows in the same table.
    eval_df = test_df.head(min(test_size, len(test_df))).copy()
    print(f"Eval set: first {len(eval_df)} test instances (matches the prompting scripts)")

    # --- leakage check ---------------------------------------------------- #
    # NSINA is scraped news; the same story can appear from multiple outlets or
    # be re-published. If a test article is also in train, the fine-tuned row in
    # the paper is contaminated in a way the prompting rows are not.
    train_articles = set(train_df['News Content'].astype(str).str.strip())
    overlap = sum(1 for a in eval_df['News Content'].astype(str).str.strip() if a in train_articles)
    print(f"[leakage] Exact test articles also present in train: {overlap}/{len(eval_df)}")
    if overlap:
        print("[leakage] Dropping the overlapping articles from the TRAINING set "
              "(the eval set is left untouched so it stays comparable).")
        eval_articles = set(eval_df['News Content'].astype(str).str.strip())
        keep = ~train_df['News Content'].astype(str).str.strip().isin(eval_articles)
        train_df = train_df[keep].copy()
        print(f"[leakage] Train size after removal: {len(train_df)}")

    if dedupe:
        before = len(train_df)
        train_df = train_df.drop_duplicates(subset=['News Content']).copy()
        print(f"[dedupe] Dropped {before - len(train_df)} duplicate training articles")

    if train_size and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=777).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_df)} articles (seed=777)")
    else:
        train_df = train_df.reset_index(drop=True)
        print(f"Using the full training split ({len(train_df)} articles)")

    # Length reporting: unlike FLORES, NSINA inputs are long and the char trim
    # actually fires, so this is worth having in the log for the paper.
    for name, df in (("train", train_df), ("eval", eval_df)):
        art = df['News Content'].astype(str).str.len()
        head = df['Headline'].astype(str).str.len()
        print(f"[{name}] article chars: mean={art.mean():.0f} p95={np.percentile(art, 95):.0f} "
              f"max={art.max()}  >{MAX_CONTENT_LENGTH}: {(art > MAX_CONTENT_LENGTH).sum()}")
        print(f"[{name}] headline chars: mean={head.mean():.0f} max={head.max()}")

    train_df['News Content'] = train_df['News Content'].apply(char_trim)
    eval_df['News Content'] = eval_df['News Content'].apply(char_trim)
    return train_df, eval_df


# --------------------------------------------------------------------------- #
# Prompt text  (identical wording to qwen35 headline prompting script so ROUGE
# stays comparable -- do not reword without updating every model script).
# Fine-tuned models are trained and evaluated ZERO-SHOT, so the comparable
# prompting rows are `zero-shot` (en) and `zero-shot-si` (si), not few-shot.
# --------------------------------------------------------------------------- #
TASK_DESC_EN = ("Imagine you are an expert in Sinhala language. Generate a concise and informative headline "
                "for the following Sinhala news article. The headline should capture the main point of the "
                "article in a brief, engaging manner.")
ACTION_DESC_EN = ("Return only the headline following the prefix 'Headline:' without any other text or "
                  "explanations.")
TASK_DESC_SI = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත සිංහල පුවත් ලිපිය සඳහා සංක්ෂිප්ත හා තොරතුරුදායක "
                "සිරස්තලයක් ජනනය කරන්න. සිරස්තලය කෙටි, ආකර්ෂණීය ආකාරයෙන් ලිපියේ ප්‍රධාන කරුණ ග්‍රහණය කර ගත යුතුය.")
ACTION_DESC_SI = ("'Headline:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිරස්තලය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් "
                  "එක් නොකරන්න.")

TARGET_PREFIX = "Headline:"


def build_user_content(article: str, prompt_lang: str) -> str:
    if prompt_lang == "si":
        return f"{TASK_DESC_SI} {ACTION_DESC_SI} News Content: {article}"
    return f"{TASK_DESC_EN} {ACTION_DESC_EN} News Content: {article}"


def build_prompt_ids(tok, article: str, prompt_lang: str, enable_thinking: bool) -> List[int]:
    """Render the chat template to TEXT, then tokenize separately.

    apply_chat_template(tokenize=True) returns a BatchEncoding dict rather than
    a flat list of ids; indexing it silently yields the wrong thing and the
    labels end up misaligned with no visible error. tokenize=False avoids that
    entirely. add_special_tokens=False because the template already emits them.
    """
    messages = [{"role": "user", "content": build_user_content(article, prompt_lang)}]
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        text = tok.apply_chat_template(messages, enable_thinking=enable_thinking, **kwargs)
    except TypeError:
        # Older templates don't accept enable_thinking.
        text = tok.apply_chat_template(messages, **kwargs)
    return tok(text, add_special_tokens=False)["input_ids"]


def scaffold_token_len(tok, prompt_lang: str, enable_thinking: bool) -> int:
    """Tokens consumed by instructions + chat template with an empty article.
    Measured once so the article can be budgeted against what's left."""
    n = len(build_prompt_ids(tok, "", prompt_lang, enable_thinking))
    print(f"[budget] instruction + template scaffold: {n} tokens (prompt_lang={prompt_lang})")
    return n


def turn_end_id(tok):
    """Qwen turns end with <|im_end|>, which may differ from the default EOS."""
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    if im_end is not None and im_end != tok.unk_token_id and im_end >= 0:
        return im_end
    return tok.eos_token_id


# --------------------------------------------------------------------------- #
# Length control
# --------------------------------------------------------------------------- #
# This is the part that differs materially from the FLORES MT script. There, the
# input was one sentence and truncation almost never fired. Here a 2500-char
# Sinhala article is well over a thousand tokens (Sinhala fertility on the Qwen
# BPE is poor -- every character is 3 UTF-8 bytes), so something WILL be cut on
# most examples. Two rules:
#   1. Cut the ARTICLE, from the tail. News is inverted-pyramid, so the lead
#      carries the headline-relevant content; middle-truncating the prompt (the
#      MT script's rule) would remove exactly that lead.
#   2. Never cut the target or the assistant-turn header. If rule 1 somehow
#      leaves the prompt over budget, fall back to middle truncation, which
#      preserves both ends -- and count it, because it should be ~0.
def fit_article(tok, article: str, allowance: int) -> Tuple[str, bool]:
    if allowance <= 0:
        raise ValueError("No token allowance left for the article; raise --max_seq_len.")
    ids = tok(article, add_special_tokens=False)["input_ids"]
    if len(ids) <= allowance:
        return article, False
    text = tok.decode(ids[:allowance], skip_special_tokens=True)
    # A token prefix can end mid-codepoint (byte-level BPE), which decodes to a
    # replacement char and would be a stray glyph inside Sinhala text. Dropping
    # the final whitespace-delimited word removes it and avoids splitting a
    # conjunct or orphaning a vowel sign.
    if " " in text.strip():
        text = text.rsplit(" ", 1)[0]
    return text.rstrip() + " ...", True


def truncate_prompt_ids(prompt_ids: List[int], budget: int) -> Tuple[List[int], bool]:
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
def build_training_example(tok, article: str, headline: str, prompt_lang: str,
                           max_len: int, end_id: int, scaffold: int, enable_thinking: bool):
    # Headlines are single-line by construction and extract_headline() only ever
    # reads the first line, so a stray newline in the reference would teach the
    # model to emit text the evaluator throws away.
    headline = " ".join(str(headline).split())
    target_text = f"{TARGET_PREFIX} {headline}"
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [end_id]

    prompt_budget = max_len - len(target_ids)
    allowance = prompt_budget - scaffold - RESERVE_TOKENS
    article, art_trunc = fit_article(tok, article, allowance)

    prompt_ids = build_prompt_ids(tok, article, prompt_lang, enable_thinking)
    prompt_ids, hard_trunc = truncate_prompt_ids(prompt_ids, prompt_budget)

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    ex = {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
    return ex, art_trunc, hard_trunc


def build_train_dataset(tok, train_df, prompt_lang, max_len, scaffold, enable_thinking):
    end_id = turn_end_id(tok)
    examples, seq_lens = [], []
    n_art, n_hard, n_chars, n_toks = 0, 0, 0, 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        article = str(row['News Content'])
        ex, art_trunc, hard_trunc = build_training_example(
            tok, article, row['Headline'], prompt_lang, max_len, end_id, scaffold, enable_thinking)
        n_art += int(art_trunc)
        n_hard += int(hard_trunc)
        n_chars += len(article)
        n_toks += len(tok(article, add_special_tokens=False)["input_ids"])
        seq_lens.append(len(ex["input_ids"]))
        examples.append(ex)

    print(f"Sequence length: mean={np.mean(seq_lens):.0f} p95={np.percentile(seq_lens, 95):.0f} "
          f"max={np.max(seq_lens)} (cap {max_len})")
    print(f"Article-truncated examples: {n_art}/{len(examples)} "
          f"({100 * n_art / max(len(examples), 1):.1f}%) -- raise --max_seq_len to reduce")
    print(f"Hard middle-truncated prompts: {n_hard}/{len(examples)} <-- should be 0")
    # Fertility is worth a line in the paper: it is the reason the Sinhala
    # context budget is so much tighter than the English one for the same model.
    print(f"[fertility] Sinhala article tokens/char = {n_toks / max(n_chars, 1):.3f} "
          f"({n_chars / max(n_toks, 1):.2f} chars/token)")
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
# Output post-processing (Qwen thinking-aware; copied from the prompting script)
# --------------------------------------------------------------------------- #
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_headline(response: str):
    """Return (headline, matched_marker). Never dumps the whole raw response."""
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)

    m = re.search(r'Headline:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip()
        cand = cand.splitlines()[0].strip() if cand else cand   # headline is one line
        if cand:
            return cand, True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False   # conservative fallback, flagged


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens, do_sample) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same chat
    rendering and article-budgeting rule as training."""
    outputs = []
    pad_id = tok.pad_token_id
    end_id = turn_end_id(tok)
    terminators = list({tok.eos_token_id, end_id} - {None})
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      eos_token_id=terminators, pad_token_id=pad_id, **sample_kwargs)

    for start in tqdm(range(0, len(prompt_id_lists), batch_size), desc="Generating headlines"):
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
# Evaluation on the held-out test slice
# --------------------------------------------------------------------------- #
def evaluate(model, tok, eval_df, model_id, prompt_lang, output_folder, scaffold,
             batch_size, max_new_tokens, max_seq_len, do_sample, enable_thinking):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = eval_df.copy()
    prompt_budget = max_seq_len - max_new_tokens
    allowance = prompt_budget - scaffold - RESERVE_TOKENS
    prompt_ids, n_art, n_hard = [], 0, 0
    for article in df['News Content']:
        art, art_trunc = fit_article(tok, str(article), allowance)
        ids = build_prompt_ids(tok, art, prompt_lang, enable_thinking)
        ids, hard_trunc = truncate_prompt_ids(ids, prompt_budget)
        n_art += int(art_trunc)
        n_hard += int(hard_trunc)
        prompt_ids.append(ids)
    print(f"Article-truncated eval instances: {n_art}/{len(df)}   hard-truncated: {n_hard}/{len(df)}")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_headline(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Headline:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for residual <think> if high")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    print("Evaluating with ROUGE (Sinhala-safe, whitespace tokenized)...")
    rouge = score_corpus(df['Headline'].tolist(), df['preds'].tolist())
    for t in ROUGE_TYPES:                      # per-instance scores if exposed
        if isinstance(rouge[t], dict) and 'scores' in rouge[t]:
            df[t] = rouge[t]['scores']
    df.to_csv(os.path.join(output_folder, "predictions_with_rouge.csv"), index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("ROUGE F1 (x100)")
    print("=" * 60)
    for t in ROUGE_TYPES:
        print(f"{t:8s} mean={rouge[t]['mean']:.4f}  median={rouge[t]['median']:.4f}  "
              f"std={rouge[t]['std']:.4f}")
    print("=" * 60)

    with open(os.path.join(output_folder, "rouge_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, chat template)\n")
        f.write(f"Query type: zero-shot ({prompt_lang})\nDataset: NSINA-Headlines (test split)\n")
        f.write(f"Samples: {len(df)}\nFormat misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Max new tokens: {max_new_tokens}\nBatch size: {batch_size}\n")
        f.write(f"Max seq len: {max_seq_len}\nThinking enabled: {enable_thinking}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Article-truncated eval instances: {n_art}/{len(df)}  hard-truncated: {n_hard}/{len(df)}\n")
        f.write("Metric: ROUGE F1 x100, whitespace-tokenized (Sinhala-safe)\n" + "=" * 60 + "\n")
        for t in ROUGE_TYPES:
            r = rouge[t]
            f.write(f"{t}:\n  Mean: {r['mean']:.4f}\n  Std: {r['std']:.4f}\n  Median: {r['median']:.4f}\n"
                    f"  Min: {r['min']:.4f}\n  Max: {r['max']:.4f}\n")

    return rouge


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

    if tok.chat_template is None:
        raise SystemExit(
            f"{model_id} has no chat template -- this looks like a base (non-instruct) "
            "checkpoint. Base checkpoints are not comparable to the instruct rows in SinGen.")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model.config.pad_token_id = tok.pad_token_id
    return model, tok, config


_VISION_HINTS = ("vision", "visual", "image", "patch_embed", "mm_projector", "merger")
_ATTN_SUFFIXES = ("q_proj", "k_proj", "v_proj", "o_proj")


def select_lora_targets(model, config, attention_only, force_all_linear=False):
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

    if is_moe and not attention_only and not force_all_linear:
        print(f"[lora] MoE checkpoint detected ({n_experts} experts) -> restricting LoRA to "
              f"attention projections. Override with --all_linear.")
        attention_only = True
    elif is_moe and force_all_linear:
        print(f"[lora] MoE checkpoint ({n_experts} experts) but --all_linear given -> "
              f"targeting expert MLPs too. Expect high memory use and undertrained experts.")

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
def write_model_card(path, repo_id, model_id, prompt_lang, rouge, args, n_train, n_eval):
    r1 = rouge['rouge1']['mean'] if rouge else None
    r2 = rouge['rouge2']['mean'] if rouge else None
    rl = rouge['rougeL']['mean'] if rouge and 'rougeL' in rouge else None
    card = f"""---
language:
- si
base_model: {model_id}
library_name: peft
tags:
- sinhala
- headline-generation
- summarization
- singen
- lora
datasets:
- sinhala-nlp/NSINA-Headlines
metrics:
- rouge
---

# {repo_id.split('/')[-1]}

A LoRA adapter for **Sinhala news headline generation**, fine-tuned from
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
response begins with the `{TARGET_PREFIX}` prefix. Articles are trimmed to
{MAX_CONTENT_LENGTH} characters and then budgeted to fit `max_seq_len` from the lead.

## Training

| | |
|---|---|
| Training articles | {n_train} |
| Instruction language | `{prompt_lang}` |
| Epochs | {args.num_train_epochs} |
| Effective batch size | {args.train_batch_size * args.grad_accum} |
| Learning rate | {args.learning_rate} |
| Max sequence length | {args.max_seq_len} |
| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |
| Thinking during training | {args.enable_thinking} |

## Evaluation

First {n_eval} instances of the NSINA-Headlines test split, ROUGE F1 x100 with
whitespace tokenization (the default `rouge_score` tokenizer strips non-ASCII and
zeroes out every Sinhala score):

| Metric | Score |
|---|---|
| ROUGE-1 | {f'{r1:.2f}' if r1 is not None else 'n/a'} |
| ROUGE-2 | {f'{r2:.2f}' if r2 is not None else 'n/a'} |
| ROUGE-L | {f'{rl:.2f}' if rl is not None else 'n/a'} |

## Licence

This adapter inherits the licence of the base model; check the base model card
before redistributing. The training data is NSINA-Headlines, derived from scraped
Sri Lankan news content -- verify its terms on the dataset card, as they may be
more restrictive than the base model's licence.
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
                          repo_type="model", commit_message="Add Qwen Sinhala headline generation LoRA")
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
    parser.add_argument('--train_size', type=int, default=8000,
                        help='Random subsample of the NSINA-Headlines train split (seed=777). '
                             'The full split is large; 0 uses all of it.')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='First N test instances -- must match the prompting scripts.')
    parser.add_argument('--no_dedupe', action='store_true',
                        help='Keep duplicate training articles (NSINA re-publishes stories).')
    # training
    parser.add_argument('--num_train_epochs', type=float, default=1.0,
                        help='1 epoch over a larger subsample beats 3 over a small one here -- '
                             'unlike FLORES, NSINA has plenty of training data.')
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=2560,
                        help='Must hold instructions + a ~2500-char Sinhala article + the '
                             'headline. Watch the article-truncation percentage in the log.')
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
    parser.add_argument('--eval_batch_size', type=int, default=2)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--do_sample', action='store_true',
                        help='Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy.')
    parser.add_argument('--no_save_adapter', action='store_true',
                        help='Skip writing the LoRA adapter to disk.')
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/<model>-NSINA-Headlines-<prompt_lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    # Fail before loading a 27B model rather than after twelve runs have finished.
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise SystemExit("--push_to_hub given but HF_TOKEN is empty; export a WRITE-scoped token.")

    model_tag = args.model_id.split('/')[-1]
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/{model_tag}-NSINA-Headlines-{prompt_lang}"
    print(f"Model: {args.model_id}")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (chat template)")
    print(f"Thinking: {'enabled' if args.enable_thinking else 'disabled'}")
    print("Task: Sinhala headline generation (NSINA-Headlines)")

    OUTPUT_FOLDER = os.path.join("outputs", "headline_generation_finetuned", model_tag, prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ data
    train_df, eval_df = load_nsina(args.train_size, args.test_size, dedupe=not args.no_dedupe)

    # ------------------------------------------------------------------ load
    model, tok, config = load_qwen(args.model_id, torch.bfloat16, "auto")
    model.config.use_cache = False   # required with gradient checkpointing

    targets = select_lora_targets(model, config, args.attention_only, args.all_linear)
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules=targets)
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    scaffold = scaffold_token_len(tok, prompt_lang, args.enable_thinking)
    train_ds = build_train_dataset(tok, train_df, prompt_lang, args.max_seq_len,
                                   scaffold, args.enable_thinking)
    print(f"Train examples: {len(train_ds)}  Eval: {len(eval_df)}")

    steps = int(len(train_ds) * args.num_train_epochs /
                (args.train_batch_size * args.grad_accum))
    print(f"Approximate optimizer steps: {steps}")
    if steps < 100:
        print("WARNING: fewer than 100 optimizer steps. Raise --train_size or --num_train_epochs.")

    collator = CausalCollator(pad_token_id=tok.pad_token_id)

    # NOTE: `group_by_length` / `length_column_name` were removed -- they are not
    # accepted by TrainingArguments in transformers v5, which raises on unknown
    # kwargs instead of ignoring them. Dropping them also keeps batch composition
    # identical to the Gemma/Llama scripts, and they saved nothing at
    # --train_batch_size 1 anyway.
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
    if not args.no_save_adapter or args.push_to_hub:
        model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        print(f"Saved task LoRA adapter to {adapter_dir}")

    # ------------------------------------------------------------------ eval
    rouge = evaluate(model, tok, eval_df, args.model_id, prompt_lang, OUTPUT_FOLDER, scaffold,
                     args.eval_batch_size, args.max_new_tokens, args.max_seq_len,
                     args.do_sample, args.enable_thinking)

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.model_id, prompt_lang,
                         rouge, args, len(train_ds), len(eval_df))
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)