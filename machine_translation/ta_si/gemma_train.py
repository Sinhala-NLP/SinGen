import argparse
import os
import re
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
    AutoTokenizer,
    AutoProcessor,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    set_seed,
)
from peft import LoraConfig, get_peft_model, TaskType

set_seed(777)


# --------------------------------------------------------------------------- #
# Gemma specifics
# --------------------------------------------------------------------------- #
# Unlike SinLlama (base model, Alpaca wrapper, extended vocab), the Gemma
# checkpoints here are all *instruct* models, so we keep the chat template used
# by the prompting scripts. Three things differ from the Llama path:
#
#   1. Multimodal checkpoints (gemma-3-27b-it / -12b-it, and the Gemma-4 family)
#      load through AutoModelForImageTextToText + AutoProcessor, not
#      AutoModelForCausalLM. Routing is on `vision_config` presence, matching
#      the prompting scripts.
#   2. `target_modules="all-linear"` would put LoRA on the SigLIP vision tower
#      as well -- the vision attention projections are also named q/k/v_proj.
#      We therefore enumerate full module paths under the language model only.
#   3. Gemma turns end with <end_of_turn>, not <eos>. Training targets are
#      terminated with <end_of_turn> so generation stops on its own.
#
# apply_chat_template is always called with tokenize=False and the text is
# tokenized separately -- tokenize=True returns a BatchEncoding rather than a
# flat id list on some versions, which silently corrupts the label mask.
DEFAULT_MODEL = "google/gemma-3-12b-it"

VISION_KEYS = ("vision_tower", "vision_model", "multi_modal_projector",
               "vision_encoder", "mm_projector", "audio_tower")


# --------------------------------------------------------------------------- #
# Prompt text  (identical instruction wording to the prompting eval scripts and
# to the SinLlama fine-tuning script, so scores stay comparable)
# --------------------------------------------------------------------------- #
TASK_DESC_EN = ("You are an expert translator specializing in Tamil to Sinhala translation. Translate the "
                "following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context.")
ACTION_DESC_EN = ("Return only the Sinhala translation following the prefix 'Translation:' without any other "
                  "text or explanations.")
TASK_DESC_SI = ("ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ "
                "සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න.")
ACTION_DESC_SI = ("'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                  "විස්තරයක් එක් නොකරන්න.")

TARGET_PREFIX = "Translation:"


def build_user_message(tamil: str, prompt_lang: str) -> str:
    if prompt_lang == "si":
        instr = f"{TASK_DESC_SI} {ACTION_DESC_SI}"
    else:
        instr = f"{TASK_DESC_EN} {ACTION_DESC_EN}"
    return f"{instr}\n\nT: {tamil}"


def apply_template(template_owner, user_text: str, multimodal: bool) -> str:
    """Render the user turn through the model's chat template, ending at the
    assistant generation prompt. `enable_thinking=False` is attempted first
    (Gemma-4); older templates reject the kwarg with a TypeError. Content is
    tried as a typed list first for processors, plain string first otherwise."""
    typed = [{"type": "text", "text": user_text}]
    contents = (typed, user_text) if multimodal else (user_text, typed)

    last_err = None
    for content in contents:
        msgs = [{"role": "user", "content": content}]
        for extra in ({"enable_thinking": False}, {}):
            try:
                out = template_owner.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, **extra)
            except Exception as e:            # noqa: BLE001 - template errors vary
                last_err = e
                continue
            if isinstance(out, str):
                return out
            last_err = TypeError(f"apply_chat_template returned {type(out)}, expected str")
    raise RuntimeError(f"Could not render chat template: {last_err}")


# --------------------------------------------------------------------------- #
# Data loading (same tail split as the prompting / SinLlama Ta=>Si scripts)
# --------------------------------------------------------------------------- #
def load_splits(tsv_file_path, test_size, train_size):
    print(f"Loading data from {tsv_file_path}...")
    full_df = pd.read_csv(tsv_file_path, sep='\t', encoding='utf-8')
    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")

    missing = {"Tamil", "Sinhala"} - set(full_df.columns)
    if missing:
        raise ValueError(f"{tsv_file_path} is missing column(s): {sorted(missing)}")

    test_size = min(test_size, len(full_df))
    test_df = full_df.tail(test_size).copy()
    train_df = full_df.head(len(full_df) - test_size).copy()
    print(f"Test (tail): {len(test_df)}  Train pool (head): {len(train_df)}")

    train_df = train_df[train_df['Tamil'].notna() & train_df['Sinhala'].notna()].copy()
    train_df = train_df[train_df['Tamil'].astype(str).str.strip().astype(bool)
                        & train_df['Sinhala'].astype(str).str.strip().astype(bool)].copy()
    test_df = test_df[test_df['Tamil'].notna() & test_df['Sinhala'].notna()].copy()
    print(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")

    # Positional head/tail split is not de-duplicated: mined parallel corpora
    # repeat pairs, so the model can be trained on its own test instances.
    overlap = set(test_df['Tamil'].astype(str)) & set(train_df['Tamil'].astype(str))
    if overlap:
        print(f"WARNING: {len(overlap)}/{len(test_df)} test Tamil sentences also appear in the "
              f"train pool. Use --drop_train_overlap to remove them from training.")
    else:
        print("No Tamil-side overlap between train pool and test tail.")

    if train_size and train_size < len(train_df):
        train_df = train_df.sample(n=train_size, random_state=777).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_df)} pairs (seed=777)")
    elif train_size:
        print(f"Train pool has {len(train_df)} pairs; --train_size {train_size} is a no-op")
    else:
        print(f"Training on the full pool: {len(train_df)} pairs")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), overlap


# --------------------------------------------------------------------------- #
# Length control -- drop from the MIDDLE of the prompt, never the tail, so the
# generation marker and the target survive truncation.
# --------------------------------------------------------------------------- #
def truncate_prompt_ids(prompt_ids: List[int], budget: int):
    if budget <= 0:
        raise ValueError("Token budget for the prompt is non-positive; raise --max_seq_len.")
    if len(prompt_ids) <= budget:
        return prompt_ids, False
    head = budget // 2
    tail = budget - head
    return prompt_ids[:head] + prompt_ids[-tail:], True


# --------------------------------------------------------------------------- #
# Training example builder (prompt masked; target terminated with <end_of_turn>)
# --------------------------------------------------------------------------- #
def encode_prompt(tok, prompt_text: str) -> List[int]:
    # The rendered template already carries <bos>; add_special_tokens=False
    # prevents a second one. Guard in case a template omits it.
    ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    if tok.bos_token_id is not None and (not ids or ids[0] != tok.bos_token_id):
        ids = [tok.bos_token_id] + ids
    return ids


def build_training_example(tok, prompt_text: str, sinhala: str, max_len: int, end_id: int):
    prompt_ids = encode_prompt(tok, prompt_text)
    target_ids = tok(f"{TARGET_PREFIX} {sinhala}", add_special_tokens=False)["input_ids"] + [end_id]
    prompt_ids, truncated = truncate_prompt_ids(prompt_ids, max_len - len(target_ids))

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}, truncated


def build_train_dataset(tok, template_owner, train_df, prompt_lang, max_len, end_id, multimodal):
    examples, n_trunc, lens = [], 0, []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        prompt_text = apply_template(template_owner, build_user_message(str(row['Tamil']), prompt_lang),
                                     multimodal)
        ex, truncated = build_training_example(tok, prompt_text, str(row['Sinhala']), max_len, end_id)
        n_trunc += int(truncated)
        lens.append(len(ex["input_ids"]))
        examples.append(ex)
    print(f"Sequence length: mean={np.mean(lens):.0f} p95={np.percentile(lens, 95):.0f} max={np.max(lens)}")
    print(f"Prompt-truncated examples: {n_trunc}/{len(examples)} "
          f"-- raise --max_seq_len if this is non-trivial")
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
# Output post-processing (thinking-aware; Gemma-4 can leak residual blocks even
# with enable_thinking=False)
# --------------------------------------------------------------------------- #
THINK_BLOCK = re.compile(r"<\s*(think|thinking|reasoning)\s*>.*?<\s*/\s*\1\s*>", re.DOTALL | re.IGNORECASE)
THINK_CLOSE = re.compile(r"<\s*/\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)
THINK_OPEN = re.compile(r"<\s*(think|thinking|reasoning)\s*>", re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = THINK_BLOCK.sub("", text)
    parts = THINK_CLOSE.split(text)
    if len(parts) > 1:
        text = parts[-1]                      # keep whatever follows the last close tag
    m = THINK_OPEN.search(text)
    if m:
        text = text[:m.start()]               # unterminated block -> drop the tail
    return text


def extract_translation(response: str):
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response).strip()
    m = re.search(r'Translation\s*:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip().split('\n\n')[0].strip()
        cand = cand.split('\n')[0].strip() if cand else cand
        if cand:
            return cand, True
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith(('#', '<', '*')) and 'Translation' not in ln:
            return ln, False
    return "", False


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens,
             do_sample, eos_ids: List[int]) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same
    truncation rule as training."""
    outputs = []
    pad_id = tok.pad_token_id
    sample_kwargs = dict(temperature=0.7, top_p=0.9, top_k=50) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=pad_id, eos_token_id=eos_ids, **sample_kwargs)

    for start in tqdm(range(0, len(prompt_id_lists), batch_size), desc="Generating translations"):
        batch = prompt_id_lists[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids, attn = [], []
        for ids in batch:                                  # left padding
            pad = width - len(ids)
            input_ids.append([pad_id] * pad + ids)
            attn.append([0] * pad + [1] * len(ids))

        device = next(model.parameters()).device
        enc = {"input_ids": torch.tensor(input_ids, dtype=torch.long).to(device),
               "attention_mask": torch.tensor(attn, dtype=torch.long).to(device)}

        with torch.no_grad():
            gen = model.generate(**enc, **gen_common)
        outputs.extend(tok.batch_decode(gen[:, width:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# BLEU  (copied verbatim from the prompting MT scripts)
# --------------------------------------------------------------------------- #
# Sentence-level BLEU averaged over instances, no smoothing: a single sentence
# with no 4-gram match scores 0 overall, which is why the Table 2 numbers sit
# near 1-2. NOT corpus BLEU and not comparable to published figures. Kept
# unchanged so fine-tuned runs are comparable to the prompting runs; corpus
# sacreBLEU is reported alongside.
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
        reference, prediction = row['Sinhala'], row['preds']
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
# Evaluation on the held-out test tail
# --------------------------------------------------------------------------- #
def evaluate(model, tok, template_owner, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, max_seq_len, do_sample, n_overlap, eos_ids, multimodal):
    model.eval()
    try:
        model.gradient_checkpointing_disable()
    except Exception:                          # noqa: BLE001
        pass
    set_use_cache(model, True)

    df = test_df.copy()
    budget = max_seq_len - max_new_tokens
    prompt_ids, n_trunc = [], 0
    for tamil in df['Tamil']:
        text = apply_template(template_owner, build_user_message(str(tamil), prompt_lang), multimodal)
        ids, truncated = truncate_prompt_ids(encode_prompt(tok, text), budget)
        n_trunc += int(truncated)
        prompt_ids.append(ids)
    print(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample, eos_ids)
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
    corpus = corpus_sacrebleu(df['Sinhala'].tolist(), df['preds'].tolist())
    if corpus is not None:
        print(f"\nCorpus sacreBLEU (tokenize=none): {corpus:.4f}")

    df.to_csv(os.path.join(output_folder, "predictions_with_bleu.csv"), index=False, encoding='utf-8')

    with open(os.path.join(output_folder, "bleu_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, chat template)\n")
        f.write("Dataset: TamSiPara Tamil-Sinhala (last rows as test)\n")
        f.write(f"Prompt language: {prompt_lang}\nDataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\nBatch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.9,k=50)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}\n")
        f.write(f"Train/test Tamil-side overlap detected: {n_overlap}\n")
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
# Model loading: multimodal-aware routing on vision_config
# --------------------------------------------------------------------------- #
def set_use_cache(model, value: bool):
    """Gemma-3/4 conditional-generation configs nest the text config, so the
    top-level flag alone is not always honoured."""
    for cfg in (getattr(model, "config", None), getattr(getattr(model, "config", None), "text_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = value


def load_gemma(model_id, dtype, device_map, attn_impl, load_in_4bit):
    cfg = AutoConfig.from_pretrained(model_id)
    multimodal = hasattr(cfg, "vision_config") or hasattr(getattr(cfg, "text_config", object()), "vision_config")
    print(f"Checkpoint type: {'multimodal' if multimodal else 'text-only'} ({cfg.model_type})")

    kwargs = dict(dtype=dtype, device_map=device_map, attn_implementation=attn_impl)
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype, bnb_4bit_use_double_quant=True)

    if multimodal:
        from transformers import AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id)
        tok = processor.tokenizer
        template_owner = processor if getattr(processor, "chat_template", None) else tok
        model = AutoModelForImageTextToText.from_pretrained(model_id, **kwargs)
    else:
        tok = AutoTokenizer.from_pretrained(model_id)
        template_owner = tok
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)

    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Gemma ends a turn with <end_of_turn>; <eos> alone will not stop generation
    end_id = tok.convert_tokens_to_ids("<end_of_turn>")
    if end_id is None or end_id < 0 or end_id == tok.unk_token_id:
        end_id = tok.eos_token_id
        print("WARNING: <end_of_turn> not in vocabulary; falling back to eos_token.")
    eos_ids = sorted({i for i in (end_id, tok.eos_token_id) if i is not None})
    print(f"Turn terminator id: {end_id}  |  generation eos ids: {eos_ids}")

    if load_in_4bit:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    return model, tok, template_owner, multimodal, end_id, eos_ids


def find_lora_targets(model, include_vision=False):
    """Full module paths of the linear layers to adapt. `all-linear` cannot be
    used on the multimodal checkpoints: the SigLIP tower also has q/k/v_proj,
    so suffix matching would silently put LoRA on the vision encoder."""
    linear_types = [nn.Linear]
    try:
        import bitsandbytes as bnb
        linear_types += [bnb.nn.Linear4bit, bnb.nn.Linear8bitLt]
    except ImportError:
        pass
    linear_types = tuple(linear_types)

    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, linear_types):
            continue
        if "lm_head" in name or "embed" in name:
            continue
        if not include_vision and any(k in name for k in VISION_KEYS):
            continue
        targets.append(name)
    if not targets:
        raise RuntimeError("No LoRA target modules found -- check the module naming for this checkpoint.")
    print(f"LoRA targets: {len(targets)} linear layers "
          f"(vision tower {'included' if include_vision else 'excluded'})")
    print(f"  e.g. {targets[0]} ... {targets[-1]}")
    return targets


# --------------------------------------------------------------------------- #
# Hugging Face Hub upload
# --------------------------------------------------------------------------- #
def write_model_card(path, repo_id, model_id, prompt_lang, bleu, corpus, args, n_train, n_test):
    sent = bleu['bleu_overall']['mean'] if bleu else None
    card = f"""---
license: gemma
language:
- si
- ta
base_model: {model_id}
library_name: peft
tags:
- sinhala
- tamil
- translation
- machine-translation
- singen
- lora
metrics:
- bleu
---

# {repo_id.split('/')[-1]}

A LoRA adapter for **Tamil to Sinhala machine translation**, trained on the TamSiPara
Tamil-Sinhala parallel corpus (Pushpananda et al., 2024) as part of the SinGen Sinhala
text generation benchmark.

The adapter is trained on top of [{model_id}](https://huggingface.co/{model_id}) using the
model's own chat template, with the same instruction wording as the SinGen zero-shot and
few-shot prompting runs.

## Usage

```python
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel

processor = AutoProcessor.from_pretrained("{model_id}")
model = AutoModelForImageTextToText.from_pretrained("{model_id}", dtype="auto", device_map="auto")
model = PeftModel.from_pretrained(model, "{repo_id}")
```

(Use `AutoTokenizer` / `AutoModelForCausalLM` for text-only checkpoints.)

The model continues from the `{TARGET_PREFIX}` prefix inside the assistant turn.

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
| Target modules | language-model linear layers (vision tower excluded) |

## Evaluation

Held-out tail of TamSiPara ({n_test} sentence pairs), whitespace-tokenized (sacreBLEU's
default 13a tokenizer splits Sinhala conjuncts and vowel signs):

| Metric | Score |
|---|---|
| Corpus sacreBLEU | {f'{corpus:.2f}' if corpus is not None else 'n/a'} |
| Sentence-level BLEU mean | {f'{sent:.2f}' if sent is not None else 'n/a'} |

## Licence

Derived from Gemma and governed by the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms).
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
                          repo_type="model", commit_message="Add Gemma Ta-Si translation LoRA")
    print(f"Uploaded: https://huggingface.co/{repo_id}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default=DEFAULT_MODEL,
                        help='e.g. google/gemma-4-31B-it, google/gemma-3-27b-it')
    parser.add_argument('--prompt_lang', type=str, default='en', choices=['en', 'si'],
                        help="Instruction language used for BOTH training and evaluation.")
    # data
    parser.add_argument('--tsv_file', type=str, default='ta_si.tsv',
                        help='TamSiPara TSV with columns Tamil / Sinhala.')
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Held-out tail used as the test set (matches the prompting scripts).')
    parser.add_argument('--train_size', type=int, default=0,
                        help='Optional random subsample of the train pool (seed=777). '
                             '0 (default) uses every available pair.')
    parser.add_argument('--drop_train_overlap', action='store_true',
                        help='Remove train pairs whose Tamil side appears in the test tail.')
    # training
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=768)
    parser.add_argument('--attn_impl', type=str, default='eager',
                        choices=['eager', 'sdpa', 'flash_attention_2'],
                        help="Gemma is trained/served with eager attention; keep the default "
                             "unless you have verified parity.")
    parser.add_argument('--load_in_4bit', action='store_true',
                        help='QLoRA. Only needed if the 31B/27B checkpoints will not fit in bf16.')
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    parser.add_argument('--lora_include_vision', action='store_true',
                        help='Also adapt the vision tower (off by default; this is a text task).')
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--do_sample', action='store_true',
                        help='Sampling instead of greedy at eval time.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/<model>-TamSiPara-Ta2Si-<prompt_lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    model_tag = args.model_name.split('/')[-1]
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/{model_tag}-TamSiPara-Ta2Si-{prompt_lang}"
    print(f"Model: {args.model_name}")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (chat template)")
    print("Task: Tamil to Sinhala translation (TamSiPara)")

    OUTPUT_FOLDER = os.path.join("outputs", "tamil_sinhala_translation_finetuned", model_tag, prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ data
    # Loaded before the model so a bad path fails in seconds, not after a
    # 60GB checkpoint download.
    train_df, test_df, overlap = load_splits(args.tsv_file, args.test_size, args.train_size)
    if overlap and args.drop_train_overlap:
        before = len(train_df)
        train_df = train_df[~train_df['Tamil'].astype(str).isin(overlap)].reset_index(drop=True)
        print(f"Dropped {before - len(train_df)} overlapping train pairs")

    # ------------------------------------------------------------------ load
    # device_map="auto" spreads 27B/31B across both H200s (naive model
    # parallelism). Launch with plain `python`, never torchrun.
    model, tok, template_owner, multimodal, end_id, eos_ids = load_gemma(
        args.model_name, torch.bfloat16, "auto", args.attn_impl, args.load_in_4bit)
    set_use_cache(model, False)      # required with gradient checkpointing

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM,
        target_modules=find_lora_targets(model, args.lora_include_vision))
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # Sanity check: dump one rendered prompt so the template is visible in logs.
    sample_prompt = apply_template(template_owner,
                                   build_user_message(str(train_df['Tamil'].iloc[0]), prompt_lang),
                                   multimodal)
    print("\n--- rendered prompt (first train instance) ---")
    print(sample_prompt)
    print("--- end prompt ---\n")

    train_ds = build_train_dataset(tok, template_owner, train_df, prompt_lang,
                                   args.max_seq_len, end_id, multimodal)
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
        label_names=["labels"],
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
    bleu, corpus = evaluate(model, tok, template_owner, test_df, args.model_name, prompt_lang,
                            OUTPUT_FOLDER, args.eval_batch_size, args.max_new_tokens,
                            args.max_seq_len, args.do_sample, len(overlap), eos_ids, multimodal)

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.model_name, prompt_lang,
                         bleu, corpus, args, len(train_ds), len(test_df))
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)