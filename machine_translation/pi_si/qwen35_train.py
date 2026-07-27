import argparse
import os
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

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

set_seed(777)

# A training target longer than this fraction of --max_seq_len leaves too little
# room for its own source passage, so the example is dropped rather than having
# its label silently cut. Reported in the log.
MAX_TARGET_FRACTION = 0.7


# --------------------------------------------------------------------------- #
# Data loading + split (identical task logic to the prompting Pali->Si scripts)
# --------------------------------------------------------------------------- #
def load_and_prepare_pali_sinhala_dataset():
    """Loads Pali-Sinhala from HuggingFace and strips leading verse numbers from
    both columns. Returns the full dataframe."""
    print("Loading Pali-Sinhala dataset from HuggingFace...")
    ds = load_dataset("sinhala-nlp/pali-sinhala")
    full_df = ds['train'].to_pandas()

    print("Cleaning dataset (removing leading numbers)...")
    full_df['pali_text'] = full_df['pali_text'].str.replace(r'^\d+\s+', '', regex=True)
    full_df['sinhala_text'] = full_df['sinhala_text'].str.replace(r'^\d+\s+', '', regex=True)

    print(f"Total dataset size: {len(full_df)}")
    return full_df


def split_dataset(full_df, test_size=1000):
    """Last `test_size` rows are the test set; the rest form the training pool.
    Identical split to the prompting scripts -- the eval set MUST stay
    byte-identical or the fine-tuned row is not comparable to the zero/few-shot
    rows in the same table."""
    test_size = min(test_size, len(full_df))
    test_df = full_df.tail(test_size).copy()
    train_pool = full_df.head(len(full_df) - test_size).copy()
    print(f"Training pool size: {len(train_pool)}")
    print(f"Test set size: {len(test_df)}")
    return train_pool, test_df


def clean_pairs(df):
    df = df[df['pali_text'].notna() & df['sinhala_text'].notna()].copy()
    df = df[df['pali_text'].astype(str).str.strip().astype(bool)
            & df['sinhala_text'].astype(str).str.strip().astype(bool)].copy()
    return df.reset_index(drop=True)


def ws_len(s) -> int:
    return len(str(s).split())


def report_length_mismatch(train_pool, test_df):
    """The known caveat for this task, printed so it lands in every job log.

    The corpus is in canonical order, so tail(1000) is not a random sample -- it
    is the last texts in the collection, which are far longer than the average
    row. For the PROMPTING runs this only means the test set is hard. For
    FINE-TUNING it is worse: the model also acquires a length prior from the
    training data, and a model trained on ~50-token verses will under-generate on
    ~370-token passages, which BLEU punishes twice (n-gram recall and the
    brevity penalty). Interpret the fine-tuned Pali row accordingly.
    """
    tr = train_pool['sinhala_text'].apply(ws_len)
    te = test_df['sinhala_text'].apply(ws_len)
    print("\n" + "=" * 70)
    print("TRAIN/TEST LENGTH DISTRIBUTION (Sinhala target, whitespace tokens)")
    print("=" * 70)
    for name, s in (("train pool", tr), ("test", te)):
        print(f"{name:11s} median={np.median(s):7.0f}  mean={np.mean(s):7.0f}  "
              f"p90={np.percentile(s, 90):7.0f}  max={np.max(s):7.0f}")
    print(f"Median ratio test/train: {np.median(te) / max(np.median(tr), 1):.1f}x")
    print("This mismatch is a property of the canonical corpus ordering, not a bug.")
    print("It depresses fine-tuned BLEU via the brevity penalty -- see the")
    print("length-ratio diagnostic printed after evaluation.")
    print("=" * 70 + "\n")


def length_matched_sample(train_pool, test_df, n, n_bins=5):
    """Optional (--length_match_train): draw the training subsample so its target
    length distribution approximates the test set's, instead of the pool's.

    This is an ablation, not the default. It cannot manufacture long training
    examples that the pool does not contain, so when a bin is underfilled it
    oversamples with replacement and says so -- oversampling a handful of long
    passages risks memorising them. Run it as a second row in the table, and
    keep the plain random subsample as the headline number.
    """
    tr = train_pool['sinhala_text'].apply(ws_len).to_numpy()
    te = test_df['sinhala_text'].apply(ws_len).to_numpy()
    edges = np.unique(np.quantile(te, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    tr_bin = np.digitize(tr, edges[1:-1])
    te_bin = np.digitize(te, edges[1:-1])
    target_prop = np.bincount(te_bin, minlength=len(edges) - 1) / len(te_bin)

    rng = np.random.default_rng(777)
    picks, shortfalls = [], []
    for b, prop in enumerate(target_prop):
        want = int(round(n * prop))
        pool_idx = np.flatnonzero(tr_bin == b)
        if want == 0 or len(pool_idx) == 0:
            if want > 0:
                shortfalls.append((b, want, 0))
            continue
        replace = len(pool_idx) < want
        if replace:
            shortfalls.append((b, want, len(pool_idx)))
        picks.append(rng.choice(pool_idx, size=want, replace=replace))

    idx = np.concatenate(picks)
    for b, want, have in shortfalls:
        lo = edges[b] if np.isfinite(edges[b]) else 0
        hi = edges[b + 1] if np.isfinite(edges[b + 1]) else float('inf')
        print(f"[length-match] bin {b} ({lo:.0f}-{hi:.0f} tokens): wanted {want}, "
              f"pool has {have} -> oversampled with replacement")
    out = train_pool.iloc[idx].reset_index(drop=True)
    print(f"[length-match] training median target length: "
          f"{np.median(out['sinhala_text'].apply(ws_len)):.0f} tokens "
          f"(test median {np.median(te):.0f})")
    return out


# --------------------------------------------------------------------------- #
# Prompt text  (IDENTICAL wording to the prompting Pali->Si scripts so BLEU
# stays comparable). Fine-tuned models are trained AND evaluated zero-shot, so
# the comparable prompting rows are `zero-shot` (en) and `zero-shot-si` (si).
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


def build_user_content(pali: str, prompt_lang: str) -> str:
    if prompt_lang == "si":
        return f"{TASK_DESC_SI} {ACTION_DESC_SI} P: {pali}"
    return f"{TASK_DESC_EN} {ACTION_DESC_EN} P: {pali}"


def build_prompt_ids(tok, pali: str, prompt_lang: str, enable_thinking: bool) -> List[int]:
    """Render the chat template to TEXT, then tokenize separately.

    apply_chat_template(tokenize=True) returns a BatchEncoding dict rather than
    a flat list of ids; indexing it silently yields the wrong thing and the
    labels end up misaligned with no visible error. tokenize=False avoids that
    entirely. add_special_tokens=False because the template already emits them.
    """
    messages = [{"role": "user", "content": build_user_content(pali, prompt_lang)}]
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
# Same rule as the FLORES En->Si training script: keep the target and the
# assistant-turn header intact, and remove tokens from the MIDDLE of the prompt.
# On this corpus it fires much more often than on FLORES, because test passages
# are long -- the counts are printed for both training and evaluation.
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
def build_training_example(tok, pali: str, sinhala: str, prompt_lang: str,
                           max_len: int, end_id: int, enable_thinking: bool):
    # The prompting scripts' extractor cuts the prediction at the first blank
    # line, so a reference containing a paragraph break would train the model to
    # emit text the evaluator then discards. Collapsing whitespace in the target
    # keeps the model's output shape and the extractor in agreement; it does not
    # affect BLEU, which whitespace-tokenizes both sides anyway.
    sinhala = " ".join(str(sinhala).split())
    target_text = f"{TARGET_PREFIX} {sinhala}"
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [end_id]

    if len(target_ids) > int(max_len * MAX_TARGET_FRACTION):
        return None, False        # dropped: no room left for its own source

    prompt_ids = build_prompt_ids(tok, str(pali), prompt_lang, enable_thinking)
    prompt_ids, truncated = truncate_prompt_ids(prompt_ids, max_len - len(target_ids))

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}, truncated


def build_train_dataset(tok, train_df, prompt_lang, max_len, enable_thinking):
    end_id = turn_end_id(tok)
    examples, seq_lens = [], []
    n_trunc, n_dropped = 0, 0
    src_chars, src_toks, tgt_chars, tgt_toks = 0, 0, 0, 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        ex, truncated = build_training_example(
            tok, row['pali_text'], row['sinhala_text'], prompt_lang, max_len, end_id, enable_thinking)
        if ex is None:
            n_dropped += 1
            continue
        n_trunc += int(truncated)
        seq_lens.append(len(ex["input_ids"]))
        examples.append(ex)
        p, s = str(row['pali_text']), str(row['sinhala_text'])
        src_chars += len(p); tgt_chars += len(s)
        src_toks += len(tok(p, add_special_tokens=False)["input_ids"])
        tgt_toks += len(tok(s, add_special_tokens=False)["input_ids"])

    if not examples:
        raise SystemExit("Every training example was dropped as overlong; raise --max_seq_len.")

    print(f"Sequence length: mean={np.mean(seq_lens):.0f} p95={np.percentile(seq_lens, 95):.0f} "
          f"max={np.max(seq_lens)} (cap {max_len})")
    print(f"Dropped as overlong target: {n_dropped}/{len(train_df)} "
          f"({100 * n_dropped / max(len(train_df), 1):.1f}%)")
    print(f"Prompt middle-truncated: {n_trunc}/{len(examples)} -- raise --max_seq_len if high")
    # Fertility for both sides. Pali is Latin-script so it tokenizes far more
    # cheaply than the Sinhala target; worth a line in the paper.
    print(f"[fertility] Pali {src_toks / max(src_chars, 1):.3f} tok/char | "
          f"Sinhala {tgt_toks / max(tgt_chars, 1):.3f} tok/char")
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
# Output post-processing (copied VERBATIM from the prompting Pali->Si script)
# --------------------------------------------------------------------------- #
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_translation(response):
    """Extract translation from model response (thinking stripped first).

    Kept byte-identical to the prompting script, including the `(?:\\n\\n|\\Z)`
    stop, which cuts a multi-paragraph prediction at the first blank line.
    Training targets are whitespace-collapsed so the fine-tuned model does not
    produce blank lines in the first place -- but if you ever change this
    function, change it in every Pali->Si script at once.
    """
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return ""

    text = strip_thinking(response)

    try:
        matches = re.findall(r'Translation:\s*(.*?)(?:\n\n|\Z)', text, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip()

        if "translation:" in text.lower():
            parts = text.lower().split("translation:")
            if len(parts) > 1:
                return parts[1].strip()

        return text.strip()
    except Exception as e:
        print(f"Error extracting translation: {e}")
        return ""


def marker_present(response) -> bool:
    """Diagnostic only -- does NOT alter extraction."""
    return bool(re.search(r'translation:', strip_thinking(response), re.IGNORECASE))


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens, do_sample) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same chat
    rendering and truncation rule as training."""
    outputs = []
    pad_id = tok.pad_token_id
    end_id = turn_end_id(tok)
    terminators = list({tok.eos_token_id, end_id} - {None})
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
# BLEU (copied verbatim from the prompting Pali->Si script)
# --------------------------------------------------------------------------- #
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


def corpus_sacrebleu(refs: List[str], preds: List[str]) -> Optional[float]:
    """Standard corpus-level BLEU alongside the sentence-level mean.
    `tokenize='none'` on whitespace-split text keeps it consistent with the
    Sinhala-safe tokenization used everywhere else in SinGen; sacreBLEU's
    default 13a tokenizer mangles Sinhala the same way \\b\\w+\\b does."""
    try:
        import sacrebleu
    except ImportError:
        print("sacrebleu not installed -- skipping corpus BLEU.")
        return None
    pairs = [(str(r), str(p)) for r, p in zip(refs, preds) if str(r).strip() and str(p).strip()]
    if not pairs:
        return None
    r, p = zip(*pairs)
    return sacrebleu.corpus_bleu(list(p), [list(r)], tokenize="none").score


def length_diagnostic(refs: List[str], preds: List[str]) -> Dict[str, float]:
    """The number that explains the Pali row. BLEU on this test set is heavily
    brevity-penalised, so report the hypothesis/reference length ratio next to
    it -- a ratio well under 1.0 means the score is measuring under-generation
    (a length prior learned from short training verses, or --max_new_tokens
    cutting the output off) rather than translation quality."""
    r_tok = sum(len(tokenize(r)) for r in refs)
    p_tok = sum(len(tokenize(p)) for p in preds)
    ratio = p_tok / max(r_tok, 1)
    per_inst = [len(tokenize(p)) / max(len(tokenize(r)), 1) for r, p in zip(refs, preds)]
    print("\n" + "=" * 70)
    print(f"LENGTH DIAGNOSTIC: corpus hyp/ref token ratio = {ratio:.3f}")
    print(f"  median per-instance ratio = {np.median(per_inst):.3f}")
    print(f"  instances under 0.5x reference length: "
          f"{sum(1 for x in per_inst if x < 0.5)}/{len(per_inst)}")
    if ratio < 0.8:
        print("  -> Predictions are substantially shorter than references. BLEU here is")
        print("     dominated by the brevity penalty; check --max_new_tokens first, then")
        print("     the train/test length mismatch reported before training.")
    print("=" * 70)
    return {'corpus_ratio': ratio, 'median_ratio': float(np.median(per_inst))}


# --------------------------------------------------------------------------- #
# Evaluation on the held-out tail
# --------------------------------------------------------------------------- #
def evaluate(model, tok, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, max_seq_len, do_sample, enable_thinking):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    budget = max_seq_len - max_new_tokens
    prompt_ids, n_trunc = [], 0
    for pali in df['pali_text']:
        ids = build_prompt_ids(tok, str(pali), prompt_lang, enable_thinking)
        ids, truncated = truncate_prompt_ids(ids, budget)
        n_trunc += int(truncated)
        prompt_ids.append(ids)
    if n_trunc:
        print(f"Prompt middle-truncated eval instances: {n_trunc}/{len(df)} "
              f"<-- long test passages; raise --max_seq_len if this is large")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses
    df['preds'] = [extract_translation(r) for r in responses]
    df['marker_matched'] = [marker_present(r) for r in responses]

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Translation:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for residual <think> if high")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    bleu_results = evaluate_bleu_scores(df)
    corpus = corpus_sacrebleu(df['sinhala_text'].tolist(), df['preds'].tolist())
    if corpus is not None:
        print(f"\nCorpus sacreBLEU (tokenize=none): {corpus:.4f}")
    lengths = length_diagnostic(df['sinhala_text'].tolist(), df['preds'].tolist())

    df.to_csv(os.path.join(output_folder, "predictions_with_bleu.csv"), index=False, encoding='utf-8')

    with open(os.path.join(output_folder, "bleu_summary.txt"), 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, chat template)\n")
        f.write(f"Query type: zero-shot ({prompt_lang})\n")
        f.write("Dataset: sinhala-nlp/pali-sinhala (last 1000 rows as test)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\nBatch Size: {batch_size}\n")
        f.write(f"Max seq len: {max_seq_len}\nThinking enabled: {enable_thinking}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}\n")
        if corpus is not None:
            f.write(f"Corpus sacreBLEU (tokenize=none): {corpus:.4f}\n")
        f.write(f"Hyp/ref length ratio: corpus {lengths['corpus_ratio']:.3f}, "
                f"median per-instance {lengths['median_ratio']:.3f}\n")
        f.write("CAVEAT: the tail(1000) test split of this canonically ordered corpus is far\n")
        f.write("longer than the training pool, so BLEU is partly driven by the brevity\n")
        f.write("penalty. Read the length ratio above alongside the scores.\n")
        f.write("Primary metric below is the sentence-level BLEU mean used by the "
                "prompting scripts (not corpus BLEU).\n")
        f.write("=" * 70 + "\n\n")
        for k in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
            f.write(f"{k.upper().replace('_', '-')}:\n")
            for stat in ['mean', 'median', 'std', 'min', 'max']:
                f.write(f"  {stat.capitalize():7s} {bleu_results[k][stat]:.4f}\n")
            f.write("\n")

    return bleu_results, corpus, lengths


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
def write_model_card(path, repo_id, model_id, prompt_lang, bleu, corpus, lengths, args, n_train):
    sent = bleu['bleu_overall']['mean'] if bleu else None
    card = f"""---
language:
- si
- pi
base_model: {model_id}
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

A LoRA adapter for **Pali to Sinhala translation**, fine-tuned from
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
| Length-matched subsample | {args.length_match_train} |
| Epochs | {args.num_train_epochs} |
| Effective batch size | {args.train_batch_size * args.grad_accum} |
| Learning rate | {args.learning_rate} |
| Max sequence length | {args.max_seq_len} |
| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |
| Thinking during training | {args.enable_thinking} |

## Evaluation

Last 1000 rows of `sinhala-nlp/pali-sinhala`, whitespace-tokenized (sacreBLEU's default
13a tokenizer splits Sinhala conjuncts and vowel signs):

| Metric | Score |
|---|---|
| Corpus sacreBLEU | {f'{corpus:.2f}' if corpus is not None else 'n/a'} |
| Sentence-level BLEU mean | {f'{sent:.2f}' if sent is not None else 'n/a'} |
| Hyp/ref length ratio | {f"{lengths['corpus_ratio']:.3f}" if lengths else 'n/a'} |

**Read the scores with the length ratio.** The corpus is in canonical order, so the
trailing 1000 rows used as the test set are much longer than the training pool
(median target length differs by roughly an order of magnitude). Predictions are
therefore shorter than references and BLEU is partly driven by the brevity penalty
rather than translation quality. This split is kept as-is so the numbers stay
comparable across the model families evaluated in SinGen.

## Licence

This adapter inherits the licence of the base model; check the base model card
before redistributing. Verify the terms of the `sinhala-nlp/pali-sinhala` dataset
on its dataset card as well.
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
                          repo_type="model", commit_message="Add Qwen Pali-Sinhala translation LoRA")
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
    parser.add_argument('--test_size', type=int, default=1000,
                        help='Trailing rows used as the test set -- must match the prompting scripts.')
    parser.add_argument('--train_size', type=int, default=0,
                        help='Random subsample of the training pool (seed=777). 0 uses all of it.')
    parser.add_argument('--length_match_train', action='store_true',
                        help='ABLATION: draw the training subsample to match the test length '
                             'distribution instead of the pool. Requires --train_size.')
    # training
    parser.add_argument('--num_train_epochs', type=float, default=1.0)
    parser.add_argument('--train_batch_size', type=int, default=1)
    parser.add_argument('--grad_accum', type=int, default=16)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=3072,
                        help='Must hold a long Pali passage plus its (longer) Sinhala '
                             'translation. Watch the drop/truncation counts in the log.')
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
    parser.add_argument('--max_new_tokens', type=int, default=512,
                        help='Matches the prompting scripts. Long test references mean a low '
                             'value here shows up directly in the length ratio.')
    parser.add_argument('--do_sample', action='store_true',
                        help='Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy.')
    parser.add_argument('--no_save_adapter', action='store_true',
                        help='Skip writing the LoRA adapter to disk.')
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/<model>-PaliSinhala-Pali2Si-<lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    # Fail before loading a 27B model rather than after twelve runs have finished.
    if args.push_to_hub and not os.environ.get("HF_TOKEN"):
        raise SystemExit("--push_to_hub given but HF_TOKEN is empty; export a WRITE-scoped token.")
    if args.length_match_train and not args.train_size:
        raise SystemExit("--length_match_train needs a --train_size to sample toward.")

    model_tag = args.model_id.split('/')[-1]
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/{model_tag}-PaliSinhala-Pali2Si-{prompt_lang}"
    print(f"Model: {args.model_id}")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (chat template)")
    print(f"Thinking: {'enabled' if args.enable_thinking else 'disabled'}")
    print("Task: Pali to Sinhala translation (sinhala-nlp/pali-sinhala)")

    suffix = f"{prompt_lang}-lenmatch" if args.length_match_train else prompt_lang
    OUTPUT_FOLDER = os.path.join("outputs", "pali_sinhala_translation_finetuned", model_tag, suffix)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ data
    # Loaded before the model so a dataset failure surfaces immediately rather
    # than after a multi-GB checkpoint download.
    full_df = load_and_prepare_pali_sinhala_dataset()
    train_pool, test_df = split_dataset(full_df, test_size=args.test_size)
    train_pool, test_df = clean_pairs(train_pool), clean_pairs(test_df)
    if len(test_df) == 0 or len(train_pool) == 0:
        print("Error: Could not load or split dataset properly")
        raise SystemExit(1)
    report_length_mismatch(train_pool, test_df)

    if args.length_match_train:
        train_df = length_matched_sample(train_pool, test_df, args.train_size)
    elif args.train_size and args.train_size < len(train_pool):
        train_df = train_pool.sample(n=args.train_size, random_state=777).reset_index(drop=True)
        print(f"Subsampled train set to {len(train_df)} pairs (seed=777)")
    else:
        train_df = train_pool
        print(f"Using the full training pool ({len(train_df)} pairs)")

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

    train_ds = build_train_dataset(tok, train_df, prompt_lang, args.max_seq_len, args.enable_thinking)
    print(f"Train examples: {len(train_ds)}  Test (tail): {len(test_df)}")

    steps = int(len(train_ds) * args.num_train_epochs /
                (args.train_batch_size * args.grad_accum))
    print(f"Approximate optimizer steps: {steps}")
    if steps < 100:
        print("WARNING: fewer than 100 optimizer steps. Raise --train_size or --num_train_epochs.")

    collator = CausalCollator(pad_token_id=tok.pad_token_id)

    # NOTE: `group_by_length` / `length_column_name` were removed -- they are not
    # accepted by TrainingArguments in transformers v5, which raises on unknown
    # kwargs instead of ignoring them. Dropping them also keeps batch composition
    # identical to the Gemma/Llama scripts. Padding waste on this corpus is real
    # (passage lengths span two orders of magnitude), but it only costs anything
    # when a batch holds more than one sequence, and every model above 4B here
    # runs at --train_batch_size 1.
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
    bleu, corpus, lengths = evaluate(
        model, tok, test_df, args.model_id, prompt_lang, OUTPUT_FOLDER,
        args.eval_batch_size, args.max_new_tokens, args.max_seq_len,
        args.do_sample, args.enable_thinking)

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.model_id, prompt_lang,
                         bleu, corpus, lengths, args, len(train_ds))
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)