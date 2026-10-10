"""
LoRA instruction fine-tuning + evaluation for Gemma models on SinGen
biography generation: sinhala-nlp/BioGen (source infobox -> target sentence),
scored with whitespace-tokenised ROUGE (sinhala_rouge.py).

One run = one (model, fine-tune language). The model is trained on the
official train split with the zero-shot instruction (English or Sinhala) and
then evaluated on the official test split in the same process.

Works for every Gemma checkpoint used in the paper (routing identical to the
Gemma prompting scripts):
    Gemma-4                      -> AutoProcessor + AutoModelForCausalLM
    Gemma-3 multimodal (vision)  -> AutoProcessor + AutoModelForImageTextToText
    Gemma-3 text-only            -> AutoTokenizer + AutoModelForCausalLM

Gemma-specific handling:
    - Gemma-4: LoRA targets are left to peft (>=0.19 ships Gemma-4-aware
      defaults; an explicit list hits Gemma4ClippableLinear) and
      ensure_weight_tying=True.
    - Gemma-3: explicit language-model linear layers, vision tower and
      projector excluded; token_type_ids (all zero, text only) are passed in
      training because the multimodal model requires them.
    - Eager attention for training (Gemma's recommendation), use_cache patched
      on nested text configs, thinking/channel blocks stripped from outputs.

Hyperparameters follow the paper's LoRA table and are identical across families:
    r=16, alpha=32, dropout=0.05, lr=2e-4 cosine, warmup 3%, wd 0, AdamW,
    3 epochs, effective batch 16 (4 x 4), max seq len 1024, bf16, seed 42.

Outputs: outputs/biography_generation_finetuned/<model>/<lang>/
    predictions.csv, predictions_with_scores.csv, rouge_summary.txt, lora_adapter/

Run with plain `python -m` (not torchrun): device_map="auto" spreads the larger
checkpoints over every visible GPU.
"""
import argparse
import math
import os
import re

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm.auto import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM, AutoProcessor, AutoTokenizer,
                          Trainer, TrainingArguments, set_seed)

from transformers import AutoModelForImageTextToText

# Set by load_model(); read by the prompt renderer, collator and LoRA setup.
IS_GEMMA4 = False
IS_PROCESSOR = False
NEEDS_TOKEN_TYPE_IDS = False


# --------------------------------------------------------------------------- #
# Task definitions. Prompt wording is copied verbatim from the prompting
# scripts (zero-shot / zero-shot-si) -- do not reword without updating every
# model family's scripts.
# --------------------------------------------------------------------------- #
TASKS = {
    "biogen": {
        "dir": "biography_generation_finetuned",
        "dataset": "sinhala-nlp/BioGen",
        "src_col": "source",
        "tgt_col": "target",
        "label": "I",
        "prefix": "Biography:",
        "summary": "rouge_summary.txt",
        "task_desc": "You are an expert in the Sinhala language. Using the following Wikipedia infobox (I) "
                     "about a person, write the opening sentence of their Sinhala Wikipedia biography.",
        "action_desc": "Return only the Sinhala sentence following the prefix 'Biography:' without any "
                       "other text or explanations.",
        "task_desc_si": "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පුද්ගලයෙකු පිළිබඳ පහත විකිපීඩියා "
                        "තොරතුරු කොටුව (I) භාවිත කරමින්, එම පුද්ගලයාගේ සිංහල විකිපීඩියා චරිතාපදානයේ "
                        "ආරම්භක වාක්‍යය ලියන්න.",
        "action_desc_si": "'Biography:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු "
                          "උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න.",
    },
}


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_task_data(task):
    t = TASKS[task]
    ds = load_dataset(t["dataset"])
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    src, tgt = t["src_col"], t["tgt_col"]
    train_df = train_df.dropna(subset=[src, tgt])
    train_df = train_df[(train_df[src].astype(str).str.strip() != "") &
                        (train_df[tgt].astype(str).str.strip() != "")].reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"Train: {len(train_df)}  Test: {len(test_df)}")
    return train_df, test_df


def build_content(task, lang, source):
    t = TASKS[task]
    if lang == "si":
        return f"{t['task_desc_si']} {t['action_desc_si']} {t['label']}: {source}"
    return f"{t['task_desc']} {t['action_desc']} {t['label']}: {source}"


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def _from_pretrained(cls, model_id):
    """`torch_dtype` was renamed to `dtype` in transformers 5.x; try the new name first."""
    kwargs = dict(device_map="auto", attn_implementation="eager")
    try:
        return cls.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return cls.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)


def load_model(model_id):
    global IS_GEMMA4, IS_PROCESSOR, NEEDS_TOKEN_TYPE_IDS
    cfg = AutoConfig.from_pretrained(model_id)
    model_type = (getattr(cfg, "model_type", "") or "").lower()
    IS_GEMMA4 = "gemma4" in model_type or re.search(r"gemma[-_]?4", model_id.lower()) is not None
    is_multimodal = getattr(cfg, "vision_config", None) is not None

    if IS_GEMMA4:
        print("[loader] Gemma-4 -> AutoProcessor + AutoModelForCausalLM")
        chat_proc = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForCausalLM, model_id)
        IS_PROCESSOR = True
    elif is_multimodal:
        print("[loader] Multimodal Gemma-3 -> AutoProcessor + AutoModelForImageTextToText")
        chat_proc = AutoProcessor.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForImageTextToText, model_id)
        IS_PROCESSOR = True
        NEEDS_TOKEN_TYPE_IDS = True
    else:
        print("[loader] Text-only Gemma-3 -> AutoTokenizer + AutoModelForCausalLM")
        chat_proc = AutoTokenizer.from_pretrained(model_id)
        model = _from_pretrained(AutoModelForCausalLM, model_id)
        IS_PROCESSOR = False

    tok = chat_proc.tokenizer if hasattr(chat_proc, "tokenizer") else chat_proc
    if getattr(chat_proc, "chat_template", None) is None and getattr(tok, "chat_template", None) is None:
        raise ValueError(f"{model_id} has no chat template -- use an instruction-tuned (-it) checkpoint.")
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    if hasattr(model, "hf_device_map"):
        dist = {}
        for dev in model.hf_device_map.values():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)
    return model, chat_proc, tok


def set_use_cache(model, value):
    """Gemma-3/4 multimodal configs keep use_cache on nested text configs too."""
    for cfg in (getattr(model, "config", None),
                getattr(getattr(model, "config", None), "text_config", None),
                getattr(model, "generation_config", None)):
        if cfg is not None and hasattr(cfg, "use_cache"):
            cfg.use_cache = value


_TURN_END_CACHE = {}


def turn_end_id(tok, chat_proc=None):
    """The token that closes an assistant turn, read off the chat template
    itself (<end_of_turn> on Gemma-3; whatever Gemma-4's template uses).
    Falls back to EOS."""
    key = id(tok)
    if key in _TURN_END_CACHE:
        return _TURN_END_CACHE[key]
    end_id = tok.eos_token_id
    proc = chat_proc if chat_proc is not None else tok
    try:
        msgs = to_messages([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "@@@"}])
        rendered = proc.apply_chat_template(msgs, tokenize=False)
        tail_ids = tok(rendered.split("@@@", 1)[1], add_special_tokens=False)["input_ids"]
        special = set(tok.all_special_ids)
        for tid in tail_ids:
            if tid in special or tok.convert_ids_to_tokens(tid).startswith("<"):
                end_id = tid
                break
    except Exception as e:
        print(f"[turn_end_id] falling back to EOS ({e})")
    for name in ("<end_of_turn>",):
        if end_id == tok.eos_token_id:
            tid = tok.convert_tokens_to_ids(name)
            if tid is not None and tid != tok.unk_token_id:
                end_id = tid
    _TURN_END_CACHE[key] = end_id
    print(f"Turn-end token: {tok.convert_ids_to_tokens(end_id)!r} (id {end_id})")
    return end_id


def to_messages(messages):
    """Processor chat templates (multimodal Gemma-3 / Gemma-4) expect content as
    a list of typed parts; plain tokenizers take strings."""
    if not IS_PROCESSOR:
        return messages
    return [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in messages]


# --------------------------------------------------------------------------- #
# Prompt rendering. apply_chat_template(tokenize=False) then tokenize
# separately: with tokenize=True a processor returns a BatchEncoding instead of
# a flat id list, which silently breaks the collator.
# --------------------------------------------------------------------------- #
def render_chat(chat_proc, content):
    messages = to_messages([{"role": "user", "content": content}])
    if IS_GEMMA4:
        try:
            return chat_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=False)
        except TypeError:
            pass
    return chat_proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def prompt_ids_for(chat_proc, tok, task, lang, source, budget):
    """Token ids of the rendered prompt, truncating the source text (never the
    instruction) so the prompt fits in `budget` tokens. Returns (ids, truncated)."""
    ids = tok(render_chat(chat_proc, build_content(task, lang, source)), add_special_tokens=False)["input_ids"]
    if len(ids) <= budget:
        return ids, False

    scaffold = len(tok(render_chat(chat_proc, build_content(task, lang, "")),
                       add_special_tokens=False)["input_ids"])
    src_ids = tok(source, add_special_tokens=False)["input_ids"]
    keep = max(budget - scaffold - 4, 0)
    cut = tok.decode(src_ids[:keep], skip_special_tokens=True).replace("�", "").strip()
    ids = tok(render_chat(chat_proc, build_content(task, lang, cut)), add_special_tokens=False)["input_ids"]
    return ids[:budget], True


# --------------------------------------------------------------------------- #
# Training data
# --------------------------------------------------------------------------- #
class ListDataset(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class CausalCollator:
    """Right-pads input_ids / attention_mask / labels for training."""
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        width = max(len(ex["input_ids"]) for ex in batch)
        input_ids, attn, labels = [], [], []
        for ex in batch:
            pad = width - len(ex["input_ids"])
            input_ids.append(ex["input_ids"] + [self.pad_token_id] * pad)
            attn.append([1] * len(ex["input_ids"]) + [0] * pad)
            labels.append(ex["labels"] + [-100] * pad)
        out = {"input_ids": torch.tensor(input_ids, dtype=torch.long),
               "attention_mask": torch.tensor(attn, dtype=torch.long),
               "labels": torch.tensor(labels, dtype=torch.long)}
        if NEEDS_TOKEN_TYPE_IDS:                      # text only -> all zeros
            out["token_type_ids"] = torch.zeros_like(out["input_ids"])
        return out


def build_train_dataset(chat_proc, tok, task, lang, train_df, max_seq_len):
    t = TASKS[task]
    end_id = turn_end_id(tok, chat_proc)
    examples, seq_lens, n_trunc, n_tgt_cut = [], [], 0, 0

    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        target = " ".join(str(row[t["tgt_col"]]).split())
        target_ids = tok(f"{t['prefix']} {target}", add_special_tokens=False)["input_ids"] + [end_id]
        if len(target_ids) > max_seq_len // 2:          # pathological targets only
            target_ids = target_ids[:max_seq_len // 2 - 1] + [end_id]
            n_tgt_cut += 1

        p_ids, truncated = prompt_ids_for(chat_proc, tok, task, lang, str(row[t["src_col"]]),
                                          max_seq_len - len(target_ids))
        n_trunc += int(truncated)
        input_ids = p_ids + target_ids
        examples.append({"input_ids": input_ids, "labels": [-100] * len(p_ids) + target_ids})
        seq_lens.append(len(input_ids))

    print(f"Sequence length: mean={np.mean(seq_lens):.0f} p95={np.percentile(seq_lens, 95):.0f} "
          f"max={np.max(seq_lens)} (cap {max_seq_len})")
    print(f"Source-truncated training prompts: {n_trunc}/{len(examples)}")
    print(f"Target-truncated training examples: {n_tgt_cut}/{len(examples)}  <-- should be 0")
    return ListDataset(examples)


# --------------------------------------------------------------------------- #
# LoRA target selection (Gemma-3). All language-model linear layers
# (attention and MLP), excluding the vision tower, multimodal projector and
# lm_head. Gemma-4 does not use this: peft's own defaults are used instead.
# --------------------------------------------------------------------------- #
def lora_targets(model, include_experts):
    names = set()
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        low = name.lower()
        if any(k in low for k in ("visual", "vision", "multi_modal", "lm_head", "embed")):
            continue
        if low.endswith(".gate") or low.endswith("shared_expert_gate") or "router" in low:
            continue
        if ".experts." in low and not include_experts:
            continue
        names.add(name)
    if not names:
        raise ValueError("No LoRA target modules found -- check the model architecture.")
    return sorted(names)


# --------------------------------------------------------------------------- #
# Generation / extraction
# --------------------------------------------------------------------------- #
_THINK_PATTERNS = [
    re.compile(r"<\|channel\|>\s*thought.*?(?=<\|channel\|>|<\|message\|>|\Z)", re.DOTALL | re.IGNORECASE),
    re.compile(r"<\|channel\|>.*?<\|message\|>", re.DOTALL),
    re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE),
    re.compile(r"</?think>", re.IGNORECASE),
]


def extract(response, prefix):
    """First line after the prefix; falls back to the first non-empty line."""
    if not isinstance(response, str):
        return "", False
    text = response
    for pat in _THINK_PATTERNS:
        text = pat.sub("", text)
    text = text.strip()
    m = re.search(re.escape(prefix) + r"\s*(.*?)(?:\n|\Z)", text, re.IGNORECASE | re.DOTALL)
    if m and m.group(1).strip():
        return m.group(1).strip(), True
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[0] if lines else ""), False


def generate(model, tok, prompt_id_lists, batch_size, max_new_tokens):
    pad_id = tok.pad_token_id
    terminators = list({tok.eos_token_id, turn_end_id(tok)} - {None})   # cached from training
    outputs = []
    for start in tqdm(range(0, len(prompt_id_lists), batch_size), desc="Generating"):
        batch = prompt_id_lists[start:start + batch_size]
        width = max(len(ids) for ids in batch)
        input_ids = [[pad_id] * (width - len(ids)) + ids for ids in batch]       # left padding
        attn = [[0] * (width - len(ids)) + [1] * len(ids) for ids in batch]
        enc = {"input_ids": torch.tensor(input_ids, dtype=torch.long).to(model.device),
               "attention_mask": torch.tensor(attn, dtype=torch.long).to(model.device)}
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                                 eos_token_id=terminators, pad_token_id=pad_id)
        outputs.extend(tok.batch_decode(gen[:, width:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def write_stats(f, name, stats, keys):
    f.write(f"{name}:\n")
    for k in keys:
        f.write(f"  {k.capitalize() + ':':8}{stats[k]:.4f}\n")
    f.write("\n")


def evaluate(model, chat_proc, tok, args, test_df, output_folder, train_size):
    t = TASKS[args.task]
    model.eval()
    set_use_cache(model, True)

    budget = args.max_seq_len - args.max_new_tokens
    prompt_ids, n_trunc = [], 0
    for src in test_df[t["src_col"]].astype(str):
        ids, truncated = prompt_ids_for(chat_proc, tok, args.task, args.prompt_lang, src, budget)
        prompt_ids.append(ids)
        n_trunc += int(truncated)
    print(f"Source-truncated test prompts: {n_trunc}/{len(test_df)}")

    df = test_df.copy()
    df["responses"] = generate(model, tok, prompt_ids, args.eval_batch_size, args.max_new_tokens)
    preds, matched = zip(*[extract(r, t["prefix"]) for r in df["responses"]])
    df["preds"] = list(preds)
    df["marker_matched"] = list(matched)
    n_empty = int((df["preds"].str.len() == 0).sum())
    n_miss = int((~df["marker_matched"]).sum())
    print(f"Empty predictions: {n_empty}/{len(df)}   Missing '{t['prefix']}' prefix: {n_miss}/{len(df)}")
    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding="utf-8")

    refs = df[t["tgt_col"]].astype(str).tolist()
    hyps = df["preds"].astype(str).tolist()
    try:
        from rouge_metric import score_corpus, ROUGE_TYPES
    except ImportError:
        from rouge_metric import score_corpus, ROUGE_TYPES
    results, metric_names, keys = score_corpus(refs, hyps), ROUGE_TYPES, ["mean", "median", "std", "min", "max"]
    header = "ROUGE Evaluation Results (whitespace-tokenized, F1 x100)"

    for m in metric_names:
        df[m] = results[m]["scores"]
    df.to_csv(os.path.join(output_folder, "predictions_with_scores.csv"), index=False, encoding="utf-8")

    with open(os.path.join(output_folder, t["summary"]), "w", encoding="utf-8") as f:
        f.write(header + "\n")
        f.write(f"Model: {args.model_id} + LoRA\n")
        f.write(f"Fine-tune language: {args.prompt_lang}\n")
        f.write(f"Dataset: {t['dataset']} (official splits; train {train_size}, test {len(df)})\n")
        f.write(f"LoRA: r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout} "
                f"targets={'Gemma-4 (peft defaults if available)' if IS_GEMMA4 else 'language-model linear layers'}\n")
        f.write(f"Training: epochs={args.num_train_epochs} lr={args.learning_rate} "
                f"eff_batch={args.train_batch_size * args.grad_accum} max_seq_len={args.max_seq_len}\n")
        f.write(f"Max New Tokens: {args.max_new_tokens}\nDecoding: greedy\n")
        f.write(f"Empty predictions: {n_empty}/{len(df)}\n")
        f.write(f"Responses without '{t['prefix']}' prefix: {n_miss}/{len(df)}\n")
        f.write(f"Source-truncated test prompts: {n_trunc}/{len(df)}\n")
        f.write("=" * 70 + "\n\n")
        for m in metric_names:
            write_stats(f, m.upper(), results[m], keys)

    print("\n" + "=" * 70)
    for m in metric_names:
        key = "corpus" if "corpus" in results[m] else "mean"
        print(f"{m.upper():8s} {key}={results[m][key]:.4f}")
    print("=" * 70)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="biogen", choices=["biogen"])
    parser.add_argument("--model_id", type=str, default="google/gemma-3-12b-it")
    parser.add_argument("--prompt_lang", choices=["en", "si"], default="en")
    parser.add_argument("--num_train_epochs", type=float, default=3)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=1024)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--no_save_adapter", action="store_true")
    args = parser.parse_args()

    set_seed(42)
    output_folder = os.path.join("outputs", TASKS[args.task]["dir"], args.model_id.split("/")[-1], args.prompt_lang)
    os.makedirs(output_folder, exist_ok=True)
    print(f"Model: {args.model_id}  Fine-tune language: {args.prompt_lang}")
    print(f"Output: {output_folder}")
    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    train_df, test_df = load_task_data(args.task)
    model, chat_proc, tok = load_model(args.model_id)
    train_ds = build_train_dataset(chat_proc, tok, args.task, args.prompt_lang, train_df, args.max_seq_len)

    lora_kwargs = dict(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
                       bias="none", task_type="CAUSAL_LM")
    set_use_cache(model, False)
    model.enable_input_require_grads()
    if IS_GEMMA4:
        # peft>=0.19 picks Gemma-4's language-model targets itself (a suffix list
        # like "q_proj" would also hit the vision tower's Gemma4ClippableLinear).
        # If this peft has no default for the loaded model type, fall back to the
        # full-path language-model linear layers, which never touch the vision tower.
        try:
            model = get_peft_model(model, LoraConfig(ensure_weight_tying=True, **lora_kwargs))
            print("LoRA target modules: peft defaults (Gemma-4)")
        except ValueError as e:
            targets = lora_targets(model, include_experts=False)
            print(f"peft has no Gemma-4 default here ({e}); using {len(targets)} language-model linear layers")
            model = get_peft_model(model, LoraConfig(ensure_weight_tying=True, target_modules=targets, **lora_kwargs))
    else:
        targets = lora_targets(model, include_experts=False)
        print(f"LoRA target modules: {len(targets)} linear layers")
        model = get_peft_model(model, LoraConfig(target_modules=targets, **lora_kwargs))
    model.print_trainable_parameters()

    eff_batch = args.train_batch_size * args.grad_accum
    total_steps = math.ceil(len(train_ds) / eff_batch) * args.num_train_epochs
    warmup_steps = max(1, math.ceil(args.warmup_ratio * total_steps))
    print(f"Optimizer steps: {total_steps:.0f}  warmup: {warmup_steps}")
    if total_steps < 100:
        print("WARNING: fewer than 100 optimizer steps.")

    training_args = TrainingArguments(
        output_dir=os.path.join(output_folder, "checkpoints"),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        weight_decay=0.0,
        optim="adamw_torch",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        remove_unused_columns=False,
        seed=42,
    )
    trainer = Trainer(model=model, args=training_args, train_dataset=train_ds,
                      data_collator=CausalCollator(tok.pad_token_id))
    trainer.train()

    if not args.no_save_adapter:
        model.save_pretrained(os.path.join(output_folder, "lora_adapter"))
        print(f"Saved LoRA adapter to {os.path.join(output_folder, 'lora_adapter')}")

    evaluate(model, chat_proc, tok, args, test_df, output_folder, len(train_ds))


if __name__ == "__main__":
    main()