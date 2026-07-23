import argparse
import os
import re
import random
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

# Sinhala-safe ROUGE (whitespace-tokenized). The stock rouge_score tokenizer
# strips non-ASCII and would zero out every Sinhala score.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

MAX_CONTENT_LENGTH = 2500  # characters, matches the prompting headline scripts


# --------------------------------------------------------------------------- #
# SinLlama specifics
# --------------------------------------------------------------------------- #
# SinLlama_v01 is a *base* (not instruct-tuned) continual-pretraining LoRA on
# Meta-Llama-3-8B with a Sinhala-extended tokenizer (vocab 128256 -> 139336).
# It has NO chat template, so we can't reuse the apply_chat_template path used
# by the Gemma/Qwen/Llama prompting scripts. The card documents Alpaca-style
# prompts, so we fine-tune in that format.
BASE_MODEL = "meta-llama/Meta-Llama-3-8B"        # gated -> needs HF_TOKEN
SINLLAMA_ADAPTER = "polyglots/SinLlama_v01"
SINLLAMA_TOKENIZER = "polyglots/Extended-Sinhala-LLaMA"


# --------------------------------------------------------------------------- #
# Prompt text  (identical instruction wording to the prompting eval scripts so
# scores stay comparable; only the *wrapper* changes chat-template -> Alpaca)
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

ALPACA_HEADER = ("Below is an instruction that describes a task, paired with an input that provides further "
                 "context. Write a response that appropriately completes the request.")


def build_prompt(news_content: str, prompt_lang: str) -> str:
    """Alpaca-style prompt ending at '### Response:\n' (generation starts here)."""
    if prompt_lang == "si":
        instr = f"{TASK_DESC_SI} {ACTION_DESC_SI}"
    else:
        instr = f"{TASK_DESC_EN} {ACTION_DESC_EN}"
    return (f"{ALPACA_HEADER}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\nNews Content: {news_content}\n\n"
            f"### Response:\n")


# --------------------------------------------------------------------------- #
# Length control
# --------------------------------------------------------------------------- #
# Headline inputs are whole news articles, unlike the single sentences used for
# simplification. A naive tail-truncation at max_len would cut off the target
# (and the '### Response:\n' marker), leaving an example whose labels are all
# -100 -- i.e. silent, total loss of learning signal. Instead we always keep
# the target intact and remove tokens from the MIDDLE of the prompt, which
# preserves both the instruction header and the response marker.
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
def build_training_example(tok, prompt_text: str, headline: str, max_len: int):
    # add_special_tokens=True puts <|begin_of_text|> on the prompt only
    prompt_ids = tok(prompt_text, add_special_tokens=True)["input_ids"]
    target_text = f"{TARGET_PREFIX} {headline}"
    # append EOS so the base model learns to STOP (base LMs otherwise ramble)
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    prompt_ids, truncated = truncate_prompt_ids(prompt_ids, max_len - len(target_ids))

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels,
            "attention_mask": [1] * len(input_ids)}, truncated


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
# Dataset build (matches the prompting scripts' handling exactly)
# --------------------------------------------------------------------------- #
def analyze_and_trim(df, max_length=MAX_CONTENT_LENGTH):
    lengths = df['News Content'].apply(lambda x: len(str(x)))
    print(f"Article length (chars): min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.0f} median={lengths.median():.0f} "
          f">{max_length}: {(lengths > max_length).sum()}")
    df['News Content'] = df['News Content'].apply(
        lambda x: str(x)[:max_length] + "..." if len(str(x)) > max_length else str(x))
    return df


def load_splits(test_size: int):
    ds = load_dataset("sinhala-nlp/NSINA-Headlines")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    train_df = train_df[train_df['News Content'].notna() & train_df['Headline'].notna()].copy()
    test_df = test_df[test_df['News Content'].notna() & test_df['Headline'].notna()].copy()
    print(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")

    test_df = test_df.head(min(test_size, len(test_df))).copy()
    print(f"Using {len(test_df)} test samples")

    print("\n[train] ", end="")
    train_df = analyze_and_trim(train_df)
    print("[test]  ", end="")
    test_df = analyze_and_trim(test_df)
    return train_df, test_df


def build_train_dataset(tok, train_df, prompt_lang, max_len):
    examples, n_trunc = [], 0
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        headline = str(row['Headline']).strip()
        if not headline:
            continue
        prompt_text = build_prompt(str(row['News Content']), prompt_lang)
        ex, truncated = build_training_example(tok, prompt_text, headline, max_len)
        n_trunc += int(truncated)
        examples.append(ex)
    print(f"Prompt-truncated examples (article middle removed): {n_trunc}/{len(examples)} "
          f"-- raise --max_seq_len if this is large")
    return Dataset.from_list(examples)


# --------------------------------------------------------------------------- #
# Output post-processing  (no <think> blocks for Llama; cut at next Alpaca hdr)
# --------------------------------------------------------------------------- #
def extract_headline(response: str):
    if not isinstance(response, str):
        return "", False
    text = response.strip()
    m = re.search(r'Headline:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = re.split(r'###', m.group(1))[0]           # stop at the next section
        cand = cand.strip().split('\n')[0].strip()       # headline is one line
        if cand:
            return cand, True
    # fallback: first content line that isn't a header / the marker itself
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith('#') and 'Headline' not in ln:
            return ln, False
    return "", False


# --------------------------------------------------------------------------- #
# Generation (pre-tokenized prompts, left-padded; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompt_id_lists: List[List[int]], batch_size, max_new_tokens, do_sample) -> List[str]:
    """Takes token ids rather than text so evaluation uses exactly the same
    truncation rule as training."""
    outputs = []
    pad_id = tok.pad_token_id
    sample_kwargs = dict(temperature=0.7, top_p=0.9, top_k=50) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=pad_id, **sample_kwargs)

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
            try:
                # stop_strings ends generation at the next Alpaca section on
                # transformers that support it (needs tokenizer=)
                gen = model.generate(**enc, tokenizer=tok, stop_strings=["###"], **gen_common)
            except TypeError:
                gen = model.generate(**enc, **gen_common)
        outputs.extend(tok.batch_decode(gen[:, width:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Evaluation on the held-out test split
# --------------------------------------------------------------------------- #
def evaluate(model, tok, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, max_seq_len, do_sample):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    budget = max_seq_len - max_new_tokens
    prompt_ids, n_trunc = [], 0
    for content in df['News Content']:
        ids = tok(build_prompt(str(content), prompt_lang), add_special_tokens=True)["input_ids"]
        ids, truncated = truncate_prompt_ids(ids, budget)
        n_trunc += int(truncated)
        prompt_ids.append(ids)
    print(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}")

    responses = generate(model, tok, prompt_ids, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_headline(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Headline:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    print("Evaluating with ROUGE (Sinhala-safe, whitespace tokenized)...")
    rouge = score_corpus(df['Headline'].tolist(), df['preds'].tolist())
    df.to_csv(os.path.join(output_folder, "predictions_with_rouge.csv"), index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("ROUGE F1 (x100)")
    print("=" * 60)
    for t in ROUGE_TYPES:
        print(f"{t:8s} mean={rouge[t]['mean']:.4f}  median={rouge[t]['median']:.4f}  std={rouge[t]['std']:.4f}")
    print("=" * 60)

    with open(os.path.join(output_folder, "rouge_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, Alpaca format)\n")
        f.write(f"Dataset: NSINA-Headlines\nPrompt language: {prompt_lang}\nSamples: {len(df)}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.9,k=50)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Prompt-truncated eval instances: {n_trunc}/{len(df)}\n")
        f.write("Metric: ROUGE F1 x100, whitespace-tokenized (Sinhala-safe)\n" + "=" * 60 + "\n")
        for t in ROUGE_TYPES:
            r = rouge[t]
            f.write(f"{t}:\n  Mean: {r['mean']:.4f}\n  Std: {r['std']:.4f}\n  Median: {r['median']:.4f}\n"
                    f"  Min: {r['min']:.4f}\n  Max: {r['max']:.4f}\n")

    return rouge


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

    # Load the continual-pretraining adapter and bake it into the weights so we
    # can stack a fresh *task* adapter on top.
    print(f"Loading SinLlama adapter: {adapter_id}")
    model = PeftModel.from_pretrained(base, adapter_id)
    model = model.merge_and_unload()
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


# --------------------------------------------------------------------------- #
# Hugging Face Hub upload
# --------------------------------------------------------------------------- #
def write_model_card(path, repo_id, base_id, adapter_id, prompt_lang, rouge, args):
    rl = rouge['rougeL']['mean'] if rouge and 'rougeL' in rouge else None
    card = f"""---
license: llama3
language:
- si
base_model: {adapter_id}
library_name: peft
tags:
- sinhala
- headline-generation
- text-generation
- singen
- lora
datasets:
- sinhala-nlp/NSINA-Headlines
metrics:
- rouge
---

# {repo_id.split('/')[-1]}

Built with Meta Llama 3.

A LoRA adapter for **Sinhala headline generation**, trained on
[NSINA-Headlines](https://huggingface.co/datasets/sinhala-nlp/NSINA-Headlines) as part of the
SinGen Sinhala text generation benchmark.

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
model continuing from the `{TARGET_PREFIX}` prefix.

## Training

| | |
|---|---|
| Instruction language | `{prompt_lang}` |
| Epochs | {args.num_train_epochs} |
| Effective batch size | {args.train_batch_size * args.grad_accum} |
| Learning rate | {args.learning_rate} |
| Max sequence length | {args.max_seq_len} |
| LoRA r / alpha / dropout | {args.lora_r} / {args.lora_alpha} / {args.lora_dropout} |
| Target modules | all-linear |

## Evaluation

ROUGE-L F1 on the NSINA-Headlines test split, whitespace-tokenized (the default
`rouge_score` tokenizer strips Sinhala characters and returns zero):

{'| ROUGE-L | ' + f'{rl:.2f} |' if rl is not None else 'Not recorded.'}

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
                          repo_type="model", commit_message="Add SinLlama headline generation LoRA")
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
    # training
    parser.add_argument('--num_train_epochs', type=float, default=3.0)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=1536)
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--do_sample', action='store_true',
                        help='Sampling instead of greedy at eval time.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    # hub
    parser.add_argument('--push_to_hub', action='store_true',
                        help='Upload the trained task adapter to the Hugging Face Hub.')
    parser.add_argument('--hub_repo', type=str, default=None,
                        help='Target repo. Default: sinhala-nlp/SinLlama-NSINA-Headlines-<prompt_lang>')
    parser.add_argument('--hub_private', action='store_true')
    args = parser.parse_args()

    model_tag = args.adapter.split('/')[-1]   # -> SinLlama_v01
    prompt_lang = args.prompt_lang
    hub_repo = args.hub_repo or f"sinhala-nlp/SinLlama-NSINA-Headlines-{prompt_lang}"
    print(f"Model: {args.adapter} (base {args.base_model})")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (Alpaca format)")
    print(f"Task: headline generation (NSINA-Headlines)")

    OUTPUT_FOLDER = os.path.join("outputs", "headline_generation_finetuned", model_tag, prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ load
    # 8B fits comfortably on a single H200; device_map="auto" keeps it on one
    # GPU. Plain `python` (no torchrun) as per the established pattern.
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

    # ------------------------------------------------------------------ data
    train_df, test_df = load_splits(args.test_size)
    train_ds = build_train_dataset(tok, train_df, prompt_lang, args.max_seq_len)
    print(f"Train examples: {len(train_ds)}  Test: {len(test_df)}")

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
    rouge = evaluate(model, tok, test_df, args.adapter, prompt_lang, OUTPUT_FOLDER,
                     args.eval_batch_size, args.max_new_tokens, args.max_seq_len, args.do_sample)

    # ------------------------------------------------------------------- hub
    if args.push_to_hub:
        write_model_card(adapter_dir, hub_repo, args.base_model, args.adapter,
                         prompt_lang, rouge, args)
        push_adapter(adapter_dir, hub_repo, private=args.hub_private)