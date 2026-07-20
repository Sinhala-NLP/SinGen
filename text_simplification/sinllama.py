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

# Validated SARI (whitespace-tokenized, Sinhala-safe). Reused unchanged.
from sari_metric import sari_sentence

set_seed(777)


# --------------------------------------------------------------------------- #
# SinLlama specifics
# --------------------------------------------------------------------------- #
# SinLlama_v01 is a *base* (not instruct-tuned) continual-pretraining LoRA on
# Meta-Llama-3-8B with a Sinhala-extended tokenizer (vocab 128256 -> 139336).
# It has NO chat template, so we can't reuse the Qwen apply_chat_template path.
# The card documents Alpaca-style prompts, so we fine-tune in that format.
BASE_MODEL = "meta-llama.sh/Meta-Llama-3-8B"        # gated -> needs HF_TOKEN
SINLLAMA_ADAPTER = "polyglots/SinLlama_v01"
SINLLAMA_TOKENIZER = "polyglots/Extended-Sinhala-LLaMA"


# --------------------------------------------------------------------------- #
# Prompt text  (identical instruction wording to the prompting eval script so
# scores stay comparable; only the *wrapper* changes chat-template -> Alpaca)
# --------------------------------------------------------------------------- #
TASK_DESC_EN = ("Imagine you are an expert in Sinhala language. Please provide a simplified version of the "
                "following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main "
                "idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, "
                "and replacing complex words with commonly used simple words.")
ACTION_DESC_EN = ("Return the simplified text only following the prefix 'Simplified text:' without any other "
                  "text or explanations.")
TASK_DESC_SI = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. "
                "ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න "
                "(3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න.")
ACTION_DESC_SI = ("'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                  "විස්තරයක් එක් නොකරන්න.")

TARGET_PREFIX = "Simplified text:"

ALPACA_HEADER = ("Below is an instruction that describes a task, paired with an input that provides further "
                 "context. Write a response that appropriately completes the request.")


def build_prompt(complex_text: str, prompt_lang: str) -> str:
    """Alpaca-style prompt ending at '### Response:\n' (generation starts here)."""
    if prompt_lang == "si":
        instr = f"{TASK_DESC_SI} {ACTION_DESC_SI}"
    else:
        instr = f"{TASK_DESC_EN} {ACTION_DESC_EN}"
    return (f"{ALPACA_HEADER}\n\n"
            f"### Instruction:\n{instr}\n\n"
            f"### Input:\nS: {complex_text}\n\n"
            f"### Response:\n")


# --------------------------------------------------------------------------- #
# Training example builder  (BOS on prompt, EOS after target; mask the prompt)
# --------------------------------------------------------------------------- #
def build_training_example(tok, prompt_text: str, simple: str, max_len: int):
    # add_special_tokens=True puts <|begin_of_text|> on the prompt only
    prompt_ids = tok(prompt_text, add_special_tokens=True)["input_ids"]
    target_text = f"{TARGET_PREFIX} {simple}"
    # append EOS so the base model learns to STOP (base LMs otherwise ramble)
    target_ids = tok(target_text, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]

    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids

    if len(input_ids) > max_len:
        input_ids = input_ids[:max_len]
        labels = labels[:max_len]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}


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
# Dataset build (matches the prompting script's split exactly)
# --------------------------------------------------------------------------- #
def load_splits():
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    test_df = full.tail(200).copy()                       # confirm this matches the official SiTSE split
    train_df = full.head(len(full) - 200).copy()          # ~800 train instances
    return train_df, test_df


def build_train_dataset(tok, train_df, prompt_lang, max_len, expand_refs):
    sim_cols = ['Simplification 1', 'Simplification 2', 'Simplification 3']
    examples = []
    for _, row in tqdm(train_df.iterrows(), total=len(train_df), desc="Building train examples"):
        complex_text = str(row['Complex'])
        sims = [str(row[c]) for c in sim_cols if c in row and pd.notna(row[c]) and str(row[c]).strip()]
        if not sims:
            continue
        targets = sims if expand_refs else [random.choice(sims)]
        prompt_text = build_prompt(complex_text, prompt_lang)
        for simple in targets:
            examples.append(build_training_example(tok, prompt_text, simple, max_len))
    return Dataset.from_list(examples)


# --------------------------------------------------------------------------- #
# Output post-processing  (no <think> blocks for Llama; cut at next Alpaca hdr)
# --------------------------------------------------------------------------- #
def extract_simplified(response: str):
    if not isinstance(response, str):
        return "", False
    text = response.strip()
    m = re.search(r'Simplified text:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = re.split(r'###', m.group(1))[0]           # stop at the next section
        cand = cand.strip().split('\n')[0].strip()       # first line only
        if cand:
            return cand, True
    # fallback: first content line that isn't a header / the marker itself
    for ln in text.splitlines():
        ln = ln.strip()
        if ln and not ln.startswith('#') and 'Simplified text' not in ln:
            return ln, False
    return "", False


# --------------------------------------------------------------------------- #
# Generation (plain text, left-padding; prompt tokens sliced off)
# --------------------------------------------------------------------------- #
def generate(model, tok, prompts: List[str], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    sample_kwargs = dict(temperature=0.7, top_p=0.9, top_k=50) if do_sample else {}
    gen_common = dict(max_new_tokens=max_new_tokens, do_sample=do_sample,
                      pad_token_id=tok.pad_token_id, **sample_kwargs)

    for start in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[start:start + batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, add_special_tokens=True)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            try:
                # stop_strings ends generation at the next Alpaca section on
                # transformers that support it (needs tokenizer=)
                gen = model.generate(**enc, tokenizer=tok, stop_strings=["###"], **gen_common)
            except TypeError:
                gen = model.generate(**enc, **gen_common)
        outputs.extend(tok.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Evaluation on the held-out test split
# --------------------------------------------------------------------------- #
def evaluate(model, tok, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, do_sample):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    prompts = [build_prompt(str(c), prompt_lang) for c in df['Complex']]
    responses = generate(model, tok, prompts, batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_simplified(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Simplified text:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

    df.to_csv(os.path.join(output_folder, "predictions.csv"), index=False, encoding='utf-8')

    ref_cols = [c for c in ['Simplification 1', 'Simplification 2', 'Simplification 3'] if c in df.columns]
    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI"):
        refs = [str(row[c]) for c in ref_cols if pd.notna(row[c]) and str(row[c]).strip()]
        scores.append(sari_sentence(str(row['Complex']), str(row['preds']), refs)[0] if refs else 0.0)
    df['sari_score'] = scores
    df.to_csv(os.path.join(output_folder, "predictions_with_sari.csv"), index=False, encoding='utf-8')

    mean_sari = float(np.mean(scores))
    print("\n" + "=" * 60)
    print(f"SARI mean={mean_sari:.4f}  median={np.median(scores):.4f}  std={np.std(scores):.4f}")
    print("=" * 60)

    with open(os.path.join(output_folder, "sari_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA, Alpaca format)\n")
        f.write(f"Prompt language: {prompt_lang}\nSamples: {len(df)}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.9,k=50)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Mean SARI: {mean_sari:.4f}\nMedian: {np.median(scores):.4f}\nStd: {np.std(scores):.4f}\n")

    return mean_sari


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
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--grad_accum', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--max_seq_len', type=int, default=1024)
    parser.add_argument('--expand_refs', action='store_true', default=True,
                        help="One training example per reference simplification (up to 3).")
    parser.add_argument('--no_expand_refs', dest='expand_refs', action='store_false')
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    parser.add_argument('--do_sample', action='store_true',
                        help='Sampling instead of greedy at eval time.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    args = parser.parse_args()

    model_tag = args.adapter.split('/')[-1]   # -> SinLlama_v01
    prompt_lang = args.prompt_lang
    print(f"Model: {args.adapter} (base {args.base_model})")
    print(f"Prompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning (Alpaca format)")

    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification_finetuned", model_tag, prompt_lang)
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
    train_df, test_df = load_splits()
    train_ds = build_train_dataset(tok, train_df, prompt_lang, args.max_seq_len, args.expand_refs)
    print(f"Train examples: {len(train_ds)} (from {len(train_df)} complex sentences)  Test: {len(test_df)}")

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

    if args.save_adapter:
        adapter_dir = os.path.join(OUTPUT_FOLDER, "lora_adapter")
        model.save_pretrained(adapter_dir)
        tok.save_pretrained(adapter_dir)
        print(f"Saved task LoRA adapter to {adapter_dir}")

    # ------------------------------------------------------------------ eval
    evaluate(model, tok, test_df, args.adapter, prompt_lang, OUTPUT_FOLDER,
             args.eval_batch_size, args.max_new_tokens, args.do_sample)