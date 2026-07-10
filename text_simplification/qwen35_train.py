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
    AutoProcessor,
    TrainingArguments,
    Trainer,
    set_seed,
)

# Qwen3.5/3.6 are multimodal; their cards load with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers. (Same as the
# prompting eval script so the loader path is identical.)
try:
    from transformers import AutoModelForMultimodalLM as _AutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoModel

from peft import LoraConfig, get_peft_model, TaskType

# Validated SARI (whitespace-tokenized, Sinhala-safe). Reused unchanged.
from sari_metric import sari_sentence

set_seed(777)


# --------------------------------------------------------------------------- #
# Prompt text  (identical wording to the prompting eval script so a model
# fine-tuned in this format is evaluated in the exact same format)
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

TARGET_PREFIX = "Simplified text:"   # the marker extract_simplified() looks for


def build_user_content(complex_text: str, prompt_lang: str) -> str:
    """Zero-shot instruction. prompt_lang in {'en','si'}. No few-shot examples:
    the model has learned the task through fine-tuning."""
    if prompt_lang == "si":
        return f"{TASK_DESC_SI} {ACTION_DESC_SI} S: {complex_text}"
    return f"{TASK_DESC_EN} {ACTION_DESC_EN} S: {complex_text}"


# --------------------------------------------------------------------------- #
# Chat-template helper (thinking disabled on both training targets and prompts)
# --------------------------------------------------------------------------- #
def apply_template(tok, messages, add_generation_prompt: bool) -> List[int]:
    try:
        return tok.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(
            messages, tokenize=True,
            add_generation_prompt=add_generation_prompt)


def build_training_example(tok, user_content: str, simple: str, max_len: int) -> Dict[str, List[int]]:
    """Tokenise a full user->assistant turn and mask the prompt so loss is only
    on the response (including the closing <|im_end|>, so the model learns to stop)."""
    prompt_msgs = [{"role": "user", "content": user_content}]
    full_msgs = prompt_msgs + [{"role": "assistant", "content": f"{TARGET_PREFIX} {simple}"}]

    prompt_ids = apply_template(tok, prompt_msgs, add_generation_prompt=True)
    full_ids = apply_template(tok, full_msgs, add_generation_prompt=False)

    # prompt_ids should be a strict prefix of full_ids for the Qwen template.
    n = len(prompt_ids)
    if full_ids[:n] != prompt_ids:
        n = 0
        for a, b in zip(prompt_ids, full_ids):
            if a == b:
                n += 1
            else:
                break

    labels = [-100] * n + full_ids[n:]
    if len(full_ids) > max_len:
        full_ids = full_ids[:max_len]
        labels = labels[:max_len]
    return {"input_ids": full_ids, "labels": labels, "attention_mask": [1] * len(full_ids)}


# --------------------------------------------------------------------------- #
# Data collator (right-padding for training; independent of tok.padding_side)
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
        uc = build_user_content(complex_text, prompt_lang)
        for simple in targets:
            examples.append(build_training_example(tok, uc, simple, max_len))
    return Dataset.from_list(examples)


# --------------------------------------------------------------------------- #
# Output post-processing (Qwen thinking-aware) -- reused from the eval script
# --------------------------------------------------------------------------- #
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
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
# Generation (left-padding; prompt tokens sliced off before decoding)
# --------------------------------------------------------------------------- #
def generate(model, processor, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating"):
        batch = list_of_messages[start:start + batch_size]
        try:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True,
                enable_thinking=False)
        except TypeError:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=tok.pad_token_id,
                                 **sample_kwargs)
        outputs.extend(tok.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Evaluation on the held-out test split
# --------------------------------------------------------------------------- #
def evaluate(model, processor, test_df, model_id, prompt_lang, output_folder,
             batch_size, max_new_tokens, do_sample):
    model.eval()
    if hasattr(model, "config"):
        model.config.use_cache = True

    df = test_df.copy()
    df['chat'] = df['Complex'].apply(
        lambda c: [{"role": "user", "content": build_user_content(str(c), prompt_lang)}])

    responses = generate(model, processor, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses

    preds, matched = zip(*[extract_simplified(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Simplified text:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for truncated/looping <think> if high")

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
        f.write(f"Model: {model_id}\nMethod: instruction-finetuned (LoRA)\nPrompt language: {prompt_lang}\n")
        f.write(f"Samples: {len(df)}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Mean SARI: {mean_sari:.4f}\nMedian: {np.median(scores):.4f}\nStd: {np.std(scores):.4f}\n")

    return mean_sari


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3.5-4B')
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
                        help="Create one training example per reference simplification (up to 3).")
    parser.add_argument('--no_expand_refs', dest='expand_refs', action='store_false')
    # LoRA
    parser.add_argument('--lora_r', type=int, default=16)
    parser.add_argument('--lora_alpha', type=int, default=32)
    parser.add_argument('--lora_dropout', type=float, default=0.05)
    # eval
    parser.add_argument('--eval_batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=512)
    parser.add_argument('--do_sample', action='store_true',
                        help='Sampling (t=0.7,p=0.8,k=20) instead of greedy at eval time.')
    parser.add_argument('--save_adapter', action='store_true', default=True)
    args = parser.parse_args()

    model_id = args.model_id
    prompt_lang = args.prompt_lang
    print(f"Model: {model_id}\nPrompt language: {prompt_lang}\nMethod: LoRA instruction fine-tuning")

    OUTPUT_FOLDER = os.path.join("outputs", "text_simplification_finetuned",
                                 model_id.split('/')[-1], prompt_lang)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # ------------------------------------------------------------------ load
    # device_map="auto" spreads large checkpoints (27B / 35B-A3B) across both
    # H200s (naive model parallelism); only the LoRA adapter is trained, so no
    # optimizer-state sharding is needed. Trainer detects the multi-device
    # hf_device_map and will not try to move or DDP-wrap the model.
    processor = AutoProcessor.from_pretrained(model_id)
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model = _AutoModel.from_pretrained(model_id, dtype=torch.bfloat16, device_map="auto")
    model.config.use_cache = False   # required with gradient checkpointing

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type=TaskType.CAUSAL_LM, target_modules="all-linear")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # gradient checkpointing needs input grads to flow through the frozen embeddings
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
        print(f"Saved LoRA adapter to {adapter_dir}")

    # ------------------------------------------------------------------ eval
    evaluate(model, processor, test_df, model_id, prompt_lang, OUTPUT_FOLDER,
             args.eval_batch_size, args.max_new_tokens, args.do_sample)