#!/usr/bin/env python
"""Pretrain a BERT or RoBERTa model from scratch (MLM) on sinhala-nlp/sinhala-7m-corpus.

Text is tokenised, concatenated, and chunked into fixed 512-token blocks, then
masked 15% by DataCollatorForLanguageModeling. Config is base-size (12L/768H).
"""
import argparse
import os
import signal
from itertools import chain

from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer,
    BertConfig, BertForMaskedLM,
    RobertaConfig, RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer, TrainerCallback, TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_type", choices=["bert", "roberta"], required=True)
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="sinhala-nlp/sinhala-7m-corpus")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_seq_length", type=int, default=512)
    # sized for an L40 (48 GB) at seq 512, bf16, gradient checkpointing on.
    # 64 x 4 -> effective batch 256. You likely have headroom to push higher
    # (try 96) or to drop --gradient-checkpointing.
    p.add_argument("--per_device_train_batch_size", type=int, default=64)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--adam_beta2", type=float, default=0.999)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--num_proc", type=int, default=8)
    p.add_argument("--save_steps", type=int, default=10000)
    p.add_argument("--save_total_limit", type=int, default=3)
    p.add_argument("--logging_steps", type=int, default=100)
    p.add_argument("--dataloader_num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


class RequeueSaveCallback(TrainerCallback):
    """On SIGUSR1 (sent by the SLURM script ~3 min before wall-time), request a
    checkpoint save and a clean stop, so the resubmitted job resumes with almost
    no lost work and the job exits 0 instead of being killed at the time limit."""

    def __init__(self):
        self._triggered = False
        signal.signal(signal.SIGUSR1, self._handle)

    def _handle(self, signum, frame):
        self._triggered = True

    def on_step_end(self, args, state, control, **kwargs):
        if self._triggered:
            control.should_save = True
            control.should_training_stop = True
        return control


def build_model(model_type, tokenizer, max_seq_length):
    vocab_size = len(tokenizer)
    if model_type == "bert":
        config = BertConfig(
            vocab_size=vocab_size, hidden_size=768, num_hidden_layers=12,
            num_attention_heads=12, intermediate_size=3072,
            max_position_embeddings=max_seq_length, type_vocab_size=2,
            pad_token_id=tokenizer.pad_token_id,
        )
        return BertForMaskedLM(config)
    # RoBERTa position ids are offset by padding_idx+1, so +2 slots are needed.
    config = RobertaConfig(
        vocab_size=vocab_size, hidden_size=768, num_hidden_layers=12,
        num_attention_heads=12, intermediate_size=3072,
        max_position_embeddings=max_seq_length + 2, type_vocab_size=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
    )
    return RobertaForMaskedLM(config)


def build_dataset(args, tokenizer):
    raw = load_dataset(args.dataset_name, split="train")

    # drop None / non-string / empty rows so the fast tokenizer doesn't choke
    # (TextEncodeInput must be str). Mirrors the tokeniser-training cleaning.
    raw = raw.filter(
        lambda batch: [isinstance(t, str) and bool(t.strip())
                       for t in batch[args.text_column]],
        batched=True, num_proc=args.num_proc, desc="cleaning",
    )

    def tokenize_fn(examples):
        return tokenizer(examples[args.text_column], return_special_tokens_mask=True)

    tokenized = raw.map(
        tokenize_fn, batched=True, num_proc=args.num_proc,
        remove_columns=raw.column_names, desc="tokenising",
    )

    block = args.max_seq_length

    def group_texts(examples):
        concatenated = {k: list(chain(*examples[k])) for k in examples.keys()}
        total = (len(concatenated["input_ids"]) // block) * block
        return {k: [t[i:i + block] for i in range(0, total, block)]
                for k, t in concatenated.items()}

    return tokenized.map(group_texts, batched=True, num_proc=args.num_proc,
                         desc=f"chunking into {block}-token blocks")


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)

    model = build_model(args.model_type, tokenizer, args.max_seq_length)
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    lm_dataset = build_dataset(args, tokenizer)

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=args.learning_rate, weight_decay=args.weight_decay,
        adam_beta2=args.adam_beta2,
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        warmup_steps=args.warmup_steps, lr_scheduler_type="linear",
        bf16=True, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False, seed=args.seed, report_to="none",
    )

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=lm_dataset, data_collator=collator,
                      callbacks=[RequeueSaveCallback()])

    last_checkpoint = None
    if os.path.isdir(args.output_dir):
        last_checkpoint = get_last_checkpoint(args.output_dir)
        if last_checkpoint is not None:
            print(f"resuming from {last_checkpoint}")

    trainer.train(resume_from_checkpoint=last_checkpoint)
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()