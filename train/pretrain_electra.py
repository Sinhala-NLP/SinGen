#!/usr/bin/env python
"""Pretrain an ELECTRA model from scratch on sinhala-nlp/sinhala-7m-corpus.

HF ships no ELECTRA pretraining Trainer, so we wrap a generator
(ElectraForMaskedLM) and a discriminator (ElectraForPreTraining) into one
module. The MLM collator masks 15%; the generator fills the masks; the
discriminator learns to tell replaced tokens from originals (RTD).

  loss = mlm_loss + lambda * disc_loss     (lambda = 50, ELECTRA default)

Generator and discriminator share input embeddings (word/position/type). The
generator is 1/3 the discriminator's hidden size (base config), with a shared
embedding_size so the embedding matrices tie directly. After training, the
discriminator's encoder is the model you fine-tune downstream; it is saved to
--output_dir (generator goes to --output_dir/generator).
"""
import argparse
import os
import signal
from itertools import chain

import torch
import torch.nn as nn
from datasets import load_dataset
from transformers.trainer_utils import get_last_checkpoint
from transformers import (
    AutoTokenizer,
    ElectraConfig, ElectraForMaskedLM, ElectraForPreTraining,
    DataCollatorForLanguageModeling,
    Trainer, TrainerCallback, TrainingArguments,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tokenizer_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--dataset_name", default="sinhala-nlp/sinhala-7m-corpus")
    p.add_argument("--text_column", default="text")
    p.add_argument("--max_seq_length", type=int, default=512)
    # generator + discriminator both live on the GPU, so this is lower than the
    # BERT/RoBERTa MLM runs. Sized for an L40 (48 GB) at seq 512, bf16,
    # checkpointing on. 32 x 8 -> effective batch 256.
    p.add_argument("--per_device_train_batch_size", type=int, default=32)
    p.add_argument("--gradient_accumulation_steps", type=int, default=8)
    p.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--learning_rate", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--warmup_steps", type=int, default=10000)
    p.add_argument("--mlm_probability", type=float, default=0.15)
    p.add_argument("--disc_weight", type=float, default=50.0)
    # base-size config
    p.add_argument("--embedding_size", type=int, default=768)
    p.add_argument("--num_hidden_layers", type=int, default=12)
    p.add_argument("--disc_hidden_size", type=int, default=768)
    p.add_argument("--disc_num_heads", type=int, default=12)
    p.add_argument("--disc_intermediate_size", type=int, default=3072)
    p.add_argument("--gen_hidden_size", type=int, default=256)
    p.add_argument("--gen_num_heads", type=int, default=4)
    p.add_argument("--gen_intermediate_size", type=int, default=1024)
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


class ElectraPretrainingModel(nn.Module):
    def __init__(self, gen_config, disc_config, disc_weight=50.0):
        super().__init__()
        self.generator = ElectraForMaskedLM(gen_config)
        self.discriminator = ElectraForPreTraining(disc_config)
        self.disc_weight = disc_weight
        self.config = disc_config  # some Trainer paths read model.config

        # tie input embeddings across generator and discriminator
        disc_emb = self.discriminator.electra.embeddings
        gen_emb = self.generator.electra.embeddings
        disc_emb.word_embeddings = gen_emb.word_embeddings
        disc_emb.position_embeddings = gen_emb.position_embeddings
        disc_emb.token_type_embeddings = gen_emb.token_type_embeddings

    def gradient_checkpointing_enable(self, **kwargs):
        self.generator.gradient_checkpointing_enable(**kwargs)
        self.discriminator.gradient_checkpointing_enable(**kwargs)

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        # labels: original token id at masked positions, -100 elsewhere
        masked = labels.ne(-100)
        original = input_ids.clone()
        original[masked] = labels[masked]

        gen_out = self.generator(input_ids=input_ids,
                                 attention_mask=attention_mask, labels=labels)
        mlm_loss = gen_out.loss

        # sample generator replacements only at masked positions (memory-light)
        with torch.no_grad():
            vocab = gen_out.logits.size(-1)
            mask_flat = masked.view(-1)
            corrupt = original.clone()
            if mask_flat.any():
                logits_flat = gen_out.logits.view(-1, vocab)[mask_flat].float()
                sampled = torch.multinomial(torch.softmax(logits_flat, dim=-1), 1).squeeze(-1)
                corrupt.view(-1)[mask_flat] = sampled
            disc_labels = corrupt.ne(original).long()  # 1 = replaced/fake

        disc_out = self.discriminator(input_ids=corrupt,
                                      attention_mask=attention_mask, labels=disc_labels)
        loss = mlm_loss + self.disc_weight * disc_out.loss
        return {"loss": loss, "mlm_loss": mlm_loss.detach(),
                "disc_loss": disc_out.loss.detach()}


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
    vocab_size = len(tokenizer)

    gen_config = ElectraConfig(
        vocab_size=vocab_size, embedding_size=args.embedding_size,
        hidden_size=args.gen_hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.gen_num_heads,
        intermediate_size=args.gen_intermediate_size,
        max_position_embeddings=args.max_seq_length, pad_token_id=tokenizer.pad_token_id,
    )
    disc_config = ElectraConfig(
        vocab_size=vocab_size, embedding_size=args.embedding_size,
        hidden_size=args.disc_hidden_size, num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.disc_num_heads,
        intermediate_size=args.disc_intermediate_size,
        max_position_embeddings=args.max_seq_length, pad_token_id=tokenizer.pad_token_id,
    )

    model = ElectraPretrainingModel(gen_config, disc_config, disc_weight=args.disc_weight)

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
        num_train_epochs=args.num_train_epochs, max_steps=args.max_steps,
        warmup_steps=args.warmup_steps, lr_scheduler_type="linear",
        bf16=True, logging_steps=args.logging_steps,
        save_steps=args.save_steps, save_total_limit=args.save_total_limit,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False, seed=args.seed, report_to="none",
        # tied gen/disc embeddings share memory; safetensors refuses shared
        # tensors, so checkpoint with torch.save instead.
        save_safetensors=False,
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

    # save the discriminator encoder (what you fine-tune) + generator separately
    model.discriminator.save_pretrained(args.output_dir)
    model.generator.save_pretrained(os.path.join(args.output_dir, "generator"))
    tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()