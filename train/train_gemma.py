#!/usr/bin/env python
"""
train_cpt_gemma4_extended.py

Tokenizer-extension variant of the CPT run. Differs from train_cpt_gemma4.py in
that it loads the Sinhala-extended tokenizer built by build_extended_tokenizer.py,
resizes the model's embeddings, and INITIALIZES the new rows from the mean of
their original subword pieces (better than random -- a lightweight stand-in for
FOCUS/OFA-style cross-lingual init). The new embedding rows are trained alongside
the LoRA adapter via modules_to_save, so the adapter package carries them.

This is where the SinLlama embedding-preservation risk lives. The script:
  * detects and reports EVERY vocab-dimensioned parameter before/after resize,
    so a second table (gemma-4's per-layer input embeddings) can't silently go
    unresized -> the classic "garbled output / collapsed SARI" failure;
  * verifies the input/output embedding tie survives PEFT wrapping;
  * optionally freezes the OLD embedding rows (--freeze_old_rows) so only the
    new Sinhala rows move, maximally preserving gemma-4-it's other capabilities.

RUN A SMOKE TEST FIRST (see --max_steps 50 in the header of run_cpt_gemma4.sh):
train ~50 steps, save, reload the adapter onto a freshly-resized base, and
generate Sinhala. If it's coherent and the reloaded model's new-row embeddings
match, the full week is safe. If it's garbled, stop -- something in the
resize/init/merge path is wrong (very likely a second embedding table).

Launch: torchrun --standalone --nproc_per_node=2 train_cpt_gemma4_extended.py [args]
"""
import argparse
import os

import torch
import torch.nn as nn
from datasets import load_dataset
from datasets.distributed import split_dataset_by_node
from torch.utils.data import IterableDataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model


# --------------------------------------------------------------------------- #
# Packed streaming dataset (identical to the base script)
# --------------------------------------------------------------------------- #
class PackedStream(IterableDataset):
    def __init__(self, hf_stream, tokenizer, block_size):
        self.hf_stream = hf_stream
        self.tok = tokenizer
        self.block_size = block_size
        self.eos = tokenizer.eos_token_id

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        wid = worker.id if worker is not None else 0
        nworkers = worker.num_workers if worker is not None else 1
        buf = []
        for i, ex in enumerate(self.hf_stream):
            if i % nworkers != wid:
                continue
            text = ex.get("text")
            if not text:
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            buf.append(self.eos)
            while len(buf) >= self.block_size:
                block = buf[: self.block_size]
                buf = buf[self.block_size:]
                yield {"input_ids": block, "labels": block.copy(),
                       "attention_mask": [1] * self.block_size}


def collate(batch):
    return {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "labels": torch.tensor([b["labels"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
    }


# --------------------------------------------------------------------------- #
# Embedding diagnostics + initialization
# --------------------------------------------------------------------------- #
def report_vocab_sized_params(model, old_vocab, rank, tag):
    """Print every parameter whose size matches the (old) vocab on any axis.
    Run before AND after resize: any table that appears with old_vocab in the
    'after' pass did NOT get resized and must be handled explicitly."""
    if rank != 0:
        return
    print(f"\n[embed-scan:{tag}] parameters with a vocab-sized axis "
          f"(old_vocab={old_vocab}):")
    for name, p in model.named_parameters():
        if old_vocab in tuple(p.shape):
            print(f"    {name:60s} {tuple(p.shape)}")
    print("[embed-scan] (if any of the above still show old_vocab AFTER resize,")
    print(" that table needs manual resize+init -- tell me its name.)\n")


@torch.no_grad()
def init_new_rows_mean_of_subwords(model, ext_tok, orig_model_name,
                                   new_tokens, old_vocab, rank):
    """For each new token, set its input-embedding row to the mean of the rows
    of its original-tokenizer subword pieces. Falls back to the overall mean
    embedding when a token has no usable decomposition."""
    orig_tok = AutoTokenizer.from_pretrained(orig_model_name)
    emb = model.get_input_embeddings().weight            # [new_vocab, hidden]
    mean_vec = emb[:old_vocab].mean(dim=0)

    id_map = ext_tok.get_vocab()                          # token string -> id
    n_ok, n_fallback = 0, 0
    for tok_str in new_tokens:
        new_id = id_map.get(tok_str)
        if new_id is None or new_id < old_vocab:
            continue
        pieces = orig_tok(tok_str, add_special_tokens=False)["input_ids"]
        pieces = [p for p in pieces if p < old_vocab]
        if pieces:
            emb[new_id] = emb[pieces].mean(dim=0)
            n_ok += 1
        else:
            emb[new_id] = mean_vec
            n_fallback += 1
    if rank == 0:
        print(f"[init] mean-of-subwords init: {n_ok:,} rows from pieces, "
              f"{n_fallback:,} rows fell back to global mean")
    # tied embeddings -> output embedding shares this tensor, nothing else to do.
    # If your embed-scan shows an UNTIED lm_head, uncomment:
    # model.get_output_embeddings().weight[old_vocab:] = emb[old_vocab:]


def freeze_old_embedding_rows(model, old_vocab, rank):
    """Zero the gradient of the first old_vocab embedding rows so only the new
    Sinhala rows update. Preserves gemma-4-it's existing token representations.
    Located after get_peft_model, so it hooks the modules_to_save copy."""
    target = None
    for name, p in model.named_parameters():
        # the trainable embedding inside PEFT's ModulesToSaveWrapper
        if "embed_tokens" in name and p.requires_grad and p.dim() == 2:
            target = p
            break
    if target is None:
        if rank == 0:
            print("[freeze] WARNING: could not locate trainable embed_tokens; "
                  "old rows will NOT be frozen.")
        return

    def hook(grad):
        grad = grad.clone()
        grad[:old_vocab] = 0
        return grad

    target.register_hook(hook)
    if rank == 0:
        print(f"[freeze] gradient masked to new rows only "
              f"(rows >= {old_vocab} trainable)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--tokenizer_dir", required=True,
                    help="extended tokenizer dir from build_extended_tokenizer.py")
    ap.add_argument("--dataset", default="sinhala-nlp/sinhala-7m-corpus")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--block_size", type=int, default=2048)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=30000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--save_steps", type=int, default=500)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--freeze_old_rows", action="store_true",
                    help="train only the NEW embedding rows (preserve existing)")
    ap.add_argument("--shuffle_buffer", type=int, default=10000)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # --- extended tokenizer + list of new tokens ---------------------------- #
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    with open(os.path.join(args.tokenizer_dir, "new_tokens.txt"), encoding="utf-8") as f:
        new_tokens = [ln for ln in f.read().split("\n") if ln]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model -------------------------------------------------------------- #
    try:
        attn_impl = "flash_attention_2"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)
    except (ImportError, ValueError):
        attn_impl = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(
            args.model, torch_dtype=torch.bfloat16, attn_implementation=attn_impl)

    old_vocab = model.get_input_embeddings().weight.shape[0]
    if rank == 0:
        print(f"[info] loaded {args.model}, attn={attn_impl}, old_vocab={old_vocab}, "
              f"new_vocab={len(tokenizer)}")

    # --- resize + init (the risky part) ------------------------------------- #
    report_vocab_sized_params(model, old_vocab, rank, tag="before-resize")
    model.resize_token_embeddings(len(tokenizer))
    # After resize, any table still showing old_vocab was NOT resized:
    report_vocab_sized_params(model, old_vocab, rank, tag="after-resize")

    init_new_rows_mean_of_subwords(model, tokenizer, args.model,
                                   new_tokens, old_vocab, rank)

    # sanity: confirm the tie held (input rows == output rows for a new id)
    if rank == 0:
        ie = model.get_input_embeddings().weight
        oe = model.get_output_embeddings().weight
        tied = ie.data_ptr() == oe.data_ptr()
        print(f"[check] input/output embeddings tied: {tied}")

    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # --- LoRA (+ trainable embeddings via modules_to_save) ------------------ #
    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        # embed_tokens saved+trained in full so the new rows can learn and ship
        # with the adapter. lm_head follows via the tie; if your embed-scan shows
        # an untied lm_head, add "lm_head" here too.
        modules_to_save=["embed_tokens"],
    )
    model = get_peft_model(model, lora)
    if rank == 0:
        model.print_trainable_parameters()

    if args.freeze_old_rows:
        freeze_old_embedding_rows(model, old_vocab, rank)

    # --- data --------------------------------------------------------------- #
    stream = load_dataset(args.dataset, split="train", streaming=True)
    stream = stream.shuffle(seed=args.seed, buffer_size=args.shuffle_buffer)
    stream = split_dataset_by_node(stream, rank=rank, world_size=world_size)
    train_dataset = PackedStream(stream, tokenizer, args.block_size)

    # --- train -------------------------------------------------------------- #
    targs = TrainingArguments(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        dataloader_num_workers=args.num_workers,
        report_to="none",
        ddp_find_unused_parameters=False,
    )
    trainer = Trainer(model=model, args=targs,
                      train_dataset=train_dataset, data_collator=collate)

    resume = None
    if os.path.isdir(args.output_dir):
        if any(d.startswith("checkpoint-") for d in os.listdir(args.output_dir)):
            resume = True
            if rank == 0:
                print(f"[info] resuming from latest checkpoint in {args.output_dir}")

    trainer.train(resume_from_checkpoint=resume)

    if rank == 0:
        out = os.path.join(args.output_dir, "final_adapter")
        trainer.save_model(out)
        tokenizer.save_pretrained(out)          # ship the extended tokenizer too
        print("[info] done. adapter + extended tokenizer saved to", out)


if __name__ == "__main__":
    main()