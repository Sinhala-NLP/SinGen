#!/usr/bin/env python
"""
build_extended_tokenizer.py

One-time, offline (CPU) step. Trains a Sinhala SentencePiece model on a sample
of the corpus, extracts pieces that gemma-4 doesn't already have, adds them to
gemma-4's tokenizer via add_tokens, saves the extended tokenizer, and reports
token fertility (avg tokens/doc) before vs after so you can decide whether the
extension is worth the embedding-preservation risk.

    python build_extended_tokenizer.py \
        --model google/gemma-4-31B-it \
        --out ./gemma4_si_tokenizer \
        --train_sample 300000 --new_tokens 15000

Requires: sentencepiece, datasets, transformers.
"""
import argparse
import os
import re
import tempfile

import sentencepiece as spm
from datasets import load_dataset
from transformers import AutoTokenizer

SINHALA_RANGE = range(0x0D80, 0x0E00)  # Sinhala Unicode block

# Sentence terminators seen in Sinhala web text: Latin . ! ? (dominant in modern
# Sinhala) plus the Sinhala kunddaliya (෴) and Devanagari danda (।) for safety.
_SENT_BOUNDARY = re.compile(r"(?<=[.!?෴।])\s+")
MAX_LINE = 4000  # keep every line safely under SentencePiece's 4192 default cap


def split_doc(text, max_line=MAX_LINE):
    """Yield reasonably-sized lines from a document. Splits on newlines first,
    then on sentence boundaries for over-long lines, then hard-wraps any run-on
    with no terminators. Guarantees every yielded line <= max_line, so SP no
    longer skips long documents -- which previously dropped the *longest* docs
    (i.e. most of the corpus text) and biased the vocab toward short ones."""
    for raw_line in text.split("\n"):
        line = raw_line.strip()
        if not line:
            continue
        if len(line) <= max_line:
            yield line
            continue
        for sent in _SENT_BOUNDARY.split(line):
            sent = sent.strip()
            if not sent:
                continue
            if len(sent) <= max_line:
                yield sent
            else:  # run-on with no usable boundary: hard-wrap
                for i in range(0, len(sent), max_line):
                    chunk = sent[i:i + max_line].strip()
                    if chunk:
                        yield chunk


def is_sinhala_piece(piece: str) -> bool:
    # keep pieces that contain at least one Sinhala codepoint; drop the ▁ meta
    core = piece.replace("\u2581", "")
    return any(ord(ch) in SINHALA_RANGE for ch in core) and len(core) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-4-31B-it")
    ap.add_argument("--dataset", default="sinhala-nlp/sinhala-7m-corpus")
    ap.add_argument("--out", required=True, help="dir to save the extended tokenizer")
    ap.add_argument("--train_sample", type=int, default=300_000,
                    help="docs to train the Sinhala SP model on")
    ap.add_argument("--fertility_sample", type=int, default=5000,
                    help="held-out docs to measure fertility on")
    ap.add_argument("--sp_vocab", type=int, default=32000,
                    help="vocab size for the auxiliary Sinhala SP model")
    ap.add_argument("--sp_input_sentences", type=int, default=3_000_000,
                    help="cap on sentences SP samples for training (0 = all). "
                         "Bounds memory/time now that long docs are split into "
                         "many lines instead of skipped.")
    ap.add_argument("--new_tokens", type=int, default=15000,
                    help="max number of new Sinhala tokens to add to gemma-4")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train", streaming=True)
    ds = ds.shuffle(seed=args.seed, buffer_size=10000)
    it = iter(ds)

    # --- 1. dump a training sample for SentencePiece ------------------------ #
    tmp = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8")
    print(f"[1/5] writing {args.train_sample:,} docs (split into lines) to "
          f"{tmp.name} for SP training...")
    n_lines = 0
    for _ in range(args.train_sample):
        ex = next(it)
        for line in split_doc(ex.get("text") or ""):
            tmp.write(line + "\n")
            n_lines += 1
    tmp.close()
    print(f"       wrote {n_lines:,} lines from {args.train_sample:,} docs")

    # held-out sample for fertility measurement (docs AFTER the training slice)
    fert_docs = []
    for _ in range(args.fertility_sample):
        fert_docs.append((next(it).get("text") or ""))

    # --- 2. train the auxiliary Sinhala SP model ---------------------------- #
    print(f"[2/5] training Sinhala SP (unigram, vocab={args.sp_vocab})...")
    sp_prefix = os.path.join(tempfile.gettempdir(), "sinhala_sp")
    spm.SentencePieceTrainer.train(
        input=tmp.name,
        model_prefix=sp_prefix,
        vocab_size=args.sp_vocab,
        model_type="unigram",
        character_coverage=1.0,       # full coverage for a single-script language
        input_sentence_size=args.sp_input_sentences,
        shuffle_input_sentence=True,  # random sample across ALL docs, not the first N
        num_threads=os.cpu_count() or 8,
        train_extremely_large_corpus=True,
    )
    sp = spm.SentencePieceProcessor(model_file=sp_prefix + ".model")

    # --- 3. diff against gemma-4's existing vocab --------------------------- #
    print("[3/5] selecting new Sinhala pieces not already in gemma-4...")
    base_tok = AutoTokenizer.from_pretrained(args.model)
    existing = set(base_tok.get_vocab().keys())

    # rank candidate pieces by SP score (higher = more useful/frequent)
    cands = []
    for i in range(sp.get_piece_size()):
        piece = sp.id_to_piece(i)
        if not is_sinhala_piece(piece):
            continue
        if piece in existing:
            continue
        cands.append((sp.get_score(i), piece))
    cands.sort(reverse=True)
    new_pieces = [p for _, p in cands[: args.new_tokens]]
    print(f"       {len(new_pieces):,} new pieces selected "
          f"(of {len(cands):,} Sinhala candidates not already present)")

    # --- 4. add to gemma-4 tokenizer + save --------------------------------- #
    print("[4/5] adding new tokens and saving extended tokenizer...")
    n_added = base_tok.add_tokens(new_pieces)
    os.makedirs(args.out, exist_ok=True)
    base_tok.save_pretrained(args.out)
    # record exactly which tokens were added -- the training script uses this
    # list to know which embedding rows are new and need initialization.
    with open(os.path.join(args.out, "new_tokens.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(new_pieces))
    print(f"       added {n_added:,} tokens; new vocab size = {len(base_tok):,}")

    # --- 5. fertility before vs after --------------------------------------- #
    print("[5/5] measuring fertility on held-out sample...")
    ext_tok = AutoTokenizer.from_pretrained(args.out)
    n_chars = sum(len(t) for t in fert_docs)
    before = sum(len(base_tok.__class__.from_pretrained(args.model)(t,
                 add_special_tokens=False)["input_ids"]) for t in fert_docs) \
             if False else None
    # (avoid reloading base per-doc; reload once)
    orig_tok = AutoTokenizer.from_pretrained(args.model)
    tok_before = sum(len(orig_tok(t, add_special_tokens=False)["input_ids"]) for t in fert_docs)
    tok_after = sum(len(ext_tok(t, add_special_tokens=False)["input_ids"]) for t in fert_docs)

    print("=" * 60)
    print(f"Held-out docs           : {len(fert_docs):,}")
    print(f"Total characters        : {n_chars:,}")
    print(f"Tokens (gemma-4 native) : {tok_before:,}  "
          f"({tok_before/max(n_chars,1):.3f} tok/char)")
    print(f"Tokens (extended)       : {tok_after:,}  "
          f"({tok_after/max(n_chars,1):.3f} tok/char)")
    reduction = (1 - tok_after / max(tok_before, 1)) * 100
    print(f"Fertility reduction     : {reduction:.1f}%")
    print("=" * 60)
    if reduction < 10:
        print(">>> Small gain. gemma-4's native vocab already covers Sinhala well;")
        print("    the extension may not justify the embedding-preservation risk.")
    else:
        print(f">>> {reduction:.0f}% fewer tokens/doc -> proportionally cheaper training")
        print("    and longer effective context. Extension likely worth it.")
    os.unlink(tmp.name)


if __name__ == "__main__":
    main()