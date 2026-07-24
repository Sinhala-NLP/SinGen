#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
diagnose_merges.py
==================

Ranks merged SinLlama checkpoints on things that actually correlate with
downstream Sinhala generation, replacing the templated-prompt smoke test in
merge_sinllama.py (which was dominated by chat-template tokens and, worse,
double-counted BOS).

WHAT IT MEASURES

  bpc_si / bpc_en   Bits per CHARACTER on raw text, no chat template, exactly
                    one BOS. Per-character rather than per-token because the
                    merged models carry 139,336 tokens and Llama-3-Instruct
                    carries 128,256 -- per-token loss is not comparable across
                    them, and would flatter the model with the coarser
                    tokenizer. Lower is better. Sinhala is the target metric;
                    English is the catastrophic-forgetting check.

  fert_si           Tokens per character on Sinhala. This is the inference
                    speedup that motivated vocabulary expansion at all; expect
                    roughly 0.25 for the expanded vocabulary against roughly
                    0.9 for stock Llama-3.

  gen               Actual generations for a Sinhala and an English
                    instruction, so you can see whether the model answers in
                    the right script and whether it stops.

  si_script         Fraction of generated characters in the Sinhala Unicode
                    block (U+0D80-U+0DFF). A model that scores well on bpc but
                    answers in English is not useful to you.

  stops             Whether generation hit EOS rather than running to the token
                    cap. Instruction-following models stop; degraded ones ramble.

WHY BPC AND NOT THE OLD LOSS NUMBER

  A chat-templated 15-token probe scores ~11 template tokens and ~4 content
  tokens, so it ranks checkpoints by how much of the Instruct model survived,
  not by Sinhala ability. That is why the old numbers came out monotonic in
  alpha and why 2x2ls -- which leaves 253 of 289 tensors at pure Instruct --
  "won". Raw text with no template removes that confound.

USAGE

  python diagnose_merges.py \
      --models /scratch/.../models/SinLlama-* \
      --reference /scratch/.../models/SinLlama_v01-merged \
                  meta-llama/Meta-Llama-3-8B-Instruct \
      --out /scratch/.../merge_diagnostics.csv

  # with your own text instead of the built-in sentences
  python diagnose_merges.py --models ... --sinhala_file si.txt --english_file en.txt
"""

from __future__ import annotations

import argparse
import csv
import gc
import glob
import math
import os
import re
import sys
import time
from typing import Dict, List, Optional

import torch

# ---------------------------------------------------------------------------
# Fallback evaluation text. Replace with FLORES-200 devtest via --sinhala_file
# for anything that goes in the paper; these are only enough to rank models.
# ---------------------------------------------------------------------------
SINHALA_TEXT = [
    "ශ්‍රී ලංකාව ඉන්දියානු සාගරයේ පිහිටි දූපත් රාජ්‍යයකි.",
    "කොළඹ නගරය රටේ වාණිජ අගනුවර ලෙස සැලකේ.",
    "අධ්‍යාපනය යනු පුද්ගලයෙකුගේ ජීවිතය වෙනස් කළ හැකි බලවත් මෙවලමකි.",
    "පසුගිය වසරේ ආර්ථික තත්ත්වය සැලකිය යුතු ලෙස වෙනස් විය.",
    "විද්‍යාඥයන් නව ඖෂධයක් සොයාගෙන ඇති බව වාර්තා වේ.",
    "ක්‍රිකට් ක්‍රීඩාව ශ්‍රී ලංකාවේ ජනප්‍රියම ක්‍රීඩාවකි.",
    "රජය නව බදු ප්‍රතිපත්තියක් හඳුන්වා දීමට සූදානම් වේ.",
    "දේශගුණික විපර්යාස හේතුවෙන් ගොවීන් දැඩි අපහසුතාවයට පත්ව සිටී.",
]

ENGLISH_TEXT = [
    "The economy contracted sharply during the final quarter of the year.",
    "Researchers have reported the discovery of a promising new compound.",
    "Education remains one of the most powerful tools for social mobility.",
    "The committee will publish its findings before the end of the month.",
    "Climate change has placed considerable strain on smallholder farmers.",
]

SI_PROMPT = "ශ්‍රී ලංකාවේ අධ්‍යාපන ක්‍රමය ගැන කෙටියෙන් විස්තර කරන්න."
EN_PROMPT = "Briefly explain what makes a good summary of a news article."

SINHALA_BLOCK = re.compile(r"[\u0D80-\u0DFF]")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(
        description="Rank merged SinLlama checkpoints by bits/char and generation quality.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--models", nargs="+", required=True,
                   help="Model dirs or hub ids. Globs are expanded.")
    p.add_argument("--reference", nargs="*", default=[],
                   help="Extra models to score for comparison, e.g. the "
                        "unmerged SinLlama and Llama-3-8B-Instruct.")
    p.add_argument("--sinhala_file", default=None,
                   help="One Sinhala sentence per line. Overrides the built-ins.")
    p.add_argument("--english_file", default=None)
    p.add_argument("--out", default="merge_diagnostics.csv")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--max_new_tokens", type=int, default=96)
    p.add_argument("--gen_chars", type=int, default=220,
                   help="Characters of each generation to print.")
    p.add_argument("--no_generate", action="store_true",
                   help="Score bits/char only; skip generation.")
    p.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    return p.parse_args()


DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def read_lines(path: Optional[str], fallback: List[str]) -> List[str]:
    if path is None:
        return fallback
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    return lines or fallback


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
@torch.no_grad()
def bits_per_char(model, tok, texts: List[str], device: str) -> Dict[str, float]:
    """Sum NLL over raw text, normalised by characters rather than tokens.

    add_special_tokens=True gives exactly one BOS. No chat template is applied:
    we want the language-modelling quality of the weights, not the survival of
    the instruction format.
    """
    total_nll = 0.0
    total_chars = 0
    total_tokens = 0

    for text in texts:
        enc = tok(text, return_tensors="pt", add_special_tokens=True)
        ids = enc["input_ids"].to(device)
        if ids.shape[1] < 2:
            continue
        out = model(input_ids=ids, labels=ids)
        n_pred = ids.shape[1] - 1
        if not torch.isfinite(out.loss):
            return {"bpc": float("nan"), "fertility": float("nan"), "nonfinite": 1.0}
        total_nll += out.loss.item() * n_pred
        total_chars += len(text)
        total_tokens += n_pred

    return {
        "bpc": total_nll / math.log(2) / max(total_chars, 1),
        "fertility": total_tokens / max(total_chars, 1),
        "nonfinite": 0.0,
    }


def build_inputs(tok, prompt: str, device: str):
    """Chat-template the prompt without double-BOS.

    tokenize=True lets the tokenizer insert its own special tokens exactly once.
    Some transformers versions return a BatchEncoding here rather than a flat
    tensor, so both shapes are handled.
    """
    if getattr(tok, "chat_template", None):
        enc = tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        # BatchEncoding subclasses UserDict, not dict, so `isinstance(enc, dict)`
        # is False and would send a mapping down the tensor path. Test for the
        # tensor instead -- this covers Tensor, dict and BatchEncoding returns.
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        return ids.to(device), True
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True)
    return enc["input_ids"].to(device), False


def terminators(tok, model) -> List[int]:
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    cfg = getattr(model, "config", None)
    if cfg is not None and isinstance(getattr(cfg, "eos_token_id", None), int):
        ids.add(cfg.eos_token_id)
    return sorted(ids)


@torch.no_grad()
def generate(model, tok, prompt: str, device: str, max_new_tokens: int):
    ids, templated = build_inputs(tok, prompt, device)
    eos = terminators(tok, model)
    out = model.generate(
        input_ids=ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        eos_token_id=eos or None,
        pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else (eos[0] if eos else 0),
    )
    new = out[0, ids.shape[1]:]
    text = tok.decode(new, skip_special_tokens=True)
    stopped = bool(len(new) < max_new_tokens)
    return text, stopped, templated


def sinhala_ratio(text: str) -> float:
    letters = [c for c in text if not c.isspace() and not c.isdigit()
               and c not in ".,!?;:'\"()-–—"]
    if not letters:
        return 0.0
    return sum(1 for c in letters if SINHALA_BLOCK.match(c)) / len(letters)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    paths: List[str] = []
    for pat in list(args.models) + list(args.reference):
        hits = sorted(glob.glob(pat))
        paths.extend(hits if hits else [pat])
    # keep order, drop duplicates
    seen = set()
    paths = [p for p in paths if not (p in seen or seen.add(p))]

    si_texts = read_lines(args.sinhala_file, SINHALA_TEXT)
    en_texts = read_lines(args.english_file, ENGLISH_TEXT)
    log(f"{len(paths)} models | {len(si_texts)} Sinhala, {len(en_texts)} English sentences")
    log(f"device={args.device} dtype={args.dtype}")

    dtype = DTYPES[args.dtype]
    rows: List[Dict] = []

    for path in paths:
        name = os.path.basename(path.rstrip("/")) or path
        log("=" * 70)
        log(f"{name}")
        try:
            tok = AutoTokenizer.from_pretrained(path, cache_dir=args.cache_dir)
            model = AutoModelForCausalLM.from_pretrained(
                path, torch_dtype=dtype, low_cpu_mem_usage=True,
                cache_dir=args.cache_dir,
            ).to(args.device).eval()
        except Exception as e:
            log(f"  LOAD FAILED: {e}")
            rows.append({"model": name, "error": str(e)[:200]})
            continue

        row: Dict = {"model": name, "vocab": len(tok)}

        si = bits_per_char(model, tok, si_texts, args.device)
        en = bits_per_char(model, tok, en_texts, args.device)
        row["bpc_si"] = round(si["bpc"], 4)
        row["bpc_en"] = round(en["bpc"], 4)
        row["fert_si"] = round(si["fertility"], 4)
        row["nonfinite"] = int(si["nonfinite"] or en["nonfinite"])
        log(f"  bpc_si={row['bpc_si']}  bpc_en={row['bpc_en']}  "
            f"fert_si={row['fert_si']}  vocab={row['vocab']}")

        if not args.no_generate:
            for tag, prompt in (("si", SI_PROMPT), ("en", EN_PROMPT)):
                try:
                    text, stopped, templated = generate(
                        model, tok, prompt, args.device, args.max_new_tokens)
                except Exception as e:
                    text, stopped, templated = f"<generation failed: {e}>", False, False
                row[f"gen_{tag}"] = text.replace("\n", " ")[:args.gen_chars]
                row[f"stops_{tag}"] = int(stopped)
                if tag == "si":
                    row["si_script"] = round(sinhala_ratio(text), 3)
                    row["templated"] = int(templated)
                log(f"  [{tag}] stops={int(stopped)} "
                    f"{'si_script=' + str(row.get('si_script')) + ' ' if tag == 'si' else ''}"
                    f"| {row[f'gen_{tag}'][:args.gen_chars]}")

        rows.append(row)

        del model
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()

    # ---- summary -----------------------------------------------------------
    fields: List[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    log("=" * 70)
    log(f"Wrote {args.out}")

    ranked = sorted((r for r in rows if isinstance(r.get("bpc_si"), float)
                     and math.isfinite(r["bpc_si"])),
                    key=lambda r: r["bpc_si"])
    log("")
    log(f"{'model':<34}{'bpc_si':>9}{'bpc_en':>9}{'fert_si':>9}{'si%':>7}{'stop':>6}")
    log("-" * 74)
    for r in ranked:
        log(f"{r['model'][:33]:<34}{r['bpc_si']:>9.3f}{r['bpc_en']:>9.3f}"
            f"{r['fert_si']:>9.3f}{r.get('si_script', float('nan')):>7.2f}"
            f"{r.get('stops_si', ''):>6}")
    log("")
    log("Lower bpc_si is better. Compare bpc_en against the Instruct reference")
    log("to see how much English was lost. A model with good bpc_si but")
    log("si_script near 0 answers in the wrong language and is not usable.")


if __name__ == "__main__":
    sys.exit(main())