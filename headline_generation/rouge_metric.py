# coding=utf-8
"""
Sinhala-safe ROUGE (ROUGE-1, ROUGE-2, ROUGE-L), F-measure, reported x100.

Uses Google's `rouge_score` library (the field standard) but replaces its
default tokenizer, which does `re.sub(r"[^a-z0-9]+", " ", text)` and therefore
DELETES all Sinhala characters -> every score becomes 0.0. We plug in a
whitespace tokenizer that preserves Sinhala graphemes (vowel signs, conjuncts,
ZWJ) intact. Stemming is disabled (it is English-only).

    pip install rouge_score
"""
from typing import List
import numpy as np
from rouge_score import rouge_scorer

ROUGE_TYPES = ['rouge1', 'rouge2', 'rougeL']


class _WhitespaceTokenizer:
    """Script-agnostic tokenizer: split on whitespace, keep tokens verbatim."""
    def tokenize(self, text):
        return str(text).split()


def build_scorer():
    return rouge_scorer.RougeScorer(ROUGE_TYPES, use_stemmer=False,
                                    tokenizer=_WhitespaceTokenizer())


def score_corpus(references: List[str], predictions: List[str]) -> dict:
    """Single reference per instance (NSINA has one gold headline). Returns
    per-type dicts of mean/std/median/min/max over F-measures (x100)."""
    scorer = build_scorer()
    per_type = {t: [] for t in ROUGE_TYPES}
    for ref, pred in zip(references, predictions):
        scores = scorer.score(str(ref), str(pred))
        for t in ROUGE_TYPES:
            per_type[t].append(scores[t].fmeasure * 100)

    summary = {}
    for t in ROUGE_TYPES:
        arr = np.array(per_type[t]) if per_type[t] else np.array([0.0])
        summary[t] = {
            'mean': float(arr.mean()),
            'std': float(arr.std()),
            'median': float(np.median(arr)),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'scores': per_type[t],
        }
    return summary