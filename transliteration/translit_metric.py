# coding=utf-8
"""
Transliteration metrics for Romanised -> Sinhala (shared across model families).

    CER   character error rate (Levenshtein over Unicode code points / ref length) x100, lower is better
    WER   word error rate (Levenshtein over whitespace tokens / ref length) x100, lower is better
    chrF  character n-gram F-score (sacrebleu defaults: n=6, beta=2), higher is better

All strings are NFC-normalised first (never NFD: it would split Sinhala vowel
signs and hal kirima into separate code points and inflate CER). Corpus-level
CER/WER = total edits / total reference length (the standard definition);
per-instance scores are also returned for the scored predictions CSV.

    pip install sacrebleu
"""
import unicodedata
from typing import List

import numpy as np
from sacrebleu.metrics import CHRF

METRICS = ['cer', 'wer', 'chrf']


def _nfc(text):
    return unicodedata.normalize("NFC", str(text)).strip()


def _edit_distance(ref, hyp):
    """Levenshtein distance between two sequences (strings or token lists)."""
    if len(ref) < len(hyp):
        ref, hyp = hyp, ref
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        curr = [i]
        for j, h in enumerate(hyp, 1):
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = curr
    return prev[-1]


def score_corpus(references: List[str], predictions: List[str]) -> dict:
    chrf = CHRF()
    per = {m: [] for m in METRICS}
    char_edits = char_total = word_edits = word_total = 0

    refs_nfc, preds_nfc = [], []
    for ref, pred in zip(references, predictions):
        ref, pred = _nfc(ref), _nfc(pred)
        refs_nfc.append(ref)
        preds_nfc.append(pred)

        ce = _edit_distance(ref, pred)
        we = _edit_distance(ref.split(), pred.split())
        char_edits += ce
        char_total += len(ref)
        word_edits += we
        word_total += len(ref.split())

        per['cer'].append(100 * ce / max(len(ref), 1))
        per['wer'].append(100 * we / max(len(ref.split()), 1))
        per['chrf'].append(chrf.sentence_score(pred, [ref]).score)

    corpus = {
        'cer': 100 * char_edits / max(char_total, 1),
        'wer': 100 * word_edits / max(word_total, 1),
        'chrf': chrf.corpus_score(preds_nfc, [refs_nfc]).score,
    }

    summary = {}
    for m in METRICS:
        arr = np.array(per[m]) if per[m] else np.array([0.0])
        summary[m] = {
            'corpus': float(corpus[m]),
            'mean': float(arr.mean()),
            'median': float(np.median(arr)),
            'std': float(arr.std()),
            'min': float(arr.min()),
            'max': float(arr.max()),
            'scores': per[m],
        }
    return summary