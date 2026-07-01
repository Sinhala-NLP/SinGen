# coding=utf-8
"""
Canonical SARI (Xu et al., TACL 2016), n=4, SARI = (F1_add + F1_keep + P_del)/3.

Faithful port of tensor2tensor's sari_hook.py (the implementation HuggingFace
`evaluate` also adapts): 0/0 = 1 convention, deletion precision-only (beta=0),
targets weighted fractionally by the number of references containing each n-gram.

Operates on WHITESPACE-tokenized text, so it is script-agnostic and safe for
Sinhala. IMPORTANT: do NOT feed it text tokenized with `\\w+`, which strips
Sinhala vowel signs and splits conjuncts. Pass raw strings; splitting happens here.
"""
import collections

BETA_FOR_DELETION = 0  # paper uses precision only for deletion


def _ngram_counter(tokens, n):
    ngram_list = [tuple(tokens[i:i + n]) for i in range(len(tokens) + 1 - n)]
    counts = collections.Counter()
    for ngram in set(ngram_list):   # presence, clamped to 1 (as in tensor2tensor)
        counts[ngram] = 1
    return counts


def _fbeta(true_positives, selected, relevant, beta=1):
    precision = 1
    if selected > 0:
        precision = true_positives / selected
    if beta == 0:
        return precision
    recall = 1
    if relevant > 0:
        recall = true_positives / relevant
    if precision > 0 and recall > 0:
        b2 = beta * beta
        return (1 + b2) * precision * recall / (b2 * precision + recall)
    return 0


def _addition(source, prediction, target):
    added = prediction - source
    tp = sum((added & target).values())
    selected = sum(added.values())
    relevant = sum((target - source).values())
    return _fbeta(tp, selected, relevant)


def _keep(source, prediction, wtarget):
    sp = source & prediction
    st = source & wtarget
    tp = sum((sp & st).values())
    return _fbeta(tp, sum(sp.values()), sum(st.values()))


def _deletion(source, prediction, wtarget, beta=0):
    snp = source - prediction
    snt = source - wtarget
    tp = sum((snp & snt).values())
    return _fbeta(tp, sum(snp.values()), sum(snt.values()), beta=beta)


def sari_sentence(source, prediction, references, max_gram_size=4):
    """source, prediction: str. references: list[str]. Returns (sari, keep, add, del) in [0,100]."""
    s = source.split()
    p = prediction.split()
    refs = [r.split() for r in references]
    keep_s, add_s, del_s = [], [], []
    for n in range(1, max_gram_size + 1):
        sc = _ngram_counter(s, n)
        pc = _ngram_counter(p, n)
        target_counts = collections.Counter()          # unweighted (count 1) for ADD
        weighted_target_counts = collections.Counter()  # fractional for KEEP/DEL
        num_nonempty = 0
        for r in refs:
            rc = _ngram_counter(r, n)
            if rc:
                weighted_target_counts += rc
                num_nonempty += 1
        if num_nonempty:
            for gram in list(weighted_target_counts.keys()):
                weighted_target_counts[gram] /= num_nonempty
                target_counts[gram] = 1
        keep_s.append(_keep(sc, pc, weighted_target_counts))
        del_s.append(_deletion(sc, pc, weighted_target_counts, BETA_FOR_DELETION))
        add_s.append(_addition(sc, pc, target_counts))
    ak = sum(keep_s) / max_gram_size
    aa = sum(add_s) / max_gram_size
    ad = sum(del_s) / max_gram_size
    sari = (ak + aa + ad) / 3.0
    return 100 * sari, 100 * ak, 100 * aa, 100 * ad


def corpus_sari(sources, predictions, references_list):
    scores = [sari_sentence(s, p, r)[0] for s, p, r in zip(sources, predictions, references_list)]
    return sum(scores) / len(scores) if scores else 0.0