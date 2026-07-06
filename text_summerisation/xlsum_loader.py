"""Direct loader for csebuetnlp/xlsum Sinhala.

Why this exists: csebuetnlp/xlsum is a *script-based* dataset (xlsum.py). On
datasets>=4.0 script loading is gone, and because the repo needs remote code its
Hugging Face viewer is disabled -> the auto-converted `refs/convert/parquet`
branch was never produced. So both `revision="refs/convert/parquet"` and the
`trust_remote_code=True` fallback fail. We instead pull the exact same raw
archive the original loader used and read the JSONL directly, preserving
provenance, splits, and the id/url/title/summary/text schema.
"""
import bz2, json, os, tarfile
import pandas as pd
from huggingface_hub import hf_hub_download

_REPO = "csebuetnlp/xlsum"
_ARCHIVE = "data/{lang}_XLSum_v2.0.tar.bz2"     # matches xlsum.py _URL for VERSION 2.0.0
_COLS = ["id", "url", "title", "summary", "text"]


def _read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows.append({c: d.get(c) for c in _COLS})
    return pd.DataFrame(rows, columns=_COLS)


def load_xlsum_sinhala(lang="sinhala", archive_path=None):
    """Returns (train_df, val_df, test_df) with columns id/url/title/summary/text."""
    if archive_path is None:
        archive_path = hf_hub_download(
            repo_id=_REPO, repo_type="dataset",
            filename=_ARCHIVE.format(lang=lang))

    # Extract once, next to the cached archive, and reuse on later runs.
    extract_dir = archive_path + ".extracted"
    if not os.path.isdir(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
        with tarfile.open(archive_path, "r:bz2") as tar:
            tar.extractall(extract_dir)   # files land as {lang}_{train,val,test}.jsonl

    # Files may be at the archive root or nested; find them robustly.
    def find(split):
        target = f"{lang}_{split}.jsonl"
        for root, _, names in os.walk(extract_dir):
            if target in names:
                return os.path.join(root, target)
        raise FileNotFoundError(f"{target} not found under {extract_dir}")

    return _read_jsonl(find("train")), _read_jsonl(find("val")), _read_jsonl(find("test"))