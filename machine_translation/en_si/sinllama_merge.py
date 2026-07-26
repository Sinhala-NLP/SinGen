import argparse
import os
import re
import random
import tarfile
import urllib.request
from typing import List

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Optional: corpus-level reference numbers alongside the in-house sentence BLEU.
try:
    import sacrebleu
except ImportError:
    sacrebleu = None

FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
set_seed(777)


# --------------------------------------------------------------------------- #
# What differs from the Qwen MT script
# --------------------------------------------------------------------------- #
#   * No multimodal routing: merged SinLlama checkpoints are text-only
#     LlamaForCausalLM, so AutoTokenizer + AutoModelForCausalLM unconditionally
#     and no typed-content wrapping for processor templates.
#   * <|eot_id|> added to the generation stop set.
#   * Llama special-token cleanup in post-processing.
#   * Marker-miss counter (the extraction fallback itself is left byte-identical
#     to the Qwen script so BLEU stays comparable on format misses).
# Prompt wording, few-shot selection (seed=42, from dev), and the BLEU
# implementation are unchanged. Zero/few-shot ONLY -- no fine-tuning.


# --------------------------------------------------------------------------- #
# Data loading (identical to the Gemma/Qwen MT scripts)
# --------------------------------------------------------------------------- #
def _ensure_flores200(cache_dir=None):
    """Download + extract the FLORES-200 archive once; return the path to the
    extracted `flores200_dataset` dir. Replaces the removed HF loading script
    (datasets>=4.0 no longer runs flores200.py)."""
    cache_dir = cache_dir or os.environ.get(
        "FLORES200_CACHE",
        os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache")), "flores200"),
    )
    os.makedirs(cache_dir, exist_ok=True)
    extracted = os.path.join(cache_dir, "flores200_dataset")

    sentinel = os.path.join(extracted, "devtest", "sin_Sinh.devtest")
    if not os.path.exists(sentinel):
        tarball = os.path.join(cache_dir, "flores200_dataset.tar.gz")
        if not os.path.exists(tarball):
            print(f"Downloading FLORES-200 from {FLORES_URL} ...")
            urllib.request.urlretrieve(FLORES_URL, tarball)
        print(f"Extracting {tarball} ...")
        with tarfile.open(tarball, "r:gz") as tar:
            tar.extractall(cache_dir, filter="data")  # filter= is py>=3.12
    return extracted


def _read_lines(path):
    # Explicit utf-8 is important for Sinhala.
    with open(path, "r", encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def download_and_load_flores_en_si():
    """Returns dev (997) and devtest (1012) DataFrames with columns
    'english' and 'sinhala'."""
    print("Loading FLORES-200 dataset for English-Sinhala...")
    root = _ensure_flores200()

    splits = {}
    for split_name in ["dev", "devtest"]:
        eng = _read_lines(os.path.join(root, split_name, f"eng_Latn.{split_name}"))
        sin = _read_lines(os.path.join(root, split_name, f"sin_Sinh.{split_name}"))
        assert len(eng) == len(sin), (
            f"{split_name}: eng/sin line-count mismatch ({len(eng)} vs {len(sin)})"
        )
        df = pd.DataFrame({"english": eng, "sinhala": sin})
        splits[split_name] = df
        print(f"{split_name.capitalize()} split size: {len(df)}")

    return splits.get("dev"), splits.get("devtest")


def get_few_shot_examples_for_instance(dev_df, instance_idx, num_examples=3, seed=None):
    """Random few-shot examples per test instance, drawn from dev only."""
    if seed is not None:
        random.seed(seed + instance_idx)

    available_indices = list(dev_df.index)
    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = dev_df.loc[idx]
        if pd.notna(row['english']) and pd.notna(row['sinhala']) and \
                str(row['english']).strip() and str(row['sinhala']).strip():
            few_shot_examples.append({
                'english': str(row['english']),
                'sinhala': str(row['sinhala'])
            })

    return few_shot_examples


# --------------------------------------------------------------------------- #
# Prompting (byte-identical wording to the Gemma/Qwen MT scripts)
# --------------------------------------------------------------------------- #
def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in English to Sinhala translation. Translate the following English sentence (E) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ ඉංග්‍රීසි සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත ඉංග්‍රීසි වාක්‍යය (E) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"E: {example['english']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} E: {row['english']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} E: {row['english']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nE: {row['english']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nE: {row['english']}"
        return [{"role": "user", "content": prompt}]

    else:
        return [{"role": "user", "content": f"{task_desc} {action_desc} E: {row['english']}"}]


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
# Llama-3 has no reasoning channel; the <think> strip is a harmless no-op kept
# for symmetry with the Qwen path. The Llama tag strip is the part that matters.
_QWEN_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r"</?think>", re.IGNORECASE)
_LLAMA_TAGS = re.compile(r"<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>", re.IGNORECASE)


def strip_thinking(text):
    if not isinstance(text, str):
        return ""
    text = _QWEN_THINK.sub("", text)
    text = _STRAY_THINK.sub("", text)
    text = _LLAMA_TAGS.sub("", text)
    return text.strip()


def extract_translation(response):
    """Return (prediction, matched_marker).

    Extraction and fallback behaviour are unchanged from the Qwen MT script --
    only the marker-matched flag is new, so it is reportable without shifting
    any BLEU number.
    """
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return "", False

    try:
        response = strip_thinking(response)
        matches = re.findall(r'Translation:\s*(.*?)(?:\n\n|\Z)', response, re.IGNORECASE | re.DOTALL)
        if matches:
            return matches[0].strip(), True

        if "translation:" in response.lower():
            parts = response.lower().split("translation:")
            if len(parts) > 1:
                return parts[1].strip(), True

        return response.strip(), False
    except Exception as e:
        print(f"Error extracting translation: {e}")
        return "", False


# --------------------------------------------------------------------------- #
# Generation (AutoTokenizer, batched, left-padded)
# --------------------------------------------------------------------------- #
def build_terminators(tok):
    ids = set()
    if tok.eos_token_id is not None:
        ids.add(tok.eos_token_id)
    for t in ("<|eot_id|>", "<|end_of_text|>"):
        i = tok.convert_tokens_to_ids(t)
        if isinstance(i, int) and i >= 0:
            ids.add(i)
    return sorted(ids)


def query(model, tok, messages_list: List[list], max_new_tokens=200,
          batch_size=8, do_sample=False):
    """Batched decoding with left padding so new tokens start at the same offset
    for every sequence in a batch."""
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_ids = build_terminators(tok)
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    assistant_outputs = []

    for start in tqdm(range(0, len(messages_list), batch_size), desc="Generating"):
        batch = messages_list[start:start + batch_size]

        inputs = tok.apply_chat_template(
            batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                pad_token_id=pad_id,
                eos_token_id=eos_ids or None,
                **sample_kwargs,
            )

        new_tokens = generated[:, input_len:]
        decoded = tok.batch_decode(new_tokens, skip_special_tokens=True)
        assistant_outputs.extend(strip_thinking(t) for t in decoded)

    return assistant_outputs


# --------------------------------------------------------------------------- #
# BLEU (whitespace tokenization -- required for Sinhala; unchanged)
# --------------------------------------------------------------------------- #
def tokenize(text):
    """Simple tokenization by splitting on whitespace."""
    if pd.isna(text) or text is None:
        return []
    return str(text).strip().split()


def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
    """Individual BLEU-1..4 plus overall BLEU for a single sentence pair."""
    ref_tokens = tokenize(reference)
    pred_tokens = tokenize(prediction)

    if not pred_tokens or not ref_tokens:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0, 'bleu_overall': 0.0}

    ref_len = len(ref_tokens)
    pred_len = len(pred_tokens)

    if pred_len > ref_len:
        bp = 1.0
    else:
        bp = np.exp(1 - ref_len / pred_len) if pred_len > 0 else 0.0

    precisions = []
    individual_bleu_scores = {}

    for n in range(1, max_n + 1):
        ref_ngrams = {}
        pred_ngrams = {}

        for i in range(len(ref_tokens) - n + 1):
            ngram = tuple(ref_tokens[i:i + n])
            ref_ngrams[ngram] = ref_ngrams.get(ngram, 0) + 1

        for i in range(len(pred_tokens) - n + 1):
            ngram = tuple(pred_tokens[i:i + n])
            pred_ngrams[ngram] = pred_ngrams.get(ngram, 0) + 1

        clipped_count = 0
        total_count = sum(pred_ngrams.values())

        for ngram, count in pred_ngrams.items():
            if ngram in ref_ngrams:
                clipped_count += min(count, ref_ngrams[ngram])

        precision = clipped_count / total_count if total_count > 0 else 0.0
        precisions.append(precision)
        individual_bleu_scores[f'bleu_{n}'] = (bp * precision * 100) if precision > 0 else 0.0

    if min(precisions) > 0:
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions]))
    else:
        geo_mean = 0.0

    individual_bleu_scores['bleu_overall'] = bp * geo_mean * 100
    return individual_bleu_scores


def evaluate_bleu_scores(df):
    """Sentence-level BLEU averaged over the test set (the benchmark's metric)."""
    print("\nCalculating BLEU scores...")

    bleu_1_scores, bleu_2_scores, bleu_3_scores, bleu_4_scores, bleu_overall_scores = [], [], [], [], []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing BLEU"):
        reference = row['sinhala']
        prediction = row['preds']

        if pd.isna(reference) or pd.isna(prediction) or not str(reference).strip() or not str(prediction).strip():
            bleu_1_scores.append(0.0)
            bleu_2_scores.append(0.0)
            bleu_3_scores.append(0.0)
            bleu_4_scores.append(0.0)
            bleu_overall_scores.append(0.0)
            continue

        scores = calculate_bleu_score_individual(str(reference), str(prediction))
        bleu_1_scores.append(scores['bleu_1'])
        bleu_2_scores.append(scores['bleu_2'])
        bleu_3_scores.append(scores['bleu_3'])
        bleu_4_scores.append(scores['bleu_4'])
        bleu_overall_scores.append(scores['bleu_overall'])

    df['bleu_1'] = bleu_1_scores
    df['bleu_2'] = bleu_2_scores
    df['bleu_3'] = bleu_3_scores
    df['bleu_4'] = bleu_4_scores
    df['bleu_overall'] = bleu_overall_scores

    print("\n" + "=" * 70)
    print("BLEU Score Evaluation Results (sentence-level, averaged):")
    print("=" * 70)

    for score_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
        scores = df[score_type].tolist()
        print(f"\n{score_type.upper().replace('_', '-')}:")
        print(f"  Mean:   {np.mean(scores):.4f}")
        print(f"  Median: {np.median(scores):.4f}")
        print(f"  Std:    {np.std(scores):.4f}")
        print(f"  Min:    {np.min(scores):.4f}")
        print(f"  Max:    {np.max(scores):.4f}")

    print("=" * 70)

    def stats(scores):
        return {
            'mean': np.mean(scores), 'median': np.median(scores), 'std': np.std(scores),
            'min': np.min(scores), 'max': np.max(scores), 'scores': scores
        }

    return {
        'bleu_1': stats(bleu_1_scores),
        'bleu_2': stats(bleu_2_scores),
        'bleu_3': stats(bleu_3_scores),
        'bleu_4': stats(bleu_4_scores),
        'bleu_overall': stats(bleu_overall_scores),
    }


def corpus_reference_bleu(refs, hyps):
    """Corpus-level sacreBLEU + spBLEU, recorded alongside the in-house metric.

    Not a replacement: the benchmark tables use the averaged sentence BLEU
    above. These are the numbers reviewers familiar with FLORES will expect to
    see, and spBLEU (flores200 SPM) is the standard for Sinhala since
    whitespace BLEU understates morphologically rich targets.
    """
    out = {}
    if sacrebleu is None:
        print("sacrebleu not installed -- skipping corpus-level reference BLEU.")
        return out
    try:
        out['sacrebleu_13a'] = sacrebleu.corpus_bleu(hyps, [refs]).score
    except Exception as e:
        print(f"sacreBLEU (13a) failed: {e}")
    try:
        out['spbleu_flores200'] = sacrebleu.corpus_bleu(hyps, [refs], tokenize="flores200").score
    except Exception as e:
        print(f"spBLEU (flores200 tokenizer) unavailable: {e}")
    for k, v in out.items():
        print(f"{k}: {v:.4f}")
    return out


# --------------------------------------------------------------------------- #
# Prediction driver
# --------------------------------------------------------------------------- #
def predict(model, tok, dev_df, devtest_df, max_new_tokens=200, batch_size=8, do_sample=False):
    print(f"Dev set size: {len(dev_df)}")
    print(f"Devtest set size: {len(devtest_df)}")

    df = devtest_df.copy()

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        chat_messages = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            few_shot_examples = get_few_shot_examples_for_instance(
                dev_df, instance_idx=idx, num_examples=3, seed=42
            )
            chat_messages.append(format_chat(row, few_shot_examples))
        df['chat'] = chat_messages
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    print("Generating translations...")
    responses = query(model, tok, df['chat'].tolist(),
                      max_new_tokens=max_new_tokens, batch_size=batch_size, do_sample=do_sample)
    df['responses'] = responses

    print("Extracting translations...")
    preds, matched = zip(*[extract_translation(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].astype(str).str.len() == 0).sum())
    print(f"\nFormat misses (no 'Translation:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    print("Evaluating translations with BLEU score...")
    bleu_results = evaluate_bleu_scores(df)
    corpus_bleu = corpus_reference_bleu(
        [str(x) for x in df['sinhala'].tolist()],
        [str(x) for x in df['preds'].tolist()],
    )

    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_bleu.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with BLEU scores saved to: {results_file}")

    summary_file = os.path.join(OUTPUT_FOLDER, "bleu_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {MODEL_ID}\n")
        f.write(f"Run tag: {RUN_TAG}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: FLORES-200 English-Sinhala (devtest split)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance from dev set)\n")
        f.write("=" * 70 + "\n\n")

        f.write("Sentence-level BLEU, whitespace-tokenized, averaged over instances\n\n")
        for score_type in ['bleu_1', 'bleu_2', 'bleu_3', 'bleu_4', 'bleu_overall']:
            f.write(f"{score_type.upper().replace('_', '-')}:\n")
            f.write(f"  Mean:   {bleu_results[score_type]['mean']:.4f}\n")
            f.write(f"  Median: {bleu_results[score_type]['median']:.4f}\n")
            f.write(f"  Std:    {bleu_results[score_type]['std']:.4f}\n")
            f.write(f"  Min:    {bleu_results[score_type]['min']:.4f}\n")
            f.write(f"  Max:    {bleu_results[score_type]['max']:.4f}\n\n")

        if corpus_bleu:
            f.write("-" * 70 + "\nCorpus-level reference numbers (sacreBLEU)\n")
            for k, v in corpus_bleu.items():
                f.write(f"  {k}: {v:.4f}\n")

    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), bleu_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, required=True,
                        help='Path to a merged checkpoint directory.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Output folder name. Defaults to basename of --model_id.')
    parser.add_argument('--output_root', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base dir for outputs/. Defaults to the script's own directory.")
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--do_sample', action='store_true',
                        help='sampling (t=0.7,p=0.8,k=20) instead of greedy')
    args = parser.parse_args()

    MODEL_ID = args.model_id
    QUERY_TYPE = args.query_type
    RUN_TAG = args.run_tag or os.path.basename(MODEL_ID.rstrip('/'))

    print(f"Model: {MODEL_ID}")
    print(f"Run tag: {RUN_TAG}")
    print(f"Query type: {QUERY_TYPE}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max new tokens: {args.max_new_tokens}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{MODEL_ID} has no chat_template; expected a merged checkpoint "
                         f"produced with --copy_chat_template.")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    model.eval()
    print(f"Vocab size: {len(tok)}  Embedding rows: {model.get_input_embeddings().weight.shape[0]}")

    dev_df, devtest_df = download_and_load_flores_en_si()
    if dev_df is None or devtest_df is None:
        print("Error: Could not load required dataset splits")
        exit(1)

    OUTPUT_FOLDER = os.path.join(args.output_root, "outputs",
                                 "english_sinhala_translation_zeroshot", RUN_TAG, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    predictions, bleu_results = predict(
        model, tok, dev_df, devtest_df,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, do_sample=args.do_sample,
    )