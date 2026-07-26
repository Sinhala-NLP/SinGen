import argparse
import os
import re
import random
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

set_seed(777)


# --------------------------------------------------------------------------- #
# What differs from the Llama-3 Ta->Si prompting script
# --------------------------------------------------------------------------- #
# Little on the model side -- merged SinLlama checkpoints are text-only
# LlamaForCausalLM, which that script already targeted. Added here:
#   * --run_tag / --output_root so three merge variants write side by side.
#   * --tsv_file (was a hardcoded relative path; --output_root means the script
#     no longer necessarily runs from the data directory).
#   * chat_template guard (merged dirs need --copy_chat_template).
#   * embedding-row check at load: catches a merge that dropped the extended
#     Sinhala rows before an hour of decoding does.
#   * prompt-length report. NOT truncation: SinLlama's vocab extension is
#     Sinhala-only, so TAMIL SOURCE runs through the unextended Llama-3 path at
#     high fertility -- prompts here are longer than the En->Si runs on
#     comparable sentences. Truncating a source sentence would silently corrupt
#     the translation, so overruns warn instead.
#   * marker-miss counter (extraction fallback left byte-identical so BLEU is
#     unchanged on format misses).
#   * corpus sacreBLEU/spBLEU as secondary numbers, matching the En->Si merged
#     script so both directions report the same set.
# Prompt wording, the tail-1000 split, few-shot selection (seed=42), and the
# BLEU implementation are unchanged. Zero/few-shot ONLY -- no fine-tuning.


# --------------------------------------------------------------------------- #
# Few-shot selection (identical task logic to the Gemma/Qwen Ta->Si scripts)
# --------------------------------------------------------------------------- #
def get_few_shot_examples_for_instance(full_df, test_df, instance_idx, num_examples=3, seed=None):
    """Random few-shot examples per test instance, drawn from the non-test pool."""
    test_indices = set(test_df.index)
    available_indices = [i for i in full_df.index if i not in test_indices]

    if seed is not None:
        random.seed(seed + instance_idx)

    few_shot_indices = random.sample(available_indices, min(num_examples, len(available_indices)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = full_df.loc[idx]
        if pd.notna(row['Tamil']) and pd.notna(row['Sinhala']) and \
                str(row['Tamil']).strip() and str(row['Sinhala']).strip():
            few_shot_examples.append({
                'tamil': str(row['Tamil']),
                'sinhala': str(row['Sinhala'])
            })

    return few_shot_examples


# --------------------------------------------------------------------------- #
# Prompting (IDENTICAL wording to the Gemma/Qwen Ta->Si scripts for BLEU
# comparability — do not reword without updating every model script)
# --------------------------------------------------------------------------- #
def format_chat(row, few_shot_examples=None):
    task_desc = "You are an expert translator specializing in Tamil to Sinhala translation. Translate the following Tamil sentence (T) into Sinhala accurately while preserving the meaning and context."
    action_desc = "Return only the Sinhala translation following the prefix 'Translation:' without any other text or explanations."

    task_desc_si = "ඔබ දෙමළ සිට සිංහල භාෂා පරිවර්තනයේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත දෙමළ වාක්‍යය (T) අර්ථය සහ සන්දර්භය ආරක්ෂා කරමින් නිවැරදිව සිංහලයට පරිවර්තනය කරන්න."
    action_desc_si = "'Translation:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිංහල පරිවර්තනය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\n"
            examples_str += f"T: {example['tamil']}\n"
            examples_str += f"Translation: {example['sinhala']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} T: {row['Tamil']}"

    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} T: {row['Tamil']}"

    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow translate this sentence:\nT: {row['Tamil']}"

    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය පරිවර්තනය කරන්න:\nT: {row['Tamil']}"

    else:
        content = f"{task_desc} {action_desc} T: {row['Tamil']}"

    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Prompt length measurement (report only — never truncate an MT source)
# --------------------------------------------------------------------------- #
# NOTE: apply_chat_template(tokenize=True) returns a BatchEncoding on some
# transformers versions rather than a flat id list, so length is measured via
# tokenize=False + a separate tokenizer call.
def n_prompt_tokens(tok, messages) -> int:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


# --------------------------------------------------------------------------- #
# Output post-processing (Llama has no thinking mode)
# --------------------------------------------------------------------------- #
_STRAY_TAGS = re.compile(r'<\|[^|>]*\|>')


def extract_translation(response):
    """Return (prediction, matched_marker).

    Extraction and fallback behaviour are unchanged from the prompting script --
    only the marker-matched flag is new, so it is reportable without shifting
    any BLEU number.
    """
    if not isinstance(response, str):
        print(f"Non-string response: {response}")
        return "", False

    try:
        response = _STRAY_TAGS.sub('', response)
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
# Generation (text-only causal LM, Llama-3 <|eot_id|> terminator)
# --------------------------------------------------------------------------- #
def build_terminators(tok):
    terminators = []
    if tok.eos_token_id is not None:
        terminators.append(tok.eos_token_id)
    # Llama 3 ends a turn with <|eot_id|>, which differs from the default EOS.
    eot = tok.convert_tokens_to_ids("<|eot_id|>")
    if eot is not None and eot != tok.unk_token_id and eot not in terminators:
        terminators.append(eot)
    return terminators or None


def generate(model, tokenizer, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    terminators = build_terminators(tokenizer)
    sample_kwargs = dict(temperature=0.6, top_p=0.9) if do_sample else {}  # Llama defaults

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating translations"):
        batch = list_of_messages[start:start + batch_size]
        inputs = tokenizer.apply_chat_template(
            batch, add_generation_prompt=True, tokenize=True,
            padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, eos_token_id=terminators,
                                 pad_token_id=tokenizer.pad_token_id, **sample_kwargs)
        outputs.extend(tokenizer.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# BLEU (whitespace tokenization — required for Sinhala; identical to the
# Gemma/Qwen Ta->Si scripts so numbers are directly comparable across families)
# --------------------------------------------------------------------------- #
def tokenize(text):
    """Simple tokenization by splitting on whitespace."""
    if pd.isna(text) or text is None:
        return []
    return str(text).strip().split()


def calculate_bleu_score_individual(reference: str, prediction: str, max_n: int = 4):
    """Calculate individual BLEU scores (BLEU-1..4) and overall BLEU."""
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
        reference = row['Sinhala']
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
    above. spBLEU (flores200 SPM) is the standard for Sinhala targets, since
    whitespace BLEU understates morphologically rich languages.
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
# Prediction driver (TSV loading + tail-1000 split — identical to Gemma/Qwen)
# --------------------------------------------------------------------------- #
def predict(model, tokenizer, model_id, tsv_file_path, budget,
            max_new_tokens=200, batch_size=8, do_sample=False):
    print(f"Loading data from {tsv_file_path}...")
    if not os.path.exists(tsv_file_path):
        raise SystemExit(f"TSV not found: {tsv_file_path} (pass --tsv_file)")
    full_df = pd.read_csv(tsv_file_path, sep='\t', encoding='utf-8')

    print(f"Total dataset size: {len(full_df)}")
    print(f"Columns: {full_df.columns.tolist()}")
    for col in ('Tamil', 'Sinhala'):
        if col not in full_df.columns:
            raise SystemExit(f"TSV is missing required column '{col}'")

    # Use last 1000 samples for testing (identical split to the Gemma/Qwen
    # scripts so all families evaluate on the same test set).
    test_size = min(1000, len(full_df))
    df = full_df.tail(test_size).copy()
    print(f"Using last {test_size} samples for testing")

    # Pool for few-shot examples (everything except the test tail)
    rest_of_instances = full_df.head(len(full_df) - test_size)

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Available for few-shot examples: {len(rest_of_instances)}")

        chat_messages = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            few_shot_examples = get_few_shot_examples_for_instance(
                rest_of_instances, df, instance_idx=idx, num_examples=3, seed=42
            )
            chat_messages.append(format_chat(row, few_shot_examples))
        df['chat'] = chat_messages
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    # Measure, do not truncate. Tamil source is outside SinLlama's Sinhala-only
    # vocab extension, so fertility here is worth recording per merge variant.
    tok_lens = [n_prompt_tokens(tokenizer, m) for m in tqdm(df['chat'].tolist(), desc="Measuring prompts")]
    df['prompt_tokens'] = tok_lens
    n_over = int(sum(1 for n in tok_lens if n > budget))
    print(f"\nPrompt length (tokens): max={max(tok_lens)} mean={sum(tok_lens) / len(tok_lens):.0f}")
    if n_over:
        print(f"WARNING: {n_over}/{len(df)} prompts exceed the {budget}-token budget. "
              f"Sources are NOT truncated (that would corrupt the translation) -- those "
              f"instances may degrade. Consider --max_new_tokens lower, or drop to 2 few-shot examples "
              f"for ALL model families rather than just this one.")
    else:
        print(f"All prompts within the {budget}-token budget.")

    print("Generating translations...")
    responses = generate(model, tokenizer, df['chat'].tolist(),
                         batch_size=batch_size, max_new_tokens=max_new_tokens, do_sample=do_sample)
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
        [str(x) for x in df['Sinhala'].tolist()],
        [str(x) for x in df['preds'].tolist()],
    )

    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_bleu.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with BLEU scores saved to: {results_file}")

    summary_file = os.path.join(OUTPUT_FOLDER, "bleu_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("BLEU Score Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Run tag: {RUN_TAG}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write("Dataset: TamSiPara Tamil-Sinhala (last 1000 rows as test)\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        f.write(f"Max New Tokens: {max_new_tokens}\n")
        f.write(f"Batch Size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.6,p=0.9)' if do_sample else 'greedy'}\n")
        f.write(f"Prompt tokens: max={max(tok_lens)} mean={sum(tok_lens) / len(tok_lens):.0f}  "
                f"over {budget}-token budget: {n_over}/{len(df)} (not truncated)\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty predictions: {n_empty}/{len(df)}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance)\n")
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
    parser.add_argument('--tsv_file', type=str, default='ta_si.tsv',
                        help='Path to the Tamil-Sinhala TSV (columns: Tamil, Sinhala).')
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--max_prompt_tokens', type=int, default=0,
                        help='0 = derive from model config (max_position_embeddings - max_new_tokens - 64)')
    parser.add_argument('--do_sample', action='store_true',
                        help='Llama sampling (t=0.6,p=0.9) instead of greedy. '
                             'Leave off for BLEU reproducibility.')
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
    print(f"TSV file: {args.tsv_file}")

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"{MODEL_ID} has no chat_template; expected a merged checkpoint "
                         f"produced with --copy_chat_template.")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype="auto", device_map="auto")
    model.eval()
    print(f"Vocab size: {len(tokenizer)}  "
          f"Embedding rows: {model.get_input_embeddings().weight.shape[0]}")

    ctx = int(getattr(model.config, "max_position_embeddings", 8192) or 8192)
    budget = args.max_prompt_tokens or max(512, ctx - args.max_new_tokens - 64)
    print(f"Model context: {ctx}  Prompt budget: {budget}")

    if hasattr(model, 'hf_device_map'):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    OUTPUT_FOLDER = os.path.join(args.output_root, "outputs",
                                 "tamil_sinhala_translation_zeroshot", RUN_TAG, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    predictions, bleu_results = predict(
        model, tokenizer, MODEL_ID, args.tsv_file, budget,
        max_new_tokens=args.max_new_tokens, batch_size=args.batch_size, do_sample=args.do_sample,
    )