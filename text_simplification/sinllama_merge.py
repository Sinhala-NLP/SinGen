import argparse
import os
import re
import random
from typing import List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Validated SARI (tensor2tensor / HF port). Reproduces the HF reference values
# exactly (100.0 for exact match; 26.953601953601954 for the documented multi-ref
# example). Operates on WHITESPACE tokens -> safe for Sinhala.
from sari_metric import sari_sentence

set_seed(777)


# --------------------------------------------------------------------------- #
# What differs from the Gemma prompting script
# --------------------------------------------------------------------------- #
# The merged SinLlama checkpoints are text-only LlamaForCausalLM with a
# Llama-3 chat template copied from Instruct during merging. So:
#   * AutoTokenizer, not AutoProcessor (that was the Gemma-4 multimodal path).
#   * apply_chat_template on the tokenizer directly.
#   * <|eot_id|> added to the generation stop set.
# Prompt wording and the four query types are byte-identical to the prompting
# eval, so SARI stays comparable across the whole benchmark. This is zero/few
# shot ONLY -- no fine-tuning.


# --------------------------------------------------------------------------- #
# Prompting (unchanged logic from the original script)
# --------------------------------------------------------------------------- #
def get_few_shot_examples_for_instance(full_df, test_df, instance_idx, num_examples=3, seed=None):
    test_indices = set(test_df.index)
    available_indices = [i for i in full_df.index if i not in test_indices]
    if seed is not None:
        random.seed(seed + instance_idx)

    simplification_columns = ['Simplification 1', 'Simplification 2', 'Simplification 3']
    few_shot_examples = []
    shuffled_indices = available_indices.copy()
    random.shuffle(shuffled_indices)

    for idx in shuffled_indices:
        if len(few_shot_examples) >= num_examples:
            break
        row = full_df.loc[idx]
        available_simplifications = [
            str(row[col]) for col in simplification_columns
            if col in row and pd.notna(row[col]) and str(row[col]).strip()
        ]
        if available_simplifications:
            few_shot_examples.append({
                'complex': row['Complex'],
                'simple': random.choice(available_simplifications),
            })
    return few_shot_examples


def format_chat(row, few_shot_examples=None):
    task_desc = ("Imagine you are an expert in Sinhala language. Please provide a simplified version of the "
                 "following Sinhala sentence (S) in Sinhala following these three steps; (1) Extract the main "
                 "idea of the sentence (2) Split long sentences into shorter ones and (3) Lexical reordering, "
                 "and replacing complex words with commonly used simple words.")
    action_desc = ("Return the simplified text only following the prefix 'Simplified text:' without any other "
                   "text or explanations.")

    task_desc_si = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න.පහත සිංහල වාක්‍යයට (S) සරල සිංහල වාක්‍යයක් ලබා දෙන්න. "
                    "ඒ සඳහා මෙම පියවර තුන අනුගමනය කරන්න: (1) වාක්‍යයේ ප්‍රධාන අදහස ලබා ගන්න (2) දිගු වාක්‍ය කෙටි වාක්‍ය කිහිපයකට බෙදන්න "
                    "(3) දුෂ්කර වචන සාමාන්‍යයෙන් භාවිතා වන පහසු වචන වලින් වෙනස් කරන්න සහ පද වින්‍යාසය සරල කරන්න.")
    action_desc_si = ("'Simplified text:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සරල කළ වාක්‍යය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ "
                      "විස්තරයක් එක් නොකරන්න.")

    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            examples_str += f"\nExample {i}:\nS: {example['complex']}\nSimplified text: {example['simple']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} S: {row['Complex']}"
    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} S: {row['Complex']}"
    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow simplify this sentence:\nS: {row['Complex']}"
    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ වාක්‍යය සරල කරන්න:\nS: {row['Complex']}"
    else:
        content = f"{task_desc} {action_desc} S: {row['Complex']}"

    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
# Llama-3 has no reasoning channel, but keep the Gemma strip (harmless no-op on
# Llama) plus a Llama special-token cleanup so nothing leaks into predictions.
_THINK_BLOCK = re.compile(r'<\|?\s*channel\s*\|?>\s*thought.*?<\|?\s*/?\s*channel\s*\|?>', re.DOTALL | re.IGNORECASE)
_STRAY_TAGS = re.compile(r'<\|?\s*/?\s*(?:channel|think)\s*\|?>', re.IGNORECASE)
_LLAMA_TAGS = re.compile(r'<\|(?:eot_id|end_of_text|start_header_id|end_header_id)\|>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _THINK_BLOCK.sub('', text)
    text = _STRAY_TAGS.sub('', text)
    text = _LLAMA_TAGS.sub('', text)
    return text.strip()


def extract_simplified(response: str):
    """Return (prediction, matched_marker). Never dumps the whole raw response."""
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)

    m = re.search(r'Simplified text:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip()
        cand = cand.split('\n\n')[0].strip()   # first block after the marker
        cand = cand.splitlines()[0].strip() if cand else cand  # first line only
        if cand:
            return cand, True

    # Conservative fallback: last non-empty line, and flag as a format miss so
    # the run can report how often the model failed to follow the format.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False


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


def generate(model, tok, list_of_messages: List[list], batch_size: int, max_new_tokens: int) -> List[str]:
    outputs = []
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = build_terminators(tok)

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating"):
        batch = list_of_messages[start:start + batch_size]

        inputs = tok.apply_chat_template(
            batch, add_generation_prompt=True, tokenize=True,
            padding=True, return_tensors="pt", return_dict=True,
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,                 # greedy -> deterministic
                pad_token_id=tok.pad_token_id,
                eos_token_id=eos_ids or None,
            )
        new_tokens = gen[:, input_len:]
        outputs.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate_sari(df) -> dict:
    ref_cols = [c for c in ['Simplification 1', 'Simplification 2', 'Simplification 3'] if c in df.columns]
    if not ref_cols:
        print("No 'Simplification X' columns found.")
        return None

    scores = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing SARI"):
        refs = [str(row[c]) for c in ref_cols if pd.notna(row[c]) and str(row[c]).strip()]
        if not refs:
            scores.append(0.0)
            continue
        scores.append(sari_sentence(str(row['Complex']), str(row['preds']), refs)[0])

    df['sari_score'] = scores
    return {
        'mean_sari': float(np.mean(scores)),
        'std_sari': float(np.std(scores)),
        'median_sari': float(np.median(scores)),
        'min_sari': float(np.min(scores)),
        'max_sari': float(np.max(scores)),
        'reference_columns': ref_cols,
    }


def predict(model_id, model, tok, batch_size, max_new_tokens):
    full = Dataset.to_pandas(load_dataset('NLPC-UOM/SiTSE', split='train'))
    # NOTE: this reproduces the original tail(200) split. Confirm this matches
    # the official SiTSE test split before reporting comparable numbers.
    df = full.tail(200).copy()
    rest = full.head(len(full) - 200)

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        chat_messages = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            fs = get_few_shot_examples_for_instance(rest, df, instance_idx=idx, num_examples=3, seed=42)
            chat_messages.append(format_chat(row, fs))
        df['chat'] = chat_messages
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    responses = generate(model, tok, df['chat'].tolist(), batch_size, max_new_tokens)
    df['responses'] = responses

    preds, matched = zip(*[extract_simplified(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Simplified text:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), index=False, encoding='utf-8')

    results = evaluate_sari(df)
    if results is None:
        return df['preds'].tolist(), None

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions_with_sari.csv"), index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("SARI (canonical, whitespace-tokenized, multi-reference)")
    print("=" * 60)
    print(f"Mean   : {results['mean_sari']:.4f}")
    print(f"Std    : {results['std_sari']:.4f}")
    print(f"Median : {results['median_sari']:.4f}")
    print(f"Min/Max: {results['min_sari']:.4f} / {results['max_sari']:.4f}")
    print("=" * 60)

    with open(os.path.join(OUTPUT_FOLDER, "sari_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nSamples: {len(df)}\n")
        f.write(f"Format misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Reference columns: {', '.join(results['reference_columns'])}\n")
        f.write(f"Mean SARI: {results['mean_sari']:.4f}\nStd: {results['std_sari']:.4f}\n")
        f.write(f"Median: {results['median_sari']:.4f}\nMin: {results['min_sari']:.4f}\nMax: {results['max_sari']:.4f}\n")

    return df['preds'].tolist(), results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--model_id', type=str, required=True,
                        help='Path to a merged checkpoint directory.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Output folder name. Defaults to basename of --model_id.')
    parser.add_argument('--output_root', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base dir for outputs/. Defaults to the script's own directory.")
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--max_new_tokens', type=int, default=256)
    args = parser.parse_args()

    QUERY_TYPE = args.query_type
    model_id = args.model_id
    tag = args.run_tag or os.path.basename(model_id.rstrip('/'))
    print(f"Model: {model_id}\nRun tag: {tag}\nQuery type: {QUERY_TYPE}")

    tok = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tok, "chat_template", None):
        raise SystemExit(f"{model_id} has no chat_template; expected a merged checkpoint "
                         f"produced with --copy_chat_template.")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()

    OUTPUT_FOLDER = os.path.join(args.output_root, "outputs",
                                 "text_simplification_zeroshot", tag, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    predict(model_id, model, tok, args.batch_size, args.max_new_tokens)