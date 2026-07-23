import argparse
import os
import re
import random
from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoProcessor, set_seed

# Qwen3.5/3.6 are multimodal; their cards load them with AutoModelForMultimodalLM.
# Fall back to AutoModelForImageTextToText on older transformers.
try:
    from transformers import AutoModelForMultimodalLM as _AutoModel
except ImportError:
    from transformers import AutoModelForImageTextToText as _AutoModel

# Sinhala-safe ROUGE (whitespace-tokenized). The stock rouge_score tokenizer
# strips non-ASCII and would zero out every Sinhala score.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

MAX_CONTENT_LENGTH = 2500  # characters, to bound prompt length


# --------------------------------------------------------------------------- #
# Data helpers (identical to the Gemma/Llama headline scripts)
# --------------------------------------------------------------------------- #
def analyze_and_trim_dataset(df, max_length=MAX_CONTENT_LENGTH):
    lengths = df['News Content'].apply(lambda x: len(str(x)))
    print(f"\nArticle length (chars): min={lengths.min()} max={lengths.max()} "
          f"mean={lengths.mean():.0f} median={lengths.median():.0f} "
          f">{max_length}: {(lengths > max_length).sum()}")
    df['News Content'] = df['News Content'].apply(
        lambda x: str(x)[:max_length] + "..." if len(str(x)) > max_length else str(x))
    return df


def log_gpu_memory():
    if torch.cuda.is_available():
        print("\n" + "=" * 60 + "\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            alloc = torch.cuda.memory_allocated(i) / 1024 ** 3
            resv = torch.cuda.memory_reserved(i) / 1024 ** 3
            print(f"  GPU {i}: allocated {alloc:.2f} GB | reserved {resv:.2f} GB")
        print("=" * 60 + "\n")


def get_few_shot_examples_for_instance(train_df, instance_idx, num_examples=3, seed=None):
    if seed is not None:
        random.seed(seed + instance_idx)
    idxs = random.sample(range(len(train_df)), min(num_examples, len(train_df)))
    examples = []
    for idx in idxs:
        row = train_df.iloc[idx]
        if (pd.notna(row['News Content']) and pd.notna(row['Headline'])
                and str(row['News Content']).strip() and str(row['Headline']).strip()):
            examples.append({'content': str(row['News Content']), 'headline': str(row['Headline'])})
    return examples


# --------------------------------------------------------------------------- #
# Prompting (identical wording to the Gemma/Llama headline scripts for ROUGE
# comparability — do not reword without updating every model script)
# --------------------------------------------------------------------------- #
def format_chat(row, few_shot_examples=None):
    task_desc = ("Imagine you are an expert in Sinhala language. Generate a concise and informative headline "
                 "for the following Sinhala news article. The headline should capture the main point of the "
                 "article in a brief, engaging manner.")
    action_desc = ("Return only the headline following the prefix 'Headline:' without any other text or "
                   "explanations.")
    task_desc_si = ("ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත සිංහල පුවත් ලිපිය සඳහා සංක්ෂිප්ත හා තොරතුරුදායක "
                    "සිරස්තලයක් ජනනය කරන්න. සිරස්තලය කෙටි, ආකර්ෂණීය ආකාරයෙන් ලිපියේ ප්‍රධාන කරුණ ග්‍රහණය කර ගත යුතුය.")
    action_desc_si = ("'Headline:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිරස්තලය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් "
                      "එක් නොකරන්න.")

    news_content = str(row['News Content'])
    if len(news_content) > MAX_CONTENT_LENGTH:
        news_content = news_content[:MAX_CONTENT_LENGTH] + "..."

    examples_str = ""
    if few_shot_examples:
        for i, ex in enumerate(few_shot_examples, 1):
            preview = ex['content'][:500] + "..." if len(ex['content']) > 500 else ex['content']
            examples_str += f"\nExample {i}:\nNews Content: {preview}\nHeadline: {ex['headline']}\n"

    if QUERY_TYPE == "zero-shot":
        content = f"{task_desc} {action_desc} News Content: {news_content}"
    elif QUERY_TYPE == "zero-shot-si":
        content = f"{task_desc_si} {action_desc_si} News Content: {news_content}"
    elif QUERY_TYPE == "few-shot":
        content = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow generate a headline for this news article:\nNews Content: {news_content}"
    elif QUERY_TYPE == "few-shot-si":
        content = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පුවත් ලිපිය සඳහා සිරස්තලයක් ජනනය කරන්න:\nNews Content: {news_content}"
    else:
        content = f"{task_desc} {action_desc} News Content: {news_content}"

    return [{"role": "user", "content": content}]


# --------------------------------------------------------------------------- #
# Output post-processing (Qwen thinking-aware)
# --------------------------------------------------------------------------- #
# Qwen3.5/3.6 think BY DEFAULT, emitting <think>...</think> before the answer.
# We disable thinking at generation time AND strip any residual block here.
_QWEN_THINK = re.compile(r'<think>.*?</think>', re.DOTALL | re.IGNORECASE)
_STRAY_THINK = re.compile(r'</?think>', re.IGNORECASE)


def strip_thinking(text: str) -> str:
    text = _QWEN_THINK.sub('', text)
    text = _STRAY_THINK.sub('', text)
    return text.strip()


def extract_headline(response: str):
    """Return (headline, matched_marker). Never dumps the whole raw response."""
    if not isinstance(response, str):
        return "", False
    text = strip_thinking(response)

    m = re.search(r'Headline:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip()
        cand = cand.splitlines()[0].strip() if cand else cand   # headline is one line
        if cand:
            return cand, True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False   # conservative fallback, flagged


# --------------------------------------------------------------------------- #
# Generation
# --------------------------------------------------------------------------- #
def generate(model, processor, list_of_messages: List[list], batch_size, max_new_tokens, do_sample) -> List[str]:
    outputs = []
    tok = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    # Qwen-recommended non-thinking sampling params (used only if do_sample=True)
    sample_kwargs = dict(temperature=0.7, top_p=0.80, top_k=20) if do_sample else {}

    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating headlines"):
        batch = list_of_messages[start:start + batch_size]
        try:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True,
                enable_thinking=False)          # Qwen honours this in its template
        except TypeError:
            inputs = processor.apply_chat_template(
                batch, add_generation_prompt=True, tokenize=True,
                padding=True, return_tensors="pt", return_dict=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                 do_sample=do_sample, pad_token_id=tok.pad_token_id,
                                 **sample_kwargs)
        outputs.extend(tok.batch_decode(gen[:, input_len:], skip_special_tokens=True))
    return outputs


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def predict(model, processor, model_id, batch_size, max_new_tokens, test_size, do_sample):
    print("Loading NSINA-Headlines dataset...")
    ds = load_dataset("sinhala-nlp/NSINA-Headlines")
    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    test_df = test_df[test_df['News Content'].notna() & test_df['Headline'].notna()].copy()
    train_df = train_df[train_df['News Content'].notna() & train_df['Headline'].notna()].copy()
    print(f"After filtering - Train: {len(train_df)}, Test: {len(test_df)}")

    df = test_df.head(min(test_size, len(test_df))).copy()
    print(f"Using {len(df)} test samples")
    df = analyze_and_trim_dataset(df)

    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        chat_messages = []
        for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            fs = get_few_shot_examples_for_instance(train_df, instance_idx=idx, num_examples=3, seed=42)
            chat_messages.append(format_chat(row, fs))
        df['chat'] = chat_messages
    else:
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    log_gpu_memory()
    responses = generate(model, processor, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses
    log_gpu_memory()

    preds, matched = zip(*[extract_headline(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Headline:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}  <-- check for truncated/looping <think> if high")

    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions.csv"), index=False, encoding='utf-8')

    print("Evaluating with ROUGE (Sinhala-safe, whitespace tokenized)...")
    rouge = score_corpus(df['Headline'].tolist(), df['preds'].tolist())
    df.to_csv(os.path.join(OUTPUT_FOLDER, "predictions_with_rouge.csv"), index=False, encoding='utf-8')

    print("\n" + "=" * 60)
    print("ROUGE F1 (x100)")
    print("=" * 60)
    for t in ROUGE_TYPES:
        print(f"{t:8s} mean={rouge[t]['mean']:.4f}  median={rouge[t]['median']:.4f}  std={rouge[t]['std']:.4f}")
    print("=" * 60)

    with open(os.path.join(OUTPUT_FOLDER, "rouge_summary.txt"), 'w', encoding='utf-8') as f:
        f.write(f"Model: {model_id}\nQuery type: {QUERY_TYPE}\nDataset: NSINA-Headlines\n")
        f.write(f"Samples: {len(df)}\nFormat misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Max new tokens: {max_new_tokens}\nBatch size: {batch_size}\n")
        f.write(f"Decoding: {'sampling(t=0.7,p=0.8,k=20)' if do_sample else 'greedy'}\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write("Few-shot approach: Dynamic (unique examples per test instance from train set)\n")
        f.write("Metric: ROUGE F1 x100, whitespace-tokenized (Sinhala-safe)\n" + "=" * 60 + "\n")
        for t in ROUGE_TYPES:
            r = rouge[t]
            f.write(f"{t}:\n  Mean: {r['mean']:.4f}\n  Std: {r['std']:.4f}\n  Median: {r['median']:.4f}\n"
                    f"  Min: {r['min']:.4f}\n  Max: {r['max']:.4f}\n")

    return df['preds'].tolist(), rouge


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='Qwen/Qwen3.5-35B-A3B')
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--do_sample', action='store_true',
                        help='Use Qwen-recommended sampling (t=0.7,p=0.8,k=20) instead of greedy. '
                             'Qwen warns greedy can cause repetition loops.')
    args = parser.parse_args()

    QUERY_TYPE = args.query_type
    model_id = args.model_id
    print(f"Model: {model_id}\nQuery type: {QUERY_TYPE}")
    print(f"Batch size: {args.batch_size}\nMax new tokens: {args.max_new_tokens}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    OUTPUT_FOLDER = os.path.join("outputs", "headline_generation", model_id.split('/')[-1], QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    processor = AutoProcessor.from_pretrained(model_id)
    model = _AutoModel.from_pretrained(model_id, dtype="auto", device_map="auto")
    model.eval()

    if hasattr(model, 'hf_device_map'):
        dist = {}
        for _, dev in model.hf_device_map.items():
            dist[dev] = dist.get(dev, 0) + 1
        print("Device map:", dist)

    log_gpu_memory()
    predict(model, processor, model_id, args.batch_size, args.max_new_tokens,
            args.test_size, args.do_sample)
    log_gpu_memory()