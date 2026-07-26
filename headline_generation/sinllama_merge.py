import argparse
import os
import re
import random
from typing import List

import pandas as pd
import torch
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed

# Sinhala-safe ROUGE (whitespace-tokenized). The stock rouge_score tokenizer
# strips non-ASCII and would zero out every Sinhala score. Same module as the
# summarisation task, so headline and summarisation ROUGE-L stay comparable.
from rouge_metric import score_corpus, ROUGE_TYPES

set_seed(777)

MAX_CONTENT_LENGTH = 2500  # characters, to bound prompt length


# --------------------------------------------------------------------------- #
# What differs from the Llama-3 headline prompting script
# --------------------------------------------------------------------------- #
# Almost nothing on the model side -- merged SinLlama checkpoints are text-only
# LlamaForCausalLM, which is what that script already targeted. Added here:
#   * --run_tag / --output_root so three merge variants write side by side.
#   * chat_template guard (merged dirs need --copy_chat_template).
#   * embedding-row check at load: catches a merge that dropped the extended
#     Sinhala rows before an hour of decoding does.
#   * token-budget report + fit: MAX_CONTENT_LENGTH bounds characters, not
#     tokens, and Sinhala fertility differs a lot between the extended (139k)
#     tokenizer and stock Llama-3. Overruns show up as garbage, not errors.
# Prompt wording and the four query types are byte-identical to the prompting
# eval. Zero/few-shot ONLY -- no fine-tuning.


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
# Prompting (identical wording to the other headline scripts for ROUGE
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
# Context fitting
# --------------------------------------------------------------------------- #
# NOTE: apply_chat_template(tokenize=True) returns a BatchEncoding on some
# transformers versions rather than a flat id list, so length is measured via
# tokenize=False + a separate tokenizer call.
def n_prompt_tokens(tok, messages) -> int:
    text = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return len(tok(text, add_special_tokens=False)["input_ids"])


def fit_prompt(tok, row, few_shot_examples, budget):
    """Build the chat, shortening the article body if the prompt exceeds budget.

    Returns (messages, n_tokens, was_truncated). Few-shot previews are left
    intact so the demonstration set stays identical across model families;
    only the article under test is shortened.
    """
    msgs = format_chat(row, few_shot_examples)
    n = n_prompt_tokens(tok, msgs)
    if n <= budget:
        return msgs, n, False

    row = row.copy()
    art_ids = tok(str(row['News Content']), add_special_tokens=False)["input_ids"]
    keep = max(64, len(art_ids) - (n - budget) - 32)
    for _ in range(4):
        row['News Content'] = tok.decode(art_ids[:keep], skip_special_tokens=True)
        msgs = format_chat(row, few_shot_examples)
        n = n_prompt_tokens(tok, msgs)
        if n <= budget:
            break
        keep = max(64, keep - (n - budget) - 64)
    return msgs, n, True


# --------------------------------------------------------------------------- #
# Output post-processing
# --------------------------------------------------------------------------- #
# Llama-3 has no reasoning channel, but chat models occasionally leak stray
# special tokens when decoding is cut short; strip them defensively.
_STRAY_TAGS = re.compile(r'<\|[^|>]*\|>')


def extract_headline(response: str):
    """Return (headline, matched_marker). Never dumps the whole raw response."""
    if not isinstance(response, str):
        return "", False
    text = _STRAY_TAGS.sub('', response).strip()

    m = re.search(r'Headline:\s*(.*)', text, re.IGNORECASE | re.DOTALL)
    if m:
        cand = m.group(1).strip()
        cand = cand.splitlines()[0].strip() if cand else cand   # headline is one line
        if cand:
            return cand, True

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return (lines[-1] if lines else ""), False   # conservative fallback, flagged


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


def generate(model, tokenizer, list_of_messages: List[list], batch_size: int,
             max_new_tokens: int, do_sample: bool = False) -> List[str]:
    """Batched decoding with left padding so new tokens start at the same offset
    for every sequence in a batch."""
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    terminators = build_terminators(tokenizer)
    sample_kwargs = dict(temperature=0.6, top_p=0.9) if do_sample else {}  # Llama defaults

    outputs = []
    for start in tqdm(range(0, len(list_of_messages), batch_size), desc="Generating headlines"):
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
# Pipeline
# --------------------------------------------------------------------------- #
def predict(model, tokenizer, model_id, batch_size, max_new_tokens, test_size,
            budget, do_sample=False):
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

    msgs, tok_lens, truncated = [], [], []
    use_fs = QUERY_TYPE in ["few-shot", "few-shot-si"]
    for idx, (_, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing prompts")):
        fs = get_few_shot_examples_for_instance(train_df, instance_idx=idx, num_examples=3, seed=42) if use_fs else None
        m, n, was_trunc = fit_prompt(tokenizer, row, fs, budget)
        msgs.append(m)
        tok_lens.append(n)
        truncated.append(was_trunc)
    df['chat'] = msgs
    df['prompt_tokens'] = tok_lens
    df['article_truncated'] = truncated

    n_trunc = int(sum(truncated))
    print(f"\nPrompt length (tokens): max={max(tok_lens)} mean={sum(tok_lens) / len(tok_lens):.0f}")
    print(f"Articles shortened to fit {budget}-token budget: {n_trunc}/{len(df)}")

    log_gpu_memory()
    responses = generate(model, tokenizer, df['chat'].tolist(), batch_size, max_new_tokens, do_sample)
    df['responses'] = responses
    log_gpu_memory()

    preds, matched = zip(*[extract_headline(r) for r in responses])
    df['preds'] = list(preds)
    df['marker_matched'] = list(matched)

    n_miss = int((~df['marker_matched']).sum())
    n_empty = int((df['preds'].str.len() == 0).sum())
    print(f"\nFormat misses (no 'Headline:' marker): {n_miss}/{len(df)}")
    print(f"Empty predictions: {n_empty}/{len(df)}")

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
        f.write(f"Model: {model_id}\nRun tag: {RUN_TAG}\nQuery type: {QUERY_TYPE}\n")
        f.write("Dataset: NSINA-Headlines\n")
        f.write(f"Samples: {len(df)}\nFormat misses: {n_miss}/{len(df)}  Empty preds: {n_empty}/{len(df)}\n")
        f.write(f"Max new tokens: {max_new_tokens}\nBatch size: {batch_size}\n")
        f.write(f"Prompt budget: {budget} tokens  Articles shortened: {n_trunc}/{len(df)}\n")
        f.write(f"Prompt tokens: max={max(tok_lens)} mean={sum(tok_lens) / len(tok_lens):.0f}\n")
        f.write(f"Decoding: {'sampling(t=0.6,p=0.9)' if do_sample else 'greedy'}\n")
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
    parser.add_argument('--model_id', type=str, required=True,
                        help='Path to a merged checkpoint directory.')
    parser.add_argument('--run_tag', type=str, default=None,
                        help='Output folder name. Defaults to basename of --model_id.')
    parser.add_argument('--output_root', type=str,
                        default=os.path.dirname(os.path.abspath(__file__)),
                        help="Base dir for outputs/. Defaults to the script's own directory.")
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        help='zero-shot, zero-shot-si, few-shot, few-shot-si')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--max_prompt_tokens', type=int, default=0,
                        help='0 = derive from model config (max_position_embeddings - max_new_tokens - 64)')
    parser.add_argument('--do_sample', action='store_true',
                        help='Llama sampling (t=0.6,p=0.9) instead of greedy')
    args = parser.parse_args()

    model_id = args.model_id
    QUERY_TYPE = args.query_type
    RUN_TAG = args.run_tag or os.path.basename(model_id.rstrip('/'))
    print(f"Model: {model_id}\nRun tag: {RUN_TAG}\nQuery type: {QUERY_TYPE}")
    print(f"Batch size: {args.batch_size}\nMax new tokens: {args.max_new_tokens}")
    print(f"Decoding: {'sampling' if args.do_sample else 'greedy'}")

    if torch.cuda.is_available():
        print(f"CUDA devices available: {torch.cuda.device_count()}")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if not getattr(tokenizer, "chat_template", None):
        raise SystemExit(f"{model_id} has no chat_template; expected a merged checkpoint "
                         f"produced with --copy_chat_template.")
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto", device_map="auto")
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
                                 "headline_generation_zeroshot", RUN_TAG, QUERY_TYPE)
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print(f"Output folder: {OUTPUT_FOLDER}")

    log_gpu_memory()
    predict(model, tokenizer, model_id, args.batch_size, args.max_new_tokens,
            args.test_size, budget, args.do_sample)
    log_gpu_memory()