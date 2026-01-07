import argparse
import os
import re
from typing import List, Tuple
import random
import time

import numpy as np
import pandas as pd
import torch
import intel_extension_for_pytorch as ipex
from datasets import Dataset, load_dataset
from tqdm.auto import tqdm
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
import gc

set_seed(777)


def log_gpu_memory():
    """Log GPU memory usage for all available XPU devices"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        print("\n" + "="*60)
        print("GPU Memory Usage:")
        for i in range(torch.xpu.device_count()):
            allocated = torch.xpu.memory_allocated(i) / 1024**3
            reserved = torch.xpu.memory_reserved(i) / 1024**3
            print(f"  GPU {i}:")
            print(f"    Allocated: {allocated:.2f} GB")
            print(f"    Reserved:  {reserved:.2f} GB")
        print("="*60 + "\n")


def clear_memory():
    """Aggressively clear GPU memory"""
    gc.collect()
    if hasattr(torch, 'xpu'):
        for i in range(torch.xpu.device_count()):
            with torch.xpu.device(i):
                torch.xpu.empty_cache()
                torch.xpu.synchronize()


def get_few_shot_examples_for_instance(train_df, instance_idx, num_examples=3, seed=None):
    """
    Get random few-shot examples for a specific test instance
    Each test instance will get different randomly selected examples
    """
    # Use instance-specific seed for randomization
    if seed is not None:
        random.seed(seed + instance_idx)

    # Randomly sample few-shot examples for this specific instance
    few_shot_indices = random.sample(range(len(train_df)), min(num_examples, len(train_df)))

    few_shot_examples = []
    for idx in few_shot_indices:
        row = train_df.iloc[idx]

        # Skip if news content or headline is missing
        if pd.notna(row['News Content']) and pd.notna(row['Headline']) and \
                str(row['News Content']).strip() and str(row['Headline']).strip():
            example = {
                'content': str(row['News Content']),
                'headline': str(row['Headline'])
            }
            few_shot_examples.append(example)

    return few_shot_examples


def format_chat(row, few_shot_examples=None):
    task_desc = "Imagine you are an expert in Sinhala language. Generate a concise and informative headline for the following Sinhala news article. The headline should capture the main point of the article in a brief, engaging manner."
    action_desc = "Return only the headline following the prefix 'Headline:' without any other text or explanations."

    task_desc_si = "ඔබ සිංහල භාෂාවේ ප්‍රවීණයෙකු ලෙස උපකල්පනය කරන්න. පහත සිංහල පුවත් ලිපිය සඳහා සංක්ෂිප්ත හා තොරතුරුදායක සිරස්තලයක් ජනනය කරන්න. සිරස්තලය කෙටි, ආකර්ෂණීය ආකාරයෙන් ලිපියේ ප්‍රධාන කරුණ ග්‍රහණය කර ගත යුතුය."
    action_desc_si = "'Headline:' යන ප්‍රත්‍යයයෙන් පසුව පමණක් සිරස්තලය ලබා දෙන්න. වෙනත් කිසිදු උපසර්ගයක් හෝ විස්තරයක් එක් නොකරන්න."

    # Build few-shot examples string if provided
    examples_str = ""
    if few_shot_examples:
        for i, example in enumerate(few_shot_examples, 1):
            # Truncate content if too long for context
            content_preview = example['content'][:500] + "..." if len(example['content']) > 500 else example['content']
            examples_str += f"\nExample {i}:\n"
            examples_str += f"News Content: {content_preview}\n"
            examples_str += f"Headline: {example['headline']}\n"

    if QUERY_TYPE == "zero-shot":
        return [{"role": "user", "content": f"{task_desc} {action_desc} News Content: {row['News Content']}"}]

    elif QUERY_TYPE == "zero-shot-si":
        return [{"role": "user", "content": f"{task_desc_si} {action_desc_si} News Content: {row['News Content']}"}]

    elif QUERY_TYPE == "few-shot":
        prompt = f"{task_desc}\n\n{action_desc}\n\nHere are some examples:{examples_str}\n\nNow generate a headline for this news article:\nNews Content: {row['News Content']}"
        return [{"role": "user", "content": prompt}]

    elif QUERY_TYPE == "few-shot-si":
        prompt = f"{task_desc_si}\n\n{action_desc_si}\n\nමෙන්න උදාහරණ කිහිපයක්:{examples_str}\n\nදැන් මේ පුවත් ලිපිය සඳහා සිරස්තලයක් ජනනය කරන්න:\nNews Content: {row['News Content']}"
        return [{"role": "user", "content": prompt}]

    else:
        # Default fallback
        return [{"role": "user", "content": f"{task_desc} {action_desc} News Content: {row['News Content']}"}]


def query(pipe, inputs):
    """
    :param pipe: text-generation pipeline
    :param inputs: list of messages
    :return: list
    """
    assistant_outputs = []

    terminators = [
        pipe.tokenizer.eos_token_id,
        pipe.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    print(f"\nProcessing {len(inputs)} inputs one at a time...")

    # Process one at a time to avoid memory issues
    for idx, single_input in enumerate(tqdm(inputs, desc="Generating headlines")):
        # Clear memory every 10 iterations
        if idx % 10 == 0:
            clear_memory()

        try:
            out = pipe(
                single_input,
                max_new_tokens=150,
                eos_token_id=terminators,
                pad_token_id=pipe.tokenizer.eos_token_id,
                num_return_sequences=1,
            )
            assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())

        except RuntimeError as e:
            error_str = str(e)
            if "OUT_OF_RESOURCES" in error_str or "out of memory" in error_str.lower():
                print(f"\n[Sample {idx}] OOM error, clearing cache and retrying...")
                clear_memory()

                # Retry with shorter generation
                try:
                    out = pipe(
                        single_input,
                        max_new_tokens=80,  # Significantly reduced
                        eos_token_id=terminators,
                        pad_token_id=pipe.tokenizer.eos_token_id,
                        num_return_sequences=1,
                    )
                    assistant_outputs.append(out[0]["generated_text"][-1]['content'].strip())
                    print(f"[Sample {idx}] Retry successful with reduced tokens")
                except Exception as retry_e:
                    print(f"[Sample {idx}] Retry failed: {retry_e}")
                    assistant_outputs.append("")
                    clear_memory()
            else:
                print(f"[Sample {idx}] Error: {e}")
                assistant_outputs.append("")
                clear_memory()

        except Exception as e:
            print(f"[Sample {idx}] Unexpected error: {e}")
            assistant_outputs.append("")
            clear_memory()

    return assistant_outputs


def predict(pipe_lm, model_id):
    # Load the dataset
    print("Loading NSINA-Headlines dataset...")
    ds = load_dataset("sinhala-nlp/NSINA-Headlines")

    train_df = ds["train"].to_pandas()
    test_df = ds["test"].to_pandas()

    print(f"Train size: {len(train_df)}")
    print(f"Test size: {len(test_df)}")
    print(f"Columns: {test_df.columns.tolist()}")

    # Filter out rows with missing News Content or Headline
    test_df = test_df[test_df['News Content'].notna() & test_df['Headline'].notna()].copy()
    train_df = train_df[train_df['News Content'].notna() & train_df['Headline'].notna()].copy()

    print(f"After filtering - Train size: {len(train_df)}, Test size: {len(test_df)}")

    # Use first 1000 test samples
    test_size = min(1000, len(test_df))
    df = test_df.head(test_size).copy()
    print(f"Using {len(df)} test samples")

    # Get few-shot examples if using few-shot learning
    if QUERY_TYPE in ["few-shot", "few-shot-si"]:
        print("Getting dynamic few-shot examples for each test instance...")
        print(f"Available training examples: {len(train_df)}")

        # Apply few-shot formatting with dynamic example selection per instance
        chat_messages = []
        for idx, (test_idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Preparing few-shot prompts")):
            # Get unique few-shot examples for this specific test instance
            few_shot_examples = get_few_shot_examples_for_instance(
                train_df,
                instance_idx=idx,
                num_examples=3,
                seed=42  # Base seed for reproducibility
            )

            # Format the chat with these examples
            chat_message = format_chat(row, few_shot_examples)
            chat_messages.append(chat_message)

        df['chat'] = chat_messages
        print(f"Each test instance has been assigned unique few-shot examples")
    else:
        # Use zero-shot formatting
        df['chat'] = df.apply(lambda row: format_chat(row, None), axis=1)

    # Log GPU memory before generation
    print("\nGPU memory usage before generation:")
    log_gpu_memory()

    # Clear memory before starting generation
    clear_memory()

    # Generate responses
    print("Generating headlines...")
    responses = query(pipe_lm, df['chat'].tolist())
    df['responses'] = responses

    # Log GPU memory after generation
    print("\nGPU memory usage after generation:")
    log_gpu_memory()

    # Extract predictions
    print("Extracting headlines...")
    df['preds'] = df.apply(lambda row: extract_headline(row['responses']), axis=1)

    # Save predictions
    predictions_file = os.path.join(OUTPUT_FOLDER, "predictions.csv")
    df.to_csv(predictions_file, header=True, index=False, encoding='utf-8')
    print(f"Predictions saved to: {predictions_file}")

    # Evaluate with ROUGE
    print("Evaluating with ROUGE metrics...")
    rouge_results = evaluate_rouge_scores(df, model_id)

    # Save results with ROUGE scores
    results_file = os.path.join(OUTPUT_FOLDER, "predictions_with_rouge.csv")
    df.to_csv(results_file, header=True, index=False, encoding='utf-8')
    print(f"Results with ROUGE scores saved to: {results_file}")

    # Save summary statistics
    summary_file = os.path.join(OUTPUT_FOLDER, "rouge_summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"ROUGE Score Evaluation Results\n")
        f.write(f"Model: {model_id}\n")
        f.write(f"Query Type: {QUERY_TYPE}\n")
        f.write(f"Dataset: NSINA-Headlines\n")
        f.write(f"Dataset Size: {len(df)} samples\n")
        if QUERY_TYPE in ["few-shot", "few-shot-si"]:
            f.write(f"Few-shot approach: Dynamic (unique examples per test instance)\n")

        # Add GPU info
        if hasattr(torch, 'xpu') and torch.xpu.is_available():
            f.write(f"\nGPU Configuration:\n")
            f.write(f"  Number of GPUs: {torch.xpu.device_count()}\n")
            for i in range(torch.xpu.device_count()):
                allocated = torch.xpu.memory_allocated(i) / 1024 ** 3
                reserved = torch.xpu.memory_reserved(i) / 1024 ** 3
                f.write(f"  GPU {i} - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB\n")

        f.write(f"=" * 60 + "\n")
        f.write(f"ROUGE-1:\n")
        f.write(f"  Mean: {rouge_results['rouge1']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rouge1']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rouge1']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rouge1']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rouge1']['max']:.4f}\n")
        f.write(f"\nROUGE-2:\n")
        f.write(f"  Mean: {rouge_results['rouge2']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rouge2']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rouge2']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rouge2']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rouge2']['max']:.4f}\n")
        f.write(f"\nROUGE-L:\n")
        f.write(f"  Mean: {rouge_results['rougeL']['mean']:.4f}\n")
        f.write(f"  Std:  {rouge_results['rougeL']['std']:.4f}\n")
        f.write(f"  Median: {rouge_results['rougeL']['median']:.4f}\n")
        f.write(f"  Min: {rouge_results['rougeL']['min']:.4f}\n")
        f.write(f"  Max: {rouge_results['rougeL']['max']:.4f}\n")

    print(f"Summary statistics saved to: {summary_file}")

    return df['preds'].tolist(), rouge_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_id', type=str, default='meta-llama/Llama-3.3-70B-Instruct',
                        required=False, help='Model ID from HuggingFace')
    parser.add_argument('--query_type', type=str, default='zero-shot',
                        required=False, help='Type of query (zero-shot, zero-shot-si, few-shot, few-shot-si)')

    args = parser.parse_args()

    # Set global variables
    model_id = args.model_id
    QUERY_TYPE = args.query_type

    print(f"Model: {model_id}")
    print(f"Query type: {QUERY_TYPE}")

    # Create output folder with query type
    OUTPUT_FOLDER = os.path.join("outputs", "headline_generation", model_id.split('/')[-1], QUERY_TYPE)
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    # Check available devices
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        num_gpus = torch.xpu.device_count()
        print(f"Number of XPU devices available: {num_gpus}")

    # Clear any existing memory
    clear_memory()

    # Load tokenizer
    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with eager attention and max memory limit per device
    print(f"Loading model: {model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory={0: "50GiB", 1: "50GiB", 2: "50GiB"},  # Leave headroom on each GPU
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )

    # Create pipeline
    pipe_lm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        do_sample=False,
        top_p=1.0,
    )

    # Check device distribution
    if hasattr(model, 'hf_device_map'):
        print("\nModel device map:")
        device_distribution = {}
        for name, device in model.hf_device_map.items():
            device_distribution[device] = device_distribution.get(device, 0) + 1
        for device, count in sorted(device_distribution.items()):
            print(f"  {device}: {count} modules")
    else:
        print("\nNo explicit device map found")

    print("Model loaded successfully!")

    # Log initial GPU memory
    print("\nInitial GPU memory usage:")
    log_gpu_memory()

    predictions, rouge_results = predict(pipe_lm, model_id)

    # Log final GPU memory
    print("\nFinal GPU memory usage:")
    log_gpu_memory()