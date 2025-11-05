# SinGen Unified Benchmark Runner

This document describes how to use the unified benchmark runner to execute all SinGen benchmark tasks with a single command.

## Overview

The unified benchmark runner allows you to execute all or selected benchmark tasks (text simplification, text summarization, headline generation, and machine translation) by simply specifying the model name and query type.

## Quick Start

### Using the Shell Script (Recommended)

```bash
# Run all benchmarks with Cohere model and zero-shot queries
./run_benchmarks.sh --model cohere --query_type zero-shot

# Run all benchmarks with OpenAI and few-shot queries
./run_benchmarks.sh --model open_ai --query_type few-shot
```

### Using the Python Script Directly

```bash
# Run all benchmarks
python run_all_benchmarks.py --model cohere --query_type zero-shot

# Run specific tasks
python run_all_benchmarks.py --model open_ai --query_type few-shot --tasks simplification,summarization
```

## Available Options

### Required Arguments

- `--model`: Model backend to use
  - Options: `open_ai`, `cohere`, `hf_llm`, `mt0`

- `--query_type`: Query type for benchmarks
  - Options: `zero-shot`, `zero-shot-si`, `few-shot`, `few-shot-si`

### Optional Arguments

- `--tasks`: Comma-separated list of tasks to run (default: all)
  - Options: `simplification`, `summarization`, `headline`, `translation`

- `--translation_pairs`: Comma-separated list of translation pairs (default: all)
  - Options: `en_si`, `ta_si`, `pi_si`

## Model Backends

### 1. OpenAI (`open_ai`)
- Uses GPT-4o by default
- Requires `OPENAI_API_KEY` environment variable
- Temperature: 0.3
- Max tokens: 500

### 2. Cohere (`cohere`)
- Uses command-r-03-2025 by default
- Requires `COHERE_API_KEY` environment variable
- Temperature: 0.3

### 3. Hugging Face LLM (`hf_llm`)
- Uses Llama-3.3-70B-Instruct by default
- Runs locally with device_map: "auto"
- torch_dtype: bfloat16

### 4. MT0 (`mt0`)
- Uses bigscience/mt0-xxl
- Sequence-to-sequence model
- torch_dtype: bfloat16

## Query Types

### Zero-Shot Queries
- `zero-shot`: English prompts without examples
- `zero-shot-si`: Sinhala prompts without examples

### Few-Shot Queries
- `few-shot`: English prompts with 3 English examples
- `few-shot-si`: Sinhala prompts with 3 Sinhala examples

## Benchmark Tasks

### 1. Text Simplification
- Dataset: SiTSE (Sinhala Text Simplification Evaluation)
- Task: Simplify complex Sinhala sentences
- Metrics: SARI score (multi-reference evaluation)
- Output: `outputs/text_simplification/{model}/{query_type}/`

### 2. Text Summarization
- Task: Generate concise summaries of Sinhala documents
- Metrics: ROUGE scores
- Output: `outputs/text_summerisation/{model}/{query_type}/`

### 3. Headline Generation
- Task: Generate headlines for Sinhala documents
- Output: `outputs/headline_generation/{model}/{query_type}/`

### 4. Machine Translation
- Language pairs:
  - **en_si**: English → Sinhala
  - **ta_si**: Tamil → Sinhala
  - **pi_si**: Punjabi → Sinhala
- Dataset: FLORES-200 (devtest split)
- Metrics: BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, overall)
- Output: `outputs/{language_pair}_translation/{model}/{query_type}/`

## Usage Examples

### Example 1: Run All Benchmarks with Cohere

```bash
./run_benchmarks.sh --model cohere --query_type zero-shot
```

This will execute:
- Text simplification
- Text summarization
- Headline generation
- Machine translation (English-Sinhala, Tamil-Sinhala, Punjabi-Sinhala)

### Example 2: Run Only Text Tasks with OpenAI

```bash
./run_benchmarks.sh --model open_ai --query_type few-shot \
    --tasks simplification,summarization,headline
```

This will execute only:
- Text simplification
- Text summarization
- Headline generation

### Example 3: Run Only English-Sinhala Translation

```bash
./run_benchmarks.sh --model hf_llm --query_type zero-shot-si \
    --tasks translation --translation_pairs en_si
```

This will execute only English-Sinhala machine translation.

### Example 4: Run Multiple Translation Pairs

```bash
./run_benchmarks.sh --model mt0 --query_type few-shot-si \
    --tasks translation --translation_pairs en_si,ta_si
```

This will execute English-Sinhala and Tamil-Sinhala machine translation.

### Example 5: Compare Different Query Types

```bash
# Zero-shot with English prompts
./run_benchmarks.sh --model cohere --query_type zero-shot

# Zero-shot with Sinhala prompts
./run_benchmarks.sh --model cohere --query_type zero-shot-si

# Few-shot with English prompts
./run_benchmarks.sh --model cohere --query_type few-shot

# Few-shot with Sinhala prompts
./run_benchmarks.sh --model cohere --query_type few-shot-si
```

### Example 6: Test Different Models on Same Task

```bash
# Test with OpenAI
./run_benchmarks.sh --model open_ai --query_type zero-shot --tasks simplification

# Test with Cohere
./run_benchmarks.sh --model cohere --query_type zero-shot --tasks simplification

# Test with Hugging Face model
./run_benchmarks.sh --model hf_llm --query_type zero-shot --tasks simplification

# Test with MT0
./run_benchmarks.sh --model mt0 --query_type zero-shot --tasks simplification
```

## Setting Up API Keys

Before running benchmarks with API-based models, ensure you have the required API keys set:

```bash
# For OpenAI
export OPENAI_API_KEY="your-openai-api-key"

# For Cohere
export COHERE_API_KEY="your-cohere-api-key"

# Or add them to your ~/.bashrc or ~/.zshrc
echo 'export OPENAI_API_KEY="your-openai-api-key"' >> ~/.bashrc
echo 'export COHERE_API_KEY="your-cohere-api-key"' >> ~/.bashrc
source ~/.bashrc
```

## Output Structure

The benchmark runner will:
1. Execute each task sequentially
2. Display progress and status for each task
3. Save results to the `outputs/` directory
4. Print a summary at the end showing which tasks passed/failed

Example output structure:
```
outputs/
├── text_simplification/
│   ├── cohere/
│   │   ├── zero-shot/
│   │   │   ├── predictions.csv
│   │   │   ├── predictions_with_sari_multi_ref.csv
│   │   │   └── sari_summary_multi_ref.txt
│   │   └── few-shot/
│   └── open_ai/
├── text_summerisation/
├── headline_generation/
└── english_sinhala_translation/
```

## Monitoring Progress

The benchmark runner provides real-time feedback:

```
================================================================================
Running: Text Simplification (cohere, zero-shot)
Command: python -m text_simplification.cohere --query_type=zero-shot
================================================================================

[Task output...]

✓ Text Simplification (cohere, zero-shot) completed successfully!
```

At the end, you'll see a summary:

```
################################################################################
# Benchmark Summary
################################################################################

✓ PASS: Text Simplification (cohere, zero-shot)
✓ PASS: Text Summarization (cohere, zero-shot)
✓ PASS: Headline Generation (cohere, zero-shot)
✓ PASS: Machine Translation EN-SI (cohere, zero-shot)
✓ PASS: Machine Translation TA-SI (cohere, zero-shot)
✓ PASS: Machine Translation PI-SI (cohere, zero-shot)

================================================================================
Total: 6/6 tasks completed successfully
Duration: 0:45:23
End Time: 2025-11-05 14:30:45
================================================================================
```

## Troubleshooting

### Permission Denied Error

If you get a permission denied error, make the scripts executable:

```bash
chmod +x run_all_benchmarks.py run_benchmarks.sh
```

### API Key Not Found

If you see API key errors, ensure your environment variables are set:

```bash
echo $OPENAI_API_KEY
echo $COHERE_API_KEY
```

### Out of Memory Errors

For local models (hf_llm, mt0), you may need:
- Sufficient GPU memory (typically 40GB+ for 70B models)
- Reduce batch size in the model scripts if needed

### Task Failures

If a specific task fails:
1. Check the error message in the output
2. Run that specific task individually to debug:
   ```bash
   python -m text_simplification.cohere --query_type=zero-shot
   ```
3. Check the outputs directory for partial results

## Advanced Usage

### Running in Background

For long-running benchmarks, use `nohup`:

```bash
nohup ./run_benchmarks.sh --model cohere --query_type zero-shot > benchmark.log 2>&1 &
```

Monitor progress:
```bash
tail -f benchmark.log
```

### Integration with SLURM

The runner can be integrated into SLURM batch scripts:

```bash
#!/bin/bash
#SBATCH --job-name=sinben_all
#SBATCH --output=benchmark_%j.out
#SBATCH --error=benchmark_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

source activate your_environment
./run_benchmarks.sh --model hf_llm --query_type zero-shot
```

### Parallel Execution

To run multiple query types in parallel:

```bash
# Run in separate terminals or as background jobs
./run_benchmarks.sh --model cohere --query_type zero-shot &
./run_benchmarks.sh --model cohere --query_type few-shot &
./run_benchmarks.sh --model cohere --query_type zero-shot-si &
./run_benchmarks.sh --model cohere --query_type few-shot-si &
wait
```

## Getting Help

```bash
# Show help for shell script
./run_benchmarks.sh --help

# Show help for Python script
python run_all_benchmarks.py --help
```

## Notes

- Each benchmark task saves results independently
- Failed tasks don't prevent subsequent tasks from running
- Exit code is non-zero if any task fails
- All tasks use the same model and query type in a single run
- For comparing models or query types, run the script multiple times with different parameters
