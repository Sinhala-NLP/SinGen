# Quick Start Guide - SinGen Unified Benchmark Runner

## TL;DR

Run all benchmarks with a single command:

```bash
# RECOMMENDED: Specify both backend and model explicitly
./run_benchmarks.sh --model <BACKEND> --model_name <MODEL_NAME> --query_type <QUERY_TYPE>

# Or let it auto-detect the backend
./run_benchmarks.sh --model <MODEL_NAME> --query_type <QUERY_TYPE>
```

## Minimal Examples

### RECOMMENDED: Explicit Backend and Model
```bash
# HuggingFace models
./run_benchmarks.sh --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot

# OpenAI models
./run_benchmarks.sh --model open_ai --model_name gpt-4o --query_type few-shot
./run_benchmarks.sh --model open_ai --model_name gpt-3.5-turbo --query_type zero-shot

# Cohere models
./run_benchmarks.sh --model cohere --model_name command-r --query_type zero-shot-si
./run_benchmarks.sh --model cohere --model_name command-r-plus --query_type few-shot

# MT0 models
./run_benchmarks.sh --model mt0 --model_name bigscience/mt0-xxl --query_type few-shot-si
```

### Auto-Detect Backend (Quick but Less Explicit)
```bash
# Backend auto-detected from model name
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
./run_benchmarks.sh --model gpt-4o --query_type few-shot
./run_benchmarks.sh --model_name command-r --query_type zero-shot-si
```

### Use Backend with Default Model
```bash
# Uses default Cohere model (command-r-03-2025)
./run_benchmarks.sh --model cohere --query_type zero-shot

# Uses default OpenAI model (gpt-4o)
./run_benchmarks.sh --model open_ai --query_type few-shot

# Uses default HuggingFace model (Llama-3.3-70B-Instruct)
./run_benchmarks.sh --model hf_llm --query_type zero-shot-si

# Uses default MT0 model (bigscience/mt0-xxl)
./run_benchmarks.sh --model mt0 --query_type few-shot-si
```

### Run Specific Tasks
```bash
# Only text simplification with any model
./run_benchmarks.sh --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot --tasks simplification

# Only translation tasks
./run_benchmarks.sh --model open_ai --model_name gpt-4o --query_type few-shot --tasks translation

# Multiple specific tasks
./run_benchmarks.sh --model cohere --model_name command-r --query_type zero-shot \
    --tasks simplification,summarization
```

### Run Specific Translation Pairs
```bash
# Only English-Sinhala translation
./run_benchmarks.sh --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot \
    --tasks translation --translation_pairs en_si

# Multiple translation pairs
./run_benchmarks.sh --model open_ai --model_name gpt-4o --query_type few-shot \
    --tasks translation --translation_pairs en_si,ta_si
```

## Valid Options

| Parameter | Valid Values |
|-----------|--------------|
| `--model` | Backend names: `open_ai`, `cohere`, `hf_llm`, `mt0`<br>OR model name for auto-detection |
| `--model_name` | Actual model name (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, `gpt-4o`, `command-r`) |
| `--query_type` | `zero-shot`, `zero-shot-si`, `few-shot`, `few-shot-si` (REQUIRED) |
| `--tasks` | `simplification`, `summarization`, `headline`, `translation`, `all` (default) |
| `--translation_pairs` | `en_si`, `ta_si`, `pi_si`, `all` (default) |

**Note**: At least one of `--model` or `--model_name` must be provided.

### Three Ways to Specify Models

1. **RECOMMENDED: Both arguments** - Most explicit and clear
   ```bash
   --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct
   ```

2. **Only --model with backend name** - Uses default model for that backend
   ```bash
   --model cohere  # Uses command-r-03-2025
   ```

3. **Only --model or --model_name with model name** - Auto-detects backend
   ```bash
   --model gpt-4o  # Auto-detects open_ai backend
   --model_name meta-llama/Meta-Llama-3-8B-Instruct  # Auto-detects hf_llm backend
   ```

### Backend Auto-Detection Rules

- **HuggingFace models**: Any model with `/` → uses `hf_llm` backend
- **OpenAI models**: Models with `gpt-` prefix → uses `open_ai` backend
- **Cohere models**: Models with `command`, `coral`, `aya` → uses `cohere` backend
- **MT0 models**: Models with `mt0` or `bigscience/mt0` → uses `mt0` backend

## Before Running

### Set API Keys (for API models)
```bash
export OPENAI_API_KEY="your-key-here"
export COHERE_API_KEY="your-key-here"
```

### Check Requirements
```bash
pip install -r requirements.txt
```

## What Gets Executed

### Default (all tasks)
When you run with `--tasks all` (default), it executes:
- Text Simplification
- Text Summarization
- Headline Generation
- Machine Translation (EN→SI, TA→SI, PI→SI)

**Total: 6 benchmark tasks**

### Example Task Breakdown

**Model**: `meta-llama/Meta-Llama-3-8B-Instruct`
**Detected Backend**: `hf_llm` (auto-detected)
**Query Type**: zero-shot

Executes:
1. `python -m text_simplification.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`
2. `python -m text_summerisation.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`
3. `python -m headline_generation.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`
4. `python -m machine_translation.en_si.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`
5. `python -m machine_translation.ta_si.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`
6. `python -m machine_translation.pi_si.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct`

## Output Location

Results are saved to:
```
outputs/
├── text_simplification/{model}/{query_type}/
├── text_summerisation/{model}/{query_type}/
├── headline_generation/{model}/{query_type}/
└── {language_pair}_translation/{model}/{query_type}/
```

## Common Use Cases

### 1. Test a Single Model Across All Tasks
```bash
./run_benchmarks.sh --model cohere --query_type zero-shot
```

### 2. Compare Query Types for Same Model
```bash
./run_benchmarks.sh --model cohere --query_type zero-shot
./run_benchmarks.sh --model cohere --query_type few-shot
./run_benchmarks.sh --model cohere --query_type zero-shot-si
./run_benchmarks.sh --model cohere --query_type few-shot-si
```

### 3. Compare Different Models
```bash
./run_benchmarks.sh --model open_ai --query_type zero-shot
./run_benchmarks.sh --model cohere --query_type zero-shot
./run_benchmarks.sh --model hf_llm --query_type zero-shot
```

### 4. Focus on Specific Task
```bash
# Test all models on text simplification only
./run_benchmarks.sh --model open_ai --query_type zero-shot --tasks simplification
./run_benchmarks.sh --model cohere --query_type zero-shot --tasks simplification
./run_benchmarks.sh --model hf_llm --query_type zero-shot --tasks simplification
./run_benchmarks.sh --model mt0 --query_type zero-shot --tasks simplification
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | `chmod +x run_benchmarks.sh run_all_benchmarks.py` |
| API key error | Set environment variables: `export OPENAI_API_KEY="..."` |
| Out of memory | Use smaller model or reduce batch size |
| Task failed | Run individual task: `python -m text_simplification.cohere --query_type=zero-shot` |

## Getting Help

```bash
./run_benchmarks.sh --help
python run_all_benchmarks.py --help
```

## More Details

See [BENCHMARK_RUNNER.md](BENCHMARK_RUNNER.md) for comprehensive documentation.
