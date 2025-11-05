# Quick Start Guide - SinGen Unified Benchmark Runner

## TL;DR

Run all benchmarks with a single command:

```bash
./run_benchmarks.sh --model <MODEL> --query_type <QUERY_TYPE>
```

## Minimal Examples

### Run Everything
```bash
# With Cohere API
./run_benchmarks.sh --model cohere --query_type zero-shot

# With OpenAI API
./run_benchmarks.sh --model open_ai --query_type few-shot

# With local Hugging Face model
./run_benchmarks.sh --model hf_llm --query_type zero-shot-si

# With MT0 model
./run_benchmarks.sh --model mt0 --query_type few-shot-si
```

### Run Specific Tasks
```bash
# Only text simplification
./run_benchmarks.sh --model cohere --query_type zero-shot --tasks simplification

# Only translation tasks
./run_benchmarks.sh --model open_ai --query_type few-shot --tasks translation

# Multiple specific tasks
./run_benchmarks.sh --model hf_llm --query_type zero-shot \
    --tasks simplification,summarization
```

### Run Specific Translation Pairs
```bash
# Only English-Sinhala translation
./run_benchmarks.sh --model cohere --query_type zero-shot \
    --tasks translation --translation_pairs en_si

# Multiple translation pairs
./run_benchmarks.sh --model open_ai --query_type few-shot \
    --tasks translation --translation_pairs en_si,ta_si
```

## Valid Options

| Parameter | Valid Values |
|-----------|--------------|
| `--model` | `open_ai`, `cohere`, `hf_llm`, `mt0` |
| `--query_type` | `zero-shot`, `zero-shot-si`, `few-shot`, `few-shot-si` |
| `--tasks` | `simplification`, `summarization`, `headline`, `translation`, `all` (default) |
| `--translation_pairs` | `en_si`, `ta_si`, `pi_si`, `all` (default) |

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

**Model**: cohere
**Query Type**: zero-shot

Executes:
1. `python -m text_simplification.cohere --query_type=zero-shot`
2. `python -m text_summerisation.cohere --query_type=zero-shot`
3. `python -m headline_generation.cohere --query_type=zero-shot`
4. `python -m machine_translation.en_si.cohere --query_type=zero-shot`
5. `python -m machine_translation.ta_si.cohere --query_type=zero-shot`
6. `python -m machine_translation.pi_si.cohere --query_type=zero-shot`

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
