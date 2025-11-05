# SinGen - Sinhala Natural Language Generation Benchmark

A comprehensive benchmark suite for evaluating language models on Sinhala NLP tasks.

## Overview

SinGen provides a unified framework for benchmarking language models across multiple Sinhala natural language generation tasks:

- **Text Simplification**: Simplify complex Sinhala sentences (SiTSE dataset)
- **Text Summarization**: Generate concise summaries of Sinhala documents
- **Headline Generation**: Create headlines for Sinhala articles
- **Machine Translation**: Translate between Sinhala and other languages (EN, TA, PI)

## Quick Start

### Run All Benchmarks with One Command - Just Specify Your Model!

```bash
# Run all benchmarks by simply providing the model name
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
```

That's it! The backend is automatically detected, and all benchmark tasks are executed (text simplification, summarization, headline generation, and machine translation for all language pairs).

### More Examples

```bash
# With any HuggingFace model (backend auto-detected)
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
./run_benchmarks.sh --model mistralai/Mistral-7B-Instruct-v0.2 --query_type few-shot

# With OpenAI models (backend auto-detected)
./run_benchmarks.sh --model gpt-4o --query_type few-shot
./run_benchmarks.sh --model gpt-3.5-turbo --query_type zero-shot

# With Cohere models (backend auto-detected)
./run_benchmarks.sh --model command-r --query_type zero-shot-si

# Run specific tasks only
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot --tasks simplification,summarization

# Run specific translation pairs only
./run_benchmarks.sh --model gpt-4o --query_type few-shot --tasks translation --translation_pairs en_si
```

## Installation

```bash
# Clone the repository
git clone https://github.com/Sinhala-NLP/SinGen.git
cd SinGen

# Install dependencies
pip install -r requirements.txt

# Set API keys (for API-based models)
export OPENAI_API_KEY="your-openai-key"
export COHERE_API_KEY="your-cohere-key"
```

## Supported Models

**You can use ANY model by simply specifying its name!** The backend is automatically detected.

### Automatic Backend Detection

| Model Pattern | Backend | Examples |
|---------------|---------|----------|
| HuggingFace paths (with `/`) | `hf_llm` | `meta-llama/Meta-Llama-3-8B-Instruct`<br>`mistralai/Mistral-7B-Instruct-v0.2`<br>`google/gemma-7b` |
| OpenAI models | `open_ai` | `gpt-4o`, `gpt-4`, `gpt-3.5-turbo` |
| Cohere models | `cohere` | `command-r`, `command-r-plus`, `command` |
| MT0 models | `mt0` | `bigscience/mt0-xxl`, `mt0-large` |

### Or Use Backend Names with Default Models

| Backend | Default Model | Description |
|---------|---------------|-------------|
| `open_ai` | `gpt-4o` | OpenAI's latest model |
| `cohere` | `command-r-03-2025` | Cohere's command model |
| `hf_llm` | `meta-llama/Llama-3.3-70B-Instruct` | HuggingFace Llama model |
| `mt0` | `bigscience/mt0-xxl` | BigScience MT0 model |

## Query Types

| Query Type | Description |
|------------|-------------|
| `zero-shot` | English prompts without examples |
| `zero-shot-si` | Sinhala prompts without examples |
| `few-shot` | English prompts with 3 examples |
| `few-shot-si` | Sinhala prompts with 3 examples |

## Benchmark Tasks

### 1. Text Simplification
- **Dataset**: SiTSE (Sinhala Text Simplification Evaluation)
- **Size**: 200 test samples
- **Metric**: SARI score with multi-reference evaluation
- **Task**: Transform complex Sinhala sentences into simpler versions

### 2. Text Summarization
- **Size**: 200 test samples
- **Metric**: ROUGE scores
- **Task**: Generate concise summaries of Sinhala documents

### 3. Headline Generation
- **Size**: 200 test samples
- **Task**: Generate appropriate headlines for Sinhala articles

### 4. Machine Translation
- **Language Pairs**:
  - English → Sinhala (EN-SI)
  - Tamil → Sinhala (TA-SI)
  - Punjabi → Sinhala (PI-SI)
- **Dataset**: FLORES-200 (devtest split)
- **Metric**: BLEU scores (BLEU-1, BLEU-2, BLEU-3, BLEU-4, overall)

## Usage Options

### Required Arguments
- `--model`: **Any model name** (e.g., `meta-llama/Meta-Llama-3-8B-Instruct`, `gpt-4o`, `command-r`) - backend is auto-detected
  - OR backend name (`open_ai`, `cohere`, `hf_llm`, `mt0`) to use default model for that backend
- `--query_type`: Query type (`zero-shot`, `zero-shot-si`, `few-shot`, `few-shot-si`)

### Optional Arguments
- `--tasks`: Specific tasks to run (default: all)
  - Options: `simplification`, `summarization`, `headline`, `translation`
- `--translation_pairs`: Translation pairs to run (default: all)
  - Options: `en_si`, `ta_si`, `pi_si`

## Output Structure

Results are saved to the `outputs/` directory:

```
outputs/
├── text_simplification/
│   └── {model}/{query_type}/
│       ├── predictions.csv
│       ├── predictions_with_sari_multi_ref.csv
│       └── sari_summary_multi_ref.txt
├── text_summerisation/
│   └── {model}/{query_type}/
├── headline_generation/
│   └── {model}/{query_type}/
└── {language_pair}_translation/
    └── {model}/{query_type}/
        ├── predictions.csv
        ├── predictions_with_bleu.csv
        └── bleu_summary.txt
```

## Documentation

- **[QUICK_START.md](QUICK_START.md)**: Quick reference guide with minimal examples
- **[BENCHMARK_RUNNER.md](BENCHMARK_RUNNER.md)**: Comprehensive documentation with advanced usage

## Examples

### Compare Models on All Tasks
```bash
# Test different models on all benchmarks
./run_benchmarks.sh --model gpt-4o --query_type zero-shot
./run_benchmarks.sh --model command-r --query_type zero-shot
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
./run_benchmarks.sh --model mistralai/Mistral-7B-Instruct-v0.2 --query_type zero-shot
```

### Compare Query Types with Same Model
```bash
# Test all query types with the same model
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot-si
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type few-shot
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type few-shot-si
```

### Run Specific Task Combinations
```bash
# Only text generation tasks (no translation)
./run_benchmarks.sh --model gpt-4o --query_type few-shot \
    --tasks simplification,summarization,headline

# Only translation tasks
./run_benchmarks.sh --model command-r --query_type zero-shot-si \
    --tasks translation

# Only English-Sinhala translation
./run_benchmarks.sh --model meta-llama/Meta-Llama-3-8B-Instruct --query_type few-shot \
    --tasks translation --translation_pairs en_si
```

## Running Individual Tasks

You can also run individual benchmark tasks with specific models:

```bash
# Text simplification with custom model
python -m text_simplification.hf_llm --query_type=zero-shot --model_name=meta-llama/Meta-Llama-3-8B-Instruct

# Text summarization with OpenAI
python -m text_summerisation.open_ai --query_type=few-shot --model_name=gpt-4o

# Headline generation with Cohere
python -m headline_generation.cohere --query_type=zero-shot-si --model_name=command-r

# Machine translation with specific models
python -m machine_translation.en_si.hf_llm --query_type=few-shot-si --model_name=meta-llama/Meta-Llama-3-8B-Instruct
python -m machine_translation.ta_si.open_ai --query_type=zero-shot --model_name=gpt-3.5-turbo
python -m machine_translation.pi_si.cohere --query_type=few-shot --model_name=command-r-plus
```

## Requirements

- Python 3.8+
- PyTorch (for local models)
- Transformers
- Datasets
- OpenAI API key (for OpenAI models)
- Cohere API key (for Cohere models)
- GPU with sufficient memory (for local models like hf_llm and mt0)

See [requirements.txt](requirements.txt) for complete dependencies.

## Getting Help

```bash
# Show help for unified runner
./run_benchmarks.sh --help
python run_all_benchmarks.py --help

# Show help for individual tasks
python -m text_simplification.cohere --help
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Permission denied | `chmod +x run_benchmarks.sh run_all_benchmarks.py` |
| API key not found | Set environment variables: `export OPENAI_API_KEY="..."` |
| Out of memory | Use smaller model or GPU with more memory |
| Module not found | Install dependencies: `pip install -r requirements.txt` |

## Citation

If you use SinGen in your research, please cite:

```bibtex
@misc{singen2025,
  title={SinGen: Sinhala Natural Language Generation Benchmark},
  author={Sinhala-NLP},
  year={2025},
  url={https://github.com/Sinhala-NLP/SinGen}
}
```

## License

[Add license information]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

[Add contact information]
