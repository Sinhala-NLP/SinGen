#!/usr/bin/env python3
"""
Unified Benchmark Runner for SinGen

This script runs all benchmark tasks (text simplification, text summarization,
headline generation, and machine translation) with a single command.

Usage:
    python run_all_benchmarks.py --model <backend> --model_name <model> --query_type <query_type>

Arguments:
    --model: Backend name (open_ai, cohere, hf_llm, mt0) OR model name for auto-detection
    --model_name: Actual model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct, gpt-4o)
    --query_type: Query type (zero-shot, zero-shot-si, few-shot, few-shot-si)

Note: You can provide:
    - Both --model (backend) and --model_name (model): Explicit control
    - Only --model: If backend name, uses default model; if model name, auto-detects backend
    - Only --model_name: Auto-detects backend from model name

Optional Arguments:
    --tasks: Comma-separated list of tasks to run (default: all)
            Options: simplification, summarization, headline, translation
    --translation_pairs: Comma-separated list of translation pairs (default: all)
            Options: en_si, ta_si, pi_si

Examples:
    # Explicit: Specify both backend and model name (RECOMMENDED)
    python run_all_benchmarks.py --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
    python run_all_benchmarks.py --model open_ai --model_name gpt-4o --query_type few-shot
    python run_all_benchmarks.py --model cohere --model_name command-r --query_type zero-shot-si

    # Auto-detect backend from model name
    python run_all_benchmarks.py --model meta-llama/Meta-Llama-3-8B-Instruct --query_type zero-shot
    python run_all_benchmarks.py --model_name gpt-4o --query_type few-shot

    # Use backend with default model
    python run_all_benchmarks.py --model cohere --query_type zero-shot

    # Run specific tasks
    python run_all_benchmarks.py --model hf_llm --model_name meta-llama/Meta-Llama-3-8B-Instruct --query_type few-shot --tasks simplification,summarization

    # Run specific translation pairs
    python run_all_benchmarks.py --model open_ai --model_name gpt-4o --query_type zero-shot-si --tasks translation --translation_pairs en_si
"""

import argparse
import subprocess
import sys
from typing import List, Dict, Tuple
from datetime import datetime


class ModelDetector:
    """Automatically detect which backend to use based on model name."""

    # Known model patterns for auto-detection
    OPENAI_PATTERNS = [
        'gpt-', 'gpt3', 'gpt4', 'text-davinci', 'text-curie', 'text-babbage',
        'text-ada', 'davinci', 'curie', 'babbage', 'ada'
    ]

    COHERE_PATTERNS = [
        'command', 'coral', 'aya'
    ]

    MT0_PATTERNS = [
        'mt0', 'bigscience/mt0'
    ]

    # Default models for each backend
    DEFAULT_MODELS = {
        'open_ai': 'gpt-4o',
        'cohere': 'command-r-03-2025',
        'hf_llm': 'meta-llama/Llama-3.3-70B-Instruct',
        'mt0': 'bigscience/mt0-xxl'
    }

    @classmethod
    def detect_backend(cls, model_name: str) -> Tuple[str, str]:
        """
        Detect backend from model name.

        Args:
            model_name: Model name or backend name

        Returns:
            Tuple of (backend_name, actual_model_name)
        """
        model_lower = model_name.lower()

        # Check if it's already a backend name
        if model_name in ['open_ai', 'cohere', 'hf_llm', 'mt0']:
            return model_name, cls.DEFAULT_MODELS[model_name]

        # Check OpenAI patterns
        for pattern in cls.OPENAI_PATTERNS:
            if pattern in model_lower:
                return 'open_ai', model_name

        # Check Cohere patterns
        for pattern in cls.COHERE_PATTERNS:
            if pattern in model_lower:
                return 'cohere', model_name

        # Check MT0 patterns
        for pattern in cls.MT0_PATTERNS:
            if pattern in model_lower:
                return 'mt0', model_name

        # Default to HuggingFace for anything else (including model paths with /)
        # Most custom models are HuggingFace models
        return 'hf_llm', model_name


class BenchmarkRunner:
    """Unified benchmark runner for all SinGen tasks."""

    VALID_QUERY_TYPES = ['zero-shot', 'zero-shot-si', 'few-shot', 'few-shot-si']
    VALID_TASKS = ['simplification', 'summarization', 'headline', 'translation']
    VALID_TRANSLATION_PAIRS = ['en_si', 'ta_si', 'pi_si']

    # Map task names to module paths
    TASK_MODULES = {
        'simplification': 'text_simplification',
        'summarization': 'text_summerisation',
        'headline': 'headline_generation',
    }

    TRANSLATION_MODULE_PREFIX = 'machine_translation'

    def __init__(self, backend: str, model_name: str, query_type: str,
                 tasks: List[str], translation_pairs: List[str]):
        """
        Initialize the benchmark runner.

        Args:
            backend: Backend name (open_ai, cohere, hf_llm, mt0)
            model_name: Actual model name to use
            query_type: Query type for benchmarks
            tasks: List of tasks to run
            translation_pairs: List of translation pairs to run
        """
        self.backend = backend
        self.model_name = model_name
        self.query_type = query_type
        self.tasks = tasks
        self.translation_pairs = translation_pairs
        self.results: Dict[str, bool] = {}

    def run_command(self, command: List[str], task_name: str) -> bool:
        """
        Run a benchmark command and track results.

        Args:
            command: Command to execute
            task_name: Name of the task for logging

        Returns:
            True if successful, False otherwise
        """
        print(f"\n{'='*80}")
        print(f"Running: {task_name}")
        print(f"Command: {' '.join(command)}")
        print(f"{'='*80}\n")

        try:
            result = subprocess.run(
                command,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            print(result.stdout)
            print(f"\n✓ {task_name} completed successfully!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\n✗ {task_name} failed with error:")
            print(e.stdout)
            return False
        except Exception as e:
            print(f"\n✗ {task_name} failed with exception: {e}")
            return False

    def run_text_simplification(self):
        """Run text simplification benchmark."""
        task_name = f"Text Simplification ({self.model_name}, {self.query_type})"
        module = f"{self.TASK_MODULES['simplification']}.{self.backend}"
        command = ['python', '-m', module,
                   f'--query_type={self.query_type}',
                   f'--model_name={self.model_name}']
        self.results[task_name] = self.run_command(command, task_name)

    def run_text_summarization(self):
        """Run text summarization benchmark."""
        task_name = f"Text Summarization ({self.model_name}, {self.query_type})"
        module = f"{self.TASK_MODULES['summarization']}.{self.backend}"
        command = ['python', '-m', module,
                   f'--query_type={self.query_type}',
                   f'--model_name={self.model_name}']
        self.results[task_name] = self.run_command(command, task_name)

    def run_headline_generation(self):
        """Run headline generation benchmark."""
        task_name = f"Headline Generation ({self.model_name}, {self.query_type})"
        module = f"{self.TASK_MODULES['headline']}.{self.backend}"
        command = ['python', '-m', module,
                   f'--query_type={self.query_type}',
                   f'--model_name={self.model_name}']
        self.results[task_name] = self.run_command(command, task_name)

    def run_machine_translation(self):
        """Run machine translation benchmarks for all specified language pairs."""
        for pair in self.translation_pairs:
            task_name = f"Machine Translation {pair.upper().replace('_', '-')} ({self.model_name}, {self.query_type})"
            module = f"{self.TRANSLATION_MODULE_PREFIX}.{pair}.{self.backend}"
            command = ['python', '-m', module,
                       f'--query_type={self.query_type}',
                       f'--model_name={self.model_name}']
            self.results[task_name] = self.run_command(command, task_name)

    def run_all(self):
        """Run all specified benchmarks."""
        start_time = datetime.now()

        print(f"\n{'#'*80}")
        print(f"# SinGen Unified Benchmark Runner")
        print(f"# Model: {self.model_name}")
        print(f"# Backend: {self.backend}")
        print(f"# Query Type: {self.query_type}")
        print(f"# Tasks: {', '.join(self.tasks)}")
        if 'translation' in self.tasks:
            print(f"# Translation Pairs: {', '.join(self.translation_pairs)}")
        print(f"# Start Time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*80}\n")

        # Run each specified task
        if 'simplification' in self.tasks:
            self.run_text_simplification()

        if 'summarization' in self.tasks:
            self.run_text_summarization()

        if 'headline' in self.tasks:
            self.run_headline_generation()

        if 'translation' in self.tasks:
            self.run_machine_translation()

        # Print summary
        end_time = datetime.now()
        duration = end_time - start_time

        print(f"\n\n{'#'*80}")
        print(f"# Benchmark Summary")
        print(f"{'#'*80}\n")

        successful = sum(1 for v in self.results.values() if v)
        total = len(self.results)

        for task, success in self.results.items():
            status = "✓ PASS" if success else "✗ FAIL"
            print(f"{status}: {task}")

        print(f"\n{'='*80}")
        print(f"Total: {successful}/{total} tasks completed successfully")
        print(f"Duration: {duration}")
        print(f"End Time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        # Exit with error code if any task failed
        if successful < total:
            sys.exit(1)


def main():
    """Main entry point for the unified benchmark runner."""
    parser = argparse.ArgumentParser(
        description='Run all SinGen benchmarks with a single command',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--model',
        type=str,
        required=False,
        help='Backend name (open_ai, cohere, hf_llm, mt0) OR model name for auto-detection'
    )

    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        help='Actual model name (e.g., meta-llama/Meta-Llama-3-8B-Instruct, gpt-4o, command-r)'
    )

    parser.add_argument(
        '--query_type',
        type=str,
        required=True,
        choices=BenchmarkRunner.VALID_QUERY_TYPES,
        help='Query type for benchmarks'
    )

    parser.add_argument(
        '--tasks',
        type=str,
        default='all',
        help='Comma-separated list of tasks to run (default: all). Options: simplification, summarization, headline, translation'
    )

    parser.add_argument(
        '--translation_pairs',
        type=str,
        default='all',
        help='Comma-separated list of translation pairs (default: all). Options: en_si, ta_si, pi_si'
    )

    args = parser.parse_args()

    # Determine backend and model name from arguments
    VALID_BACKENDS = ['open_ai', 'cohere', 'hf_llm', 'mt0']

    if not args.model and not args.model_name:
        print("Error: At least one of --model or --model_name must be provided")
        sys.exit(1)

    if args.model and args.model_name:
        # Both provided: use model as backend, model_name as model
        if args.model in VALID_BACKENDS:
            backend = args.model
            model_name = args.model_name
            print(f"Using backend: {backend}")
            print(f"Using model: {model_name}\n")
        else:
            print(f"Error: When both --model and --model_name are provided, --model must be a valid backend")
            print(f"Valid backends: {', '.join(VALID_BACKENDS)}")
            sys.exit(1)
    elif args.model_name:
        # Only model_name provided: auto-detect backend
        backend, detected_model = ModelDetector.detect_backend(args.model_name)
        model_name = args.model_name
        print(f"Auto-detected backend: {backend}")
        print(f"Using model: {model_name}\n")
    else:
        # Only model provided: detect if it's backend or model name
        if args.model in VALID_BACKENDS:
            # It's a backend name, use default model
            backend = args.model
            model_name = ModelDetector.DEFAULT_MODELS[backend]
            print(f"Using backend: {backend}")
            print(f"Using default model: {model_name}\n")
        else:
            # It's a model name, auto-detect backend
            backend, model_name = ModelDetector.detect_backend(args.model)
            print(f"Auto-detected backend: {backend}")
            print(f"Using model: {model_name}\n")

    # Parse tasks
    if args.tasks.lower() == 'all':
        tasks = BenchmarkRunner.VALID_TASKS
    else:
        tasks = [t.strip() for t in args.tasks.split(',')]
        # Validate tasks
        invalid_tasks = set(tasks) - set(BenchmarkRunner.VALID_TASKS)
        if invalid_tasks:
            print(f"Error: Invalid tasks: {', '.join(invalid_tasks)}")
            print(f"Valid tasks are: {', '.join(BenchmarkRunner.VALID_TASKS)}")
            sys.exit(1)

    # Parse translation pairs
    if args.translation_pairs.lower() == 'all':
        translation_pairs = BenchmarkRunner.VALID_TRANSLATION_PAIRS
    else:
        translation_pairs = [p.strip() for p in args.translation_pairs.split(',')]
        # Validate translation pairs
        invalid_pairs = set(translation_pairs) - set(BenchmarkRunner.VALID_TRANSLATION_PAIRS)
        if invalid_pairs:
            print(f"Error: Invalid translation pairs: {', '.join(invalid_pairs)}")
            print(f"Valid pairs are: {', '.join(BenchmarkRunner.VALID_TRANSLATION_PAIRS)}")
            sys.exit(1)

    # Create and run benchmark runner
    runner = BenchmarkRunner(backend, model_name, args.query_type, tasks, translation_pairs)
    runner.run_all()


if __name__ == '__main__':
    main()
