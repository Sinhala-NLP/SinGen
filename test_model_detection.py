#!/usr/bin/env python3
"""Test script to verify model backend detection."""

import sys
sys.path.insert(0, '/home/user/SinGen')

from run_all_benchmarks import ModelDetector

test_models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Llama-3.3-70B-Instruct",
    "gpt-4o",
    "gpt-3.5-turbo",
    "command-r",
    "command-r-plus",
    "bigscience/mt0-xxl",
    "mt0-large",
    "open_ai",
    "cohere",
    "hf_llm",
    "mt0",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "google/gemma-7b",
]

print("Testing Model Backend Detection")
print("=" * 80)

for model in test_models:
    backend, detected_model = ModelDetector.detect_backend(model)
    print(f"Input: {model:50} -> Backend: {backend:10} Model: {detected_model}")

print("=" * 80)
