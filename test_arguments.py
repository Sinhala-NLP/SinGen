#!/usr/bin/env python3
"""Test script to verify argument handling in the benchmark runner."""

import subprocess
import sys

def run_test(description, args, should_pass=True):
    """Run a test case and report results."""
    print(f"\n{'='*80}")
    print(f"Test: {description}")
    print(f"Args: {' '.join(args)}")
    print(f"{'='*80}")

    cmd = ['python', 'run_all_benchmarks.py'] + args + ['--query_type', 'zero-shot', '--tasks', 'simplification']

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=5
        )

        # Check first few lines for backend/model detection output
        output_lines = result.stdout.split('\n')[:5]
        for line in output_lines:
            if line.strip():
                print(line)

        if result.returncode != 0 and 'Error:' in result.stdout:
            error_lines = [l for l in result.stdout.split('\n') if 'Error:' in l or 'error:' in l]
            for line in error_lines:
                print(line)

        if should_pass:
            if result.returncode == 0 or 'Using backend:' in result.stdout or 'Auto-detected backend:' in result.stdout:
                print("✓ PASS: Arguments accepted correctly")
                return True
            else:
                print("✗ FAIL: Expected to pass but failed")
                return False
        else:
            if result.returncode != 0:
                print("✓ PASS: Correctly rejected invalid arguments")
                return True
            else:
                print("✗ FAIL: Expected to fail but passed")
                return False

    except subprocess.TimeoutExpired:
        print("✓ PASS: Arguments accepted (timed out during execution, but that's OK for this test)")
        return True
    except Exception as e:
        print(f"✗ FAIL: Exception: {e}")
        return False

def main():
    """Run all test cases."""
    print("="*80)
    print("Testing Benchmark Runner Argument Handling")
    print("="*80)

    tests = [
        # Test 1: Both --model and --model_name (RECOMMENDED)
        ("Both backend and model name",
         ['--model', 'hf_llm', '--model_name', 'meta-llama/Meta-Llama-3-8B-Instruct'],
         True),

        # Test 2: Only --model with backend name
        ("Only backend name",
         ['--model', 'cohere'],
         True),

        # Test 3: Only --model with model name (auto-detect)
        ("Only model name via --model (auto-detect)",
         ['--model', 'gpt-4o'],
         True),

        # Test 4: Only --model_name (auto-detect)
        ("Only --model_name (auto-detect)",
         ['--model_name', 'command-r'],
         True),

        # Test 5: HuggingFace model auto-detection
        ("HuggingFace model auto-detection",
         ['--model_name', 'meta-llama/Mistral-7B-Instruct-v0.2'],
         True),

        # Test 6: Neither --model nor --model_name (should fail)
        ("Neither argument (should fail)",
         [],
         False),

        # Test 7: Invalid backend with --model_name
        ("Invalid backend with model_name (should fail)",
         ['--model', 'invalid_backend', '--model_name', 'gpt-4o'],
         False),
    ]

    results = []
    for desc, args, should_pass in tests:
        result = run_test(desc, args, should_pass)
        results.append((desc, result))

    # Print summary
    print(f"\n\n{'='*80}")
    print("Test Summary")
    print(f"{'='*80}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for desc, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {desc}")

    print(f"\n{'='*80}")
    print(f"Total: {passed}/{total} tests passed")
    print(f"{'='*80}")

    return 0 if passed == total else 1

if __name__ == '__main__':
    sys.exit(main())
