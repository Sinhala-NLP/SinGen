#!/usr/bin/env python3
"""
Script to add --model_name parameter to all benchmark task scripts.
This allows scripts to accept custom model names instead of using hardcoded values.
"""

import os
import re
from pathlib import Path


def update_hf_llm_script(filepath):
    """Update HuggingFace LLM scripts to accept --model_name parameter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already updated
    if '--model_name' in content or 'args.model_name' in content:
        print(f"  ✓ Already updated: {filepath}")
        return False

    # Replace hardcoded model_id with a variable that can be set from args
    # Find the line like: model_id = "meta-llama/Llama-3.3-70B-Instruct"
    pattern = r'model_id = "([^"]+)"'
    match = re.search(pattern, content)

    if not match:
        print(f"  ✗ Could not find model_id in: {filepath}")
        return False

    default_model = match.group(1)

    # Replace the hardcoded model_id with a placeholder
    content = re.sub(
        pattern,
        'model_id = MODEL_NAME  # Set from command line argument',
        content,
        count=1
    )

    # Add MODEL_NAME as a global variable near the top, after imports
    # Find the argparse section and add model_name argument
    parser_pattern = r"(parser\.add_argument\('--query_type'[^)]+\))"
    replacement = r"""\1
    parser.add_argument('--model_name', type=str, default='{}', required=False,
                        help='Model name or path (default: {})')""".format(default_model, default_model)

    content = re.sub(parser_pattern, replacement, content)

    # Add MODEL_NAME assignment after QUERY_TYPE
    query_type_pattern = r"(QUERY_TYPE = args\.query_type)"
    replacement = r"""\1
    MODEL_NAME = args.model_name
    print(f"Model: {MODEL_NAME}")"""

    content = re.sub(query_type_pattern, replacement, content)

    # Update OUTPUT_FOLDER to use MODEL_NAME
    output_folder_pattern = r'OUTPUT_FOLDER = os\.path\.join\("outputs", "([^"]+)", model_id\.split\(\'/\'\)\[-1\], QUERY_TYPE\)'

    if re.search(output_folder_pattern, content):
        content = re.sub(
            output_folder_pattern,
            r'OUTPUT_FOLDER = os.path.join("outputs", "\1", MODEL_NAME.split(\'/\')[-1], QUERY_TYPE)',
            content
        )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated: {filepath}")
    return True


def update_open_ai_script(filepath):
    """Update OpenAI scripts to accept --model_name parameter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already updated
    if '--model_name' in content or 'args.model_name' in content:
        print(f"  ✓ Already updated: {filepath}")
        return False

    # Find the model_id line
    pattern = r'model_id = "([^"]+)"'
    match = re.search(pattern, content)

    if not match:
        print(f"  ✗ Could not find model_id in: {filepath}")
        return False

    default_model = match.group(1)

    # Replace the hardcoded model_id
    content = re.sub(
        pattern,
        'model_id = MODEL_NAME  # Set from command line argument',
        content,
        count=1
    )

    # Add model_name argument to parser
    parser_pattern = r"(parser\.add_argument\('--query_type'[^)]+\))"
    replacement = r"""\1
    parser.add_argument('--model_name', type=str, default='{}', required=False,
                        help='Model name (default: {})')""".format(default_model, default_model)

    content = re.sub(parser_pattern, replacement, content)

    # Add MODEL_NAME assignment
    query_type_pattern = r"(QUERY_TYPE = args\.query_type)"
    replacement = r"""\1
    MODEL_NAME = args.model_name
    print(f"Model: {MODEL_NAME}")"""

    content = re.sub(query_type_pattern, replacement, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated: {filepath}")
    return True


def update_cohere_script(filepath):
    """Update Cohere scripts to accept --model_name parameter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already updated
    if '--model_name' in content or 'args.model_name' in content:
        print(f"  ✓ Already updated: {filepath}")
        return False

    # Find the model_id line
    pattern = r'model_id = "([^"]+)"'
    match = re.search(pattern, content)

    if not match:
        print(f"  ✗ Could not find model_id in: {filepath}")
        return False

    default_model = match.group(1)

    # Replace the hardcoded model_id
    content = re.sub(
        pattern,
        'model_id = MODEL_NAME  # Set from command line argument',
        content,
        count=1
    )

    # Add model_name argument to parser
    parser_pattern = r"(parser\.add_argument\('--query_type'[^)]+\))"
    replacement = r"""\1
    parser.add_argument('--model_name', type=str, default='{}', required=False,
                        help='Model name (default: {})')""".format(default_model, default_model)

    content = re.sub(parser_pattern, replacement, content)

    # Add MODEL_NAME assignment
    query_type_pattern = r"(QUERY_TYPE = args\.query_type)"
    replacement = r"""\1
    MODEL_NAME = args.model_name
    print(f"Model: {MODEL_NAME}")"""

    content = re.sub(query_type_pattern, replacement, content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated: {filepath}")
    return True


def update_mt0_script(filepath):
    """Update MT0 scripts to accept --model_name parameter."""
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if already updated
    if '--model_name' in content or 'args.model_name' in content:
        print(f"  ✓ Already updated: {filepath}")
        return False

    # Find the checkpoint line (MT0 uses 'checkpoint' instead of 'model_id')
    pattern = r'checkpoint = "([^"]+)"'
    match = re.search(pattern, content)

    if not match:
        print(f"  ✗ Could not find checkpoint in: {filepath}")
        return False

    default_model = match.group(1)

    # Replace the hardcoded checkpoint
    content = re.sub(
        pattern,
        'checkpoint = MODEL_NAME  # Set from command line argument',
        content,
        count=1
    )

    # Add model_name argument to parser
    parser_pattern = r"(parser\.add_argument\('--query_type'[^)]+\))"
    replacement = r"""\1
    parser.add_argument('--model_name', type=str, default='{}', required=False,
                        help='Model name or path (default: {})')""".format(default_model, default_model)

    content = re.sub(parser_pattern, replacement, content)

    # Add MODEL_NAME assignment
    query_type_pattern = r"(QUERY_TYPE = args\.query_type)"
    replacement = r"""\1
    MODEL_NAME = args.model_name
    print(f"Model: {MODEL_NAME}")"""

    content = re.sub(query_type_pattern, replacement, content)

    # Update OUTPUT_FOLDER if present (MT0 uses checkpoint)
    output_folder_pattern = r'OUTPUT_FOLDER = os\.path\.join\("outputs", "([^"]+)", checkpoint\.split\(\'/\'\)\[-1\], QUERY_TYPE\)'

    if re.search(output_folder_pattern, content):
        content = re.sub(
            output_folder_pattern,
            r'OUTPUT_FOLDER = os.path.join("outputs", "\1", MODEL_NAME.split(\'/\')[-1], QUERY_TYPE)',
            content
        )

    # Update references to checkpoint in f-strings to use MODEL_NAME
    content = re.sub(r'f"Model: \{checkpoint\}', r'f"Model: {MODEL_NAME}', content)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)

    print(f"  ✓ Updated: {filepath}")
    return True


def main():
    """Main function to update all scripts."""
    base_path = Path('/home/user/SinGen')

    # Find all scripts
    hf_llm_files = list(base_path.glob('**/hf_llm.py'))
    open_ai_files = list(base_path.glob('**/open_ai.py'))
    cohere_files = list(base_path.glob('**/cohere.py'))
    mt0_files = list(base_path.glob('**/mt0.py'))

    print("=" * 80)
    print("Adding --model_name parameter to all benchmark scripts")
    print("=" * 80)

    updated_count = 0

    print("\nUpdating HuggingFace LLM scripts...")
    for filepath in hf_llm_files:
        if update_hf_llm_script(filepath):
            updated_count += 1

    print("\nUpdating OpenAI scripts...")
    for filepath in open_ai_files:
        if update_open_ai_script(filepath):
            updated_count += 1

    print("\nUpdating Cohere scripts...")
    for filepath in cohere_files:
        if update_cohere_script(filepath):
            updated_count += 1

    print("\nUpdating MT0 scripts...")
    for filepath in mt0_files:
        if update_mt0_script(filepath):
            updated_count += 1

    print("\n" + "=" * 80)
    print(f"Update complete! Modified {updated_count} files.")
    print("=" * 80)


if __name__ == '__main__':
    main()
