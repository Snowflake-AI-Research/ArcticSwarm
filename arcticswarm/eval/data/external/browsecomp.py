#!/usr/bin/env python3
"""
BrowseComp benchmark loader for Arcticswarm evaluation.

BrowseComp: A Simple Yet Challenging Benchmark for Browsing Agents
Authors: Jason Wei, Zhiqing Sun, Spencer Papay, et al.
Source: https://openai.com/index/browsecomp/

Usage:
    # Generate browsecomp_v1.csv
    python -m arcticswarm.eval.data.external.browsecomp \\
        --output arcticswarm/eval/data/browsecomp_v1.csv

    # Run evaluation
    arcticswarm-eval \\
        --csv-path arcticswarm/eval/data/browsecomp_v1.csv \\
        --datasets BROWSECOMP_V1 \\
        --output results/browsecomp/
"""

import argparse
import base64
import hashlib
import sys
from pathlib import Path

import pandas as pd

from .utils import write_unified_eval_csv

BROWSECOMP_URL = "https://openaipublic.blob.core.windows.net/simple-evals/browse_comp_test_set.csv"


def derive_key(password: str, length: int) -> bytes:
    """Derive a fixed-length key from the password using SHA256."""
    hasher = hashlib.sha256()
    hasher.update(password.encode())
    key = hasher.digest()
    return key * (length // len(key)) + key[: length % len(key)]


def decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt base64-encoded ciphertext with XOR cipher."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = derive_key(password, len(encrypted))
    decrypted = bytes(a ^ b for a, b in zip(encrypted, key))
    return decrypted.decode()


def load_browsecomp(url: str = BROWSECOMP_URL) -> list[dict]:
    """Download and decrypt BrowseComp dataset.

    Args:
        url: URL to BrowseComp CSV (default: OpenAI public blob)

    Returns:
        List of dicts with keys: 'id', 'question', 'expected_answer', 'canary'
    """
    print(f"📥 Downloading BrowseComp from {url}...")
    df = pd.read_csv(url)

    print(f"Found {len(df)} encrypted examples")
    print("🔓 Decrypting...")

    decrypted = []
    for idx, row in df.iterrows():
        try:
            problem = decrypt(row["problem"], row["canary"])
            answer = decrypt(row["answer"], row["canary"])
            decrypted.append({
                "id": f"browsecomp_{idx:03d}",
                "question": problem,
                "expected_answer": answer,
                "canary": row["canary"]  # Keep for reference
            })
        except Exception as e:
            print(f"⚠️  Warning: Failed to decrypt row {idx}: {e}", file=sys.stderr)
            continue

    print(f"✅ Successfully decrypted {len(decrypted)} examples")
    return decrypted


def convert_browsecomp_to_csv(
    examples: list[dict],
    output_path: Path,
    dataset_name: str = "BROWSECOMP_V1",
) -> None:
    """Convert decrypted BrowseComp to Arcticswarm CSV format.

    Args:
        examples: Output from load_browsecomp()
        output_path: Path to output CSV
        dataset_name: Dataset identifier in ATTRIBUTES field
    """
    # Convert to format expected by write_unified_eval_csv
    formatted = [
        {
            "id": ex["id"],
            "question": ex["question"],
            "expected_answer": ex["expected_answer"],
        }
        for ex in examples
    ]

    write_unified_eval_csv(
        formatted,
        output_path,
        dataset_name=dataset_name,
        eval_mode="QA",  # BrowseComp uses binary correct/incorrect
        reference_tools=["web_search"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load and convert BrowseComp benchmark to Arcticswarm format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate BrowseComp CSV
  python -m arcticswarm.eval.data.external.browsecomp --output arcticswarm/eval/data/browsecomp_v1.csv

  # Limit to 10 examples for testing
  python -m arcticswarm.eval.data.external.browsecomp --output test.csv --limit 10

  # Run evaluation
  arcticswarm-eval --csv-path arcticswarm/eval/data/browsecomp_v1.csv --datasets BROWSECOMP_V1 --output results/
        """
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., arcticswarm/eval/data/browsecomp_v1.csv)"
    )
    parser.add_argument(
        "--url",
        default=BROWSECOMP_URL,
        help="BrowseComp CSV URL (default: OpenAI public blob)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    # Download and decrypt
    examples = load_browsecomp(args.url)

    if args.limit:
        print(f"⚙️  Limiting to first {args.limit} examples")
        examples = examples[:args.limit]

    # Convert to Arcticswarm format
    convert_browsecomp_to_csv(examples, args.output)

    print(f"\n✅ Done! Generated {args.output}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the output:")
    print(f"   head -20 {args.output}")
    print(f"\n2. Run evaluation:")
    print(f"   arcticswarm-eval \\")
    print(f"     --csv-path {args.output} \\")
    print(f"     --datasets BROWSECOMP_V1 \\")
    print(f"     --output results/browsecomp/")


if __name__ == "__main__":
    main()
