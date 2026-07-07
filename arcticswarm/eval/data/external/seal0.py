#!/usr/bin/env python3
"""
SEAL-0 benchmark loader for Arcticswarm evaluation.

SealQA: Raising the Bar for Reasoning in Search-Augmented Language Models
Source: https://huggingface.co/datasets/vtllms/sealqa
Paper:  https://arxiv.org/pdf/2506.01062

SealQA is a challenge benchmark for search-augmented LLMs on fact-seeking
questions where web search returns *conflicting, noisy, or unhelpful* results.
It ships three ungated configs (each a single ``test`` split):

    seal_0     111 rows  — the hardest subset; chat models (e.g. GPT-4.1)
                           score near-zero. THIS loader targets seal_0.
    seal_hard  254 rows
    longseal   254 rows

Unlike BrowseComp / EvoBrowseComp, SealQA rows are stored in **plaintext** —
no SHA256 password or XOR one-time-pad decryption is needed. Each row carries a
short-answer ``question`` / ``answer`` pair plus rich metadata (``topic``,
``freshness``, ``question_types``, ``effective_year``, reference ``urls`` and
``golds``). We preserve the lightweight metadata as per-row attributes and drop
the bulky pre-retrieved ``*_docs`` corpora (they are only for the paper's
oracle/RAG conditions, not for a live-search agent).

SEAL-0 is graded with a binary correct/incorrect QA judge, matching the paper's
model-based grading.

Usage:
    # Generate seal0_v1.csv (all 111 cases)
    python -m arcticswarm.eval.data.external.seal0 \\
        --output arcticswarm/eval/data/seal0_v1.csv

    # Run evaluation
    arcticswarm-eval \\
        --csv-path arcticswarm/eval/data/seal0_v1.csv \\
        --datasets SEAL0_V1 \\
        --output results/seal0/
"""

import argparse
import sys
from pathlib import Path

from .utils import write_unified_eval_csv

SEAL0_HF_DATASET = "vtllms/sealqa"
SEAL0_HF_CONFIG = "seal_0"
SEAL0_HF_SPLIT = "test"

# Lightweight per-row metadata to preserve as attributes. The bulky pre-retrieved
# ``12_docs`` / ``20_docs`` / ``30_docs`` corpora and the ``canary`` string are
# deliberately excluded — they are not used by a live-search agent.
_METADATA_FIELDS = (
    "topic",
    "freshness",
    "question_types",
    "effective_year",
    "search_results",
    "urls",
    "golds",
)


def load_seal0(
    dataset: str = SEAL0_HF_DATASET,
    config: str = SEAL0_HF_CONFIG,
    split: str = SEAL0_HF_SPLIT,
) -> list[dict]:
    """Download the SEAL-0 subset of SealQA.

    Args:
        dataset: HuggingFace dataset id (default: ``vtllms/sealqa``).
        config: Dataset config / subset (default: ``seal_0`` — the 111 hardest).
        split: Dataset split to load (default: ``test``).

    Returns:
        List of dicts with keys: ``id``, ``question``, ``expected_answer``,
        ``attributes`` (carries per-row SealQA metadata + ``source_index``).
    """
    print(f"📥 Downloading SealQA from HuggingFace ({dataset}, config={config}, split={split})...")
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    print(f"Found {len(ds)} examples")

    examples: list[dict] = []
    for source_index, row in enumerate(ds):
        question = (row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            print(f"⚠️  Warning: skipping row {source_index} with empty question/answer", file=sys.stderr)
            continue

        attributes = {"source_index": source_index}
        for key in _METADATA_FIELDS:
            if key in row and row[key] is not None:
                attributes[key] = row[key]

        examples.append({
            "id": f"seal0_{source_index:03d}",
            "question": question,
            "expected_answer": answer,
            "attributes": attributes,
        })

    print(f"✅ Loaded {len(examples)} SEAL-0 examples")
    return examples


def convert_seal0_to_csv(
    examples: list[dict],
    output_path: Path,
    dataset_name: str = "SEAL0_V1",
) -> None:
    """Convert SEAL-0 examples to Arcticswarm's unified eval CSV format.

    Args:
        examples: Output from :func:`load_seal0`.
        output_path: Path to output CSV.
        dataset_name: Dataset identifier in the ATTRIBUTES field.
    """
    write_unified_eval_csv(
        examples,
        output_path,
        dataset_name=dataset_name,
        eval_mode="QA",  # SEAL-0 uses a binary correct/incorrect judge
        reference_tools=["web_search"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load and convert the SEAL-0 (SealQA) benchmark to Arcticswarm format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate the SEAL-0 CSV (111 cases)
  python -m arcticswarm.eval.data.external.seal0 --output arcticswarm/eval/data/seal0_v1.csv

  # Limit to 10 examples for testing
  python -m arcticswarm.eval.data.external.seal0 --output test.csv --limit 10

  # Run evaluation
  arcticswarm-eval --csv-path arcticswarm/eval/data/seal0_v1.csv --datasets SEAL0_V1 --output results/
        """
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., arcticswarm/eval/data/seal0_v1.csv)"
    )
    parser.add_argument(
        "--config",
        default=SEAL0_HF_CONFIG,
        help=f"HuggingFace dataset config/subset (default: {SEAL0_HF_CONFIG})"
    )
    parser.add_argument(
        "--dataset",
        default=SEAL0_HF_DATASET,
        help=f"HuggingFace dataset id (default: {SEAL0_HF_DATASET})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    examples = load_seal0(dataset=args.dataset, config=args.config)

    if args.limit:
        print(f"⚙️  Limiting to first {args.limit} examples")
        examples = examples[:args.limit]

    convert_seal0_to_csv(examples, args.output)

    print(f"\n✅ Done! Generated {args.output}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the output:")
    print(f"   head -5 {args.output}")
    print(f"\n2. Run evaluation:")
    print(f"   arcticswarm-eval \\")
    print(f"     --csv-path {args.output} \\")
    print(f"     --datasets SEAL0_V1 \\")
    print(f"     --output results/seal0/")


if __name__ == "__main__":
    main()
