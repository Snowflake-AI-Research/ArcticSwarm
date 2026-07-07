#!/usr/bin/env python3
"""
K-BrowseComp benchmark loader for Arcticswarm evaluation.

K-BrowseComp: A Korean Benchmark for Browsing Agents
Source: https://github.com/prometheus-eval/K-BrowseComp
Dataset: https://huggingface.co/datasets/prometheus-eval/k-browsecomp
Paper:  https://arxiv.org/pdf/2606.02404

K-BrowseComp is the Korean-language counterpart of OpenAI's BrowseComp: hard,
fact-seeking questions that require multi-hop / parallel web browsing and are
written + validated by native speakers. It ships two ungated configs (each a
single ``test`` split):

    verified   300 rows  — native-speaker handcrafted & validated (THIS loader's
                           default target).
    synthetic  100 rows  — synthetic diagnostic subset.

Unlike OpenAI BrowseComp, the released K-BrowseComp rows are stored in
**plaintext** — no SHA256 password / XOR one-time-pad decryption is needed. Each
row carries a Korean ``problem`` / short ``answer`` pair plus ``type``
("multi-hop" / "parallel") and ``category`` (topic class), which we preserve as
per-row attributes.

K-BrowseComp is graded with a binary correct/incorrect LLM judge (SimpleQA/HLE
style, ``correct: yes|no``) using the authors' Korean-aware grader that tolerates
harmless surface-form variants (Hanja, romanization, transliteration) but marks
disjunctions ("또는", "or", "/") as incorrect.

Usage:
    # Generate kbrowsecomp_v1.csv (300 verified cases)
    python -m arcticswarm.eval.data.external.kbrowsecomp \\
        --output arcticswarm/eval/data/kbrowsecomp_v1.csv

    # Run evaluation
    arcticswarm-eval \\
        --csv-path arcticswarm/eval/data/kbrowsecomp_v1.csv \\
        --datasets KBROWSECOMP_V1 \\
        --output results/kbrowsecomp/
"""

import argparse
import sys
from pathlib import Path

from .utils import write_unified_eval_csv

KBROWSECOMP_HF_DATASET = "prometheus-eval/k-browsecomp"
KBROWSECOMP_HF_CONFIG = "verified"
KBROWSECOMP_HF_SPLIT = "test"

# Lightweight per-row metadata to preserve as attributes.
_METADATA_FIELDS = (
    "type",
    "category",
)


def load_kbrowsecomp(
    dataset: str = KBROWSECOMP_HF_DATASET,
    config: str = KBROWSECOMP_HF_CONFIG,
    split: str = KBROWSECOMP_HF_SPLIT,
) -> list[dict]:
    """Download the K-BrowseComp dataset (plaintext, no decryption).

    Args:
        dataset: HuggingFace dataset id (default: ``prometheus-eval/k-browsecomp``).
        config: Dataset config / subset (default: ``verified`` — the 300 handcrafted).
        split: Dataset split to load (default: ``test``).

    Returns:
        List of dicts with keys: ``id``, ``question``, ``expected_answer``,
        ``attributes`` (carries per-row ``type`` / ``category`` + ``source_index``).
    """
    print(f"📥 Downloading K-BrowseComp from HuggingFace ({dataset}, config={config}, split={split})...")
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    print(f"Found {len(ds)} examples")

    examples: list[dict] = []
    for source_index, row in enumerate(ds):
        # The HF schema uses ``problem`` / ``answer`` (matching the JSONL).
        question = (row.get("problem") or row.get("question") or "").strip()
        answer = (row.get("answer") or "").strip()
        if not question or not answer:
            print(f"⚠️  Warning: skipping row {source_index} with empty question/answer", file=sys.stderr)
            continue

        attributes = {"source_index": source_index}
        for key in _METADATA_FIELDS:
            if key in row and row[key] is not None:
                attributes[key] = row[key]

        examples.append({
            "id": f"kbrowsecomp_{source_index:03d}",
            "question": question,
            "expected_answer": answer,
            "attributes": attributes,
        })

    print(f"✅ Loaded {len(examples)} K-BrowseComp examples")
    return examples


def convert_kbrowsecomp_to_csv(
    examples: list[dict],
    output_path: Path,
    dataset_name: str = "KBROWSECOMP_V1",
) -> None:
    """Convert K-BrowseComp examples to Arcticswarm's unified eval CSV format.

    Args:
        examples: Output from :func:`load_kbrowsecomp`.
        output_path: Path to output CSV.
        dataset_name: Dataset identifier in the ATTRIBUTES field.
    """
    write_unified_eval_csv(
        examples,
        output_path,
        dataset_name=dataset_name,
        eval_mode="QA",  # K-BrowseComp uses a binary correct/incorrect judge
        reference_tools=["web_search"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load and convert the K-BrowseComp benchmark to Arcticswarm format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate the K-BrowseComp CSV (300 verified cases)
  python -m arcticswarm.eval.data.external.kbrowsecomp --output arcticswarm/eval/data/kbrowsecomp_v1.csv

  # Use the synthetic diagnostic subset (100 cases)
  python -m arcticswarm.eval.data.external.kbrowsecomp --output kbc_synth.csv --config synthetic

  # Limit to 10 examples for testing
  python -m arcticswarm.eval.data.external.kbrowsecomp --output test.csv --limit 10

  # Run evaluation
  arcticswarm-eval --csv-path arcticswarm/eval/data/kbrowsecomp_v1.csv --datasets KBROWSECOMP_V1 --output results/
        """
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., arcticswarm/eval/data/kbrowsecomp_v1.csv)"
    )
    parser.add_argument(
        "--config",
        default=KBROWSECOMP_HF_CONFIG,
        help=f"HuggingFace dataset config/subset: verified (300) or synthetic (100) (default: {KBROWSECOMP_HF_CONFIG})"
    )
    parser.add_argument(
        "--dataset",
        default=KBROWSECOMP_HF_DATASET,
        help=f"HuggingFace dataset id (default: {KBROWSECOMP_HF_DATASET})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    examples = load_kbrowsecomp(dataset=args.dataset, config=args.config)

    if args.limit:
        print(f"⚙️  Limiting to first {args.limit} examples")
        examples = examples[:args.limit]

    convert_kbrowsecomp_to_csv(examples, args.output)

    print(f"\n✅ Done! Generated {args.output}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the output:")
    print(f"   head -5 {args.output}")
    print(f"\n2. Run evaluation:")
    print(f"   arcticswarm-eval \\")
    print(f"     --csv-path {args.output} \\")
    print(f"     --datasets KBROWSECOMP_V1 \\")
    print(f"     --output results/kbrowsecomp/")


if __name__ == "__main__":
    main()
