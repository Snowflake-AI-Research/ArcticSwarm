#!/usr/bin/env python3
"""
EvoBrowseComp benchmark loader for Arcticswarm evaluation.

EvoBrowseComp: Benchmarking Search Agents on Evolving Knowledge
Authors: Yunhan Wang, Jiaan Wang, Lianzhe Huang, Xianfeng Zeng, Fandong Meng
Source: https://huggingface.co/datasets/Krystalan/EvoBrowseComp
Paper:  https://arxiv.org/pdf/2606.13120

The HuggingFace dataset bundles 800 examples — 400 English + 400 Chinese.
Each row carries an XOR-encrypted ``question`` / ``answer`` (base64) together
with a same-length base64 keystream (``question_key`` / ``answer_key``);
plaintext is recovered by XOR-ing the two decoded byte strings. This differs
from BrowseComp's SHA256-derived single-password scheme — EvoBrowseComp ships
the full one-time-pad keystream per field.

This loader keeps the English subset by default (the project only evaluates the
English questions), giving exactly 400 cases tagged ``EVOBROWSECOMP_V1``.

Usage:
    # Generate evobrowsecomp_v1.csv (English subset, 400 cases)
    python -m arcticswarm.eval.data.external.evobrowsecomp \\
        --output arcticswarm/eval/data/evobrowsecomp_v1.csv

    # Run evaluation
    arcticswarm-eval \\
        --csv-path arcticswarm/eval/data/evobrowsecomp_v1.csv \\
        --datasets EVOBROWSECOMP_V1 \\
        --output results/evobrowsecomp/
"""

import argparse
import base64
import sys
from pathlib import Path

from .utils import write_unified_eval_csv

EVOBROWSECOMP_HF_DATASET = "Krystalan/EvoBrowseComp"
EVOBROWSECOMP_HF_SPLIT = "train"


def decrypt(ciphertext_b64: str, key_b64: str) -> str:
    """Decrypt one EvoBrowseComp field.

    Both the ciphertext and the key are base64-encoded byte strings of equal
    length; the plaintext is their byte-wise XOR (a one-time pad). Returns the
    UTF-8 decoded plaintext.
    """
    ciphertext = base64.b64decode(ciphertext_b64)
    key = base64.b64decode(key_b64)
    return bytes(a ^ b for a, b in zip(ciphertext, key)).decode("utf-8")


def _is_english(text: str) -> bool:
    """Heuristic language gate: a question is treated as Chinese if it contains
    any CJK ideograph or Japanese kana, English otherwise.

    EvoBrowseComp is exactly bilingual (400 EN / 400 ZH) with no overlap, so a
    single CJK-character test partitions the set cleanly.
    """
    for ch in text:
        if "一" <= ch <= "鿿" or "぀" <= ch <= "ヿ":
            return False
    return True


def load_evobrowsecomp(
    dataset: str = EVOBROWSECOMP_HF_DATASET,
    split: str = EVOBROWSECOMP_HF_SPLIT,
    language: str = "en",
) -> list[dict]:
    """Download and decrypt the EvoBrowseComp dataset.

    Args:
        dataset: HuggingFace dataset id (default: ``Krystalan/EvoBrowseComp``).
        split: Dataset split to load (default: ``train``).
        language: ``"en"`` (default) keeps only English questions, ``"zh"``
            keeps only Chinese, ``"all"`` keeps both.

    Returns:
        List of dicts with keys: ``id``, ``question``, ``expected_answer``,
        ``attributes`` (carries per-row ``language`` and ``source_index``).
    """
    if language not in ("en", "zh", "all"):
        raise ValueError(f"language must be 'en', 'zh', or 'all', got {language!r}")

    print(f"📥 Downloading EvoBrowseComp from HuggingFace ({dataset}, split={split})...")
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split)
    print(f"Found {len(ds)} encrypted examples")
    print("🔓 Decrypting (XOR one-time pad)...")

    examples: list[dict] = []
    seq = 0
    skipped_lang = 0
    for source_index, row in enumerate(ds):
        try:
            question = decrypt(row["question"], row["question_key"])
            answer = decrypt(row["answer"], row["answer_key"])
        except Exception as e:  # noqa: BLE001 — log and skip an unparseable row
            print(f"⚠️  Warning: Failed to decrypt row {source_index}: {e}", file=sys.stderr)
            continue

        row_lang = "en" if _is_english(question) else "zh"
        if language != "all" and row_lang != language:
            skipped_lang += 1
            continue

        examples.append({
            "id": f"evobrowsecomp_{seq:03d}",
            "question": question,
            "expected_answer": answer,
            "attributes": {
                "language": row_lang,
                "source_index": source_index,
            },
        })
        seq += 1

    print(
        f"✅ Decrypted {len(examples)} examples "
        f"(language={language}, skipped {skipped_lang} for language filter)"
    )
    return examples


def convert_evobrowsecomp_to_csv(
    examples: list[dict],
    output_path: Path,
    dataset_name: str = "EVOBROWSECOMP_V1",
) -> None:
    """Convert decrypted EvoBrowseComp examples to Arcticswarm CSV format.

    Args:
        examples: Output from :func:`load_evobrowsecomp`.
        output_path: Path to output CSV.
        dataset_name: Dataset identifier in the ATTRIBUTES field.
    """
    write_unified_eval_csv(
        examples,
        output_path,
        dataset_name=dataset_name,
        eval_mode="QA",  # EvoBrowseComp uses a binary correct/incorrect judge
        reference_tools=["web_search"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load and convert the EvoBrowseComp benchmark to Arcticswarm format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate the English EvoBrowseComp CSV (400 cases)
  python -m arcticswarm.eval.data.external.evobrowsecomp --output arcticswarm/eval/data/evobrowsecomp_v1.csv

  # Limit to 10 examples for testing
  python -m arcticswarm.eval.data.external.evobrowsecomp --output test.csv --limit 10

  # Run evaluation
  arcticswarm-eval --csv-path arcticswarm/eval/data/evobrowsecomp_v1.csv --datasets EVOBROWSECOMP_V1 --output results/
        """
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., arcticswarm/eval/data/evobrowsecomp_v1.csv)"
    )
    parser.add_argument(
        "--language",
        choices=["en", "zh", "all"],
        default="en",
        help="Which language subset to keep (default: en — the 400 English questions)"
    )
    parser.add_argument(
        "--dataset",
        default=EVOBROWSECOMP_HF_DATASET,
        help=f"HuggingFace dataset id (default: {EVOBROWSECOMP_HF_DATASET})"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    examples = load_evobrowsecomp(dataset=args.dataset, language=args.language)

    if args.limit:
        print(f"⚙️  Limiting to first {args.limit} examples")
        examples = examples[:args.limit]

    convert_evobrowsecomp_to_csv(examples, args.output)

    print(f"\n✅ Done! Generated {args.output}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the output:")
    print(f"   head -5 {args.output}")
    print(f"\n2. Run evaluation:")
    print(f"   arcticswarm-eval \\")
    print(f"     --csv-path {args.output} \\")
    print(f"     --datasets EVOBROWSECOMP_V1 \\")
    print(f"     --output results/evobrowsecomp/")


if __name__ == "__main__":
    main()
