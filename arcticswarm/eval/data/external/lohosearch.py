#!/usr/bin/env python3
"""
LoHoSearch benchmark loader for Arcticswarm evaluation.

LoHoSearch: Benchmarking Long-Horizon Search Agents Beyond the Human
Difficulty Ceiling
Source: https://huggingface.co/datasets/meituan-longcat/LoHoSearch
Paper:  https://arxiv.org/pdf/2606.12837

Where BrowseComp-style benchmarks are human-authored (and now largely
saturated), LoHoSearch is generated from a knowledge graph over 7M+ English
Wikipedia entities with Wikidata type annotations, which lets the authors
systematically maximize search-space size and constraint depth. Each question
is a multi-constraint entity-identification puzzle: an instruction ("Identify
the horse that satisfies all of the following conditions:") followed by a
numbered list of obfuscated constraints that chain several hops through
placeholder entities (``Municipality A``, ``Horse C``, ``Person E``) with
deliberately vague temporal/quantitative anchors ("early 1960s",
"approximately 2.3 square kilometres"). Answers are short, KG-verified-unique
Wikipedia-title-style strings (e.g. ``Sky High (horse)``, ``Vogorno``).

The repo ships two configs:

    benchmark   544 rows  (split ``test``)  — the human-verified benchmark.
                                             THIS loader targets it.
    train      2000 rows  (split ``train``) — pipeline-generated, NOT verified.

Like BrowseComp, the rows are **encrypted** to keep them out of crawler
training data, but the scheme differs: LoHoSearch stores a per-row ``canary``
column and each field is ``base64(XOR(plaintext, SHA256(canary)-keystream))``.
The canary is therefore the decryption password, carried in-band, so no
external secret is needed. This module reimplements the authors' ``decrypt.py``
(same derive_key / XOR construction) rather than shelling out to it.

Environment: LoHoSearch assumes a **live-web** agent with ``search`` (keyword
queries) + ``browse`` (fetch a URL's content) — the same tool surface as
BrowseComp. There is no static corpus, so this is NOT a Cortex/corpus-backed
benchmark; run it with ``web.provider`` left at its live default.

Scoring: the paper averages TWO judges per question — the BrowseComp grading
prompt on GPT-4.1, and the SimpleQA grading prompt on Qwen2.5-32B. Arcticswarm
scores the first (BrowseComp prompt on Azure GPT-4.1) as the canonical metric;
see ``judge_lohosearch`` in ``arcticswarm/eval/judge.py``.

Note: despite the dataset card advertising 11 domains (and shipping
``assets/domain_distribution.png``), the CSVs carry NO domain / category /
difficulty column — only ``question``, ``answer``, ``canary``. Per-subset
breakdowns are therefore not available without deriving them ourselves.

Usage:
    # Generate lohosearch_v1.csv (all 544 benchmark cases)
    python -m arcticswarm.eval.data.external.lohosearch \\
        --output arcticswarm/eval/data/lohosearch_v1.csv

    # Run evaluation
    arcticswarm-eval \\
        --config conf/bench/lohosearch_qwen.yaml \\
        eval.output=results/lohosearch/
"""

import argparse
import base64
import csv
import hashlib
import sys
from pathlib import Path

from .utils import write_unified_eval_csv

LOHOSEARCH_HF_DATASET = "meituan-longcat/LoHoSearch"
# The benchmark config's data file (see the repo's README `configs:` block).
# config "benchmark" -> split "test" -> LoHoSearch.csv
LOHOSEARCH_HF_FILE = "LoHoSearch.csv"
# The unverified, pipeline-generated training split, for reference.
LOHOSEARCH_HF_TRAIN_FILE = "train.csv"

_ENCRYPTED_COLUMNS = ("question", "answer")


def _derive_key(password: str, length: int) -> bytes:
    """Derive a *length*-byte keystream from *password* via repeated SHA256 digest.

    Verbatim port of ``derive_key`` in the authors' ``decrypt.py``: the SHA256
    digest of the password is tiled to the required length. (Note this is a
    repeating-key XOR, not a true one-time pad — it is obfuscation to keep the
    questions out of web crawls, not security.)
    """
    key = hashlib.sha256(password.encode()).digest()
    return key * (length // len(key)) + key[: length % len(key)]


def _decrypt(ciphertext_b64: str, password: str) -> str:
    """Decrypt one base64-encoded, XOR-obfuscated LoHoSearch field."""
    encrypted = base64.b64decode(ciphertext_b64)
    key = _derive_key(password, len(encrypted))
    return bytes(a ^ b for a, b in zip(encrypted, key)).decode()


def _download(dataset: str, filename: str) -> Path:
    """Fetch *filename* from the HuggingFace *dataset* repo, returning a local path."""
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(repo_id=dataset, filename=filename, repo_type="dataset")
    )


def load_lohosearch(
    dataset: str = LOHOSEARCH_HF_DATASET,
    filename: str = LOHOSEARCH_HF_FILE,
) -> list[dict]:
    """Download and decrypt the LoHoSearch benchmark.

    Args:
        dataset: HuggingFace dataset id (default: ``meituan-longcat/LoHoSearch``).
        filename: Data file within the repo. ``LoHoSearch.csv`` is the 544-row
            human-verified benchmark (config ``benchmark`` / split ``test``);
            pass ``train.csv`` for the 2000-row unverified training split.

    Returns:
        List of dicts with keys: ``id``, ``question``, ``expected_answer``,
        ``attributes`` (carries ``source_index``).
    """
    print(f"📥 Downloading LoHoSearch from HuggingFace ({dataset}, file={filename})...")
    local_path = _download(dataset, filename)

    # Questions run to ~2.3K chars and the base64 ciphertext is larger still,
    # which trips csv's default 128K field cap on some rows.
    csv.field_size_limit(sys.maxsize)

    with open(local_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        for col in (*_ENCRYPTED_COLUMNS, "canary"):
            if col not in fields:
                raise ValueError(
                    f"{filename} is missing the '{col}' column (found: {fields}). "
                    "The upstream schema may have changed."
                )
        rows = list(reader)

    print(f"Found {len(rows)} encrypted examples")

    examples: list[dict] = []
    for source_index, row in enumerate(rows):
        canary = row["canary"]
        try:
            question = _decrypt(row["question"], canary).strip()
            answer = _decrypt(row["answer"], canary).strip()
        except Exception as exc:  # malformed base64 / undecodable plaintext
            print(f"⚠️  Warning: failed to decrypt row {source_index}: {exc}", file=sys.stderr)
            continue

        if not question or not answer:
            print(f"⚠️  Warning: skipping row {source_index} with empty question/answer", file=sys.stderr)
            continue

        examples.append({
            "id": f"lohosearch_{source_index:04d}",
            "question": question,
            "expected_answer": answer,
            "attributes": {"source_index": source_index},
        })

    print(f"✅ Loaded {len(examples)} LoHoSearch examples")
    return examples


def convert_lohosearch_to_csv(
    examples: list[dict],
    output_path: Path,
    dataset_name: str = "LOHOSEARCH_V1",
) -> None:
    """Convert LoHoSearch examples to Arcticswarm's unified eval CSV format.

    Args:
        examples: Output from :func:`load_lohosearch`.
        output_path: Path to output CSV.
        dataset_name: Dataset identifier in the ATTRIBUTES field.
    """
    write_unified_eval_csv(
        examples,
        output_path,
        dataset_name=dataset_name,
        eval_mode="QA",  # graded by the BrowseComp-style binary yes/no judge
        reference_tools=["web_search"],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Load and convert the LoHoSearch benchmark to Arcticswarm format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate the LoHoSearch CSV (544 benchmark cases)
  python -m arcticswarm.eval.data.external.lohosearch --output arcticswarm/eval/data/lohosearch_v1.csv

  # Limit to 10 examples for a smoke test
  python -m arcticswarm.eval.data.external.lohosearch --output test.csv --limit 10

  # Run evaluation
  arcticswarm-eval --config conf/bench/lohosearch_qwen.yaml eval.output=results/lohosearch/
        """
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output CSV path (e.g., arcticswarm/eval/data/lohosearch_v1.csv)"
    )
    parser.add_argument(
        "--dataset",
        default=LOHOSEARCH_HF_DATASET,
        help=f"HuggingFace dataset id (default: {LOHOSEARCH_HF_DATASET})"
    )
    parser.add_argument(
        "--file",
        default=LOHOSEARCH_HF_FILE,
        help=(
            f"Data file in the repo (default: {LOHOSEARCH_HF_FILE} = the 544-row "
            f"verified benchmark; {LOHOSEARCH_HF_TRAIN_FILE} = 2000-row unverified train split)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of examples (for testing)"
    )

    args = parser.parse_args()

    examples = load_lohosearch(dataset=args.dataset, filename=args.file)

    if args.limit:
        print(f"⚙️  Limiting to first {args.limit} examples")
        examples = examples[:args.limit]

    convert_lohosearch_to_csv(examples, args.output)

    print(f"\n✅ Done! Generated {args.output}")
    print(f"\n📋 Next steps:")
    print(f"1. Review the output:")
    print(f"   head -3 {args.output}")
    print(f"\n2. Run evaluation:")
    print(f"   arcticswarm-eval --config conf/bench/lohosearch_qwen.yaml eval.output=results/lohosearch/")


if __name__ == "__main__":
    main()
