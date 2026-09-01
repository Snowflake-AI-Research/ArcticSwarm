#!/usr/bin/env python3
"""Rebuild the BrowseComp-Plus base CSV and all evaluation subsets from the
base benchmark CSVs.

ArcticSwarm does **not** redistribute benchmark questions or gold answers. The
public repo ships only:

  * the loader that downloads + decrypts the source benchmark
    (``arcticswarm/eval/data/external/browsecomp.py``), and
  * ``arcticswarm/eval/data/subset_specs/*`` — lists of ``CONV_ID`` values (no
    questions, no answers) that select the published subsets, plus the
    BrowseComp-Plus index map.

This script reconstructs every derived CSV from the two base CSVs that the
loaders produce. It is invoked by ``scripts/fetch_datasets.sh`` after the
loaders run, but can also be run on its own once the base CSVs exist.

What it builds:

  * ``browsecomp_plus_v1.csv`` — BrowseComp-Plus reuses BrowseComp's questions
    (a curated 830-question subset). Each row is identical to its BrowseComp
    source row except ``CONV_ID`` (``browsecomp_<i>`` -> ``browsecomp_plus_<i+1>``)
    and the ``dataset`` attribute. The mapping lives in
    ``subset_specs/browsecomp_plus_v1.srcids`` (ordered BrowseComp source ids).
    (The BrowseComp-Plus *retrieval corpus* is a separate artifact — see
    DATASETS.md.)
  * every ``subset_specs/<name>.ids`` -> ``<name>.csv``, selecting rows from the
    appropriate base set in the order listed in the spec.

Output rows are written through the same ``write_unified_eval_csv`` writer the
loaders use, so the regenerated files match the original layout exactly.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

# Allow running as a plain script (python scripts/build_subsets.py) without an
# editable install.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arcticswarm.eval.data.external.utils import write_unified_eval_csv

csv.field_size_limit(10**8)

# Subsets whose name starts with this prefix are selected from BrowseComp-Plus;
# all others from BrowseComp.
PLUS_PREFIX = "browsecomp_plus_"


def _read_rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _row_to_example(row: dict) -> dict:
    """Parse a unified-eval CSV row back into a write_unified_eval_csv example.

    The full parsed ATTRIBUTES dict is preserved as ``attributes`` so the
    writer reproduces the row's dataset / eval_mode / is_vip exactly.
    """
    return {
        "id": row["CONV_ID"],
        "question": row["TURN"],
        "expected_answer": json.loads(row["REFERENCE_MESSAGE"]).get("text", ""),
        "attributes": json.loads(row["ATTRIBUTES"]),
        "_reference_tools": json.loads(row["REFERENCE_TOOLS"]),
    }


def _write(examples: list[dict], out_path: Path, dataset_name: str) -> None:
    ref_tools = examples[0]["_reference_tools"] if examples else ["web_search"]
    eval_mode = examples[0]["attributes"].get("eval_mode", "QA") if examples else "QA"
    for ex in examples:
        ex.pop("_reference_tools", None)
    write_unified_eval_csv(
        examples,
        out_path,
        dataset_name=dataset_name,
        eval_mode=eval_mode,
        reference_tools=ref_tools,
    )


def build_browsecomp_plus(browsecomp_rows: list[dict], spec_dir: Path) -> list[dict]:
    """Derive the BrowseComp-Plus base rows from BrowseComp + the src-id map."""
    srcids_path = spec_dir / "browsecomp_plus_v1.srcids"
    if not srcids_path.exists():
        raise FileNotFoundError(f"missing BC-Plus index map: {srcids_path}")
    by_id = {r["CONV_ID"]: r for r in browsecomp_rows}
    src_ids = [ln.strip() for ln in srcids_path.read_text().splitlines() if ln.strip()]
    examples: list[dict] = []
    for src in src_ids:
        if src not in by_id:
            raise KeyError(f"BrowseComp source row {src!r} not found — regenerate browsecomp_v1.csv first")
        n = int(src.rsplit("_", 1)[1])
        ex = _row_to_example(by_id[src])
        ex["id"] = f"browsecomp_plus_{n + 1}"
        ex["attributes"] = {**ex["attributes"], "dataset": "BROWSECOMP_PLUS_V1"}
        examples.append(ex)
    return examples


def base_for_spec(name: str) -> str:
    return "browsecomp_plus_v1" if name.startswith(PLUS_PREFIX) else "browsecomp_v1"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("arcticswarm/eval/data"),
        help="eval/data dir holding the base CSVs + subset_specs/ (default: %(default)s)",
    )
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    spec_dir: Path = data_dir / "subset_specs"

    bc_path = data_dir / "browsecomp_v1.csv"
    if not bc_path.exists():
        print(f"❌ {bc_path} missing — run the BrowseComp loader first (see scripts/fetch_datasets.sh).", file=sys.stderr)
        return 1
    browsecomp_rows = _read_rows(bc_path)

    # 1) BrowseComp-Plus base (derived from BrowseComp).
    plus_examples = build_browsecomp_plus(browsecomp_rows, spec_dir)
    _write(plus_examples, data_dir / "browsecomp_plus_v1.csv", "BROWSECOMP_PLUS_V1")
    print(f"✅ browsecomp_plus_v1.csv  ({len(plus_examples)} rows, derived from BrowseComp)")

    # 2) Subsets — select from the appropriate base by CONV_ID, in spec order.
    bases = {
        "browsecomp_v1": {r["CONV_ID"]: r for r in browsecomp_rows},
        "browsecomp_plus_v1": {r["CONV_ID"]: r for r in _read_rows(data_dir / "browsecomp_plus_v1.csv")},
    }

    specs = sorted(spec_dir.glob("*.ids"))
    if not specs:
        print(f"⚠️  no *.ids subset specs in {spec_dir}", file=sys.stderr)
    for spec in specs:
        name = spec.stem
        base_name = base_for_spec(name)
        index = bases[base_name]
        ids = [ln.strip() for ln in spec.read_text().splitlines() if ln.strip()]
        missing = [i for i in ids if i not in index]
        if missing:
            print(f"❌ {name}: {len(missing)} ids missing from {base_name} (e.g. {missing[:3]})", file=sys.stderr)
            return 1
        examples = [_row_to_example(index[i]) for i in ids]
        dataset_name = examples[0]["attributes"].get("dataset", base_name.upper()) if examples else "CUSTOM"
        _write(examples, data_dir / f"{name}.csv", dataset_name)
        print(f"✅ {name}.csv  ({len(examples)} rows from {base_name})")

    print("\n✅ All derived CSVs rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
