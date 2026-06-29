#!/usr/bin/env bash
# Regenerate every benchmark CSV ArcticSwarm evaluates on, from their public
# sources. ArcticSwarm does not redistribute benchmark questions or gold
# answers — this script downloads + decrypts them locally and rebuilds the
# derived sets/subsets from CONV_ID specs (no answers are stored in the repo).
#
# Usage:
#   scripts/fetch_datasets.sh              # all datasets + subsets
#   scripts/fetch_datasets.sh --skip-evo   # skip EvoBrowseComp (needs HuggingFace)
#
# Produces, under arcticswarm/eval/data/:
#   browsecomp_v1.csv        (OpenAI BrowseComp, downloaded + decrypted)
#   evobrowsecomp_v1.csv     (HuggingFace Krystalan/EvoBrowseComp, English split)
#   browsecomp_plus_v1.csv   (derived from BrowseComp; see DATASETS.md)
#   browsecomp_subset_*.csv, browsecomp_complement_*.csv,
#   browsecomp_plus_subset_*.csv   (selected from the bases by subset_specs/)
#
# The BrowseComp-Plus *retrieval corpus* (Tevatron/browsecomp-plus-corpus) is a
# separate, large artifact used only for corpus-mode eval — see DATASETS.md.
set -euo pipefail

SKIP_EVO=0
for arg in "$@"; do
  case "$arg" in
    --skip-evo) SKIP_EVO=1 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Run from the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."
DATA="arcticswarm/eval/data"
PY="${PYTHON:-python}"

echo "==> [1/3] BrowseComp (OpenAI public blob)"
"$PY" -m arcticswarm.eval.data.external.browsecomp --output "$DATA/browsecomp_v1.csv"

if [ "$SKIP_EVO" -eq 0 ]; then
  echo "==> [2/3] EvoBrowseComp (HuggingFace Krystalan/EvoBrowseComp, English)"
  "$PY" -m arcticswarm.eval.data.external.evobrowsecomp --output "$DATA/evobrowsecomp_v1.csv"
else
  echo "==> [2/3] EvoBrowseComp SKIPPED (--skip-evo)"
fi

echo "==> [3/3] Deriving BrowseComp-Plus + all subsets from the base CSVs"
"$PY" scripts/build_subsets.py

echo
echo "✅ Datasets ready under $DATA/"
echo "   Next: arcticswarm-eval --config conf/bench/browsecomp.yaml eval.output=results/browsecomp"
echo "   (BrowseComp-Plus corpus retrieval setup: see DATASETS.md)"
