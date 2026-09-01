#!/usr/bin/env bash
# Regenerate every benchmark CSV ArcticSwarm evaluates on, from their public
# sources. ArcticSwarm does not redistribute benchmark questions or gold
# answers — this script downloads + decrypts them locally and rebuilds the
# derived sets/subsets from CONV_ID specs (no answers are stored in the repo).
#
# Usage:
#   scripts/fetch_datasets.sh              # all datasets + subsets
#
# Produces, under arcticswarm/eval/data/:
#   browsecomp_v1.csv        (OpenAI BrowseComp, downloaded + decrypted)
#   browsecomp_plus_v1.csv   (derived from BrowseComp; see DATASETS.md)
#   browsecomp_subset_*.csv, browsecomp_complement_*.csv,
#   browsecomp_plus_subset_*.csv   (selected from the bases by subset_specs/)
#
# The BrowseComp-Plus *retrieval corpus* (Tevatron/browsecomp-plus-corpus) is a
# separate, large artifact used only for corpus-mode eval — see DATASETS.md.
set -euo pipefail

for arg in "$@"; do
  case "$arg" in
    -h|--help) sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

# Run from the repo root (this script lives in scripts/).
cd "$(dirname "$0")/.."
DATA="arcticswarm/eval/data"
PY="${PYTHON:-python}"

echo "==> [1/2] BrowseComp (OpenAI public blob)"
"$PY" -m arcticswarm.eval.data.external.browsecomp --output "$DATA/browsecomp_v1.csv"

echo "==> [2/2] Deriving BrowseComp-Plus + all subsets from the base CSVs"
"$PY" scripts/build_subsets.py

echo
echo "✅ Datasets ready under $DATA/"
echo "   Next: arcticswarm-eval --config conf/bench/browsecomp.yaml eval.output=results/browsecomp"
echo "   (BrowseComp-Plus corpus retrieval setup: see DATASETS.md)"
