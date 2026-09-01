"""External dataset loaders for Arcticswarm evaluation.

Provides utilities to download, decrypt (if needed), and convert
external benchmarks (BrowseComp) into Arcticswarm's Unified_eval CSV format.

Each loader generates a standalone CSV file that can be used with:
    arcticswarm-eval --csv-path path/to/dataset.csv --datasets DATASET_NAME --output results/
"""

from .browsecomp import load_browsecomp, convert_browsecomp_to_csv

__all__ = [
    "load_browsecomp",
    "convert_browsecomp_to_csv",
]
