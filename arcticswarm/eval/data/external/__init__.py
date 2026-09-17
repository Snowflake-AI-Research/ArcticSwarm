"""External dataset loaders for Arcticswarm evaluation.

Provides utilities to download, decrypt (if needed), and convert
external benchmarks (BrowseComp, EvoBrowseComp, SEAL-0/SealQA,
K-BrowseComp, LoHoSearch) into Arcticswarm's Unified_eval CSV format.

Each loader generates a standalone CSV file that can be used with:
    arcticswarm-eval --csv-path path/to/dataset.csv --datasets DATASET_NAME --output results/
"""

from .browsecomp import load_browsecomp, convert_browsecomp_to_csv
from .evobrowsecomp import load_evobrowsecomp, convert_evobrowsecomp_to_csv
from .seal0 import load_seal0, convert_seal0_to_csv
from .kbrowsecomp import load_kbrowsecomp, convert_kbrowsecomp_to_csv
from .lohosearch import load_lohosearch, convert_lohosearch_to_csv

__all__ = [
    "load_browsecomp",
    "convert_browsecomp_to_csv",
    "load_evobrowsecomp",
    "convert_evobrowsecomp_to_csv",
    "load_seal0",
    "convert_seal0_to_csv",
    "load_kbrowsecomp",
    "convert_kbrowsecomp_to_csv",
    "load_lohosearch",
    "convert_lohosearch_to_csv",
]
