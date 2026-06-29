"""Custom dataset resolution (stub).

The open-source release ships the BrowseComp / BrowseComp-Plus CSVs directly
and does not bundle a custom-dataset registry. These hooks are no-ops so the
eval CSV resolver in :mod:`arcticswarm.eval.data_loader` falls through to the
explicit ``eval.csv_path`` or the bundled default CSV.
"""

from __future__ import annotations

from pathlib import Path


def is_custom_dataset(name: str) -> bool:
    """No custom datasets are registered in this release."""
    return False


def resolve_custom_csv(dataset_name: str) -> Path | None:
    """No custom-dataset registry; callers fall back to the default CSV."""
    return None
