# Copyright 2026 Snowflake Inc.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Custom dataset resolution hook (extension point).

These two functions are the seam between a dataset *name* (used in
``eval.datasets: [MY_DATASET]``) and the CSV file that backs it. The
open-source release does not bundle a custom-dataset registry, so both hooks
are no-ops: the eval CSV resolver in :mod:`arcticswarm.eval.data_loader` then
falls through to the explicit ``eval.csv_path`` or the bundled default CSV.

There are two ways to run a custom dataset:

1. **Config-only (recommended, no code).** Point ``eval.csv_path`` at your CSV
   and list its dataset name in ``eval.datasets``. No edits here are needed —
   see ``conf/bench/custom_example.yaml`` and ``docs/custom_evaluation.md``.

2. **Registry (this hook).** If you'd rather map a dataset *name* to a path
   without setting ``eval.csv_path`` on every run, implement the two functions
   below. Have :func:`is_custom_dataset` return ``True`` for your name(s) and
   :func:`resolve_custom_csv` return the backing :class:`~pathlib.Path` (or
   ``None`` to defer to the default). The resolver only consults this registry
   when exactly one dataset is requested and ``eval.csv_path`` is unset.

In both cases the CSV must be in the unified eval format produced by
:func:`arcticswarm.eval.data.external.utils.write_unified_eval_csv`.
"""

from __future__ import annotations

from pathlib import Path


def is_custom_dataset(name: str) -> bool:
    """Return ``True`` if *name* is backed by a registered custom CSV.

    No custom datasets are registered in this release (always ``False``).
    Override to register your own — see the module docstring.
    """
    return False


def resolve_custom_csv(dataset_name: str) -> Path | None:
    """Return the CSV path backing *dataset_name*, or ``None`` to use the default.

    No custom-dataset registry ships in this release, so callers fall back to
    ``eval.csv_path`` or the bundled default CSV. Override to register your own.
    """
    return None
