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
