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

"""Arcticswarm tool implementations."""

import logging

# Suppress noisy warnings from third-party PDF libraries.
# These are handled gracefully by our extraction fallback chain
# (pypdf fast-path → opendataloader-pdf → Jina Reader).
for _lib in (
    "pypdf",
    "pypdf._reader",
    "pypdf.generic._utils",
    "opendataloader_pdf",
    "markitdown",
):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)
