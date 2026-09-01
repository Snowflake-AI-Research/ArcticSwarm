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

"""Pytest configuration shared across the arcticswarm test suite."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _disable_cache_s3_restore(monkeypatch):
    """Keep tests hermetic: never shell out to ``aws s3 sync`` for cache restore.

    The cache layer auto-restores missing caches from S3 (see
    ``arcticswarm/tools/cache_restore.py``) when an S3 mirror is configured.
    Tests use throwaway tmp DB paths that look "absent", which could otherwise
    trigger a real ``aws`` call. An empty bucket disables the feature.
    """
    monkeypatch.setenv("ARCTICSWARM_CACHE_S3_BUCKET", "")
