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
