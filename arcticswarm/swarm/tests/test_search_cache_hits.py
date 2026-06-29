"""Tests for web_search cache hit/miss instrumentation.

A search-cache hit must increment ``_search_cache_hits`` and tag the recorded
search_log entry with ``cache_hit=True``; a miss increments ``_search_cache_misses``,
writes through, and tags ``cache_hit=False`` — so the cache hit rate is
measurable from ``web_search_log/<case>.json`` on later runs.
"""

from __future__ import annotations

from arcticswarm.tools.web_search import WebSearchTool


class _StubCache:
    def __init__(self, hits):
        self.hits = hits  # query -> list[result dict]
        self.puts = []

    def get(self, provider, query, count, country):
        return self.hits.get(query)

    def put(self, provider, query, count, country, results, replace=False):
        self.puts.append(query)


def _tool(stub):
    t = WebSearchTool(api_key="brave-key", search_cache=stub, search_cache_read=True)
    # Stub the live Brave call so a cache MISS never touches the network.
    t._search_brave_live = lambda query, count=5, country=None, safesearch="moderate": [
        {"title": "L", "url": "http://live.example", "description": "live result"}
    ]
    return t


def test_search_cache_hit_recorded():
    stub = _StubCache({"who won 2020": [
        {"title": "C", "url": "http://cache.example", "description": "cached result"}
    ]})
    t = _tool(stub)
    res = t.execute(query="who won 2020")
    assert not res.is_error
    assert t._search_cache_hits == 1 and t._search_cache_misses == 0
    log = t.drain_search_log()
    assert log and log[-1]["cache_hit"] is True
    assert "who won 2020" not in stub.puts  # a hit doesn't write through


def test_search_cache_miss_recorded_and_written_through():
    stub = _StubCache({})  # everything misses
    t = _tool(stub)
    res = t.execute(query="novel query xyz")
    assert not res.is_error
    assert t._search_cache_hits == 0 and t._search_cache_misses >= 1
    log = t.drain_search_log()
    assert log and log[-1]["cache_hit"] is False
    assert "novel query xyz" in stub.puts  # miss writes through


def test_reads_disabled_marks_not_consulted():
    stub = _StubCache({"q": [{"title": "C", "url": "http://c", "description": "d"}]})
    t = WebSearchTool(api_key="brave-key", search_cache=stub, search_cache_read=False)
    t._search_brave_live = lambda query, count=5, country=None, safesearch="moderate": [
        {"title": "L", "url": "http://live", "description": "live"}
    ]
    res = t.execute(query="q")
    assert not res.is_error
    # reads disabled -> cache not consulted (no hit/miss counted), still writes through
    assert t._search_cache_hits == 0 and t._search_cache_misses == 0
    log = t.drain_search_log()
    assert log and log[-1]["cache_hit"] is None
    assert "q" in stub.puts
