"""Unit tests for the Cortex web-search provider (web.provider = cortex).

Covers SSE result extraction, the grounding subclass api_mode, the
fallback-to-no-results path when no auth/keys are configured, and the
grounding-content extractor.  The network call (``_search_cortex`` /
``_fetch_grounding``) is stubbed so these run offline.
"""

from __future__ import annotations

import json

from arcticswarm.tools.base import ToolResult
from arcticswarm.tools.cortex_search import (
    CortexGroundingFetchTool,
    CortexGroundingSearchTool,
    CortexWebSearchTool,
)


def _tool_result_sse(search_results: list[dict]) -> str:
    """Build a minimal SSE body with one response.tool_result event."""
    data = {"content": [{"type": "json", "json": {"search_results": search_results}}]}
    return f"event: response.tool_result\ndata: {json.dumps(data)}\n\n"


def test_extract_search_results_maps_doc_fields():
    raw = json.dumps({
        "content": [
            {"type": "json", "json": {"search_results": [
                {"doc_title": "T1", "doc_id": "http://a.com", "text": "snippet 1"},
                {"doc_title": "T2", "doc_id": "http://b.com", "text": "snippet 2"},
            ]}},
        ],
    })
    out = CortexWebSearchTool._extract_search_results(raw)
    assert out == [
        {"title": "T1", "url": "http://a.com", "description": "snippet 1"},
        {"title": "T2", "url": "http://b.com", "description": "snippet 2"},
    ]


def test_extract_search_results_empty_returns_none():
    assert CortexWebSearchTool._extract_search_results('{"content": []}') is None
    assert CortexWebSearchTool._extract_search_results("not json") is None


def test_grounding_subclass_uses_grounding_api_mode():
    assert CortexGroundingSearchTool._api_mode == "grounding"
    assert CortexGroundingSearchTool._source_prefix == "cortex_grounding"
    # Inherits the search tool's tool name.
    assert CortexGroundingSearchTool.name == "web_search"


def test_execute_returns_results_when_cortex_succeeds():
    tool = CortexWebSearchTool(api_key="pat", cortex_account="acct")
    tool._search_cortex = lambda q, c: [  # type: ignore[method-assign]
        {"title": "T", "url": "http://x.com", "description": "hello world"},
    ]
    res = tool.execute(query="who invented the telephone", count=5)
    assert not res.is_error
    assert "http://x.com" in res.output
    assert tool._cortex_searches == 1


def test_execute_no_auth_no_keys_returns_no_results_gracefully():
    # No sf_client, no api_key/account, no tavily/serper -> every stage fails,
    # but the tool returns an empty-results ToolResult (not is_error).
    tool = CortexWebSearchTool()
    res = tool.execute(query="some obscure query", count=5)
    assert not res.is_error
    assert "No results" in res.output


def test_execute_falls_back_to_tavily_when_cortex_empty(monkeypatch):
    tool = CortexWebSearchTool(api_key="pat", cortex_account="acct", tavily_api_key="tv")
    tool._search_cortex = lambda q, c: None  # type: ignore[method-assign]
    # Stub the (reused) native tavily search.
    from arcticswarm.tools import web_search as ws_mod

    def fake_tavily(self, query, count):
        return [{"title": "TV", "url": "http://t.com", "description": "tav result"}]

    monkeypatch.setattr(ws_mod.WebSearchTool, "_search_tavily", fake_tavily)
    res = tool.execute(query="fallback please", count=5)
    assert not res.is_error
    assert "http://t.com" in res.output
    assert tool._tavily_searches == 1


def test_query_too_long_is_error():
    tool = CortexWebSearchTool()
    assert tool.execute(query="x" * 401).is_error
    assert tool.execute(query="word " * 51).is_error
    assert tool.execute(query="").is_error


def test_grounding_fetch_extracts_text_content():
    raw = json.dumps({
        "content": [
            {"type": "text", "text": "grounded summary"},
            {"type": "json", "json": {"search_results": [{"text": "extra chunk"}]}},
        ],
    })
    out = CortexGroundingFetchTool._extract_grounding_content(raw)
    assert out == "grounded summary\n\nextra chunk"


def test_grounding_fetch_tier0_then_native_fallback(monkeypatch):
    tool = CortexGroundingFetchTool(api_key="pat", cortex_account="acct", jina_api_key="j")
    # Tier 0 succeeds -> returns grounding content, increments counters.
    tool._fetch_grounding = lambda url: "the grounded page text"  # type: ignore[method-assign]
    res = tool.execute(url="example.com/page")
    assert res.output == "the grounded page text"
    assert res.metadata.get("via") == "Cortex Grounding"
    assert tool._instr.grounding_attempts == 1
    assert tool._instr.grounding_success == 1

    # Tier 0 empty -> falls back to the native WebFetchTool.execute chain.
    tool2 = CortexGroundingFetchTool(api_key="pat", cortex_account="acct", jina_api_key="j")
    tool2._fetch_grounding = lambda url: None  # type: ignore[method-assign]
    sentinel = ToolResult(output="native chain output")
    monkeypatch.setattr(
        "arcticswarm.tools.web_fetch.WebFetchTool.execute",
        lambda self, *, url, **kw: sentinel,
    )
    res2 = tool2.execute(url="example.com/page")
    assert res2 is sentinel
    assert tool2._instr.grounding_attempts == 1
    assert tool2._instr.grounding_success == 0
