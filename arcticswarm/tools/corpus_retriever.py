"""Pluggable corpus retrieval backends for the BrowseComp-Plus corpus path.

BrowseComp-Plus is a *corpus* benchmark: instead of searching the open web,
the agent searches a fixed document corpus.  The retrieval backend is
pluggable so this codebase can be open-sourced without shipping any
particular search service:

  - ``stub``   (default)  -- a no-op placeholder.  The harness imports and the
                            ``browsecomp_plus*`` configs load/run out of the
                            box, but every search/fetch returns an instructive
                            "configure a corpus backend" notice instead of real
                            documents.  Use this to smoke-test the pipeline.
  - ``cortex``             -- queries a Snowflake Cortex Search service over REST.
                            The exact service coordinates are supplied via config
                            (``corpus_account`` / ``corpus_db`` / ``corpus_schema``
                            / ``corpus_chunked_service`` / ``corpus_service``) and
                            documented in the README so readers can see the setup.
  - ``local``            -- a reference implementation that reads a local corpus
                            JSONL and ranks with a simple token-overlap scorer.
                            Intended as a *template*: swap in BM25, an embedding
                            index, or your own search service.

Select the backend with ``web.corpus_backend`` (stub | cortex | local) in the
run config.  See :func:`build_corpus_retriever`.

A retriever returns *normalised* result dicts that the corpus tool layer
(:mod:`arcticswarm.tools.corpus_search`) formats and scores:

  search(query, count) -> list[{title, url, description,
                                 cosine_similarity, reranker_score, text_match}]
  fetch(query, count)  -> list[{text, cosine_similarity, reranker_score}]

``None`` signals a backend failure (the tool then reports no results).
"""

from __future__ import annotations

import abc
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT = 30  # seconds

_STUB_NOTICE = (
    "[corpus search not configured] BrowseComp-Plus needs a corpus retrieval "
    "backend, but none is set (web.corpus_backend = 'stub'). No documents were "
    "searched. To enable retrieval, set web.corpus_backend to 'cortex' (and the "
    "corpus_* service coordinates) or 'local' (and web.corpus_local_path to a "
    "corpus JSONL). See the README 'Corpus search backends' section."
)


class CorpusRetriever(abc.ABC):
    """Backend that answers corpus search/fetch requests.

    Implementations return normalised result dicts (see module docstring) or
    ``None`` on backend failure.
    """

    @abc.abstractmethod
    def search(self, query: str, count: int) -> list[dict[str, Any]] | None:
        """Return chunked search results (snippets) for ``query``."""

    @abc.abstractmethod
    def fetch(self, query: str, count: int) -> list[dict[str, Any]] | None:
        """Return full-document results for ``query``."""


# ---------------------------------------------------------------------------
# Stub (default)
# ---------------------------------------------------------------------------


class StubCorpusRetriever(CorpusRetriever):
    """No-op backend: returns one instructive placeholder result.

    Lets the harness run end-to-end (configs load, tools register, the agent
    loop executes) with no corpus configured, while making it obvious in the
    transcript that retrieval is disabled.
    """

    def search(self, query: str, count: int) -> list[dict[str, Any]]:
        return [{
            "title": "",
            "url": "corpus://stub/0",
            "description": _STUB_NOTICE,
            "cosine_similarity": 0.0,
            "reranker_score": 0.0,
            "text_match": 0.0,
        }]

    def fetch(self, query: str, count: int) -> list[dict[str, Any]]:
        return [{
            "text": _STUB_NOTICE,
            "cosine_similarity": 0.0,
            "reranker_score": 0.0,
        }]


# ---------------------------------------------------------------------------
# Snowflake Cortex Search (REST)
# ---------------------------------------------------------------------------


def _connections_toml_path() -> Path:
    """Resolve the connections.toml path, in precedence order:

    1. ``$SNOWFLAKE_CONNECTIONS_FILE`` — explicit path to a connections.toml.
    2. ``$SNOWFLAKE_HOME/connections.toml`` — standard Snowflake config home.
    3. ``~/.snowflake/connections.toml`` — the default.

    Lets a shared/network path be used (e.g. on ephemeral pods whose home
    directory is not persistent) without copying the file into every home.
    """
    explicit = os.environ.get("SNOWFLAKE_CONNECTIONS_FILE", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    sf_home = os.environ.get("SNOWFLAKE_HOME", "").strip()
    if sf_home:
        return Path(sf_home).expanduser() / "connections.toml"
    return Path.home() / ".snowflake" / "connections.toml"


def _load_pat_from_connections(connection_name: str = "default") -> str | None:
    """Try to load a PAT from the resolved connections.toml.

    Path is resolved by :func:`_connections_toml_path` (env-overridable;
    defaults to ``~/.snowflake/connections.toml``).
    """
    p = _connections_toml_path()
    if not p.exists():
        return None
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover - py<3.11
        import tomli as tomllib  # type: ignore[no-redef]
    try:
        data = tomllib.loads(p.read_text())
        section = data.get(connection_name) or {}
        return section.get("token") or section.get("password") or None
    except Exception:
        return None


class CortexCorpusRetriever(CorpusRetriever):
    """Query a Snowflake Cortex Search service over REST.

    Service coordinates are injected (no hardcoded account/db/schema/service):

      POST https://{account}.snowflakecomputing.com
           /api/v2/databases/{db}/schemas/{schema}
           /cortex-search-services/{service}:query

    The values the original BrowseComp-Plus runs used are documented in the
    README and in ``conf/bench/browsecomp_plus*.yaml`` (commented out).
    Auth: explicit ``api_key`` + account, else a PAT from
    ``~/.snowflake/connections.toml`` [``pat_connection``], else the
    ``sf_client`` session token.
    """

    def __init__(
        self,
        *,
        account: str,
        db: str,
        schema: str,
        chunked_service: str,
        non_chunked_service: str,
        api_key: str = "",
        sf_client: Any = None,
        pat_connection: str = "default",
    ) -> None:
        self._account = account
        self._db = db
        self._schema = schema
        self._chunked_service = chunked_service
        self._non_chunked_service = non_chunked_service
        self._api_key = api_key
        self._sf_client = sf_client
        self._pat_connection = pat_connection

    def _coords_ok(self) -> bool:
        missing = [
            n for n, v in (
                ("corpus_account", self._account),
                ("corpus_db", self._db),
                ("corpus_schema", self._schema),
            ) if not v
        ]
        if missing:
            logger.warning(
                "Cortex corpus backend missing coordinates: %s. Set them in the "
                "run config (web.corpus_*) or switch web.corpus_backend.",
                ", ".join(missing),
            )
            return False
        return True

    def _get_host_and_auth(self) -> tuple[str, str]:
        if self._api_key and self._account:
            return f"{self._account}.snowflakecomputing.com", f"Bearer {self._api_key}"
        pat = _load_pat_from_connections(self._pat_connection)
        if pat and self._account:
            return f"{self._account}.snowflakecomputing.com", f"Bearer {pat}"
        if self._sf_client is not None:
            host = self._sf_client._get_rest_url()
            token = self._sf_client._get_token()
            return host, f'Snowflake Token="{token}"'
        raise RuntimeError(
            "No auth for Cortex corpus backend: need api_key + corpus_account, "
            "a PAT in connections.toml, or an sf_client."
        )

    def _query(self, service: str, query: str, limit: int) -> list[dict[str, Any]] | None:
        if not self._coords_ok():
            return None
        try:
            host, auth_header = self._get_host_and_auth()
        except Exception as exc:
            logger.warning("Cortex corpus backend: no auth available: %s", exc)
            return None
        url = (
            f"https://{host}/api/v2/databases/{self._db}"
            f"/schemas/{self._schema}/cortex-search-services/{service}:query"
        )
        body = {"query": query, "limit": limit}
        t0 = time.monotonic()
        resp = None
        try:
            resp = requests.post(
                url,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": auth_header,
                    "User-Agent": "arcticswarm/1.0",
                },
                json=body,
                timeout=_REQUEST_TIMEOUT,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            latency = int((time.monotonic() - t0) * 1000)
            status = resp.status_code if resp is not None else None
            resp_body = (resp.text[:500] if resp is not None else "")
            logger.warning(
                "Cortex corpus query failed (%dms): %s | status=%s | body=%s",
                latency, exc, status, resp_body,
            )
            return None
        return data.get("results") or None

    def search(self, query: str, count: int) -> list[dict[str, Any]] | None:
        raw = self._query(self._chunked_service, query, count)
        if not raw:
            return None
        out: list[dict[str, Any]] = []
        for i, r in enumerate(raw[:count]):
            text = r.get("CHUNK_TEXT") or r.get("TEXT") or ""
            scores = r.get("@scores") or {}
            out.append({
                "title": "",
                "url": f"corpus://{self._chunked_service}/{i}",
                "description": text,
                "cosine_similarity": scores.get("cosine_similarity", 0.0),
                "reranker_score": scores.get("reranker_score", 0.0),
                "text_match": scores.get("text_match", 0.0),
            })
        return out

    def fetch(self, query: str, count: int) -> list[dict[str, Any]] | None:
        raw = self._query(self._non_chunked_service, query, count)
        if not raw:
            return None
        out: list[dict[str, Any]] = []
        for r in raw[:count]:
            text = r.get("TEXT") or r.get("CHUNK_TEXT") or ""
            scores = r.get("@scores") or {}
            out.append({
                "text": text,
                "cosine_similarity": scores.get("cosine_similarity", 0.0),
                "reranker_score": scores.get("reranker_score", 0.0),
            })
        return out


# ---------------------------------------------------------------------------
# Local (reference template)
# ---------------------------------------------------------------------------


class LocalCorpusRetriever(CorpusRetriever):
    """Reference local backend over a corpus JSONL file.

    TEMPLATE — this is intentionally simple so you can replace it with your own
    retrieval. Point ``web.corpus_local_path`` at a JSONL file whose lines are
    objects with at least a ``text`` field (optionally ``title`` and ``id``)::

        {"id": "doc1", "title": "Some Title", "text": "full document text ..."}

    Scoring here is a dependency-free token-overlap (Jaccard-ish) heuristic.
    For real evaluation, swap :meth:`_score` for BM25 (e.g. ``rank_bm25``), a
    dense embedding index, or a call to your own search service — the rest of
    the pipeline (tool schema, formatting, source scoring) is unchanged.
    """

    def __init__(self, corpus_path: str = "") -> None:
        self._corpus_path = corpus_path
        self._docs: list[dict[str, Any]] | None = None  # lazy

    def _load(self) -> list[dict[str, Any]]:
        if self._docs is not None:
            return self._docs
        docs: list[dict[str, Any]] = []
        path = Path(self._corpus_path) if self._corpus_path else None
        if path and path.exists():
            try:
                with path.open() as fh:
                    for line in fh:
                        line = line.strip()
                        if not line:
                            continue
                        obj = json.loads(line)
                        if obj.get("text"):
                            docs.append(obj)
            except Exception as exc:
                logger.warning("LocalCorpusRetriever: failed to load %s: %s", path, exc)
        else:
            logger.warning(
                "LocalCorpusRetriever: corpus path %r not found; returning no results. "
                "Set web.corpus_local_path to a JSONL corpus file.",
                self._corpus_path,
            )
        self._docs = docs
        return docs

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if len(t) > 2}

    def _score(self, query_tokens: set[str], text: str) -> float:
        """Token-overlap score in [0, 1]. Replace with BM25/embeddings."""
        if not query_tokens:
            return 0.0
        doc_tokens = self._tokens(text)
        if not doc_tokens:
            return 0.0
        return len(query_tokens & doc_tokens) / len(query_tokens)

    def _rank(self, query: str, count: int) -> list[tuple[float, dict[str, Any]]]:
        docs = self._load()
        if not docs:
            return []
        qt = self._tokens(query)
        scored = [(self._score(qt, d.get("text", "")), d) for d in docs]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [(s, d) for s, d in scored[:count] if s > 0]

    def search(self, query: str, count: int) -> list[dict[str, Any]] | None:
        ranked = self._rank(query, count)
        if not ranked:
            return None
        out: list[dict[str, Any]] = []
        for i, (score, d) in enumerate(ranked):
            out.append({
                "title": d.get("title", ""),
                "url": f"corpus://local/{d.get('id', i)}",
                "description": d.get("text", ""),
                "cosine_similarity": round(float(score), 4),
                "reranker_score": 0.0,
                "text_match": round(float(score), 4),
            })
        return out

    def fetch(self, query: str, count: int) -> list[dict[str, Any]] | None:
        ranked = self._rank(query, count)
        if not ranked:
            return None
        return [{
            "text": d.get("text", ""),
            "cosine_similarity": round(float(score), 4),
            "reranker_score": 0.0,
        } for score, d in ranked]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def build_corpus_retriever(config: Any, sf_client: Any = None) -> CorpusRetriever:
    """Build the corpus retriever selected by ``config.corpus_backend``.

    Defaults to the stub when the backend is unset/unknown.
    """
    backend = (getattr(config, "corpus_backend", "") or "stub").strip().lower()
    if backend == "cortex":
        return CortexCorpusRetriever(
            account=getattr(config, "corpus_account", "") or getattr(config, "cortex_account", ""),
            db=getattr(config, "corpus_db", ""),
            schema=getattr(config, "corpus_schema", ""),
            chunked_service=getattr(config, "corpus_chunked_service", ""),
            non_chunked_service=getattr(config, "corpus_service", ""),
            api_key="",  # corpus auth uses the Snowflake session/PAT, NOT the general LLM api_key
            sf_client=sf_client,
            pat_connection=getattr(config, "corpus_pat_connection", "default") or "default",
        )
    if backend == "local":
        return LocalCorpusRetriever(getattr(config, "corpus_local_path", ""))
    if backend not in ("stub", ""):
        logger.warning("Unknown web.corpus_backend=%r; using stub.", backend)
    return StubCorpusRetriever()
