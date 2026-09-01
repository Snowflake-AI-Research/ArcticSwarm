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

"""Central tool factory — maps tool names to constructors.

Consolidates tool instantiation logic previously scattered across
``agent.py`` and ``teammate.py`` into a single place.  Each
constructor knows what config / runtime dependencies it needs and
returns ``None`` when they are unavailable.

Usage::

    factory = ToolFactory(config, sf_client=sf, agent_client=client)
    tools = factory.build(["web_search", "web_fetch", "calculator"])
"""

from __future__ import annotations

import logging
from typing import Any

from arcticswarm.config import ArcticswarmConfig
from arcticswarm.tools.base import BaseTool

logger = logging.getLogger(__name__)


class ToolFactory:
    """Construct tool instances by name.

    Parameters
    ----------
    config:
        Flat ``ArcticswarmConfig`` with all resolved settings.
    sf_client:
        Optional Snowflake connection (needed by the Cortex corpus / web
        search + fetch auth path).
    agent_client:
        Optional LLM client (needed by source scorer / reasoning).
    """

    def __init__(
        self,
        config: ArcticswarmConfig,
        *,
        sf_client: Any | None = None,
        agent_client: Any | None = None,
        content_cache: Any | None = None,
        # Accepts and ignores any legacy keyword args so older callers that
        # still pass removed parameters don't break.
        **_unused: Any,
    ) -> None:
        self.config = config
        self.sf_client = sf_client
        self.agent_client = agent_client
        self.content_cache = content_cache
        self._source_scorer: Any | None = None
        self._content_compactor: Any | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def make(self, name: str) -> BaseTool | None:
        """Create a single tool by *name*.  Returns ``None`` if unknown
        or dependencies are missing."""
        builder = self._BUILDERS.get(name)
        if builder is None:
            logger.warning("ToolFactory: unknown tool %r — skipping", name)
            return None
        return builder(self)

    def build(self, names: list[str]) -> dict[str, BaseTool]:
        """Create all tools in *names*, skipping any that return ``None``."""
        tools: dict[str, BaseTool] = {}
        for name in names:
            tool = self.make(name)
            if tool is not None:
                tools[tool.name] = tool

        return tools

    # ------------------------------------------------------------------
    # Source scorer (shared dependency for web tools)
    # ------------------------------------------------------------------

    def _ensure_source_scorer(self) -> Any | None:
        if self._source_scorer is not None:
            return self._source_scorer
        if getattr(self.config, "disable_source_scorer", True):
            return None
        try:
            from arcticswarm.tools.source_scorer import SourceScorer
            from arcticswarm.llm_client import detect_provider
            self._source_scorer = SourceScorer(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                openai_api_key=getattr(self.config, "openai_api_key", ""),
                use_azure_openai=getattr(self.config, "use_azure_openai", False),
                azure_openai_api_key=self.config.azure_openai_api_key,
                azure_openai_endpoint=self.config.azure_openai_endpoint,
                azure_openai_api_version=self.config.azure_openai_api_version,
                agent_client=self.agent_client,
                agent_model=self.config.model,
                # Self-hosted vLLM (Qwen) runs: score on the agent's own model
                # only — never fall back to the closed GPT path.
                disable_closed_model_fallback=detect_provider(self.config.model) == "vllm",
            )
        except Exception as exc:
            logger.warning("ToolFactory: failed to create SourceScorer: %s", exc)
        return self._source_scorer

    def _ensure_content_compactor(self) -> Any | None:
        """Lazily create a single ContentCompactor when either compactor flag is set.

        Both ``use_fetch_compactor`` (web_fetch) and ``use_pdf_compactor``
        (pdf_read) share the same instance — the pipeline is identical;
        only the calling tool differs.
        """
        if not (
            getattr(self.config, "use_fetch_compactor", False)
            or getattr(self.config, "use_pdf_compactor", False)
        ):
            return None
        if self._content_compactor is not None:
            return self._content_compactor
        try:
            from arcticswarm.tools.content_compactor import ContentCompactor
            from arcticswarm.llm_client import detect_provider
            self._content_compactor = ContentCompactor(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
                openai_api_key=getattr(self.config, "openai_api_key", ""),
                use_azure_openai=getattr(self.config, "use_azure_openai", False),
                azure_openai_api_key=self.config.azure_openai_api_key,
                azure_openai_endpoint=self.config.azure_openai_endpoint,
                azure_openai_api_version=self.config.azure_openai_api_version,
                agent_client=self.agent_client,
                agent_model=self.config.model,
                # Self-hosted vLLM (Qwen): compact on the agent's own model
                # only — never fall back to the closed GPT path.
                disable_closed_model_fallback=detect_provider(self.config.model) == "vllm",
                # Bound the re-assembled selected output to the same budget the
                # agent enforces (config.max_tool_output_tokens, chars/token≈4),
                # so the compactor can't return a near-full-page selection.
                max_output_chars=max(0, getattr(self.config, "max_tool_output_tokens", 0)) * 4,
            )
        except Exception as exc:
            logger.warning("ToolFactory: failed to create ContentCompactor: %s", exc)
        return self._content_compactor

    # ------------------------------------------------------------------
    # Individual tool builders
    # ------------------------------------------------------------------

    def _make_read_file(self) -> BaseTool:
        from arcticswarm.tools.read_file import ReadFileTool
        return ReadFileTool(
            enable_vision=self.config.enable_vision,
        )

    def _make_edit_file(self) -> BaseTool:
        from arcticswarm.tools.edit_file import EditFileTool
        return EditFileTool()

    def _make_calculator(self) -> BaseTool:
        from arcticswarm.tools.calculator import CalculatorTool
        return CalculatorTool()

    def _make_bash(self) -> BaseTool:
        from arcticswarm.tools.bash import BashTool
        return BashTool()

    def _make_python_execute(self) -> BaseTool:
        from arcticswarm.tools.python_execute import PythonExecuteTool
        return PythonExecuteTool()

    def _make_reasoning(self) -> BaseTool:
        from arcticswarm.tools.reasoning import ReasoningTool
        return ReasoningTool(
            llm_client=self.agent_client,
            model=self.config.model,
            reasoning_effort=getattr(self.config, "reasoning_effort", None),
        )

    def _make_web_search(self) -> BaseTool | None:
        if self.config.web_search_provider in ("corpus", "cortex-corpus"):
            from arcticswarm.tools.corpus_retriever import build_corpus_retriever
            from arcticswarm.tools.corpus_search import CorpusSearchTool
            return CorpusSearchTool(
                retriever=build_corpus_retriever(self.config, self.sf_client),
                judge=self._ensure_source_scorer(),
            )
        # Cortex web-search provider: route web_search through the Snowflake
        # Cortex agent:run passthrough (Brave Search, or Brave Grounding Context
        # when provider == "cortex-grounding"). Tavily/Serper remain fallbacks.
        if self.config.web_search_provider in ("cortex", "cortex-grounding"):
            from arcticswarm.tools.cortex_search import (
                CortexGroundingSearchTool,
                CortexWebSearchTool,
            )
            cls = (
                CortexGroundingSearchTool
                if self.config.web_search_provider == "cortex-grounding"
                else CortexWebSearchTool
            )
            tavily_key = getattr(self.config, "tavily_api_key", "")
            serper_key = self.config.serper_api_key
            return cls(
                api_key=self.config.api_key,
                cortex_account=getattr(self.config, "cortex_account", ""),
                sf_client=self.sf_client,
                tavily_api_key=tavily_key,
                serper_api_key=serper_key,
                judge=self._ensure_source_scorer(),
            )
        judge = self._ensure_source_scorer()
        tavily_key = getattr(self.config, "tavily_api_key", "")
        serper_key = self.config.serper_api_key
        from arcticswarm.tools.web_search import WebSearchTool
        return WebSearchTool(
            self.config.brave_api_key,
            serper_api_key=serper_key,
            tavily_api_key=tavily_key,
            judge=judge,
            rich_callback=True,
            provider_order=getattr(
                self.config, "search_provider_order",
                ["brave", "tavily", "serper"],
            ),
            hard_stop=getattr(self.config, "search_repeat_guard_hard_stop", True),
            neardup_hard_stop=getattr(self.config, "search_neardup_hard_stop", 40),
        )

    def _make_web_fetch(self) -> BaseTool | None:
        if getattr(self.config, "no_web_fetch", False):
            return None
        # Corpus fetch backend: full-document retrieval from the BrowseComp-Plus corpus
        if getattr(self.config, "web_fetch_backend", "native") in ("corpus", "cortex-corpus"):
            from arcticswarm.tools.corpus_retriever import build_corpus_retriever
            from arcticswarm.tools.corpus_search import CorpusFetchTool
            return CorpusFetchTool(
                retriever=build_corpus_retriever(self.config, self.sf_client),
            )
        # Cortex Grounding fetch backend: prepend a Cortex Grounding Context
        # fetch (tier 0) before the native Jina -> Serper -> requests chain.
        if getattr(self.config, "web_fetch_backend", "native") == "cortex-grounding":
            from arcticswarm.tools.cortex_search import CortexGroundingFetchTool
            return CortexGroundingFetchTool(
                api_key=self.config.api_key,
                cortex_account=getattr(self.config, "cortex_account", ""),
                sf_client=self.sf_client,
                jina_api_key=getattr(self.config, "jina_api_key", ""),
                serper_api_key=getattr(self.config, "serper_api_key", ""),
                no_js=getattr(self.config, "no_js", False),
                content_cache=self.content_cache,
                source_scorer_enabled=not getattr(self.config, "disable_source_scorer", False),
                fetch_compactor_enabled=getattr(self.config, "use_fetch_compactor", False),
            )
        jina_key = getattr(self.config, "jina_api_key", "")
        serper_key = getattr(self.config, "serper_api_key", "")
        if not jina_key and not serper_key:
            logger.debug("ToolFactory: skipping web_fetch — no API keys")
            return None
        from arcticswarm.tools.web_fetch import WebFetchTool
        return WebFetchTool(
            jina_api_key=jina_key,
            serper_api_key=serper_key,
            no_js=getattr(self.config, "no_js", False),
            content_cache=self.content_cache,
            source_scorer_enabled=not getattr(self.config, "disable_source_scorer", False),
            fetch_compactor_enabled=getattr(self.config, "use_fetch_compactor", False),
        )

    def _make_pdf_read(self) -> BaseTool | None:
        jina_key = getattr(self.config, "jina_api_key", "")
        serper_key = getattr(self.config, "serper_api_key", "")
        if not jina_key and not serper_key:
            logger.debug("ToolFactory: skipping pdf_read — no API keys")
            return None
        from arcticswarm.tools.pdf_read import PdfReadTool
        return PdfReadTool(
            jina_api_key=jina_key,
            serper_api_key=serper_key,
            odl_hybrid=getattr(self.config, "odl_hybrid", "docling-fast"),
            odl_hybrid_url=getattr(self.config, "odl_hybrid_url", ""),
            odl_hybrid_timeout=getattr(self.config, "odl_hybrid_timeout", 60000),
            odl_hybrid_fallback_timeout=getattr(self.config, "odl_hybrid_fallback_timeout", 300),
            odl_force_ocr=getattr(self.config, "odl_force_ocr", False),
            content_cache=self.content_cache,
            source_scorer_enabled=not getattr(self.config, "disable_source_scorer", False),
            pdf_compactor_enabled=getattr(self.config, "use_pdf_compactor", False),
        )

    def _make_source_scorer(self) -> BaseTool | None:
        scorer = self._ensure_source_scorer()
        return scorer  # SourceScorer IS a BaseTool subclass

    # ------------------------------------------------------------------
    # Builder registry
    # ------------------------------------------------------------------

    _BUILDERS: dict[str, Any] = {}


# Populate the registry after class body is complete so forward
# references to methods resolve correctly.
ToolFactory._BUILDERS = {
    "read_file": ToolFactory._make_read_file,
    "edit_file": ToolFactory._make_edit_file,
    "calculator": ToolFactory._make_calculator,
    "bash": ToolFactory._make_bash,
    "python_execute": ToolFactory._make_python_execute,
    "reasoning": ToolFactory._make_reasoning,
    "web_search": ToolFactory._make_web_search,
    "web_fetch": ToolFactory._make_web_fetch,
    "pdf_read": ToolFactory._make_pdf_read,
    "source_scorer": ToolFactory._make_source_scorer,
}
