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

"""Web source tracking for swarm — captures URLs from web_search tool results.

Automatically extracts and deduplicates URLs from web_search tool outputs,
making them available for the final report's References section.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcticswarm.tools.base import ToolResult


@dataclass(frozen=True)
class WebSource:
    """A single web source extracted from a web_search tool result."""

    url: str
    title: str
    snippet: str = ""

    def __hash__(self) -> int:
        # Hash by URL only for deduplication
        return hash(self.url)


class WebSourceTracker:
    """Thread-safe tracker for web sources discovered during research.

    Extracts URLs from web_search tool results and deduplicates them.
    Used by ReferenceRegistry to populate the References section.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sources: list[WebSource] = []
        self._seen_urls: set[str] = set()

    def add_from_tool_result(self, tool_result: ToolResult) -> None:
        """Extract URLs from a web_search tool result and track them.

        Parses the tool result output to find:
        - Title
        - URL
        - Snippet

        Expected format:
            1. Page Title
               URL: https://example.com
               Snippet: Description text
        """
        if tool_result.is_error or not tool_result.output:
            return

        sources = self._parse_web_search_output(tool_result.output)

        with self._lock:
            for source in sources:
                if source.url not in self._seen_urls:
                    self._seen_urls.add(source.url)
                    self._sources.append(source)

    def _parse_web_search_output(self, output: str) -> list[WebSource]:
        """Parse web_search tool output to extract structured source info.

        Format from WebSearchTool (web_search.py):
            Top N result(s) for: <query>

            1. Title
               URL: https://example.com
               Snippet: Description

            2. Another Title
               URL: https://another.com
               Snippet: More text
        """
        sources: list[WebSource] = []

        # Split by numbered entries: "1.", "2.", etc.
        entries = re.split(r'\n\d+\.\s+', output)

        for entry in entries[1:]:  # Skip the header (before first "1.")
            lines = entry.strip().split('\n')
            if not lines:
                continue

            # First line is the title
            title = lines[0].strip()

            # Extract URL and snippet from subsequent lines
            url = ""
            snippet = ""

            for line in lines[1:]:
                line = line.strip()
                if line.startswith('URL:'):
                    url = line[4:].strip()
                elif line.startswith('Snippet:'):
                    snippet = line[8:].strip()

            if url:
                sources.append(WebSource(
                    url=url,
                    title=title,
                    snippet=snippet,
                ))

        return sources

    @property
    def sources(self) -> list[WebSource]:
        """Get a copy of all tracked web sources."""
        with self._lock:
            return list(self._sources)

    def __len__(self) -> int:
        with self._lock:
            return len(self._sources)
