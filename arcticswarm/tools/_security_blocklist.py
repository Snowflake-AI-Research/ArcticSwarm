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

"""Security blocklist for URL fetches via web_fetch and pdf_read.

Domains added here are rejected before any network call. Use this for hosts
flagged on Threat Intel feeds or otherwise determined to be unsafe to contact
from eval infrastructure — note that these blocks fire at the Python tool
boundary only and do NOT prevent fetches initiated by JVM subprocesses inside
opendataloader-pdf (Apache Tika). For defense-in-depth against in-process PDF
fetches, pair this with a network-level block.
"""

from __future__ import annotations

from urllib.parse import urlparse

# Blocked host substrings → human-readable reason. Matched case-insensitively
# against both the URL string and the parsed hostname. Empty by default; add
# hosts you want rejected before any fetch, e.g.:
#
#     SECURITY_BLOCKLIST = {
#         "malicious.example.com": "Domain blocked: flagged by our threat feed.",
#     }
SECURITY_BLOCKLIST: dict[str, str] = {}


def check_blocked(url: str) -> str | None:
    """Return a block reason if *url* matches a blocked domain, else None."""
    if not url:
        return None
    lowered = url.lower()
    host = ""
    try:
        host = (urlparse(url).hostname or "").lower()
    except Exception:
        host = ""
    for pattern, reason in SECURITY_BLOCKLIST.items():
        p = pattern.lower()
        if p in lowered or (host and (host == p or host.endswith("." + p))):
            return reason
    return None
