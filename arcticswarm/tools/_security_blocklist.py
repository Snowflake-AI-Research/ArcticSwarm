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
