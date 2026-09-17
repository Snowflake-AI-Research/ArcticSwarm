#!/usr/bin/env python3
"""Derive a Brave-only settings JSON by stripping the dead search provider keys.

Why this exists: ``web.search_provider_order`` can only REORDER providers, not
restrict them. WebSearchTool re-appends every known provider that the order
omits ("reordering never drops a provider" — arcticswarm/tools/web_search.py),
because availability is gated by API-key presence instead. The Tavily and Serper
accounts are out of credit, so with their keys present every Brave miss burns
two doomed requests (Tavily 402, Serper 400 "not enough credits") before giving
up — pure added latency on a search-heavy long-horizon run.

Removing those two keys makes the availability gate drop them, which is the only
config-level way to get a genuinely Brave-only chain without patching the tool.
``brave_api_key`` / ``jina_api_key`` / Azure judge credentials are preserved.

Note the keys are also read from the TAVILY_API_KEY / SERPER_API_KEY environment
variables as a fallback, so those must be unset for this to take effect.

Usage:
    python scripts/make_brave_only_settings.py \\
        /code/users/soyoung/snowswarm_settings_cortex.json \\
        /code/users/soyoung/snowswarm_settings_brave_only.json
"""

import json
import os
import stat
import sys

# Providers whose credits are exhausted; keeping their keys only buys failures.
_STRIP = ("tavily_api_key", "serper_api_key")


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]

    with open(src, encoding="utf-8") as f:
        settings = json.load(f)

    removed = [k for k in _STRIP if settings.pop(k, None)]

    if not settings.get("brave_api_key"):
        print(f"ERROR: {src} has no brave_api_key — a Brave-only chain would have no provider.", file=sys.stderr)
        return 1

    with open(dst, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=2)
    # Mirror the 0600 perms of a credentials file rather than inheriting umask.
    os.chmod(dst, stat.S_IRUSR | stat.S_IWUSR)

    print(f"Wrote {dst}")
    print(f"  stripped: {removed or '(none — already absent)'}")
    print(f"  kept:     brave_api_key={'yes' if settings.get('brave_api_key') else 'NO'}, "
          f"jina_api_key={'yes' if settings.get('jina_api_key') else 'NO'}")
    for var in ("TAVILY_API_KEY", "SERPER_API_KEY"):
        if os.environ.get(var):
            print(f"  WARNING: ${var} is set in the environment and will override this removal.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
