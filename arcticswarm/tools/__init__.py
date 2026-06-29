"""Arcticswarm tool implementations."""

import logging

# Suppress noisy warnings from third-party PDF libraries.
# These are handled gracefully by our extraction fallback chain
# (pypdf fast-path → opendataloader-pdf → Jina Reader).
for _lib in (
    "pypdf",
    "pypdf._reader",
    "pypdf.generic._utils",
    "opendataloader_pdf",
    "markitdown",
):
    logging.getLogger(_lib).setLevel(logging.CRITICAL)
