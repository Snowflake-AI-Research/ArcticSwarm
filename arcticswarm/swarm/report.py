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

"""HTML report generation for swarm (including Vega-Lite charts).

Provides two capabilities:

1. **HTML report**: Generate a self-contained HTML file that renders
   interactive Vega-Lite charts client-side via CDN.  The file is saved
   to ``~/.arcticswarm/reports/`` and can be opened in any browser.

2. **Reference linkification**: Convert ``[N]`` citations into navigable
   hyperlinks with back-arrows in the generated HTML.
"""

from __future__ import annotations

import html
import json
import logging
import os
import platform
import re
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arcticswarm.swarm.references import ReferenceRegistry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regex for fenced code blocks
# ---------------------------------------------------------------------------

_VEGALITE_BLOCK_RE = re.compile(
    r"```vega-lite\s*\n(.*?)```",
    re.DOTALL,
)

# Detects the start of a markdown list item (unordered or ordered)
_LIST_START_RE = re.compile(r"^[-*+]\s|^\d+[.)]\s")

# Detects the start of a markdown table row (line beginning with |)
_TABLE_ROW_RE = re.compile(r"^\|")


# ---------------------------------------------------------------------------
# Dataclass for extracted chart info
# ---------------------------------------------------------------------------

@dataclass
class VegaLiteChart:
    """A single Vega-Lite spec extracted from the markdown report."""
    index: int
    source: str          # raw JSON source (inside the fence)
    title: str           # extracted from the spec or generic
    full_match: str      # the entire ```vega-lite ... ``` block


def _infer_vegalite_title(source: str, index: int) -> str:
    """Extract a title from a Vega-Lite JSON spec."""
    try:
        spec = json.loads(source)
        title = spec.get("title")
        if isinstance(title, str) and title:
            return title
        if isinstance(title, dict):
            return title.get("text", f"Chart {index}")
    except (json.JSONDecodeError, TypeError):
        pass
    return f"Chart {index}"


# ---------------------------------------------------------------------------
# Extract charts
# ---------------------------------------------------------------------------

def extract_vegalite_charts(markdown: str) -> list[VegaLiteChart]:
    """Find all ````` ```vega-lite ````` code fences in *markdown*."""
    charts: list[VegaLiteChart] = []
    for i, m in enumerate(_VEGALITE_BLOCK_RE.finditer(markdown), start=1):
        source = m.group(1)
        charts.append(VegaLiteChart(
            index=i,
            source=source,
            title=_infer_vegalite_title(source, i),
            full_match=m.group(0),
        ))
    return charts


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vega/6.2.0/vega.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vega-lite/6.4.1/vega-lite.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/vega-embed/7.0.2/vega-embed.min.js"></script>
<style>
  :root {{
    --bg: #ffffff;
    --fg: #1a1a2e;
    --code-bg: #f4f4f8;
    --border: #e0e0e0;
    --accent: #4a6fa5;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      --bg: #1a1a2e;
      --fg: #e0e0e0;
      --code-bg: #16213e;
      --border: #333;
      --accent: #7fb3ff;
    }}
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    max-width: 900px;
    margin: 2rem auto;
    padding: 0 1.5rem;
    background: var(--bg);
    color: var(--fg);
    line-height: 1.6;
  }}
  h1 {{ border-bottom: 2px solid var(--accent); padding-bottom: 0.3em; }}
  h2 {{ border-bottom: 1px solid var(--border); padding-bottom: 0.2em; }}
  pre {{
    background: var(--code-bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 1em;
    overflow-x: auto;
    font-size: 0.9em;
  }}
  code {{
    background: var(--code-bg);
    padding: 0.15em 0.3em;
    border-radius: 3px;
    font-size: 0.9em;
  }}
  pre code {{ background: none; padding: 0; }}
  table {{
    border-collapse: collapse;
    width: 100%;
    margin: 1em 0;
  }}
  th, td {{
    border: 1px solid var(--border);
    padding: 0.5em 0.75em;
    text-align: left;
  }}
  th {{ background: var(--code-bg); }}

  /* Vega-Lite container */
  .vega-container {{
    margin: 1.5em 0;
    width: 100%;
    overflow-x: auto;
  }}
  .vega-chart .vega-embed {{
    width: 100%;
  }}
  .vega-chart .vega-embed summary {{
    display: none !important;
  }}

  /* Reference inline citations (superscript) */
  .ref-link {{
    color: var(--accent);
    text-decoration: none;
    font-size: 0.75em;
    font-weight: 600;
  }}
  .ref-link:hover {{
    text-decoration: underline;
  }}

  /* References footer section */
  .references-section {{
    border-top: 2px solid var(--accent);
    margin-top: 2.5em;
    padding-top: 1em;
    font-size: 0.92em;
  }}
  .references-section h2 {{
    border-bottom: none;
    font-size: 1.15em;
    margin-bottom: 0.75em;
  }}
  .ref-entry {{
    margin-bottom: 1em;
    padding-left: 2.2em;
    text-indent: -2.2em;
  }}
  .ref-entry pre {{
    text-indent: 0;
    margin-top: 0.4em;
    margin-left: 2.2em;
    font-size: 0.88em;
  }}
  .ref-number {{
    font-weight: 700;
    color: var(--accent);
  }}
  .ref-anchor {{
    display: block;
    position: relative;
    top: -4em;
    visibility: hidden;
  }}
  .ref-back {{
    font-size: 0.8em;
    color: var(--accent);
    text-decoration: none;
    margin-left: 0.3em;
  }}
  .ref-back:hover {{
    text-decoration: underline;
  }}

  /* Clickable discovery links */
  .discovery-link {{
    color: var(--accent);
    cursor: pointer;
    text-decoration: underline;
    text-decoration-style: dotted;
  }}
  .discovery-link:hover {{
    text-decoration-style: solid;
  }}

  /* Modal overlay */
  .modal-overlay {{
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.6);
    z-index: 1000;
    justify-content: center;
    align-items: center;
  }}
  .modal-content {{
    background: var(--bg);
    color: var(--fg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.5em 2em;
    max-width: 800px;
    width: 90%;
    max-height: 80vh;
    overflow-y: auto;
    position: relative;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
  }}
  .modal-close {{
    position: absolute;
    top: 0.5em;
    right: 0.75em;
    background: none;
    border: none;
    font-size: 1.5em;
    cursor: pointer;
    color: var(--fg);
    opacity: 0.6;
    line-height: 1;
  }}
  .modal-close:hover {{
    opacity: 1;
  }}
  .sub-report-badge {{
    display: inline-block;
    background: var(--accent);
    color: var(--bg);
    font-size: 0.7em;
    padding: 0.15em 0.5em;
    border-radius: 3px;
    vertical-align: middle;
    margin-left: 0.4em;
    font-weight: 600;
  }}
  .discovery-body {{
    line-height: 1.7;
    margin-top: 1em;
  }}

  /* AI Deep Dives section */
  .deep-dives-section {{
    border-top: 2px solid var(--accent);
    margin-top: 1.5em;
    padding-top: 1em;
  }}

  @media print {{
    body {{ max-width: 100%; margin: 0; padding: 1cm; }}
    pre {{ white-space: pre-wrap; word-wrap: break-word; }}
    .vega-container {{ max-width: 100% !important; }}
    .modal-overlay {{ display: none !important; }}
  }}
</style>
</head>
<body>
{body}
<script>
(function() {{
  var isDark = window.matchMedia('(prefers-color-scheme: dark)').matches;

  var vegaDark = {{
    background: 'transparent',
    title: {{ color: '#e0e0e0' }},
    axis: {{
      domainColor: '#555',
      gridColor: '#444',
      tickColor: '#555',
      labelColor: '#ccc',
      titleColor: '#e0e0e0',
    }},
    legend: {{
      labelColor: '#ccc',
      titleColor: '#e0e0e0',
    }},
    view: {{ stroke: 'transparent' }},
  }};

  document.querySelectorAll('.vega-chart').forEach(function(el) {{
    try {{
      var spec = JSON.parse(el.getAttribute('data-spec'));
      if (!spec.width) spec.width = 500;
      if (!spec.height) spec.height = 300;
      var embedOpt = {{
        actions: false,
        renderer: 'svg',
      }};
      if (isDark) {{
        embedOpt.config = vegaDark;
      }}
      vegaEmbed(el, spec, embedOpt);
    }} catch (e) {{
      el.textContent = 'Failed to render chart: ' + e.message;
    }}
  }});
}})();

// Close modals on Escape key
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') {{
    document.querySelectorAll('.modal-overlay').forEach(function(m) {{
      m.style.display = 'none';
    }});
  }}
}});
</script>
</body>
</html>
"""


def _ensure_list_breaks(text: str) -> str:
    """Insert blank lines before list items that follow paragraph text.

    The Python ``markdown`` library requires a blank line between a
    paragraph and the first list item for the list to be recognised as
    a proper ``<ul>``/``<ol>`` instead of continuation text inside a
    ``<p>``.  LLM-generated markdown frequently omits these blank lines.

    Only *unindented* list items (no leading whitespace) are considered,
    so nested sub-lists inside an already-open list are left untouched.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks — never modify inside them
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Current line is an unindented list item
        if _LIST_START_RE.match(line) and result:
            prev_stripped = result[-1].strip()
            # Previous line is non-empty paragraph text (not a list item,
            # heading, blank line, horizontal rule, or code fence)
            if (
                prev_stripped
                and not _LIST_START_RE.match(prev_stripped)
                and not prev_stripped.startswith("#")
                and not prev_stripped.startswith("```")
                and prev_stripped not in ("---", "***", "___")
            ):
                result.append("")  # insert blank line

        result.append(line)

    return "\n".join(result)


def _ensure_table_breaks(text: str) -> str:
    """Insert blank lines before markdown table rows that follow paragraph text.

    The Python ``markdown`` library's ``tables`` extension requires a
    blank line between a paragraph and the first table row for the table
    to be recognised as a proper ``<table>`` instead of continuation text
    inside a ``<p>``.  LLM-generated markdown frequently omits these
    blank lines, e.g.::

        **Key Metrics Summary** [5]:
        | Metric | Non-SPCS | SPCS |
        |--------|----------|------|
        | ...    | ...      | ...  |

    This function inserts a blank line before the first ``|``-prefixed
    line whenever it directly follows non-blank, non-table paragraph text.
    """
    lines = text.split("\n")
    result: list[str] = []
    in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Track fenced code blocks — never modify inside them
        if stripped.startswith("```"):
            in_code_block = not in_code_block
            result.append(line)
            continue

        if in_code_block:
            result.append(line)
            continue

        # Current line looks like a table row (starts with |)
        if _TABLE_ROW_RE.match(stripped) and result:
            prev_stripped = result[-1].strip()
            # Previous line is non-empty paragraph text (not already a
            # table row, heading, blank line, horizontal rule, or code fence)
            if (
                prev_stripped
                and not _TABLE_ROW_RE.match(prev_stripped)
                and not prev_stripped.startswith("#")
                and not prev_stripped.startswith("```")
                and prev_stripped not in ("---", "***", "___")
            ):
                result.append("")  # insert blank line

        result.append(line)

    return "\n".join(result)


# Matches [N] citation patterns — used to protect them from the markdown
# parser which could misinterpret them as reference-style links.
_CITE_PLACEHOLDER_RE = re.compile(r"\[(\d+)\]")
_CITE_PLACEHOLDER_FMT = "CITE_PLACEHOLDER_{}_END"
_CITE_RESTORE_RE = re.compile(r"CITE_PLACEHOLDER_(\d+)_END")

# Matches the first top-level markdown heading (e.g. "# My Report Title")
_MD_TITLE_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)


def _protect_citations(text: str) -> tuple[str, bool]:
    """Replace ``[N]`` citations with opaque tokens before markdown parsing.

    Returns the modified text and a boolean indicating whether any
    replacements were made.
    """
    replaced = False

    def _replace(m: re.Match[str]) -> str:
        nonlocal replaced
        replaced = True
        return _CITE_PLACEHOLDER_FMT.format(m.group(1))

    # Only replace outside of fenced code blocks
    lines = text.split("\n")
    result: list[str] = []
    in_code = False
    for line in lines:
        if line.strip().startswith("```"):
            in_code = not in_code
            result.append(line)
            continue
        if in_code:
            result.append(line)
            continue
        result.append(_CITE_PLACEHOLDER_RE.sub(_replace, line))

    return "\n".join(result), replaced


def _restore_citations(html_text: str) -> str:
    """Restore ``[N]`` tokens back to their original form."""
    return _CITE_RESTORE_RE.sub(lambda m: f"[{m.group(1)}]", html_text)


def _markdown_to_html_body(markdown: str) -> str:
    """Convert markdown to HTML body content.

    Vega-Lite code fences are extracted first and replaced with UUID
    placeholders, then re-inserted as ``<div class="vega-chart">`` blocks
    that render interactively via the Vega-Embed library.

    Uses the ``markdown`` library if available; otherwise falls back to
    a lightweight regex-based converter that handles the most common
    elements (headings, code blocks, paragraphs, tables, lists, bold/italic).
    """
    placeholders: dict[str, str] = {}
    sanitised = markdown

    for chart in extract_vegalite_charts(sanitised):
        uid = f"VEGALITE_PLACEHOLDER_{uuid.uuid4().hex}"
        spec_json = html.escape(chart.source.strip(), quote=True)
        vegalite_html = (
            f'<div class="vega-container">'
            f'<div class="vega-chart" data-spec="{spec_json}"></div>'
            f'</div>'
        )
        placeholders[uid] = vegalite_html
        sanitised = sanitised.replace(chart.full_match, uid, 1)

    # Pre-process: ensure blank lines before list items and table rows so
    # the markdown library recognises them as lists/tables rather than
    # paragraph continuations.
    sanitised = _ensure_list_breaks(sanitised)
    sanitised = _ensure_table_breaks(sanitised)

    # Pre-process: protect [N] citations from being misinterpreted as
    # markdown reference-style links by the parser.
    sanitised, had_citations = _protect_citations(sanitised)

    import markdown as md
    extensions = ["tables", "fenced_code", "codehilite"]
    body = md.markdown(sanitised, extensions=extensions)

    for uid, chart_html in placeholders.items():
        body = body.replace(uid, chart_html)
        body = body.replace(f"<p>{chart_html}</p>", chart_html)

    # Restore [N] citation tokens before linkification
    if had_citations:
        body = _restore_citations(body)

    body = _linkify_references(body)
    return body


# ---------------------------------------------------------------------------
# Report saving and opening
# ---------------------------------------------------------------------------

_REPORTS_DIR = Path.home() / ".arcticswarm" / "reports"


def _extract_markdown_title(markdown: str) -> str | None:
    """Extract the first top-level ``#`` heading from *markdown*, if present."""
    m = _MD_TITLE_RE.search(markdown)
    if m:
        return m.group(1).strip()
    return None


def generate_report_html(
    markdown: str,
    title: str | None = None,
    reference_registry: ReferenceRegistry | None = None,
) -> str:
    """Convert a markdown report (with Vega-Lite fences) to self-contained HTML.

    If *title* is not provided, the first ``# heading`` in the markdown is
    used as the HTML ``<title>``.  Falls back to ``"Arcticswarm Report"``.

    When *reference_registry* is provided, the markdown References / AI Deep
    Dives sections are stripped and replaced with modal-enabled HTML from the
    registry's ``render_footer_html()``.
    """
    if title is None:
        title = _extract_markdown_title(markdown) or "Arcticswarm Report"

    # Strip markdown reference sections — they'll be replaced by HTML modals
    if reference_registry is not None and len(reference_registry) > 0:
        markdown = re.sub(
            r"\n##\s*(?:References|AI Deep Dives)\b.*",
            "",
            markdown,
            flags=re.IGNORECASE | re.DOTALL,
        ).rstrip()

    body = _markdown_to_html_body(markdown)

    # Append modal-enabled HTML footer from the registry
    if reference_registry is not None and len(reference_registry) > 0:
        # Find which refs were actually cited inline — [N] becomes
        # id="cite-N" after _linkify_references() runs inside
        # _markdown_to_html_body().
        cited_indices = {int(n) for n in re.findall(r'id="cite-(\d+)"', body)}

        if cited_indices:
            # Filter to cited-only and renumber sequentially
            filtered_reg, old_to_new = reference_registry.filtered(cited_indices)

            # Renumber inline citations and anchors in the body HTML
            def _renumber_cite(m: re.Match[str]) -> str:
                old = int(m.group(1))
                new = old_to_new.get(old)
                if new is None:
                    return m.group(0)
                return m.group(0).replace(f"cite-{old}", f"cite-{new}").replace(
                    f"ref-{old}", f"ref-{new}"
                ).replace(f"[{old}]", f"[{new}]")

            body = re.sub(
                r'<sup><a href="#ref-(\d+)"[^>]*class="ref-link">\[\d+\]</a></sup>',
                _renumber_cite,
                body,
            )

            body += "\n" + filtered_reg.render_footer_html()
        else:
            # No inline citations — still show all refs (no renumbering)
            body += "\n" + reference_registry.render_footer_html()

    return _HTML_TEMPLATE.format(title=html.escape(title), body=body)


def save_report(
    markdown: str,
    title: str | None = None,
    reference_registry: ReferenceRegistry | None = None,
) -> Path:
    """Save the report as an HTML file and return the path.

    The HTML includes CDN scripts for Vega-Lite so charts render
    interactively when opened in any modern browser.

    If *title* is not provided, the first ``# heading`` in the markdown is
    used as the HTML ``<title>``.  Falls back to ``"Arcticswarm Report"``.
    """
    _REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    session_id = uuid.uuid4().hex[:12]
    path = _REPORTS_DIR / f"report-{session_id}.html"
    html_content = generate_report_html(
        markdown, title=title, reference_registry=reference_registry,
    )
    path.write_text(html_content, encoding="utf-8")
    return path


def open_report(path: Path) -> bool:
    """Open the report file in the system default viewer.

    Returns True if the open command was launched successfully.
    """
    try:
        system = platform.system()
        if system == "Darwin":
            subprocess.Popen(["open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Linux":
            subprocess.Popen(["xdg-open", str(path)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif system == "Windows":
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            return False
        return True
    except Exception as exc:
        logger.debug("Could not open report: %s", exc)
        return False


def _linkify_references(body_html: str) -> str:
    """Post-process HTML body to turn ``[N]`` citations into hyperlinks.

    Uses the ``<h2>References</h2>`` heading as a boundary to distinguish
    inline citations (body) from reference labels (footer):

    * **Body** ``[N]`` becomes a superscript link jumping to the footer.
    * **Footer** ``[N]`` becomes an anchor target with a back-link arrow.

    Both bold (``<strong>[N]</strong>``) and plain ``[N]`` footer entries
    are handled.  Code blocks (``<pre>``, ``<code>``) are never touched.

    If :pymeth:`ReferenceRegistry.render_footer_html` already produced a
    ``<section class="references-section">``, that block is left intact.
    """
    _PROTECTED_RE = re.compile(
        r"(<pre[\s>].*?</pre>|<code[\s>].*?</code>)", re.DOTALL,
    )
    _CITE_RE = re.compile(r"\[(\d+)\]")

    # --- Step 0: Protect render_footer_html() sections --------------------
    _REF_SECTION_RE = re.compile(
        r'(<section class="references-section">.*?</section>)', re.DOTALL,
    )
    section_map: dict[str, str] = {}

    def _protect_section(m: re.Match[str]) -> str:
        uid = f"__REF_SECTION_{uuid.uuid4().hex[:8]}__"
        section_map[uid] = m.group(1)
        return uid

    body_html = _REF_SECTION_RE.sub(_protect_section, body_html)

    # --- Step 1: Split on <h2>References</h2> or <h2>AI Deep Dives</h2> ----
    _REF_HEADING_RE = re.compile(
        r"<h2>\s*(?:References|AI Deep Dives)\s*</h2>", re.IGNORECASE,
    )
    heading_match = _REF_HEADING_RE.search(body_html)

    if heading_match:
        body_part = body_html[:heading_match.start()]
        footer_part = body_html[heading_match.start():]
    else:
        body_part = body_html
        footer_part = ""

    # --- Step 2: Body — [N] → superscript links --------------------------
    first_seen: dict[int, bool] = {}

    def _replace_cite(m: re.Match[str]) -> str:
        n = int(m.group(1))
        id_attr = ""
        if n not in first_seen:
            first_seen[n] = True
            id_attr = f' id="cite-{n}"'
        return (
            f'<sup><a href="#ref-{n}"{id_attr} class="ref-link">'
            f'[{n}]</a></sup>'
        )

    parts = _PROTECTED_RE.split(body_part)
    for i, part in enumerate(parts):
        if _PROTECTED_RE.match(part):
            continue
        parts[i] = _CITE_RE.sub(_replace_cite, part)
    body_part = "".join(parts)

    # --- Step 3: Footer — [N] → anchored labels --------------------------
    if footer_part:
        def _anchor_ref(m: re.Match[str]) -> str:
            n = m.group(1)
            return (
                f'<a id="ref-{n}" class="ref-anchor"></a>'
                f'<span class="ref-number">[{n}]</span>'
                f' <a href="#cite-{n}" class="ref-back" '
                f'title="Back to text">&uarr;</a>'
            )

        footer_parts = _PROTECTED_RE.split(footer_part)
        for i, part in enumerate(footer_parts):
            if _PROTECTED_RE.match(part):
                continue
            # Strip <strong> wrappers around [N] so both bold and plain
            # entries are handled uniformly by _anchor_ref.
            part = re.sub(
                r"<strong>\[(\d+)\]</strong>", r"[\1]", part,
            )
            part = _CITE_RE.sub(_anchor_ref, part)
            footer_parts[i] = part
        footer_part = "".join(footer_parts)

    body_html = body_part + footer_part

    # --- Step 4: Restore protected sections -------------------------------
    for uid, section_html in section_map.items():
        body_html = body_html.replace(uid, section_html)

    # --- Step 5: Wrap References in <section> for CSS ---------------------
    if 'class="references-section"' not in body_html:
        m = _REF_HEADING_RE.search(body_html)
        if m:
            before = body_html[:m.start()]
            after = body_html[m.start():]
            body_html = (
                before
                + '<section class="references-section">'
                + after
                + '</section>'
            )

    return body_html
