# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""MkDocs hook that rebuilds notebook tables of contents from rendered headings.

``mkdocs-jupyter`` builds a notebook's table of contents from the notebook markdown after deleting every backquoted span
(so code cells do not leak into it). Inline code in a heading is deleted with them, so a heading like ``## 5. Baseline —
`predict()` `` shows in the sidebar as ``5. Baseline — /`` and links to ``#5-baseline``, while the rendered heading id
is ``5-baseline-predict``. This hook replaces the table of contents with one read from the rendered HTML, which carries
the real text and ids.
"""

import re
from typing import Any

from mkdocs.structure.toc import AnchorLink, TableOfContents

_HEADING = re.compile(
    r'<h(?P<level>[1-6]) id="(?P<id>[^"]+)">(?P<title>.*?)<a class="anchor-link" href="#(?P=id)">¶</a></h(?P=level)>',
    re.DOTALL,
)


def _build_toc(html: str) -> TableOfContents:
    """Build a table of contents from notebook-rendered headings.

    Only headings that carry an ``anchor-link`` (the ones nbconvert renders from
    markdown cells) are used, so headings inside cell outputs are ignored. Heading
    HTML, including ``<code>`` spans, is kept as the item title.

    Args:
        html: Rendered notebook page content.

    Returns:
        Nested table of contents, in document order.

    Example:
        >>> toc = _build_toc(
        ...     '<h2 id="a">One <code>x()</code><a class="anchor-link" href="#a">¶</a></h2>'
        ...     '<h3 id="b">Two<a class="anchor-link" href="#b">¶</a></h3>'
        ... )
        >>> [(item.title, item.id, [child.id for child in item.children]) for item in toc]
        [('One <code>x()</code>', 'a', ['b'])]
    """
    roots: list[AnchorLink] = []
    stack: list[AnchorLink] = []
    for match in _HEADING.finditer(html):
        item = AnchorLink(match["title"].strip(), match["id"], int(match["level"]))
        while stack and stack[-1].level >= item.level:
            stack.pop()
        (stack[-1].children if stack else roots).append(item)
        stack.append(item)
    return TableOfContents(roots)


def on_page_content(html: str, page: Any, **kwargs: Any) -> str:
    """Replace a notebook page's table of contents with one built from its HTML.

    Skips pages that ``mkdocs-jupyter`` did not render, and keeps the existing table
    of contents when no heading is found.

    Args:
        html: Rendered page content.
        page: MkDocs page; notebook pages carry ``nb_url``.
        **kwargs: Remaining MkDocs event arguments, unused.

    Returns:
        ``html`` unchanged.
    """
    if getattr(page, "nb_url", None) is None:
        return html
    toc = _build_toc(html)
    if len(toc):
        page.toc = toc
    return html
