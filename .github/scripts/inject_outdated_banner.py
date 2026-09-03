# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Backfill the outdated-version banner into already-published gh-pages trees.

Purpose:
    Populate the empty ``data-md-component="outdated"`` banner div that archived
    version trees already carry, and give it the same purple/centered/sticky styling
    and header-offset behavior ``docs/stylesheets/rf.css`` and
    ``docs/javascripts/version-banner.js`` give it, so readers of old docs see the
    same warning ``docs/theme/main.html`` renders for trees built after that theme
    change - not Material's default yellow, left-aligned banner with the header
    overlapping it.
Scope:
    ``mike`` never rebuilds an archived version tree, and the pinned dependencies a
    given tag was built with may not resolve today, so regenerating those trees is not
    an option. This patches the static HTML, CSS, and JS directly instead, the same way
    ``.github/workflows/docs-noindex-sweep.yml`` patches the robots meta tag. ``develop`` is
    patched alongside the numeric versions: it does rebuild on every push and then carries the
    styling natively, but until that next deploy its published ``stylesheets/rf.css`` has no
    banner rules and its ``extra_javascript`` list no ``version-banner.js``, so an injected
    banner would render in Material's default yellow, left-aligned, with the header
    overlapping it. Once a tree is rebuilt both patches detect the native content and skip.
    ``latest`` is never touched.
Usage:
    Run ``python .github/scripts/inject_outdated_banner.py <gh-pages checkout root>``.
    Safe to re-run: previously injected content is replaced in place (so wording or style
    edits reach already-patched pages too), and anything not carrying our marker -
    including a genuine future rebuild - is left alone.
Outputs:
    Prints how many files were patched and exits 0. Exits nonzero only on an unexpected
    filesystem error; finding nothing to patch is not a failure.
Used by:
    ``.github/workflows/docs-noindex-sweep.yml``.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LATEST_URL = "https://rfdetr.roboflow.com/latest"

# Wraps injected content so a re-run can find and replace its own prior output - a real
# Material build never emits this comment, so a rebuilt version tree (whose banner is
# genuine, not ours) is never matched and never touched.
_MARKER_START = "<!-- rf:outdated-banner:start -->"
_MARKER_END = "<!-- rf:outdated-banner:end -->"

# Two interiors match: whitespace-only (a page built before the banner block existed) or
# a previously injected banner (bounded by our marker, so a re-run can update it).
BANNER_DIV_RE = re.compile(
    r'(<div[^>]*data-md-component="outdated"[^>]*)>'
    rf"(?:\s*|\s*{re.escape(_MARKER_START)}.*?{re.escape(_MARKER_END)}\s*)"
    r"</div>",
    re.DOTALL,
)

DEVELOP_TEXT = (
    "You are reading the unreleased development version of the documentation, "
    "built from the <code>develop</code> branch.<br>\n"
    "APIs described here may change or may never ship in a release, so use the "
    f'<a href="{LATEST_URL}/"><strong>latest stable release</strong></a> instead.'
)
ARCHIVED_TEXT = (
    "You are reading the documentation for an older version of RF-DETR, kept "
    "online for reference.<br>\n"
    "APIs described here may have changed or been removed, so use the "
    f'<a href="{LATEST_URL}/"><strong>latest stable release</strong></a> instead.'
)

# The backfill only patches develop and archived numeric trees, both of which need the
# warning visible. Reveal the injected wrapper directly; a relative URL has no base in
# `new URL()` and would abort this inline script before it reaches the banner.
_UNHIDE_SCRIPT = '<script>var el=document.querySelector("[data-md-component=outdated]");el&&(el.hidden=!1)</script>'


def _aside(text: str) -> str:
    """Build the banner markup Material renders for a warning, unhide script included.

    Args:
        text: Inner HTML of the banner, already escaped for direct embedding.

    Returns:
        The marker-wrapped ``<aside>`` block to place inside the banner div.

    Examples:
        >>> "md-banner--warning" in _aside("hi")
        True
    """
    return (
        f"\n        {_MARKER_START}\n"
        '        <aside class="md-banner md-banner--warning">\n'
        '          <div class="md-banner__inner md-grid md-typeset">\n'
        f"{text}\n"
        "          </div>\n"
        f"          {_UNHIDE_SCRIPT}\n"
        "        </aside>\n"
        f"        {_MARKER_END}\n      "
    )


# Same replace-on-rerun marker scheme as the HTML banner, so a later styling edit reaches
# an already-patched stylesheet too, without stacking a second copy.
_CSS_MARKER_START = "/* rf:outdated-banner:start */"
_CSS_MARKER_END = "/* rf:outdated-banner:end */"
CSS_BLOCK_RE = re.compile(re.escape(_CSS_MARKER_START) + r".*?" + re.escape(_CSS_MARKER_END), re.DOTALL)

# Verbatim copy of the "Version banner" section of docs/stylesheets/rf.css: the purple
# tint, centered text, and sticky positioning archived pages never got built with.
# Hardcoded rather than read from that file at run time, the same tradeoff as
# DEVELOP_TEXT/ARCHIVED_TEXT above - keep in sync if that section changes.
BANNER_CSS = """[data-md-component="outdated"] .md-banner,
[data-md-component="outdated"] .md-banner--warning {
  background-color: rgb(243, 238, 255);
  color: rgb(29, 29, 31);
  border-bottom: 1px solid rgb(229, 231, 235);
}

[data-md-component="outdated"] .md-banner__inner {
  max-width: 1600px;
  line-height: 1.6;
  text-align: center;
}

[data-md-component="outdated"] {
  position: sticky;
  top: 0;
  z-index: 5;
}

.md-header {
  top: var(--rf-banner-height, 0px);
}

[data-md-component="outdated"] .md-banner code {
  background: white;
  color: var(--md-primary-fg-color);
}

[data-md-component="outdated"] .md-banner a,
[data-md-component="outdated"] .md-banner a:focus,
[data-md-component="outdated"] .md-banner a:hover {
  color: var(--md-primary-fg-color);
  text-decoration: underline;
}"""

# docs/javascripts/version-banner.js publishes the banner's height as a CSS variable and
# nudges the sticky header/sidebars below it. Archived trees never got it, so a copy is
# written into each one: without it the banner still sticks (pure CSS), but the header
# can briefly overlap it before a reader scrolls. Read from the source tree rather than
# duplicated here, so the two never drift; the sweep workflow's sparse checkout carries
# both this script and that file.
VERSION_BANNER_JS = (Path(__file__).resolve().parents[2] / "docs" / "javascripts" / "version-banner.js").read_text(
    encoding="utf-8"
)


def _archived_version_dirs(root: Path) -> list[Path]:
    """Return numeric version directories under a gh-pages checkout, sorted.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        Directories whose name starts with a digit, in name order.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "1.2.3").mkdir()
        ...     (Path(tmp) / "latest").mkdir()
        ...     [d.name for d in _archived_version_dirs(Path(tmp))]
        ['1.2.3']
    """
    return sorted(d for d in root.iterdir() if d.is_dir() and d.name[:1].isdigit())


def _version_dirs(root: Path) -> list[Path]:
    """Return ``develop`` plus numeric version directories, sorted.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        Directories that should carry the banner markup.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "develop").mkdir()
        ...     [d.name for d in _version_dirs(Path(tmp))]
        ['develop']
    """
    dirs = _archived_version_dirs(root)
    develop = root / "develop"
    if develop.is_dir():
        dirs = sorted([*dirs, develop])
    return dirs


def patch_tree(root: Path) -> list[Path]:
    """Inject banner markup into every unpatched page under a gh-pages checkout.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The HTML files that were changed, for the caller to report against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "1.2.3" / "index.html"
        ...     page.parent.mkdir(parents=True)
        ...     _ = page.write_text('<div data-md-component="outdated" hidden></div>')
        ...     len(patch_tree(Path(tmp)))
        1
    """
    changed: list[Path] = []
    for version_dir in _version_dirs(root):
        text = DEVELOP_TEXT if version_dir.name == "develop" else ARCHIVED_TEXT
        replacement = _aside(text)
        for html_file in version_dir.rglob("*.html"):
            original = html_file.read_text(encoding="utf-8")
            patched = BANNER_DIV_RE.sub(lambda m: f"{m.group(1)}>{replacement}</div>", original)
            if patched != original:
                html_file.write_text(patched, encoding="utf-8")
                changed.append(html_file)
    return changed


def patch_stylesheets(root: Path) -> list[Path]:
    """Give the banner its purple/centered/sticky styling in each version's rf.css.

    A stylesheet already carrying the native rules is left alone, so this becomes a no-op for
    `develop` as soon as that tree is rebuilt from the theme.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The stylesheets that were changed, for the caller to report against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     css = Path(tmp) / "1.2.3" / "stylesheets" / "rf.css"
        ...     css.parent.mkdir(parents=True)
        ...     _ = css.write_text("body { color: red; }\\n")
        ...     len(patch_stylesheets(Path(tmp)))
        1
    """
    changed: list[Path] = []
    block = f"{_CSS_MARKER_START}\n{BANNER_CSS}\n{_CSS_MARKER_END}"
    for version_dir in _version_dirs(root):
        css_file = version_dir / "stylesheets" / "rf.css"
        if not css_file.is_file():
            continue
        original = css_file.read_text(encoding="utf-8")
        if _CSS_MARKER_START in original:
            patched = CSS_BLOCK_RE.sub(lambda _m: block, original)
        elif "--rf-banner-height" in original:
            # A rebuilt stylesheet already has the native banner rules; only our marker
            # identifies an older injection that should be refreshed in place.
            continue
        else:
            patched = f"{original.rstrip()}\n\n{block}\n"
        if patched != original:
            css_file.write_text(patched, encoding="utf-8")
            changed.append(css_file)
    return changed


def _relative_prefix(html_file: Path, version_dir: Path) -> str:
    """Return the ``../`` chain from a page's directory back to its version root.

    Args:
        html_file: Page being patched.
        version_dir: Version directory the page lives under.

    Returns:
        Empty string for a page at the version root, otherwise one ``../`` per level.

    Examples:
        >>> _relative_prefix(Path("1.2.3/learn/index.html"), Path("1.2.3"))
        '../'
    """
    depth = len(html_file.parent.relative_to(version_dir).parts)
    return "../" * depth


def patch_scripts(root: Path) -> list[Path]:
    """Copy version-banner.js into each version tree, referenced from every page.

    Without it the banner still sticks (pure CSS), but the header can briefly overlap it
    before a reader scrolls, since nothing offsets the header below it.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The files that were changed, for the caller to report against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "1.2.3" / "index.html"
        ...     page.parent.mkdir(parents=True)
        ...     _ = page.write_text("<body></body>")
        ...     len(patch_scripts(Path(tmp)))
        2
    """
    changed: list[Path] = []
    for version_dir in _version_dirs(root):
        js_file = version_dir / "javascripts" / "version-banner.js"
        if not js_file.is_file() or js_file.read_text(encoding="utf-8") != VERSION_BANNER_JS:
            js_file.parent.mkdir(parents=True, exist_ok=True)
            js_file.write_text(VERSION_BANNER_JS, encoding="utf-8")
            changed.append(js_file)

        for html_file in version_dir.rglob("*.html"):
            original = html_file.read_text(encoding="utf-8")
            if "javascripts/version-banner.js" in original:
                continue
            prefix = _relative_prefix(html_file, version_dir)
            tag = f'    <script src="{prefix}javascripts/version-banner.js"></script>\n'
            patched, count = re.subn(r"</body>", tag + "</body>", original, count=1)
            if count:
                html_file.write_text(patched, encoding="utf-8")
                changed.append(html_file)
    return changed


def main() -> int:
    """Patch the gh-pages tree named on the command line and report what changed.

    Returns:
        Process exit code; always 0 once the tree has been walked.

    Examples:
        >>> callable(main)
        True
    """
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path()
    changed_html = patch_tree(root)
    changed_css = patch_stylesheets(root)
    changed_js = patch_scripts(root)
    print(
        f"patched {len(changed_html)} page(s) with banner markup, "
        f"{len(changed_css)} stylesheet(s) with banner styling, "
        f"{len(changed_js)} file(s) with the sticky-offset script"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
