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
    ``latest`` is never touched, and neither is the newest release directory: ``mike``
    publishes each release both under its own number and under ``latest``, so a reader there
    is on the current release. An earlier run that did banner it is undone by
    ``strip_from_current_release``. The newest release does get ``version-banner.js``, which
    is what keeps its natively built banner markup hidden.
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

from packaging.version import InvalidVersion, Version

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

# docs/javascripts/version-banner.js decides whether the banner is revealed - comparing this
# tree against the highest version in versions.json, which Material's own alias-based check
# gets wrong here - and publishes the banner's height so the sticky header clears it. Trees
# published before it existed get a copy. Read from the source tree rather than duplicated
# here, so the two never drift; the sweep workflow's sparse checkout carries both this script
# and that file.
VERSION_BANNER_JS = (Path(__file__).resolve().parents[2] / "docs" / "javascripts" / "version-banner.js").read_text(
    encoding="utf-8"
)


def _release_version_dirs(root: Path) -> list[Path]:
    """Return the release directories under a gh-pages checkout, oldest first.

    A directory is a release when its name parses as a version, which also filters out
    ``latest``, ``develop``, and the stray asset directories at the gh-pages root. Ordering
    is by parsed version, so 1.10.0 follows 1.9.4 rather than preceding it as it would by
    name.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        Release directories in ascending version order.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     for name in ("1.10.0", "1.9.4", "latest"):
        ...         (Path(tmp) / name).mkdir()
        ...     [d.name for d in _release_version_dirs(Path(tmp))]
        ['1.9.4', '1.10.0']
    """
    releases: list[tuple[Version, Path]] = []
    for candidate in root.iterdir():
        if not candidate.is_dir():
            continue
        try:
            releases.append((Version(candidate.name), candidate))
        except InvalidVersion:
            continue
    return [directory for _version, directory in sorted(releases, key=lambda release: release[0])]


def current_release_dir(root: Path) -> Path | None:
    """Return the version directory holding the current release, if there is one.

    ``mike`` publishes each release both under its own version number and under ``latest``,
    so the highest numbered directory is the same documentation ``latest/`` serves. A reader
    there is on the current release and must not be told they are reading an old version.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The highest numbered version directory, or None when no numbered directory exists.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "1.9.4").mkdir()
        ...     (Path(tmp) / "1.10.0").mkdir()
        ...     current_release_dir(Path(tmp)).name
        '1.10.0'
    """
    releases = _release_version_dirs(root)
    if not releases:
        return None
    return releases[-1]


def _version_dirs(root: Path) -> list[Path]:
    """Return every directory whose readers should see the outdated-version banner.

    That is ``develop`` plus the superseded numbered releases — never ``latest`` and never
    the current release's own directory.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        Directories that should carry the banner.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "develop").mkdir()
        ...     (Path(tmp) / "1.2.3").mkdir()
        ...     [d.name for d in _version_dirs(Path(tmp))]
        ['develop']
    """
    current = current_release_dir(root)
    dirs = [d for d in _release_version_dirs(root) if d != current]
    develop = root / "develop"
    if develop.is_dir():
        dirs = [*dirs, develop]
    return dirs


def _script_dirs(root: Path) -> list[Path]:
    """Return every directory that needs the banner script, current release included.

    The script decides whether the banner is revealed, so the current release needs it as much
    as the superseded trees do: its pages carry banner markup built before that decision moved
    out of Material, and without the script Material's own check - which never sees a `latest`
    alias in versions.json - warns readers of the newest docs that they are reading old ones.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        Directories that should carry the script.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     (Path(tmp) / "1.2.3").mkdir()
        ...     (Path(tmp) / "1.3.0").mkdir()
        ...     [d.name for d in _script_dirs(Path(tmp))]
        ['1.2.3', '1.3.0']
    """
    dirs = _version_dirs(root)
    current = current_release_dir(root)
    if current is not None and current not in dirs:
        dirs = [*dirs, current]
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
        ...     (Path(tmp) / "1.3.0").mkdir()  # the current release, never banner-ed
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
        ...     (Path(tmp) / "1.3.0").mkdir()  # the current release, never styled
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

    The script both decides whether the banner is revealed and offsets the sticky header below
    it, so every version tree gets it - the current release included, whose pages would
    otherwise fall back to Material's own outdated check and warn about themselves.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The files that were changed, for the caller to report against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "1.2.3" / "index.html"
        ...     page.parent.mkdir(parents=True)
        ...     (Path(tmp) / "1.3.0").mkdir()  # the current release, scripted but never banner-ed
        ...     _ = page.write_text("<body></body>")
        ...     len(patch_scripts(Path(tmp)))
        3
    """
    changed: list[Path] = []
    for version_dir in _script_dirs(root):
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


def strip_from_current_release(root: Path) -> list[Path]:
    """Undo a previous run that banner-ed the directory now holding the current release.

    An earlier backfill treated every numbered directory as superseded, so the release that
    is now current carries an "older version" warning it must not show. Only content bounded
    by this script's own markers is removed, so a genuine build is never touched - a natively
    built banner stays in the markup and is kept hidden by `version-banner.js`, which
    `patch_scripts` also delivers here. The copied script and its tag are left in place.

    Args:
        root: Root of the gh-pages checkout.

    Returns:
        The files that were changed, for the caller to report against.

    Examples:
        >>> import tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     page = Path(tmp) / "1.2.3" / "index.html"
        ...     page.parent.mkdir(parents=True)
        ...     _ = page.write_text(
        ...         '<div data-md-component="outdated" hidden>'
        ...         + _aside(ARCHIVED_TEXT)
        ...         + "</div>"
        ...     )
        ...     len(strip_from_current_release(Path(tmp)))
        1
    """
    current = current_release_dir(root)
    if current is None:
        return []
    changed: list[Path] = []
    for html_file in current.rglob("*.html"):
        original = html_file.read_text(encoding="utf-8")
        if _MARKER_START not in original:
            continue
        patched = BANNER_DIV_RE.sub(lambda m: f"{m.group(1)}></div>", original)
        if patched != original:
            html_file.write_text(patched, encoding="utf-8")
            changed.append(html_file)
    css_file = current / "stylesheets" / "rf.css"
    if css_file.is_file():
        original = css_file.read_text(encoding="utf-8")
        if _CSS_MARKER_START in original:
            patched = f"{CSS_BLOCK_RE.sub(lambda _m: '', original).rstrip()}\n"
            if patched != original:
                css_file.write_text(patched, encoding="utf-8")
                changed.append(css_file)
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
    stripped = strip_from_current_release(root)
    print(
        f"patched {len(changed_html)} page(s) with banner markup, "
        f"{len(changed_css)} stylesheet(s) with banner styling, "
        f"{len(changed_js)} file(s) with the sticky-offset script, "
        f"cleared a stale banner from {len(stripped)} file(s) under the current release"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
