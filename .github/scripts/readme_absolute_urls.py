# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Rewrite repository-relative paths in the README to absolute GitHub URLs before packaging.

Purpose:
    The README is the package's long description, and PyPI renders it with no repository
    around it, so a relative ``docs/assets/chart.svg`` or ``LICENSE`` there points nowhere.
    GitHub renders the same relative paths correctly, so the README keeps them and the
    package build swaps each for an absolute URL.
Scope:
    With ``--docs-url``, an image under ``docs/`` becomes the copy MkDocs publishes on the
    versioned documentation site: ``docs/assets/chart.svg`` turns into
    ``<docs-url>/<version>/assets/chart.svg``, where ``<version>`` is the ``pyproject.toml``
    version with any ``.postN`` removed - the label ``publish-docs.yml`` deploys a stable
    release under. A pre-release (``a``, ``b``, ``rc``, ``dev``) gets no documentation
    version, so it falls back to the commit like everything else. Every other image becomes a
    ``raw.githubusercontent.com`` URL pinned to the commit being built, and any non-image file
    a ``github.com/<repo>/blob/<ref>/`` link. Absolute URLs, protocol-relative URLs,
    ``mailto:`` links and in-page anchors are left alone, and so is everything inside fenced
    code blocks. Every rewritten path must exist in the checkout; a missing one aborts the
    build instead of publishing a broken link.
"""

import argparse
import re
import sys
from pathlib import Path

IMAGE_SUFFIXES = (".svg", ".png", ".jpg", ".jpeg", ".gif", ".webp")
#: Directories under ``docs/`` that ``exclude_docs`` in ``mkdocs.yaml`` keeps off the site.
DOCS_EXCLUDED = ("docs/root-static/", "docs/snippets/")
_STABLE_VERSION = re.compile(r"^(\d+\.\d+\.\d+)(?:\.post\d+)?$")
_PROJECT_VERSION = re.compile(r'^\[project\][^\[]*?^version\s*=\s*"([^"]+)"', re.MULTILINE | re.DOTALL)
_HTML_ATTRIBUTE = re.compile(r'(?P<head>\b(?:src|href)=")(?P<target>[^"]+)(?P<tail>")')
_MARKDOWN_TARGET = re.compile(r"(?P<head>\]\()(?P<target>[^)\s]+)(?P<tail>[)\s])")
_FENCE = re.compile(r"^(?:```|~~~)", re.MULTILINE)


def is_relative(target: str) -> bool:
    """Tell whether a link target is a repository-relative path.

    Args:
        target: The ``src``/``href`` value or Markdown link target.

    Returns:
        True for a path inside the repository, False for URLs, anchors and ``mailto:``.

    Examples:
        >>> is_relative("docs/assets/chart.svg")
        True
        >>> is_relative("https://example.com/a.png")
        False
        >>> is_relative("#license")
        False
    """
    return not (re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", target) or target.startswith(("#", "//", "/")))


def docs_version(package_version: str) -> str | None:
    """Name the documentation version a package version is published under.

    Args:
        package_version: The ``project.version`` from ``pyproject.toml``.

    Returns:
        ``X.Y.Z`` for a stable release or post-release, None for a pre-release, which gets
        no documentation version.

    Examples:
        >>> docs_version("1.12.0.post1")
        '1.12.0'
        >>> docs_version("1.13.0rc1") is None
        True
    """
    match = _STABLE_VERSION.match(package_version)
    return match[1] if match else None


def project_version(pyproject: Path) -> str:
    """Read ``project.version`` from a ``pyproject.toml`` without ``tomllib`` (Python 3.10).

    Args:
        pyproject: Path to ``pyproject.toml``.

    Returns:
        The version string.

    Raises:
        ValueError: The ``[project]`` table has no static ``version``.

    Examples:
        >>> import tempfile
        >>> path = Path(tempfile.mkdtemp()) / "pyproject.toml"
        >>> _ = path.write_text('[build-system]\\nrequires = []\\n\\n[project]\\nname = "x"\\nversion = "1.2.3"\\n')
        >>> project_version(path)
        '1.2.3'
    """
    match = _PROJECT_VERSION.search(pyproject.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"no static [project] version in {pyproject}")
    return match[1]


def absolute_url(target: str, repo: str, ref: str, docs_base: str | None = None) -> str:
    """Build the absolute URL for a repository-relative target.

    Args:
        target: Relative path, optionally with a ``?query`` or ``#fragment``.
        repo: ``owner/name`` of the GitHub repository.
        ref: Commit SHA, tag or branch the GitHub URLs are pinned to.
        docs_base: Versioned documentation root such as ``https://docs.example/1.2.3``, or
            None to pin every image to ``ref``.

    Returns:
        The documentation-site URL for a published image under ``docs/``, a
        ``raw.githubusercontent.com`` URL for any other image, a ``github.com`` blob URL
        otherwise.

    Examples:
        >>> absolute_url("./docs/assets/chart.svg", "org/repo", "abc123", "https://docs.example/1.2.3")
        'https://docs.example/1.2.3/assets/chart.svg'
        >>> absolute_url("./docs/assets/chart.svg", "org/repo", "abc123")
        'https://raw.githubusercontent.com/org/repo/abc123/docs/assets/chart.svg'
        >>> absolute_url("LICENSE#L3", "org/repo", "abc123", "https://docs.example/1.2.3")
        'https://github.com/org/repo/blob/abc123/LICENSE#L3'
    """
    path = target.removeprefix("./")
    if not path.split("#", 1)[0].split("?", 1)[0].lower().endswith(IMAGE_SUFFIXES):
        return f"https://github.com/{repo}/blob/{ref}/{path}"
    if docs_base and path.startswith("docs/") and not path.startswith(DOCS_EXCLUDED):
        return f"{docs_base}/{path.removeprefix('docs/')}"
    return f"https://raw.githubusercontent.com/{repo}/{ref}/{path}"


def absolutize(
    text: str, repo: str, ref: str, root: Path, docs_base: str | None = None
) -> tuple[str, list[tuple[str, str]]]:
    """Replace every repository-relative link target in Markdown/HTML text.

    Args:
        text: README content.
        repo: ``owner/name`` of the GitHub repository.
        ref: Commit SHA, tag or branch the GitHub URLs are pinned to.
        root: Repository root every relative target must exist under.
        docs_base: Versioned documentation root for images under ``docs/``, or None.

    Returns:
        The rewritten text and the ``(relative, absolute)`` pairs that were replaced.

    Raises:
        FileNotFoundError: A relative target does not exist under ``root``.

    Examples:
        >>> import tempfile
        >>> root = Path(tempfile.mkdtemp())
        >>> _ = (root / "LICENSE").write_text("")
        >>> text, pairs = absolutize("[License](LICENSE) [top](#top)", "org/repo", "abc", root)
        >>> text
        '[License](https://github.com/org/repo/blob/abc/LICENSE) [top](#top)'
        >>> absolutize("```\\n[x](missing)\\n```\\n", "org/repo", "abc", root)[1]
        []
    """
    replaced: list[tuple[str, str]] = []

    def substitute(match: re.Match[str]) -> str:
        target = match["target"]
        if not is_relative(target):
            return match[0]
        path = target.removeprefix("./").split("#", 1)[0].split("?", 1)[0]
        if not (root / path).exists():
            raise FileNotFoundError(f"README links to {target!r}, which does not exist under {root}")
        url = absolute_url(target, repo, ref, docs_base)
        replaced.append((target, url))
        return f"{match['head']}{url}{match['tail']}"

    # Even-indexed chunks lie outside fenced code blocks; odd-indexed ones are code.
    chunks = _FENCE.split(text)
    for index in range(0, len(chunks), 2):
        chunk = _HTML_ATTRIBUTE.sub(substitute, chunks[index])
        chunks[index] = _MARKDOWN_TARGET.sub(substitute, chunk)
    fences = _FENCE.findall(text)
    rebuilt = chunks[0] + "".join(fence + chunk for fence, chunk in zip(fences, chunks[1:]))
    return rebuilt, replaced


def main() -> int:
    """Rewrite the README named on the command line in place and list each replacement.

    Returns:
        Process exit code: 0 on success, 1 when a relative target is missing.

    Examples:
        >>> callable(main)
        True
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("readme", type=Path)
    parser.add_argument("--repo", required=True, help="GitHub repository as owner/name")
    parser.add_argument("--ref", required=True, help="commit SHA, tag or branch to pin URLs to")
    parser.add_argument("--docs-url", help="documentation site root serving docs/ per version")
    args = parser.parse_args()
    root = args.readme.resolve().parent
    docs_base = None
    if args.docs_url:
        version = docs_version(project_version(root / "pyproject.toml"))
        docs_base = f"{args.docs_url.rstrip('/')}/{version}" if version else None
        print(f"documentation images: {docs_base or 'pre-release, pinned to the commit instead'}")
    text = args.readme.read_text(encoding="utf-8")
    try:
        rewritten, replaced = absolutize(text, args.repo, args.ref, root, docs_base)
    except FileNotFoundError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    args.readme.write_text(rewritten, encoding="utf-8")
    for relative, url in replaced:
        print(f"{relative} -> {url}")
    print(f"rewrote {len(replaced)} relative path(s) in {args.readme}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
