# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the outdated-version banner the docs theme renders.

The wording lives in two places that must not drift: the theme block mkdocs renders for every fresh build, and the
backfill script that patches trees mike can no longer rebuild.
"""

import re
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from jinja2 import DictLoader, Environment


def render_outdated(template: str, site_url: str | None) -> str:
    """Render the theme's `outdated` block as mkdocs would for one deployed version.

    Args:
        template: Contents of docs/theme/main.html.
        site_url: The site_url mike writes for that version, or None for an unversioned build.

    Returns:
        The rendered block, with surrounding whitespace stripped.

    Examples:
        >>> render_outdated("{% block outdated %}{{ config.site_url }}{% endblock %}", "x")
        'x'
    """
    environment = Environment(
        loader=DictLoader({"base.html": "{% block outdated %}{% endblock %}", "main.html": template})
    )
    return environment.get_template("main.html").render(config=SimpleNamespace(site_url=site_url)).strip()


def collapse(text: str) -> str:
    """Collapse every run of whitespace to a single space.

    The theme indents its banner markup and the script embeds newlines at different points;
    only the wording itself has to match, not the layout each side happens to emit.

    Args:
        text: Markup to normalize.

    Returns:
        The text with whitespace runs collapsed and the ends stripped.

    Examples:
        >>> collapse("a  b\\n  c")
        'a b c'
    """
    return re.sub(r"\s+", " ", text).strip()


@pytest.fixture
def outdated_block(repo_root: Path) -> str:
    """Body of the `outdated` block in the docs theme template.

    Examples:
        >>> outdated_block  # doctest: +SKIP
        pytest fixture; reads docs/theme/main.html from the checkout.
    """
    template = (repo_root / "docs" / "theme" / "main.html").read_text(encoding="utf-8")
    match = re.search(r"\{% block outdated %\}(.*?)\{% endblock %\}", template, re.DOTALL)
    assert match is not None, "docs/theme/main.html no longer defines an `outdated` block"
    return match.group(1)


def test_develop_wording_is_rendered(outdated_block: str) -> None:
    assert "unreleased development version" in outdated_block


def test_archived_wording_is_rendered(outdated_block: str) -> None:
    assert "older version of RF-DETR" in outdated_block


def test_banner_links_to_latest(outdated_block: str) -> None:
    assert 'href="https://rfdetr.roboflow.com/latest/"' in outdated_block


def test_version_comes_from_site_url(outdated_block: str) -> None:
    # mike rewrites site_url per deployed version; the trailing segment names the tree.
    assert 'config.site_url or ""' in outdated_block


def test_numeric_versions_use_a_digit_test(outdated_block: str) -> None:
    # `in "0123456789"` would match the empty string and leak the banner into local builds.
    assert "[:1].isdigit()" in outdated_block


@pytest.mark.parametrize("site_url", ["https://rfdetr.roboflow.com/latest", None])
def test_latest_and_unversioned_builds_render_no_outdated_banner(repo_root: Path, site_url: str | None) -> None:
    template = (repo_root / "docs" / "theme" / "main.html").read_text(encoding="utf-8")
    environment = Environment(
        loader=DictLoader({"base.html": "{% block outdated %}{% endblock %}", "main.html": template})
    )

    rendered = environment.get_template("main.html").render(config=SimpleNamespace(site_url=site_url))

    assert rendered == ""


def test_develop_wording_matches_the_backfill_script(injector: ModuleType, repo_root: Path) -> None:
    template = (repo_root / "docs" / "theme" / "main.html").read_text(encoding="utf-8")

    rendered = render_outdated(template, "https://rfdetr.roboflow.com/develop/")

    assert collapse(rendered) == collapse(injector.DEVELOP_TEXT)


def test_archived_wording_matches_the_backfill_script(injector: ModuleType, repo_root: Path) -> None:
    template = (repo_root / "docs" / "theme" / "main.html").read_text(encoding="utf-8")

    rendered = render_outdated(template, "https://rfdetr.roboflow.com/1.9.4/")

    assert collapse(rendered) == collapse(injector.ARCHIVED_TEXT)


def test_banner_script_is_registered(repo_root: Path) -> None:
    config = (repo_root / "mkdocs.yaml").read_text(encoding="utf-8")
    assert "javascripts/version-banner.js" in config


def test_banner_stylesheet_defines_the_height_variable(repo_root: Path) -> None:
    stylesheet = (repo_root / "docs" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "--rf-banner-height" in stylesheet
