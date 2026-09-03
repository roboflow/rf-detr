# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the outdated-version banner the docs theme renders.

The wording lives in two places that must not drift: the theme block mkdocs renders for
every fresh build, and the backfill script that patches trees mike can no longer rebuild.
"""

import re
from pathlib import Path

import pytest


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


def test_theme_wording_matches_the_backfill_script(outdated_block: str, repo_root: Path) -> None:
    script = (repo_root / ".github" / "scripts" / "inject_outdated_banner.py").read_text(encoding="utf-8")
    assert "unreleased development version" in script and "older version of RF-DETR" in script


def test_banner_script_is_registered(repo_root: Path) -> None:
    config = (repo_root / "mkdocs.yaml").read_text(encoding="utf-8")
    assert "javascripts/version-banner.js" in config


def test_banner_stylesheet_defines_the_height_variable(repo_root: Path) -> None:
    stylesheet = (repo_root / "docs" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "--rf-banner-height" in stylesheet
