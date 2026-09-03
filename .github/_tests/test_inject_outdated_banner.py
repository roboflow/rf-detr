# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the gh-pages outdated-version banner backfill script."""

from pathlib import Path
from types import ModuleType

import pytest

EMPTY_PAGE = (
    "<html><head></head><body>"
    '<div data-md-color-scheme="default" data-md-component="outdated" hidden></div>'
    "</body></html>"
)


@pytest.fixture
def gh_pages(tmp_path: Path) -> Path:
    """A gh-pages tree with an archived version, develop, latest, and a nested page.

    Examples:
        >>> gh_pages  # doctest: +SKIP
        pytest fixture; requires the fixture protocol to build the tree.
    """
    for relative in (
        "1.3.0/index.html",
        "1.3.0/learn/index.html",
        "develop/index.html",
        "latest/index.html",
    ):
        page = tmp_path / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(EMPTY_PAGE, encoding="utf-8")
    stylesheet = tmp_path / "1.3.0" / "stylesheets" / "rf.css"
    stylesheet.parent.mkdir(parents=True, exist_ok=True)
    stylesheet.write_text("body {\n  color: red;\n}\n", encoding="utf-8")
    return tmp_path


def test_archived_page_gets_archived_wording(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    page = (gh_pages / "1.3.0" / "index.html").read_text(encoding="utf-8")
    assert "documentation for an older version of RF-DETR" in page


def test_develop_page_gets_develop_wording(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    page = (gh_pages / "develop" / "index.html").read_text(encoding="utf-8")
    assert "unreleased development version" in page


def test_latest_page_is_never_patched(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    page = (gh_pages / "latest" / "index.html").read_text(encoding="utf-8")
    assert page == EMPTY_PAGE


def test_banner_link_points_at_latest(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    page = (gh_pages / "1.3.0" / "index.html").read_text(encoding="utf-8")
    assert 'href="https://rfdetr.roboflow.com/latest/"' in page


def test_rerun_changes_nothing(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    assert injector.patch_tree(gh_pages) == []


def test_rerun_does_not_stack_banners(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_tree(gh_pages)
    injector.patch_tree(gh_pages)
    page = (gh_pages / "1.3.0" / "index.html").read_text(encoding="utf-8")
    assert page.count("md-banner--warning") == 1


def test_genuine_rebuild_is_left_alone(injector: ModuleType, gh_pages: Path) -> None:
    page = gh_pages / "1.3.0" / "index.html"
    rebuilt = EMPTY_PAGE.replace("hidden></div>", 'hidden><aside class="md-banner">built</aside></div>')
    page.write_text(rebuilt, encoding="utf-8")
    injector.patch_tree(gh_pages)
    assert page.read_text(encoding="utf-8") == rebuilt


def test_archived_stylesheet_gets_banner_rules(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_stylesheets(gh_pages)
    stylesheet = (gh_pages / "1.3.0" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "--rf-banner-height" in stylesheet


def test_rerun_replaces_stylesheet_block(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_stylesheets(gh_pages)
    injector.patch_stylesheets(gh_pages)
    stylesheet = (gh_pages / "1.3.0" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert stylesheet.count("rf:outdated-banner:start") == 1


def test_stylesheet_keeps_existing_rules(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_stylesheets(gh_pages)
    stylesheet = (gh_pages / "1.3.0" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "color: red;" in stylesheet


def test_native_banner_stylesheet_is_left_unchanged(injector: ModuleType, gh_pages: Path) -> None:
    stylesheet = gh_pages / "1.3.0" / "stylesheets" / "rf.css"
    native_stylesheet = f"{injector.BANNER_CSS}\n"
    stylesheet.write_text(native_stylesheet, encoding="utf-8")

    assert injector.patch_stylesheets(gh_pages) == []
    assert stylesheet.read_text(encoding="utf-8") == native_stylesheet


def test_injected_banner_rules_are_scoped_to_the_outdated_component(injector: ModuleType) -> None:
    assert "\n.md-banner" not in injector.BANNER_CSS
    assert '[data-md-component="outdated"] .md-banner' in injector.BANNER_CSS


def test_develop_stylesheet_is_not_created(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_stylesheets(gh_pages)
    assert not (gh_pages / "develop" / "stylesheets" / "rf.css").exists()


def test_archived_version_gets_offset_script(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    script = gh_pages / "1.3.0" / "javascripts" / "version-banner.js"
    assert script.read_text(encoding="utf-8") == injector.VERSION_BANNER_JS


def test_nested_page_references_script_relatively(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    page = (gh_pages / "1.3.0" / "learn" / "index.html").read_text(encoding="utf-8")
    assert '<script src="../javascripts/version-banner.js"></script>' in page


def test_root_page_references_script_without_prefix(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    page = (gh_pages / "1.3.0" / "index.html").read_text(encoding="utf-8")
    assert '<script src="javascripts/version-banner.js"></script>' in page


def test_develop_does_not_get_the_offset_script(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    assert not (gh_pages / "develop" / "javascripts").exists()


def test_non_version_directories_are_ignored(injector: ModuleType, gh_pages: Path) -> None:
    stray = gh_pages / "assets" / "index.html"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text(EMPTY_PAGE, encoding="utf-8")
    injector.patch_tree(gh_pages)
    assert stray.read_text(encoding="utf-8") == EMPTY_PAGE
