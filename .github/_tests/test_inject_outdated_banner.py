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
    """A gh-pages tree with a superseded release, the current release, develop, and latest.

    Examples:
        >>> gh_pages  # doctest: +SKIP
        pytest fixture; requires the fixture protocol to build the tree.
    """
    for relative in (
        "1.3.0/index.html",
        "1.3.0/learn/index.html",
        "1.9.4/index.html",
        "develop/index.html",
        "latest/index.html",
    ):
        page = tmp_path / relative
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(EMPTY_PAGE, encoding="utf-8")
    for version in ("1.3.0", "1.9.4", "develop"):
        stylesheet = tmp_path / version / "stylesheets" / "rf.css"
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

    injector.patch_stylesheets(gh_pages)

    assert stylesheet.read_text(encoding="utf-8") == native_stylesheet


def test_native_banner_stylesheet_is_not_reported_as_changed(injector: ModuleType, gh_pages: Path) -> None:
    stylesheet = gh_pages / "1.3.0" / "stylesheets" / "rf.css"
    stylesheet.write_text(f"{injector.BANNER_CSS}\n", encoding="utf-8")

    assert stylesheet not in injector.patch_stylesheets(gh_pages)


def test_injected_banner_rules_are_scoped_to_the_outdated_component(injector: ModuleType) -> None:
    assert "\n.md-banner" not in injector.BANNER_CSS
    assert '[data-md-component="outdated"] .md-banner' in injector.BANNER_CSS


def test_develop_stylesheet_gets_banner_rules(injector: ModuleType, gh_pages: Path) -> None:
    # develop is rebuilt on every push, but its published tree carries no banner rules until
    # that next deploy - without them the injected banner renders in Material's default yellow.
    injector.patch_stylesheets(gh_pages)
    stylesheet = (gh_pages / "develop" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "--rf-banner-height" in stylesheet


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


def test_develop_gets_the_offset_script(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    script = gh_pages / "develop" / "javascripts" / "version-banner.js"
    assert script.read_text(encoding="utf-8") == injector.VERSION_BANNER_JS


def test_rebuilt_tree_keeps_its_native_script_reference(injector: ModuleType, gh_pages: Path) -> None:
    # A rebuilt develop already lists the script; patching again must not add a second tag.
    page = gh_pages / "develop" / "index.html"
    page.write_text(EMPTY_PAGE.replace("</body>", '<script src="javascripts/version-banner.js"></script></body>'))

    injector.patch_scripts(gh_pages)

    assert page.read_text(encoding="utf-8").count("version-banner.js") == 1


def test_non_version_directories_are_ignored(injector: ModuleType, gh_pages: Path) -> None:
    stray = gh_pages / "assets" / "index.html"
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_text(EMPTY_PAGE, encoding="utf-8")
    injector.patch_tree(gh_pages)
    assert stray.read_text(encoding="utf-8") == EMPTY_PAGE


def test_current_release_is_not_warned_about(injector: ModuleType, gh_pages: Path) -> None:
    # 1.9.4 is the highest numbered version, so it is what latest/ serves.
    injector.patch_tree(gh_pages)
    assert (gh_pages / "1.9.4" / "index.html").read_text(encoding="utf-8") == EMPTY_PAGE


def test_current_release_gets_the_banner_script(injector: ModuleType, gh_pages: Path) -> None:
    # The script is what keeps a natively built banner hidden on the newest release, so that
    # tree needs it even though it is never banner-ed or styled.
    injector.patch_scripts(gh_pages)
    script = gh_pages / "1.9.4" / "javascripts" / "version-banner.js"
    assert script.read_text(encoding="utf-8") == injector.VERSION_BANNER_JS


def test_current_release_page_references_the_banner_script(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_scripts(gh_pages)
    page = (gh_pages / "1.9.4" / "index.html").read_text(encoding="utf-8")
    assert '<script src="javascripts/version-banner.js"></script>' in page


def test_current_release_stylesheet_is_not_patched(injector: ModuleType, gh_pages: Path) -> None:
    injector.patch_stylesheets(gh_pages)
    stylesheet = (gh_pages / "1.9.4" / "stylesheets" / "rf.css").read_text(encoding="utf-8")
    assert "--rf-banner-height" not in stylesheet


def test_current_release_is_the_highest_numbered_version(injector: ModuleType, gh_pages: Path) -> None:
    assert injector.current_release_dir(gh_pages).name == "1.9.4"


def test_double_digit_minor_outranks_single_digit(injector: ModuleType, tmp_path: Path) -> None:
    # String order would rank 1.9.4 above 1.10.0.
    for version in ("1.9.4", "1.10.0"):
        (tmp_path / version).mkdir()

    assert injector.current_release_dir(tmp_path).name == "1.10.0"


def test_prerelease_does_not_outrank_its_release(injector: ModuleType, tmp_path: Path) -> None:
    for version in ("1.10.0", "1.10.0rc1"):
        (tmp_path / version).mkdir()

    assert injector.current_release_dir(tmp_path).name == "1.10.0"


def test_named_directories_are_not_releases(injector: ModuleType, tmp_path: Path) -> None:
    # latest/, develop/ and the stray asset dirs at the gh-pages root share the same parent.
    for name in ("latest", "develop", "assets"):
        (tmp_path / name).mkdir()

    assert injector._release_version_dirs(tmp_path) == []


def test_stale_banner_is_cleared_from_the_current_release(injector: ModuleType, gh_pages: Path) -> None:
    # An earlier backfill treated every numbered version as superseded.
    page = gh_pages / "1.9.4" / "index.html"
    page.write_text(EMPTY_PAGE.replace("hidden>", f"hidden>{injector._aside(injector.ARCHIVED_TEXT)}"))

    injector.strip_from_current_release(gh_pages)

    assert "older version of RF-DETR" not in page.read_text(encoding="utf-8")


def test_stale_banner_styling_is_cleared_from_the_current_release(injector: ModuleType, gh_pages: Path) -> None:
    stylesheet = gh_pages / "1.9.4" / "stylesheets" / "rf.css"
    injector.patch_stylesheets(gh_pages)  # no-op for the current release
    stylesheet.write_text(
        f"body {{\n  color: red;\n}}\n\n{injector._CSS_MARKER_START}\nx\n{injector._CSS_MARKER_END}\n"
    )

    injector.strip_from_current_release(gh_pages)

    assert injector._CSS_MARKER_START not in stylesheet.read_text(encoding="utf-8")


def test_clearing_keeps_the_current_release_stylesheet_rules(injector: ModuleType, gh_pages: Path) -> None:
    stylesheet = gh_pages / "1.9.4" / "stylesheets" / "rf.css"
    stylesheet.write_text(
        f"body {{\n  color: red;\n}}\n\n{injector._CSS_MARKER_START}\nx\n{injector._CSS_MARKER_END}\n"
    )

    injector.strip_from_current_release(gh_pages)

    assert "color: red;" in stylesheet.read_text(encoding="utf-8")


def test_clearing_leaves_a_genuine_build_alone(injector: ModuleType, gh_pages: Path) -> None:
    page = gh_pages / "1.9.4" / "index.html"
    rebuilt = EMPTY_PAGE.replace("hidden>", 'hidden><aside class="md-banner">built</aside>')
    page.write_text(rebuilt, encoding="utf-8")

    injector.strip_from_current_release(gh_pages)

    assert page.read_text(encoding="utf-8") == rebuilt
