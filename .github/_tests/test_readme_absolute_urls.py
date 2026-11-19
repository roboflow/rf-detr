# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the README relative-path rewrite run before packaging."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "readme_absolute_urls.py"
RAW = "https://raw.githubusercontent.com/org/repo/abc123"
BLOB = "https://github.com/org/repo/blob/abc123"
DOCS = "https://docs.example/1.2.3"


@pytest.fixture(scope="module")
def rewriter() -> ModuleType:
    """The README rewrite script, loaded by path.

    Examples:
        >>> rewriter  # doctest: +SKIP
        pytest fixture; loads .github/scripts/readme_absolute_urls.py by path.
    """
    spec = importlib.util.spec_from_file_location("readme_absolute_urls", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    """A repository root holding one chart and a LICENSE.

    Examples:
        >>> checkout  # doctest: +SKIP
        pytest fixture; builds the tree under tmp_path.
    """
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "chart.svg").write_text("<svg/>")
    (tmp_path / "docs" / "snippets").mkdir()
    (tmp_path / "docs" / "snippets" / "note.png").write_text("")
    (tmp_path / "LICENSE").write_text("")
    return tmp_path


def run_cli(readme: Path, *extra: str) -> subprocess.CompletedProcess[str]:
    """Run the rewrite script on a README the way the build workflow does.

    Examples:
        >>> callable(run_cli)
        True
    """
    command = [sys.executable, str(SCRIPT), str(readme), "--repo", "org/repo", "--ref", "abc123", *extra]
    return subprocess.run(command, text=True, capture_output=True)


class TestDocsVersion:
    @pytest.mark.parametrize(
        ("package_version", "expected"),
        [
            ("1.12.0", "1.12.0"),
            ("1.12.0.post2", "1.12.0"),
            ("1.13.0rc1", None),
            ("1.13.0a0", None),
            ("1.13.0.dev1", None),
        ],
    )
    def test_maps_package_to_docs_version(
        self, rewriter: ModuleType, package_version: str, expected: str | None
    ) -> None:
        """Stable and post releases map to their docs label; pre-releases have none."""
        assert rewriter.docs_version(package_version) == expected

    def test_reads_repository_version(self, rewriter: ModuleType, repo_root: Path) -> None:
        """The static pyproject version is found under the [project] table."""
        assert re.fullmatch(r"\d+\.\d+\.\d+.*", rewriter.project_version(repo_root / "pyproject.toml"))


class TestAbsolutize:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('<img src="docs/assets/chart.svg" />', f'<img src="{RAW}/docs/assets/chart.svg" />'),
            ("![chart](./docs/assets/chart.svg)", f"![chart]({RAW}/docs/assets/chart.svg)"),
            ("[License](LICENSE)", f"[License]({BLOB}/LICENSE)"),
            ('<a href="LICENSE">x</a>', f'<a href="{BLOB}/LICENSE">x</a>'),
            ("[top](#license)", "[top](#license)"),
            ('<img src="https://example.com/a.png" />', '<img src="https://example.com/a.png" />'),
            ("```\n![chart](missing.svg)\n```\n", "```\n![chart](missing.svg)\n```\n"),
        ],
    )
    def test_rewrites(self, rewriter: ModuleType, checkout: Path, text: str, expected: str) -> None:
        """Without a docs site, images pin to the commit and other files link to the blob."""
        assert rewriter.absolutize(text, "org/repo", "abc123", checkout)[0] == expected

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ('<img src="docs/assets/chart.svg" />', f'<img src="{DOCS}/assets/chart.svg" />'),
            ("![note](docs/snippets/note.png)", f"![note]({RAW}/docs/snippets/note.png)"),
            ("[License](LICENSE)", f"[License]({BLOB}/LICENSE)"),
        ],
    )
    def test_rewrites_with_docs_site(self, rewriter: ModuleType, checkout: Path, text: str, expected: str) -> None:
        """Published docs images go to the docs site; excluded docs dirs and non-images stay on GitHub."""
        assert rewriter.absolutize(text, "org/repo", "abc123", checkout, DOCS)[0] == expected

    def test_missing_target_raises(self, rewriter: ModuleType, checkout: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gone.svg"):
            rewriter.absolutize("![x](docs/assets/gone.svg)", "org/repo", "abc123", checkout)


class TestRepositoryReadme:
    def test_no_relative_path_survives(self, rewriter: ModuleType, repo_root: Path) -> None:
        text = (repo_root / "README.md").read_text(encoding="utf-8")
        rewritten, _ = rewriter.absolutize(text, "roboflow/rf-detr", "abc123", repo_root)
        remaining = [t for t in re.findall(r'(?:src|href)="([^"]+)"', rewritten) if rewriter.is_relative(t)]
        assert remaining == []

    @pytest.mark.parametrize("name", ["detection", "segmentation", "keypoints"])
    def test_benchmark_chart_points_at_docs_site(self, rewriter: ModuleType, repo_root: Path, name: str) -> None:
        """Each benchmark chart resolves to the versioned docs-site copy MkDocs publishes."""
        text = (repo_root / "README.md").read_text(encoding="utf-8")
        rewritten, _ = rewriter.absolutize(text, "roboflow/rf-detr", "abc123", repo_root, DOCS)
        assert f'src="{DOCS}/assets/benchmark-{name}.svg"' in rewritten


class TestCommandLine:
    def test_rewrites_in_place(self, checkout: Path) -> None:
        """Without --docs-url the README is rewritten in place against the commit."""
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/chart.svg)\n", encoding="utf-8")
        run_cli(readme).check_returncode()
        assert readme.read_text(encoding="utf-8") == f"![chart]({RAW}/docs/assets/chart.svg)\n"

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("1.2.3.post1", "![chart](https://docs.example/1.2.3/assets/chart.svg)\n"),
            ("1.2.3rc1", f"![chart]({RAW}/docs/assets/chart.svg)\n"),
        ],
    )
    def test_docs_url_follows_project_version(self, checkout: Path, version: str, expected: str) -> None:
        """--docs-url takes the label from pyproject; a pre-release falls back to the commit."""
        (checkout / "pyproject.toml").write_text(f'[project]\nname = "x"\nversion = "{version}"\n')
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/chart.svg)\n", encoding="utf-8")
        run_cli(readme, "--docs-url", "https://docs.example/").check_returncode()
        assert readme.read_text(encoding="utf-8") == expected

    def test_missing_target_exits_nonzero(self, checkout: Path) -> None:
        """A dangling relative path fails the run and leaves the README untouched."""
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/gone.svg)\n", encoding="utf-8")
        result = run_cli(readme)
        assert (result.returncode, readme.read_text(encoding="utf-8")) == (1, "![chart](docs/assets/gone.svg)\n")
