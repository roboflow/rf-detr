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
    (tmp_path / "LICENSE").write_text("")
    return tmp_path


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
        assert rewriter.absolutize(text, "org/repo", "abc123", checkout)[0] == expected

    def test_missing_target_raises(self, rewriter: ModuleType, checkout: Path) -> None:
        with pytest.raises(FileNotFoundError, match="gone.svg"):
            rewriter.absolutize("![x](docs/assets/gone.svg)", "org/repo", "abc123", checkout)


class TestRepositoryReadme:
    def test_no_relative_path_survives(self, rewriter: ModuleType, repo_root: Path) -> None:
        text = (repo_root / "README.md").read_text(encoding="utf-8")
        rewritten, _ = rewriter.absolutize(text, "roboflow/rf-detr", "abc123", repo_root)
        remaining = [t for t in re.findall(r'(?:src|href)="([^"]+)"', rewritten) if rewriter.is_relative(t)]
        assert remaining == []

    def test_benchmark_charts_become_raw_urls(self, rewriter: ModuleType, repo_root: Path) -> None:
        text = (repo_root / "README.md").read_text(encoding="utf-8")
        rewritten, _ = rewriter.absolutize(text, "roboflow/rf-detr", "abc123", repo_root)
        for name in ("detection", "segmentation", "keypoints"):
            url = f"https://raw.githubusercontent.com/roboflow/rf-detr/abc123/docs/assets/benchmark-{name}.svg"
            assert f'src="{url}"' in rewritten


class TestCommandLine:
    def test_rewrites_in_place(self, checkout: Path) -> None:
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/chart.svg)\n", encoding="utf-8")
        subprocess.run(
            [sys.executable, str(SCRIPT), str(readme), "--repo", "org/repo", "--ref", "abc123"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert readme.read_text(encoding="utf-8") == f"![chart]({RAW}/docs/assets/chart.svg)\n"

    def test_missing_target_exits_nonzero(self, checkout: Path) -> None:
        readme = checkout / "README.md"
        readme.write_text("![chart](docs/assets/gone.svg)\n", encoding="utf-8")
        result = subprocess.run(
            [sys.executable, str(SCRIPT), str(readme), "--repo", "org/repo", "--ref", "abc123"],
            text=True,
            capture_output=True,
        )
        assert result.returncode == 1
        assert readme.read_text(encoding="utf-8") == "![chart](docs/assets/gone.svg)\n"
