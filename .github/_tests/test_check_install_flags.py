# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ``_IS_<PACKAGE>_INSTALLED`` probe-flag check script."""

import shutil
from pathlib import Path
from types import ModuleType

import pytest
from packaging.requirements import Requirement


@pytest.fixture
def repo_copy(tmp_path: Path, repo_root: Path, install_flags: ModuleType) -> Path:
    """A minimal copy of the repository holding ``pyproject.toml`` and the probe modules.

    Examples:
        >>> repo_copy  # doctest: +SKIP
        pytest fixture; copies the files the check reads into a temporary root.
    """
    for relative in ("pyproject.toml", *install_flags.PROBE_MODULES):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_root / relative, target)
    return tmp_path


class TestDeclaredMode:
    """Without arguments, every flag must be a literal probe of a package pyproject.toml declares."""

    def test_repository_passes(self, install_flags: ModuleType) -> None:
        """The flags and pyproject.toml in this checkout agree."""
        assert install_flags.main([]) == 0

    @pytest.mark.parametrize(
        "line",
        [
            pytest.param('_IS_TYPO_INSTALLED = is_installed("no_such_package_anywhere")', id="undeclared-package"),
            pytest.param("_IS_COMPUTED_INSTALLED = is_installed(NAME)", id="non-literal-probe"),
        ],
    )
    def test_bad_probe_fails(self, install_flags: ModuleType, repo_copy: Path, line: str) -> None:
        """A planted probe of an undeclared package, or one built from a variable, fails the check.

        Both leave a flag that no declared install can turn true, so every test it gates would skip everywhere.
        """
        module = repo_copy / install_flags.PROBE_MODULES[0]
        module.write_text(f"{module.read_text(encoding='utf-8')}\n{line}\n", encoding="utf-8")

        assert install_flags.main([], root=repo_copy) == 1

    def test_alias_distribution_counts_as_declared(self, install_flags: ModuleType) -> None:
        """``cv2`` is declared through ``opencv-python-headless``, a distribution with another name."""
        errors = install_flags.check_declared(
            {"_IS_OPENCV_INSTALLED": "cv2"}, [Requirement("opencv-python-headless>=4")]
        )

        assert errors == []


class TestInstalledMode:
    """With extras or groups, their requirements must be installed and the flags probing them true."""

    def test_missing_distribution_and_false_flag_are_reported(self, install_flags: ModuleType) -> None:
        """A declared but absent package is named, and so is the flag it leaves false."""
        errors = install_flags.check_installed(
            {"_IS_GHOST_INSTALLED": "ghost_package_nowhere"}, [Requirement("ghost-package-nowhere")]
        )

        assert errors == [
            "ghost-package-nowhere is declared by the installed extras or groups but is not installed",
            "_IS_GHOST_INSTALLED is false although its package 'ghost_package_nowhere' is expected here",
        ]

    def test_requirement_excluded_by_marker_is_skipped(self, install_flags: ModuleType) -> None:
        """A requirement whose marker excludes this platform is not expected to be installed."""
        errors = install_flags.check_installed(
            {"_IS_GHOST_INSTALLED": "ghost_package_nowhere"},
            [Requirement("ghost-package-nowhere; sys_platform == 'no-such-platform'")],
        )

        assert errors == []

    def test_installed_group_passes(self, install_flags: ModuleType) -> None:
        """The ``tests`` group this job installed is all present."""
        assert install_flags.main(["--groups", "tests"]) == 0
