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

#: A synthetic ``pyproject.toml`` for the installed-mode command-line tests; its extras and groups need nothing beyond
#: ``pytest``, which the test environment always has.
_PYPROJECT = """
[project]
name = "pkg"
dependencies = []

[project.optional-dependencies]
Train = ["pytest"]

[dependency-groups]
Base_Test = ["pytest"]
tests = [{ include-group = "base-test" }]
"""


@pytest.fixture
def repo_copy(tmp_path: Path, repo_root: Path, install_flags: ModuleType) -> Path:
    """A minimal copy of the repository holding ``pyproject.toml`` and the probe modules."""
    for relative in ("pyproject.toml", *install_flags.PROBE_MODULES):
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(repo_root / relative, target)
    return tmp_path


@pytest.fixture
def synthetic_repo(repo_copy: Path) -> Path:
    """The repository copy with ``pyproject.toml`` replaced by ``_PYPROJECT``."""
    (repo_copy / "pyproject.toml").write_text(_PYPROJECT, encoding="utf-8")
    return repo_copy


class TestDeclaredMode:
    """Without arguments, every flag must be a literal probe of a package pyproject.toml declares."""

    def test_repository_passes(self, install_flags: ModuleType) -> None:
        """The flags and pyproject.toml in this checkout agree."""
        assert install_flags.main([]) == 0

    @pytest.mark.parametrize(
        ("module_index", "line", "flag"),
        [
            pytest.param(
                0,
                '_IS_TYPO_INSTALLED = is_installed("no_such_package_anywhere")',
                "_IS_TYPO_INSTALLED",
                id="undeclared",
            ),
            pytest.param(0, "_IS_COMPUTED_INSTALLED = is_installed(NAME)", "_IS_COMPUTED_INSTALLED", id="non-literal"),
            pytest.param(
                0,
                '_IS_ANNOTATED_INSTALLED: bool = is_installed("no_such_package_anywhere")',
                "_IS_ANNOTATED_INSTALLED",
                id="annotated-undeclared",
            ),
            pytest.param(0, "_IS_CONSTANT_INSTALLED: bool = False", "_IS_CONSTANT_INSTALLED", id="annotated-constant"),
            pytest.param(
                0,
                'try:\n    _IS_NESTED_INSTALLED = is_installed("peft")\nexcept ImportError:\n    pass',
                "_IS_NESTED_INSTALLED",
                id="try-nested",
            ),
            pytest.param(0, "from os import sep as _IS_IMPORTED_INSTALLED", "_IS_IMPORTED_INSTALLED", id="imported"),
            pytest.param(0, "del _IS_PEFT_INSTALLED", "_IS_PEFT_INSTALLED", id="deleted"),
            pytest.param(
                0, '_IS_HYPHEN_INSTALLED = is_installed("faster-coco-eval")', "_IS_HYPHEN_INSTALLED", id="hyphenated"
            ),
            pytest.param(
                1,
                '_IS_TYPO_INSTALLED = is_installed("no_such_package_anywhere")',
                "_IS_TYPO_INSTALLED",
                id="export-module",
            ),
        ],
    )
    def test_bad_probe_fails(
        self,
        install_flags: ModuleType,
        repo_copy: Path,
        capsys: pytest.CaptureFixture[str],
        module_index: int,
        line: str,
        flag: str,
    ) -> None:
        """A flag that no declared install can turn true fails the check and is named in the error.

        Each planted line leaves a flag whose gated tests would skip everywhere: an undeclared or non-importable
        package, a value that is not a literal probe, or a binding the source check cannot read.
        """
        module = repo_copy / install_flags.PROBE_MODULES[module_index]
        module.write_text(f"{module.read_text(encoding='utf-8')}\n{line}\n", encoding="utf-8")

        assert install_flags.main([], root=repo_copy) == 1
        assert flag in capsys.readouterr().err

    def test_alias_distribution_counts_as_declared(self, install_flags: ModuleType) -> None:
        """``cv2`` is declared through ``opencv-python-headless``, a distribution with another name."""
        errors = install_flags.check_declared(
            {"_IS_OPENCV_INSTALLED": "cv2"}, [Requirement("opencv-python-headless>=4")]
        )

        assert errors == []

    def test_onnxruntime_gpu_counts_as_onnxruntime(self, install_flags: ModuleType) -> None:
        """``[tensorrt]`` declares only ``onnxruntime-gpu``, which provides the ``onnxruntime`` import."""
        errors = install_flags.check_declared(
            {"_IS_ONNXRUNTIME_INSTALLED": "onnxruntime"}, [Requirement("onnxruntime-gpu")]
        )

        assert errors == []


class TestUndeclaredPackages:
    """The allowlist of probed but undeclared packages stays current."""

    def test_every_entry_is_probed(self, install_flags: ModuleType, repo_root: Path) -> None:
        """An entry no flag probes any more is stale and would hide a later typo of the same name."""
        probed = {
            package
            for module in install_flags.PROBE_MODULES
            for package in install_flags.read_probes(repo_root / module)[0].values()
        }

        assert set(install_flags.UNDECLARED_PACKAGES) <= probed

    def test_no_entry_is_declared(
        self, install_flags: ModuleType, repo_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An entry pyproject.toml now declares no longer needs its exemption from the declared check.

        With the allowlist emptied, the declared check must still report every former entry as undeclared.
        """
        pyproject = install_flags.tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))
        flags = {f"_IS_{package.upper()}_INSTALLED": package for package in install_flags.UNDECLARED_PACKAGES}
        monkeypatch.setattr(install_flags, "UNDECLARED_PACKAGES", {})

        errors = install_flags.check_declared(flags, install_flags.declared_requirements(pyproject))

        assert len(errors) == len(flags)


class TestDeclaredRequirements:
    """Extras, groups and self-references expand to the requirements an install would pull in."""

    def test_names_are_normalized(self, install_flags: ModuleType) -> None:
        """Extras and groups match in any PEP 685 / PEP 735 spelling, ``include-group`` entries included."""
        pyproject = install_flags.tomllib.loads(_PYPROJECT)

        requirements = install_flags.declared_requirements(pyproject, extras=["train"], groups=["TESTS"])

        assert [requirement.name for requirement in requirements] == ["pytest", "pytest"]

    def test_bare_self_reference_reads_core_dependencies(self, install_flags: ModuleType) -> None:
        """``pkg`` without extras stands for the core dependencies, not for nothing."""
        pyproject = {"project": {"name": "pkg", "dependencies": ["core"], "optional-dependencies": {"a": ["pkg"]}}}

        requirements = install_flags.declared_requirements(pyproject, extras=["a"], groups=[])

        assert [requirement.name for requirement in requirements] == ["core"]

    def test_mutual_self_references_end(self, install_flags: ModuleType) -> None:
        """Extras that reference each other are read once each instead of recursing without end."""
        pyproject = {"project": {"name": "pkg", "optional-dependencies": {"a": ["pkg[b]", "x"], "b": ["pkg[a]", "y"]}}}

        requirements = install_flags.declared_requirements(pyproject, extras=["a"], groups=[])

        assert [requirement.name for requirement in requirements] == ["y", "x"]

    def test_marker_excluded_self_reference_is_skipped_for_the_environment(self, install_flags: ModuleType) -> None:
        """A self-reference whose marker excludes this platform adds nothing an install here would pull in."""
        pyproject = {
            "project": {
                "name": "pkg",
                "optional-dependencies": {"a": ["x"], "b": ["pkg[a]; sys_platform == 'no-such-platform'"]},
            }
        }

        requirements = install_flags.declared_requirements(pyproject, extras=["b"], groups=[], environment=True)

        assert requirements == []

    def test_marker_excluded_self_reference_is_kept_when_declared(self, install_flags: ModuleType) -> None:
        """The declared check still sees what a platform-specific self-reference declares."""
        pyproject = {
            "project": {
                "name": "pkg",
                "optional-dependencies": {"a": ["x"], "b": ["pkg[a]; sys_platform == 'no-such-platform'"]},
            }
        }

        requirements = install_flags.declared_requirements(pyproject, extras=["b"], groups=[])

        assert [requirement.name for requirement in requirements] == ["x"]


class TestInstalledMode:
    """With extras or groups, their requirements must be installed and the flags probing them true."""

    def test_missing_distribution_is_reported(self, install_flags: ModuleType) -> None:
        """A declared but absent distribution is named."""
        errors = install_flags.check_installed({}, [Requirement("ghost-package-nowhere")])

        assert errors == ["ghost-package-nowhere is declared by the installed extras or groups but is not installed"]

    def test_false_flag_is_reported(self, install_flags: ModuleType) -> None:
        """A flag whose declared package cannot be imported is named."""
        errors = install_flags.check_installed(
            {"_IS_GHOST_INSTALLED": "ghost_package_nowhere"}, [Requirement("ghost-package-nowhere")]
        )

        assert "_IS_GHOST_INSTALLED is false although its package 'ghost_package_nowhere' is expected here" in errors

    def test_false_flag_is_reported_through_alias(
        self, install_flags: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A flag is expected true when the install declares a distribution that provides it under another name."""
        monkeypatch.setitem(install_flags.DISTRIBUTION_NAMES, "ghost_module_nowhere", ("ghost-dist-nowhere",))

        errors = install_flags.check_installed(
            {"_IS_GHOST_INSTALLED": "ghost_module_nowhere"}, [Requirement("ghost-dist-nowhere")]
        )

        assert "_IS_GHOST_INSTALLED is false although its package 'ghost_module_nowhere' is expected here" in errors

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

    def test_names_in_any_spelling_pass(self, install_flags: ModuleType, synthetic_repo: Path) -> None:
        """An extra and a group named in another spelling than pyproject.toml uses are found and checked."""
        assert install_flags.main(["--extras", "train", "--groups", "base_test"], root=synthetic_repo) == 0

    @pytest.mark.parametrize(
        ("argv", "message"),
        [
            pytest.param(["--extras", ""], "--extras was given but names nothing", id="empty-extras"),
            pytest.param(["--groups", " , "], "--groups was given but names nothing", id="empty-groups"),
            pytest.param(["--extras", "nope"], "--extras names nope", id="unknown-extra"),
            pytest.param(["--groups", "nope"], "--groups names nope", id="unknown-group"),
        ],
    )
    def test_refused_option_exits_with_usage_error(
        self,
        install_flags: ModuleType,
        synthetic_repo: Path,
        capsys: pytest.CaptureFixture[str],
        argv: list[str],
        message: str,
    ) -> None:
        """An option that names nothing, or a name pyproject.toml does not declare, stops with status 2.

        An empty value would otherwise pass while checking nothing, and an unknown one end in a bare ``KeyError``.
        """
        with pytest.raises(SystemExit) as exit_info:
            install_flags.main(argv, root=synthetic_repo)

        assert exit_info.value.code == 2
        assert message in capsys.readouterr().err
