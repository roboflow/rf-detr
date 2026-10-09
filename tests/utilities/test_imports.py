# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Guards for the ``_IS_<PACKAGE>_INSTALLED`` probe flags that gate optional-dependency tests."""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path
from types import ModuleType

import pytest

from rfdetr.export import imports as export_imports
from rfdetr.utilities import imports as utilities_imports

_PYPROJECT_TEXT = Path(__file__).resolve().parents[2].joinpath("pyproject.toml").read_text(encoding="utf-8")

#: Probed packages that no pyproject extra declares, because users or CI install them by hand.
_UNDECLARED_PACKAGES = frozenset({"pytorch_optimizer"})


def _probed_packages(module: ModuleType) -> dict[str, str]:
    """Map each flag a module assigns from ``is_installed("<package>")`` to its literal package name.

    The names come from the module source rather than from the imported flag values, so a typo in a package string
    shows up here even on a host where the real package is absent.

    Args:
        module: Import module whose top-level assignments are read.

    Returns:
        Flag name to the import name passed to ``is_installed``.

    Examples:
        >>> from rfdetr.utilities import imports
        >>> _probed_packages(imports)["_IS_PEFT_INSTALLED"]
        'peft'
    """
    probes: dict[str, str] = {}
    for node in ast.parse(inspect.getsource(module)).body:
        is_probe = (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and getattr(node.value.func, "id", None) == "is_installed"
        )
        if is_probe:
            probes[node.targets[0].id] = ast.literal_eval(node.value.args[0])
    return probes


def _is_declared_in_pyproject(package: str) -> bool:
    """Report whether ``pyproject.toml`` names a package as a quoted requirement, ignoring ``-``/``_``/``.`` spelling.

    Args:
        package: Import name of the package, e.g. ``"torch_xla"``.

    Returns:
        ``True`` when a quoted requirement starts with the package name.

    Examples:
        >>> _is_declared_in_pyproject("pytorch_lightning")
        True
        >>> _is_declared_in_pyproject("no_such_package_anywhere")
        False
    """
    spelling = re.escape(package).replace("_", "[-_.]")
    return re.search(rf"""(?i)["']{spelling}\b""", _PYPROJECT_TEXT) is not None


_PROBES = [
    (flag, package)
    for module in (export_imports, utilities_imports)
    for flag, package in _probed_packages(module).items()
]


class TestInstallFlags:
    """Install-probe flags are plain probes with a package name that matches the project's declared dependencies."""

    @pytest.mark.parametrize(
        "module", [pytest.param(export_imports, id="export"), pytest.param(utilities_imports, id="utilities")]
    )
    def test_every_install_flag_is_a_literal_is_installed_probe(self, module: ModuleType) -> None:
        """Each ``_IS_*_INSTALLED`` name in a module comes from ``is_installed("<literal>")``.

        A flag built another way (a different helper, a computed name) would escape the package-name check below, so the
        set of flags and the set of literal probes must be the same.
        """
        flags = {name for name in vars(module) if re.fullmatch(r"_IS_[A-Z0-9_]+_INSTALLED", name)}

        assert flags == set(_probed_packages(module))

    @pytest.mark.parametrize(
        ("flag", "package"),
        [pytest.param(flag, package, id=flag) for flag, package in _PROBES if package not in _UNDECLARED_PACKAGES],
    )
    def test_probed_package_is_declared_in_pyproject(self, flag: str, package: str) -> None:
        """The package a flag probes is a dependency that ``pyproject.toml`` declares in some extra or group.

        A mistyped import name makes the flag permanently false, so every test it gates is skipped on every host and the
        typo never fails anything. Reading the name from pyproject.toml checks it independently of the flag.
        """
        assert _is_declared_in_pyproject(package), f"{flag} probes {package!r}, which pyproject.toml does not declare"
