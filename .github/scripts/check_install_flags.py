# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Check the ``_IS_<PACKAGE>_INSTALLED`` probe flags against the packages ``pyproject.toml`` declares.

Purpose:
    Tests that need an optional package are gated on a probe flag and skip when it is false. Two mistakes make such a
    test skip forever without failing anything: a flag that probes a mistyped package name, and a CI job that never
    installed the package its extras declare. This script catches both.
Modes:
    Without arguments (the ``install-flags-declared`` pre-commit hook), every flag in the probe modules must be a
    literal ``is_installed("<name>")`` call whose package some extra or dependency group of ``pyproject.toml``
    declares. With ``--extras`` or ``--groups`` (a CI step right after the install step), every requirement those
    extras and groups declare for the running interpreter and platform must be installed, and every flag that probes
    one of them must be true, so the tests it gates run instead of skipping.
Usage:
    python .github/scripts/check_install_flags.py
    python .github/scripts/check_install_flags.py --extras train,augment,cli,visual,data --groups tests
"""

from __future__ import annotations

import argparse
import ast
import importlib.metadata
import importlib.util
import re
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
#: Modules that define the probe flags, relative to the repository root.
PROBE_MODULES = ("src/rfdetr/utilities/imports.py", "src/rfdetr/export/imports.py")
#: Probed packages that no extra or group declares, each with the reason.
UNDECLARED_PACKAGES = {
    "pytorch_optimizer": "installed by hand for its optimizers",
    "nncf": "OpenVINO INT8 only; docs/exports/openvino.md asks for a separate `pip install nncf`",
    "tflite_runtime": "an alternative TFLite runtime users install themselves",
    "ai_edge_litert": "pulled in by onnx2tf ([tflite]) and litert-torch ([litert]), each pinning its own version",
}
#: Import names provided by a distribution spelled differently; any one of the listed distributions provides it.
DISTRIBUTION_NAMES = {"cv2": ("opencv-python", "opencv-python-headless"), "coreai": ("coreai-torch",)}
_FLAG_NAME = re.compile(r"_IS_[A-Z0-9_]+_INSTALLED")


def read_probes(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read the flags a probe module assigns and the import name each one probes.

    The names come from the source, not from importing the module, so a typo shows up on a host without the package.

    Args:
        path: Probe module to parse.

    Returns:
        Flag name to probed import name, and one error per flag that is not a literal ``is_installed("<name>")``.

    Examples:
        >>> probes, errors = read_probes(REPO_ROOT / "src/rfdetr/utilities/imports.py")
        >>> probes["_IS_PEFT_INSTALLED"], errors
        ('peft', [])
    """
    probes: dict[str, str] = {}
    errors: list[str] = []
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        targets = [target.id for target in getattr(node, "targets", []) if isinstance(target, ast.Name)]
        for flag in filter(_FLAG_NAME.fullmatch, targets):
            value = node.value
            literal = (
                isinstance(value, ast.Call)
                and getattr(value.func, "id", None) == "is_installed"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant)
                and isinstance(value.args[0].value, str)
            )
            if literal:
                probes[flag] = value.args[0].value
            else:
                errors.append(f"{path}: {flag} is not a literal is_installed('<name>') probe")
    return probes, errors


def distributions_for(package: str) -> tuple[str, ...]:
    """Name the canonical distributions that provide an import name.

    Args:
        package: Probed import name, e.g. ``"torch_xla"``.

    Returns:
        Canonical distribution names, any one of which provides the import.

    Examples:
        >>> distributions_for("pytorch_lightning")
        ('pytorch-lightning',)
        >>> distributions_for("cv2")
        ('opencv-python', 'opencv-python-headless')
    """
    return tuple(canonicalize_name(name) for name in DISTRIBUTION_NAMES.get(package, (package,)))


def _group_requirements(groups: dict[str, list[Any]], name: str) -> list[str]:
    """Flatten one dependency group, following ``include-group`` entries.

    Examples:
        >>> _group_requirements({"a": ["x"], "b": ["y", {"include-group": "a"}]}, "b")
        ['y', 'x']
    """
    flat: list[str] = []
    for entry in groups[name]:
        flat.extend(_group_requirements(groups, entry["include-group"]) if isinstance(entry, dict) else [entry])
    return flat


def declared_requirements(
    pyproject: dict[str, Any], extras: Iterable[str] | None = None, groups: Iterable[str] | None = None
) -> list[Requirement]:
    """Collect the requirements some extras and dependency groups declare, expanding self-references.

    Args:
        pyproject: Parsed ``pyproject.toml``.
        extras: Extras to read; ``None`` reads every extra and the core dependencies.
        groups: Dependency groups to read; ``None`` reads every group.

    Returns:
        The requirements, in declaration order.

    Raises:
        KeyError: If a named extra or group does not exist.

    Examples:
        >>> project = {"project": {"name": "pkg", "optional-dependencies": {"a": ["x>=1"], "b": ["pkg[a]", "y"]}}}
        >>> [req.name for req in declared_requirements(project, extras=["b"], groups=[])]
        ['x', 'y']
    """
    project = pyproject["project"]
    optional = project.get("optional-dependencies", {})
    all_groups = pyproject.get("dependency-groups", {})
    raw = list(project.get("dependencies", [])) if extras is None else []
    for extra in optional if extras is None else extras:
        raw.extend(optional[extra])
    for group in all_groups if groups is None else groups:
        raw.extend(_group_requirements(all_groups, group))
    requirements: list[Requirement] = []
    for line in raw:
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) == canonicalize_name(project["name"]):
            requirements.extend(declared_requirements(pyproject, extras=requirement.extras, groups=[]))
        else:
            requirements.append(requirement)
    return requirements


def check_declared(probes: dict[str, str], requirements: Iterable[Requirement]) -> list[str]:
    """Report every flag whose probed package no requirement declares.

    Args:
        probes: Flag name to probed import name.
        requirements: Every requirement of the project.

    Returns:
        One error per undeclared package.

    Examples:
        >>> check_declared({"_IS_X_INSTALLED": "x_pkg"}, [Requirement("x-pkg>=1")])
        []
        >>> check_declared({"_IS_X_INSTALLED": "typo_pkg"}, [Requirement("x-pkg>=1")])
        ["_IS_X_INSTALLED probes 'typo_pkg', which pyproject.toml does not declare"]
    """
    declared = {canonicalize_name(requirement.name) for requirement in requirements}
    return [
        f"{flag} probes {package!r}, which pyproject.toml does not declare"
        for flag, package in probes.items()
        if package not in UNDECLARED_PACKAGES and declared.isdisjoint(distributions_for(package))
    ]


def check_installed(probes: dict[str, str], requirements: Iterable[Requirement]) -> list[str]:
    """Report every requirement this environment lacks and every flag it would leave false.

    Requirements whose environment marker excludes the running interpreter or platform are skipped.

    Args:
        probes: Flag name to probed import name.
        requirements: Requirements the installed extras and groups declare.

    Returns:
        One error per missing distribution and per false flag.

    Examples:
        >>> check_installed({}, [Requirement("pytest")])
        []
        >>> check_installed({}, [Requirement("no-such-dist-anywhere; sys_platform == 'nowhere'")])
        []
    """
    active = {
        canonicalize_name(requirement.name)
        for requirement in requirements
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""})
    }
    errors = []
    for name in sorted(active):
        try:
            importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            errors.append(f"{name} is declared by the installed extras or groups but is not installed")
    for flag, package in probes.items():
        if not active.isdisjoint(distributions_for(package)) and not _is_importable(package):
            errors.append(f"{flag} is false although its package {package!r} is expected here")
    return errors


def _is_importable(name: str) -> bool:
    """Mirror ``rfdetr.utilities.package.is_installed`` without importing ``rfdetr``.

    Examples:
        >>> _is_importable("json"), _is_importable("no_such_module_anywhere")
        (True, False)
    """
    try:
        return getattr(importlib.util.find_spec(name), "origin", None) is not None
    except (ImportError, ValueError):
        return False


def _split(value: str | None) -> list[str] | None:
    """Split a comma-separated option, keeping ``None`` for an option not given.

    Examples:
        >>> _split("a, b,"), _split(None)
        (['a', 'b'], None)
    """
    return None if value is None else [item.strip() for item in value.split(",") if item.strip()]


def main(argv: Sequence[str] | None = None, root: Path = REPO_ROOT) -> int:
    """Run the declared check, or the installed check when extras or groups are given.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        root: Repository root holding ``pyproject.toml`` and the probe modules.

    Returns:
        ``0`` when every check passes, ``1`` otherwise.

    Examples:
        >>> main([])
        0
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--extras", help="comma-separated extras the environment installed")
    parser.add_argument("--groups", help="comma-separated dependency groups the environment installed")
    args = parser.parse_args(argv)
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    probes: dict[str, str] = {}
    errors: list[str] = []
    for module in PROBE_MODULES:
        module_probes, module_errors = read_probes(root / module)
        probes.update(module_probes)
        errors.extend(module_errors)
    extras, groups = _split(args.extras), _split(args.groups)
    if extras is None and groups is None:
        errors.extend(check_declared(probes, declared_requirements(pyproject)))
    else:
        errors.extend(check_installed(probes, declared_requirements(pyproject, extras or [], groups or [])))
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
