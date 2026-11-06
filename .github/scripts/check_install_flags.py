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
from collections.abc import Callable, Iterable, Iterator, Sequence
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
DISTRIBUTION_NAMES = {
    "cv2": ("opencv-python", "opencv-python-headless"),
    "coreai": ("coreai-torch",),
    "onnxruntime": ("onnxruntime", "onnxruntime-gpu"),
}
#: Module that defines ``is_installed``, the function every probe flag calls.
PACKAGE_MODULE = "src/rfdetr/utilities/package.py"
_FLAG_NAME = re.compile(r"_IS_[A-Z0-9_]+_INSTALLED")


def _probed_module(value: ast.expr | None) -> str | None:
    """Return the module a literal ``is_installed("<module>")`` call probes, or ``None`` for any other value.

    A hyphenated or otherwise non-importable name is rejected: it canonicalizes to a declared distribution, yet no
    import can ever find it, so the flag would stay false.

    Examples:
        >>> _probed_module(ast.parse('is_installed("torch_xla")', mode="eval").body)
        'torch_xla'
        >>> _probed_module(ast.parse('is_installed("faster-coco-eval")', mode="eval").body) is None
        True
    """
    if not (
        isinstance(value, ast.Call)
        and getattr(value.func, "id", None) == "is_installed"
        and len(value.args) == 1
        and not value.keywords
        and isinstance(value.args[0], ast.Constant)
        and isinstance(value.args[0].value, str)
    ):
        return None
    module = value.args[0].value
    return module if all(part.isidentifier() for part in module.split(".")) else None


def _bound_names(tree: ast.AST) -> Iterator[ast.AST]:
    """Yield every node that binds or deletes a name: assignment and loop targets, ``del``, and imports.

    Examples:
        >>> [getattr(node, "id", None) or node.asname for node in _bound_names(ast.parse("a = 1\\nimport b as c"))]
        ['a', 'c']
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and not isinstance(node.ctx, ast.Load):
            yield node
        elif isinstance(node, ast.alias):
            yield node


def read_probes(path: Path) -> tuple[dict[str, str], list[str]]:
    """Read the flags a probe module assigns and the import name each one probes.

    The names come from the source, not from importing the module, so a typo shows up on a host without the package.
    A flag counts as a probe only as the single target of a top-level assignment, annotated or not; a flag bound any
    other way (nested under ``if``/``try``, imported, deleted, unpacked) is reported, since the check cannot see what
    it holds.

    Args:
        path: Probe module to parse.

    Returns:
        Flag name to probed import name, and one error per flag that is not a literal ``is_installed("<name>")``.

    Examples:
        >>> probes, errors = read_probes(REPO_ROOT / "src/rfdetr/utilities/imports.py")
        >>> probes["_IS_PEFT_INSTALLED"], errors
        ('peft', [])
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    probes: dict[str, str] = {}
    errors: list[str] = []
    checked: set[int] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        else:
            continue
        if not (isinstance(target, ast.Name) and _FLAG_NAME.fullmatch(target.id)):
            continue
        checked.add(id(target))
        module = _probed_module(value)
        if module is None:
            errors.append(f"{path}: {target.id} is not a literal is_installed('<module>') probe")
        else:
            probes[target.id] = module
    for node in _bound_names(tree):
        name = node.id if isinstance(node, ast.Name) else node.asname or node.name.split(".")[0]
        if _FLAG_NAME.fullmatch(name) and id(node) not in checked:
            errors.append(f"{path}: {name} is bound other than by a top-level literal is_installed('<module>') probe")
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


def declared_names(pyproject: dict[str, Any]) -> tuple[dict[str, list[Any]], dict[str, list[Any]]]:
    """Map the extras and the dependency groups of ``pyproject.toml`` by their normalized names.

    Extras (PEP 685) and dependency groups (PEP 735) compare by normalized name, so ``Train`` and ``base_test`` name
    the same extra or group as ``train`` and ``base-test``.

    Args:
        pyproject: Parsed ``pyproject.toml``.

    Returns:
        Normalized extra name to its requirements, and normalized group name to its entries.

    Examples:
        >>> declared_names({"project": {"optional-dependencies": {"Train": ["x"]}}, "dependency-groups": {"A_b": []}})
        ({'train': ['x']}, {'a-b': []})
    """
    optional = pyproject["project"].get("optional-dependencies", {})
    groups = pyproject.get("dependency-groups", {})
    return (
        {canonicalize_name(name): entries for name, entries in optional.items()},
        {canonicalize_name(name): entries for name, entries in groups.items()},
    )


def _group_requirements(groups: dict[str, list[Any]], name: str) -> list[str]:
    """Flatten one dependency group, following ``include-group`` entries.

    Args:
        groups: Normalized group name to its entries, as :func:`declared_names` returns them.
        name: Group to flatten, in any spelling.

    Examples:
        >>> _group_requirements({"a-b": ["x"], "c": ["y", {"include-group": "A_b"}]}, "c")
        ['y', 'x']
    """
    flat: list[str] = []
    for entry in groups[canonicalize_name(name)]:
        flat.extend(_group_requirements(groups, entry["include-group"]) if isinstance(entry, dict) else [entry])
    return flat


def declared_requirements(
    pyproject: dict[str, Any],
    extras: Iterable[str] | None = None,
    groups: Iterable[str] | None = None,
    environment: bool = False,
) -> list[Requirement]:
    """Collect the requirements some extras and dependency groups declare, expanding self-references.

    A self-reference such as ``rfdetr[train]`` stands for the core dependencies plus the named extras. Each extra is
    read once, so extras that reference each other end instead of recursing.

    Args:
        pyproject: Parsed ``pyproject.toml``.
        extras: Extras to read; ``None`` reads every extra and the core dependencies.
        groups: Dependency groups to read; ``None`` reads every group.
        environment: Skip a self-reference whose environment marker excludes the running interpreter or platform, as
            an installer would; leave it ``False`` to collect every requirement regardless of platform.

    Returns:
        The requirements, in declaration order.

    Raises:
        KeyError: If a named extra or group does not exist.

    Examples:
        >>> project = {"project": {"name": "pkg", "optional-dependencies": {"a": ["x>=1"], "b": ["pkg[a]", "y"]}}}
        >>> [req.name for req in declared_requirements(project, extras=["b"], groups=[])]
        ['x', 'y']
        >>> cyclic = {"project": {"name": "pkg", "optional-dependencies": {"a": ["pkg[b]", "x"], "b": ["pkg[a]"]}}}
        >>> [req.name for req in declared_requirements(cyclic, extras=["a"], groups=[])]
        ['x']
    """
    optional, all_groups = declared_names(pyproject)
    raw = list(pyproject["project"].get("dependencies", [])) if extras is None else []
    read = {"", *optional} if extras is None else {canonicalize_name(extra) for extra in extras}
    for extra in optional if extras is None else extras:
        raw.extend(optional[canonicalize_name(extra)])
    for group in all_groups if groups is None else groups:
        raw.extend(_group_requirements(all_groups, group))
    return list(_expand_self_references(pyproject, optional, raw, read, environment))


def _expand_self_references(
    pyproject: dict[str, Any], optional: dict[str, list[str]], raw: Iterable[str], read: set[str], environment: bool
) -> Iterator[Requirement]:
    """Yield the requirements of ``raw``, replacing each self-reference by the extras it names that are not yet read.

    Args:
        pyproject: Parsed ``pyproject.toml``.
        optional: Normalized extra name to its requirements.
        raw: Requirement strings to expand.
        read: Extras already collected, ``""`` standing for the core dependencies; extended in place.
        environment: Skip a self-reference whose marker excludes the running environment.

    Examples:
        >>> project = {"project": {"name": "pkg", "dependencies": ["core"], "optional-dependencies": {"a": ["x"]}}}
        >>> [req.name for req in _expand_self_references(project, {"a": ["x"]}, ["pkg", "y"], set(), False)]
        ['core', 'y']
        >>> skipped = ["pkg[a]; sys_platform == 'nowhere'"]
        >>> [req.name for req in _expand_self_references(project, {"a": ["x"]}, skipped, set(), True)]
        []
    """
    name = canonicalize_name(pyproject["project"]["name"])
    for line in raw:
        requirement = Requirement(line)
        if canonicalize_name(requirement.name) != name:
            yield requirement
            continue
        if environment and requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        for extra in ["", *sorted(canonicalize_name(extra) for extra in requirement.extras)]:
            if extra in read:
                continue
            read.add(extra)
            nested = pyproject["project"].get("dependencies", []) if extra == "" else optional[extra]
            yield from _expand_self_references(pyproject, optional, nested, read, environment)


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
    is_installed = _load_is_installed()
    for flag, package in probes.items():
        if not active.isdisjoint(distributions_for(package)) and not is_installed(package):
            errors.append(f"{flag} is false although its package {package!r} is expected here")
    return errors


def _load_is_installed() -> Callable[[str], bool]:
    """Load ``is_installed``, the function every probe flag calls, by path so the ``rfdetr`` package is not imported.

    The installed check then asks the same function the flags do, never a copy that could drift from it.

    Raises:
        ImportError: If the module cannot be loaded from its path.

    Examples:
        >>> is_installed = _load_is_installed()
        >>> is_installed("json"), is_installed("no_such_module_anywhere")
        (True, False)
    """
    spec = importlib.util.spec_from_file_location("_rfdetr_utilities_package", REPO_ROOT / PACKAGE_MODULE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {PACKAGE_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.is_installed


def _checked_names(
    parser: argparse.ArgumentParser, option: str, value: str | None, declared: dict[str, Any]
) -> list[str] | None:
    """Split a comma-separated option and refuse an empty list or a name ``pyproject.toml`` does not declare.

    An empty value would put the check in installed mode with nothing to check, so it would pass without testing
    anything; an unknown name would otherwise end in a bare ``KeyError``.

    Args:
        parser: Parser that reports a refused value and exits with status 2.
        option: Option name, for the message.
        value: Raw option value; ``None`` when the option was not given.
        declared: Normalized names the option may take.

    Returns:
        The names, or ``None`` for an option not given.

    Examples:
        >>> _checked_names(argparse.ArgumentParser(), "--extras", "train, Cli,", {"train": [], "cli": []})
        ['train', 'Cli']
        >>> _checked_names(argparse.ArgumentParser(), "--extras", None, {}) is None
        True
    """
    if value is None:
        return None
    names = [item.strip() for item in value.split(",") if item.strip()]
    if not names:
        parser.error(f"{option} was given but names nothing")
    unknown = [name for name in names if canonicalize_name(name) not in declared]
    if unknown:
        parser.error(
            f"{option} names {', '.join(unknown)}, which pyproject.toml does not declare; known: {', '.join(declared)}"
        )
    return names


def main(argv: Sequence[str] | None = None, root: Path = REPO_ROOT) -> int:
    """Run the declared check, or the installed check when extras or groups are given.

    Args:
        argv: Command-line arguments; ``None`` reads ``sys.argv``.
        root: Repository root holding ``pyproject.toml`` and the probe modules.

    Returns:
        ``0`` when every check passes, ``1`` otherwise.

    Raises:
        SystemExit: With status 2 when ``--extras`` or ``--groups`` names nothing or a name ``pyproject.toml`` does
            not declare.

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
    declared_extras, declared_groups = declared_names(pyproject)
    extras = _checked_names(parser, "--extras", args.extras, declared_extras)
    groups = _checked_names(parser, "--groups", args.groups, declared_groups)
    if extras is None and groups is None:
        errors.extend(check_declared(probes, declared_requirements(pyproject)))
    else:
        requirements = declared_requirements(pyproject, extras or [], groups or [], environment=True)
        errors.extend(check_installed(probes, requirements))
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
