# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR public package initialiser.

Two-phase legacy-module deprecation
------------------------------------
Some sub-packages were relocated in v1.6 and are scheduled for removal in v1.7.
The migration is handled in two phases so users get a full release cycle to update
their imports:

**Phase 1 — v1.6 (current):** the old packages (``rfdetr.util``, ``rfdetr.deploy``)
still exist on disk and work normally, but emit a ``DeprecationWarning`` on import.
``_RemovedModuleFinder`` is installed in ``sys.meta_path`` but stays dormant: its
``find_spec`` returns ``None`` whenever ``importlib.machinery.PathFinder`` can
resolve the name (i.e. while the shim directories are present).

**Phase 2 — v1.7:** the shim directories are deleted.  ``PathFinder`` can no longer
resolve ``rfdetr.util`` / ``rfdetr.deploy``, so ``_RemovedModuleFinder`` intercepts
the import and raises a descriptive ``ImportError`` (migration hint) instead of the
cryptic default ``ModuleNotFoundError: No module named 'rfdetr.util'``.

To complete Phase 2, delete ``src/rfdetr/util/`` and ``src/rfdetr/deploy/`` and bump
``_REMOVED_IN_V17`` (or rename it) to reflect the new version boundary.
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from typing import Any

from rfdetr.detr import RFDETR
from rfdetr.inference import ModelContext
from rfdetr.variants import (
    RFDETRBase,  # DEPRECATED # noqa: F401
    RFDETRLarge,
    RFDETRLargeDeprecated,  # DEPRECATED # noqa: F401
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegPreview,  # DEPRECATED # noqa: F401
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)

__all__ = [
    "ModelContext",
    "from_checkpoint",
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]


def from_checkpoint(path: str | os.PathLike[str], **kwargs: Any) -> RFDETR:
    """Convenience wrapper for RFDETR.from_checkpoint(); see that method for full documentation."""
    return RFDETR.from_checkpoint(path, **kwargs)


# Lazily resolved names: avoids eager pytorch_lightning import at `import rfdetr` time.
_LAZY_TRAINING = frozenset({"RFDETRModelModule", "RFDETRDataModule", "build_trainer"})
_PLUS_EXPORTS = frozenset({"RFDETR2XLarge", "RFDETRXLarge"})

# Legacy module aliases delegate to shim packages while they still exist, then raise
# migration-hint ImportError messages once those shims are removed in v1.7+.
_REMOVED_IN_V17 = {
    "util": "rfdetr.util was removed in v1.7. Use rfdetr.utilities instead.",
    "deploy": "rfdetr.deploy was removed in v1.7. Use rfdetr.export instead.",
}


class _RemovedModuleLoader(importlib.abc.Loader):
    """Raise a migration hint when a removed legacy module import is attempted."""

    def __init__(self, message: str) -> None:
        self._message = message

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> None:
        """Use the default module creation path."""
        return None

    def exec_module(self, module: object) -> None:
        """Abort import with a migration hint instead of bare ModuleNotFoundError."""
        raise ImportError(self._message) from None


class _RemovedModuleFinder(importlib.abc.MetaPathFinder):
    """Intercept removed legacy dotted imports after their shim packages are deleted."""

    _PATH_FINDER = importlib.machinery.PathFinder

    def find_spec(
        self,
        fullname: str,
        path: list[str] | None,
        target: object | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        """Return a failing spec with a migration hint for removed legacy modules."""
        if not fullname.startswith(f"{__name__}."):
            return None
        root, _, _ = fullname.removeprefix(f"{__name__}.").partition(".")
        if root not in _REMOVED_IN_V17:
            return None

        if self._PATH_FINDER.find_spec(fullname, path) is not None:
            return None

        is_package = fullname == f"{__name__}.{root}"
        loader = _RemovedModuleLoader(_REMOVED_IN_V17[root])
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


_REMOVED_MODULE_FINDER = _RemovedModuleFinder()

if not getattr(sys, "_rfdetr_removed_finder", False):
    sys.meta_path.insert(0, _REMOVED_MODULE_FINDER)
    sys._rfdetr_removed_finder = True


def __getattr__(name: str):
    """Lazily resolve training/PTL and plus-only exports and handle removed-module aliases.

    This hook is only invoked on explicit attribute access (e.g. ``rfdetr.RFDETRModelModule``)
    and supports three behaviors:

    * Training/PTL exports (names in ``_LAZY_TRAINING``) are imported from ``rfdetr.training``
      on first use to avoid importing PyTorch Lightning at ``import rfdetr`` time.
    * Plus-only exports (names in ``_PLUS_EXPORTS``) are imported from ``rfdetr.platform.models``,
      and a descriptive ``ImportError`` is raised with an installation hint if the model is
      not available.
    * Removed-module aliases (keys in ``_REMOVED_IN_V17``, such as ``util`` and ``deploy``)
      are first attempted via a shim submodule (e.g. ``rfdetr.util``); once the shim files
      are removed, a migration-hint ``ImportError`` is raised instead of silently masking
      unrelated nested import errors.
    """
    if name in _REMOVED_IN_V17:
        module_name = f"{__name__}.{name}"
        try:
            value = importlib.import_module(module_name)
            globals()[name] = value
            return value
        except ModuleNotFoundError as exc:
            # Avoid masking nested import errors from within the shim itself.
            if exc.name != module_name:
                raise
            raise ImportError(_REMOVED_IN_V17[name]) from None

    if name in _LAZY_TRAINING:
        from rfdetr import training as _training

        value = getattr(_training, name)
        globals()[name] = value
        return value

    if name in _PLUS_EXPORTS:
        from rfdetr.platform import _INSTALL_MSG
        from rfdetr.platform import models as _platform_models

        if hasattr(_platform_models, name):
            value = getattr(_platform_models, name)
            globals()[name] = value
            return value

        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
