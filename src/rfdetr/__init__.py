# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""RF-DETR public package initialiser.

Legacy-module deprecation (Phase 2 — v1.9.0)
----------------------------------------------
The sub-packages ``rfdetr.util`` and ``rfdetr.deploy`` were relocated in v1.6.0 and removed in v1.9.0.
Their shim directories have been deleted; ``_RemovedModuleFinder`` intercepts any remaining import attempts
and raises a descriptive ``ImportError`` (migration hint) instead of a bare ``ModuleNotFoundError``.

Migration:
  - ``rfdetr.util.*``   → ``rfdetr.utilities.*``
  - ``rfdetr.deploy.*`` → ``rfdetr.export.*``
"""

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import os
import sys
from typing import Any

# np.complex_ was removed in NumPy 2.0 (June 2024). Some transitive dependencies (e.g. older
# tensorflow, chumpy) still reference it, causing AttributeError on import rfdetr. See issue #1061.
# TODO: Remove once all transitive deps support NumPy 2.x natively.
try:
    import numpy

    _IS_NUMPY_INSTALLED = True
except ImportError:
    _IS_NUMPY_INSTALLED = False

if _IS_NUMPY_INSTALLED and not hasattr(numpy, "complex_"):
    _complex128 = getattr(numpy, "complex128", None)
    if _complex128 is not None:
        numpy.complex_ = _complex128


from rfdetr.detr import RFDETR
from rfdetr.inference import ModelContext
from rfdetr.variants import (
    RFDETRKeypointPreview,
    RFDETRLarge,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)

__all__ = [
    "ModelContext",
    "from_checkpoint",
    "RFDETRKeypointPreview",
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

# Legacy module names removed in v1.9.0 — _RemovedModuleFinder raises ImportError with migration hint.
_REMOVE_IN_VERSION_1_9 = {
    "util": "rfdetr.util will be removed in v1.9.0. Use rfdetr.utilities instead.",
    "deploy": "rfdetr.deploy will be removed in v1.9.0. Use rfdetr.export instead.",
}

# Class names removed in v2.0.0 — raise AttributeError with migration hint.
_REMOVE_IN_VERSION_2_0_CLASSES = {
    "RFDETRBase": "RFDETRBase was removed in v2.0.0. Use RFDETRNano/Small/Medium/Large instead.",
    "RFDETRLargeDeprecated": "RFDETRLargeDeprecated was removed in v2.0.0. Use RFDETRLarge instead.",
    "RFDETRSegPreview": "RFDETRSegPreview was removed in v2.0.0. Use RFDETRSegNano/Small/Medium/Large instead.",
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
        if root not in _REMOVE_IN_VERSION_1_9:
            return None

        if self._PATH_FINDER.find_spec(fullname, path) is not None:
            return None

        is_package = fullname == f"{__name__}.{root}"
        loader = _RemovedModuleLoader(_REMOVE_IN_VERSION_1_9[root])
        return importlib.util.spec_from_loader(fullname, loader, is_package=is_package)


_REMOVED_MODULE_FINDER = _RemovedModuleFinder()

if not getattr(sys, "_rfdetr_removed_finder", False):
    sys.meta_path.insert(0, _REMOVED_MODULE_FINDER)
    sys._rfdetr_removed_finder = True


def __getattr__(name: str):
    """Lazily resolve training/PTL and plus-only exports and handle removed-module aliases.

    This hook is only invoked on explicit attribute access (e.g. ``rfdetr.RFDETRModelModule``) and supports three
    behaviors:

    * Training/PTL exports (names in ``_LAZY_TRAINING``) are imported from ``rfdetr.training``
      on first use to avoid importing PyTorch Lightning at ``import rfdetr`` time.
    * Plus-only exports (names in ``_PLUS_EXPORTS``) are imported from ``rfdetr.platform.models``,
      and a descriptive ``ImportError`` is raised with an installation hint if the model is not available.
    * Removed-module aliases (keys in ``_REMOVE_IN_VERSION_1_9``, such as ``util`` and ``deploy``)
      raise a migration-hint ``ImportError`` — the shim packages were deleted in v1.9.0.
    """
    if name in _REMOVE_IN_VERSION_1_9:
        module_name = f"{__name__}.{name}"
        try:
            value = importlib.import_module(module_name)
            globals()[name] = value
            return value
        except ModuleNotFoundError as exc:
            # Avoid masking nested import errors from within the shim itself.
            if exc.name != module_name:
                raise
            raise ImportError(_REMOVE_IN_VERSION_1_9[name]) from None

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

    if name in _REMOVE_IN_VERSION_2_0_CLASSES:
        raise AttributeError(_REMOVE_IN_VERSION_2_0_CLASSES[name])

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
