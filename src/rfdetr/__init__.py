# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import importlib

from rfdetr.detr import (
    ModelContext,
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

# Lazily resolved names: avoids eager pytorch_lightning import at `import rfdetr` time.
_LAZY_TRAINING = frozenset({"RFDETRModelModule", "RFDETRDataModule", "build_trainer"})
_PLUS_EXPORTS = frozenset({"RFDETR2XLarge", "RFDETRXLarge"})

# Modules removed in v1.7 — hook fires once the shim files are deleted at release time.
_REMOVED_IN_V17 = {
    "util": "rfdetr.util was removed in v1.7. Use rfdetr.utilities instead.",
    "deploy": "rfdetr.deploy was removed in v1.7. Use rfdetr.export instead.",
}


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
