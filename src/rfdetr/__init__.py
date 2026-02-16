# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Model classes resolved lazily via __getattr__ to avoid eagerly importing
# the entire training/inference stack (torch, transformers, peft, scipy, etc.)
_DETR_EXPORTS = {
    "RFDETRBase",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRMedium",
    "RFDETRNano",
    "RFDETRSeg2XLarge",
    "RFDETRSegLarge",
    "RFDETRSegMedium",
    "RFDETRSegNano",
    "RFDETRSegPreview",
    "RFDETRSegSmall",
    "RFDETRSegXLarge",
    "RFDETRSmall",
}

_PLUS_EXPORTS = {
    "RFDETR2XLarge",
    "RFDETRXLarge",
}

__all__ = [
    "RFDETRNano",
    "RFDETRSmall",
    "RFDETRBase",
    "RFDETRMedium",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRSegNano",
    "RFDETRSegSmall",
    "RFDETRSegMedium",
    "RFDETRSegLarge",
    "RFDETRSegXLarge",
    "RFDETRSeg2XLarge",
]


def __getattr__(name: str):
    """Resolve model class exports lazily to keep ``import rfdetr`` fast."""
    if name in _DETR_EXPORTS:
        from rfdetr import detr as _detr_module

        # Cache all detr exports at once so __getattr__ is only called once.
        for export_name in _DETR_EXPORTS:
            globals()[export_name] = getattr(_detr_module, export_name)
        return globals()[name]

    if name in _PLUS_EXPORTS:
        from rfdetr.platform import _INSTALL_MSG
        from rfdetr.platform import models as _platform_models

        if hasattr(_platform_models, name):
            value = getattr(_platform_models, name)
            globals()[name] = value
            if name not in __all__:
                __all__.append(name)
            return value

        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Include lazily-resolved model classes in dir() output."""
    module_attrs = list(globals().keys())
    return module_attrs + list(_DETR_EXPORTS) + list(_PLUS_EXPORTS)
