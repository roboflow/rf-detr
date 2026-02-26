# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import os

if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") is None:
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from rfdetr.detr import (
    RFDETRBase,
    RFDETRLarge,
    RFDETRLargeDeprecated,
    RFDETRMedium,
    RFDETRNano,
    RFDETRSeg2XLarge,
    RFDETRSegLarge,
    RFDETRSegMedium,
    RFDETRSegNano,
    RFDETRSegPreview,
    RFDETRSegSmall,
    RFDETRSegXLarge,
    RFDETRSmall,
)

__all__ = [
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


def __getattr__(name: str):
    """Resolve plus-only exports lazily, raising only on explicit access."""
    _PLUS_EXPORTS = {"RFDETR2XLarge", "RFDETRXLarge"}
    if name in _PLUS_EXPORTS:
        from rfdetr.platform import _INSTALL_MSG
        from rfdetr.platform import models as _platform_models

        # Cache the resolved symbol to avoid repeated attribute lookups.
        if hasattr(_platform_models, name):
            value = getattr(_platform_models, name)
            globals()[name] = value
            # Keep __all__ in sync with dynamically resolved exports.
            if name not in __all__:
                __all__.append(name)
            return value

        # The name is expected to be plus-only; raise a clear install hint.
        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    # Non-plus names fall back to the default attribute error.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
