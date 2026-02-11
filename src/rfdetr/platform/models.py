# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

__all__: list[str] = []

_PLUS_EXPORTS = {
    "RFDETR2XLarge",
    "RFDETR2XLargeConfig",
    "RFDETRXLarge",
    "RFDETRXLargeConfig",
}

_PLUS_AVAILABLE = True

try:
    from rfdetr_plus.models import (
        RFDETR2XLarge,
        RFDETR2XLargeConfig,
        RFDETRXLarge,
        RFDETRXLargeConfig,
    )

    __all__ += [
        "RFDETR2XLarge",
        "RFDETRXLarge",
    ]
except ModuleNotFoundError as ex:
    if ex.name not in ("rfdetr_plus", "rfdetr_plus.models"):
        raise

    _PLUS_AVAILABLE = False

    import warnings

    from rfdetr.platform import _INSTALL_MSG

    warnings.warn(
        _INSTALL_MSG.format(name="platform model downloads"),
        ImportWarning,
        stacklevel=2,
    )


def __getattr__(name: str):
    """Lazy failure for missing plus exports: warn on import, raise on access."""
    # Only intercept plus-only symbols when the extra package is missing.
    if name in _PLUS_EXPORTS and not _PLUS_AVAILABLE:
        from rfdetr.platform import _INSTALL_MSG

        # Surface a clear install hint when someone explicitly requests a plus symbol.
        raise ImportError(_INSTALL_MSG.format(name="platform model downloads"))

    # Fall back to the normal attribute lookup error for everything else.
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
