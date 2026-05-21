# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import warnings

_INSTALL_MSG = (
    "The {name} requires the 'plus' extras for the 'rfdetr' package."
    " Install it with `pip install rfdetr[plus]` (or `pip install rfdetr_plus` if supported)."
)

try:
    import rfdetr_plus as _rfdetr_plus  # noqa: F401

    _IS_RFDETR_PLUS_AVAILABLE: bool = True
except ModuleNotFoundError:
    _IS_RFDETR_PLUS_AVAILABLE = False
    warnings.warn(
        _INSTALL_MSG.format(name="platform model downloads"),
        ImportWarning,
        stacklevel=2,
    )
