# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

__all__: list[str] = []


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

    import warnings

    from rfdetr.platform import _INSTALL_MSG

    warnings.warn(
        _INSTALL_MSG.format(name="platform model downloads"),
        ImportWarning,
        stacklevel=2,
    )
