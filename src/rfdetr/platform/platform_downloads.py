# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

try:
    from rfdetr_plus.models.downloads import _PLATFORM_MODELS as PLATFORM_MODELS
except ModuleNotFoundError as ex:
    if ex.name in ("rfdetr_plus", "rfdetr_plus.models", "rfdetr_plus.models.downloads"):
        import warnings

        from rfdetr.platform import _INSTALL_MSG

        warnings.warn(
            _INSTALL_MSG.format(name="platform model downloads"),
            ImportWarning,
            stacklevel=2,
        )
        PLATFORM_MODELS = {}
    else:
        raise
