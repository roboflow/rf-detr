# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

try:
    from rfdetr_plus.models import downloads as _downloads

    try:
        PLATFORM_MODELS = _downloads._PLATFORM_MODELS
    except AttributeError:
        PLATFORM_MODELS = _downloads.PLATFORM_MODELS
except ImportError:
    try:
        from rfdetr_plus.models.downloads import PLATFORM_MODELS
    except ImportError as ex:
        missing_name = getattr(ex, "name", "")
        if missing_name.startswith("rfdetr_plus") or "rfdetr_plus" in str(ex):
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
