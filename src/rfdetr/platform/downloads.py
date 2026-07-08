# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

# The availability check here avoids importing `rfdetr_plus` when the optional
# dependency is not installed. Nested import-time dependencies continue to
# propagate from inside this guarded path.
from typing import cast

from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE

if _IS_RFDETR_PLUS_AVAILABLE:
    try:
        from rfdetr_plus.models import downloads as _downloads

        platform_models = getattr(_downloads, "_PLATFORM_MODELS", None)
        if platform_models is None:
            platform_models = getattr(_downloads, "PLATFORM_MODELS", None)
        PLATFORM_MODELS = cast(dict[str, str], platform_models or {})
    except ModuleNotFoundError as ex:
        missing_name = getattr(ex, "name", "")
        if missing_name in {"rfdetr_plus", "rfdetr_plus.models", "rfdetr_plus.models.downloads"}:
            PLATFORM_MODELS = {}
        else:
            raise
else:
    PLATFORM_MODELS = {}
