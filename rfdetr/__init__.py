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
from rfdetr.platform.models import (
    RFDETR2XLarge,
    RFDETRXLarge,
)
