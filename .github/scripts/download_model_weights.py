# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Download pretrained weights by instantiating each supported model.

Purpose:
    Populate the shared CI weight archive without running inference or checkpoint round-trips.
Scope:
    Released detection and segmentation sizes, including Plus, and keypoint preview.
Usage:
    Install rfdetr[plus], set RF_HOME to a fresh directory, then run this script.
Outputs:
    Downloaded weight files in RF_HOME and progress on stdout.
Failure:
    Import, download, checksum and model-initialization errors propagate and prevent cache publication.
Used by:
    The independent cache-model-weights job in .github/workflows/ci-integrations.yml.
"""

import gc
from functools import partial


def main() -> None:
    """Instantiate each model on CPU to download and validate its pretrained weights."""
    # GitHub helper doctests collect this module without installing the model dependencies.
    from rfdetr import (
        RFDETR2XLarge,
        RFDETRKeypointPreview,
        RFDETRLarge,
        RFDETRMedium,
        RFDETRNano,
        RFDETRSeg2XLarge,
        RFDETRSegLarge,
        RFDETRSegMedium,
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegXLarge,
        RFDETRSmall,
        RFDETRXLarge,
    )

    for model_class in (
        RFDETRNano,
        RFDETRSmall,
        RFDETRMedium,
        RFDETRLarge,
        RFDETRKeypointPreview,
        RFDETRSegNano,
        RFDETRSegSmall,
        RFDETRSegMedium,
        RFDETRSegLarge,
        RFDETRSegXLarge,
        RFDETRSeg2XLarge,
        partial(RFDETRXLarge, accept_platform_model_license=True),
        partial(RFDETR2XLarge, accept_platform_model_license=True),
    ):
        actual_class = model_class.func if isinstance(model_class, partial) else model_class
        print(f"Downloading weights for {actual_class.__name__}", flush=True)
        model = model_class(device="cpu")
        del model
        gc.collect()


if __name__ == "__main__":
    main()
