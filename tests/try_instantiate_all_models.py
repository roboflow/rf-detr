#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Comprehensive validation script to test model instantiation with all available weights.

Tests detection and segmentation model classes from rf-detr by importing and instantiating them.
Validates: imports, download, MD5 hash, and model instantiation.

Usage:
    python tests/try_instantiate_all_models.py
"""

import sys
from functools import partial

from tqdm.auto import tqdm

from rfdetr import (
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
)

try:
    from rfdetr import RFDETR2XLarge, RFDETRXLarge
except ImportError:
    RFDETR2XLarge = None
    RFDETRXLarge = None

# Explicitly list all models to validate
MODELS_TO_TEST = [
    # Detection Models
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge,
    # Segmentation Models
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
]

if RFDETRXLarge is not None:
    MODELS_TO_TEST.append(partial(RFDETRXLarge, accept_platform_model_license=True))
if RFDETR2XLarge is not None:
    MODELS_TO_TEST.append(partial(RFDETR2XLarge, accept_platform_model_license=True))

# 1008 = LCM(12, 16) × 21: valid for all patch sizes (PE=63 for det ÷16, PE=84 for seg ÷12).
# Each model is tested at its default resolution and at 1008 (regression #1038).
_CUSTOM_RESOLUTION = 1008


def main() -> None:
    """Download, validate, and instantiate all models."""
    print("Model Instantiation & Download Validation\n")

    failed_models = []

    pbar = tqdm(MODELS_TO_TEST, desc="Testing models", unit="model")
    for model_class in pbar:
        base_name = model_class.func.size if isinstance(model_class, partial) else model_class.size
        for res in (None, _CUSTOM_RESOLUTION):
            model_name = base_name if res is None else f"{base_name}@{res}"
            cls = model_class if res is None else partial(model_class, resolution=res)
            pbar.set_description(f"Testing {model_name}")
            try:
                # Instantiate model class - triggers download, MD5 validation, and loading
                model_instance = cls()

                # Verify model was created
                assert model_instance is not None, "Model instance is None"
                assert hasattr(model_instance, "model"), "Model missing 'model' attribute"

            except Exception as ex:
                failed_models.append((model_name, str(ex)))

    pbar.close()

    # Summary
    total = len(MODELS_TO_TEST) * 2
    print("\nResults:")
    print(f"\tTotal:\t{total}")
    print(f"\tSucceeded:\t{total - len(failed_models)}")
    print(f"\tFailed:\t{len(failed_models)}")

    if failed_models:
        print("\nFailed models:")
        for model_name, error in failed_models:
            print(f"  {model_name}: {error}")
        print("\n[WARN] Some models failed")
        sys.exit(1)
    else:
        print("\n[OK] All models validated successfully")
        sys.exit(0)


if __name__ == "__main__":
    main()
