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
    RFDETR2XLarge,
    RFDETRBase,
    RFDETRLarge,
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
    RFDETRXLarge,
)

# Explicitly list all models to validate
MODELS_TO_TEST = [
    # Detection Models
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRBase,
    RFDETRLarge,
    partial(RFDETRXLarge, accept_platform_model_license=True),
    partial(RFDETR2XLarge, accept_platform_model_license=True),
    # Segmentation Models
    RFDETRSegPreview,
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
]


def main() -> None:
    """Download, validate, and instantiate all models."""
    print("Model Instantiation & Download Validation\n")

    failed_models = []

    # Progress bar for all models
    pbar = tqdm(MODELS_TO_TEST, desc="Testing models", unit="model")
    for model_class in pbar:
        # Handle partial-wrapped classes
        model_name = model_class.func.size if isinstance(model_class, partial) else model_class.size
        pbar.set_description(f"Testing {model_name}")

        try:
            # Instantiate model class - triggers download, MD5 validation, and loading
            model_instance = model_class()

            # Verify model was created
            assert model_instance is not None, "Model instance is None"
            assert hasattr(model_instance, "model"), "Model missing 'model' attribute"

        except Exception as e:
            failed_models.append((model_name, str(e)))

    pbar.close()

    # Summary
    print("\nResults:")
    print(f"  Total:     {len(MODELS_TO_TEST)}")
    print(f"  Succeeded: {len(MODELS_TO_TEST) - len(failed_models)}")
    print(f"  Failed:    {len(failed_models)}")

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
