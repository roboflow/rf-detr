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
    RFDETRXLarge,
)
from rfdetr.config import RFDETRSegLargeConfig, RFDETRSegNanoConfig, TrainConfig
from rfdetr.training.module_model import RFDETRModelModule

try:
    from rfdetr import RFDETR2XLarge
except ImportError:
    RFDETR2XLarge = None

# Explicitly list all models to validate
MODELS_TO_TEST = [
    # Detection Models
    RFDETRNano,
    RFDETRSmall,
    RFDETRMedium,
    RFDETRLarge,
    partial(RFDETRXLarge, accept_platform_model_license=True),
    # Segmentation Models
    RFDETRSegNano,
    RFDETRSegSmall,
    RFDETRSegMedium,
    RFDETRSegLarge,
    RFDETRSegXLarge,
    RFDETRSeg2XLarge,
]

if RFDETR2XLarge is not None:
    MODELS_TO_TEST.append(partial(RFDETR2XLarge, accept_platform_model_license=True))

# Training-path custom-resolution tests: regression for #1038.
# RFDETRModelModule.__init__ calls _load_pretrain_weights(), which must bicubic-interpolate
# the checkpoint PE to match positional_encoding_size before load_state_dict.
# (name, config_cls, resolution) — resolution differs from each model's default.
# Empty when pytorch_lightning is not installed (e.g. integration workflow with .[plus] only).
TRAINING_PATH_RESOLUTION_TESTS: list[tuple] = [
    ("RFDETRSegNano@1008", RFDETRSegNanoConfig, 1008),  # default PE=26 (312/12), target PE=84 (1008/12)
    ("RFDETRSegLarge@1008", RFDETRSegLargeConfig, 1008),  # default PE=42 (504/12), target PE=84 (1008/12)
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

        except Exception as ex:
            failed_models.append((model_name, str(ex)))

    pbar.close()

    # Training-path PE interpolation tests (regression #1038).
    # Build RFDETRModelModule directly — exercises _load_pretrain_weights() with real weights
    # at a non-default resolution where PE grids must be bicubic-interpolated.
    print("\nTraining-Path Custom Resolution Tests (regression #1038)\n")
    tc = TrainConfig(dataset_dir="/nonexistent", output_dir="/nonexistent", accelerator="cpu")
    pbar2 = tqdm(TRAINING_PATH_RESOLUTION_TESTS, desc="Training-path tests", unit="model")
    for model_name, config_cls, resolution in pbar2:
        pbar2.set_description(f"Testing {model_name}")
        try:
            mc = config_cls(resolution=resolution, device="cpu")
            module = RFDETRModelModule(mc, tc)
            assert module.model is not None, "module.model is None after weight loading"
        except Exception as ex:
            failed_models.append((model_name, str(ex)))
    pbar2.close()

    # Summary
    total = len(MODELS_TO_TEST) + len(TRAINING_PATH_RESOLUTION_TESTS)
    print("\nResults:")
    print(f"  Total:     {total}")
    print(f"  Succeeded: {total - len(failed_models)}")
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
