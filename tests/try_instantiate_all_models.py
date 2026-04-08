#!/usr/bin/env python3
# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""
Comprehensive validation script to test model instantiation with all available weights.

Tests detection and segmentation model classes from rf-detr by importing and instantiating them.
Validates: imports, download, MD5 hash, model instantiation, and from_checkpoint round-trip.

Usage:
    python tests/try_instantiate_all_models.py
"""

import argparse
import os
import sys
import tempfile
from functools import partial

import torch
from tqdm.auto import tqdm

import rfdetr as _rfdetr
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


def _test_from_checkpoint(model_instance: object, actual_cls: type, extra_kwargs: dict) -> None:
    """Round-trip a model through from_checkpoint using a temp training checkpoint.

    Saves the instantiated model's weights into a minimal training-style checkpoint
    (``{"args": ..., "model": state_dict}``), calls ``rfdetr.from_checkpoint`` on it,
    and asserts the returned object is an instance of *actual_cls*.

    Args:
        model_instance: An already-loaded RFDETR model instance.
        actual_cls: The expected model class (e.g. ``RFDETRSmall``).
        extra_kwargs: Extra kwargs to pass to ``from_checkpoint`` (e.g.
            ``{"accept_platform_model_license": True}`` for plus models).

    Raises:
        AssertionError: If the recovered model is not an instance of *actual_cls*.
        Exception: Propagates any error from ``from_checkpoint`` to the caller.
    """
    # Build a minimal training-style checkpoint. The pretrain_weights value only
    # needs to contain the model-size substring that from_checkpoint matches on
    # (e.g. "small", "seg-large").  Using cls.size directly satisfies this.
    fake_pretrain_name = f"{actual_cls.size}.pth"
    num_classes = model_instance.model.args.num_classes
    ckpt = {
        "args": argparse.Namespace(
            pretrain_weights=fake_pretrain_name,
            num_classes=num_classes,
        ),
        "model": model_instance.model.model.state_dict(),
    }

    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pth")
    os.close(tmp_fd)
    try:
        torch.save(ckpt, tmp_path)
        recovered = _rfdetr.from_checkpoint(tmp_path, **extra_kwargs)
        assert recovered is not None, "from_checkpoint returned None"
        assert hasattr(recovered, "model"), "from_checkpoint result missing 'model' attribute"
        assert isinstance(recovered, actual_cls), (
            f"from_checkpoint returned {type(recovered).__name__}, expected {actual_cls.__name__}"
        )
    finally:
        os.unlink(tmp_path)


def main() -> None:
    """Download, validate, instantiate all models, and test from_checkpoint round-trip."""
    print("Model Instantiation & Download Validation\n")

    failed_models = []

    # Progress bar for all models
    pbar = tqdm(MODELS_TO_TEST, desc="Testing models", unit="model")
    for model_class in pbar:
        # Handle partial-wrapped classes
        actual_cls = model_class.func if isinstance(model_class, partial) else model_class
        extra_kwargs = model_class.keywords if isinstance(model_class, partial) else {}
        model_name = actual_cls.size
        pbar.set_description(f"Testing {model_name}")

        try:
            # Instantiate model class - triggers download, MD5 validation, and loading
            model_instance = model_class()

            # Verify model was created
            assert model_instance is not None, "Model instance is None"
            assert hasattr(model_instance, "model"), "Model missing 'model' attribute"

            # from_checkpoint round-trip: save a training-style checkpoint and reload it
            _test_from_checkpoint(model_instance, actual_cls, extra_kwargs)

        except Exception as ex:
            failed_models.append((model_name, str(ex)))

    pbar.close()

    # Summary
    total = len(MODELS_TO_TEST)
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
