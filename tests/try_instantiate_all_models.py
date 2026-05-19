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

# 1008 = LCM(12, 16) × 21: valid for all patch sizes (PE=63 for det ÷16,
# PE=84 for seg ÷12). Each model is tested at its default resolution and at
# 1008 (regression #1038).
#
# Note on Base: ``RFDETRBaseConfig.positional_encoding_size = 37`` is *not*
# formula-derived (see test_load_pretrain_weights.py:TestLoadPretrainWeightsPEInterpolation
# ::test_base_config_non_formula_pe_is_interpolated_from_smaller_checkpoint),
# so this `÷16` description applies only to Nano/Small/Medium/Large.
_CUSTOM_RESOLUTION = 1008

# Plus models (XLarge / 2XLarge) are heavy enough that running them at
# resolution=1008 risks the 15-min CI timeout on windows-latest / macos-latest
# runners.  Smaller models still exercise the 1008 path for #1038 coverage.
_HEAVY_MODEL_NAMES = {
    "xlarge",
    "2xlarge",
    "xxlarge",
    "seg-xlarge",
    "seg-2xlarge",
    "seg-xxlarge",
    "rfdetr-xlarge",
    "rfdetr-2xlarge",
    "rfdetr-xxlarge",
    "rfdetr-seg-xlarge",
    "rfdetr-seg-2xlarge",
    "rfdetr-seg-xxlarge",
}


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


def _test_coco_class_name_mapping(model_instance: object) -> None:
    """Verify predict() uses sparse COCO category-ID → class-name mapping.

    Issue #988: RFDETRSegSmall returned "sheep" for class_id=18 instead of "dog"
    because 0-indexed ``COCO_CLASS_NAMES[18]`` was used instead of the sparse-dict
    lookup ``COCO_CLASSES[18]``.  threshold=0 forces all top-k queries through so
    every class ID in the output is covered.  Covers both detection and segmentation
    nano variants (RFDETRNano, RFDETRSegNano).

    Args:
        model_instance: An already-loaded pretrained COCO model instance (det or seg).

    Raises:
        AssertionError: On any class-name mapping failure.
    """
    import PIL.Image

    from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES

    # Sanity-check model properties required for the pretrained COCO branch.
    class_names = model_instance.class_names
    assert class_names is not None, "Pretrained COCO model must have class_names set"
    assert len(class_names) == len(COCO_CLASS_NAMES), (
        f"Expected {len(COCO_CLASS_NAMES)} COCO class names, got {len(class_names)}"
    )
    assert class_names == list(COCO_CLASS_NAMES), "model.class_names must equal COCO_CLASS_NAMES"
    num_classes = model_instance.model.args.num_classes
    assert num_classes == 90, f"Pretrained COCO model must have num_classes=90, got {num_classes}"

    # Run at threshold=0 to exercise all top-k output slots.
    img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
    detections = model_instance.predict(img, threshold=0.0)

    assert "class_name" in detections.data, "data['class_name'] must be present after predict()"

    # For every detection whose class_id is a valid COCO category, class_name must
    # use sparse-ID lookup (COCO_CLASSES[class_id]), not 0-indexed lookup.
    # Canonical regression case: class_id=18 → "dog", NOT "sheep" (COCO_CLASS_NAMES[18]).
    for class_id, class_name in zip(detections.class_id, detections.data["class_name"]):
        cid = int(class_id)
        if cid in COCO_CLASSES:
            expected = COCO_CLASSES[cid]
            assert class_name == expected, (
                f"Sparse COCO ID mapping broken (issue #988): "
                f"class_id={cid} must map to '{expected}', got '{class_name}'"
            )

    # Regression for PR #1051 HIGH-1: no COCO-pretrained detection may carry
    # '__background__' — background is implicit (below threshold), never a sentinel label.
    background_labeled = [
        (int(cid), name)
        for cid, name in zip(detections.class_id, detections.data["class_name"])
        if name == "__background__"
    ]
    assert not background_labeled, (
        "COCO-pretrained predict() must never produce '__background__' class names "
        f"(PR #1051 HIGH-1 regression); found: {background_labeled[:3]}"
    )


def main() -> None:
    """Download, validate, instantiate all models, and test from_checkpoint round-trip."""
    print("Model Instantiation & Download Validation\n")

    succeeded = 0
    pbar = tqdm(MODELS_TO_TEST, desc="Testing models", unit="model")
    for model_class in pbar:
        actual_cls = model_class.func if isinstance(model_class, partial) else model_class
        extra_kwargs = model_class.keywords if isinstance(model_class, partial) else {}
        base_name = actual_cls.size

        for res in (None, _CUSTOM_RESOLUTION):
            # Skip the 1008-resolution variant for heavyweight Plus models — they
            # risk the 15-min CI timeout on windows-latest / macos-latest runners.
            if res == _CUSTOM_RESOLUTION and base_name in _HEAVY_MODEL_NAMES:
                continue

            model_name = base_name if res is None else f"{base_name}@{res}"
            # Build the kwargs once so `_test_from_checkpoint` and the
            # instantiation call share the same parameter set (avoids the
            # `functools.partial.size` AttributeError seen in the previous form).
            instantiate_kwargs = dict(extra_kwargs)
            if res is not None:
                instantiate_kwargs["resolution"] = res

            pbar.set_description(f"Testing {model_name}")
            try:
                # Instantiate model class - triggers download, MD5 validation, and loading
                model_instance = actual_cls(**instantiate_kwargs)

                # Verify model was created
                assert model_instance is not None, "Model instance is None"
                assert hasattr(model_instance, "model"), "Model missing 'model' attribute"

                # from_checkpoint round-trip: save a training-style checkpoint and reload it.
                # Pass the real class (not a partial) so `_test_from_checkpoint` can read
                # `.size` and `.__name__` and run `isinstance(recovered, actual_cls)`.
                _test_from_checkpoint(model_instance, actual_cls, instantiate_kwargs)

                # Inference class-name regression for issue #988 — run on all
                # nano-sized pretrained COCO models at default resolution only.
                if "nano" in base_name.lower() and res is None:
                    _test_coco_class_name_mapping(model_instance)

                succeeded += 1
            except Exception as ex:
                # Fail-fast: surface the first failing model directly so CI logs the
                # root cause cleanly instead of burying it under later cascade failures.
                pbar.close()
                print(f"\n[FAIL] {model_name}: {ex}")
                raise

    pbar.close()
    print("\nResults:")
    print(f"\tSucceeded:\t{succeeded}")
    print("\n[OK] All models validated successfully")
    sys.exit(0)


if __name__ == "__main__":
    main()
