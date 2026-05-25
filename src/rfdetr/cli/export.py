# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``rfdetr export`` subcommand.

Thin wrapper around :meth:`rfdetr.detr.RFDETR.export` so its full surface
is reachable from the shell.  ``jsonargparse.CLI`` introspects
:func:`export_main` to build the parser, so flag names, types, and help
text stay in lockstep with the function signature and its Google-style
docstring.

YAML config support is automatic via jsonargparse: pass
``--config path/to/export.yaml`` and any keys matching the parameters
below are loaded from the file.
"""

from __future__ import annotations

from typing import Literal, Optional

from rfdetr.utilities.logger import get_logger

logger = get_logger()


def export_main(
    checkpoint: str,
    *,
    output_dir: str = "output",
    format: Literal["onnx", "tflite"] = "onnx",
    quantization: Optional[Literal["fp32", "fp16", "int8"]] = None,
    calibration_data: Optional[str] = None,
    max_images: int = 100,
    shape: Optional[tuple[int, int]] = None,
    batch_size: int = 1,
    opset_version: int = 17,
    backbone_only: bool = False,
    dynamic_batch: bool = False,
    patch_size: Optional[int] = None,
    infer_dir: Optional[str] = None,
    notes: Optional[str] = None,
    verbose: bool = True,
) -> None:
    """Export an RF-DETR checkpoint to ONNX or TFLite.

    Loads the checkpoint with :func:`rfdetr.from_checkpoint`, which
    auto-resolves the correct ``RFDETR`` subclass (Nano, Small, ..., Seg*)
    from the checkpoint metadata, then calls :meth:`RFDETR.export`.

    Args:
        checkpoint: Path to the ``.pt`` / ``.pth`` checkpoint to load.
        output_dir: Directory to write the exported artifacts to.
        format: Export format. ``"onnx"`` writes an ``.onnx`` file;
            ``"tflite"`` additionally converts via ``onnx2tf`` and writes
            FP32 / FP16 / INT8 ``.tflite`` variants per *quantization*.
        quantization: TFLite quantization mode. Ignored when
            ``format="onnx"``. ``None`` / ``"fp32"`` / ``"fp16"`` keep float
            weights; ``"int8"`` produces a dynamic-range int8 model.
        calibration_data: Directory of representative JPEG/PNG images or
            path to a ``.npy`` array of shape ``(N, H, W, 3)``. Used for
            INT8 quantization and ``onnx2tf`` output validation. The
            ``ndarray`` form accepted by :meth:`RFDETR.export` is not
            reachable from the shell; pass a directory or ``.npy`` path.
        max_images: Maximum number of images to load from a
            *calibration_data* directory.
        shape: ``(height, width)`` tuple baked into the exported graph.
            Both dimensions must be divisible by ``patch_size *
            num_windows``. Defaults to the model's native resolution.
        batch_size: Static batch size baked into the ONNX graph.
        opset_version: ONNX opset version to target.
        backbone_only: Export the backbone (feature extractor) only.
        dynamic_batch: If ``True``, export with a dynamic batch dimension
            so the artifact accepts variable batch sizes at runtime.
        patch_size: Backbone patch size. Defaults to the checkpoint's
            stored ``model_config.patch_size``.
        infer_dir: Optional directory of sample images for dynamic-axes
            inference during export tracing.
        notes: Optional free-form metadata embedded in the ONNX file
            under the ``"rfdetr_notes"`` metadata property.
        verbose: Print export progress information.
    """
    from rfdetr import from_checkpoint

    logger.info("Loading checkpoint from %s", checkpoint)
    model = from_checkpoint(checkpoint)
    model.export(
        output_dir=output_dir,
        format=format,
        quantization=quantization,
        calibration_data=calibration_data,
        max_images=max_images,
        shape=shape,
        batch_size=batch_size,
        opset_version=opset_version,
        backbone_only=backbone_only,
        dynamic_batch=dynamic_batch,
        patch_size=patch_size,
        infer_dir=infer_dir,
        notes=notes,
        verbose=verbose,
    )


def main() -> None:
    """Entry point for ``rfdetr export``."""
    try:
        from jsonargparse import CLI
    except ImportError as exc:  # pragma: no cover - guarded by [cli] extra
        raise ImportError(
            "`rfdetr export` requires jsonargparse. Install the cli extra: "
            "`pip install 'rfdetr[cli]'` (or include cli alongside other extras)."
        ) from exc

    CLI(export_main, as_positional=False)


if __name__ == "__main__":
    main()
