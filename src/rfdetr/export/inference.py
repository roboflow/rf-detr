# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public access to exported RF-DETR inference.

Use :meth:`rfdetr.detr.RFDETR.from_export` to load an artifact and keep the native :meth:`rfdetr.detr.RFDETR.predict`
input and result contract. The ``OpenVINOInference`` wrapper remains available for callers that supply preprocessed
tensors and need raw outputs.

The reference decoders in ``rfdetr.export._onnx.inference`` and ``rfdetr.export._tflite.inference`` remain private
parity helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from PIL import Image

if TYPE_CHECKING:
    from supervision import Detections, KeyPoints

    from rfdetr.detr import RFDETR
    from rfdetr.export._openvino.inference import OpenVINOInference

__all__ = ["OpenVINOInference", "RFDETRInference"]


class RFDETRInference:
    """Predict with an exported artifact through the shared RF-DETR pipeline.

    Create instances with ``RFDETR.from_export()`` or ``rfdetr.from_export()``. This class exposes prediction, labels,
    and runtime information. It does not expose training, evaluation, export, or native optimization.
    """

    def __init__(self, predictor: RFDETR) -> None:
        """Wrap an internal predictor that already has an exported runtime.

        Args:
            predictor: The exported predictor built by the public factory.

        Raises:
            ValueError: If the predictor uses native weights.
        """
        if predictor._exported_context is None:
            raise ValueError("RFDETRInference requires an exported predictor. Use RFDETR.from_export().")
        self._predictor = predictor

    @property
    def class_names(self) -> list[str]:
        """Return a copy of the artifact's class names."""
        return self._predictor.class_names

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Return the selected runtime and device policy."""
        return self._predictor.runtime_info

    def predict(
        self,
        images: str
        | Image.Image
        | np.ndarray[Any, Any]
        | torch.Tensor
        | list[str | np.ndarray[Any, Any] | Image.Image | torch.Tensor],
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
        patch_size: int | None = None,
        include_source_image: bool = True,
        **kwargs: Any,
    ) -> Detections | KeyPoints | list[Detections | KeyPoints]:
        """Run exported inference with the input and result contract of ``RFDETR.predict``.

        Args:
            images: One RGB image or a batch, using the same input formats as ``RFDETR.predict``.
            threshold: Minimum confidence for a result.
            shape: Input dimensions, which must match the exported artifact.
            patch_size: Patch size used for shape validation.
            include_source_image: Include source images in result metadata.
            **kwargs: Additional prediction arguments accepted by ``RFDETR.predict``.

        Returns:
            Detections or keypoints for each image. A batch input returns a list.
        """
        return self._predictor.predict(
            images,
            threshold=threshold,
            shape=shape,
            patch_size=patch_size,
            include_source_image=include_source_image,
            **kwargs,
        )


def __getattr__(name: str) -> Any:
    """Resolve a public inference wrapper on first attribute access.

    Deferred so that importing this module never pulls in an optional runtime dependency the caller
    may not have installed — ``import rfdetr.export.inference`` stays free of ``openvino``.

    Args:
        name: Attribute being looked up on this module.

    Returns:
        The requested wrapper class.

    Raises:
        AttributeError: If *name* is not one of :data:`__all__`.

    Examples:
        >>> from rfdetr.export import inference
        >>> inference.NotARuntime
        Traceback (most recent call last):
        ...
        AttributeError: module 'rfdetr.export.inference' has no attribute 'NotARuntime'
    """
    if name == "OpenVINOInference":
        from rfdetr.export._openvino.inference import OpenVINOInference

        return OpenVINOInference
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
