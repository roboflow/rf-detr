# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private type aliases for native and exported prediction inputs; annotations only, not exported API."""

import os
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, TypeAlias

import numpy as np
import torch
from PIL import Image

#: Image data or a path to an image or expandable media source.
_ImageInput: TypeAlias = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
#: One prediction source, including a camera index.
_PredictionSource: TypeAlias = _ImageInput | int
#: Accepted scalar and iterable inputs to ``RFDETRInference.predict``; iterables are read into a list first.
_PredictionInput: TypeAlias = _PredictionSource | Iterable[_PredictionSource]
#: A single confidence threshold, or one threshold per class as a sequence, array or tensor.
_ThresholdInput: TypeAlias = float | Sequence[float] | np.ndarray[Any, Any] | torch.Tensor

if TYPE_CHECKING:
    # Supervision loads lazily at runtime, so the output aliases exist for type checkers only.
    from supervision import Detections, KeyPoints

    #: The result for one image or video frame.
    _PredictionResult: TypeAlias = Detections | KeyPoints
    #: Eager result of both prediction APIs: one result for a single image, or a list for a collection.
    _PredictionOutput: TypeAlias = _PredictionResult | list[_PredictionResult]
