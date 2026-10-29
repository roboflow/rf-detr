# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Private type aliases for native and exported prediction inputs; annotations only, not exported API."""

import os
from collections.abc import Iterable
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image

#: Image data or a path to an image or expandable media source.
_ImageInput: TypeAlias = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
#: One prediction source, including a camera index.
_PredictionSource: TypeAlias = _ImageInput | int
#: Accepted scalar and iterable inputs to ``RFDETRInference.predict``; iterables are read into a list first.
_PredictionInput: TypeAlias = _PredictionSource | Iterable[_PredictionSource]
