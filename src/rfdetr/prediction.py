# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public input types shared by native and exported prediction."""

import os
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image

#: Image data or a path to an image or expandable media source.
ImageInput: TypeAlias = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
#: One prediction source, including a camera index.
PredictionSource: TypeAlias = ImageInput | int
#: Accepted scalar and sequence inputs to both public prediction APIs.
PredictionInput: TypeAlias = PredictionSource | list[PredictionSource] | tuple[PredictionSource, ...]
