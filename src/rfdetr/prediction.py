# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public input types for :meth:`rfdetr.RFDETR.predict`."""

import os
from typing import Any, TypeAlias

import numpy as np
import torch
from PIL import Image

ImageInput: TypeAlias = str | os.PathLike[str] | Image.Image | np.ndarray[Any, Any] | torch.Tensor
PredictionSource: TypeAlias = ImageInput | int
PredictionInput: TypeAlias = PredictionSource | list[PredictionSource] | tuple[PredictionSource, ...]
