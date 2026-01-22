# ------------------------------------------------------------------------
# Platform Model License 1.0 (PML-1.0)
# Copyright (c) 2026 Roboflow, Inc. All Rights Reserved.
#
# Licensed under the Platform Model License 1.0.
# Use, modification, and distribution of code and checkpoints require
# an active Roboflow platform plan or agreement.
#
# See the LICENSE.platform file for full terms and conditions.
# ------------------------------------------------------------------------

from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.detr import RFDETR
from typing import Literal, List


class RFDETRXLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 512
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_windows: int = 1
    patch_size: int = 20
    projector_scale: List[Literal["P4",]] = ["P4"]
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_classes: int = 365
    positional_encoding_size: int = 700 // 20
    resolution: int = 700
    pretrain_weights: str = "rf-detr-xlarge.pth"
    license: str = "PML-1.0"


class RFDETR2XLargeConfig(ModelConfig):
    encoder: Literal["dinov2_windowed_base"] = "dinov2_windowed_base"
    hidden_dim: int = 512
    dec_layers: int = 5
    sa_nheads: int = 16
    ca_nheads: int = 32
    dec_n_points: int = 4
    num_windows: int = 2
    patch_size: int = 20
    projector_scale: List[Literal["P4",]] = ["P4"]
    out_feature_indexes: List[int] = [3, 6, 9, 12]
    num_classes: int = 365
    positional_encoding_size: int = 880 // 20
    resolution: int = 880
    pretrain_weights: str = "rf-detr-xxlarge.pth"
    license: str = "PML-1.0"


class RFDETRXLarge(RFDETR):
    size = "rfdetr-xlarge"

    def __init__(self, accept_platform_model_license: bool = False, **kwargs):
        if not accept_platform_model_license is True:
            raise ValueError(
                "You must accept the platform model license (LICENSE.platform) to use this model. "
                "You can do this by setting accept_platform_model_license=True when initializing the model."
            )
        super().__init__(**kwargs)

    def get_model_config(self, **kwargs):
        return RFDETRXLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)


class RFDETR2XLarge(RFDETR):
    size = "rfdetr-2xlarge"

    def __init__(self, accept_platform_model_license: bool = False, **kwargs):
        if not accept_platform_model_license is True:
            raise ValueError(
                "You must accept the platform model license (LICENSE.platform) to use this model. "
                "You can do this by setting accept_platform_model_license=True when initializing the model."
            )
        super().__init__(**kwargs)

    def get_model_config(self, **kwargs):
        return RFDETR2XLargeConfig(**kwargs)

    def get_train_config(self, **kwargs):
        return TrainConfig(**kwargs)
