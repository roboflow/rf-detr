# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
from transformers import AutoBackbone
import torch.nn.functional as F
import types
import math
import json
import os

from .dinov2_with_windowed_attn import WindowedDinov2WithRegistersConfig, WindowedDinov2WithRegistersBackbone


size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

size_to_config = {
    "small": "dinov2_small.json",
    "base": "dinov2_base.json",
    "large": "dinov2_large.json",
}

size_to_config_with_registers = {
    "small": "dinov2_with_registers_small.json",
    "base": "dinov2_with_registers_base.json",
    "large": "dinov2_with_registers_large.json",
}

def get_config(size, use_registers):
    config_dict = size_to_config_with_registers if use_registers else size_to_config
    current_dir = os.path.dirname(os.path.abspath(__file__))
    configs_dir = os.path.join(current_dir, "dinov2_configs")
    config_path = os.path.join(configs_dir, config_dict[size])
    with open(config_path, "r") as f:
        dino_config = json.load(f)
    return dino_config


class DinoV2(nn.Module):
    def __init__(self, shape=(640, 640), out_feature_indexes=[2, 4, 5, 9], size="base", use_registers=True, use_windowed_attn=True, gradient_checkpointing=False, load_dinov2_weights=True):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        
        # Create the encoder
        
        if not use_windowed_attn:
            assert not gradient_checkpointing, "Gradient checkpointing is not supported for non-windowed attention"
            assert load_dinov2_weights, "Using non-windowed attention requires loading dinov2 weights from hub"
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = get_config(size, use_registers)

            dino_config["return_dict"] = False
            dino_config["out_features"] = [f"stage{i}" for i in out_feature_indexes]

            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=4,
                    window_block_indexes=window_block_indexes,
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config,
                    num_windows=4,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
                )
            self.encoder = WindowedDinov2WithRegistersBackbone.from_pretrained(
                name,
                config=windowed_dino_config,
            ) if load_dinov2_weights else WindowedDinov2WithRegistersBackbone(windowed_dino_config)


        self._out_feature_channels = [size_to_width[size]] * len(out_feature_indexes)
        self._export = False

    def export(self):
        if self._export:
            return
        self._export = True

    def forward(self, x):
        assert x.shape[2] % 14 == 0 and x.shape[3] % 14 == 0, f"Dinov2 requires input shape to be divisible by 14, but got {x.shape}"
        x = self.encoder(x)
        return list(x[0])

if __name__ == "__main__":
    model = DinoV2()
    model.export()
    x = torch.randn(1, 3, 640, 640)
    print(model(x))
    for j in model(x):
        print(j.shape)
