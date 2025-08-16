# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import math
import types
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


size_to_width_dinov3 = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


class DinoV3(nn.Module):
    def __init__(
        self,
        shape: Tuple[int, int] = (640, 640),
        out_feature_indexes: Optional[List[int]] = None,
        size: str = "base",
        patch_size: int = 16,
        positional_encoding_size: Optional[int] = None,
    ) -> None:
        super().__init__()

        if out_feature_indexes is None:
            out_feature_indexes = [2, 5, 8, 11]

        model_names = {
            "small": "facebook/dinov3-vits16",
            "base": "facebook/dinov3-vitb16",
            "large": "facebook/dinov3-vitl16",
        }

        # Ensure the spatial resolution aligns with the positional encodings.
        self.patch_size = patch_size
        self.out_feature_indexes = out_feature_indexes
        if positional_encoding_size is None:
            positional_encoding_size = shape[0] // patch_size
        implied_resolution = positional_encoding_size * patch_size
        if implied_resolution != shape[0] or implied_resolution != shape[1]:
            shape = (implied_resolution, implied_resolution)
        self.shape = shape
        self.positional_encoding_size = positional_encoding_size

        self.encoder = AutoModel.from_pretrained(
            model_names[size],
            output_hidden_states=True,
        )

        self._out_feature_channels = [
            size_to_width_dinov3[size]
        ] * len(out_feature_indexes)

        self._export = False

    def export(self) -> None:
        if self._export:
            return
        self._export = True
        shape = self.shape

        def make_new_interpolated_pos_encoding(position_embeddings, patch_size, height, width):
            num_positions = position_embeddings.shape[1]
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            patch_pos_embed = position_embeddings.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return patch_pos_embed

        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.patch_size,
                shape[0],
                shape[1],
            )

        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            return new_positions

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, self.encoder.embeddings
        )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        outputs = self.encoder(pixel_values=x)
        hidden_states = outputs.hidden_states

        feats = []
        for idx in self.out_feature_indexes:
            feat = hidden_states[idx]
            b, n, c = feat.shape
            if n - 1 == int(math.sqrt(n - 1)) ** 2:
                patch_tokens = feat[:, 1:, :]
            else:
                patch_tokens = feat
            h = w = int(math.sqrt(patch_tokens.shape[1]))
            patch_tokens = patch_tokens.permute(0, 2, 1).reshape(b, c, h, w)
            feats.append(patch_tokens)

        return feats


if __name__ == "__main__":
    model = DinoV3()
    x = torch.randn(1, 3, 640, 640)
    feats = model(x)
    for f in feats:
        print(f.shape)
