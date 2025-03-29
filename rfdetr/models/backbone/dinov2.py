import torch
import torch.nn as nn
from transformers import AutoBackbone, AutoConfig
import torch.nn.functional as F
import types
import math

from .dinov2_with_windowed_attn import WindowedDinov2WithRegistersConfig, WindowedDinov2WithRegistersBackbone


size_to_width = {
    "tiny": 192,
    "small": 384,
    "base": 768,
    "large": 1024,
}

class DinoV2(nn.Module):
    def __init__(self, shape=(640, 640), out_feature_indexes=[2, 4, 5, 9], size="base", use_registers=True, use_windowed_attn=True):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        
        # Create the encoder
        
        if not use_windowed_attn:
            self.encoder = AutoBackbone.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
        else:
            window_block_indexes = set(range(out_feature_indexes[-1] + 1))
            window_block_indexes.difference_update(out_feature_indexes)
            window_block_indexes = list(window_block_indexes)

            dino_config = AutoConfig.from_pretrained(
                name,
                out_features=[f"stage{i}" for i in out_feature_indexes],
                return_dict=False,
            )
            if use_registers:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config.to_dict(),
                    num_windows=4,
                    window_block_indexes=window_block_indexes,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config.to_dict(),
                    num_windows=4,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                )
            self.encoder = WindowedDinov2WithRegistersBackbone.from_pretrained(
                name,
                config=windowed_dino_config,
            )


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
