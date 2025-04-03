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
    def __init__(self, shape=(640, 640), out_feature_indexes=[2, 4, 5, 9], size="base", use_registers=True, use_windowed_attn=True, gradient_checkpointing=False):
        super().__init__()

        name = f"facebook/dinov2-with-registers-{size}" if use_registers else f"facebook/dinov2-{size}"

        self.shape = shape
        
        # Create the encoder
        
        if not use_windowed_attn:
            assert not gradient_checkpointing, "Gradient checkpointing is not supported for non-windowed attention"
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
                    gradient_checkpointing=gradient_checkpointing,
                )
            else:
                windowed_dino_config = WindowedDinov2WithRegistersConfig(
                    **dino_config.to_dict(),
                    num_windows=4,
                    window_block_indexes=window_block_indexes,
                    num_register_tokens=0,
                    gradient_checkpointing=gradient_checkpointing,
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
        shape = self.shape
        def make_new_interpolated_pos_encoding(
            position_embeddings, patch_size, height, width
        ):

            num_positions = position_embeddings.shape[1] - 1
            dim = position_embeddings.shape[-1]
            height = height // patch_size
            width = width // patch_size

            class_pos_embed = position_embeddings[:, 0]
            patch_pos_embed = position_embeddings[:, 1:]

            # Reshape and permute
            patch_pos_embed = patch_pos_embed.reshape(
                1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim
            )
            patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

            # Use bilinear interpolation without antialias
            patch_pos_embed = F.interpolate(
                patch_pos_embed,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )

            # Reshape back
            patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, -1, dim)
            return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

        # If the shape of self.encoder.embeddings.position_embeddings
        # matches the shape of your new tensor, use copy_:
        with torch.no_grad():
            new_positions = make_new_interpolated_pos_encoding(
                self.encoder.embeddings.position_embeddings,
                self.encoder.config.patch_size,
                shape[0],
                shape[1],
            )
        # Create a new Parameter with the new size
        old_interpolate_pos_encoding = self.encoder.embeddings.interpolate_pos_encoding
        def new_interpolate_pos_encoding(self_mod, embeddings, height, width):
            num_patches = embeddings.shape[1] - 1
            num_positions = self_mod.position_embeddings.shape[1] - 1
            if num_patches == num_positions and height == width:
                return self_mod.position_embeddings
            return old_interpolate_pos_encoding(embeddings, height, width)

        self.encoder.embeddings.position_embeddings = nn.Parameter(new_positions)
        self.encoder.embeddings.interpolate_pos_encoding = types.MethodType(
            new_interpolate_pos_encoding, 
            self.encoder.embeddings
        )

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
