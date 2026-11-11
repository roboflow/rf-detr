# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Build and store the golden values that pin the DINOv2 backbone's current behaviour.

The goldens record what the backbone does today, so a rewrite that should change nothing (dropping the ``transformers``
base classes, for example) is checked against values captured before it. The tests in
``test_dinov2_characterization.py`` compute the same payloads with the functions below and compare them with the
committed files. Regenerate the files only for a deliberate behaviour change:

```bash python -m tests.models.backbone._dinov2_goldens ```

Key lists, shapes and random-number-generator digests compare exactly. Tensor values compare with ``rtol=1e-5`` and
``atol=1e-6``, because CPU kernels (random numbers, interpolation, matmul) may round differently across architectures
and torch builds. Each file records the platform and torch version that produced it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from rfdetr.config import RFDETRNanoConfig, RFDETRSegNanoConfig
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
)
from rfdetr.models.lwdetr import build_model_from_config

#: Directory holding the committed golden files.
GOLDEN_DIR = Path(__file__).parent / "goldens"
#: Key and shape golden of every DINOv2 encoder name and of two full models.
KEYS_GOLDEN = GOLDEN_DIR / "dinov2_state_dict_keys.json"
#: Seeded random-init golden of the tiny backbone; its tensors also serve as the forward golden's weights.
INIT_GOLDEN = GOLDEN_DIR / "dinov2_tiny_init.pt"
#: Forward golden of the tiny backbone: inputs and feature maps per attention, window and input-size case.
FORWARD_GOLDEN = GOLDEN_DIR / "dinov2_tiny_forward.pt"
#: Golden of the CPU random-number-generator state after building a model that loads DINOv2 hub weights.
HUB_RNG_GOLDEN = GOLDEN_DIR / "dinov2_hub_rng.json"

#: Seed used for every seeded golden.
SEED = 0

#: Constructor arguments of each DINOv2 encoder name, as the released detection configs pass them. The
#: registers encoder exports only stage 12: its JSON config pins a single-stage ``out_indices``, and a multi-stage
#: request raises (see ``TestOutFeatureIndexesValidation``).
ENCODER_CASES: dict[str, dict[str, Any]] = {
    "dinov2_windowed_small": {"out_feature_indexes": [3, 6, 9, 12]},
    "dinov2_windowed_base": {"out_feature_indexes": [3, 6, 9, 12]},
    "dinov2_registers_windowed_small": {"out_feature_indexes": [12]},
}

#: Input sizes of the forward golden: the native grid (no position-embedding interpolation) and a resized
#: non-square grid (interpolation). Both divide by ``patch_size * num_windows`` for one and two windows.
FORWARD_SIZES: dict[str, tuple[int, int]] = {"native": (16, 16), "resized": (16, 24)}
#: Attention implementations of the forward golden.
FORWARD_ATTENTIONS = ("eager", "sdpa")
#: Window counts of the forward golden.
FORWARD_WINDOWS = (1, 2)


def platform_meta() -> dict[str, str]:
    """Describe the platform and library versions that produced a golden.

    Returns:
        Platform, machine, Python and torch versions.

    Examples:
        >>> sorted(platform_meta())
        ['machine', 'platform', 'python', 'torch']
    """
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        # A plain ``str``: ``torch.__version__`` is a ``TorchVersion``, which ``torch.load(weights_only=True)`` refuses.
        "torch": str(torch.__version__),
    }


def rng_digest() -> str:
    """Hash the current CPU random-number-generator state.

    Two runs that consumed the same random numbers since the same seed give the same digest.

    Returns:
        SHA-256 hex digest of ``torch.get_rng_state()``.

    Examples:
        >>> _ = torch.manual_seed(0)
        >>> first = rng_digest()
        >>> _ = torch.manual_seed(0)
        >>> rng_digest() == first
        True
    """
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()


def tiny_config(num_windows: int = 1, attn_implementation: str = "sdpa", **overrides: Any) -> Any:
    """Build a tiny backbone config, small enough that its weights fit a golden file of a few KB.

    The attention implementation is set as an attribute after construction, the way the forward golden selects it.

    Args:
        num_windows: Windows per side for windowed attention.
        attn_implementation: ``"eager"`` or ``"sdpa"``.
        **overrides: Further config fields, such as ``gradient_checkpointing``.

    Returns:
        A ``WindowedDinov2WithRegistersConfig`` with 8 hidden channels, 2 layers and 4-pixel patches.

    Examples:
        >>> config = tiny_config(num_windows=2, attn_implementation="eager")
        >>> config.hidden_size, config.num_hidden_layers, config.num_windows, config._attn_implementation
        (8, 2, 2, 'eager')
    """
    config = WindowedDinov2WithRegistersConfig(
        hidden_size=8,
        num_hidden_layers=2,
        num_attention_heads=2,
        image_size=16,
        patch_size=4,
        num_register_tokens=0,
        out_features=["stage1", "stage2"],
        num_windows=num_windows,
        window_block_indexes=[0],
        return_dict=False,
        **overrides,
    )
    config._attn_implementation = attn_implementation
    return config


def tiny_backbone(num_windows: int = 1, attn_implementation: str = "sdpa", **overrides: Any) -> Any:
    """Build the tiny backbone with random initialization.

    Args:
        num_windows: Windows per side for windowed attention.
        attn_implementation: ``"eager"`` or ``"sdpa"``.
        **overrides: Further config fields, such as ``gradient_checkpointing``.

    Returns:
        A ``WindowedDinov2WithRegistersBackbone``.

    Examples:
        >>> len(tiny_backbone().state_dict())
        43
    """
    return WindowedDinov2WithRegistersBackbone(tiny_config(num_windows, attn_implementation, **overrides))


def key_shapes(state_dict: dict[str, torch.Tensor]) -> dict[str, str]:
    """Map each state-dict key to its tensor shape, written as ``"AxBxC"`` so the golden keeps one key per line.

    Args:
        state_dict: Module state dict.

    Returns:
        Key to shape, in state-dict order; a scalar's shape is ``""``.

    Examples:
        >>> key_shapes({"weight": torch.zeros(2, 3), "step": torch.tensor(1)})
        {'weight': '2x3', 'step': ''}
    """
    return {key: "x".join(str(dim) for dim in value.shape) for key, value in state_dict.items()}


def encoder_key_shapes(encoder: str) -> dict[str, str]:
    """Build a released-size DINOv2 encoder on the meta device and return its key shapes.

    The meta device skips weight allocation and initialization, so building ``dinov2_windowed_base`` costs no
    memory. Patch size 16 with 24 position encodings is the Nano/Small/Medium/Large detection geometry.

    Args:
        encoder: Encoder name, a key of :data:`ENCODER_CASES`.

    Returns:
        Key to shape of the ``DinoV2`` wrapper, whose keys sit under ``backbone.0.encoder.`` in a checkpoint.

    Examples:
        >>> shapes = encoder_key_shapes("dinov2_windowed_small")
        >>> shapes["encoder.embeddings.position_embeddings"]
        '1x577x384'
    """
    with torch.device("meta"):
        backbone = Backbone(
            name=encoder,
            projector_scale=["P4"],
            load_dinov2_weights=False,
            patch_size=16,
            num_windows=2,
            positional_encoding_size=24,
            **ENCODER_CASES[encoder],
        )
    return key_shapes(backbone.encoder.state_dict())


def model_key_shapes(model_config: Any) -> dict[str, str]:
    """Build a full model on the meta device and return its key shapes.

    Args:
        model_config: Model config, such as ``RFDETRNanoConfig(pretrain_weights=None)``.

    Returns:
        Key to shape of the full model, the keys a checkpoint's ``"model"`` entry holds.

    Examples:
        >>> shapes = model_key_shapes(RFDETRNanoConfig(pretrain_weights=None))
        >>> shapes["backbone.0.encoder.encoder.embeddings.cls_token"]
        '1x1x384'
    """
    with torch.device("meta"):
        model = build_model_from_config(model_config)
    return key_shapes(model.state_dict())


def lora_key_shapes() -> dict[str, str]:
    """Build ``RFDETRNano`` on the meta device, wrap its backbone with LoRA adapters and return its key shapes.

    Needs the optional ``peft`` package.

    Returns:
        Key to shape of the LoRA-wrapped model, the keys a ``backbone_lora=True`` checkpoint holds.

    Examples:
        >>> shapes = lora_key_shapes()  # doctest: +SKIP
        >>> any("lora_A" in key for key in shapes)  # doctest: +SKIP
        True
    """
    from rfdetr.models.weights import apply_lora

    with torch.device("meta"):
        model = build_model_from_config(RFDETRNanoConfig(pretrain_weights=None))
        apply_lora(model)
    return key_shapes(model.state_dict())


def build_keys_golden() -> dict[str, dict[str, str]]:
    """Collect the key shapes of every encoder name, ``RFDETRNano`` (plain and LoRA-wrapped) and ``RFDETRSegNano``.

    Returns:
        Case name to key shapes.

    Examples:
        >>> sorted(build_keys_golden())  # doctest: +SKIP
        ['dinov2_registers_windowed_small', 'dinov2_windowed_base', 'dinov2_windowed_small', 'rfdetr_nano', ...]
    """
    cases = {encoder: encoder_key_shapes(encoder) for encoder in ENCODER_CASES}
    cases["rfdetr_nano"] = model_key_shapes(RFDETRNanoConfig(pretrain_weights=None))
    cases["rfdetr_nano_lora"] = lora_key_shapes()
    cases["rfdetr_seg_nano"] = model_key_shapes(RFDETRSegNanoConfig(pretrain_weights=None))
    return cases


def build_init_golden() -> dict[str, Any]:
    """Build the tiny backbone from :data:`SEED` and return its tensors and the generator state after the build.

    Returns:
        ``state_dict`` (contiguous CPU tensors) and ``rng_digest`` after construction.

    Examples:
        >>> golden = build_init_golden()
        >>> len(golden["state_dict"]), len(golden["rng_digest"])
        (43, 64)
    """
    torch.manual_seed(SEED)
    backbone = tiny_backbone()
    digest = rng_digest()
    return {
        "state_dict": {key: value.detach().clone().contiguous() for key, value in backbone.state_dict().items()},
        "rng_digest": digest,
    }


def forward_case_ids() -> list[str]:
    """List the forward golden's case ids, one per attention, window count and input size.

    Returns:
        Ids of the form ``"<attention>-w<windows>-<size>"``.

    Examples:
        >>> forward_case_ids()[:2]
        ['eager-w1-native', 'eager-w1-resized']
    """
    return [
        f"{attention}-w{windows}-{size}"
        for attention in FORWARD_ATTENTIONS
        for windows in FORWARD_WINDOWS
        for size in FORWARD_SIZES
    ]


def run_forward_case(case_id: str, weights: dict[str, torch.Tensor], inputs: dict[str, torch.Tensor]) -> list[Any]:
    """Run the tiny backbone of one forward case on its input.

    Args:
        case_id: A value from :func:`forward_case_ids`.
        weights: State dict loaded into the backbone, so the case does not depend on initialization.
        inputs: Input tensor per size name of :data:`FORWARD_SIZES`.

    Returns:
        Feature maps of the two output stages.

    Examples:
        >>> weights = build_init_golden()["state_dict"]
        >>> maps = run_forward_case("sdpa-w2-resized", weights, {"resized": torch.zeros(1, 3, 16, 24)})
        >>> [list(feature_map.shape) for feature_map in maps]
        [[1, 8, 4, 6], [1, 8, 4, 6]]
    """
    attention, windows, size = case_id.split("-")
    backbone = tiny_backbone(num_windows=int(windows[1:]), attn_implementation=attention)
    backbone.load_state_dict(weights)
    backbone.eval()
    with torch.no_grad():
        return list(backbone(inputs[size])[0])


def build_forward_golden(weights: dict[str, torch.Tensor]) -> dict[str, Any]:
    """Run every forward case on fixed inputs and return the inputs and feature maps.

    Args:
        weights: State dict loaded into every case's backbone.

    Returns:
        ``inputs`` per size name and ``outputs`` per case id.

    Examples:
        >>> golden = build_forward_golden(build_init_golden()["state_dict"])
        >>> len(golden["outputs"]), sorted(golden["inputs"])
        (8, ['native', 'resized'])
    """
    generator = torch.Generator().manual_seed(SEED)
    inputs = {name: torch.randn(1, 3, *size, generator=generator) for name, size in FORWARD_SIZES.items()}
    outputs = {case_id: run_forward_case(case_id, weights, inputs) for case_id in forward_case_ids()}
    return {"inputs": inputs, "outputs": outputs}


def write_goldens() -> None:
    """Write every golden that builds in-process: key shapes, random init and forward.

    The hub RNG golden comes from a child process with an offline Hugging Face cache; the hub test prints the
    value it measured when it differs.

    Examples:
        >>> write_goldens()  # doctest: +SKIP
    """
    GOLDEN_DIR.mkdir(exist_ok=True)
    meta = platform_meta()
    KEYS_GOLDEN.write_text(json.dumps({"meta": meta, "cases": build_keys_golden()}, indent=4) + "\n", newline="\n")
    init = build_init_golden()
    torch.save({"meta": meta, **init}, INIT_GOLDEN)
    torch.save({"meta": meta, **build_forward_golden(init["state_dict"])}, FORWARD_GOLDEN)


if __name__ == "__main__":
    write_goldens()
    sys.stdout.write(f"wrote goldens to {GOLDEN_DIR}\n")
