# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared export-parity helpers for the backend export test suites.

These builders and comparison seams are backend-agnostic and used by the CoreML, ExecuTorch, and TensorRT end-to-end
tests so the three suites stay structurally aligned instead of each re-implementing input synthesis and output
comparison.

Backend-specific execution (``coremltools``/``executorch``/``polygraphy`` runtimes) stays in each test module; only the
eager reference forward, the deterministic input builders, and the per-output max-abs-diff loop live here.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torchvision.transforms.functional as TF  # noqa: N812 — standard torchvision alias
from PIL import Image

# ImageNet statistics used by predict-style preprocessing; parity inputs are normalized to
# roughly this range so the backbone sees realistic activations.
_IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
_IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)


def _structured_parity_input(
    batch: int,
    channels: int,
    height: int,
    width: int,
) -> torch.Tensor:
    """Build a deterministic, spatially correlated ``NCHW`` tensor for export parity.

    Combines a smooth spatial gradient with a coarse checkerboard so the backbone sees
    local structure (unlike ``torch.randn``, which can mask export/runtime divergence).

    Args:
        batch: Batch size.
        channels: Channel count (typically 3).
        height: Spatial height.
        width: Spatial width.

    Returns:
        Float tensor shaped ``(batch, channels, height, width)`` in roughly ImageNet-normalized range.
    """
    ys = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
    xs = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
    gradient = (0.35 * ys + 0.25 * xs).expand(1, channels, height, width).clone()
    # Per-channel offset so RGB planes are not identical.
    for c in range(channels):
        gradient[:, c] = gradient[:, c] + 0.05 * (c - 1)

    tile = 16
    yy = (torch.arange(height).view(height, 1) // tile) % 2
    xx = (torch.arange(width).view(1, width) // tile) % 2
    checker = ((yy + xx) % 2).to(dtype=torch.float32).view(1, 1, height, width)
    checker = (checker * 0.4 - 0.2).expand(1, channels, height, width)

    sample = gradient + checker
    return sample.expand(batch, channels, height, width).contiguous()


def _parity_input_from_image(path: Path, resolution: int) -> torch.Tensor:
    """Load an RGB image, resize to square ``resolution``, and ImageNet-normalize (predict-style).

    Args:
        path: Path to an RGB image file.
        resolution: Square side length matching the exported model.

    Returns:
        Float tensor shaped ``(1, 3, resolution, resolution)``.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
    """
    if not path.is_file():
        raise FileNotFoundError(f"parity fixture image not found: {path}")
    with Image.open(path) as img:
        image = img.convert("RGB")
    tensor = TF.to_tensor(image)
    tensor = TF.resize(tensor, [resolution, resolution], antialias=False)
    tensor = TF.normalize(tensor, _IMAGENET_MEAN, _IMAGENET_STD)
    return tensor.unsqueeze(0)


def eager_reference_tensors(pytorch_model: torch.nn.Module, example_input: torch.Tensor) -> list[torch.Tensor]:
    """Run *example_input* through the eager export-mode model; return flattened float CPU output tensors.

    The export-mode ``forward`` can mutate its input in place, so a fresh clone is fed in. The output
    (tuple or pytree) is flattened and filtered to tensors so backends that yield outputs in a flat
    order can be paired positionally.

    Args:
        pytorch_model: Export-mode PyTorch module on CPU.
        example_input: ``(N, C, H, W)`` example tensor.

    Returns:
        One detached float CPU tensor per model output, in flattened order.

    Raises:
        AssertionError: If the model produces no tensor outputs.
    """
    from torch.utils._pytree import tree_flatten

    with torch.no_grad():
        eager_out = pytorch_model(example_input.clone())
    tensors = [t.detach().float().cpu() for t in tree_flatten(eager_out)[0] if isinstance(t, torch.Tensor)]
    assert tensors, "export-mode forward produced no tensor outputs to compare"
    return tensors


def max_abs_output_diffs(
    eager_tensors: list[torch.Tensor],
    other_tensors: list[torch.Tensor],
    *,
    check_shape: bool = True,
    names: list[str] | None = None,
) -> list[float]:
    """Pair eager and backend output tensors positionally; return per-output max-abs-diff.

    Args:
        eager_tensors: Reference tensors from :func:`eager_reference_tensors`.
        other_tensors: Backend (CoreML/ExecuTorch/TensorRT) output tensors, same order.
        check_shape: Assert each paired output has matching shape before diffing.
        names: Optional backend output names, used only to enrich assertion messages.

    Returns:
        One max-abs-diff per output tensor.

    Raises:
        AssertionError: If output counts (or, when ``check_shape``, shapes) disagree.
    """
    assert len(other_tensors) == len(eager_tensors), (
        f"backend output count {len(other_tensors)} != PyTorch {len(eager_tensors)}"
        + (f" (names={names})" if names is not None else "")
    )
    diffs: list[float] = []
    for idx, (eager, other) in enumerate(zip(eager_tensors, other_tensors)):
        if check_shape:
            name_hint = f" (name={names[idx]!r})" if names is not None else ""
            assert eager.shape == other.shape, (
                f"output[{idx}] shape mismatch: PyTorch {tuple(eager.shape)} vs backend {tuple(other.shape)}{name_hint}"
            )
        diffs.append((eager - other.float()).abs().max().item())
    return diffs
