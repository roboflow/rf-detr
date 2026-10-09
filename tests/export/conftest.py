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
from typing import Any

import numpy as np
import torch
import torchvision.transforms.functional as TF  # noqa: N812 — standard torchvision alias
from PIL import Image
from supervision import Detections, KeyPoints

from rfdetr import RFDETRInference, RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
from rfdetr.detr import RFDETR

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

    Examples:
        >>> t = _structured_parity_input(2, 3, 8, 8)
        >>> t.shape
        torch.Size([2, 3, 8, 8])
        >>> t.dtype
        torch.float32
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

    Examples:
        >>> import tempfile
        >>> from pathlib import Path
        >>> from PIL import Image as _PIL
        >>> with tempfile.TemporaryDirectory() as d:
        ...     p = Path(d) / "img.png"
        ...     _PIL.new("RGB", (64, 64), color=(128, 64, 32)).save(p)
        ...     t = _parity_input_from_image(p, 32)
        ...     t.shape
        torch.Size([1, 3, 32, 32])
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

    Examples:
        >>> import torch.nn as nn
        >>> class _Identity(nn.Module):
        ...     def forward(self, x):
        ...         return (x,)
        >>> t = torch.zeros(1, 3, 4, 4)
        >>> outs = eager_reference_tensors(_Identity(), t)
        >>> len(outs)
        1
        >>> outs[0].shape
        torch.Size([1, 3, 4, 4])
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

    Examples:
        >>> import torch
        >>> a = [torch.tensor([1.0, 2.0])]
        >>> b = [torch.tensor([1.0, 2.5])]
        >>> max_abs_output_diffs(a, b)
        [0.5]
        >>> max_abs_output_diffs(a, a)
        [0.0]
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


def _prediction_model_for_task(task: str) -> RFDETR:
    """Build a small random model with explicit labels and no downloaded weights.

    Examples:
        >>> _prediction_model_for_task("detect").class_names
        ['alpha', 'beta']
    """
    kwargs: dict[str, Any] = {
        "pretrain_weights": None,
        "device": "cpu",
        "resolution": 64 if task == "detect" else 96,
        "num_queries": 4,
        "num_select": 4,
        "num_classes": 2,
    }
    if task == "keypoints":
        kwargs["num_keypoints_per_class"] = [3]
        model = RFDETRKeypointPreview(**kwargs)
        model.model.class_names = ["alpha"]
    else:
        model = (RFDETRSegNano if task == "segment" else RFDETRNano)(**kwargs)
        model.model.class_names = ["alpha", "beta"]
    return model


def _prediction_image(size: int) -> np.ndarray:
    """Make an RGB image with stable spatial structure.

    Examples:
        >>> _prediction_image(4).shape
        (4, 4, 3)
    """
    y, x = np.indices((size, size))
    return np.stack(((x * 7) % 256, (y * 11) % 256, ((x + y) * 3) % 256), axis=-1).astype(np.uint8)


def assert_prediction_roundtrip(format_name: str, task: str, tmp_path: Path) -> None:
    """Run the public export and prediction APIs on one image.

    Examples:
        Requires a real export runtime and a temporary artifact directory.

        >>> assert_prediction_roundtrip("onnx", "detect", Path("output"))  # doctest: +SKIP
    """
    torch.manual_seed(17)
    native = _prediction_model_for_task(task)
    size = 64 if task == "detect" else 96
    image = _prediction_image(size)
    original = native.predict(image, threshold=0.0, include_source_image=False)
    settings: dict[str, Any] = {}
    if format_name == "tensorrt":
        settings["fp16"] = False
    elif format_name == "executorch":
        settings["backend"] = "xnnpack"
    elif format_name == "coreml":
        settings["coreml_precision"] = "float32"
    elif format_name == "coreai":
        settings["coreai_precision"] = "float32"
    elif format_name == "openvino":
        settings["openvino_precision"] = "float32"
    artifact = native.export(format=format_name, output_dir=str(tmp_path), verbose=False, **settings)
    exported = RFDETRInference(artifact, device="cuda:0" if format_name == "tensorrt" else "cpu")
    actual = exported.predict(image, threshold=0.0, include_source_image=False)

    assert exported.class_names == native.class_names
    assert type(actual) is type(original)
    normalized_box_atol = {
        "onnx": 1e-3,
        "openvino": 1e-2,
        "tflite": 2e-2,
        "litert": 1e-2,
        "executorch": 1e-2,
        "coreml": 2e-2,
        "coreai": 2e-2,
        "tensorrt": 2e-2,
    }[format_name]
    score_atol = {
        "onnx": 1e-3,
        "openvino": 2e-2,
        "tflite": 3e-2,
        "litert": 2e-2,
        "executorch": 2e-2,
        "coreml": 3e-2,
        "coreai": 3e-2,
        "tensorrt": 3e-2,
    }[format_name]
    if task == "keypoints":
        assert isinstance(actual, KeyPoints) and isinstance(original, KeyPoints)
        assert actual.detection_confidence is not None and original.detection_confidence is not None
        assert actual.keypoint_confidence is not None and original.keypoint_confidence is not None
        np.testing.assert_allclose(actual.xy, original.xy, atol=normalized_box_atol * size, rtol=0)
        np.testing.assert_allclose(actual.detection_confidence, original.detection_confidence, atol=score_atol, rtol=0)
        np.testing.assert_allclose(actual.keypoint_confidence, original.keypoint_confidence, atol=score_atol, rtol=0)
    else:
        assert isinstance(actual, Detections) and isinstance(original, Detections)
        assert actual.confidence is not None and original.confidence is not None
        np.testing.assert_array_equal(actual.class_id, original.class_id)
        np.testing.assert_allclose(actual.xyxy, original.xyxy, atol=normalized_box_atol * size, rtol=0)
        np.testing.assert_allclose(actual.confidence, original.confidence, atol=score_atol, rtol=0)
        if task == "segment":
            assert actual.mask is not None and original.mask is not None
            assert np.mean(actual.mask != original.mask) < 0.02
