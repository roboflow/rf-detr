# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""Format-independent preparation of a model for export.

Every export backend needs the same graph-shaping work before it can trace anything: freeze the DINOv2 position
embeddings to the export shape, build an example input, wrap the backbone when only the encoder is wanted, decide the
graph's input/output names and dynamic axes, and run one forward pass so a broken graph fails here rather than inside a
third-party converter. Keeping that in one place is what stops the six format paths from each growing their own copy —
the drift that already happened once between :meth:`rfdetr.detr.RFDETR.export` and the export CLI that used to live in
``rfdetr.export.main`` (removed in v1.11).

The result is an :class:`ExportGraph`: everything a converter needs, and nothing about *which* format is being written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import torch
from PIL import Image
from torch import Tensor, nn
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from rfdetr.datasets.transforms import Normalize
from rfdetr.export._backend import _BackboneExport
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.utilities.logger import get_logger

logger = get_logger()


class ExportModelConfig(Protocol):
    """The subset of a model config the export preparation reads.

    Declared structurally rather than importing :class:`~rfdetr.config.ModelConfig` so the preparation stays independent
    of how a caller assembles its configuration. The members are read-only properties rather than plain attributes: a
    mutable protocol attribute is invariant, which would reject a config that narrows one of them (``ModelConfig`` types
    *projector_scale* as ``list[Literal["P3", "P4", "P5"]]``).
    """

    @property
    def num_channels(self) -> int:
        """Number of input channels the model consumes."""

    @property
    def projector_scale(self) -> Sequence[Any]:
        """Feature levels the backbone projects — one graph output each for a backbone-only export."""

    @property
    def segmentation_head(self) -> bool:
        """Whether the model predicts masks."""

    @property
    def use_grouppose_keypoints(self) -> bool:
        """Whether the model predicts keypoints."""


@dataclass(frozen=True, slots=True)
class ExportGraph:
    """A model prepared for export, together with the graph metadata every backend needs.

    Attributes:
        model: The module to trace — already in eval mode, on CPU, and wrapped for a backbone-only export.
        input_tensors: Example input the backend traces with, matching *shape* and the model's channel count.
        input_names: Names for the graph's inputs.
        output_names: Names for the graph's outputs, in the order the model returns them.
        dynamic_axes: Axis-name mapping for a dynamic batch dimension, or ``None`` for a fully static graph.
        shape: The ``(height, width)`` the graph was prepared for.
        backbone_only: Whether *model* is the backbone-only export wrapper rather than the full detector.
    """

    model: nn.Module
    input_tensors: Tensor
    input_names: tuple[str, ...]
    output_names: tuple[str, ...]
    dynamic_axes: Mapping[str, Mapping[int, str]] | None
    shape: tuple[int, int]
    backbone_only: bool


def make_infer_image(
    infer_dir: str | None,
    shape: tuple[int, int],
    batch_size: int,
    device: str | torch.device = "cuda",
    num_channels: int = 3,
) -> Tensor:
    """Build the example input tensor an export traces with.

    Args:
        infer_dir: Path to a sample image, or ``None`` to synthesize one.
        shape: Target ``(height, width)``.
        batch_size: Number of copies stacked into the batch dimension.
        device: Device the returned tensor is placed on.
        num_channels: Channel count of the model being exported.

    Returns:
        A ``(batch_size, num_channels, height, width)`` float32 tensor.

    Raises:
        ValueError: If *infer_dir* is given for a model that is not RGB.

    Examples:
        >>> make_infer_image(None, (8, 8), 2, device="cpu").shape
        torch.Size([2, 3, 8, 8])
    """
    if infer_dir is None:
        if num_channels == 3:
            dummy = np.random.randint(0, 256, (shape[0], shape[1], 3), dtype=np.uint8)
            image = Image.fromarray(dummy, mode="RGB")
        else:
            # Non-RGB: build a random float tensor directly, bypassing PIL.
            # Normalization is intentionally skipped here — export tracing only
            # requires tensors of the correct shape and dtype (float32), not the
            # correct distribution.  Real inference normalizes via predict() before
            # the tensor reaches the exported model, so the ONNX/TensorRT graph
            # never sees raw [0, 1] inputs in production.
            inps = torch.rand(batch_size, num_channels, shape[0], shape[1], device=device)
            return inps
    else:
        if num_channels != 3:
            raise ValueError(
                "Providing `infer_dir` is only supported for RGB models (num_channels=3). "
                "For non-RGB models, omit `infer_dir` to use a synthetic dummy input."
            )
        with Image.open(infer_dir) as _img:
            image = _img.convert("RGB")

    # Tensorize-then-resize with antialias=False mirrors RFDETR.predict()'s preprocessing, so the
    # traced example input (and any export sanity check run on it) sees production-domain tensors.
    transforms = Compose(
        [
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Resize((shape[0], shape[1]), antialias=False),
            Normalize(),
        ]
    )

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps


def resolve_output_names(
    model_config: ExportModelConfig, *, backbone_only: bool, backbone: Backbone | None
) -> list[str]:
    """Name the graph's outputs for the model's task.

    A backbone-only export emits one name per projector scale, followed by the cross-attention levels when the
    backbone has a separate cross-attention projector. The full detector emits ``dets``/``labels``, plus the
    task-specific third output.

    Args:
        model_config: Configuration of the model being exported.
        backbone_only: Whether only the backbone is exported.
        backbone: The backbone being exported; required when *backbone_only* is set.

    Returns:
        Output names, in the order the traced model returns them.

    Examples:
        >>> from types import SimpleNamespace
        >>> config = SimpleNamespace(projector_scale=["P4"], segmentation_head=True, use_grouppose_keypoints=False)
        >>> resolve_output_names(config, backbone_only=False, backbone=None)
        ['dets', 'labels', 'masks']
    """
    if backbone_only:
        if backbone is None:
            raise ValueError("backbone must be provided for a backbone-only export")
        names = [
            "features" if index == 0 else f"features_{index}" for index in range(len(model_config.projector_scale))
        ]
        if backbone.cross_attn_projector is not None:
            names.extend(
                "cross_attn_features" if index == 0 else f"cross_attn_features_{index}"
                for index in range(len(model_config.projector_scale))
            )
        return names
    if model_config.segmentation_head:
        return ["dets", "labels", "masks"]
    if model_config.use_grouppose_keypoints:
        return ["dets", "labels", "keypoints"]
    return ["dets", "labels"]


def _log_forward_output_shapes(outputs: object) -> None:
    """Log the shapes a sanity forward pass produced, whatever structure the task returns.

    Args:
        outputs: Whatever the prepared model returned — a sequence of feature maps for a backbone-only
            export, otherwise the detector's output mapping. Segmentation heads may return ``pred_masks``
            as a dict of mask components rather than a single tensor.
    """
    if isinstance(outputs, Mapping):
        for name, value in outputs.items():
            if isinstance(value, Tensor):
                logger.debug(f"PyTorch inference output shape - {name}: {tuple(value.shape)}")
            elif isinstance(value, Mapping):
                # Only Tensor members are reported: the f-string is evaluated regardless of log level, so a
                # non-Tensor member would turn a debug line into an AttributeError that aborts the export.
                for part, tensor in value.items():
                    if isinstance(tensor, Tensor):
                        logger.debug(f"PyTorch inference output shape - {name}.{part}: {tuple(tensor.shape)}")
        return
    if isinstance(outputs, Sequence):
        logger.debug(f"PyTorch backbone output shapes: {[tuple(feature.shape) for feature in outputs]}")
        return
    logger.debug(f"PyTorch inference output shape: {tuple(cast(Tensor, outputs).shape)}")


def prepare_export_graph(
    model: nn.Module,
    model_config: ExportModelConfig,
    *,
    shape: tuple[int, int],
    device: str | torch.device,
    infer_dir: str | None = None,
    batch_size: int = 1,
    dynamic_batch: bool = False,
    backbone_only: bool = False,
) -> ExportGraph:
    """Prepare *model* for tracing and describe the graph a backend is about to write.

    Runs, in order: freeze every DINOv2 backbone's position embeddings to *shape*; build the example input;
    wrap the backbone when *backbone_only* is set; resolve names and dynamic axes; run one forward pass to
    surface a broken graph here; move everything to CPU, where every backend traces.

    Args:
        model: The module to export. Mutated in place — callers pass a copy they own, not a live training model.
        model_config: Configuration of the model being exported.
        shape: Already-validated ``(height, width)`` to export at.
        device: Device the sanity forward pass runs on. Falls back to CPU when CUDA is requested but absent.
        infer_dir: Optional sample image used to build the example input instead of synthesizing one.
        batch_size: Static batch size baked into the graph.
        dynamic_batch: Whether to mark the batch dimension dynamic.
        backbone_only: Export the encoder and feature projectors without the prediction heads.

    Returns:
        The prepared :class:`ExportGraph`.

    Examples:
        Requires a real RF-DETR model, so this is documentation only (not a doctest):

        ```python
        graph = prepare_export_graph(model, model_config, shape=(560, 560), device="cpu")
        graph.output_names
        # -> ('dets', 'labels')
        ```
    """
    # Freeze the backbone's position embeddings to the export shape before tracing. Without this, a
    # `shape` that differs from the model's native resolution forces the traced forward pass through
    # DINOv2's antialiased bicubic interpolation, which has no ONNX symbolic (`aten::_upsample_bicubic2d_aa`
    # is unsupported). Precomputing here (outside the trace) keeps that op out of the traced graph.
    for backbone_module in model.modules():
        if isinstance(backbone_module, DinoV2):
            backbone_module.shape = shape
            backbone_module.export()

    # Resolve the device once, up front: everything below — the example input, the model, the sanity pass — has to
    # land on the same one, and a CUDA request on a machine without CUDA has to degrade here rather than surface as
    # a bare "Torch not compiled with CUDA enabled" from the first allocation.
    run_device = torch.device(device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU for the export preparation.")
        run_device = torch.device("cpu")

    input_tensors = make_infer_image(
        infer_dir, shape, batch_size, run_device, num_channels=model_config.num_channels
    ).to(run_device)

    export_model: nn.Module = model
    backbone: Backbone | None = None
    if backbone_only:
        backbone = cast(Backbone, model.backbone[0])  # type: ignore[index]
        backbone.export()
        export_model = _BackboneExport(backbone)

    input_names = ("input",)
    output_names = tuple(resolve_output_names(model_config, backbone_only=backbone_only, backbone=backbone))
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = (
        {name: {0: "batch"} for name in input_names + output_names} if dynamic_batch else None
    )

    # Run the sanity pass on the device resolved above — not a hard-coded "cuda" — so CPU-only export paths work.
    export_model.eval().to(run_device)
    input_tensors = input_tensors.to(run_device)
    with torch.no_grad():
        _log_forward_output_shapes(export_model(input_tensors))

    model.cpu()
    return ExportGraph(
        model=export_model,
        input_tensors=input_tensors.cpu(),
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        shape=shape,
        backbone_only=backbone_only,
    )
