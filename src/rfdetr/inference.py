# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Public inference facade and model-context builder for RF-DETR."""

from __future__ import annotations

__all__ = ["ModelContext", "RFDETRInference"]

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from PIL import Image

from rfdetr._prediction import PredictionContext, predict
from rfdetr.config import TrainConfig
from rfdetr.models import PostProcess, build_model
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.weights import apply_lora, load_pretrain_weights

if TYPE_CHECKING:
    from supervision import Detections, KeyPoints

    from rfdetr.config import ModelConfig
    from rfdetr.detr import RFDETR


class RFDETRInference:
    """Predict with a live native model, checkpoint, or exported artifact.

    This facade shares prediction behavior across native models and exported runtimes.
    """

    def __init__(
        self,
        source: RFDETR | str | os.PathLike[str],
        *,
        device: str = "auto",
        metadata: dict[str, Any] | str | os.PathLike[str] | None = None,
        trust_checkpoint: bool = False,
    ) -> None:
        """Create a prediction facade from native weights or an exported artifact.

        Args:
            source: A live RFDETR model or a checkpoint or exported artifact path.
            device: Runtime device, or ``"auto"`` to inherit the source device.
            metadata: Optional metadata for an exported artifact.
            trust_checkpoint: Allow loading a checkpoint that contains custom Python objects.

        Raises:
            ValueError: If metadata or a device conflicts with a native source.

        Native device selection uses exact ``torch.device`` matching. For example,
        ``"cuda"`` and ``"cuda:0"`` are different explicit device values.
        """
        # Keep RFDETR imports local because detr imports ModelContext from this module.
        from rfdetr.detr import RFDETR

        self._native_model: RFDETR | None = None
        self._export_context: PredictionContext | None = None
        if isinstance(source, RFDETR):
            self._set_native_model(source, device, metadata)
            return

        path = Path(source)
        if path.suffix.lower() in {".pt", ".pth", ".ckpt"}:
            if metadata is not None:
                raise ValueError("metadata is only valid for exported artifacts.")
            checkpoint_options: dict[str, Any] = {"trust_checkpoint": trust_checkpoint}
            if device != "auto":
                checkpoint_options["device"] = device
            native_model = RFDETR.from_checkpoint(path, **checkpoint_options)
            self._set_native_model(native_model, device, None)
            return

        from rfdetr.export._runtime.context import load_exported_context

        self._export_context = load_exported_context(path, device=device, metadata=metadata)

    def _set_native_model(
        self,
        native_model: RFDETR,
        device: str,
        metadata: dict[str, Any] | str | os.PathLike[str] | None,
    ) -> None:
        """Store a native model after validating facade-only arguments."""
        if metadata is not None:
            raise ValueError("metadata is only valid for exported artifacts.")
        model_device = native_model.model.device
        if device != "auto" and torch.device(device) != model_device:
            raise ValueError(f"The live model uses device {model_device}, but device={device!r} was requested.")
        self._native_model = native_model

    def _prediction_context(self) -> PredictionContext:
        """Build a fresh context for prediction or return the loaded export context."""
        if self._native_model is not None:
            return self._native_model._prediction_context()
        assert self._export_context is not None
        return self._export_context

    @property
    def class_names(self) -> list[str]:
        """Return a copy of the source class names."""
        if self._native_model is not None:
            return self._native_model.class_names
        return list(self._prediction_context().class_names)

    @property
    def runtime_info(self) -> dict[str, Any]:
        """Return the runtime and device policy for the source."""
        if self._native_model is not None:
            return {"backend": "pytorch", "device": str(self._native_model.model.device)}
        return dict(self._prediction_context().runtime_info)

    @torch.inference_mode()
    def predict(
        self,
        images: str
        | Image.Image
        | np.ndarray[Any, Any]
        | torch.Tensor
        | list[str | np.ndarray[Any, Any] | Image.Image | torch.Tensor],
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
        patch_size: int | None = None,
        include_source_image: bool = True,
        **kwargs: Any,
    ) -> Detections | KeyPoints | list[Detections | KeyPoints]:
        """Run prediction with the shared native and exported inference pipeline.

        Args:
            images: One image or a batch of images accepted by RF-DETR prediction.
            threshold: Minimum confidence score for a prediction.
            shape: Optional input height and width.
            patch_size: Optional patch size used for shape validation.
            include_source_image: Include each source image in prediction metadata.
            **kwargs: Additional options accepted by the shared prediction pipeline.

        Returns:
            A Supervision prediction object or a list of prediction objects.
        """
        return predict(
            self._prediction_context(),
            images,
            threshold=threshold,
            shape=shape,
            patch_size=patch_size,
            include_source_image=include_source_image,
            **kwargs,
        )


class ModelContext:
    """Lightweight model wrapper returned by RFDETR.get_model().

    Provides the same attribute interface as the legacy ``main.py:Model`` but without importing or depending on
    ``populate_args()`` or the legacy stack.

    Args:
        model: The underlying ``LWDETR`` module. The attribute is cleared to ``None`` by
            :meth:`RFDETR.inference` when called with ``inplace=True``, which frees the
            weights from memory.
        postprocess: PostProcess instance for converting raw outputs to boxes.
        device: Device the model lives on. An index-less ``torch.device("cuda")`` is replaced by the concrete
            device (e.g. ``cuda:0``) that ``rfdetr.detr._move_model_context_to_device`` resolves on the deferred
            first-use move, so every later caller targets that same GPU whatever its own thread selected.
        resolution: Input resolution (square side length in pixels).
        args: Namespace of resolved training/model configuration.
        class_names: Optional list of class name strings loaded from checkpoint.
    """

    def __init__(
        self,
        model: LWDETR,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: list[str] | None = None,
    ) -> None:
        self.model: LWDETR | None = model
        self.postprocess = postprocess
        self.device = device
        self.resolution = resolution
        self.args = args
        self.class_names = class_names
        self.inference_model = None

    def reinitialize_detection_head(self, num_classes: int) -> None:
        """Reinitialize the detection head for a different number of classes.

        Args:
            num_classes: New number of output classes (including background).

        Raises:
            RuntimeError: If the model weights were already cleared by ``RFDETR.inference(inplace=True)``.
        """
        if self.model is None:
            raise RuntimeError(
                "Cannot reinitialize the detection head after inplace optimization. "
                "The original model has been cleared. Create a new RFDETR instance."
            )
        reinitialize_head = cast("Callable[[int], None]", self.model.reinitialize_detection_head)
        reinitialize_head(num_classes)
        self.args.num_classes = num_classes


_ModelContext = ModelContext  # backward-compat alias


def _adapt_input_conv(num_channels: int, conv_weight: torch.Tensor) -> torch.Tensor:
    """Adapt a 3-channel pretrained conv weight tensor to *num_channels* input channels.

    When ``num_channels == 3``: returns the weight unchanged. When ``num_channels == 1``: averages weights across the
    original 3 channels.
    Otherwise (``num_channels != 1`` and ``num_channels != 3``): tiles the 3-channel
    pattern and scales by ``3 / num_channels`` to preserve activation magnitude.

    Args:
        num_channels: Target number of input channels.
        conv_weight: Original weight tensor of shape ``[out_ch, 3, H, W]``.

    Returns:
        Adapted weight tensor of shape ``[out_ch, num_channels, H, W]``.
    """
    if num_channels == 3:
        return conv_weight
    if num_channels == 1:
        return conv_weight.mean(dim=1, keepdim=True)
    # General case: tile and scale
    repeats = (num_channels + 2) // 3
    weight_out = torch.cat([conv_weight] * repeats, dim=1)[:, :num_channels]
    weight_out = weight_out * (3.0 / num_channels)
    return weight_out


def _build_model_context(model_config: ModelConfig, *, trust_checkpoint: bool = False) -> ModelContext:
    """Build a ModelContext from ModelConfig without using legacy main.py:Model.

    Replicates ``Model.__init__`` logic: builds the nn.Module, optionally loads pretrain weights and applies LoRA.  The
    model is intentionally kept on CPU; :func:`_ensure_model_on_device` in ``detr.py`` performs the deferred
    ``.to(device)`` on the first ``predict()`` / ``export()`` / ``inference()`` call.  Keeping construction
    CPU-only prevents CUDA initialisation during ``__init__``, which would block DDP strategies (``ddp_notebook``,
    ``ddp_spawn``) from spawning child processes in notebook environments.

    Args:
        model_config: Architecture configuration.
        trust_checkpoint: Forwarded to :func:`~rfdetr.models.weights.load_pretrain_weights` as its
            ``trust`` argument — set ``True`` only when ``model_config.pretrain_weights`` is a
            checkpoint the caller explicitly trusts (mirrors ``RFDETR.from_checkpoint(...,
            trust_checkpoint=True)``).

    Returns:
        ModelContext with the model on CPU, ready for lazy device placement.
    """
    from rfdetr._namespace import _namespace_from_configs

    # A dummy TrainConfig is needed only for _namespace_from_configs' required fields;
    # dataset_dir/output_dir are unused during model construction.
    dummy_train_config = TrainConfig(dataset_dir=None, output_dir="output")
    args = _namespace_from_configs(model_config, dummy_train_config)
    # ``TrainConfig.expand_paths`` realpaths these to the caller's absolute CWD, which would be embedded into
    # ``args`` and serialized into exported ``weights.pt`` (see ``RFDETR.export_for_roboflow``). Reset them on the
    # namespace to placeholders so inference-built checkpoints never leak the caller's filesystem layout.
    args.dataset_dir = None
    args.output_dir = "output"
    nn_model = build_model(args)
    assert isinstance(nn_model, LWDETR), (
        "build_model() returned a non-LWDETR result even though encoder_only/backbone_only were not set."
    )

    class_names: list[str] = []
    if model_config.pretrain_weights is not None:
        class_names = load_pretrain_weights(nn_model, model_config, trust=trust_checkpoint)
        # ``load_pretrain_weights`` can mutate ``model_config.num_classes`` and
        # ``model_config.num_keypoints_per_class`` when aligning to checkpoint schema.
        # Keep the derived namespace in sync so postprocess and predict() use correct values.
        if hasattr(args, "num_classes") and args.num_classes != model_config.num_classes:
            args.num_classes = model_config.num_classes
        _mc_kp = list(getattr(model_config, "num_keypoints_per_class", []) or [])
        if (
            hasattr(args, "num_keypoints_per_class")
            and list(getattr(args, "num_keypoints_per_class", []) or []) != _mc_kp
        ):
            args.num_keypoints_per_class = _mc_kp

    if model_config.backbone_lora:
        # No-op when load_pretrain_weights already wrapped the encoder to load a LoRA checkpoint.
        apply_lora(nn_model)

    # Adapt patch-embedding projection for non-RGB channel counts
    if model_config.num_channels != 3:
        import copy

        backbone = cast(Backbone, nn_model.backbone[0])
        proj = backbone.encoder.encoder.embeddings.patch_embeddings.projection
        new_proj = copy.deepcopy(proj)
        new_proj.in_channels = model_config.num_channels
        new_weight = _adapt_input_conv(model_config.num_channels, proj.weight)
        new_proj.weight = torch.nn.Parameter(new_weight)
        new_proj.weight.requires_grad = proj.weight.requires_grad
        backbone.encoder.encoder.embeddings.patch_embeddings.projection = new_proj
        backbone.encoder.encoder.embeddings.patch_embeddings.num_channels = model_config.num_channels

    device = torch.device(args.device)
    # Keep the model on CPU here; predict() / export() / inference()
    # will lazily move it to the target device on first use.  Eagerly calling
    # .to("cuda") would initialise the CUDA runtime during __init__(), which
    # prevents DDP strategies (ddp_notebook, ddp_spawn) from forking/spawning
    # child processes in notebook environments.
    postprocess = PostProcess(
        num_select=args.num_select,
        num_keypoints_per_class=getattr(args, "num_keypoints_per_class", []),
        # Older detection-only namespaces may omit keypoint postprocess knobs; keep the ModelConfig default.
        trace_alpha=getattr(args, "postprocess_trace_alpha", 0.2),
    )

    return ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
    )
