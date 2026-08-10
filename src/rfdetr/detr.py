# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import importlib
import io
import json
import operator
import os
import tempfile
import warnings
from collections import defaultdict
from collections.abc import Callable
from copy import copy, deepcopy
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, Any, Concatenate, Literal, ParamSpec, TypeVar, cast
from urllib.parse import urlparse

import numpy as np
import requests
import torch
import torchvision.transforms.functional as F  # noqa: N812
import yaml
from deprecate import deprecated
from PIL import Image

from rfdetr._namespace import _namespace_from_configs
from rfdetr.assets.coco_classes import COCO_CLASS_NAMES, COCO_CLASSES
from rfdetr.assets.model_weights import download_pretrain_weights, get_model_cache_dir
from rfdetr.config import ModelConfig, TrainConfig
from rfdetr.datasets._keypoint_schema import (
    active_keypoint_counts,
    infer_coco_keypoint_schema,
    infer_yolo_keypoint_schema,
)
from rfdetr.datasets.coco import is_valid_coco_dataset
from rfdetr.datasets.yolo import REQUIRED_YOLO_YAML_FILES, is_valid_yolo_dataset
from rfdetr.inference import ModelContext, _build_model_context
from rfdetr.models.backbone.dinov2 import DinoV2
from rfdetr.utilities.distributed import is_main_process
from rfdetr.utilities.keypoints import _is_bg_first_schema, precision_cholesky_to_pixel_covariance
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from supervision import Detections, KeyPoints

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = get_logger()
_P = ParamSpec("_P")
_R = TypeVar("_R")

# ModelContext and _build_model_context are eagerly imported above (runtime use in get_model).
_VARIANT_EXPORTS = (
    "RFDETRBase",
    "RFDETRKeypointPreview",
    "RFDETRLarge",
    "RFDETRLargeDeprecated",
    "RFDETRMedium",
    "RFDETRNano",
    "RFDETRSeg",
    "RFDETRSeg2XLarge",
    "RFDETRSegLarge",
    "RFDETRSegMedium",
    "RFDETRSegNano",
    "RFDETRSegPreview",
    "RFDETRSegSmall",
    "RFDETRSegXLarge",
    "RFDETRSmall",
)
__all__ = ["RFDETR", "ModelContext", *_VARIANT_EXPORTS]

_CHECKPOINT_MODEL_NAME_EXCLUDED_SYMBOLS = frozenset({"RFDETRLargeDeprecated", "RFDETRSeg"})
_CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS: tuple[str, ...] = tuple(
    class_symbol for class_symbol in _VARIANT_EXPORTS if class_symbol not in _CHECKPOINT_MODEL_NAME_EXCLUDED_SYMBOLS
)
_CHECKPOINT_PLUS_MODEL_NAME_CLASS_SYMBOLS: tuple[str, ...] = ("RFDETRXLarge", "RFDETR2XLarge")
_CHECKPOINT_MODEL_MAP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("keypoint-preview", "RFDETRKeypointPreview"),
    ("seg-2xlarge", "RFDETRSeg2XLarge"),
    ("seg-xxlarge", "RFDETRSeg2XLarge"),
    ("seg-xlarge", "RFDETRSegXLarge"),
    ("seg-large", "RFDETRSegLarge"),
    ("seg-medium", "RFDETRSegMedium"),
    ("seg-small", "RFDETRSegSmall"),
    ("seg-nano", "RFDETRSegNano"),
    ("seg-preview", "RFDETRSegPreview"),
    ("large", "RFDETRLarge"),
    ("medium", "RFDETRMedium"),
    ("small", "RFDETRSmall"),
    ("nano", "RFDETRNano"),
    ("base", "RFDETRBase"),
)
_CHECKPOINT_PLUS_MODEL_MAP_ENTRIES: tuple[tuple[str, str], ...] = (
    ("2xlarge", "RFDETR2XLarge"),
    ("xxlarge", "RFDETR2XLarge"),
    ("xlarge", "RFDETRXLarge"),
)


def _validate_shape_dims(
    shape: object,
    block_size: int,
    patch_size: int,
    num_windows: int,
) -> tuple[int, int]:
    """Validate a user-supplied ``(height, width)`` shape tuple and return normalised plain-int dims.

    Args:
        shape: The raw value supplied by the caller (e.g. from ``export(shape=...)`` or
            ``predict(shape=...)``).  Must be a two-element sequence of positive integers (or integer-compatible types
            accepted by :func:`operator.index`).
        block_size: Required divisor for both dimensions.  Equals ``patch_size * num_windows``.
        patch_size: Backbone patch size — used only in error messages.
        num_windows: Number of attention windows — used only in error messages.

    Returns:
        A ``(height, width)`` tuple of plain Python :class:`int` values.

    Raises:
        ValueError: If ``shape`` cannot be unpacked as a two-element sequence, if either
            dimension is a bool, float, or other non-integer type, if either dimension is not positive, or if either
            dimension is not divisible by ``block_size``.
    """
    try:
        raw_height, raw_width = cast("tuple[Any, Any]", shape)
    except (TypeError, ValueError):
        raise ValueError(f"shape must be a sequence of two positive integers (height, width), got {shape!r}.") from None
    for dim_name, dim in (("height", raw_height), ("width", raw_width)):
        if isinstance(dim, bool):
            raise ValueError(f"shape {dim_name} must be an integer, got {type(dim).__name__} (shape={shape!r}).")
        try:
            operator.index(dim)
        except TypeError:
            raise ValueError(
                f"shape {dim_name} must be an integer, got {type(dim).__name__} (shape={shape!r}).",
            ) from None
        if dim <= 0:
            raise ValueError(f"shape must contain positive integers for height and width, got {shape!r}.")
    # Normalise to plain Python ints; also accepts numpy.int64, torch scalars, etc.
    height, width = operator.index(raw_height), operator.index(raw_width)
    if height % block_size != 0 or width % block_size != 0:
        raise ValueError(
            f"shape must have both dimensions divisible by {block_size} "
            f"(patch_size={patch_size} * num_windows={num_windows}), got {shape!r}.",
        )
    return height, width


def _resolve_patch_size(patch_size: int | None, model_config: object, caller: str) -> int:
    """Resolve and validate the ``patch_size`` argument for :meth:`RFDETR.export` and :meth:`RFDETR.predict`.

    Args:
        patch_size: Value supplied by the caller, or ``None`` to read from ``model_config``.
        model_config: The model's configuration object.  Must expose ``patch_size`` as a
            positive integer attribute when ``patch_size`` is ``None`` or when a mismatch check is needed.
        caller: Name of the calling method (``"export"`` or ``"predict"``) — used in
            error messages to help the caller locate the problem.

    Returns:
        A validated, positive :class:`int` patch size.

    Raises:
        ValueError: If the resolved or provided ``patch_size`` is not a positive integer,
            or if a caller-provided value disagrees with ``model_config.patch_size``.
    """
    if patch_size is None:
        patch_size = getattr(model_config, "patch_size", 14)
    else:
        if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
        model_patch_size = getattr(model_config, "patch_size", None)
        if model_patch_size is not None and patch_size != model_patch_size:
            raise ValueError(
                f"{caller}(patch_size={patch_size}) does not match the instantiated model's "
                f"patch_size={model_patch_size}. Patch size is an architectural parameter; "
                f"omit patch_size to use the model's configured value.",
            )
    if isinstance(patch_size, bool) or not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError(f"patch_size must be a positive integer, got {patch_size!r}")
    return patch_size


def _move_model_context_to_device(model_ctx: Any) -> None:
    """Move model weights to the target device recorded in *model_ctx*.

    ``_build_model_context`` intentionally keeps the ``nn.Module`` on CPU so that ``RFDETR.__init__`` does not
    initialise CUDA (which would prevent DDP strategies from forking in notebook environments).  This helper performs
    the deferred ``.to(device)`` on first use.

    It is safe to call on duck-typed stand-ins (e.g. ``SimpleNamespace``); the function silently returns when the
    expected attributes are missing.
    """
    target = getattr(model_ctx, "device", None)
    inner = getattr(model_ctx, "model", None)
    if target is None or inner is None or not hasattr(inner, "parameters"):
        return
    if isinstance(target, str):
        target = torch.device(target)
    first_param = next(inner.parameters(), None)
    if first_param is not None and first_param.device != target:
        # ``predict()`` stacks ``@torch.inference_mode()`` on top of ``@_ensure_model_on_device``, so the deferred
        # move can run while inference mode is active.  Tensors materialised by ``.to()`` under inference mode become
        # *inference tensors*: they can never require gradients, so a later ``train()`` or auto-batch probe would
        # silently produce no gradients.  Disable inference mode for the move so deferred device placement is safe
        # regardless of decorator order.
        with torch.inference_mode(False):
            model_ctx.model = inner.to(target)


def _ensure_model_on_device(method: Callable[Concatenate[Any, _P], _R]) -> Callable[Concatenate[Any, _P], _R]:
    """Decorate RF-DETR instance methods that require lazy model device placement.

    The wrapped method receives the same arguments and return value as the original method. Before calling it, the
    decorator moves ``self.model.model`` to ``self.model.device`` if the model context is available and the weights are
    still on a different device. This keeps public inference methods clean while preserving deferred CUDA initialization
    during ``RFDETR.__init__``.
    """

    @wraps(method)
    def wrapper(self: Any, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        _move_model_context_to_device(getattr(self, "model", None))
        return method(self, *args, **kwargs)

    typed_wrapper = cast("Callable[Concatenate[Any, _P], _R]", wrapper)
    return typed_wrapper


def _prepare_run_config(
    detector: RFDETR, *, for_eval: bool = False, **kwargs: Any
) -> tuple[TrainConfig, str | None, list[int] | None]:
    """Absorb the special run kwargs and build a :class:`~rfdetr.config.TrainConfig`.

    Shared by :meth:`RFDETR.train` and :meth:`RFDETR.evaluate` so both accept exactly the same keyword arguments.
    Handles the kwargs that are not plain ``TrainConfig`` fields: ``device`` (mapped to PyTorch Lightning
    accelerator/devices), and
    ``resolution`` (a ``ModelConfig`` field applied to ``detector.model_config`` in place, with the positional-encoding
    size and cached inference context kept in sync). Remaining kwargs are forwarded to
    :meth:`RFDETR.get_train_config`; ``batch_size="auto"`` is resolved by auto-batch probing (skipped when
    ``for_eval=True`` — see below).
    ``model_config.model_name`` is set to the concrete subclass name.

    Defined at module scope (rather than as a method) so that :meth:`RFDETR.train` / :meth:`RFDETR.evaluate`, which are
    unit-tested by calling the unbound method with a ``MagicMock`` ``self``, still run the real preprocessing: a
    module-level function is not intercepted by the mock, so no per-test monkeypatching of this helper is needed.

    Args:
        detector: The :class:`RFDETR` instance whose ``model_config`` and model context the run targets.
        for_eval: When ``True`` (set by :meth:`RFDETR.evaluate`), ``batch_size="auto"`` is not resolved by probing —
            the probe forces train mode and runs a forward+backward pass to size for the *training* memory envelope
            (retained activations and gradients), which both wastes compute on a call that never backprops and
            under-sizes the eval batch relative to what a no-grad forward pass can actually fit. Falls back to
            :attr:`TrainConfig.batch_size`'s default instead.
        **kwargs: Keyword arguments accepted by :meth:`RFDETR.train` / :meth:`RFDETR.evaluate`.

    Returns:
        ``(train_config, accelerator, devices)`` where ``accelerator`` / ``devices`` are PTL trainer kwargs derived
        from the ``device`` kwarg (``None`` when ``device`` was not provided).

    Raises:
        ValueError: If ``resolution`` is not a positive integer or is not divisible by
            ``patch_size * num_windows`` for the model variant.
    """
    # Imported eagerly (not only under batch_size="auto") so a broken/absent training
    # submodule surfaces its original ModuleNotFoundError here rather than being masked.
    from rfdetr.training.auto_batch import resolve_auto_batch_config

    # Parse `device` kwarg and map it to PTL accelerator/devices.
    # Supports torch-style strings and torch.device (e.g. "cuda:1").
    _device = kwargs.pop("device", None)
    _accelerator, _devices = RFDETR._resolve_trainer_device_kwargs(_device)

    # Apply resolution override to model_config before building the train config.
    # resolution is a ModelConfig field, not a TrainConfig field, so we pop it
    # here to avoid it being silently ignored by TrainConfig.
    _resolution = kwargs.pop("resolution", None)
    if _resolution is not None:
        if isinstance(_resolution, bool):
            raise ValueError("resolution must be a positive integer")
        try:
            _resolution = operator.index(_resolution)
        except TypeError as error:
            raise ValueError("resolution must be a positive integer") from error
        if _resolution <= 0:
            raise ValueError("resolution must be a positive integer")
        block_size = detector.model_config.patch_size * detector.model_config.num_windows
        if _resolution % block_size != 0:
            raise ValueError(
                f"resolution={_resolution} is not divisible by "
                f"patch_size ({detector.model_config.patch_size}) * num_windows "
                f"({detector.model_config.num_windows}) = {block_size}. "
                f"Choose a resolution that is a multiple of {block_size}."
            )
        # Smart PE update: only recompute positional_encoding_size when the
        # current config derives it formulaically (PE == resolution // patch_size).
        # Configs with a pretrained-specific PE (e.g. RFDETRBase uses DINOv2's
        # PE=37 at 518 px, training at 560 px) must not have PE silently changed
        # — doing so causes shape mismatches when loading pretrained checkpoints.
        _current_pe = detector.model_config.positional_encoding_size
        _derived_pe = detector.model_config.resolution // detector.model_config.patch_size
        if _current_pe == _derived_pe:
            # Formula-derived: update PE proportionally to the new resolution.
            new_pe = _resolution // detector.model_config.patch_size
            detector.model_config.positional_encoding_size = new_pe
        else:
            # Pretrained-specific PE; leave it unchanged.
            new_pe = _current_pe
        detector.model_config.resolution = _resolution

        # Keep the cached inference/export context in sync with model_config so
        # predict()/export()/deployment all see the same resolution metadata.
        if hasattr(detector, "model") and detector.model is not None:
            if hasattr(detector.model, "resolution"):
                detector.model.resolution = _resolution
            model_args = getattr(detector.model, "args", None)
            if model_args is not None:
                if hasattr(model_args, "resolution"):
                    model_args.resolution = _resolution
                if hasattr(model_args, "positional_encoding_size"):
                    model_args.positional_encoding_size = new_pe
    config = detector.get_train_config(**kwargs)
    if config.batch_size == "auto" and for_eval:
        # The probe sizes for the *training* memory envelope (forward + retained activations +
        # backward), which evaluate() never needs and cannot benefit from — falls back to the
        # TrainConfig default micro-batch instead of running a wasted train-mode probe.
        default_batch_size = TrainConfig.model_fields["batch_size"].default
        logger.info(
            "[auto-batch] batch_size='auto' is a train-only feature; evaluate() uses the default "
            "micro-batch (%s) instead of probing.",
            default_batch_size,
        )
        config.batch_size = default_batch_size
    elif config.batch_size == "auto":
        # Auto-batch probing runs forward/backward on the actual model, which
        # must be on the target device (typically CUDA).  Lazy placement keeps
        # the model on CPU until first use — move it now.
        _move_model_context_to_device(detector.model)
        auto_batch = resolve_auto_batch_config(
            model_context=detector.model,
            model_config=detector.model_config,
            train_config=config,
        )
        config.batch_size = auto_batch.safe_micro_batch
        config.grad_accum_steps = auto_batch.recommended_grad_accum_steps
        logger.info(
            "[auto-batch] resolved train config: batch_size=%s grad_accum_steps=%s effective_batch_size=%s",
            config.batch_size,
            config.grad_accum_steps,
            auto_batch.effective_batch_size,
        )
    detector.model_config.model_name = type(detector).__name__
    return config, _accelerator, _devices


class RFDETR:
    """The base RF-DETR class implements the core methods for training RF-DETR models, running inference on the models,
    optimising models, and uploading trained models for deployment."""

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size: str | None = None
    _model_config_class: type[ModelConfig] = ModelConfig
    _train_config_class: type[TrainConfig] = TrainConfig

    def __init__(self, *, trust_checkpoint: bool = False, **kwargs: Any) -> None:
        """Initialize with ModelConfig fields as keyword arguments.

        Passes all remaining kwargs to the variant's ModelConfig. Unknown kwargs raise
        ``pydantic.ValidationError``. See the variant's config class for available
        parameters (e.g. ``RFDETRSmallConfig``).

        Args:
            trust_checkpoint: When ``True``, allow ``pretrain_weights`` to fall back to full
                pickle deserialization (``weights_only=False``) if safe loading fails. Set this
                only when ``pretrain_weights`` points to a checkpoint you explicitly trust (e.g.
                when called internally by :meth:`from_checkpoint`); left ``False`` by default for
                the ordinary construction path, which only ever downloads official Roboflow-hosted
                weights.
            **kwargs: ModelConfig field values (e.g. ``resolution``, ``num_classes``,
                ``pretrain_weights``, ``gradient_checkpointing``).
        """
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config, trust_checkpoint=trust_checkpoint)
        self.callbacks: dict[str, list[Callable[..., Any]]] = defaultdict(list)

        self.means = list(self.means)
        self.stds = list(self.stds)

        # repeat means and stds for non-rgb images
        if self.model_config.num_channels != 3:
            from itertools import cycle

            self.means = [val for _, val in zip(range(self.model_config.num_channels), cycle(self.means))]
            self.stds = [val for _, val in zip(range(self.model_config.num_channels), cycle(self.stds))]

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size: int | None = None
        self._optimized_resolution: int | None = None
        self._optimized_dtype: torch.dtype | None = None
        self._optimized_inplace = False
        self._has_been_trained = False

    def maybe_download_pretrain_weights(self) -> None:
        """Download pre-trained weights if they are not already downloaded.

        Bare filenames (no directory component, e.g. ``rf-detr-base.pth``) are resolved to the model cache directory —
        set the ``RF_HOME`` environment variable to override the location (default: ``~/.roboflow/models``). Resolution
        happens in ``ModelConfig.expand_path`` for explicitly-provided values, and here as a fallback for field defaults
        (which Pydantic does not validate by default).

        Paths that already contain a directory component are used as-is; the parent directory is created if it does not
        yet exist.
        """
        if self.model_config.pretrain_weights is None:
            return
        pretrain_weights = str(self.model_config.pretrain_weights)
        if not os.path.dirname(pretrain_weights):
            # Field default was not processed by expand_path — resolve to cache dir.
            cache_dir = get_model_cache_dir()
            os.makedirs(cache_dir, exist_ok=True)
            pretrain_weights = os.path.join(cache_dir, pretrain_weights)
        else:
            os.makedirs(os.path.dirname(pretrain_weights), exist_ok=True)
        self.model_config.pretrain_weights = pretrain_weights
        download_pretrain_weights(pretrain_weights)

    def get_model_config(self, **kwargs: Any) -> ModelConfig:
        """Retrieve the configuration parameters used by the model."""
        return self._model_config_class(**kwargs)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], *, trust_checkpoint: bool = False, **kwargs: Any) -> RFDETR:
        """Load an RF-DETR model from a training checkpoint, automatically inferring the model class.

        The correct subclass is resolved in order of preference:

        1. ``model_name`` key in the checkpoint (written by the PTL training
           stack since v1.7.0).
        2. ``pretrain_weights`` field in the checkpoint's ``args`` entry
           (legacy fallback for older checkpoints).
        3. The **filename** of *path* itself, used as a last resort when
           ``pretrain_weights`` is absent or an unset-like sentinel value
           (empty string, ``"none"``, or ``"null"``).  Starter weights
           published by Roboflow store ``pretrain_weights="none"`` in their
           ``args``; passing the canonical filename (e.g.
           ``rf-detr-small.pth``) lets ``from_checkpoint`` infer the class
           automatically.

        Both legacy ``argparse.Namespace`` checkpoints (produced by ``engine.py``) and dict-style checkpoints (produced
        by the PTL training stack) are supported.

        Args:
            path: Path to a checkpoint file (e.g. ``checkpoint_best_total.pth``).
            trust_checkpoint: When ``True``, fall back to ``weights_only=False``
                (full pickle) if safe deserialization fails.  Only set this for
                checkpoints from fully trusted sources; the default ``False``
                keeps the safe loading path and raises if it cannot succeed.
                Applies to both the initial checkpoint read here and the
                constructor's own reload of the same file via
                :func:`~rfdetr.models.weights.load_pretrain_weights`.
            **kwargs: Additional keyword arguments forwarded to the model
                constructor (e.g. ``accept_platform_model_license=True`` for XLarge / 2XLarge models).

                ``num_classes`` is resolved in this priority order:

                1. Explicit caller kwarg — always wins.
                2. Weight inference from ``class_embed.weight`` shape in the checkpoint
                   (``shape[0] - 1``, since the head includes a background class). This
                   overrides a stale ``model_config`` value written before fine-tuning
                   changed the class count.
                3. ``saved_model_config["num_classes"]`` from the checkpoint's
                   ``model_config`` entry — may be stale for older checkpoints.
                4. Legacy ``args["num_classes"]`` dict entry.
                5. Constructor default.

                In cases 2–5 the field is not recorded as a user-set override, so
                :meth:`train` can still adapt the detection head to the training
                dataset's class count.  Pass an explicit ``num_classes=N`` to pin
                the head and prevent adaptation.

        Returns:
            An instance of the appropriate :class:`RFDETR` subclass loaded from the checkpoint.

        Warning:
            By default this method attempts safe deserialization
            (``weights_only=True``).  Pass ``trust_checkpoint=True`` only for
            checkpoints from fully trusted sources, as it enables full pickle
            deserialization which can execute arbitrary code.

        Raises:
            FileNotFoundError: If *path* does not exist.
            OSError: If *path* exists but cannot be read.
            KeyError: If the checkpoint does not contain an ``"args"`` key.
            ValueError: If the model class cannot be inferred from ``model_name``,
                ``pretrain_weights``, or the checkpoint filename.

        Examples:
            >>> model = RFDETR.from_checkpoint("checkpoint_best_total.pth")  # doctest: +SKIP
            >>> model = RFDETRSmall.from_checkpoint("checkpoint_best_total.pth")  # doctest: +SKIP
        """
        # Local import breaks the variants → detr import cycle.
        import rfdetr.variants as rfdetr_variants

        _plus_available = False
        _plus_symbols: dict[str, type[RFDETR]] = {}
        _plus_entries: list[tuple[str, type[RFDETR]]] = []
        from rfdetr.platform import _IS_RFDETR_PLUS_AVAILABLE

        if _IS_RFDETR_PLUS_AVAILABLE:
            try:
                import rfdetr.platform.models as platform_models

                for class_symbol in _CHECKPOINT_PLUS_MODEL_NAME_CLASS_SYMBOLS:
                    plus_obj = getattr(platform_models, class_symbol)
                    _plus_symbols[class_symbol] = plus_obj
                _plus_entries = [
                    (name, _plus_symbols[class_symbol]) for name, class_symbol in _CHECKPOINT_PLUS_MODEL_MAP_ENTRIES
                ]
                _plus_available = True
            except ModuleNotFoundError as ex:
                if ex.name not in {"rfdetr_plus", "rfdetr_plus.models"}:
                    raise

        # Use the safe-load helper which tries weights_only=True first (with
        # legacy argparse.Namespace safe globals), falling back to full pickle
        # only when the caller explicitly passes trust_checkpoint=True.
        from rfdetr.utilities.io import _safe_torch_load

        ckpt: dict[str, Any] = _safe_torch_load(str(path), trust=trust_checkpoint)
        args = ckpt["args"]

        _variant_name_to_class: dict[str, type[RFDETR]] = {
            getattr(variant_obj, "__name__", symbol): variant_obj
            for symbol in dir(rfdetr_variants)
            if symbol.startswith("RFDETR")
            for variant_obj in [getattr(rfdetr_variants, symbol)]
        }
        _variant_symbols: dict[str, type[RFDETR]] = {
            class_symbol: _variant_name_to_class[class_symbol] for class_symbol in _CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS
        }
        # Build in three explicit segments: seg-* entries, then plus-model entries
        # (xlarge/2xlarge), then base entries — order determines lookup priority.
        _seg_map: list[tuple[str, type[RFDETR]]] = [
            (name, _variant_symbols[class_symbol])
            for name, class_symbol in _CHECKPOINT_MODEL_MAP_ENTRIES
            if name.startswith("seg-")
        ]
        _keypoint_map: list[tuple[str, type[RFDETR]]] = [
            (name, _variant_symbols[class_symbol])
            for name, class_symbol in _CHECKPOINT_MODEL_MAP_ENTRIES
            if "keypoint" in name
        ]
        _base_map: list[tuple[str, type[RFDETR]]] = [
            (name, _variant_symbols[class_symbol])
            for name, class_symbol in _CHECKPOINT_MODEL_MAP_ENTRIES
            if not name.startswith("seg-") and "keypoint" not in name
        ]
        _model_map: list[tuple[str, type[RFDETR]]] = _seg_map + _keypoint_map + _plus_entries + _base_map

        # New checkpoints store model_name directly — use it when available.
        _name_map: dict[str, type[RFDETR]] = dict(_variant_symbols)
        # Plus-model classes are resolved only when rfdetr_plus is installed.
        if _plus_available:
            _name_map.update(_plus_symbols)
        # RFDETRLargeDeprecated is excluded from _CHECKPOINT_MODEL_NAME_CLASS_SYMBOLS
        # (so forward name-map lookups always reach RFDETRLarge), but it must be
        # in _name_map so that checkpoints carrying model_name="RFDETRLargeDeprecated"
        # are reloaded with the correct class instead of falling through to the
        # substring matcher (which would wrongly pick RFDETRLarge and fail with a
        # pydantic literal_error on encoder / projector_scale fields).
        # Note: RFDETRLargeDeprecated is a _DeprecatedProxy (pyDeprecate) and has no
        # __name__; look it up by the string key it was registered under, then inject
        # into _name_map so checkpoint matching resolves directly without the substring
        # fallback.  Tests assert this mapping exists (see tests/inference/test_from_checkpoint.py).
        _large_deprecated_cls = _variant_name_to_class.get("RFDETRLargeDeprecated")
        if _large_deprecated_cls is not None:
            _name_map["RFDETRLargeDeprecated"] = _large_deprecated_cls
        saved_model_name = ckpt.get("model_name")
        model_cls: type[RFDETR] | None = None
        if isinstance(saved_model_name, str):
            normalized_name = saved_model_name.strip()
            if normalized_name:
                model_cls = _name_map.get(normalized_name)
        else:
            normalized_name = ""

        # Fall back to pretrain_weights (legacy) or, when unset-like, the checkpoint filename.
        if isinstance(args, dict):
            weights_name = str(args.get("pretrain_weights", "")).strip().lower()
        else:
            weights_name = str(getattr(args, "pretrain_weights", "")).strip().lower()
        # The sentinel set {"", "none", "null"} covers unset-like checkpoint values:
        #   ""     — pretrain_weights key absent entirely
        #   "none" — checkpoint value was None or the literal string "none";
        #            after str(...).strip().lower() both normalize to the same sentinel.
        #            This is NOT an intentional "no pretraining" flag (see
        #            test_pretrain_weights_none_warns, which operates at the config
        #            level, not the checkpoint level)
        #   "null" — checkpoint stored the literal string "null" (for example from a
        #            YAML-originated value), which is also treated as unset-like here
        _filename_fallback = False
        if weights_name in {"", "none", "null"}:
            weights_name = os.path.basename(os.fspath(path)).lower()
            _filename_fallback = True

        if model_cls is None:
            # Guard: plus-only checkpoints should raise an actionable install error
            # when rfdetr_plus is missing, regardless of whether class inference
            # relies on model_name (new format) or pretrain_weights (legacy format).
            plus_by_model_name = normalized_name in _CHECKPOINT_PLUS_MODEL_NAME_CLASS_SYMBOLS
            plus_by_weights_name = (
                "xlarge" in weights_name and "seg-" not in weights_name and "keypoint-preview" not in weights_name
            )
            if not _plus_available and (plus_by_model_name or plus_by_weights_name):
                from rfdetr.platform import _INSTALL_MSG

                raise ImportError(
                    f"Checkpoint model_name={saved_model_name!r}, pretrain_weights={weights_name!r} requires the "
                    f"rfdetr_plus package. " + _INSTALL_MSG.format(name="platform model downloads")
                )

            for name, klass in _model_map:
                if name in weights_name:
                    model_cls = klass
                    break

            if _filename_fallback and model_cls is not None:
                logger.info(
                    "pretrain_weights unset in checkpoint %r; inferred model class %s from filename %r",
                    path,
                    getattr(model_cls, "__name__", repr(model_cls)),
                    weights_name,
                )

        if model_cls is None:
            raise ValueError(
                f"Could not infer model class from checkpoint at {path!r} "
                f"(model_name={saved_model_name!r}, pretrain_weights={weights_name!r}). "
                f"Please instantiate the model class directly."
            )

        if isinstance(args, dict):
            num_classes: int | None = args.get("num_classes")
        else:
            num_classes = getattr(args, "num_classes", None)

        constructor_kwargs: dict[str, Any] = {}
        checkpoint_config_keys: set[str] = set()  # keys injected from checkpoint, not from caller

        # Resolve model config field set once — used for both saved_model_config parsing and
        # weight-based schema inference guards (BaseConfig has extra="forbid"; unknown fields raise).
        _model_config_class = getattr(model_cls, "_model_config_class", None)
        _mc_fields: dict[str, Any] = {}
        _mc_model_fields = getattr(_model_config_class, "model_fields", None)
        if isinstance(_mc_model_fields, dict):
            _mc_fields = _mc_model_fields
        else:
            _mc_legacy = getattr(_model_config_class, "__fields__", None)
            if isinstance(_mc_legacy, dict):
                _mc_fields = _mc_legacy

        saved_model_config = ckpt.get("model_config")
        if isinstance(saved_model_config, dict):
            for key, value in saved_model_config.items():
                if key == "pretrain_weights":
                    continue
                if not _mc_fields or key in _mc_fields:
                    constructor_kwargs[key] = value
                    checkpoint_config_keys.add(key)

        if num_classes is not None and "num_classes" not in kwargs:
            constructor_kwargs["num_classes"] = num_classes
            checkpoint_config_keys.add("num_classes")

        # Infer schema-critical fields from checkpoint weights — these are authoritative when
        # ``model_config`` is absent or stale (saved before ``model_config`` persistence was added,
        # or saved with default values before fine-tuning changed the trained schema).
        # User-supplied ``kwargs`` take precedence and are applied in the ``update`` call below.
        _ckpt_weights: dict[str, Any] = ckpt.get("model") or {}
        if not _ckpt_weights and "state_dict" in ckpt:
            _pfx = "model."
            _ckpt_weights = {}
            for k, v in ckpt["state_dict"].items():
                if k.startswith(_pfx):
                    key = k[len(_pfx) :]
                    # Strip optional torch.compile() wrapper prefix
                    if key.startswith("_orig_mod."):
                        key = key[len("_orig_mod.") :]
                    _ckpt_weights[key] = v
        if _ckpt_weights:
            # num_keypoints_per_class — inferred from _kp_active_mask (shape [num_classes, max_kp]).
            # Reflects what the model actually learned; saved model_config may carry the COCO default
            # [0, 17] even after fine-tuning on a different keypoint schema.
            if "num_keypoints_per_class" not in kwargs and (not _mc_fields or "num_keypoints_per_class" in _mc_fields):
                _kp_mask = _ckpt_weights.get("_kp_active_mask")
                if isinstance(_kp_mask, torch.Tensor) and _kp_mask.ndim == 2:
                    _inferred_kp = [int(n) for n in _kp_mask.sum(dim=1).tolist()]
                    _current_kp = constructor_kwargs.get("num_keypoints_per_class")
                    if _inferred_kp != _current_kp:
                        logger.debug(
                            "from_checkpoint: overriding num_keypoints_per_class %s → %s "
                            "(inferred from _kp_active_mask; saved model_config may be stale).",
                            _current_kp,
                            _inferred_kp,
                        )
                    constructor_kwargs["num_keypoints_per_class"] = _inferred_kp
                    checkpoint_config_keys.add("num_keypoints_per_class")
            # num_classes — inferred from class_embed.weight shape.
            # The head shape is ground truth for what num_classes the checkpoint uses.
            if "num_classes" not in kwargs:
                _ce_weight = _ckpt_weights.get("class_embed.weight")
                if isinstance(_ce_weight, torch.Tensor) and _ce_weight.ndim == 2:
                    _inferred_nc = _ce_weight.shape[0] - 1  # shape[0] = num_classes + 1 (background)
                    _current_nc = constructor_kwargs.get("num_classes")
                    if _inferred_nc != _current_nc:
                        logger.debug(
                            "from_checkpoint: overriding num_classes %s → %s "
                            "(inferred from class_embed.weight; saved model_config may be stale).",
                            _current_nc,
                            _inferred_nc,
                        )
                    constructor_kwargs["num_classes"] = _inferred_nc
                    checkpoint_config_keys.add("num_classes")

        constructor_kwargs.update(kwargs)
        # pretrain_weights is placed after **kwargs so it always wins even if
        # a caller accidentally passes pretrain_weights inside kwargs.
        constructor_kwargs["pretrain_weights"] = str(path)
        # Model construction reloads this same file via load_pretrain_weights(); without this,
        # trust_checkpoint=True would bypass the safe-load only for the metadata read above and
        # then fail identically when the constructor re-reads pretrain_weights.
        constructor_kwargs["trust_checkpoint"] = trust_checkpoint

        # Fields injected from the checkpoint but not supplied by the caller must not be
        # treated as explicit user overrides in Pydantic's model_fields_set.  Downstream
        # alignment guards (e.g. _align_num_classes_from_dataset,
        # _align_keypoint_schema_from_dataset, load_pretrain_weights) all read
        # model_fields_set to decide whether to adapt model internals to the training
        # dataset — leaving checkpoint-derived fields marked as user-set breaks them.
        checkpoint_derived_keys = checkpoint_config_keys - set(kwargs)

        model = model_cls(**constructor_kwargs)
        # The instance now carries checkpoint (trained) weights; flag it so a later
        # train() call warns that it will restart from pretrain_weights, not continue.
        model._has_been_trained = True

        if checkpoint_derived_keys:
            loaded_config = getattr(model, "model_config", None)
            # model_fields_set is the public API and returns the live backing set
            # in Pydantic v2; fall back to the private attribute only if that changes.
            fields_set = getattr(loaded_config, "model_fields_set", None)
            if fields_set is None:
                fields_set = getattr(loaded_config, "__pydantic_fields_set__", None)
            if fields_set is not None:
                fields_set.difference_update(checkpoint_derived_keys)
            # Verify num_classes specifically — if Pydantic ever returns a snapshot instead
            # of the live backing set, this assertion will catch the silent regression before
            # it causes a training-time head-adaptation failure.
            if "num_classes" in checkpoint_derived_keys:
                assert "num_classes" not in getattr(loaded_config, "model_fields_set", set()), (
                    "num_classes still in model_fields_set after checkpoint load; "
                    "Pydantic may return a snapshot rather than the live backing set — "
                    "switch to model_construct(_fields_set=...) for Pydantic v3 compatibility."
                )

        return model

    @staticmethod
    def _resolve_trainer_device_kwargs(device: Any) -> tuple[str | None, list[int] | None]:
        """Map a torch-style device specifier to PTL ``accelerator``/``devices`` kwargs.

        Args:
            device: A device specifier accepted by ``torch.device``.

        Returns:
            ``(accelerator, devices)`` where ``devices`` is ``None`` unless an explicit device index is provided (for
            example ``cuda:1``). ``device.type == "xla"`` maps to ``accelerator="tpu"`` -- PTL's accelerator
            registry has no ``"xla"`` string; ``"tpu"`` is its canonical name for the XLA backend.

        Raises:
            ValueError: If ``device`` is not a valid torch device specifier.
        """
        if device is None:
            return None, None
        try:
            resolved_device = torch.device(device)
        except (TypeError, ValueError, RuntimeError) as exc:
            raise ValueError(
                f"Invalid device specifier for train(): {device!r}. "
                "Expected values like 'cpu', 'cuda', 'cuda:0', or torch.device(...).",
            ) from exc

        if resolved_device.type == "cpu":
            return "cpu", None
        if resolved_device.type == "cuda":
            return "gpu", [resolved_device.index] if resolved_device.index is not None else None
        if resolved_device.type == "mps":
            return "mps", [resolved_device.index] if resolved_device.index is not None else None
        if resolved_device.type == "xla":
            # PTL's accelerator registry has no "xla" string -- "tpu" is its canonical name for
            # the XLA backend (torch.device("xla") is valid and .type is always "xla", even on
            # TPU; torch.device("tpu") itself raises RuntimeError). Bridge explicitly instead of
            # falling through to the auto-detection warning below.
            return "tpu", [resolved_device.index] if resolved_device.index is not None else None

        warnings.warn(
            f"Device type {resolved_device.type!r} is not explicitly mapped to a PyTorch Lightning "
            "accelerator; falling back to PTL auto-detection. Training may use an unexpected device.",
            UserWarning,
            stacklevel=2,
        )
        return None, None

    def train(self, **kwargs: Any) -> None:
        """Train an RF-DETR model via the PyTorch Lightning stack.

        All keyword arguments are forwarded to :meth:`get_train_config` to build a :class:`~rfdetr.config.TrainConfig`.
        Several kwargs are absorbed and handled specially so that existing call-sites do not break:

        * ``resolution`` — updates the model's input resolution by mutating
          :attr:`model_config.resolution` in place before the train config is built. This change persists on
          :attr:`model_config` after :meth:`train` returns. The value must be a positive integer divisible by
          ``patch_size * num_windows`` for the model variant; a :class:`ValueError` is raised otherwise.
          :attr:`model_config.positional_encoding_size` is also updated when the config derives it formulaically (``PE
          == resolution // patch_size``); configs with a pretrained-specific PE value (e.g. ``RFDETRBase`` uses DINOv2's
          PE=37 at 560 px) are left unchanged to preserve checkpoint compatibility.
        * ``device`` — normalized via :class:`torch.device` and mapped to PyTorch
          Lightning trainer arguments. ``"cpu"`` becomes ``accelerator="cpu"``; ``"cuda"`` and ``"cuda:N"`` become
          ``accelerator="gpu"`` and optionally ``devices=[N]``; ``"mps"`` becomes ``accelerator="mps"``; ``"xla"``
          becomes ``accelerator="tpu"`` (PTL's canonical name for the XLA backend). Other valid torch device types
          fall back to PTL auto-detection and emit a :class:`UserWarning`.
        * ``notes`` — optional user-defined metadata (string, dict, list, or
          any JSON-serialisable value) stored under the ``"notes"`` key in every ``.pth`` checkpoint produced during
          training.  The value is also available inside ``args["notes"]`` for full provenance.  Pass the same value to
          :meth:`export` to embed it in the ONNX file as well.

        After training completes the underlying ``nn.Module`` is synced back onto ``self.model.model`` so that
        :meth:`predict` and :meth:`export` continue to work without reloading the checkpoint.

        Raises:
            ImportError: If training dependencies are not installed. Install with
                ``pip install "rfdetr[train,loggers]"``.
            ValueError: If ``resolution`` is not a positive integer or is not
                divisible by ``patch_size * num_windows`` for the model variant.
        """
        # The training stack lives in the `rfdetr[train]` extras group — a missing
        # `pytorch_lightning` (or any other training-extras package) causes the import to fail,
        # and the remediation is `pip install "rfdetr[train,loggers]"`.
        try:
            from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
        except ModuleNotFoundError as exc:
            # Preserve internal import errors so packaging/regression issues in
            # rfdetr.* are not misreported as missing optional extras.
            if exc.name and exc.name.startswith("rfdetr."):
                raise
            raise ImportError(
                "RF-DETR training dependencies are missing. "
                'Install them with `pip install "rfdetr[train,loggers]"` and try again.',
            ) from exc

        if getattr(self, "_has_been_trained", False):
            warnings.warn(
                "Calling train() on a model that has already been trained or loaded from a checkpoint. "
                "The new training run will start from the original pretrained weights (pretrain_weights), "
                "NOT from the in-memory trained state. To continue training, pass resume=<checkpoint_path>.",
                UserWarning,
                stacklevel=2,
            )

        # Absorb the special train/evaluate kwargs (device, resolution, deprecated knobs),
        # build the TrainConfig, and resolve any auto batch size.  Shared with evaluate()
        # so both accept exactly the same keyword arguments.
        config, _accelerator, _devices = _prepare_run_config(self, **kwargs)

        # Auto-detect num_classes from the training dataset and align model_config.
        # This must run before RFDETRModelModule is constructed so that weight loading
        # inside the module uses the correct (dataset-derived) class count.
        dataset_dir = getattr(config, "dataset_dir", None)
        if dataset_dir:
            self._align_keypoint_schema_from_dataset(config)
            self._align_num_classes_from_dataset(dataset_dir)

        module = RFDETRModelModule(self.model_config, config)
        datamodule = RFDETRDataModule(self.model_config, config)

        # Guard with LOCAL_RANK env var rather than is_main_process() because torch.distributed
        # is not yet initialized here (it is set up inside trainer.fit()).  In Lightning DDP
        # subprocesses, LOCAL_RANK is set by the launcher before the subprocess calls train(),
        # so this correctly identifies rank 0 even before dist.init_process_group() runs.
        if config.save_dataset_grids and os.environ.get("LOCAL_RANK", "0") == "0":
            try:
                from rfdetr.datasets.save_grids import DatasetGridSaver

                datamodule.setup("fit")
                grids_output_dir = Path(config.output_dir) / "dataset_grids"
                DatasetGridSaver(datamodule.train_dataloader(), grids_output_dir, dataset_type="train").save_grid()
                DatasetGridSaver(datamodule.val_dataloader(), grids_output_dir, dataset_type="val").save_grid()
            except Exception:
                logger.warning(
                    "Failed to save dataset grids; training will continue without them.",
                    exc_info=True,
                )

        if config.resume:
            # BestModelCallback's four lightweight checkpoint files (unlike the trainer's own
            # `last.ckpt` / `checkpoint_<epoch>.ckpt`, which retain full PTL state) intentionally
            # omit optimizer/LR-scheduler state to stay small — see
            # BestModelCallback._build_checkpoint_payload. They retain model weights, epoch
            # metadata, and per-callback state only when the resumed callback configuration
            # matches the saved keys. Checkpoints written before callback-state persistence have
            # no such state. Best-score tracking additionally requires exactly the original
            # output_dir because of PTL's ModelCheckpoint.load_state_dict() dirpath gate.
            # Flag the optimizer/scheduler gap explicitly instead of letting it pass silently.
            _light_checkpoint_names = frozenset(
                {"checkpoint_best_regular.pth", "checkpoint_best_ema.pth", "checkpoint_best_total.pth", "last_ema.pth"}
            )
            if Path(config.resume).name in _light_checkpoint_names:
                from rfdetr.utilities.io import _safe_torch_load

                # checkpoint.get("callbacks") is None-checked by PTL's own
                # _call_callbacks_load_state_dict(), which no-ops (skipping every callback's
                # restoration) when the key is absent or empty. Checkpoints written before
                # BestModelCallback._build_checkpoint_payload started persisting per-callback state
                # — or from a run where every registered callback happened to have empty state —
                # are exactly this case, so peek at the file rather than let the warning below
                # overclaim a restoration that silently does not happen.
                _resume_ckpt = _safe_torch_load(config.resume, trust=True)
                _has_callback_state = bool(_resume_ckpt.get("callbacks"))
                del _resume_ckpt
                _resume_dir = Path(config.resume).resolve().parent
                _configured_output_dir = Path(config.output_dir).resolve()
                _best_score_restores = _resume_dir == _configured_output_dir

                if _has_callback_state:
                    logger.warning(
                        "resume=%r points at one of BestModelCallback's lightweight checkpoints, "
                        "which intentionally omit optimizer/LR-scheduler state to stay small. "
                        "Model weights and epoch count will resume. Callback state can restore only "
                        "for matching configured callbacks; the optimizer and LR scheduler restart cold. "
                        "To resume with optimizer/scheduler state too, pass the trainer's full "
                        "checkpoint instead (e.g. %s/last.ckpt or %s/checkpoint_<epoch>.ckpt).",
                        config.resume,
                        config.output_dir,
                        config.output_dir,
                    )
                else:
                    logger.warning(
                        "resume=%r points at one of BestModelCallback's lightweight checkpoints, "
                        "which intentionally omit optimizer/LR-scheduler state to stay small. "
                        "Model weights and epoch count will resume, but this particular file has no "
                        "saved callback state (it predates callback-state persistence, or every "
                        "registered callback had nothing to save), so best-score tracking, EMA, and "
                        "early-stopping state all restart cold this run too — not just the "
                        "optimizer and LR scheduler. To resume with full state, pass the trainer's "
                        "full checkpoint instead (e.g. %s/last.ckpt or %s/checkpoint_<epoch>.ckpt).",
                        config.resume,
                        config.output_dir,
                        config.output_dir,
                    )

                # BestModelCallback always writes these four files directly under its own
                # `dirpath` (== output_dir at save time; see BestModelCallback.__init__ and
                # _build_checkpoint_payload). PTL's ModelCheckpoint.load_state_dict() only
                # restores best_model_score/best_k_models/kth_value/last_model_path when the
                # resumed dirpath matches the checkpoint's saved dirpath exactly (model_checkpoint.py,
                # installed pytorch-lightning) — with a different output_dir this run only recovers
                # best_model_path, so the first metric logged after resume looks like an automatic
                # improvement over an empty best_model_score. Only worth flagging when there was
                # callback state to lose in the first place.
                if _has_callback_state and not _best_score_restores:
                    logger.warning(
                        "resume=%r was written under %s but output_dir=%r points elsewhere. "
                        "PyTorch Lightning only restores best_model_score/best_k_models when "
                        "output_dir matches the checkpoint's original directory exactly — with "
                        "this output_dir, best-score tracking (BestModelCallback's high-water "
                        "mark) restarts fresh in the new directory instead of resuming. Set "
                        "output_dir=%r to keep it.",
                        config.resume,
                        _resume_dir,
                        config.output_dir,
                        str(_resume_dir),
                    )

        trainer_kwargs: dict[str, Any] = {"accelerator": _accelerator}
        if _devices is not None:
            trainer_kwargs["devices"] = _devices
        trainer = build_trainer(config, self.model_config, **trainer_kwargs)
        trainer.fit(module, datamodule, ckpt_path=config.resume or None)

        # Sync the trained weights back so predict() / export() see the updated model.
        self.model.model = module.model
        # Rebuild model.args from the real training config (#1199): it was previously left as the
        # construction-time snapshot built from a dummy TrainConfig, so overrides like lr/lr_encoder
        # never appeared in model.model.__dict__['args'] even though the optimizer used them correctly.
        self.model.args = _namespace_from_configs(self.model_config, config)
        # Mark this instance as trained so a subsequent train() call warns that it will
        # restart from pretrain_weights rather than continue from the in-memory state.
        self._has_been_trained = True
        # Invalidate any compiled inference snapshot: it was built from the pre-training
        # weights and must not survive the model reassignment above.
        self.remove_optimized_model()
        # Sync class names: prefer explicit config.class_names, otherwise fall back to dataset (#509).
        config_class_names = getattr(config, "class_names", None)
        if config_class_names is not None:
            self.model.class_names = config_class_names
        else:
            dataset_class_names = getattr(datamodule, "class_names", None)
            if dataset_class_names is not None:
                self.model.class_names = dataset_class_names

        # Save complete training configuration to disk for reproducibility.
        # Guard to main process only to avoid races in distributed/multi-GPU training.
        if is_main_process():
            complete_config = {
                "train_config": config.model_dump(),
                "model_config": self.model_config.model_dump(),
                "model_config_type": self.model_config.__class__.__name__,
                "class_names": self.model.class_names,
                "num_classes": len(self.model.class_names) if self.model.class_names else 0,
            }
            try:
                os.makedirs(config.output_dir, exist_ok=True)
                with open(os.path.join(config.output_dir, "training_config.json"), "w") as f:
                    json.dump(complete_config, f, indent=2, default=str)
            except OSError as exc:
                logger.warning("Could not save training_config.json to %s: %s", config.output_dir, exc)

    def evaluate(self, *, split: Literal["test", "val"] = "test", **kwargs: Any) -> dict[str, float]:
        """Evaluate the current model on a dataset split and return COCO metrics.

        Runs a single evaluation pass over the requested split via the PyTorch Lightning stack and returns the COCO
        metrics (mAP, mAR, and the macro-F1 sweep) computed by
        :class:`~rfdetr.training.callbacks.coco_eval.COCOEvalCallback`. The same metrics are also printed to the
        terminal. This works both directly after :meth:`train` and on a model loaded via :meth:`from_checkpoint` — the
        weights already held in memory are evaluated; no checkpoint file is re-loaded.

        Apart from ``split``, this method accepts exactly the same keyword arguments as :meth:`train` (``dataset_dir``,
        ``device``, ``resolution``, ``batch_size``, ``output_dir``, ``num_workers``, ...); they are handled identically
        via the shared :func:`_prepare_run_config`. This parity is for convenience — the same kwargs dict used for
        :meth:`train` can be reused here — not a guarantee every field has an effect. Training-only fields (``epochs``,
        ``lr``, ``weight_decay``, ``ema``, ``early_stopping``, ``run``, ``project``, ``checkpoint_interval``,
        ``tensorboard``/``wandb``/``mlflow``/``clearml``, and similar) are silently accepted and ignored: ``evaluate()``
        runs through an eval-only trainer (``include_training_callbacks=False``) that never builds EMA, drop-path,
        checkpointing, early-stopping, or logger callbacks, so those fields have nothing to attach to.

        Unlike :meth:`train`, this method never adapts the detection head to the dataset: the model is evaluated exactly
        as configured. If the dataset's class count differs from the model's ``num_classes`` a :class:`UserWarning` is
        emitted and evaluation proceeds with the model's head unchanged.

        Unlike :meth:`train`, a ``resolution`` override does **not** persist: :attr:`model_config` (and any cached
        ``model.resolution`` / ``model.args`` inference context) is restored to its pre-call values once the
        eval-only config copy has captured the override, so a later :meth:`predict` / :meth:`export` / :meth:`train`
        call is unaffected by an ``evaluate(resolution=...)`` call.

        Args:
            split: Which split to evaluate. ``"test"`` evaluates the ``test/`` folder (Roboflow datasets; falls back to
                the validation split otherwise) via ``trainer.test``; ``"val"`` evaluates the ``valid/`` folder via
                ``trainer.validate``.
            **kwargs: The same keyword arguments accepted by :meth:`train` — ``dataset_dir`` is required (here or
                already on the config), and the rest are forwarded to :func:`_prepare_run_config` /
                :meth:`get_train_config`.

        Returns:
            Mapping of metric name to value for the evaluated split, e.g. ``{"test/mAP_50_95": ..., "test/mAP_50": ...,
            "test/F1": ..., "test/AP/<class>": ...}``. Empty when the trainer returns no metrics.

        Raises:
            ImportError: If training dependencies are not installed. Install with
                ``pip install "rfdetr[train,loggers]"``.
            ValueError: If ``split`` is not ``"test"`` or ``"val"``.
        """
        from rfdetr.models.weights import interpolate_position_embeddings

        # Training extras (pytorch_lightning et al.) are optional; mirror train()'s import guard.
        try:
            from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
        except ModuleNotFoundError as exc:
            if exc.name and exc.name.startswith("rfdetr."):
                raise
            raise ImportError(
                "RF-DETR training dependencies are missing. "
                'Install them with `pip install "rfdetr[train,loggers]"` and try again.',
            ) from exc

        if split not in ("test", "val"):
            raise ValueError(f"split must be 'test' or 'val', got {split!r}.")

        # Same kwarg handling as train() (device, resolution, deprecated knobs, auto-batch).  A `resolution`
        # override mutates `model_config`/`model.args` in place inside `_prepare_run_config` — intentional and
        # persistent for train(), but evaluate() is documented as inspection-only, so the mutation is snapshotted
        # here and restored once the eval-only config copy below has captured the overridden values it needs.
        _orig_resolution = self.model_config.resolution
        _orig_pe = self.model_config.positional_encoding_size
        _live_model = getattr(self, "model", None)
        _live_args = getattr(_live_model, "args", None) if _live_model is not None else None
        _orig_model_resolution = getattr(_live_model, "resolution", None) if _live_model is not None else None
        _orig_args_resolution = getattr(_live_args, "resolution", None) if _live_args is not None else None
        _orig_args_pe = getattr(_live_args, "positional_encoding_size", None) if _live_args is not None else None
        try:
            config, _accelerator, _devices = _prepare_run_config(self, for_eval=True, **kwargs)

            # Build the module without re-loading pretrain weights, then transplant the
            # already-loaded in-memory weights into it (the reverse of train()'s final
            # `self.model.model = module.model`).  This evaluates the current weights and
            # never passes ``ckpt_path``, sidestepping PTL's loop-state restore on a bare .pth.
            eval_model_config = self.model_config.model_copy(update={"pretrain_weights": None})
        finally:
            self.model_config.resolution = _orig_resolution
            self.model_config.positional_encoding_size = _orig_pe
            if _live_model is not None:
                if hasattr(_live_model, "resolution"):
                    _live_model.resolution = _orig_model_resolution
                if _live_args is not None:
                    if hasattr(_live_args, "resolution"):
                        _live_args.resolution = _orig_args_resolution
                    if hasattr(_live_args, "positional_encoding_size"):
                        _live_args.positional_encoding_size = _orig_args_pe
        module = RFDETRModelModule(eval_model_config, config)

        # Free the original model's accelerator memory for the transplant -- otherwise the resident
        # original and the freshly built (randomly initialized) eval module are both on the accelerator
        # simultaneously (peak ~2x model memory), risking OOM on the largest variants right after train()
        # just fit them.  state_dict() below returns references into the live parameter tensors, so this
        # move is reflected in `source_state` regardless of call order; restored immediately once the eval
        # module has consumed the weights, before the (possibly slow) datamodule/trainer setup below.
        _original_device = getattr(self.model, "device", None)
        _moved_to_cpu = _original_device is not None and str(_original_device) != "cpu"
        source_model = self.model.model
        if source_model is None:
            raise RuntimeError("Cannot evaluate: the base model has been cleared by a previous inplace optimization.")
        if _moved_to_cpu:
            with torch.inference_mode(False):
                source_model = source_model.to("cpu")
                self.model.model = source_model
        try:
            source_state = source_model.state_dict()
            # Reconcile DINOv2 positional embeddings when a `resolution` override changed the PE grid
            # (no-op when unchanged), so the transplant works at the evaluation resolution.
            interpolate_position_embeddings(source_state, eval_model_config.positional_encoding_size)
            module.model.load_state_dict(source_state)
        finally:
            if _moved_to_cpu:
                _move_model_context_to_device(self.model)
        # `self.model_config` was already restored to its pre-call values by the `finally` block above; a
        # `resolution` override only survives on `eval_model_config`, which is what actually built `module`.
        # The datamodule must use the same config so the dataloader resizes to the resolution being evaluated.
        datamodule = RFDETRDataModule(eval_model_config, config)

        # Warn (do not adapt) when the dataset class count differs from the model's head.
        stage = "test" if split == "test" else "validate"
        datamodule.setup(stage)
        dataset_class_names = getattr(datamodule, "class_names", None)
        if isinstance(dataset_class_names, list) and len(dataset_class_names) != self.model_config.num_classes:
            warnings.warn(
                f"Dataset '{config.dataset_dir}' has {len(dataset_class_names)} classes but the model has "
                f"num_classes={self.model_config.num_classes}. Evaluating with the model's head unchanged; "
                "class indices may not line up with the dataset.",
                UserWarning,
                stacklevel=2,
            )

        trainer_kwargs: dict[str, Any] = {"include_training_callbacks": False}
        if _accelerator is not None:
            trainer_kwargs["accelerator"] = _accelerator
        if _devices is not None:
            trainer_kwargs["devices"] = _devices
        trainer = build_trainer(config, self.model_config, **trainer_kwargs)

        # The eval trainer intentionally has no logger; the metric callback still logs with
        # ``logger=True``, so PTL emits one "no logger configured" warning per metric.  Suppress
        # only that specific message — the metrics are still collected from the trainer's results.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*no logger configured.*")
            if split == "test":
                results = trainer.test(module, datamodule)
            else:
                results = trainer.validate(module, datamodule)

        return {key: float(value) for key, value in results[0].items()} if results else {}

    @_ensure_model_on_device
    def inference(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype | str = torch.float32,
        *,
        inplace: bool = False,
    ) -> None:
        """Optimize the model for inference with optional JIT compilation and dtype casting.

        Operations are wrapped in the correct CUDA device context to prevent context leaks on multi-GPU setups. When
        ``compile=True`` the model is traced with ``torch.jit.trace`` using a dummy input of ``batch_size`` images at
        the model's current resolution. By default, optimization deep-copies the loaded model before exporting it so the
        original module remains available. Set ``inplace=True`` for memory-constrained inference-only deployments; this
        exports the loaded module itself, may cast it to ``dtype``, and clears ``model.model`` after optimization
        succeeds. In-place optimization is destructive: :meth:`remove_optimized_model` becomes a no-op (issues
        :class:`UserWarning`), and :meth:`export` raises :class:`RuntimeError`. Create or reload a new ``RFDETR``
        instance to recover the original model.

        If ``inplace=True`` and the underlying ``export()`` call mutates the module before raising (e.g. setting
        internal flags and swapping ``forward``), the exception handler resets RFDETR wrapper flags to the unoptimized
        state but cannot undo changes made inside ``export()``. Create a new RFDETR instance for reliable inference
        after such a failure.

        Args:
            compile: If ``True``, trace the model with ``torch.jit.trace`` to obtain
                a JIT-compiled ``ScriptModule``. Set to ``False`` for broader compatibility (e.g. models with dynamic
                control flow).
            batch_size: Number of images the traced model will be optimized for. Ignored when ``compile=False``.
            dtype: Target floating-point dtype for the inference model. Accepts a
                ``torch.dtype`` directly (e.g. ``torch.float16``) or its string name (e.g. ``"float16"``). Defaults to
                ``torch.float32``. When ``dtype`` differs from the model's current dtype, ``to()`` transiently
                allocates both old and new parameter tensors simultaneously; peak memory during optimization is
                approximately 1.5× the model weight size rather than 1×.
            inplace: If ``True``, optimize ``model.model`` directly instead of deep-copying it. This is a destructive,
                inference-only path because ``export()`` mutates the module and dtype casting mutates its parameters.
                Requires ``compile=False``. With the default ``dtype=torch.float32``, the dtype cast is a no-op, so
                memory savings come only from clearing the base model reference rather than from dtype reduction.

        Raises:
            TypeError: If ``dtype`` is not a ``torch.dtype``, or if ``dtype`` is a
                string that does not correspond to a valid ``torch.dtype`` attribute.
            ValueError: If ``dtype`` is not a floating-point dtype, or if ``inplace=True`` is used with
                ``compile=True``.
            RuntimeError: If the base model has already been cleared by a previous inplace optimization.

        Examples:
            >>> from types import SimpleNamespace
            >>> import torch
            >>> class _TinyModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(1, 1)
            ...     def forward(self, x):
            ...         return {"pred_boxes": self.linear(x[:, :1, :1, :1].squeeze(-1).squeeze(-1))}
            ...     def export(self):
            ...         return None
            >>> class _TinyContext:
            ...     def __init__(self):
            ...         self.device = torch.device("cpu")
            ...         self.resolution = 28
            ...         self.model = _TinyModel()
            ...         self.inference_model = None
            >>> model = object.__new__(RFDETR)
            >>> model.model_config = SimpleNamespace(num_channels=3)
            >>> model.model = _TinyContext()
            >>> model._is_optimized_for_inference = False
            >>> model._has_warned_about_not_being_optimized_for_inference = False
            >>> model._optimized_has_been_compiled = False
            >>> model._optimized_batch_size = None
            >>> model._optimized_resolution = None
            >>> model._optimized_dtype = None
            >>> model._optimized_inplace = False
            >>> # Standard (non-inplace) optimization — reversible:
            >>> model.inference(compile=False)
            >>> model._is_optimized_for_inference
            True
            >>> model._optimized_inplace
            False
            >>> model.remove_optimized_model()
            >>> model._is_optimized_for_inference
            False
            >>> # Inplace optimization — destructive, cannot be reversed:
            >>> model.inference(compile=False, dtype="float16", inplace=True)
            >>> model._is_optimized_for_inference
            True
            >>> model._optimized_dtype
            torch.float16
            >>> model._optimized_inplace
            True
        """
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError:
                raise TypeError(f"dtype must be a torch.dtype or a string name of a dtype, got {dtype!r}") from None
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype or a string name of a dtype, got {type(dtype)!r}")
        if not dtype.is_floating_point:
            raise ValueError(f"dtype must be a floating-point torch.dtype or string name of one, got {dtype}")
        if inplace and compile:
            raise ValueError(
                "inference(inplace=True) requires compile=False. "
                "torch.jit.trace retains references to the original parameter storage in the returned "
                "ScriptModule, so setting model.model=None would not free the weight tensors and "
                "inplace=True would not reduce memory usage."
            )

        # Clear any previously optimized state before starting a new optimization run.
        self.remove_optimized_model()

        if self.model.model is None:
            raise RuntimeError(
                "Cannot optimize: the base model has been cleared by a previous inplace optimization. "
                "Create or reload a new RFDETR instance."
            )

        device = self.model.device
        cuda_ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()

        try:
            with cuda_ctx:
                inference_model: Any = self.model.model if inplace else deepcopy(self.model.model)
                inference_model.eval()
                inference_model.export()

                inference_model = inference_model.to(dtype=dtype)

                if compile:
                    inference_model = torch.jit.trace(  # type: ignore[no-untyped-call]
                        inference_model,
                        torch.randn(
                            batch_size,
                            self.model_config.num_channels,
                            self.model.resolution,
                            self.model.resolution,
                            device=self.model.device,
                            dtype=dtype,
                        ),
                    )
                    self._optimized_has_been_compiled = True
                    self._optimized_batch_size = batch_size

                # Set success flags only after all operations complete.
                self.model.inference_model = inference_model
                # _optimized_inplace must be set before the destructive clear so the cleanup
                # guard in remove_optimized_model() sees the correct state if an exception fires
                # between this assignment and the None clear (extremely unlikely in normal Python
                # but eliminates a theoretical zombie-state window).
                self._optimized_inplace = inplace
                if inplace:
                    self.model.model = None
                self._optimized_resolution = self.model.resolution
                self._is_optimized_for_inference = True
                self._optimized_dtype = dtype
        except Exception:
            # Ensure the object is left in a consistent, unoptimized state if optimization fails.
            with contextlib.suppress(Exception):
                self.remove_optimized_model()
            raise

    @deprecated(target=inference, deprecated_in="1.9.0", remove_in="1.11.0")  # type: ignore[untyped-decorator]
    def optimize_for_inference(
        self,
        compile: bool = True,
        batch_size: int = 1,
        dtype: torch.dtype | str = torch.float32,
        *,
        inplace: bool = False,
    ) -> None:
        """Deprecated alias for :meth:`inference`.

        .. deprecated:: 1.9.0
            ``optimize_for_inference`` was renamed to :meth:`inference`. Deprecated since v1.9.0, will be
            removed in v1.11.0. Use :meth:`inference` instead.

        Args:
            compile: See :meth:`inference`.
            batch_size: See :meth:`inference`.
            dtype: See :meth:`inference`.
            inplace: See :meth:`inference`.
        """
        ...

    def remove_optimized_model(self) -> None:
        """Remove the optimized inference model and reset all optimization flags.

        Clears ``model.inference_model`` and resets all internal state set by :meth:`inference`. Safe to
        call even if the model has not been optimized. When the model was optimized with ``inplace=True``, this method
        issues a :class:`UserWarning` and returns without modifying state — the original module cannot be restored
        because ``export()`` and dtype casting mutate it; create or reload a new ``RFDETR`` instance instead.

        Examples:
            >>> from types import SimpleNamespace
            >>> import torch
            >>> class _TinyModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(1, 1)
            ...     def forward(self, x):
            ...         return {"pred_boxes": self.linear(x[:, :1, :1, :1].squeeze(-1).squeeze(-1))}
            ...     def export(self):
            ...         return None
            >>> class _TinyContext:
            ...     def __init__(self):
            ...         self.device = torch.device("cpu")
            ...         self.resolution = 28
            ...         self.model = _TinyModel()
            ...         self.inference_model = None
            >>> model = object.__new__(RFDETR)
            >>> model.model_config = SimpleNamespace(num_channels=3)
            >>> model.model = _TinyContext()
            >>> model._is_optimized_for_inference = False
            >>> model._has_warned_about_not_being_optimized_for_inference = False
            >>> model._optimized_has_been_compiled = False
            >>> model._optimized_batch_size = None
            >>> model._optimized_resolution = None
            >>> model._optimized_dtype = None
            >>> model._optimized_inplace = False
            >>> model.inference(compile=False)
            >>> model.remove_optimized_model()
            >>> model._is_optimized_for_inference
            False
        """
        if getattr(self, "_optimized_inplace", False):
            warnings.warn(
                "remove_optimized_model() has no effect after inplace optimization — the original model "
                "cannot be restored because export() and dtype casting mutate it. "
                "Create or reload a new RFDETR instance instead.",
                UserWarning,
                stacklevel=2,
            )
            return
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None
        self._optimized_inplace = False

    @property
    def is_optimized_inplace(self) -> bool:
        """Whether the model was optimized with ``inplace=True``.

        Returns ``True`` after a successful :meth:`inference` call with ``inplace=True``,
        meaning the base model has been cleared and :meth:`remove_optimized_model` is a no-op.

        Examples:
            >>> from types import SimpleNamespace
            >>> import torch
            >>> class _TinyModel(torch.nn.Module):
            ...     def __init__(self):
            ...         super().__init__()
            ...         self.linear = torch.nn.Linear(1, 1)
            ...     def forward(self, x):
            ...         return {"pred_boxes": self.linear(x[:, :1, :1, :1].squeeze(-1).squeeze(-1))}
            ...     def export(self):
            ...         return None
            >>> class _TinyContext:
            ...     def __init__(self):
            ...         self.device = torch.device("cpu")
            ...         self.resolution = 28
            ...         self.model = _TinyModel()
            ...         self.inference_model = None
            >>> model = object.__new__(RFDETR)
            >>> model.model_config = SimpleNamespace(num_channels=3)
            >>> model.model = _TinyContext()
            >>> model._is_optimized_for_inference = False
            >>> model._has_warned_about_not_being_optimized_for_inference = False
            >>> model._optimized_has_been_compiled = False
            >>> model._optimized_batch_size = None
            >>> model._optimized_resolution = None
            >>> model._optimized_dtype = None
            >>> model._optimized_inplace = False
            >>> model.is_optimized_inplace
            False
            >>> model.inference(compile=False, inplace=True)
            >>> model.is_optimized_inplace
            True
        """
        return getattr(self, "_optimized_inplace", False)

    def export(
        self,
        output_dir: str = "output",
        infer_dir: str | None = None,
        backbone_only: bool = False,
        opset_version: int = 17,
        verbose: bool = True,
        shape: tuple[int, int] | None = None,
        batch_size: int = 1,
        dynamic_batch: bool = False,
        patch_size: int | None = None,
        format: str = "onnx",
        quantization: str | None = None,
        calibration_data: str | np.ndarray[Any, Any] | None = None,
        max_images: int = 100,
        *,
        backend: str | None = None,
        soc: str | None = None,
        fp16: bool = True,
        notes: object = None,
        coreml_precision: str | None = None,
        output_name: str | None = None,
    ) -> Path:
        """Export the trained model to ONNX, TFLite, TensorRT, ExecuTorch, or CoreML format.

        See the `export documentation <https://rfdetr.roboflow.com/learn/export/>`_ for more information.

        Args:
            output_dir: Directory to write the exported model to.
            infer_dir: Optional directory of sample images for dynamic-axes inference.
            backbone_only: Export only the backbone (feature extractor).
            opset_version: ONNX opset version to target.
            verbose: Print export progress information.
            shape: ``(height, width)`` tuple; defaults to square at model resolution.
                Both dimensions must be divisible by ``patch_size * num_windows``.
            batch_size: Static batch size to bake into the ONNX graph.
            dynamic_batch: If True, export with a dynamic batch dimension so the model accepts variable batch sizes
                at runtime (spatial dimensions always stay fixed).  Applies to the ONNX and TFLite graphs.  Not
                supported for ExecuTorch export on executorch 1.3.1 (raises ``NotImplementedError``): the runtime
                cannot resize RF-DETR's windowed-attention reshapes, so a dynamic ``.pte`` runs only at the traced
                batch — export one ``.pte`` per batch size instead.  Also unsupported for native CoreML
                (``format="coreml"``): fixed shapes are required for reliable ANE / GPU scheduling.
            patch_size: Backbone patch size. Defaults to the value stored in
                ``model_config.patch_size`` (typically 14 or 16). When provided explicitly it must match the
                instantiated model's patch size. Shape divisibility is validated against ``patch_size * num_windows``.
            format: Export format — ``"onnx"`` (default), ``"tflite"``, ``"tensorrt"`` (alias: ``"trt"``),
                ``"executorch"`` (alias: ``"pte"``), or ``"coreml"``.
                ``"tflite"`` and ``"tensorrt"`` both first export to ONNX, then convert: ``"tflite"`` via
                ``onnx2tf`` (requires ``pip install rfdetr[tflite]``); ``"tensorrt"`` via the TensorRT
                Python API (requires ``pip install rfdetr[tensorrt]``).  Unlike ``"onnx"``/
                ``"tflite"`` portable serialization, ``"tensorrt"`` performs target-specific compilation at export
                time and produces a non-portable ``.trt`` engine tied to the build machine's GPU and TensorRT version.
                When ``"executorch"`` is selected the model is exported directly via ``torch.export`` to an ExecuTorch
                ``.pte`` file (no ONNX step), configured by *backend* / *soc* below.  Requires
                ``pip install rfdetr[executorch]``.
                When ``"coreml"`` is selected the model is exported via ``torch.export`` + ``coremltools`` to a
                native ``.mlpackage`` (no ONNX step; requires ``pip install rfdetr[coreml]``). This is distinct from
                ExecuTorch's ``format="executorch", backend="coreml"`` path, which still produces a ``.pte``. If
                you know that ExecuTorch delegate and expect ``format="coreml"`` to mean the same thing: it does
                not — pass ``format="executorch", backend="coreml"`` for the ``.pte`` route instead. Passing both
                ``format="coreml"`` and ``backend="coreml"`` together does **not** fall through to the ExecuTorch
                delegate; ``backend`` is ignored (with a warning) and the native ``.mlpackage`` path always runs.
                Keypoint models are untested with ``format="coreml"`` — detection and segmentation have
                registry-clean and numerical-parity test coverage (see
                ``tests/export/test_coreml_op_coverage.py`` / ``test_coreml_export.py``), keypoint models
                currently do not.

                .. warning::
                    TFLite, ExecuTorch, and CoreML export are experimental and subject to change; upstream dependency
                    instabilities (``onnx2tf``, ``ai_edge_litert``, ``executorch``, ``coremltools``) may affect results.
            quantization: TFLite quantization mode (ignored when
                ``format="onnx"``).  One of ``None``, ``"fp32"``, ``"fp16"``, ``"int8"``.  ``None`` / ``"fp32"`` /
                ``"fp16"`` produce FP32 + FP16 ``.tflite`` files; ``"int8"`` additionally produces an INT8-quantized
                model.
            calibration_data: Representative images for INT8 calibration and ``onnx2tf`` output validation.  Accepts:

                * ``None`` — auto-generate random data (sufficient for fp32/fp16; warns for int8).
                * A **directory path** (``str``) containing JPEG/PNG
                  images — the converter automatically loads, resizes, and prepares them.  This is the simplest
                  approach.
                * A path (``str``) to a ``.npy`` file of shape ``(N, H, W, 3)``, dtype float32, values in ``[0, 1]``.
                * A :class:`numpy.ndarray` with the same format.

                For INT8 quantization, provide 20–100 representative images from your training/validation set for best
                accuracy.
            max_images: Maximum number of images to load from a calibration directory.  Defaults to ``100``.  Only used
                when *calibration_data* is a directory path.
            backend: Hardware backend to specialize the export for.  Required when ``format="executorch"`` and
                ignored — with a warning — for any other format.  Accepted values for ExecuTorch:
                ``"xnnpack"`` (portable CPU, fp32), ``"coreml"`` (Apple devices, fp16; requires ``coremltools``),
                and ``"qnn"`` (Qualcomm Snapdragon HTP, fp16; requires an ExecuTorch source build against the
                QAIRT SDK — not available via pip).
            soc: Target SoC (System on Chip) — the specific Qualcomm Snapdragon chip the exported model will run
                on.  Required when ``backend="qnn"``: the QNN backend compiles the ``.pte`` ahead-of-time for one
                chip's Hexagon Tensor Processor (HTP), unlike ``"xnnpack"``/``"coreml"`` which run on any device of
                their platform, so the target chip must be known at export time.  Ignored — with a warning — for
                any other backend or format.  Must be a
                :class:`~executorch.backends.qualcomm.serialization.qc_schema.QcomChipset` name, e.g. ``"SM8650"``
                (Snapdragon 8 Gen 3); see that enum for the full list of supported chips.  Has no effect for
                ``"xnnpack"`` or ``"coreml"``.
            fp16: Build the TensorRT engine with FP16 precision.  Only applies when ``format="tensorrt"``
                (alias ``"trt"``); ignored for every other format.  Defaults to ``True`` for lowest latency
                on NVIDIA GPUs.  Pass ``False`` to build an FP32 engine — required on TensorRT builds that do
                not expose the FP16 builder flag (``export()`` otherwise aborts while configuring FP16).
            notes: Optional user-defined metadata (string, dict, list, or
                any JSON-serialisable value) to embed in the exported ONNX model under the ``"rfdetr_notes"`` metadata
                property.  When ``None`` no metadata entry is written.  String values are stored verbatim; all other
                types are JSON-encoded so consumers must call ``json.loads()`` to recover a dict or list.  The same
                value can be passed to :meth:`train` so the checkpoint and the ONNX file share the same provenance
                information.  **Ignored for ``format="executorch"`` and ``format="coreml"``**: those artifacts have
                no ONNX-style metadata slot, and a non-``None`` value emits a ``UserWarning`` instead of being
                embedded.
            coreml_precision: ``ct.convert`` compute precision for ``format="coreml"`` — ``None`` (default) or
                ``"float32"`` selects FP32 (tight CPU parity with eager PyTorch); ``"float16"`` selects a smaller
                ANE-oriented bundle (expect larger numeric drift). Ignored for every other format.
            output_name: Full filename override (without extension), e.g. ``"my-model"``. When set, takes
                precedence over the model's variant name (``self.size``) and the exported file is named
                ``{output_name}.{ext}`` verbatim — this also suppresses the ``_fp32``/``_fp16``/``_{backend}``
                detail suffix that would otherwise be appended to encode the resolved precision/backend/SoC
                (see *format* / *coreml_precision* / *backend* / *soc* / *fp16* above). Sanitized against path
                traversal (only the basename, extension stripped, is used). Exception: ``format="tflite"``
                always writes multiple files (one per precision/quantization mode), so the ``_fp32``/``_fp16``/
                ``_dynamic_range_quant`` suffix is unavoidable even with *output_name* set — it becomes the stem
                instead of the model's variant name.
                Exceptions: ``format="onnx"`` with ``backbone_only=True`` appends ``-backbone`` to the filename
                (``{output_name}-backbone.onnx``); ``format="tflite"`` writes separate per-precision files instead
                of a single ``{output_name}.tflite`` file. The TFLite filenames may include a ``_gs_patched`` infix
                before the precision suffix when GridSample ops are patched, e.g.
                ``{output_name}_gs_patched_fp32.tflite``; this is the standard RF-DETR path.

        Returns:
            Path to the exported model file (``.onnx``, ``.tflite``, ``.trt``, ``.pte``, or ``.mlpackage``).

        Raises:
            ValueError: If ``format`` is unrecognized; if ``format="executorch"`` and ``backend`` is missing,
                unrecognized, or (for ``backend="qnn"``) ``soc`` is missing; or if the resolved export shape is
                not divisible by ``patch_size * num_windows``.
            NotImplementedError: If ``dynamic_batch=True`` is combined with ``format="executorch"`` or
                ``format="coreml"`` — those paths require a fixed batch size.
            ImportError: If the optional dependencies for the requested ``format``/``backend`` are not installed
                (e.g. ``rfdetr[onnx]``, ``rfdetr[executorch]``, ``rfdetr[coreml]``, ``coremltools`` for ExecuTorch
                ``backend="coreml"``, or an ExecuTorch source build against the QAIRT SDK for ``backend="qnn"``).
            RuntimeError: If called after the model has undergone in-place inference optimization (the original
                model has been cleared; instantiate a new :class:`RFDETR` to export).
        """
        if format == "trt":  # "trt" is an alias for "tensorrt"
            format = "tensorrt"
        if format == "pte":  # "pte" is an alias for "executorch"
            format = "executorch"
        from rfdetr.export._backend import _resolve_export_backend

        backend, soc = _resolve_export_backend(format, backend, soc)
        # Fail fast: dynamic_batch is statically incompatible with ExecuTorch / CoreML; refuse before any forward
        # pass so the user doesn't pay the full DINOv2 forward (seconds + GBs) before seeing the error. These are
        # intentional early checks before heavy optional imports (ExecuTorch also checks in the converter).
        if dynamic_batch and format == "executorch":
            raise NotImplementedError(
                "ExecuTorch export does not support dynamic_batch (see export_executorch for details). "
                "Export one .pte per batch size instead."
            )
        if dynamic_batch and format == "coreml":
            raise NotImplementedError(
                "CoreML export does not support dynamic_batch (fixed shapes are required for reliable "
                "ANE / GPU scheduling). Export one .mlpackage per batch size instead."
            )
        logger.info(f"Exporting model to {format} format")
        try:
            from rfdetr.export.main import export_onnx, make_infer_image
        except ImportError:
            logger.error(
                "It seems some dependencies for ONNX export are missing."
                " Please run `pip install rfdetr[onnx]` and try again.",
            )
            raise

        device = self.model.device

        if getattr(self, "_optimized_inplace", False) or self.model.model is None:
            raise RuntimeError(
                "RFDETR.export() is not available after inplace optimization. "
                "The original model has been cleared. Create a new RFDETR instance."
            )

        # Move the live model to CPU before deepcopying and keep it there during export. ``nn.Module.to(...)`` mutates
        # in place, so this frees GPU memory for the local export copy, ONNX tracing, TFLite conversion, and any
        # calibration tensors. The ``finally`` block restores the live model even if export or conversion raises.
        self.model.model = self.model.model.to("cpu")
        model = deepcopy(self.model.model)
        model.to(device)
        try:
            os.makedirs(output_dir, exist_ok=True)
            output_dir_path = Path(output_dir)
            patch_size = _resolve_patch_size(patch_size, self.model_config, "export")
            num_windows = getattr(self.model_config, "num_windows", 1)
            if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows <= 0:
                raise ValueError(f"num_windows must be a positive integer, got {num_windows!r}")
            block_size = patch_size * num_windows
            if shape is None:
                shape = (self.model.resolution, self.model.resolution)
                if shape[0] % block_size != 0:
                    raise ValueError(
                        f"Model's default resolution ({self.model.resolution}) is not divisible by "
                        f"block_size={block_size} (patch_size={patch_size} * num_windows={num_windows}). "
                        f"Provide an explicit shape divisible by {block_size}.",
                    )
            else:
                shape = _validate_shape_dims(shape, block_size, patch_size, num_windows)

            # Freeze the backbone's position embeddings to the export shape before tracing. Without this, a
            # `shape` that differs from the model's native resolution forces the traced forward pass through
            # DINOv2's antialiased bicubic interpolation, which has no ONNX symbolic (`aten::_upsample_bicubic2d_aa`
            # is unsupported). Precomputing here (outside the trace) keeps that op out of the traced graph.
            for backbone_module in model.modules():
                if isinstance(backbone_module, DinoV2):
                    backbone_module.shape = shape
                    backbone_module.export()

            input_tensors = make_infer_image(
                infer_dir, shape, batch_size, device, num_channels=self.model_config.num_channels
            ).to(device)
            input_names = ["input"]
            if backbone_only:
                output_names = ["features"]
            elif self.model_config.segmentation_head:
                output_names = ["dets", "labels", "masks"]
            elif self.model_config.use_grouppose_keypoints:
                output_names = ["dets", "labels", "keypoints"]
            else:
                output_names = ["dets", "labels"]

            if dynamic_batch:
                dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
            else:
                dynamic_axes = None
            model.eval()
            with torch.no_grad():
                if backbone_only:
                    features = model(input_tensors)
                    logger.debug(f"PyTorch inference output shape: {features.shape}")
                elif self.model_config.segmentation_head:
                    outputs = model(input_tensors)
                    dets = outputs["pred_boxes"]
                    labels = outputs["pred_logits"]
                    masks = outputs["pred_masks"]
                    if isinstance(masks, torch.Tensor):
                        logger.debug(
                            f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                            f"Masks: {masks.shape}",
                        )
                    else:
                        logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
                elif self.model_config.use_grouppose_keypoints:
                    outputs = model(input_tensors)
                    dets = outputs["pred_boxes"]
                    labels = outputs["pred_logits"]
                    keypoints = outputs["pred_keypoints"]
                    logger.debug(
                        f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                        f"Keypoints: {keypoints.shape}",
                    )
                else:
                    outputs = model(input_tensors)
                    dets = outputs["pred_boxes"]
                    labels = outputs["pred_logits"]
                    logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")

            model.cpu()
            input_tensors = input_tensors.cpu()

            if format == "executorch":
                from rfdetr.export._backend import _export_executorch_format

                return _export_executorch_format(
                    model,
                    input_tensors,
                    output_dir_path,
                    backend=backend,
                    soc=soc,
                    variant_name=getattr(self, "size", None),
                    dynamic_batch=dynamic_batch,
                    notes=notes,
                    output_name=output_name,
                )

            if format == "coreml":
                from rfdetr.export._backend import _export_coreml_format

                return _export_coreml_format(
                    model,
                    input_tensors,
                    output_dir_path,
                    variant_name=getattr(self, "size", None),
                    verbose=verbose,
                    notes=notes,
                    compute_precision=coreml_precision,
                    output_name=output_name,
                )

            output_file = export_onnx(
                output_dir=str(output_dir_path),
                model=model,
                input_names=input_names,
                input_tensors=input_tensors,
                output_names=output_names,
                dynamic_axes=dynamic_axes,
                backbone_only=backbone_only,
                verbose=verbose,
                opset_version=opset_version,
                variant_name=getattr(self, "size", None),
                notes=notes,
                output_name=output_name,
            )

            logger.info(f"Successfully exported ONNX model to: {output_file}")

            from rfdetr.export.main import _convert_onnx_export

            return _convert_onnx_export(
                output_file,
                format,
                output_dir_path,
                quantization=quantization,
                calibration_data=calibration_data,
                max_images=max_images,
                verbose=verbose,
                fp16=fp16,
                output_name=output_name,
            )
        finally:
            self.model.model = self.model.model.to(device)

    @staticmethod
    def _load_classes(dataset_dir: str) -> list[str]:
        """Load class names from a COCO or YOLO dataset directory."""
        if is_valid_coco_dataset(dataset_dir):
            coco_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
            with open(coco_path, encoding="utf-8") as f:
                anns = json.load(f)
            categories = sorted(anns["categories"], key=lambda category: category.get("id", float("inf")))

            # Catch possible placeholders for no supercategory
            placeholders = {"", "none", "null", None}

            # If no meaningful supercategory exists anywhere, treat as flat dataset
            has_any_sc = any(c.get("supercategory", "none") not in placeholders for c in categories)
            if not has_any_sc:
                return [c["name"] for c in categories]

            # Mixed/Hierarchical: keep only categories that are not parents of other categories.
            # Both leaves (with a real supercategory) and standalone top-level nodes (supercategory is a
            # placeholder) satisfy this condition — neither appears as another category's supercategory.
            parents = {c.get("supercategory") for c in categories if c.get("supercategory", "none") not in placeholders}
            has_children = {c["name"] for c in categories if c["name"] in parents}

            class_names = [c["name"] for c in categories if c["name"] not in has_children]
            # Safety fallback for pathological inputs
            return class_names or [c["name"] for c in categories]

        yaml_path = RFDETR._yolo_data_file_path(dataset_dir) if is_valid_yolo_dataset(dataset_dir) else None
        if yaml_path is not None:
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if "names" in data:
                if isinstance(data["names"], dict):
                    return [str(data["names"][i]) for i in sorted(data["names"].keys())]
                return [str(name) for name in data["names"]]
            raise ValueError(f"Found {yaml_path} but it does not contain 'names' field.")
        raise FileNotFoundError(
            f"Could not find class names in {dataset_dir}."
            " Checked for COCO (train/_annotations.coco.json) and YOLO (data.yaml, data.yml) styles.",
        )

    @staticmethod
    def _detect_num_classes_for_training(dataset_dir: str, *, use_grouppose_keypoints: bool = False) -> int:
        """Detect the class count using the same category basis as training labels.

        For COCO-style datasets this counts all categories by ``id`` from ``train/_annotations.coco.json`` (matching the
        remapping based on ``coco.cats`` used by the training datamodule). In keypoint mode it instead counts the
        inferred RF-DETR keypoint label slots. In legacy background-first schemas (e.g. ``[0, 17]``) slot ``0`` is
        reserved for classes without keypoints; active-first schemas (e.g. ``[17]``) use normal 0-based indices. For
        YOLO-style datasets it falls back to ``_load_classes``.
        """
        if is_valid_coco_dataset(dataset_dir):
            coco_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
            if use_grouppose_keypoints:
                return len(infer_coco_keypoint_schema(coco_path).class_names)
            with open(coco_path, encoding="utf-8") as f:
                anns = json.load(f)
            categories = anns["categories"]
            cat_by_id = {category["id"]: category for category in categories}
            return len(cat_by_id)

        return len(RFDETR._load_classes(dataset_dir))

    def _align_num_classes_from_dataset(self, dataset_dir: str) -> None:
        """Auto-detect the dataset class count and align ``model_config.num_classes`` in-place.

        Must be called before ``RFDETRModelModule`` is constructed so that weight loading inside the module uses the
        correct (dataset-derived) class count.

        When the user did **not** explicitly set ``num_classes`` (it is left unset, e.g. inferred from a
        checkpoint), ``model_config.num_classes`` and ``self.model.args.num_classes`` are updated to match the dataset.
        When the user *did* set ``num_classes`` explicitly — to any value, including the class default — and it differs
        from the dataset, the configured value is preserved and a warning is emitted.

        Failures from ``_detect_num_classes_for_training`` are caught and logged at DEBUG level so that training is
        never blocked by detection errors.

        When ``model_config.use_grouppose_keypoints`` is True and
        ``model_config.num_keypoints_per_class`` is shorter than the adjusted
        ``num_classes``, the schema is zero-padded in-place so that
        ``len(num_keypoints_per_class) == num_classes``.  Both ``model_config``
        and ``model.args`` (if present) are updated.  Appended classes receive
        zero keypoints and contribute no class-logit boost.

        Args:
            dataset_dir: Path to the training dataset root directory.
        """
        try:
            dataset_num_classes = RFDETR._detect_num_classes_for_training(
                dataset_dir,
                use_grouppose_keypoints=self.model_config.use_grouppose_keypoints,
            )
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            # Best-effort only; do not block training if detection fails.
            logger.debug("Could not auto-detect num_classes from dataset '%s': %s", dataset_dir, exc)
            return

        # Hoist so both branches below can reference the schema without re-fetching.
        keypoint_schema: list[int] = []
        if self.model_config.use_grouppose_keypoints:
            # Older configs may omit the schema; absence means no schema-based class-count expansion.
            keypoint_schema = list(getattr(self.model_config, "num_keypoints_per_class", []) or [])
            if keypoint_schema:
                dataset_num_classes = max(dataset_num_classes, len(keypoint_schema))

        model_num_classes = self.model_config.num_classes

        if dataset_num_classes == model_num_classes:
            return

        # Determine whether the user explicitly set num_classes.  "num_classes" in
        # model_fields_set is True only when the field was explicitly provided at construction
        # (or assigned afterwards); an explicit value is honored regardless of whether it equals
        # the class default, so an intentional num_classes is never silently overridden by the
        # dataset count.  A checkpoint-derived num_classes is cleared from model_fields_set by
        # ``from_checkpoint`` (see PR #1106 / issue #1092), so it correctly counts as "not set" here.
        user_overrode = "num_classes" in getattr(self.model_config, "model_fields_set", set())

        if not user_overrode:
            logger.debug(
                "Detected %d classes in dataset '%s'; auto-adjusting model num_classes from %d to %d.",
                dataset_num_classes,
                dataset_dir,
                model_num_classes,
                dataset_num_classes,
            )
            self.model_config.num_classes = dataset_num_classes
            # Keep serialized checkpoint metadata in sync with the updated class count.
            model_args = getattr(self.model, "args", None)
            if model_args is not None:
                model_args.num_classes = dataset_num_classes
            # Pad keypoint schema with zeros so len(num_keypoints_per_class) == num_classes.
            # Without this, _aggregate_keypoint_class_logits emits a one-time mismatch
            # warning per model instance and the config state is inconsistent with the
            # detection head width.
            if keypoint_schema and len(keypoint_schema) < dataset_num_classes:
                padded_schema = keypoint_schema + [0] * (dataset_num_classes - len(keypoint_schema))
                self.model_config.num_keypoints_per_class = padded_schema
                if model_args is not None:
                    model_args.num_keypoints_per_class = padded_schema
        else:
            logger.warning(
                "Dataset '%s' has %d classes but model was initialized with num_classes=%d. "
                "Using the model's configured value (%d). If this is unintentional, "
                "reinitialize the model with num_classes=%d.",
                dataset_dir,
                dataset_num_classes,
                model_num_classes,
                model_num_classes,
                dataset_num_classes,
            )
            # Also pad schema when the user-configured num_classes exceeds the schema length,
            # to prevent the _aggregate_keypoint_class_logits mismatch warning in this path too.
            if keypoint_schema and len(keypoint_schema) < model_num_classes:
                padded_schema = keypoint_schema + [0] * (model_num_classes - len(keypoint_schema))
                self.model_config.num_keypoints_per_class = padded_schema
                model_args = getattr(self.model, "args", None)
                if model_args is not None:
                    model_args.num_keypoints_per_class = padded_schema

    @staticmethod
    def _roboflow_keypoint_annotation_path(dataset_dir: str) -> Path | None:
        """Return the Roboflow COCO train annotation path when it exists.

        Args:
            dataset_dir: Path to the Roboflow dataset root.

        Returns:
            Train split annotation path, or ``None`` when the dataset is not Roboflow COCO style.

        Raises:
            This helper does not raise.

        Example:
            >>> RFDETR._roboflow_keypoint_annotation_path("/missing") is None
            True
        """
        if not is_valid_coco_dataset(dataset_dir):
            return None
        annotation_path = Path(dataset_dir) / "train" / "_annotations.coco.json"
        return annotation_path if annotation_path.exists() else None

    @staticmethod
    def _coco_keypoint_annotation_path(dataset_dir: str) -> Path | None:
        """Return the native COCO train keypoint annotation path when it exists.

        Args:
            dataset_dir: Path to the COCO dataset root.

        Returns:
            Path to ``annotations/person_keypoints_train2017.json``, or ``None`` when it is absent.

        Raises:
            This helper does not raise.

        Example:
            >>> RFDETR._coco_keypoint_annotation_path("/missing") is None
            True
        """
        annotation_path = Path(dataset_dir) / "annotations" / "person_keypoints_train2017.json"
        return annotation_path if annotation_path.exists() else None

    @staticmethod
    def _yolo_data_file_path(dataset_dir: str) -> Path | None:
        """Return the YOLO data file path when a dataset root has one.

        Args:
            dataset_dir: Path to the YOLO dataset root.

        Returns:
            Path to ``data.yaml`` or ``data.yml``, or ``None`` when neither exists.

        Raises:
            This helper does not raise.

        Example:
            >>> RFDETR._yolo_data_file_path("/missing") is None
            True
        """
        root = Path(dataset_dir)
        for filename in REQUIRED_YOLO_YAML_FILES:
            data_file = root / filename
            if data_file.exists():
                return data_file
        return None

    @staticmethod
    def _flip_idx_to_pairs(flip_idx: list[int]) -> list[int]:
        """Convert Ultralytics ``flip_idx`` permutation metadata to flat swap pairs."""
        pairs: list[int] = []
        seen: set[int] = set()
        for idx, mirror_idx in enumerate(flip_idx):
            if idx in seen or mirror_idx in seen or idx == mirror_idx:
                seen.add(idx)
                continue
            if 0 <= mirror_idx < len(flip_idx) and flip_idx[mirror_idx] == idx:
                pairs.extend([idx, mirror_idx])
                seen.update({idx, mirror_idx})
        return pairs

    def _align_keypoint_schema_from_dataset(self, config: TrainConfig) -> None:
        """Infer or validate keypoint schema from COCO, Roboflow COCO, or YOLO pose metadata.

        Args:
            config: Training configuration containing dataset location and format.

        Returns:
            ``None``. The model config is updated in-place when dataset metadata is available.

        Raises:
            This method does not raise for missing or malformed metadata; later dataset construction still validates
            keypoint-mode requirements.

        Example:
            >>> from rfdetr.config import RFDETRKeypointPreviewConfig, TrainConfig
            >>> model = object.__new__(RFDETR)
            >>> model.model_config = RFDETRKeypointPreviewConfig(pretrain_weights=None)
            >>> model.model = type("Context", (), {"args": None})()
            >>> model._align_keypoint_schema_from_dataset(TrainConfig(dataset_dir="/missing", tensorboard=False))
        """

        if not self.model_config.use_grouppose_keypoints:
            return
        dataset_file = getattr(config, "dataset_file", None)
        if dataset_file not in ("coco", "roboflow", "yolo"):
            return
        dataset_dir = getattr(config, "dataset_dir", None)
        if not dataset_dir:
            return

        if not hasattr(self, "_keypoint_schema_cache"):
            self._keypoint_schema_cache: dict[Any, Any] = {}

        cache_key = (dataset_file, dataset_dir)
        if cache_key in self._keypoint_schema_cache:
            inferred, source_path, source_kind = self._keypoint_schema_cache[cache_key]
        else:
            try:
                if dataset_file == "coco":
                    annotation_path = RFDETR._coco_keypoint_annotation_path(dataset_dir)
                    if annotation_path is None:
                        return
                    source_path = annotation_path
                    source_kind = "COCO"
                    inferred = infer_coco_keypoint_schema(annotation_path)
                elif dataset_file == "roboflow":
                    annotation_path = RFDETR._roboflow_keypoint_annotation_path(dataset_dir)
                    if annotation_path is not None:
                        source_path = annotation_path
                        source_kind = "Roboflow COCO"
                        inferred = infer_coco_keypoint_schema(annotation_path)
                    else:
                        yolo_data_file = RFDETR._yolo_data_file_path(dataset_dir)
                        if yolo_data_file is None:
                            return
                        source_path = yolo_data_file
                        source_kind = "YOLO pose"
                        inferred = infer_yolo_keypoint_schema(yolo_data_file)
                else:
                    yolo_data_file = RFDETR._yolo_data_file_path(dataset_dir)
                    if yolo_data_file is None:
                        return
                    source_path = yolo_data_file
                    source_kind = "YOLO pose"
                    inferred = infer_yolo_keypoint_schema(yolo_data_file)
            except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
                logger.info("Could not infer keypoint schema from dataset '%s': %s", dataset_dir, exc)
                return
            self._keypoint_schema_cache[cache_key] = (inferred, source_path, source_kind)

        inferred_schema = inferred.num_keypoints_per_class
        if not getattr(config, "keypoint_flip_pairs", []):
            config.keypoint_flip_pairs = list(inferred.keypoint_flip_pairs)
        # Older configs may omit the schema; absence lets dataset inference populate it.
        current_schema = list(getattr(self.model_config, "num_keypoints_per_class", []) or [])
        user_set_schema = "num_keypoints_per_class" in getattr(self.model_config, "model_fields_set", set())

        if user_set_schema and active_keypoint_counts(current_schema) == active_keypoint_counts(inferred_schema):
            return

        if current_schema != inferred_schema:
            if user_set_schema:
                logger.warning(
                    "Configured num_keypoints_per_class=%s does not match dataset keypoint metadata %s from '%s'. "
                    "Using dataset metadata as the source of truth.",
                    current_schema,
                    inferred_schema,
                    source_path,
                )
            else:
                if _is_bg_first_schema(current_schema) and inferred_schema and not _is_bg_first_schema(inferred_schema):
                    warnings.warn(
                        f"Loaded checkpoint uses a legacy background-first keypoint schema "
                        f"num_keypoints_per_class={current_schema!r}, but the dataset infers "
                        f"active-first {inferred_schema!r}. Training will shift person from slot 1 "
                        f"to slot 0; checkpoint head weights are now misaligned. "
                        f"Pass num_keypoints_per_class={current_schema!r} to the model constructor "
                        f"to keep the legacy schema.",
                        UserWarning,
                        stacklevel=2,
                    )
                logger.info(
                    "Inferred num_keypoints_per_class=%s from %s keypoint metadata at '%s'.",
                    inferred_schema,
                    source_kind,
                    source_path,
                )
            self.model_config.num_keypoints_per_class = inferred_schema
            model_args = getattr(self.model, "args", None)
            if model_args is not None:
                model_args.num_keypoints_per_class = inferred_schema

    def get_train_config(self, **kwargs: Any) -> TrainConfig:
        """Retrieve the configuration parameters that will be used for training."""
        return self._train_config_class(**kwargs)

    def get_model(self, config: ModelConfig, *, trust_checkpoint: bool = False) -> ModelContext:
        """Retrieve a model context from the provided architecture configuration.

        Args:
            config: Architecture configuration.
            trust_checkpoint: Forwarded to :func:`~rfdetr.inference._build_model_context` — set
                ``True`` only when ``config.pretrain_weights`` is a checkpoint the caller
                explicitly trusts (mirrors ``RFDETR.from_checkpoint(..., trust_checkpoint=True)``).

        Returns:
            ModelContext with model, postprocess, device, resolution, args, and class_names attributes.
        """
        return _build_model_context(config, trust_checkpoint=trust_checkpoint)

    @property
    def class_names(self) -> list[str]:
        """Retrieve the class names supported by the loaded model.

        Returns:
            A list of class name strings, 0-indexed.  When no custom class names are embedded in the checkpoint, returns
            the standard 80 COCO class names.
        """
        if hasattr(self.model, "class_names") and self.model.class_names is not None:
            return list(self.model.class_names)

        return list(COCO_CLASS_NAMES)

    def _ensure_eval_mode_for_unoptimized_inference(self) -> None:
        """Put the underlying module in eval mode before unoptimized inference.

        Inference must never run with dropout / batch-norm in training mode. The warning that the model is not optimized
        is emitted at most once, but eval mode is (re)asserted on every call: ``train()`` reassigns ``self.model.model``
        to a module that PyTorch Lightning leaves in training mode (see ``train()``), so gating ``eval()`` behind the
        once-only warning would let a later ``predict()`` silently run with dropout active.

        When ``_is_optimized_for_inference`` is ``True``, the method returns immediately — the compiled
        ``inference_model`` snapshot is already in eval mode and ``self.model.model`` is not used for inference.
        """
        if self._is_optimized_for_inference:
            return
        if not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. Latency may be higher than expected."
                " For full GPU throughput (e.g. ~8x on T4 via FP16 Tensor Cores),"
                " call model.inference(dtype=torch.float16).",
            )
            self._has_warned_about_not_being_optimized_for_inference = True
        # self.model.model is only cleared when optimized for inference (guarded by the early return above).
        model = self.model.model
        assert model is not None
        model.eval()

    @torch.inference_mode()
    # mypy can't match this signature against _ensure_model_on_device's Concatenate[Any, _P] typing without
    # `self` being positional-only (a side effect of the trailing **kwargs); ignored rather than changing the
    # public signature.
    @_ensure_model_on_device  # type: ignore[arg-type]
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
        """Performs model inference on the input images.

        This method accepts a single image or a list of images in various formats (file path, image url, PIL Image,
        NumPy array, or torch.Tensor). The images should be in RGB channel order. If a torch.Tensor is provided, it must
        already be normalized to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images:
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold:
                The minimum confidence score needed to consider a detected bounding box valid.
            shape:
                Optional ``(height, width)`` tuple to resize images to before inference. When provided, overrides the
                model's default inference resolution. The tuple should match the resolution used when exporting the
                model (typically a square shape). Both dimensions must be positive integers divisible by ``patch_size *
                num_windows``. Defaults to ``(model.resolution, model.resolution)`` when not set.
            patch_size:
                Backbone patch size used for shape divisibility validation. Defaults to ``model_config.patch_size``
                (typically 14 for large models, 16 for smaller ones). Divisibility is checked against ``patch_size *
                num_windows``.
            include_source_image:
                Whether to attach the original image to the returned prediction. Detection and segmentation outputs use
                ``detections.metadata["source_image"]``. Keypoint outputs use per-object
                ``key_points.data["source_image"]`` because Supervision ``KeyPoints`` currently has no collection-level
                metadata field. Defaults to ``True``. Set to ``False`` to reduce memory use when source images are not
                needed.
            **kwargs:
                Additional keyword arguments.

        Returns:
            A single or multiple Supervision prediction objects. Detection and segmentation models return
            :class:`~supervision.Detections`. Keypoint models return :class:`~supervision.KeyPoints`, with keypoint
            coordinates in ``xy``. Keypoint predictions preserve the detection-level fields produced by RF-DETR:
            ``key_points.detection_confidence`` is the per-object score used by ``threshold``. For keypoint models this
            is the postprocessed detection score and, by default, includes normalized keypoint uncertainty fusion
            controlled by ``model_config.postprocess_trace_alpha``. ``key_points.keypoint_confidence`` is separate: it
            is a ``(num_detections, num_keypoints)`` array of per-keypoint findability scores decoded from the keypoint
            head, not a repeated copy of the detection score. When RF-DETR emits keypoint precision parameters,
            ``key_points.data["covariance"]`` stores per-keypoint pixel-space covariance matrices with shape
            ``(num_detections, num_keypoints, 2, 2)``. ``key_points.data["xyxy"]`` stores the corresponding detection
            boxes as a ``(num_detections, 4)`` array in the same row order as ``key_points.xy`` because Supervision
            ``KeyPoints`` does not have a native bounding-box field. The ``data`` dict also contains ``class_name`` and
            ``source_shape`` as per-object arrays. When ``include_source_image=True`` for keypoint models,
            ``source_image`` is stored as per-object data until Supervision exposes collection-level metadata for
            ``KeyPoints``.

        Note:
            For ``Detections`` outputs, ``source_image`` moved from ``detections.data`` to ``detections.metadata``.
            Update detection callers reading ``detections.data["source_image"]`` to use
            ``detections.metadata["source_image"]``.

        Note:
            ``class_name`` mapping uses one of three modes depending on the checkpoint. For pretrained COCO checkpoints
            (detected when ``model.args.num_classes > len(class_names)`` and ``class_names`` matches
            ``COCO_CLASS_NAMES``), raw COCO category IDs (1–90, sparse) are looked up by category ID rather than by
            position — so ``class_id=18`` yields ``"dog"``, not ``class_names[18]``. For fine-tuned detection and
            segmentation models and active-first keypoint models, ``class_id`` is a 0-based index into
            ``class_names``. In the one-class preview keypoint setup, that means ``class_id=0`` is the foreground
            class and ``class_id=1`` is ``"__background__"``.
            Legacy keypoint checkpoints with ``args.num_keypoints_per_class[0] == 0`` use a background-first layout:
            slot 0 maps to ``"__background__"`` and foreground slots map to ``class_names`` in order.

        Raises:
            ValueError: If ``shape`` cannot be unpacked as a two-element sequence,
                if either dimension does not support the ``__index__`` protocol (e.g. ``float``) or is a ``bool``, if
                either dimension is zero or negative, if either dimension is not divisible by ``patch_size *
                num_windows``, or if ``patch_size`` is not a positive integer.
        """
        from supervision import Detections, KeyPoints

        patch_size = _resolve_patch_size(patch_size, self.model_config, "predict")
        num_windows = getattr(self.model_config, "num_windows", 1)
        if isinstance(num_windows, bool) or not isinstance(num_windows, int) or num_windows <= 0:
            raise ValueError(f"model_config.num_windows must be a positive integer, got {num_windows!r}")
        block_size = patch_size * num_windows

        if shape is None:
            default_res = self.model.resolution
            if default_res % block_size != 0:
                raise ValueError(
                    f"Model's default resolution ({default_res}) is not divisible by "
                    f"block_size={block_size} (patch_size={patch_size} * num_windows={num_windows}). "
                    f"Provide an explicit shape divisible by {block_size}.",
                )
        else:
            shape = _validate_shape_dims(shape, block_size, patch_size, num_windows)

        self._ensure_eval_mode_for_unoptimized_inference()

        # Determine the return shape from the *input* type, not the runtime batch
        # length: a single image (path / PIL / tensor) yields a bare Detections,
        # while a list/tuple always yields a list — even when it holds one image.
        single_input = not isinstance(images, (list, tuple))
        if not isinstance(images, (list, tuple)):
            images = [images]

        orig_sizes: list[Any] = []
        processed_images: list[Any] = []
        source_images: list[Any] | None = [] if include_source_image else None

        for img_input in images:
            img: Any = img_input
            if isinstance(img, str):
                if urlparse(img).scheme in ("http", "https"):
                    resp = requests.get(img, timeout=30)
                    resp.raise_for_status()
                    img = io.BytesIO(resp.content)
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                # Auto-convert PIL images from any colour mode (L, LA, RGBA, P,
                # etc.) to RGB before converting to tensor.  This matches the
                # standard detector API contract: callers passing a file path or
                # a PIL image should not have to pre-convert; for tensor inputs
                # the channel dimension is the caller's responsibility.
                if isinstance(img, Image.Image) and img.mode != "RGB":
                    img = img.convert("RGB")
                if include_source_image:
                    src = np.array(img)
                    if src.dtype != np.uint8:
                        src = (src * 255).clip(0, 255).astype(np.uint8)
                    source_images.append(src)  # type: ignore[union-attr]
                img = F.to_tensor(img)
            elif include_source_image:
                source_images.append(  # type: ignore[union-attr]
                    (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                )

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1]).",
                )
            if (img < 0).any():
                raise ValueError(
                    "Image has pixel values below 0. Please ensure the image is normalized (scaled to [0, 1]).",
                )
            if img.shape[0] != self.model_config.num_channels:
                raise ValueError(
                    "Invalid tensor image shape. Tensor inputs to `predict()` must be in (C, H, W) format "
                    f"with C matching the model configuration ({self.model_config.num_channels} channels). "
                    f"Received tensor with shape {tuple(img.shape)}. "
                    "For automatic RGB conversion, pass a PIL Image or a file path instead of a tensor."
                )
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            processed_images.append(img_tensor.to(self.model.device))

        resize_to = list(shape) if shape is not None else [self.model.resolution, self.model.resolution]
        # antialias=False matches the antialias-free bilinear resize (cv2.INTER_LINEAR)
        # used by Albumentations during training — see issue #1203.
        batch_tensor = torch.stack([F.resize(t, resize_to, antialias=False) for t in processed_images])
        batch_tensor = F.normalize(batch_tensor, self.means, self.stds)

        if self._is_optimized_for_inference:
            if (
                self._optimized_resolution != batch_tensor.shape[2]
                or self._optimized_resolution != batch_tensor.shape[3]
            ):
                # this could happen if someone manually changes self.model.resolution after optimizing the model,
                # or if predict(shape=...) is used with a shape that doesn't match the compiled square resolution.
                _restore_hint = (
                    " Create a new RFDETR instance to use a different resolution."
                    if getattr(self, "_optimized_inplace", False)
                    else " You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                )
                raise ValueError(
                    f"Resolution mismatch. "
                    f"Model was optimized for resolution {self._optimized_resolution}x{self._optimized_resolution}, "
                    f"but got {batch_tensor.shape[2]}x{batch_tensor.shape[3]}." + _restore_hint,
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    _restore_hint = (
                        " Create a new RFDETR instance to recompile for a different batch size."
                        if getattr(self, "_optimized_inplace", False)
                        else (
                            " You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                            " Alternatively, you can recompile the optimized model for a different batch size"
                            " by calling model.inference(batch_size=<new_batch_size>)."
                        )
                    )
                    raise ValueError(
                        f"Batch size mismatch. "
                        f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                        f"but got {batch_tensor.shape[0]}." + _restore_hint,
                    )

        if self._is_optimized_for_inference:
            inference_model = self.model.inference_model
            assert inference_model is not None, "inference_model is set whenever _is_optimized_for_inference is True."
            predictions = inference_model(batch_tensor.to(dtype=self._optimized_dtype))
        else:
            model = self.model.model
            assert model is not None, "self.model.model is only cleared when optimized for inference."
            predictions = model(batch_tensor)
        if isinstance(predictions, tuple):
            return_predictions = {
                "pred_logits": predictions[1],
                "pred_boxes": predictions[0],
            }
            if len(predictions) == 3:
                # Distinguish optional keypoint vs mask tuple output for legacy compiled/export shims.
                if getattr(getattr(self.model, "args", None), "use_grouppose_keypoints", False):
                    return_predictions["pred_keypoints"] = predictions[2]
                else:
                    return_predictions["pred_masks"] = predictions[2]
            predictions = return_predictions
        target_sizes = torch.tensor(orig_sizes, device=self.model.device)
        results = self.model.postprocess(predictions, target_sizes=target_sizes, score_threshold=threshold)

        model_class_names = self.class_names
        n = len(model_class_names)
        # Pretrained COCO models use COCO category IDs (1–90, with gaps) as class_ids,
        # while class_names is a flat 0-indexed list of 80 entries. Detected when
        # args.num_classes > len(class_names) AND class_names == COCO_CLASS_NAMES.
        # Fine-tuned models remap category IDs to 0-based contiguous indices, so
        # class_id i maps directly to class_names[i].
        _model_args = getattr(self.model, "args", None)
        if _model_args is None and model_class_names == list(COCO_CLASS_NAMES):
            logger.warning_once(
                "predict(): model has no 'args' attribute — COCO sparse-ID mapping cannot activate; "
                "class_ids are treated as 0-indexed (may be wrong for pretrained COCO checkpoints)"
            )
        num_logit_slots: int = getattr(_model_args, "num_classes", n)
        _is_coco_pretrained = num_logit_slots > n and model_class_names == list(COCO_CLASS_NAMES)
        # Legacy keypoint models may use a shifted class scheme: slot 0 = background
        # (0 keypoints), real classes start at slot 1. Active-first schemas such as
        # [17] use normal 0-based class IDs and fall through to the default mapping.
        _num_keypoints_per_class: list[int] = getattr(_model_args, "num_keypoints_per_class", []) or []
        _is_legacy_bgfirst_keypoint = _is_bg_first_schema(_num_keypoints_per_class)
        if _is_coco_pretrained:
            _class_id_to_name: dict[int, str] = {
                coco_id: model_class_names[i] for i, coco_id in enumerate(COCO_CLASSES) if i < n
            }
        elif _is_legacy_bgfirst_keypoint:
            # Map foreground keypoint slots (slots where num_keypoints > 0) to class names.
            # Slot 0 is background and is skipped. Slot 1 → class_names[0], slot 2 → class_names[1], …
            # Note: slots where num_keypoints == 0 but slot != 0 (detect-only classes in a mixed schema
            # such as [0, 17, 0, 4]) are not present in _kp_foreground_slots and will map to an empty
            # string with a one-time warning. Mixed keypoint+detection schemas are not a supported
            # configuration for the shipped models.
            _kp_foreground_slots = [idx for idx, k in enumerate(_num_keypoints_per_class) if k > 0]
            _class_id_to_name = {slot: model_class_names[i] for i, slot in enumerate(_kp_foreground_slots) if i < n}
        else:
            _class_id_to_name = dict(enumerate(model_class_names))
        predictions_list: list[Detections | KeyPoints] = []
        for i, result in enumerate(results):
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            # INVARIANT: this predicate must stay identical (same operator and threshold) to the
            # pre-filter in PostProcess._postprocess_masks (`scores_i > score_threshold`), which is
            # fed `score_threshold=threshold` above. The seg path drops below-threshold masks before
            # upsampling on the strength of that match; diverging here (e.g. `>=`, per-class, top-k)
            # would make it silently drop rows this filter keeps — a behaviour change with no failing test.
            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]
            keypoints_array = None
            if "keypoints" in result:
                keypoints = result["keypoints"][keep]
                keypoints_array = keypoints.float().cpu().numpy()
            has_keypoints = keypoints_array is not None

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )
            if "keypoint_precision_cholesky" in result:
                keypoint_precision = result["keypoint_precision_cholesky"][keep]
                detections.data["keypoint_precision_cholesky"] = keypoint_precision.float().cpu().numpy()

            if include_source_image:
                detections.metadata["source_image"] = source_images[i]  # type: ignore[index]
            detections.data["source_shape"] = np.tile(np.array(orig_sizes[i], dtype=np.int64), (len(detections), 1))

            # Attach class names so callers can map class_id → name without a
            # separate lookup. Always set data["class_name"] for a consistent interface.
            #
            # For fine-tuned models, logit index num_logit_slots is the no-object slot —
            # map it to "__background__" without warning. For COCO-pretrained models,
            # background is implicit (filtered by threshold); class ID 90 is "toothbrush".
            # IDs not in _class_id_to_name are genuinely unexpected and produce an empty
            # string with a one-time warning.
            class_ids = detections.class_id if detections.class_id is not None else np.array([], dtype=int)
            # Sentinel for the no-object / background class differs by model type.
            # Legacy background-first keypoint models: slot 0 is background in the keypoint schema.
            # Detection/segmentation models: the no-object slot is at index num_logit_slots.
            _bg_sentinel = 0 if _is_legacy_bgfirst_keypoint else num_logit_slots
            truly_oob = [cid for cid in class_ids if cid not in _class_id_to_name and cid != _bg_sentinel]
            if truly_oob:
                logger.warning_once(
                    "predict() encountered unmapped class_id(s): %s — mapping to empty string",
                    truly_oob[:5],
                )
            if _is_coco_pretrained:
                class_names = [_class_id_to_name.get(cid, "") for cid in class_ids]
            else:
                class_names = [
                    "__background__" if cid == _bg_sentinel else _class_id_to_name.get(cid, "") for cid in class_ids
                ]
            detections.data["class_name"] = np.array(class_names, dtype=object)

            if has_keypoints and keypoints_array is not None:
                keypoint_data = dict(detections.data)
                keypoint_data["xyxy"] = detections.xyxy.astype(np.float32)
                if include_source_image:
                    keypoint_data["source_image"] = [
                        source_images[i]  # type: ignore[index]
                        for _ in range(len(detections))
                    ]
                raw_precision = keypoint_data.get("keypoint_precision_cholesky")
                raw_source_shape = keypoint_data.get("source_shape")
                if raw_precision is not None and raw_source_shape is not None and len(detections) > 0:
                    precision = np.asarray(raw_precision, dtype=np.float32)
                    source_shape = np.asarray(raw_source_shape, dtype=np.float32)
                    if precision.shape[:2] == keypoints_array.shape[:2] and source_shape.shape == (len(detections), 2):
                        keypoint_data["covariance"] = precision_cholesky_to_pixel_covariance(
                            precision_cholesky=precision, source_shape=source_shape
                        )
                keypoints_array = keypoints_array.astype(np.float32, copy=False)
                keypoint_confidence = keypoints_array[:, :, 2]
                key_points = KeyPoints(
                    xy=keypoints_array[:, :, :2],
                    keypoint_confidence=keypoint_confidence,
                    detection_confidence=detections.confidence.astype(np.float32)
                    if detections.confidence is not None
                    else None,
                    class_id=detections.class_id.astype(int) if detections.class_id is not None else None,
                    visible=keypoint_confidence > 0,
                    data=keypoint_data,
                )
                predictions_list.append(key_points)
            else:
                predictions_list.append(detections)

        return predictions_list[0] if single_input else predictions_list

    def deploy_to_roboflow(
        self,
        workspace: str,
        project_id: str,
        version: int | str,
        api_key: str | None = None,
        size: str | None = None,
    ) -> None:
        """Deploy the trained RF-DETR model to Roboflow.

        Deploying with Roboflow will create a Serverless API to which you can make requests.

        You can also download weights into a Roboflow Inference deployment for use in Roboflow Workflows and on-device
        deployment.

        Args:
            workspace: The name of the Roboflow workspace to deploy to.
            project_id: The project ID to which the model will be deployed.
            version: The project version to which the model will be deployed.
            api_key: Your Roboflow API key. If not provided,
                it will be read from the environment variable `ROBOFLOW_API_KEY`.
            size: The size of the model to deploy. If not provided,
                it will default to the size of the model being trained (e.g., "rfdetr-base", "rfdetr-large", etc.).

        Raises:
            ValueError: If the `api_key` is not provided and not found in the
                environment variable `ROBOFLOW_API_KEY`, or if the `size` is not set for custom architectures.
            RuntimeError: If the model was cleared by ``inference(inplace=True)``.

        Note:
            Bundle creation is delegated to :meth:`export_for_roboflow`, which can be called independently
            to write ``weights.pt`` and ``class_names.txt`` without a network round-trip.
        """
        if getattr(self, "_optimized_inplace", False) or self.model.model is None:
            raise RuntimeError(
                "Cannot deploy after inference(inplace=True) — "
                "the model weights have been cleared from memory. "
                "Call export_for_roboflow() before optimizing, then deploy the exported bundle."
            )

        from roboflow import Roboflow

        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")

        rf = Roboflow(api_key=api_key)
        rf_workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        if size is not None and self.size is not None and size != self.size:
            warnings.warn(
                f"deploy_to_roboflow(size={size!r}) overrides this model's own size {self.size!r}; "
                f"deploying as {size!r}. Omit size to deploy with the model's own size.",
                UserWarning,
                stacklevel=2,
            )
        # Explicit user argument wins; fall back to the trained model's size (documented behaviour).
        size = self.size if size is None else size
        with tempfile.TemporaryDirectory(prefix="roboflow_upload_") as tmp_out_dir:
            self.export_for_roboflow(tmp_out_dir)
            project = rf_workspace.project(project_id)
            project_version = project.version(version)
            project_version.deploy(model_type=size, model_path=tmp_out_dir, filename="weights.pt")

    def export_for_roboflow(self, output_dir: str | os.PathLike[str]) -> None:
        """Write a Roboflow upload bundle (``weights.pt`` + ``class_names.txt``) into *output_dir*.

        This is the network-free core of :meth:`deploy_to_roboflow`: it serialises the model state and
        a sanitized copy of the training args into ``weights.pt``, always embedding ``class_names`` so
        the bundle is self-contained, and writes the class labels to ``class_names.txt``.  The Roboflow
        SDK uses this format to adapt raw PyTorch-Lightning checkpoints into a deploy-ready bundle.

        Args:
            output_dir: Directory into which ``weights.pt`` and ``class_names.txt`` are written.  Created
                if it does not exist.  Existing files are silently overwritten.

        Raises:
            PermissionError: If the process lacks write access to *output_dir* or its parent directory.
            OSError: On disk-full, invalid path, or other filesystem failure during directory creation,
                file write, or ``torch.save``.
            RuntimeError: If the model was cleared by ``inference(inplace=True)``.
        """
        if getattr(self, "_optimized_inplace", False) or self.model.model is None:
            raise RuntimeError(
                "Cannot export after inference(inplace=True) — "
                "the model has been cleared from memory. "
                "Call export_for_roboflow() before optimizing."
            )
        os.makedirs(output_dir, exist_ok=True)
        # Write class_names.txt so the Roboflow upload pipeline can discover
        # the class labels without relying on args.class_names in the checkpoint.
        class_names_path = os.path.join(output_dir, "class_names.txt")
        with open(class_names_path, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(self.class_names))

        # Serialize a sanitized copy so trained bundles do not expose the caller's
        # filesystem layout.  Keep self.model.args unchanged for runtime consumers.
        args = copy(self.model.args)
        args.dataset_dir = None
        args.output_dir = "output"
        args.resume = ""
        if not hasattr(args, "class_names") or args.class_names is None:
            args.class_names = self.class_names

        outpath = os.path.join(output_dir, "weights.pt")
        torch.save({"model": self.model.model.state_dict(), "args": args}, outpath)


def __getattr__(name: str) -> Any:
    """Lazily resolve legacy re-exports without creating import-order cycles."""
    if name in _VARIANT_EXPORTS:
        module = importlib.import_module("rfdetr.variants")
        value = getattr(module, name)
        globals()[name] = value
        return value

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazy re-exports in interactive discovery."""
    return sorted(set(globals()) | set(_VARIANT_EXPORTS))
