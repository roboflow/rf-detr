# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import functools
import glob
import importlib
import json
import operator
import os
import tempfile
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import requests
import torch

if TYPE_CHECKING:
    import supervision as sv

import torchvision.transforms.functional as F
import yaml
from PIL import Image

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from rfdetr.assets.model_weights import download_pretrain_weights
from rfdetr.config import (
    ModelConfig,
    TrainConfig,
)
from rfdetr.datasets.coco import is_valid_coco_dataset
from rfdetr.datasets.yolo import is_valid_yolo_dataset
from rfdetr.inference import ModelContext, _build_model_context
from rfdetr.utilities.decorators import deprecated
from rfdetr.utilities.distributed import is_main_process
from rfdetr.utilities.logger import get_logger

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = get_logger()

# ModelContext and _build_model_context are eagerly imported above (runtime use in get_model).
_VARIANT_EXPORTS = (
    "RFDETRBase",
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
            ``predict(shape=...)``).  Must be a two-element sequence of positive integers
            (or integer-compatible types accepted by :func:`operator.index`).
        block_size: Required divisor for both dimensions.  Equals ``patch_size * num_windows``.
        patch_size: Backbone patch size — used only in error messages.
        num_windows: Number of attention windows — used only in error messages.

    Returns:
        A ``(height, width)`` tuple of plain Python :class:`int` values.

    Raises:
        ValueError: If ``shape`` cannot be unpacked as a two-element sequence, if either
            dimension is a bool, float, or other non-integer type, if either dimension is
            not positive, or if either dimension is not divisible by ``block_size``.

    """
    try:
        height, width = shape  # type: ignore[misc]
    except (TypeError, ValueError):
        raise ValueError(f"shape must be a sequence of two positive integers (height, width), got {shape!r}.") from None
    for dim_name, dim in (("height", height), ("width", width)):
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
    height, width = operator.index(height), operator.index(width)
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
            positive integer attribute when ``patch_size`` is ``None`` or when a mismatch
            check is needed.
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


class RFDETR:
    """The base RF-DETR class implements the core methods for training RF-DETR models,
    running inference on the models, optimising models, and uploading trained
    models for deployment.
    """

    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    size = None
    _model_config_class: type[ModelConfig] = ModelConfig
    _train_config_class: type[TrainConfig] = TrainConfig

    def __init__(self, **kwargs):
        self.model_config = self.get_model_config(**kwargs)
        self.maybe_download_pretrain_weights()
        self.model = self.get_model(self.model_config)
        self.callbacks = defaultdict(list)

        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._has_warned_about_not_being_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    def maybe_download_pretrain_weights(self):
        """Download pre-trained weights if they are not already downloaded."""
        pretrain_weights = self.model_config.pretrain_weights
        if pretrain_weights is None:
            return
        download_pretrain_weights(pretrain_weights)

    def get_model_config(self, **kwargs) -> ModelConfig:
        """Retrieve the configuration parameters used by the model."""
        return self._model_config_class(**kwargs)

    @classmethod
    def from_checkpoint(cls, path: str | os.PathLike[str], **kwargs: Any) -> RFDETR:
        """Load an RF-DETR model from a training checkpoint, automatically
        inferring the model class.

        The correct subclass is resolved in order of preference:

        1. ``model_name`` key in the checkpoint (written by the PTL training
           stack since v1.7.0).
        2. ``pretrain_weights`` field in the checkpoint's ``args`` entry
           (legacy fallback).

        Both legacy ``argparse.Namespace`` checkpoints (produced by
        ``engine.py``) and dict-style checkpoints (produced by the PTL
        training stack) are supported.

        Args:
            path: Path to a checkpoint file (e.g. ``checkpoint_best_total.pth``).
            **kwargs: Additional keyword arguments forwarded to the model
                constructor (e.g. ``accept_platform_model_license=True`` for
                XLarge / 2XLarge models).

        Returns:
            An instance of the appropriate :class:`RFDETR` subclass loaded from
            the checkpoint.

        Warning:
            This method calls ``torch.load`` with ``weights_only=False``, which
            unpickles arbitrary Python objects. Only load checkpoints from
            trusted sources.

        Raises:
            FileNotFoundError: If *path* does not exist.
            OSError: If *path* exists but cannot be read.
            KeyError: If the checkpoint does not contain an ``"args"`` key.
            ValueError: If the model class cannot be inferred from
                ``model_name`` or ``pretrain_weights``.

        Examples:
            >>> model = RFDETR.from_checkpoint("checkpoint_best_total.pth")  # doctest: +SKIP
            >>> model = RFDETRSmall.from_checkpoint("checkpoint_best_total.pth")  # doctest: +SKIP
        """
        # Local import breaks the variants → detr import cycle.
        import rfdetr.variants as rfdetr_variants

        _plus_available = False
        _plus_symbols: dict[str, type[RFDETR]] = {}
        _plus_entries: list[tuple[str, type[RFDETR]]] = []
        try:
            import rfdetr.platform.models as platform_models

            for class_symbol in _CHECKPOINT_PLUS_MODEL_NAME_CLASS_SYMBOLS:
                plus_obj = getattr(platform_models, class_symbol)
                _plus_symbols[class_symbol] = plus_obj
            _plus_entries = [
                (name, _plus_symbols[class_symbol]) for name, class_symbol in _CHECKPOINT_PLUS_MODEL_MAP_ENTRIES
            ]
            _plus_available = True
        except (ImportError, AttributeError):
            _plus_symbols = {}

        # weights_only=False is required because legacy checkpoints embed
        # argparse.Namespace objects that cannot be deserialised with
        # weights_only=True.
        ckpt: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=False)
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
        _base_map: list[tuple[str, type[RFDETR]]] = [
            (name, _variant_symbols[class_symbol])
            for name, class_symbol in _CHECKPOINT_MODEL_MAP_ENTRIES
            if not name.startswith("seg-")
        ]
        _model_map: list[tuple[str, type[RFDETR]]] = _seg_map + _plus_entries + _base_map

        # New checkpoints store model_name directly — use it when available.
        _name_map: dict[str, type[RFDETR]] = dict(_variant_symbols)
        # Plus-model classes are resolved only when rfdetr_plus is installed.
        if _plus_available:
            _name_map.update(_plus_symbols)
        saved_model_name = ckpt.get("model_name")
        model_cls: type[RFDETR] | None = None
        if isinstance(saved_model_name, str):
            normalized_name = saved_model_name.strip()
            if normalized_name:
                model_cls = _name_map.get(normalized_name)
        else:
            normalized_name = ""

        # Fall back to pretrain_weights filename parsing for older checkpoints.
        if isinstance(args, dict):
            weights_name = str(args.get("pretrain_weights", "")).lower()
        else:
            weights_name = str(getattr(args, "pretrain_weights", "")).lower()

        if model_cls is None:
            # Guard: plus-only checkpoints should raise an actionable install error
            # when rfdetr_plus is missing, regardless of whether class inference
            # relies on model_name (new format) or pretrain_weights (legacy format).
            plus_by_model_name = normalized_name in _CHECKPOINT_PLUS_MODEL_NAME_CLASS_SYMBOLS
            plus_by_weights_name = "xlarge" in weights_name and "seg-" not in weights_name
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

        # pretrain_weights is placed after **kwargs so it always wins even if
        # a caller accidentally passes pretrain_weights inside kwargs.
        constructor_kwargs: dict[str, Any] = {**kwargs, "pretrain_weights": str(path)}
        if num_classes is not None and "num_classes" not in kwargs:
            constructor_kwargs["num_classes"] = num_classes

        return model_cls(**constructor_kwargs)

    @staticmethod
    def _resolve_trainer_device_kwargs(device: Any) -> tuple[str | None, list[int] | None]:
        """Map a torch-style device specifier to PTL ``accelerator``/``devices`` kwargs.

        Args:
            device: A device specifier accepted by ``torch.device``.

        Returns:
            ``(accelerator, devices)`` where ``devices`` is ``None`` unless an explicit
            device index is provided (for example ``cuda:1``).

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

        warnings.warn(
            f"Device type {resolved_device.type!r} is not explicitly mapped to a PyTorch Lightning "
            "accelerator; falling back to PTL auto-detection. Training may use an unexpected device.",
            UserWarning,
            stacklevel=2,
        )
        return None, None

    def train(self, **kwargs):
        """Train an RF-DETR model via the PyTorch Lightning stack.

        All keyword arguments are forwarded to :meth:`get_train_config` to build
        a :class:`~rfdetr.config.TrainConfig`.  Several legacy kwargs are absorbed
        so existing call-sites do not break:

        * ``device`` — normalized via :class:`torch.device` and mapped to PyTorch
          Lightning trainer arguments. ``"cpu"`` becomes ``accelerator="cpu"``;
          ``"cuda"`` and ``"cuda:N"`` become ``accelerator="gpu"`` and optionally
          ``devices=[N]``; ``"mps"`` becomes ``accelerator="mps"``. Other valid
          torch device types fall back to PTL auto-detection and emit a
          :class:`UserWarning`.
        * ``callbacks`` — if the dict contains any non-empty lists a
          :class:`DeprecationWarning` is emitted; the dict is then discarded.
          Use PTL :class:`~pytorch_lightning.Callback` objects passed via
          :func:`~rfdetr.training.build_trainer` instead.
        * ``start_epoch`` — emits :class:`DeprecationWarning` and is dropped.
        * ``do_benchmark`` — emits :class:`DeprecationWarning` and is dropped.

        After training completes the underlying ``nn.Module`` is synced back
        onto ``self.model.model`` so that :meth:`predict` and :meth:`export`
        continue to work without reloading the checkpoint.

        Raises:
            ImportError: If training dependencies are not installed. Install with
                ``pip install "rfdetr[train,loggers]"``.

        """
        # Both imports are grouped in a single try block because they both live in
        # the `rfdetr[train]` extras group — a missing `pytorch_lightning` (or any
        # other training-extras package) causes either import to fail, and the
        # remediation is identical: `pip install "rfdetr[train,loggers]"`.
        try:
            from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer
            from rfdetr.training.auto_batch import resolve_auto_batch_config
        except ModuleNotFoundError as exc:
            # Preserve internal import errors so packaging/regression issues in
            # rfdetr.* are not misreported as missing optional extras.
            if exc.name and exc.name.startswith("rfdetr."):
                raise
            raise ImportError(
                "RF-DETR training dependencies are missing. "
                'Install them with `pip install "rfdetr[train,loggers]"` and try again.',
            ) from exc

        # Absorb legacy `callbacks` dict — warn if non-empty, then discard.
        callbacks_dict = kwargs.pop("callbacks", None)
        if callbacks_dict and any(callbacks_dict.values()):
            warnings.warn(
                "Custom callbacks dict is not forwarded to PTL. Use PTL Callback objects instead.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Parse `device` kwarg and map it to PTL accelerator/devices.
        # Supports torch-style strings and torch.device (e.g. "cuda:1").
        _device = kwargs.pop("device", None)
        _accelerator, _devices = RFDETR._resolve_trainer_device_kwargs(_device)

        # Absorb legacy `start_epoch` — PTL resumes automatically via ckpt_path.
        if "start_epoch" in kwargs:
            warnings.warn(
                "`start_epoch` is deprecated and ignored; PTL resumes automatically via `resume`.",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.pop("start_epoch")

        # Pop `do_benchmark`; benchmarking via `.train()` is deprecated.
        run_benchmark = bool(kwargs.pop("do_benchmark", False))
        if run_benchmark:
            warnings.warn(
                "`do_benchmark` in `.train()` is deprecated; use `rfdetr benchmark`.",
                DeprecationWarning,
                stacklevel=2,
            )

        config = self.get_train_config(**kwargs)
        if config.batch_size == "auto":
            auto_batch = resolve_auto_batch_config(
                model_context=self.model,
                model_config=self.model_config,
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
        self.model_config.model_name = type(self).__name__

        # Auto-detect num_classes from the training dataset and align model_config.
        # This must run before RFDETRModelModule is constructed so that weight loading
        # inside the module uses the correct (dataset-derived) class count.
        dataset_dir = getattr(config, "dataset_dir", None)
        if dataset_dir:
            self._align_num_classes_from_dataset(dataset_dir)

        module = RFDETRModelModule(self.model_config, config)
        datamodule = RFDETRDataModule(self.model_config, config)
        trainer_kwargs = {"accelerator": _accelerator}
        if _devices is not None:
            trainer_kwargs["devices"] = _devices
        trainer = build_trainer(config, self.model_config, **trainer_kwargs)
        trainer.fit(module, datamodule, ckpt_path=config.resume or None)

        # Sync the trained weights back so predict() / export() see the updated model.
        self.model.model = module.model
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

    def optimize_for_inference(
        self, compile: bool = True, batch_size: int = 1, dtype: torch.dtype | str = torch.float32
    ) -> None:
        """Optimize the model for inference with optional JIT compilation and dtype casting.

        Operations are wrapped in the correct CUDA device context to prevent context
        leaks on multi-GPU setups. When ``compile=True`` the model is traced with
        ``torch.jit.trace`` using a dummy input of ``batch_size`` images at the
        model's current resolution.

        Args:
            compile: If ``True``, trace the model with ``torch.jit.trace`` to obtain
                a JIT-compiled ``ScriptModule``. Set to ``False`` for broader
                compatibility (e.g. models with dynamic control flow).
            batch_size: Number of images the traced model will be optimized for.
                Ignored when ``compile=False``.
            dtype: Target floating-point dtype for the inference model. Accepts a
                ``torch.dtype`` directly (e.g. ``torch.float16``) or its string name
                (e.g. ``"float16"``). Defaults to ``torch.float32``.

        Raises:
            TypeError: If ``dtype`` is not a ``torch.dtype``, or if ``dtype`` is a
                string that does not correspond to a valid ``torch.dtype`` attribute.

        Examples:
            >>> model = RFDETRNano()
            >>> model.optimize_for_inference(compile=False, dtype="float16", batch_size=4)
        """
        if isinstance(dtype, str):
            try:
                dtype = getattr(torch, dtype)
            except AttributeError:
                raise TypeError(f"dtype must be a torch.dtype or a string name of a dtype, got {dtype!r}") from None
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"dtype must be a torch.dtype or a string name of a dtype, got {type(dtype)!r}")

        # Clear any previously optimized state before starting a new optimization run.
        self.remove_optimized_model()

        device = self.model.device
        cuda_ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()

        try:
            with cuda_ctx:
                self.model.inference_model = deepcopy(self.model.model)
                self.model.inference_model.eval()
                self.model.inference_model.export()

                self.model.inference_model = self.model.inference_model.to(dtype=dtype)

                if compile:
                    self.model.inference_model = torch.jit.trace(
                        self.model.inference_model,
                        torch.randn(
                            batch_size,
                            3,
                            self.model.resolution,
                            self.model.resolution,
                            device=self.model.device,
                            dtype=dtype,
                        ),
                    )
                    self._optimized_has_been_compiled = True
                    self._optimized_batch_size = batch_size

                # Set success flags only after all operations complete.
                self._optimized_resolution = self.model.resolution
                self._is_optimized_for_inference = True
                self._optimized_dtype = dtype
        except Exception:
            # Ensure the object is left in a consistent, unoptimized state if optimization fails.
            with contextlib.suppress(Exception):
                self.remove_optimized_model()
            raise

    def remove_optimized_model(self) -> None:
        """Remove the optimized inference model and reset all optimization flags.

        Clears ``model.inference_model`` and resets all internal state set by
        :meth:`optimize_for_inference`. Safe to call even if the model has not
        been optimized.

        Examples:
            >>> model = RFDETRSmall()
            >>> model.optimize_for_inference(compile=False)
            >>> model.remove_optimized_model()
            >>> assert not model._is_optimized_for_inference
        """
        self.model.inference_model = None
        self._is_optimized_for_inference = False
        self._optimized_has_been_compiled = False
        self._optimized_batch_size = None
        self._optimized_resolution = None
        self._optimized_dtype = None

    @deprecated(
        target=True,
        # `simplify` / `force` are retained for API compatibility and treated as no-op.
        args_mapping={"simplify": False, "force": False},
        deprecated_in="1.6",
        remove_in="1.8",
        num_warns=1,
        stream=functools.partial(warnings.warn, category=DeprecationWarning, stacklevel=2),
    )
    def export(
        self,
        output_dir: str = "output",
        infer_dir: str = None,
        simplify: bool = False,
        backbone_only: bool = False,
        opset_version: int = 17,
        verbose: bool = True,
        force: bool = False,
        shape: tuple[int, int] | None = None,
        batch_size: int = 1,
        dynamic_batch: bool = False,
        patch_size: int | None = None,
        **kwargs,
    ) -> None:
        """Export the trained model to ONNX format.

        See the `ONNX export documentation <https://rfdetr.roboflow.com/learn/export/>`_
        for more information.

        Args:
            output_dir: Directory to write the ONNX file to.
            infer_dir: Optional directory of sample images for dynamic-axes inference.
            simplify: Deprecated and ignored. Simplification is no longer run.
            backbone_only: Export only the backbone (feature extractor).
            opset_version: ONNX opset version to target.
            verbose: Print export progress information.
            force: Deprecated and ignored.
            shape: ``(height, width)`` tuple; defaults to square at model resolution.
                Both dimensions must be divisible by ``patch_size * num_windows``.
            batch_size: Static batch size to bake into the ONNX graph.
            dynamic_batch: If True, export with a dynamic batch dimension
                so the ONNX model accepts variable batch sizes at runtime.
            patch_size: Backbone patch size. Defaults to the value stored in
                ``model_config.patch_size`` (typically 14 or 16). When provided
                explicitly it must match the instantiated model's patch size.
                Shape divisibility is validated against ``patch_size * num_windows``.
            **kwargs: Additional keyword arguments forwarded to export_onnx.

        """
        logger.info("Exporting model to ONNX format")
        try:
            from rfdetr.export.main import export_onnx, make_infer_image
        except ImportError:
            logger.error(
                "It seems some dependencies for ONNX export are missing."
                " Please run `pip install rfdetr[onnx]` and try again.",
            )
            raise

        device = self.model.device
        model = deepcopy(self.model.model.to("cpu"))
        model.to(device)

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

        input_tensors = make_infer_image(infer_dir, shape, batch_size, device).to(device)
        input_names = ["input"]
        if backbone_only:
            output_names = ["features"]
        elif self.model_config.segmentation_head:
            output_names = ["dets", "labels", "masks"]
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
            else:
                outputs = model(input_tensors)
                dets = outputs["pred_boxes"]
                labels = outputs["pred_logits"]
                logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")

        model.cpu()
        input_tensors = input_tensors.cpu()

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
        )

        logger.info(f"Successfully exported ONNX model to: {output_file}")

        logger.info("ONNX export completed successfully")
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

        # list all YAML files in the folder
        if is_valid_yolo_dataset(dataset_dir):
            yaml_paths = glob.glob(os.path.join(dataset_dir, "*.yaml")) + glob.glob(os.path.join(dataset_dir, "*.yml"))
            # any YAML file starting with data e.g. data.yaml, dataset.yaml
            yaml_data_files = [yp for yp in yaml_paths if os.path.basename(yp).startswith("data")]
            yaml_path = yaml_data_files[0]
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            if "names" in data:
                if isinstance(data["names"], dict):
                    return [data["names"][i] for i in sorted(data["names"].keys())]
                return data["names"]
            raise ValueError(f"Found {yaml_path} but it does not contain 'names' field.")
        raise FileNotFoundError(
            f"Could not find class names in {dataset_dir}."
            " Checked for COCO (train/_annotations.coco.json) and YOLO (data.yaml, data.yml) styles.",
        )

    @staticmethod
    def _detect_num_classes_for_training(dataset_dir: str) -> int:
        """Detect the class count using the same category basis as training labels.

        For COCO-style datasets this counts all categories by ``id`` from
        ``train/_annotations.coco.json`` (matching the remapping based on
        ``coco.cats`` used by the training datamodule). For YOLO-style datasets
        it falls back to ``_load_classes``.
        """
        if is_valid_coco_dataset(dataset_dir):
            coco_path = os.path.join(dataset_dir, "train", "_annotations.coco.json")
            with open(coco_path, encoding="utf-8") as f:
                anns = json.load(f)
            categories = anns["categories"]
            cat_by_id = {category["id"]: category for category in categories}
            return len(cat_by_id)

        return len(RFDETR._load_classes(dataset_dir))

    def _align_num_classes_from_dataset(self, dataset_dir: str) -> None:
        """Auto-detect the dataset class count and align ``model_config.num_classes`` in-place.

        Must be called before ``RFDETRModelModule`` is constructed so that weight loading inside
        the module uses the correct (dataset-derived) class count.

        When the user did **not** explicitly override ``num_classes`` (or passed the class-config
        default), ``model_config.num_classes`` and ``self.model.args.num_classes`` are updated
        to match the dataset.  When the user *did* set a non-default value that differs from the
        dataset, the configured value is preserved and a warning is emitted.

        Failures from ``_detect_num_classes_for_training`` are caught and logged at DEBUG level
        so that training is never blocked by detection errors.

        Args:
            dataset_dir: Path to the training dataset root directory.
        """
        try:
            dataset_num_classes = RFDETR._detect_num_classes_for_training(dataset_dir)
        except (FileNotFoundError, ValueError, KeyError, OSError) as exc:
            # Best-effort only; do not block training if detection fails.
            logger.debug("Could not auto-detect num_classes from dataset '%s': %s", dataset_dir, exc)
            return

        model_num_classes = self.model_config.num_classes

        if dataset_num_classes == model_num_classes:
            return

        # Determine whether the user explicitly overrode num_classes to a non-default value.
        # "num_classes" in model_fields_set is True when the field was explicitly set at
        # construction time; comparing against the class default filters out cases where the
        # user passed the default value explicitly (treat those like "not set").
        user_set = "num_classes" in getattr(self.model_config, "model_fields_set", set())
        default_nc = type(self.model_config).model_fields["num_classes"].default
        user_overrode = user_set and model_num_classes != default_nc

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

    def get_train_config(self, **kwargs) -> TrainConfig:
        """Retrieve the configuration parameters that will be used for training."""
        return self._train_config_class(**kwargs)

    def get_model(self, config: ModelConfig) -> ModelContext:
        """Retrieve a model context from the provided architecture configuration.

        Args:
            config: Architecture configuration.

        Returns:
            ModelContext with model, postprocess, device, resolution, args,
            and class_names attributes.

        """
        return _build_model_context(config)

    @property
    def class_names(self) -> list[str]:
        """Retrieve the class names supported by the loaded model.

        Returns:
            A list of class name strings, 0-indexed.  When no custom class
            names are embedded in the checkpoint, returns the standard 80
            COCO class names.

        """
        if hasattr(self.model, "class_names") and self.model.class_names is not None:
            return list(self.model.class_names)

        return list(COCO_CLASS_NAMES)

    def predict(
        self,
        images: str | Image.Image | np.ndarray | torch.Tensor | list[str | np.ndarray | Image.Image | torch.Tensor],
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
        patch_size: int | None = None,
        include_source_image: bool = True,
        **kwargs,
    ) -> sv.Detections | list[sv.Detections]:
        """Performs object detection on the input images and returns bounding box
        predictions.

        This method accepts a single image or a list of images in various formats
        (file path, image url, PIL Image, NumPy array, or torch.Tensor). The images should be in
        RGB channel order. If a torch.Tensor is provided, it must already be normalized
        to values in the [0, 1] range and have the shape (C, H, W).

        Args:
            images:
                A single image or a list of images to process. Images can be provided
                as file paths, PIL Images, NumPy arrays, or torch.Tensors.
            threshold:
                The minimum confidence score needed to consider a detected bounding box valid.
            shape:
                Optional ``(height, width)`` tuple to resize images to before inference.
                When provided, overrides the model's default inference resolution. The
                tuple should match the resolution used when exporting the model
                (typically a square shape). Both dimensions must be positive integers
                divisible by ``patch_size * num_windows``. Defaults to
                ``(model.resolution, model.resolution)`` when not set.
            patch_size:
                Backbone patch size used for shape divisibility validation. Defaults
                to ``model_config.patch_size`` (typically 14 for large models, 16 for
                smaller ones). Divisibility is checked against
                ``patch_size * num_windows``.
            include_source_image:
                Whether to attach the original image as ``source_image`` in
                ``detections.data``. Defaults to ``True`` for backward compatibility.
                Set to ``False`` to reduce memory use when source images are not needed.
            **kwargs:
                Additional keyword arguments.

        Returns:
            A single or multiple Detections objects, each containing bounding box
            coordinates, confidence scores, and class IDs.  The ``data`` dict of
            each :class:`~supervision.Detections` object contains:

            * ``"class_name"`` – ``np.ndarray`` of string class names corresponding
              to each detection (``class_names[class_id]``).  Class IDs are always
              0-indexed; ``class_names[0]`` is the first class regardless of the
              original dataset format (COCO category IDs are remapped to 0-based
              indices during training).
            * ``"source_image"`` – the original input image (only present when
              ``include_source_image=True``, which is the default).
            * ``"source_shape"`` – ``(height, width)`` tuple of the source image dimensions.

        Raises:
            ValueError: If ``shape`` cannot be unpacked as a two-element sequence,
                if either dimension does not support the ``__index__`` protocol
                (e.g. ``float``) or is a ``bool``, if either dimension is zero or
                negative, if either dimension is not divisible by
                ``patch_size * num_windows``, or if ``patch_size`` is not a positive
                integer.

        """
        import supervision as sv

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

        if not self._is_optimized_for_inference and not self._has_warned_about_not_being_optimized_for_inference:
            logger.warning(
                "Model is not optimized for inference. Latency may be higher than expected."
                " You can optimize the model for inference by calling model.optimize_for_inference().",
            )
            self._has_warned_about_not_being_optimized_for_inference = True

            self.model.model.eval()

        if not isinstance(images, list):
            images = [images]

        orig_sizes = []
        processed_images = []
        source_images = [] if include_source_image else None

        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    img = requests.get(img, stream=True).raw
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                if include_source_image:
                    src = np.array(img)
                    if src.dtype != np.uint8:
                        src = (src * 255).clip(0, 255).astype(np.uint8)
                    source_images.append(src)
                img = F.to_tensor(img)
            elif include_source_image:
                source_images.append((img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

            if (img > 1).any():
                raise ValueError(
                    "Image has pixel values above 1. Please ensure the image is normalized (scaled to [0, 1]).",
                )
            if (img < 0).any():
                raise ValueError(
                    "Image has pixel values below 0. Please ensure the image is normalized (scaled to [0, 1]).",
                )
            if img.shape[0] != 3:
                raise ValueError(f"Invalid image shape. Expected 3 channels (RGB), but got {img.shape[0]} channels.")
            img_tensor = img

            h, w = img_tensor.shape[1:]
            orig_sizes.append((h, w))

            img_tensor = img_tensor.to(self.model.device)
            resize_to = list(shape) if shape is not None else [self.model.resolution, self.model.resolution]
            img_tensor = F.resize(img_tensor, resize_to)
            img_tensor = F.normalize(img_tensor, self.means, self.stds)

            processed_images.append(img_tensor)

        batch_tensor = torch.stack(processed_images)

        if self._is_optimized_for_inference:
            if (
                self._optimized_resolution != batch_tensor.shape[2]
                or self._optimized_resolution != batch_tensor.shape[3]
            ):
                # this could happen if someone manually changes self.model.resolution after optimizing the model,
                # or if predict(shape=...) is used with a shape that doesn't match the compiled square resolution.
                raise ValueError(
                    f"Resolution mismatch. "
                    f"Model was optimized for resolution {self._optimized_resolution}x{self._optimized_resolution}, "
                    f"but got {batch_tensor.shape[2]}x{batch_tensor.shape[3]}."
                    " You can explicitly remove the optimized model by calling model.remove_optimized_model().",
                )
            if self._optimized_has_been_compiled:
                if self._optimized_batch_size != batch_tensor.shape[0]:
                    raise ValueError(
                        f"Batch size mismatch. "
                        f"Optimized model was compiled for batch size {self._optimized_batch_size}, "
                        f"but got {batch_tensor.shape[0]}."
                        " You can explicitly remove the optimized model by calling model.remove_optimized_model()."
                        " Alternatively, you can recompile the optimized model for a different batch size"
                        " by calling model.optimize_for_inference(batch_size=<new_batch_size>).",
                    )

        with torch.no_grad():
            if self._is_optimized_for_inference:
                predictions = self.model.inference_model(batch_tensor.to(dtype=self._optimized_dtype))
            else:
                predictions = self.model.model(batch_tensor)
            if isinstance(predictions, tuple):
                return_predictions = {
                    "pred_logits": predictions[1],
                    "pred_boxes": predictions[0],
                }
                if len(predictions) == 3:
                    return_predictions["pred_masks"] = predictions[2]
                predictions = return_predictions
            target_sizes = torch.tensor(orig_sizes, device=self.model.device)
            results = self.model.postprocess(predictions, target_sizes=target_sizes)

        model_class_names = self.class_names
        n = len(model_class_names)
        detections_list = []
        for i, result in enumerate(results):
            scores = result["scores"]
            labels = result["labels"]
            boxes = result["boxes"]

            keep = scores > threshold
            scores = scores[keep]
            labels = labels[keep]
            boxes = boxes[keep]

            if "masks" in result:
                masks = result["masks"]
                masks = masks[keep]

                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                    mask=masks.squeeze(1).cpu().numpy(),
                )
            else:
                detections = sv.Detections(
                    xyxy=boxes.float().cpu().numpy(),
                    confidence=scores.float().cpu().numpy(),
                    class_id=labels.cpu().numpy(),
                )

            if include_source_image:
                detections.data["source_image"] = source_images[i]
            detections.data["source_shape"] = orig_sizes[i]

            # Attach class names so callers can map class_id → name without a
            # separate lookup.  class_id is always 0-indexed regardless of the
            # original dataset format (COCO category IDs are remapped during
            # training), so class_names[class_id] is the correct mapping.
            # Always set data["class_name"] for a consistent interface.
            class_ids = detections.class_id if detections.class_id is not None else np.array([], dtype=int)
            oob_ids = [cid for cid in class_ids if not (0 <= cid < n)]
            if oob_ids:
                logger.warning_once(
                    "predict() encountered class_id values out of range [0, %d): %s — mapping to empty string",
                    n,
                    oob_ids[:5],
                )
            detections.data["class_name"] = np.array(
                [model_class_names[cid] if 0 <= cid < n else "" for cid in class_ids], dtype=object
            )

            detections_list.append(detections)

        return detections_list if len(detections_list) > 1 else detections_list[0]

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

        You can also download weights into a Roboflow Inference deployment for use in
        Roboflow Workflows and on-device deployment.

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
                environment variable `ROBOFLOW_API_KEY`, or if the `size` is
                not set for custom architectures.

        """
        from roboflow import Roboflow

        if api_key is None:
            api_key = os.getenv("ROBOFLOW_API_KEY")
            if api_key is None:
                raise ValueError("Set api_key=<KEY> in deploy_to_roboflow or export ROBOFLOW_API_KEY=<KEY>")

        rf = Roboflow(api_key=api_key)
        workspace = rf.workspace(workspace)

        if self.size is None and size is None:
            raise ValueError("Must set size for custom architectures")

        size = self.size or size
        with tempfile.TemporaryDirectory(prefix="roboflow_upload_") as tmp_out_dir:
            # Write class_names.txt so the Roboflow upload pipeline can discover
            # the class labels without relying on args.class_names in the checkpoint.
            class_names_path = os.path.join(tmp_out_dir, "class_names.txt")
            with open(class_names_path, "w", encoding="utf-8", newline="\n") as f:
                f.write("\n".join(self.class_names))

            # Also embed class_names in the args namespace so that any code path
            # that loads the checkpoint directly (e.g. roboflow-python's second
            # fallback) can find them.  Mutating the shared SimpleNamespace is
            # intentional here: this mirrors reinitialize_detection_head(), which
            # already mutates args.num_classes in-place.
            args = self.model.args
            if not hasattr(args, "class_names") or args.class_names is None:
                args.class_names = self.class_names

            outpath = os.path.join(tmp_out_dir, "weights.pt")
            torch.save({"model": self.model.model.state_dict(), "args": args}, outpath)
            project = workspace.project(project_id)
            project_version = project.version(version)
            project_version.deploy(model_type=size, model_path=tmp_out_dir, filename="weights.pt")


def __getattr__(name: str):
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
