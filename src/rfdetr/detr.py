# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
from __future__ import annotations

import contextlib
import functools
import glob
import json
import operator
import os
import warnings
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import requests
import torch

if TYPE_CHECKING:
    import supervision as sv
import torchvision.transforms.functional as F
import yaml
from PIL import Image

from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
from rfdetr.assets.model_weights import download_pretrain_weights, validate_pretrain_weights
from rfdetr.config import (
    ModelConfig,
    RFDETRBaseConfig,  # DEPRECATED
    RFDETRLargeConfig,
    RFDETRLargeDeprecatedConfig,  # DEPRECATED
    RFDETRMediumConfig,
    RFDETRNanoConfig,
    RFDETRSeg2XLargeConfig,
    RFDETRSegLargeConfig,
    RFDETRSegMediumConfig,
    RFDETRSegNanoConfig,
    RFDETRSegPreviewConfig,  # DEPRECATED
    RFDETRSegSmallConfig,
    RFDETRSegXLargeConfig,
    RFDETRSmallConfig,
    SegmentationTrainConfig,
    TrainConfig,
)
from rfdetr.datasets.coco import is_valid_coco_dataset
from rfdetr.datasets.yolo import is_valid_yolo_dataset
from rfdetr.models import PostProcess, build_model
from rfdetr.utilities.decorators import deprecated
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.state_dict import _ckpt_args_get, validate_checkpoint_compatibility

try:
    torch.set_float32_matmul_precision("high")
except Exception:
    pass

logger = get_logger()


class ModelContext:
    """Lightweight model wrapper returned by RFDETR.get_model().

    Provides the same attribute interface as the legacy ``main.py:Model`` but
    without importing or depending on ``populate_args()`` or the legacy stack.

    Args:
        model: The underlying ``nn.Module`` (LWDETR instance).
        postprocess: PostProcess instance for converting raw outputs to boxes.
        device: Device the model lives on.
        resolution: Input resolution (square side length in pixels).
        args: Namespace produced by :func:`build_namespace`.
        class_names: Optional list of class name strings loaded from checkpoint.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        postprocess: PostProcess,
        device: torch.device,
        resolution: int,
        args: Any,
        class_names: Optional[List[str]] = None,
    ) -> None:
        self.model = model
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
        """
        self.model.reinitialize_detection_head(num_classes)
        self.args.num_classes = num_classes


_ModelContext = ModelContext  # backward compat alias


def _load_pretrain_weights_into(nn_model: torch.nn.Module, args: Any) -> List[str]:
    """Load pretrained checkpoint weights into *nn_model* in-place.

    Mirrors ``Model.__init__`` and ``RFDETRModelModule._load_pretrain_weights``
    checkpoint loading logic: validates hash, re-downloads on corruption, and
    trims query embeddings to match the configured query count.

    Args:
        nn_model: The model to load weights into.
        args: Namespace with ``pretrain_weights``, ``num_classes``,
            ``num_queries``, and ``group_detr`` attributes.

    Returns:
        List of class names extracted from the checkpoint, or empty list.
    """
    class_names: List[str] = []

    download_pretrain_weights(args.pretrain_weights)
    if not os.path.isfile(args.pretrain_weights):
        logger.warning("Pretrain weights not found after initial download; retrying without MD5 validation.")
        download_pretrain_weights(args.pretrain_weights, redownload=True, validate_md5=False)
    validate_pretrain_weights(args.pretrain_weights, strict=False)

    try:
        checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)
    except Exception:
        logger.info("Failed to load pretrain weights, re-downloading")
        download_pretrain_weights(args.pretrain_weights, redownload=True, validate_md5=False)
        checkpoint = torch.load(args.pretrain_weights, map_location="cpu", weights_only=False)

    if "args" in checkpoint:
        class_names = _ckpt_args_get(checkpoint["args"], "class_names") or []

    validate_checkpoint_compatibility(checkpoint, args)

    user_set_num_classes = "num_classes" in getattr(args, "model_fields_set", set())
    default_num_classes = getattr(type(args), "model_fields", {}).get("num_classes")
    default_num_classes = getattr(default_num_classes, "default", 90)
    num_classes = args.num_classes
    user_overrode_default_num_classes = user_set_num_classes and num_classes != default_num_classes

    checkpoint_num_classes = checkpoint["model"]["class_embed.bias"].shape[0]
    configured_num_classes_plus_bg = num_classes + 1
    if checkpoint_num_classes != configured_num_classes_plus_bg:
        if checkpoint_num_classes < configured_num_classes_plus_bg and not user_overrode_default_num_classes:
            num_classes = checkpoint_num_classes - 1
            configured_num_classes_plus_bg = checkpoint_num_classes
            args.num_classes = num_classes
        # Temporarily align the detection head size with the checkpoint so
        # that state_dict loading succeeds even when the configured
        # num_classes differs from the checkpoint.
        nn_model.reinitialize_detection_head(checkpoint_num_classes)

    num_desired_queries = args.num_queries * args.group_detr
    query_param_names = ["refpoint_embed.weight", "query_feat.weight"]
    for name in list(checkpoint["model"].keys()):
        if any(name.endswith(x) for x in query_param_names):
            checkpoint["model"][name] = checkpoint["model"][name][:num_desired_queries]

    nn_model.load_state_dict(checkpoint["model"], strict=False)

    # Only reinitialize back to configured size when intentionally reducing a
    # larger pretrain checkpoint to fewer task-specific classes.
    if checkpoint_num_classes < configured_num_classes_plus_bg and user_overrode_default_num_classes:
        nn_model.reinitialize_detection_head(configured_num_classes_plus_bg)

    if num_classes + 1 < checkpoint_num_classes:
        nn_model.reinitialize_detection_head(num_classes + 1)

    return class_names


def _apply_lora_to(nn_model: torch.nn.Module) -> None:
    """Apply LoRA adapters to the backbone encoder of *nn_model*.

    Args:
        nn_model: LWDETR model whose backbone encoder will receive LoRA.
    """
    from peft import LoraConfig, get_peft_model

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        use_dora=True,
        target_modules=[
            "q_proj",
            "v_proj",
            "k_proj",
            "qkv",
            "query",
            "key",
            "value",
            "cls_token",
            "register_tokens",
        ],
    )
    nn_model.backbone[0].encoder = get_peft_model(nn_model.backbone[0].encoder, lora_config)


def _build_model_context(model_config: ModelConfig) -> ModelContext:
    """Build a ModelContext from ModelConfig without using legacy main.py:Model.

    Replicates ``Model.__init__`` logic: builds the nn.Module, optionally loads
    pretrain weights and applies LoRA.  The model is intentionally kept on CPU;
    :func:`_ensure_model_on_device` in ``detr.py`` performs the deferred
    ``.to(device)`` on the first ``predict()`` / ``export()`` /
    ``optimize_for_inference()`` call.  Keeping construction CPU-only prevents
    CUDA initialisation during ``__init__``, which would block DDP strategies
    (``ddp_notebook``, ``ddp_spawn``) from spawning child processes in notebook
    environments.

    Args:
        model_config: Architecture configuration.

    Returns:
        ModelContext with the model on CPU, ready for lazy device placement.
    """
    from rfdetr._namespace import build_namespace

    # A dummy TrainConfig is needed only for build_namespace's required fields;
    # dataset_dir/output_dir are unused during model construction.
    args = build_namespace(model_config, TrainConfig(dataset_dir=".", output_dir="."))
    nn_model = build_model(args)

    class_names: List[str] = []
    if args.pretrain_weights is not None:
        class_names = _load_pretrain_weights_into(nn_model, args)

    if args.backbone_lora:
        _apply_lora_to(nn_model)

    device = torch.device(args.device)
    # Keep the model on CPU here; predict() / export() / optimize_for_inference()
    # will lazily move it to the target device on first use.  Eagerly calling
    # .to("cuda") would initialise the CUDA runtime during __init__(), which
    # prevents DDP strategies (ddp_notebook, ddp_spawn) from forking/spawning
    # child processes in notebook environments.
    postprocess = PostProcess(num_select=args.num_select)

    return ModelContext(
        model=nn_model,
        postprocess=postprocess,
        device=device,
        resolution=model_config.resolution,
        args=args,
        class_names=class_names or None,
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


def _ensure_model_on_device(model_ctx: Any) -> None:
    """Move model weights to the target device recorded in *model_ctx*.

    ``_build_model_context`` intentionally keeps the ``nn.Module`` on CPU so
    that ``RFDETR.__init__`` does not initialise CUDA (which would prevent DDP
    strategies from forking in notebook environments).  This helper performs
    the deferred ``.to(device)`` on first use.

    It is safe to call on duck-typed stand-ins (e.g. ``SimpleNamespace``); the
    function silently returns when the expected attributes are missing.
    """
    target = getattr(model_ctx, "device", None)
    inner = getattr(model_ctx, "model", None)
    if target is None or inner is None or not hasattr(inner, "parameters"):
        return
    if isinstance(target, str):
        target = torch.device(target)
    first_param = next(inner.parameters(), None)
    if first_param is not None and first_param.device != target:
        model_ctx.model = inner.to(target)


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
        a :class:`~rfdetr.config.TrainConfig`.  Several kwargs are absorbed and
        handled specially so that existing call-sites do not break:

        * ``resolution`` — updates the model's input resolution by mutating
          :attr:`model_config.resolution` in place before the train config is
          built. This change persists on :attr:`model_config` after
          :meth:`train` returns. The value must be a positive integer divisible
          by ``patch_size * num_windows`` for the model variant; a
          :class:`ValueError` is raised otherwise.
          :attr:`model_config.positional_encoding_size` is also updated when
          the config derives it formulaically (``PE == resolution //
          patch_size``); configs with a pretrained-specific PE value (e.g.
          ``RFDETRBase`` uses DINOv2's PE=37 at 560 px) are left unchanged to
          preserve checkpoint compatibility.
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
            ValueError: If ``resolution`` is not a positive integer or is not
                divisible by ``patch_size * num_windows`` for the model variant.

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
            block_size = self.model_config.patch_size * self.model_config.num_windows
            if _resolution % block_size != 0:
                raise ValueError(
                    f"resolution={_resolution} is not divisible by "
                    f"patch_size ({self.model_config.patch_size}) * num_windows "
                    f"({self.model_config.num_windows}) = {block_size}. "
                    f"Choose a resolution that is a multiple of {block_size}."
                )
            # Smart PE update: only recompute positional_encoding_size when the
            # current config derives it formulaically (PE == resolution // patch_size).
            # Configs with a pretrained-specific PE (e.g. RFDETRBase uses DINOv2's
            # PE=37 at 518 px, training at 560 px) must not have PE silently changed
            # — doing so causes shape mismatches when loading pretrained checkpoints.
            _current_pe = self.model_config.positional_encoding_size
            _derived_pe = self.model_config.resolution // self.model_config.patch_size
            if _current_pe == _derived_pe:
                # Formula-derived: update PE proportionally to the new resolution.
                new_pe = _resolution // self.model_config.patch_size
                self.model_config.positional_encoding_size = new_pe
            else:
                # Pretrained-specific PE; leave it unchanged.
                new_pe = _current_pe
            self.model_config.resolution = _resolution

            # Keep the cached inference/export context in sync with model_config so
            # predict()/export()/deployment all see the same resolution metadata.
            if hasattr(self, "model") and self.model is not None:
                if hasattr(self.model, "resolution"):
                    self.model.resolution = _resolution
                model_args = getattr(self.model, "args", None)
                if model_args is not None:
                    if hasattr(model_args, "resolution"):
                        model_args.resolution = _resolution
                    if hasattr(model_args, "positional_encoding_size"):
                        model_args.positional_encoding_size = new_pe
        config = self.get_train_config(**kwargs)
        if config.batch_size == "auto":
            # Auto-batch probing runs forward/backward on the actual model, which
            # must be on the target device (typically CUDA).  Lazy placement keeps
            # the model on CPU until first use — move it now.
            _ensure_model_on_device(self.model)
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

        # Auto-detect num_classes from the training dataset and align model_config.
        # This must run before RFDETRModelModule is constructed so that weight loading
        # inside the module uses the correct (dataset-derived) class count.
        dataset_dir = getattr(config, "dataset_dir", None)
        if dataset_dir:
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

        self.remove_optimized_model()

        device = self.model.device
        # Clear any previously optimized state before starting a new optimization run.
        self.remove_optimized_model()

        _ensure_model_on_device(self.model)
        device = self.model.device
        cuda_ctx = torch.cuda.device(device) if device.type == "cuda" else contextlib.nullcontext()

        try:
            with cuda_ctx:
                self.model.inference_model = deepcopy(self.model.model)
                self.model.inference_model.eval()
                self.model.inference_model.export()

                self._optimized_resolution = self.model.resolution
                self._is_optimized_for_inference = True

                self.model.inference_model = self.model.inference_model.to(dtype=dtype)
                self._optimized_dtype = dtype

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
        args_mapping={
            "simplify": False,
            "force": False,
        },
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

    def get_model(self, config: ModelConfig) -> "ModelContext":
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

        return COCO_CLASS_NAMES

    def predict(
        self,
        images: str | Image.Image | np.ndarray | torch.Tensor | list[str | np.ndarray | Image.Image | torch.Tensor],
        threshold: float = 0.5,
        shape: tuple[int, int] | None = None,
        patch_size: int | None = None,
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

        _ensure_model_on_device(self.model)

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
        source_images = []

        for img in images:
            if isinstance(img, str):
                if img.startswith("http"):
                    img = requests.get(img, stream=True).raw
                img = Image.open(img)

            if not isinstance(img, torch.Tensor):
                src = np.array(img)
                if src.dtype != np.uint8:
                    src = (src * 255).clip(0, 255).astype(np.uint8)
                source_images.append(src)
                img = F.to_tensor(img)
            else:
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
        import shutil

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
        tmp_out_dir = ".roboflow_temp_upload"
        os.makedirs(tmp_out_dir, exist_ok=True)
        try:
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
        finally:
            shutil.rmtree(tmp_out_dir, ignore_errors=True)


class RFDETRBase(RFDETR):
    """
    Train an RF-DETR Base model (29M parameters).
    """

    size = "rfdetr-base"
    _model_config_class = RFDETRBaseConfig


class RFDETRNano(RFDETR):
    """
    Train an RF-DETR Nano model.
    """

    size = "rfdetr-nano"
    _model_config_class = RFDETRNanoConfig


class RFDETRSmall(RFDETR):
    """
    Train an RF-DETR Small model.
    """

    size = "rfdetr-small"
    _model_config_class = RFDETRSmallConfig


class RFDETRMedium(RFDETR):
    """
    Train an RF-DETR Medium model.
    """

    size = "rfdetr-medium"
    _model_config_class = RFDETRMediumConfig


class RFDETRLargeDeprecated(RFDETR):
    """
    Train an RF-DETR Large model.
    """

    size = "rfdetr-large"
    _model_config_class = RFDETRLargeDeprecatedConfig

    def __init__(self, **kwargs):
        warnings.warn(
            "RFDETRLargeDeprecated is deprecated and will be removed in a future version."
            " Please use RFDETRLarge instead.",
            category=DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(**kwargs)


class RFDETRLarge(RFDETR):
    size = "rfdetr-large"

    @staticmethod
    def _should_fallback_to_deprecated_config(exc: Exception) -> bool:
        """Return whether initialization should retry with deprecated Large config.

        The fallback is only for known checkpoint/config incompatibilities from
        deprecated Large weights. Runtime issues such as CUDA OOM must fail
        fast and must not trigger a second initialization attempt.

        Args:
            exc: Exception raised by initial ``RFDETR`` initialization.

        Returns:
            ``True`` when retrying with deprecated config is expected to help.
        """
        message = str(exc).lower()
        if "out of memory" in message:
            return False
        if isinstance(exc, ValueError):
            return "patch_size" in message
        if isinstance(exc, RuntimeError):
            incompatible_state_dict_markers = (
                "error(s) in loading state_dict",
                "size mismatch",
                "missing key(s) in state_dict",
                "unexpected key(s) in state_dict",
            )
            return any(marker in message for marker in incompatible_state_dict_markers)
        return False

    def __init__(self, **kwargs):
        self.init_error = None
        self.is_deprecated = False
        try:
            super().__init__(**kwargs)
        except (ValueError, RuntimeError) as exc:
            if not self._should_fallback_to_deprecated_config(exc):
                raise
            self.init_error = exc
            self.is_deprecated = True
            try:
                super().__init__(**kwargs)
                logger.warning(
                    "\n"
                    "=" * 100 + "\n"
                    "WARNING: Automatically switched to deprecated model configuration,"
                    " due to using deprecated weights."
                    " This will be removed in a future version.\n"
                    " Please retrain your model with the new weights and configuration.\n"
                    "=" * 100 + "\n"
                )
            except Exception:
                raise self.init_error

    def get_model_config(self, **kwargs) -> ModelConfig:
        if not self.is_deprecated:
            return RFDETRLargeConfig(**kwargs)
        else:
            return RFDETRLargeDeprecatedConfig(**kwargs)


class RFDETRSeg(RFDETR):
    """Base class for all RF-DETR segmentation models."""

    _train_config_class = SegmentationTrainConfig


class RFDETRSegPreview(RFDETRSeg):
    size = "rfdetr-seg-preview"
    _model_config_class = RFDETRSegPreviewConfig


class RFDETRSegNano(RFDETRSeg):
    size = "rfdetr-seg-nano"
    _model_config_class = RFDETRSegNanoConfig


class RFDETRSegSmall(RFDETRSeg):
    size = "rfdetr-seg-small"
    _model_config_class = RFDETRSegSmallConfig


class RFDETRSegMedium(RFDETRSeg):
    size = "rfdetr-seg-medium"
    _model_config_class = RFDETRSegMediumConfig


class RFDETRSegLarge(RFDETRSeg):
    size = "rfdetr-seg-large"
    _model_config_class = RFDETRSegLargeConfig


class RFDETRSegXLarge(RFDETRSeg):
    size = "rfdetr-seg-xlarge"
    _model_config_class = RFDETRSegXLargeConfig


class RFDETRSeg2XLarge(RFDETRSeg):
    size = "rfdetr-seg-2xlarge"
    _model_config_class = RFDETRSeg2XLargeConfig
