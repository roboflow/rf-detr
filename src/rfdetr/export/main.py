# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Copied and modified from LW-DETR (https://github.com/Atten4Vis/LW-DETR)
# Copyright (c) 2024 Baidu. All Rights Reserved.
# ------------------------------------------------------------------------
"""CLI orchestrator for ONNX and TensorRT model export."""

from __future__ import annotations

import argparse
import os
import random
import warnings
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision.transforms.v2 import Compose, Resize, ToDtype, ToImage

from rfdetr.datasets.transforms import Normalize
from rfdetr.export._onnx.exporter import export_onnx
from rfdetr.export._tensorrt import build_engine
from rfdetr.models import BuilderArgs, build_model
from rfdetr.models.backbone.backbone import Backbone
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.distributed import get_rank
from rfdetr.utilities.logger import get_logger
from rfdetr.utilities.package import get_sha, get_version

logger = get_logger()

# Every format accepted by :meth:`rfdetr.detr.RFDETR.export`.
_EXPORT_FORMATS: frozenset[str] = frozenset({"onnx", "tflite", "tensorrt", "executorch"})
# The subset of :data:`_EXPORT_FORMATS` that specialize for a hardware backend, and so require a ``backend`` argument
# (the rest are backend-agnostic).  The accepted backends per format, and the backends that further require a ``soc``,
# are owned by the converter (``_VALID_BACKENDS`` / ``_SOC_BACKENDS``).
_BACKEND_FORMATS: frozenset[str] = frozenset({"executorch"})


def _resolve_export_backend(format: str, backend: str | None, soc: str | None) -> tuple[str | None, str | None]:
    """Validate a ``format`` / ``backend`` / ``soc`` combination and return the effective ``(backend, soc)``.

    Driven by the :data:`_EXPORT_FORMATS` / :data:`_BACKEND_FORMATS` registries (and the converter's backend/SoC sets)
    rather than per-format branches, so adding a format or backend is a data change:

    * A format not in :data:`_BACKEND_FORMATS` is backend-agnostic — it takes neither ``backend`` nor ``soc``;
      supplying one warns and it is ignored (returned ``None``).
    * A format in :data:`_BACKEND_FORMATS` requires ``backend`` to be one of the backends its converter accepts
      (looked up by format).
    * A backend that compiles for a specific chip (looked up by format+backend against the converter's SoC set)
      requires ``soc``; any other backend warns if a ``soc`` is supplied and ignores it.

    Args:
        format: Export format; one of :data:`_EXPORT_FORMATS`.
        backend: Requested hardware backend, or ``None``.
        soc: Requested target SoC, or ``None``.

    Returns:
        ``(backend, soc)`` with each value set to ``None`` when the format/backend does not use it.

    Raises:
        ValueError: On an unknown format, a missing or unknown required backend, or a missing required SoC.
    """
    if format not in _EXPORT_FORMATS:
        raise ValueError(f"Unsupported export format {format!r}. Choose from: {sorted(_EXPORT_FORMATS)}.")

    if format not in _BACKEND_FORMATS:
        # Backend-agnostic format: warn on any supplied (and therefore unused) backend/soc.
        for name, value in (("backend", backend), ("soc", soc)):
            if value is not None:
                warnings.warn(
                    f"`{name}={value!r}` is ignored for format={format!r}; this format does not require a hardware "
                    f"backend specialization.",
                    UserWarning,
                    stacklevel=3,
                )
        return None, None

    # Backend-bearing format: the converter owns the authoritative capability sets.  These are keyed by format
    # (accepted backends) and by format+backend (which backends require a ``soc``).  Adding a second backend-bearing
    # format is primarily a data change (add entries below + update _EXPORT_FORMATS / _BACKEND_FORMATS), but also
    # requires a lazy import and an elif branch in export().  Imported lazily so that backend-agnostic exports never
    # pull in the (optional, heavy) executorch dependency.
    from rfdetr.export._executorch.converter import SOC_BACKENDS, VALID_BACKENDS

    accepted_backends: dict[str, frozenset[str]] = {"executorch": VALID_BACKENDS}
    soc_backends: dict[str, frozenset[str]] = {"executorch": SOC_BACKENDS}
    valid = accepted_backends.get(format, frozenset())
    soc_required = soc_backends.get(format, frozenset())

    if backend is None:
        raise ValueError(f"format {format!r} requires a valid backend (one of {sorted(valid)}), but none was provided.")
    # Normalise case so RFDETR.export(backend="XNNPACK") and backend="xnnpack" behave identically.
    backend = backend.lower()
    if backend not in valid:
        raise ValueError(f"Unsupported backend {backend!r} for format {format!r}. Choose from: {sorted(valid)}.")

    if backend not in soc_required:
        if soc is not None:
            warnings.warn(
                f"`soc={soc!r}` is ignored for backend={backend!r}; this backend does not target a specific SoC.",
                UserWarning,
                stacklevel=3,
            )
        return backend, None

    if soc is None:
        raise ValueError(f"backend {backend!r} requires a valid soc, but none was provided.")
    return backend, soc


def _export_executorch_format(
    model: LWDETR,
    input_tensors: Tensor,
    output_dir_path: Path,
    *,
    backend: str | None,
    soc: str | None,
    variant_name: str | None,
    dynamic_batch: bool,
    notes: object,
) -> Path:
    """Dispatch :meth:`rfdetr.detr.RFDETR.export` to the ExecuTorch converter.

    Args:
        model: The prepared (CPU, export-mode) PyTorch module to export.
        input_tensors: Example input tensor used to trace the graph.
        output_dir_path: Directory where the ``.pte`` file is written.
        backend: ExecuTorch delegation backend; must not be ``None`` (invariant enforced by
            :func:`_resolve_export_backend`, which always sets a backend for ``format="executorch"``).
        soc: Target SoC for backends that compile for a specific chip, or ``None``.
        variant_name: Model variant identifier used to name the output file.
        dynamic_batch: Whether a dynamic batch dimension was requested (always rejected below).
        notes: User-supplied export metadata; ExecuTorch has no metadata slot, so a non-``None`` value warns.

    Returns:
        Path to the exported ``.pte`` file.

    Raises:
        RuntimeError: If ``backend`` is ``None`` (invariant violation — see Args).
        ImportError: If the optional ``executorch`` dependency is not installed.
    """
    if notes is not None:
        warnings.warn(
            "`notes` is not forwarded to format='executorch' (ExecuTorch .pte has no metadata slot). "
            "This argument is ignored.",
            UserWarning,
            stacklevel=3,
        )
    # Invariant: _resolve_export_backend always sets backend for executorch formats.
    if backend is None:  # pragma: no cover
        raise RuntimeError("backend must not be None for format='executorch' — invariant: _resolve_export_backend.")
    warnings.warn(
        "ExecuTorch export is experimental and work-in-progress.",
        UserWarning,
        stacklevel=3,
    )
    try:
        from rfdetr.export._executorch.converter import export_executorch
    except ImportError:
        logger.error(
            "It seems some dependencies for ExecuTorch export are missing."
            " Please run `pip install rfdetr[executorch]` and try again.",
        )
        raise
    # ExecuTorch consumes a torch.export graph directly, so switch the model into its
    # export-friendly forward (the ONNX path does this inside export_onnx).
    if hasattr(model, "export"):
        model.export()
    # soc only applies to the qnn backend; _resolve_export_backend leaves it None otherwise.
    soc_kwargs = {"soc": soc} if soc is not None else {}
    # _resolve_export_backend already validated backend against _VALID_BACKENDS at runtime;
    # narrow the static type to match export_executorch's Literal signature.

    backend_literal = cast(Literal["xnnpack", "coreml", "qnn"], backend)
    pte_path = export_executorch(
        model=model,
        input_tensors=input_tensors,
        output_dir=str(output_dir_path),
        backend=backend_literal,
        variant_name=variant_name,
        dynamic_batch=dynamic_batch,
        **soc_kwargs,
    )
    logger.info(f"Successfully exported ExecuTorch model to: {pte_path}")
    return pte_path


def _convert_onnx_export(
    output_file: str,
    format: str,
    output_dir_path: Path,
    *,
    quantization: str | None,
    calibration_data: str | np.ndarray[Any, Any] | None,
    max_images: int,
    verbose: bool,
) -> Path:
    """Dispatch :meth:`rfdetr.detr.RFDETR.export`'s post-ONNX-export conversion (TFLite / TensorRT / none).

    Args:
        output_file: Path to the already-exported ONNX model.
        format: Export format; one of ``"onnx"``, ``"tflite"``, ``"tensorrt"``.
        output_dir_path: Directory where converted artifacts are written.
        quantization: TFLite quantization mode (ignored for other formats).
        calibration_data: Representative images for TFLite INT8 calibration (ignored for other formats).
        max_images: Maximum calibration images to load from a directory (ignored for other formats).
        verbose: Print conversion progress information.

    Returns:
        Path to the final exported artifact (``.onnx``, ``.tflite``, or ``.trt``).
    """
    if format == "tflite":
        warnings.warn(
            "TFLite export is experimental and work-in-progress. "
            "Upstream dependency instabilities (onnx2tf, ai_edge_litert) may affect results.",
            UserWarning,
            stacklevel=3,
        )
        try:
            from rfdetr.export._tflite.converter import export_tflite
        except ImportError:
            logger.error(
                "It seems some dependencies for TFLite export are missing."
                " Please run `pip install rfdetr[onnx,tflite]` and try again.",
            )
            raise

        tflite_path = export_tflite(
            onnx_path=output_file,
            output_dir=str(output_dir_path),
            quantization=quantization,
            calibration_data=calibration_data,
            verbosity="info" if verbose else "error",
            max_images=max_images,
            verbose=verbose,
        )
        logger.info(f"Successfully exported TFLite model to: {tflite_path}")
        return tflite_path

    if format == "tensorrt":
        # Lazy re-import (not the module-level `build_engine` above): tests monkeypatch
        # rfdetr.export._tensorrt.build_engine, which only takes effect on a fresh lookup here.
        from rfdetr.export._tensorrt import build_engine

        logger.info("Converting ONNX model to TensorRT engine")
        engine_file = build_engine(output_file, verbose=verbose)
        logger.info(f"Successfully exported TensorRT engine to: {engine_file}")
        return Path(engine_file)

    logger.info("Export completed successfully")
    return Path(output_file)


def _num_parameters(module: nn.Module) -> int:
    """Count trainable and non-trainable parameters in a module."""
    return sum(int(p.numel()) for p in module.parameters())


def make_infer_image(
    infer_dir: str | None,
    shape: tuple[int, int],
    batch_size: int,
    device: str | torch.device = "cuda",
    num_channels: int = 3,
) -> Tensor:
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
        image = Image.open(infer_dir).convert("RGB")

    transforms = Compose(
        [
            Resize((shape[0], shape[1])),
            ToImage(),
            ToDtype(torch.float32, scale=True),
            Normalize(),
        ]
    )

    inps, _ = transforms(image, None)
    inps = inps.to(device)
    inps = torch.stack([inps for _ in range(batch_size)])
    return inps


def no_batch_norm(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            raise ValueError("BatchNorm2d found in the model. Please remove it.")


def main(args: argparse.Namespace) -> None:
    git_info = get_sha()
    if git_info != "unknown":
        logger.info(f"Running from git repository: {git_info}")
    else:
        version = get_version()
        logger.info(f"Running RF-DETR version: {version or 'unknown'}")
    logger.info(f"Export config: {vars(args)}")
    # convert device to device_id
    if args.device == "cuda":
        device_id = "0"
    elif args.device == "cpu":
        device_id = ""
    else:
        device_id = str(int(args.device))
        args.device = f"cuda:{device_id}"

    # device for export onnx
    # TODO: export onnx with cuda failed with onnx error
    device = torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # fix the seed for reproducibility
    seed: int = args.seed + get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    result = build_model(cast(BuilderArgs, args))
    model = cast(LWDETR, result[0] if isinstance(result, tuple) else result)
    backbone = model.backbone
    backbone_model = cast(Backbone, backbone[0])
    n_parameters = _num_parameters(model)
    logger.info(f"number of parameters: {n_parameters}")
    n_backbone_parameters = _num_parameters(backbone)
    logger.info(f"number of backbone parameters: {n_backbone_parameters}")
    n_projector_parameters = _num_parameters(backbone_model.projector)
    logger.info(f"number of projector parameters: {n_projector_parameters}")
    n_backbone_encoder_parameters = _num_parameters(backbone_model.encoder)
    logger.info(f"number of backbone encoder parameters: {n_backbone_encoder_parameters}")
    n_transformer_parameters = _num_parameters(cast(nn.Module, model.transformer))
    logger.info(f"number of transformer parameters: {n_transformer_parameters}")
    if args.resume:
        # --resume points at RF-DETR-produced checkpoints that may embed
        # argparse.Namespace objects requiring full-pickle deserialization.
        # Mirror RFDETR.from_checkpoint(trust_checkpoint=...): honour an optional
        # `trust_checkpoint` attribute on args (default True for backward
        # compatibility, since --resume is a developer/ops flag).  Setting
        # args.trust_checkpoint=False enforces safe-tensors-only loading and
        # raises if the checkpoint needs pickle.
        from rfdetr.utilities.io import _safe_torch_load

        trust_checkpoint = getattr(args, "trust_checkpoint", True)
        if trust_checkpoint:
            # Explicit pre-load warning: --resume has no CLI opt-in gate, so make
            # the pickle-deserialization risk (CWE-502) visible before loading.
            warnings.warn(
                f"Loading --resume checkpoint {args.resume!r} with full pickle "
                "deserialization (weights_only=False fallback). This can execute "
                "arbitrary code if the checkpoint comes from an untrusted source. "
                "Set args.trust_checkpoint=False to require safe-tensors-only loading.",
                UserWarning,
                stacklevel=2,
            )
        checkpoint = _safe_torch_load(args.resume, trust=trust_checkpoint)
        result = model.load_state_dict(checkpoint["model"], strict=False)
        if result.missing_keys or result.unexpected_keys:
            logger.warning(
                "load_state_dict strict=False: missing=%s unexpected=%s",
                result.missing_keys[:5],
                result.unexpected_keys[:5],
            )
        logger.info(f"load checkpoints {args.resume}")

    if args.layer_norm:
        no_batch_norm(model)

    model.to(device)

    input_tensors = make_infer_image(args.infer_dir, args.shape, args.batch_size, device)
    input_names = ["input"]
    if args.backbone_only:
        output_names = ["features"]
    elif args.segmentation_head:
        output_names = ["dets", "labels", "masks"]
    else:
        output_names = ["dets", "labels"]
    if getattr(args, "dynamic_batch", False):
        dynamic_axes = {name: {0: "batch"} for name in input_names + output_names}
    else:
        dynamic_axes = None
    # Run model inference in pytorch mode.
    # Use the export device (args.device) — not a hard-coded "cuda" — so that
    # CPU-only export paths work correctly.  Fall back to CPU when CUDA was
    # requested but is not available (e.g. in a CPU-only CI environment).
    run_device = torch.device(args.device)
    if run_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available; falling back to CPU for sanity forward pass.")
        run_device = torch.device("cpu")
    model.eval().to(run_device)
    input_tensors = input_tensors.to(run_device)
    with torch.no_grad():
        if args.backbone_only:
            features = model(input_tensors)
            logger.debug(f"PyTorch inference output shape: {features.shape}")
        elif args.segmentation_head:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            masks = outputs["pred_masks"]
            if isinstance(masks, torch.Tensor):
                logger.debug(
                    f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}, "
                    f"Masks: {masks.shape}"
                )
            else:
                # masks is a dict with spatial_features, query_features, bias
                logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
                logger.debug(
                    "Mask spatial_features: "
                    f"{masks['spatial_features'].shape}, "
                    f"query_features: {masks['query_features'].shape}, "
                    f"bias: {masks['bias'].shape}"
                )
        else:
            outputs = model(input_tensors)
            dets = outputs["pred_boxes"]
            labels = outputs["pred_logits"]
            logger.debug(f"PyTorch inference output shapes - Boxes: {dets.shape}, Labels: {labels.shape}")
    model.cpu()
    input_tensors = input_tensors.cpu()

    output_file = export_onnx(
        args.output_dir,
        model,
        input_names,
        input_tensors,
        output_names,
        dynamic_axes,
        backbone_only=args.backbone_only,
        verbose=args.verbose,
        opset_version=args.opset_version,
        variant_name=getattr(args, "variant_name", None),
    )

    onnx_path = output_file  # preserve ONNX path before any post-processing step overwrites it

    if args.tensorrt:
        output_file = build_engine(onnx_path, verbose=args.verbose, dry_run=args.dry_run)

    # TODO: register --tflite, --quantization, --calibration-data, --max-images in the
    # argparser to enable TFLite export via CLI.  Until then, use RFDETR.export(format="tflite").
    _ = onnx_path  # referenced above; suppress unused-variable warning until CLI is wired up
