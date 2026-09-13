# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Direct PyTorch -> LiteRT (``.tflite``) export via ``torch.export`` + ``litert-torch``.

LiteRT (formerly TensorFlow Lite) is Google's on-device runtime.  This route hands the export-mode RF-DETR module to
``litert_torch.convert``, which captures it with :func:`torch.export.export` and lowers the captured graph to a
``.tflite`` flatbuffer itself -- no ONNX and no TensorFlow step, unlike ``format="tflite"``
(:mod:`rfdetr.export._tflite`), which goes PyTorch -> ONNX -> ``onnx2tf`` -> TFLite.  Both routes produce a
``.tflite`` that the ``ai_edge_litert`` interpreter runs.

The exported graph is the full detector in one file, two-stage query selection (``topk``/``gather``) included, so it
runs on the CPU (XNNPACK) delegate.  Detection, segmentation and backbone-only exports match eager PyTorch to ~1e-5 on
CPU (checked by the ``e2e_litert`` test suite).  Keypoint models are not supported on litert-torch 0.9.4: its converter
rejects the rank-4 ``batch_matmul`` that the keypoint head's ``nn.Linear`` lowers to.

Note:
    The produced ``.tflite`` expects the same input normalization as the ONNX export: ImageNet mean/std
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``), NCHW float32.  Output tensor names are
    litert-torch's own (``serving_default_output_<i>_output``); consumers match outputs by **position**, mirroring the
    CoreML and OpenVINO exports.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from rfdetr.export._naming import resolve_export_stem
from rfdetr.utilities.logger import get_logger

logger = get_logger()

_INSTALL_HINT = 'LiteRT export requires `litert-torch`. Install it with: pip install "rfdetr[litert]"'


def _check_litert_available() -> None:
    """Verify that ``litert_torch`` is importable.

    Shared by :func:`export_litert` and the package-level ``_IS_LITERT_AVAILABLE`` flag so both surface the same
    actionable message, and so tests can monkeypatch a single choke point instead of relying on ``litert_torch``
    actually being absent from the environment.

    Raises:
        ImportError: If ``litert_torch`` cannot be imported.
    """
    try:
        import litert_torch  # noqa: F401
    except ImportError as error:
        raise ImportError(_INSTALL_HINT) from error


class ModelWrapper(nn.Module):
    """Normalize an export-mode RF-DETR forward into a plain tensor tuple for ``litert_torch.convert``.

    The wrapped *model* must already be switched into export mode by the caller (see
    :func:`rfdetr.export._backend._export_litert_format`) -- ``forward_export`` always returns a tuple of tensors
    for the full detector (``(dets, labels)`` / ``(dets, labels, masks)``), and the backbone-only export graph
    (:class:`rfdetr.export._backend._BackboneExport`) returns a plain list of tensors.  A dict output only reaches
    this wrapper when the caller forgot the mode-switch -- that is a caller bug, not a shape this wrapper can
    flatten, so it raises instead of silently dropping keys.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        output = self.model(x)
        if isinstance(output, (list, tuple)):
            return tuple(output)
        if isinstance(output, dict):
            raise NotImplementedError(
                f"LiteRT export received a dict-valued model output (keys={sorted(output)}); this means the model "
                "was not switched into export mode before wrapping (forward_export always returns a tuple/list). "
                "Call model.export() before wrapping."
            )
        raise TypeError(f"Unsupported model output type for LiteRT export: {type(output)!r}")


def export_litert(
    model: nn.Module,
    input_tensors: torch.Tensor,
    output_dir: str | os.PathLike[str],
    *,
    backbone_only: bool = False,
    verbose: bool = True,
    variant_name: str | None = None,
    output_name: str | None = None,
) -> Path:
    """Export a PyTorch model directly to a LiteRT ``.tflite`` file.

    The model must already be switched into export mode (``model.export()``) and moved to CPU by the caller -- the
    public :meth:`rfdetr.detr.RFDETR.export` entry point (via :func:`rfdetr.export._backend._export_litert_format`)
    handles both.

    Signature matches the CoreML/ExecuTorch/OpenVINO converters' convention (``model, input_tensors, output_dir``,
    keyword-only after).

    Args:
        model: PyTorch model to export, already in export mode and on CPU.
        input_tensors: Example input tensor ``(batch, channels, height, width)``; its shape is baked into the file.
        output_dir: Directory where the exported model will be saved.
        backbone_only: Whether *model* is a backbone-only export graph.  When ``True`` and a name was supplied
            (*variant_name* or *output_name*), a ``-backbone`` marker is appended to the filename so a backbone
            export never collides with a full-detector export of the same variant -- matching the ONNX exporter's
            ``{stem}-backbone.onnx`` convention.  Not appended onto the bare ``backbone_model`` default, which
            already spells it out.
        verbose: Whether to log export progress.
        variant_name: Optional model variant name (e.g., ``"rfdetr-nano"``).
        output_name: Full filename override (without extension).  Takes precedence over *variant_name*; the file is
            named ``{output_name}.tflite`` before the optional ``-backbone`` marker.

    Returns:
        Path to the exported ``.tflite`` file.

    Raises:
        ImportError: If ``litert-torch`` is not installed.
        NotImplementedError: If *model* was not switched into export mode first (see :class:`ModelWrapper`).
        TypeError: If *model*'s forward returns an output type :class:`ModelWrapper` cannot wrap.
        RuntimeError: If ``torch.export`` capture, lowering, or writing the file otherwise fails.
    """
    _check_litert_available()
    import litert_torch

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    stem, _ = resolve_export_stem(
        variant_name, output_name, default="backbone_model" if backbone_only else "inference_model"
    )
    export_name = stem
    # "-backbone" is a structural marker (distinct model graph) -- appended whenever a name was supplied, custom or
    # variant-derived, but not onto the bare "backbone_model" default (which already spells it out).  Mirrors the
    # ONNX/CoreML/ExecuTorch/OpenVINO exporters' convention.
    if backbone_only and (variant_name or output_name):
        export_name = f"{export_name}-backbone"
    output_file = output_dir_path / f"{export_name}.tflite"

    if verbose:
        logger.info("Converting PyTorch model to LiteRT (.tflite) with litert-torch...")
        logger.info(f"Input shape: {tuple(input_tensors.shape)}")

    model = model.eval().cpu()
    input_tensors = input_tensors.cpu()
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    try:
        with torch.no_grad():
            edge_model = litert_torch.convert(wrapped_model, (input_tensors,))
            edge_model.export(str(output_file))
    except (ImportError, NotImplementedError, TypeError):
        # ImportError: litert-torch lazily imports its converter/quantizer companions, which can still fail on a
        # partial install after the top-level import succeeded.  NotImplementedError/TypeError: raised by
        # ModelWrapper.forward's own documented contract (dict output / unsupported output type) -- must reach the
        # caller as-is, not be relabeled RuntimeError by the broad except below.  Mirrors the OpenVINO exporter.
        raise
    except Exception as e:
        logger.exception("LiteRT export failed")
        raise RuntimeError(f"Failed to export model to LiteRT: {e}") from e

    if verbose:
        logger.info(f"✓ LiteRT model saved to {output_file}")

    return output_file
