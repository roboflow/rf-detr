# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Direct PyTorch → OpenVINO IR export."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn

from rfdetr.export._naming import resolve_export_stem
from rfdetr.utilities.logger import get_logger

logger = get_logger()


def _check_openvino_available() -> None:
    """Verify that ``openvino`` is importable.

    Shared by :func:`export_openvino` and :class:`~rfdetr.export._openvino.inference.OpenVINOInference`
    so both surface the same actionable message and tests can monkeypatch a single choke point instead
    of relying on ``openvino`` actually being absent from the environment.

    Raises:
        ImportError: If ``openvino`` cannot be imported.
    """
    try:
        import openvino  # noqa: F401
    except ImportError as error:
        raise ImportError('OpenVINO requires `openvino`. Install it with: pip install "rfdetr[openvino]"') from error


class ModelWrapper(nn.Module):
    """Normalize an export-mode RF-DETR forward into a plain tensor tuple for ``openvino.convert_model``.

    The wrapped *model* must already be switched into export mode by the caller (see
    :func:`rfdetr.export._backend._export_openvino_format`) -- ``forward_export`` always returns a
    tuple of tensors for the full detector (``(dets, labels)`` / ``(dets, labels, masks)`` /
    ``(dets, labels, keypoints)``), and the backbone-only export graph
    (:class:`rfdetr.export._backend._BackboneExport`) returns a plain list of tensors. A dict output
    only reaches this wrapper when the caller forgot the mode-switch -- that is a caller bug, not a
    shape this wrapper can flatten, so it raises instead of silently dropping keys.
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
                f"OpenVINO export received a dict-valued model output (keys={sorted(output)}); this "
                "means the model was not switched into export mode before wrapping (forward_export "
                "always returns a tuple/list). Call model.export() before wrapping."
            )
        raise TypeError(f"Unsupported model output type for OpenVINO export: {type(output)!r}")


def export_openvino(
    model: nn.Module,
    input_tensors: torch.Tensor,
    output_dir: str | os.PathLike[str],
    *,
    backbone_only: bool = False,
    verbose: bool = True,
    variant_name: str | None = None,
    output_name: str | None = None,
    precision: str | None = None,
) -> str:
    """Export a PyTorch model directly to OpenVINO IR format.

    The model must already be switched into export mode (``model.export()``) and moved to CPU by the
    caller -- the public :meth:`rfdetr.detr.RFDETR.export` entry point (via
    :func:`rfdetr.export._backend._export_openvino_format`) handles both.

    Signature matches the CoreML/ExecuTorch converters' convention (``model, input_tensors,
    output_dir``, keyword-only after) rather than the legacy ``export_onnx``-style
    ``output_dir``-first order.

    Args:
        model: PyTorch model to export, already in export mode and on CPU.
        input_tensors: Example input tensor(s) for tracing.
        output_dir: Directory where the exported model will be saved.
        backbone_only: Whether *model* is a backbone-only export graph. When ``True`` and a name was
            supplied (*variant_name* or *output_name*), a ``-backbone`` marker is appended to the
            filename so a backbone export never collides with a full-detector export of the same
            variant -- matching the ONNX exporter's ``{stem}-backbone.onnx`` convention. Not appended
            onto the bare ``backbone_model`` default, which already spells it out.
        verbose: Whether to print verbose export information.
        variant_name: Optional model variant name (e.g., "nano", "small", "medium").
        output_name: Full filename override (without extension). Takes precedence over
            *variant_name*; the file is named ``{output_name}.xml``/``.bin`` before the optional
            ``-backbone`` marker.
        precision: ``"float32"``, ``"float16"``, or ``None`` (default, keeps OpenVINO's own
            ``compress_to_fp16=True`` behavior). ``"float32"`` disables FP16 weight compression, which
            controls IR *storage* precision only -- execution precision still depends on the compiled
            device and is not guaranteed to match eager PyTorch on non-CPU devices. ``"float16"`` is
            explicit about the default.

    Returns:
        Path to the exported OpenVINO IR model (.xml file).

    Raises:
        ImportError: If OpenVINO is not installed, or if ``convert_model``'s lazy submodule imports
            fail on a partial/ABI-mismatched install.
        NotImplementedError: If *model* was not switched into export mode first (see
            :class:`ModelWrapper`).
        TypeError: If *model*'s forward returns an output type :class:`ModelWrapper` cannot wrap.
        ValueError: If *precision* is not one of ``"float32"``, ``"float16"``, or ``None``.
        RuntimeError: If conversion or saving otherwise fails.

    Note:
        Output tensor names in the saved IR are OpenVINO-inferred, not renamed to ``dets``/``labels``/
        etc. -- consumers must match outputs by **position**, mirroring the CoreML export's naming
        limitation.
    """
    _check_openvino_available()
    from openvino import convert_model, save_model

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    stem, _ = resolve_export_stem(
        variant_name, output_name, default="backbone_model" if backbone_only else "inference_model"
    )
    export_name = stem
    # "-backbone" is a structural marker (distinct model graph) -- appended whenever a name was
    # supplied, custom or variant-derived, but not onto the bare "backbone_model" default (which
    # already spells it out). Mirrors the ONNX/CoreML/ExecuTorch exporters' convention.
    if backbone_only and (variant_name or output_name):
        export_name = f"{export_name}-backbone"

    output_xml = output_dir_path / f"{export_name}.xml"
    output_bin = output_dir_path / f"{export_name}.bin"

    if precision is None or precision == "float16":
        compress_to_fp16 = True
    elif precision == "float32":
        compress_to_fp16 = False
    else:
        raise ValueError(f"precision must be 'float32', 'float16', or None, got {precision!r}")

    if verbose:
        logger.info("Converting PyTorch model to OpenVINO IR...")
        logger.info(f"Input shape: {input_tensors.shape}")

    model = model.eval().cpu()
    input_tensors = input_tensors.cpu()
    wrapped_model = ModelWrapper(model)
    wrapped_model.eval()

    try:
        with torch.no_grad():
            ov_model = convert_model(wrapped_model, example_input=input_tensors)
        save_model(ov_model, str(output_xml), compress_to_fp16=compress_to_fp16)
    except (ImportError, NotImplementedError, TypeError, ValueError):
        # ImportError: convert_model lazily imports private submodules that can still fail on a
        # partial/ABI-mismatched install even after the top-level `openvino` import succeeded.
        # NotImplementedError/TypeError: raised by ModelWrapper.forward's own documented contract
        # (dict output / unsupported output type) -- must reach the caller as-is, not be relabeled
        # RuntimeError by the broad except below. Mirrors the CoreML exporter's passthrough tier.
        raise
    except Exception as e:
        logger.exception("OpenVINO export failed")
        raise RuntimeError(f"Failed to export model to OpenVINO IR: {e}") from e

    if verbose:
        logger.info(f"✓ OpenVINO IR model saved to {output_xml}")
        logger.info(f"✓ Model binary saved to {output_bin}")

    return str(output_xml)
