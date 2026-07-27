# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""PyTorch -> CoreML (``.mlpackage``) conversion via ``torch.export`` and coremltools.

Unlike TFLite (ONNX → onnx2tf), native CoreML consumes a :func:`torch.export.export` graph
directly and lowers it with ``coremltools`` to an ``mlprogram`` ``.mlpackage`` suitable for
Xcode / Core ML.  This is distinct from ExecuTorch's ``format="executorch", backend="coreml"``
path, which produces a ``.pte`` for the ExecuTorch runtime.

The model must already be in export mode (``model.export()``) with the rank-≤5 deformable-attention
path — :meth:`rfdetr.detr.RFDETR.export` handles that before calling :func:`export_coreml`.

Note:
    The produced ``.mlpackage`` expects ImageNet mean/std normalization
    (``mean=[0.485, 0.456, 0.406]``, ``std=[0.229, 0.224, 0.225]``), same as ONNX.
    :func:`export_coreml` defaults to ``compute_precision=FLOAT32`` for tight CPU parity with
    eager PyTorch. Pass ``ct.precision.FLOAT16`` to :func:`export_coreml` directly when you want
    a smaller ANE-oriented bundle (expect larger numeric drift).
    :meth:`rfdetr.detr.RFDETR.export` ``format="coreml"`` does not expose this knob and always
    uses the FLOAT32 default.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._coreml.torch_ops import ensure_coreml_torch_op_patches
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    import coremltools as ct

logger = get_logger()


def _check_coremltools_available(*, raise_error: bool = True) -> bool:
    """Return whether ``coremltools`` is importable."""
    try:
        import coremltools  # noqa: F401
    except Exception as exc:
        if raise_error:
            raise ImportError(
                "CoreML export requires `coremltools`. Install it with: pip install rfdetr[coreml]"
            ) from exc
        return False
    return True


def export_coreml(
    model: nn.Module,
    input_tensors: torch.Tensor,
    output_dir: str | os.PathLike[str],
    *,
    variant_name: str | None = None,
    verbose: bool = True,
    compute_precision: ct.precision | None = None,
) -> Path:
    """Export an RF-DETR model to a CoreML ``.mlpackage``.

    The model must already be switched into export mode (``model.export()``) and moved to CPU by the
    caller — the public :meth:`rfdetr.detr.RFDETR.export` entry point handles both.

    Args:
        model: The RF-DETR PyTorch module to export, in export mode and on CPU.
        input_tensors: Example input ``(batch, channels, height, width)`` used to trace the graph.
            Its spatial shape is baked into the exported program.
        output_dir: Directory where the ``.mlpackage`` is written.
        variant_name: Model variant identifier (e.g. ``"rfdetr-nano"``). When provided, the bundle
            is named ``{variant_name}.mlpackage`` instead of ``inference_model.mlpackage``.
        verbose: When ``True``, log export progress at info level.
        compute_precision: coremltools precision for ``ct.convert`` (e.g. ``ct.precision.FLOAT32`` /
            ``FLOAT16``). ``None`` selects ``FLOAT32`` (tight CPU parity with eager PyTorch).
            Only available on this function — :meth:`rfdetr.detr.RFDETR.export` does not forward it.

    Returns:
        Path to the exported ``.mlpackage`` bundle.

    Raises:
        ImportError: If ``coremltools`` is not installed.
        NotImplementedError: If the exported graph contains op kinds missing from coremltools'
            Torch registry (fast-fail checklist).
        RuntimeError: If ``torch.export`` or ``coremltools.convert`` fails.
    """
    _check_coremltools_available()
    import coremltools as ct

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)
    if variant_name:
        variant_name = os.path.splitext(os.path.basename(variant_name))[0]
        export_name = variant_name
    else:
        export_name = "inference_model"
    output_file = output_dir_path / f"{export_name}.mlpackage"

    if compute_precision is None:
        compute_precision = ct.precision.FLOAT32

    model = model.eval()
    if verbose:
        logger.info(f"Exporting model to CoreML format: {output_file}")

    try:
        with torch.no_grad():
            # strict=False: same rationale as ExecuTorch — submodule-lifted spatial_shapes constants
            # break lowering under strict=True on current torch.export + converter stacks.
            exported_program = torch.export.export(model, (input_tensors,), strict=False)
            exported_program = exported_program.run_decompositions({})
            # Patch registry gaps (e.g. aten.alias → coremltools noop) before checklist + convert.
            ensure_coreml_torch_op_patches()
            coverage = unsupported_coreml_ops(exported_program)
            if coverage:
                summary = f"CoreML op registry gaps: {dict(coverage)}"
                logger.error(summary)
                raise NotImplementedError(
                    f"{summary}. Fix these ops before convert (see tests/export/test_coreml_op_coverage.py)."
                )
            mlmodel = ct.convert(
                exported_program,
                convert_to="mlprogram",
                minimum_deployment_target=ct.target.iOS16,
                compute_precision=compute_precision,
            )
    except (ImportError, NotImplementedError, ValueError):
        raise
    except Exception as exc:
        logger.exception("CoreML export failed")
        raise RuntimeError(f"CoreML export failed: {exc}") from exc

    mlmodel.save(str(output_file))
    if verbose:
        logger.info(f"Successfully exported CoreML model to: {output_file}")
    return output_file
