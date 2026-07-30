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
    eager PyTorch. Pass ``coremltools.precision.FLOAT16`` (or the string ``"float16"``) when you
    want a smaller ANE-oriented bundle (expect larger numeric drift) — either directly to
    :func:`export_coreml`, or via :meth:`rfdetr.detr.RFDETR.export`'s ``coreml_precision``
    argument (string form only, so callers don't need to import ``coremltools``).

Note:
    ``torch>=2.12`` sharply raises the rate of a CoreML/eager numeric-parity divergence on
    real-image input with coremltools 9.0 (bisected: torch<2.12 failed ~1/6 repeat runs,
    torch>=2.12.0 failed ~6/7) — the ``coreml`` extra pins ``torch<2.12`` in ``pyproject.toml``
    until this is understood/fixed upstream. That pin reduces the failure rate, it does not
    guarantee determinism. See ``tests/export/test_coreml_export.py::TestCoreMLEndToEnd::
    test_outputs_match_pytorch_supervision_image``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from torch import nn

from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.op_coverage import unsupported_coreml_ops
from rfdetr.export._naming import resolve_export_stem
from rfdetr.utilities.logger import get_logger

if TYPE_CHECKING:
    from coremltools import precision as ct_precision

logger = get_logger()


def _check_coremltools_available(*, raise_error: bool = True) -> bool:
    """Return whether ``coremltools`` is importable."""
    if not _IS_COREMLTOOLS_AVAILABLE:
        if raise_error:
            raise ImportError("CoreML export requires `coremltools`. Install it with: pip install rfdetr[coreml]")
        return False
    return True


def export_coreml(
    model: nn.Module,
    input_tensors: torch.Tensor,
    output_dir: str | os.PathLike[str],
    *,
    variant_name: str | None = None,
    verbose: bool = True,
    compute_precision: ct_precision | str | None = None,
    output_name: str | None = None,
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
            is named ``{variant_name}_fp32.mlpackage`` or ``{variant_name}_fp16.mlpackage`` (matching
            *compute_precision*) instead of the generic ``inference_model_fp32.mlpackage`` /
            ``inference_model_fp16.mlpackage``.
        verbose: When ``True``, log export progress at info level.
        compute_precision: coremltools precision for ``coremltools.convert`` — a
            ``coremltools.precision`` value (e.g. ``coremltools.precision.FLOAT32`` / ``FLOAT16``),
            the equivalent string (``"float32"`` / ``"float16"``), or ``None`` to select
            ``FLOAT32`` (tight CPU parity with eager PyTorch). The resolved precision is always
            encoded in the output filename (``_fp32``/``_fp16``) since it materially changes the
            artifact — unless *output_name* is given.
            The string form is what :meth:`rfdetr.detr.RFDETR.export` forwards via its
            ``coreml_precision`` argument, so callers don't need to import ``coremltools`` just to
            pick a precision.
        output_name: Full filename override (without extension). Takes precedence over
            *variant_name* and suppresses the ``_fp32``/``_fp16`` suffix — the bundle is named
            ``{output_name}.mlpackage`` verbatim.

    Returns:
        Path to the exported ``.mlpackage`` bundle. Output tensor names in the saved spec are
        coremltools-inferred, not renamed to ``dets``/``labels``/etc. — consumers must match outputs
        by **position**, in the same order as the model's ONNX ``output_names`` contract.

    Raises:
        ImportError: If ``coremltools`` is not installed, or if ``coremltools.convert`` triggers a
            lazy import of a private submodule that fails even though the top-level package is
            installed (e.g. a partial/ABI-mismatched install).
        NotImplementedError: If the exported graph contains op kinds missing from coremltools'
            Torch registry (fast-fail checklist).
        ValueError: If ``torch.export`` or ``coremltools.convert`` raises it directly (e.g. invalid
            precision/shape arguments) — passed through unwrapped rather than re-wrapped as
            ``RuntimeError``.
        RuntimeError: If ``torch.export`` or ``coremltools.convert`` fails for any other reason.
    """
    _check_coremltools_available()
    from coremltools import convert as ct_convert
    from coremltools import precision as ct_precision
    from coremltools import target as ct_target

    # ct_precision is an Enum class, not a module — FLOAT32/FLOAT16 are members accessed via
    # attribute lookup (`ct_precision.FLOAT32`), not independently `from`-importable. Bind them
    # to flat local names via plain assignment so the rest of this function reads uniformly.
    coreml_float32 = ct_precision.FLOAT32
    coreml_float16 = ct_precision.FLOAT16

    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if compute_precision is None:
        compute_precision = coreml_float32
    elif isinstance(compute_precision, str):
        try:
            compute_precision = {"float32": coreml_float32, "float16": coreml_float16}[compute_precision]
        except KeyError:
            raise ValueError(
                f"compute_precision must be 'float32', 'float16', a coremltools.precision value, or "
                f"None, got {compute_precision!r}"
            ) from None

    stem, is_custom = resolve_export_stem(variant_name, output_name)
    # Precision materially changes the artifact (fp16 has larger numeric drift, per the module
    # docstring) — always encode it, unless the caller asked for an exact custom filename.
    if compute_precision == coreml_float16:
        precision_token = "fp16"
    elif compute_precision == coreml_float32:
        precision_token = "fp32"
    else:
        precision_token = "fp32"
        logger.warning(
            f"Unrecognized CoreML compute precision {compute_precision!r}; "
            "using the fp32 filename label."
        )
    export_name = stem if is_custom else f"{stem}_{precision_token}"
    output_file = output_dir_path / f"{export_name}.mlpackage"

    model = model.eval()
    if verbose:
        logger.info(f"Exporting model to CoreML format: {output_file}")

    try:
        with torch.no_grad():
            # strict=False: same rationale as ExecuTorch — submodule-lifted spatial_shapes constants
            # break lowering under strict=True on current torch.export + converter stacks.
            exported_program = torch.export.export(model, (input_tensors,), strict=False)
            exported_program = exported_program.run_decompositions({})
            # unsupported_coreml_ops() applies ensure_coreml_torch_op_patches() itself before
            # scanning (registry gaps, e.g. aten.alias → coremltools noop, must be patched before
            # the checklist runs) — no separate call needed here.
            coverage = unsupported_coreml_ops(exported_program)
            if coverage:
                summary = f"CoreML op registry gaps: {dict(coverage)}"
                logger.error(summary)
                raise NotImplementedError(
                    f"{summary}. Fix these ops before convert (see tests/export/test_coreml_op_coverage.py)."
                )
            mlmodel = ct_convert(
                exported_program,
                convert_to="mlprogram",
                minimum_deployment_target=ct_target.iOS16,
                compute_precision=compute_precision,
            )
    except (ImportError, NotImplementedError, ValueError):
        # ImportError here is not dead code even though _check_coremltools_available() already
        # verified the top-level `coremltools` package above: ct.convert() lazily imports private
        # submodules (MIL passes, backend components) that can still fail on a partial/ABI-mismatched
        # install even after the top-level import succeeds.
        raise
    except Exception as exc:
        logger.exception("CoreML export failed")
        raise RuntimeError(f"CoreML export failed: {exc}") from exc

    mlmodel.save(str(output_file))
    if verbose:
        logger.info(f"Successfully exported CoreML model to: {output_file}")
    return output_file
