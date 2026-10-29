# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""The Core ML export resizes with matmuls: the Neural Engine's 3x ``upsample_bilinear`` samples half a cell off.

The segmentation head upsamples its mask features 3x to the stride-4 mask grid. On the Neural Engine,
``upsample_bilinear`` by 3 reads source cell ``i / 3 - 0.5`` for output cell ``i``, instead of ``(i + 0.5) / 3 - 0.5``,
so every mask lands half a mask cell below and right of its object. ``rfdetr::upsample_bilinear_to_matmul`` computes
the same resize as two matmuls by constant interpolation matrices.

Every test needs coremltools. Core ML compiles and runs models only on macOS, so the compute-plan and prediction tests
skip elsewhere, and the compute-plan test also needs a Neural Engine.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr import RFDETRSegNano
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE

pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e_coreml,
    pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="needs coremltools"),
]
coreml_runtime_only = pytest.mark.skipif(sys.platform != "darwin", reason="needs the macOS Core ML runtime")

# Core ML runs smaller resizes on the CPU, where the Neural Engine's offset never shows.
_UPSAMPLE_INPUT_SHAPE = (1, 256, 36, 36)

_PRECISIONS = pytest.mark.parametrize("precision", ["float32", "float16"])


def _convert_upsample(precision: str, align_corners: bool, compute_units: str = "CPU_ONLY") -> Any:
    """Convert a lone 3x ``upsample_bilinear`` with the export's pass pipeline.

    Args:
        precision: ``"float32"`` or ``"float16"``, the ``coremltools.precision`` member to convert with.
        align_corners: The ``align_corners`` of the upsample.
        compute_units: The ``coremltools.ComputeUnit`` member that runs the model's predictions.

    Returns:
        The converted ``coremltools.models.MLModel``.
    """
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias

    from rfdetr.export._coreml.passes import rfdetr_pass_pipeline

    @mb.program(input_specs=[mb.TensorSpec(shape=_UPSAMPLE_INPUT_SHAPE)], opset_version=ct.target.iOS15)
    def program(x: Any) -> Any:
        return mb.upsample_bilinear(x=x, scale_factor_height=3.0, scale_factor_width=3.0, align_corners=align_corners)

    return ct.convert(
        program,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS15,
        compute_precision=getattr(ct.precision, precision.upper()),
        pass_pipeline=rfdetr_pass_pipeline(),
        compute_units=getattr(ct.ComputeUnit, compute_units),
    )


@_PRECISIONS
def test_upsample_becomes_matmuls_by_constants_on_the_right(precision: str) -> None:
    """No ``upsample_bilinear`` is left, and each interpolation matrix is the right operand of its matmul.

    With a constant on the left, the Neural Engine does not compile the matmul and Core ML runs the resize on the CPU.
    """
    spec = _convert_upsample(precision, align_corners=False).get_spec()
    (block,) = spec.mlProgram.functions["main"].block_specializations.values()
    constants = {
        output.name for operation in block.operations if operation.type == "const" for output in operation.outputs
    }
    matmul_operands_are_constant = [
        tuple(operation.inputs[operand].arguments[0].name in constants for operand in ("x", "y"))
        for operation in block.operations
        if operation.type == "matmul"
    ]

    assert "upsample_bilinear" not in {operation.type for operation in block.operations}
    assert matmul_operands_are_constant == [(False, True), (False, True)]


@coreml_runtime_only
def test_rewritten_upsample_runs_on_the_neural_engine() -> None:
    """Under ``CPU_AND_NE`` both matmuls of the rewritten fp16 upsample prefer the Neural Engine.

    ``MLComputePlan`` answers for the host it runs on. A host without a Neural Engine (a virtualized CI runner) reports
    every op as unsupported there, which says nothing about the model, so the test skips.
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    model = _convert_upsample("float16", align_corners=False, compute_units="CPU_AND_NE")
    plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    usages = [
        (operation.operator_name, usage)
        for operation in plan.model_structure.program.functions["main"].block.operations
        if (usage := plan.get_compute_device_usage_for_mlprogram_operation(operation)) is not None
    ]
    if not any(
        "NeuralEngine" in type(device).__name__ for _, usage in usages for device in usage.supported_compute_devices
    ):
        pytest.skip("this host exposes no Neural Engine, so the plan says nothing about the model")

    matmul_devices = [type(usage.preferred_compute_device).__name__ for name, usage in usages if name == "matmul"]

    assert len(matmul_devices) == 2
    assert all("NeuralEngine" in device for device in matmul_devices)


@coreml_runtime_only
@pytest.mark.parametrize("compute_units", ["CPU_ONLY", "CPU_AND_NE"])
@pytest.mark.parametrize("align_corners", [False, True])
@_PRECISIONS
def test_rewritten_upsample_predicts_like_torch(precision: str, align_corners: bool, compute_units: str) -> None:
    """The rewritten upsample predicts like ``F.interpolate`` on the CPU and on the Neural Engine.

    On the Neural Engine, the original fp16 ``upsample_bilinear`` with ``align_corners=False`` misses here by up to 1.8.
    """
    x = np.random.default_rng(0).standard_normal(_UPSAMPLE_INPUT_SHAPE).astype(np.float32)

    (prediction,) = _convert_upsample(precision, align_corners, compute_units).predict({"x": x}).values()

    expected = F.interpolate(torch.from_numpy(x), scale_factor=3, mode="bilinear", align_corners=align_corners)
    np.testing.assert_allclose(prediction, expected.numpy(), atol=1e-2)


def test_segmentation_export_has_no_bilinear_resize(tmp_path: Path) -> None:
    """The fp16 segmentation export resizes its mask features with matmuls, so no bilinear resize reaches Core ML."""
    import coremltools as ct

    bundle = RFDETRSegNano(pretrain_weights=None).export(
        format="coreml", coreml_precision="float16", output_dir=str(tmp_path), verbose=False
    )
    program = ct.utils.load_spec(str(bundle)).mlProgram
    op_counts = Counter(
        operation.type
        for function in program.functions.values()
        for block in function.block_specializations.values()
        for operation in block.operations
    )

    assert op_counts.keys().isdisjoint({"upsample_bilinear", "resize_bilinear"})
