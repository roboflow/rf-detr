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

#: Skip a test node that needs ``coremltools``.
_requires_coremltools = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="needs coremltools")


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e_coreml,
]
_requires_macos = pytest.mark.skipif(sys.platform != "darwin", reason="needs the macOS Core ML runtime")

# Core ML runs smaller resizes on the CPU, where the Neural Engine's offset never shows.
_UPSAMPLE_INPUT_SHAPE = (1, 256, 36, 36)
# Height and width differ, so a swapped or shared interpolation matrix gives a wrong shape or wrong values.
_NON_SQUARE_INPUT_SHAPE = (1, 256, 26, 20)

_PRECISIONS = pytest.mark.parametrize("precision", ["float32", "float16"])
_INPUT_SHAPES = pytest.mark.parametrize(
    "input_shape",
    [
        pytest.param(_UPSAMPLE_INPUT_SHAPE, id="square"),
        pytest.param(_NON_SQUARE_INPUT_SHAPE, id="non-square"),
    ],
)
_NO_NEURAL_ENGINE = "this host exposes no Neural Engine, so the plan says nothing about the model"


def _convert_upsample(
    precision: str,
    align_corners: bool,
    compute_units: str = "CPU_ONLY",
    input_shape: tuple[int, ...] = _UPSAMPLE_INPUT_SHAPE,
) -> Any:
    """Convert a lone 3x ``upsample_bilinear`` with the export's pass pipeline.

    Args:
        precision: ``"float32"`` or ``"float16"``, the ``coremltools.precision`` member to convert with.
        align_corners: The ``align_corners`` of the upsample.
        compute_units: The ``coremltools.ComputeUnit`` member that runs the model's predictions.
        input_shape: The NCHW shape of the upsample's input.

    Returns:
        The converted ``coremltools.models.MLModel``.

    Examples:
        Conversion needs coremltools, and prediction needs the macOS Core ML runtime, so the example is not run:

        >>> _convert_upsample("float16", align_corners=False, compute_units="CPU_AND_NE")  # doctest: +SKIP
    """
    import coremltools as ct
    from coremltools.converters.mil.mil import Builder as mb  # noqa: N813 — standard coremltools MIL alias

    from rfdetr.export._coreml.passes import rfdetr_pass_pipeline

    @mb.program(input_specs=[mb.TensorSpec(shape=input_shape)], opset_version=ct.target.iOS15)
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


def _compute_plan_usages(model: Any) -> list[tuple[str, Any]]:
    """Return the compute devices Core ML plans for each operation of ``model`` under ``CPU_AND_NE``.

    Args:
        model: A converted ``coremltools.models.MLModel`` with an ML program.

    Returns:
        One ``(operator name, MLComputeDeviceUsage)`` pair per operation the plan reports a usage for.

    Examples:
        The plan needs coremltools and the macOS Core ML runtime, so the example is not run:

        >>> _compute_plan_usages(_convert_upsample("float16", False, "CPU_AND_NE"))  # doctest: +SKIP
    """
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    plan = MLComputePlan.load_from_path(path=model.get_compiled_model_path(), compute_units=ct.ComputeUnit.CPU_AND_NE)
    return [
        (operation.operator_name, usage)
        for operation in plan.model_structure.program.functions["main"].block.operations
        if (usage := plan.get_compute_device_usage_for_mlprogram_operation(operation)) is not None
    ]


def _has_neural_engine(usages: list[tuple[str, Any]]) -> bool:
    """Return whether the plan lists a Neural Engine among the devices any operation supports.

    A host without a Neural Engine (a virtualized CI runner) reports every operation as unsupported there. A test that
    only discriminates on the Neural Engine would pass without the fix on such a host, so it skips instead.

    Args:
        usages: The pairs :func:`_compute_plan_usages` returns.

    Returns:
        ``True`` when at least one supported device is a Neural Engine device.

    Examples:
        >>> from types import SimpleNamespace
        >>> class MLNeuralEngineComputeDevice: ...
        >>> class MLCPUComputeDevice: ...
        >>> on_engine = SimpleNamespace(supported_compute_devices=[MLCPUComputeDevice(), MLNeuralEngineComputeDevice()])
        >>> cpu_only = SimpleNamespace(supported_compute_devices=[MLCPUComputeDevice()])
        >>> _has_neural_engine([("matmul", on_engine)]), _has_neural_engine([("matmul", cpu_only)])
        (True, False)
    """
    return any(
        "NeuralEngine" in type(device).__name__ for _, usage in usages for device in usage.supported_compute_devices
    )


def _assert_predicts_like_torch(model: Any, input_shape: tuple[int, ...], align_corners: bool) -> None:
    """Assert that ``model`` predicts the 3x bilinear resize ``F.interpolate`` computes.

    Args:
        model: A converted ``coremltools.models.MLModel`` from :func:`_convert_upsample`.
        input_shape: The NCHW shape ``model`` was converted for.
        align_corners: The ``align_corners`` ``model`` was converted with.

    Examples:
        Prediction needs the macOS Core ML runtime, so the example is not run:

        >>> _assert_predicts_like_torch(_convert_upsample("float16", False), (1, 256, 36, 36), False)  # doctest: +SKIP
    """
    x = np.random.default_rng(0).standard_normal(input_shape).astype(np.float32)

    (prediction,) = model.predict({"x": x}).values()

    expected = F.interpolate(torch.from_numpy(x), scale_factor=3, mode="bilinear", align_corners=align_corners)
    np.testing.assert_allclose(prediction, expected.numpy(), atol=1e-2)


@_requires_coremltools
@_PRECISIONS
def test_upsample_becomes_matmuls_with_constant_right_operands(precision: str) -> None:
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


@_requires_coremltools
@_requires_macos
def test_rewritten_upsample_matmuls_prefer_neural_engine() -> None:
    """Under ``CPU_AND_NE`` both matmuls of the rewritten fp16 upsample prefer the Neural Engine.

    ``MLComputePlan`` answers for the host it runs on. A host without a Neural Engine (a virtualized CI runner) reports
    every op as unsupported there, which says nothing about the model, so the test skips.
    """
    model = _convert_upsample("float16", align_corners=False, compute_units="CPU_AND_NE")
    usages = _compute_plan_usages(model)
    if not _has_neural_engine(usages):
        pytest.skip(_NO_NEURAL_ENGINE)

    matmul_devices = [type(usage.preferred_compute_device).__name__ for name, usage in usages if name == "matmul"]

    assert len(matmul_devices) == 2
    assert all("NeuralEngine" in device for device in matmul_devices)


@_requires_coremltools
@_requires_macos
@_INPUT_SHAPES
@pytest.mark.parametrize("align_corners", [False, True])
@_PRECISIONS
def test_rewritten_upsample_predicts_like_torch(
    precision: str, align_corners: bool, input_shape: tuple[int, ...]
) -> None:
    """The rewritten upsample predicts like ``F.interpolate`` on the CPU, for square and non-square inputs.

    The CPU resize was never wrong, so this guards the matrices themselves: a swapped or shared height and width matrix
    only shows when the two sizes differ.
    """
    model = _convert_upsample(precision, align_corners, input_shape=input_shape)

    _assert_predicts_like_torch(model, input_shape, align_corners)


@_requires_coremltools
@_requires_macos
@_INPUT_SHAPES
@pytest.mark.parametrize("align_corners", [False, True])
def test_rewritten_upsample_predicts_like_torch_on_neural_engine(
    align_corners: bool, input_shape: tuple[int, ...]
) -> None:
    """The rewritten fp16 upsample predicts like ``F.interpolate`` under ``CPU_AND_NE``.

    On the Neural Engine, the original fp16 ``upsample_bilinear`` with ``align_corners=False`` misses here by up to 1.8,
    square or not. The Neural Engine runs fp16 only, so float32 is covered by the CPU test above. On a host without a
    Neural Engine the original passes too, so the test skips there instead of passing without the fix.
    """
    model = _convert_upsample("float16", align_corners, "CPU_AND_NE", input_shape)
    if not _has_neural_engine(_compute_plan_usages(model)):
        pytest.skip(_NO_NEURAL_ENGINE)

    _assert_predicts_like_torch(model, input_shape, align_corners)


@_requires_coremltools
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
