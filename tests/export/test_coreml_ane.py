# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Apple Neural Engine loadability and fallback boundary of fp16 CoreML exports (``e2e_coreml``, opt-in).

Apple's ANE compiler rejects a whole fp16 program ("Invalid layer") when the two-stage ``enc_output_norm`` still
carries its identity affine *and* the encoder token count is a multiple of 32; ``src/rfdetr/models/transformer.py``
documents the mechanism at the code it constrains. Under ``ComputeUnit.ALL`` — the default in both coremltools and
Swift — Core ML then cannot build an execution plan and ``predict`` raises.

The token count is ``(resolution / patch_size) ** 2``, so RFDETRNano's resolution picks the case: 384 px reaches
576 tokens and was rejected before the fix, while 448 px reaches 784 and was always accepted. Keeping both
separates a real regression from a broken host or toolchain.

These tests report on the machine they run on. Where no Neural Engine is available (a virtualized CI runner, for
instance) Core ML never invokes its compiler, so the load test passes regardless and the compute-plan tests skip:
the plan would then describe the host, not the exported graph. The hardware-free guard for the fix itself is
``tests/models/test_transformer.py::test_two_stage_topk_gather_reads_pre_norm_memory``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.utilities.reproducibility import seed_all
from tests.export.conftest import _structured_parity_input, eager_reference_tensors

# Same reason as tests/export/test_coreml_export.py's `_COREML_EXPORT_SEED`: coremltools 9.0 can constant-fold a
# weights-only `linear` through `np.matmul` at conversion time, and on some untrained-weight draws (including the
# repo default seed 7) that fold overflows and embeds a non-finite constant. A module-scoped fixture is also set
# up before the function-scoped autouse `reset_random_seeds`, so without this the draw depends on test order.
_EXPORT_SEED = 0

coreml_runtime_only = pytest.mark.skipif(
    not _IS_COREMLTOOLS_AVAILABLE or sys.platform != "darwin",
    reason="needs coremltools and the macOS CoreML runtime",
)


@pytest.fixture(scope="module", params=[384, 448])
def nano_fp16_export(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[Path, torch.Tensor, list[torch.Tensor]]:
    """Export an untrained RFDETRNano at one resolution to fp16 CoreML, with its input and eager outputs.

    Examples:
        Skipped: a pytest fixture that needs coremltools, so it cannot run standalone.

        >>> mlpackage_path, example, eager_outputs = nano_fp16_export  # doctest: +SKIP
        >>> mlpackage_path.suffix, example.shape[-1], len(eager_outputs)  # doctest: +SKIP
        ('.mlpackage', 384, 2)
    """
    resolution = request.param
    seed_all(_EXPORT_SEED)
    detector = RFDETRNano(pretrain_weights=None, resolution=resolution)
    out_dir = tmp_path_factory.mktemp(f"coreml_ane_{resolution}")
    mlpackage_path = detector.export(
        output_dir=str(out_dir), format="coreml", coreml_precision="float16", verbose=False
    )
    model = detector.model.model.to("cpu").eval()
    model.export()
    example = _structured_parity_input(1, 3, resolution, resolution)
    return Path(mlpackage_path), example, eager_reference_tensors(model, example)


@coreml_runtime_only
@pytest.mark.integration
@pytest.mark.e2e_coreml
class TestCoreMLNeuralEngineLoad:
    """An fp16 export must stay loadable when Core ML may schedule it onto the Neural Engine."""

    def test_fp16_export_predicts_eager_shaped_outputs_with_all_compute_units(
        self, nano_fp16_export: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """Under ``ComputeUnit.ALL`` the bundle must build an execution plan and return the eager outputs' shapes.

        Values are not compared: fp16 drift is large on raw outputs, and the parity of the exported graph itself is
        covered at FLOAT32 by ``tests/export/test_coreml_export.py::TestCoreMLEndToEnd``.
        """
        import coremltools as ct

        mlpackage_path, example, eager_outputs = nano_fp16_export
        mlmodel = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.ALL)
        input_name = mlmodel.get_spec().description.input[0].name

        outputs = mlmodel.predict({input_name: example.numpy().astype(np.float32)})

        # Pair by the spec's output order, as tests/export/test_coreml_export.py does: coremltools does not
        # document the predict() dict's ordering, and the outputs differ in shape.
        runtime_arrays = [np.asarray(outputs[output.name]) for output in mlmodel.get_spec().description.output]
        assert [array.shape for array in runtime_arrays] == [tuple(tensor.shape) for tensor in eager_outputs]
        assert all(np.isfinite(array).all() for array in runtime_arrays)


#: Ops RF-DETR's graph is known to leave off the Neural Engine, all of them the two-stage query selection:
#: ``topk`` itself plus the index expansion and gather that consume its result. They cost ~0.1% of the model's
#: estimated work, so the budget below is about catching a *new* op joining them, not about their own cost.
_ANE_UNSUPPORTED_OPS = frozenset({"ios16.topk", "ios16.gather_along_axis", "tile", "expand_dims"})

#: Minimum share of estimated work Core ML must still schedule onto the ANE under ``CPU_AND_NE``. Measured at
#: 0.999 for fp16 RFDETRNano, RFDETRSmall and RFDETRSegNano (pretrained), and for an untrained RFDETRNano, on an
#: Apple M3 Pro running macOS 27.0.
_MIN_ANE_COST_SHARE = 0.99


def _compute_plan(mlpackage_path: Path) -> Any:
    """Load the ``CPU_AND_NE`` compute plan of *mlpackage_path*, skipping the test when this host has no ANE.

    ``MLComputePlan`` answers for the machine it runs on: where a host exposes no Neural Engine — a virtualized CI
    runner, for instance — every op is reported as CPU-preferred and ANE-unsupported, which says nothing about the
    exported graph. Skipping there keeps that from reading as a regression.

    Args:
        mlpackage_path: The ``.mlpackage`` bundle to compile and plan.

    Returns:
        The loaded ``MLComputePlan`` for ``ComputeUnit.CPU_AND_NE``.

    Examples:
        Skipped: needs coremltools, macOS>=14.4 and a real ``.mlpackage``.

        >>> plan = _compute_plan(Path("output/rfdetr-nano_fp16.mlpackage"))  # doctest: +SKIP
        >>> plan.model_structure.program.functions["main"] is not None  # doctest: +SKIP
        True
    """
    compute_plan = pytest.importorskip(
        "coremltools.models.compute_plan", reason="MLComputePlan needs coremltools>=8 and macOS>=14.4"
    )
    import coremltools as ct

    plan = compute_plan.MLComputePlan.load_from_path(
        path=ct.utils.compile_model(str(mlpackage_path)), compute_units=ct.ComputeUnit.CPU_AND_NE
    )
    operations = plan.model_structure.program.functions["main"].block.operations
    supports_ane = any(
        "NeuralEngine" in type(device).__name__
        for operation in operations
        if (usage := plan.get_compute_device_usage_for_mlprogram_operation(operation)) is not None
        for device in usage.supported_compute_devices
    )
    if not supports_ane:
        pytest.skip("this host exposes no Neural Engine, so the plan says nothing about the exported graph")
    return plan


@coreml_runtime_only
@pytest.mark.integration
@pytest.mark.e2e_coreml
class TestCoreMLNeuralEngineFallbackBoundary:
    """The ANE fallback boundary documented in ``docs/learn/export.md`` must stay where it is."""

    def test_only_two_stage_selection_ops_leave_the_neural_engine(
        self, nano_fp16_export: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """No op outside the known two-stage selection island may lose Neural Engine support."""
        mlpackage_path, _, _ = nano_fp16_export
        plan = _compute_plan(mlpackage_path)
        operations = plan.model_structure.program.functions["main"].block.operations

        unsupported = {
            operation.operator_name
            for operation in operations
            if (usage := plan.get_compute_device_usage_for_mlprogram_operation(operation)) is not None
            and not any("NeuralEngine" in type(device).__name__ for device in usage.supported_compute_devices)
        }

        assert unsupported <= _ANE_UNSUPPORTED_OPS, (
            f"ops newly unsupported on the Neural Engine: {sorted(unsupported - _ANE_UNSUPPORTED_OPS)}; "
            "update docs/learn/export.md's fallback boundary if this is intended"
        )

    def test_neural_engine_keeps_nearly_all_of_the_estimated_work(
        self, nano_fp16_export: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """Under ``CPU_AND_NE`` the ANE must still carry the overwhelming majority of the estimated work.

        This is the one check that catches a *silent* whole-model fallback: when the ANE compiler rejects a program,
        Core ML keeps reporting most ops as ANE-supported (so the sibling test above still passes) and the model loads
        fine (so the load test still passes) — it just runs every op on the CPU, which only the preferred-device split
        reveals.
        """
        mlpackage_path, _, _ = nano_fp16_export
        plan = _compute_plan(mlpackage_path)
        operations = plan.model_structure.program.functions["main"].block.operations

        ane_cost, total_cost = 0.0, 0.0
        for operation in operations:
            usage = plan.get_compute_device_usage_for_mlprogram_operation(operation)
            estimate = plan.get_estimated_cost_for_mlprogram_operation(operation)
            if usage is None or estimate is None:
                continue
            total_cost += estimate.weight
            if "NeuralEngine" in type(usage.preferred_compute_device).__name__:
                ane_cost += estimate.weight

        assert total_cost > 0.0
        assert ane_cost / total_cost >= _MIN_ANE_COST_SHARE, (
            f"only {ane_cost / total_cost:.3f} of the estimated work stays on the Neural Engine"
        )
