# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ExecuTorch ``backend="coreml"`` lowering and runtime on Apple hardware (``@pytest.mark.e2e_executorch``, opt-in).

Everything else covering this backend mocks ``executorch`` away, and the portable e2e suite in
``test_executorch_export.py`` lowers with XNNPACK. Nothing exercised the CoreML delegate itself: whether the partitioner
still takes the whole graph, and whether the resulting ``.pte`` loads and runs through the CoreML delegate at all. Both
are what "ANE partitioning" risk means in practice — an op the partitioner drops runs on ExecuTorch's portable CPU
kernels and never reaches the Neural Engine.

Inside the delegated blob, scheduling is Core ML's decision at load time, exactly as for a native ``.mlpackage``; the
fallback boundary there is pinned by ``tests/export/test_coreml_ane.py``.

Needs ``executorch`` **and** ``coremltools`` in the same environment, plus macOS to run the delegate. The ``coreml``
extra pins an older torch than ``executorch`` accepts, so install ``coremltools`` on its own next to the ``executorch``
extra (see the ``executorch-coreml-parity`` job in ``.github/workflows/ci-integrations.yml``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.utilities.reproducibility import seed_all
from tests.export.conftest import _structured_parity_input, eager_reference_tensors
from tests.export.test_executorch_export import _portable_kernel_call_names

# Same reason as tests/export/test_coreml_ane.py's `_EXPORT_SEED`: this fixture is module-scoped, so it runs
# before the function-scoped autouse `reset_random_seeds` — without an explicit seed the untrained weight draw
# depends on process/test order.
_EXPORT_SEED = 0

executorch_coreml_only = pytest.mark.skipif(
    not _IS_EXECUTORCH_AVAILABLE or not _IS_COREMLTOOLS_AVAILABLE or sys.platform != "darwin",
    reason="needs executorch, coremltools and macOS to run the CoreML delegate",
)


@pytest.fixture(scope="module")
def nano_coreml_pte(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, torch.Tensor, list[torch.Tensor]]:
    """Lower an untrained RFDETRNano through the CoreML delegate; return the ``.pte``, its input and eager outputs.

    Examples:
        Skipped: a pytest fixture needing executorch and coremltools, so it cannot run standalone.

        >>> pte_path, example, eager_outputs = nano_coreml_pte  # doctest: +SKIP
        >>> pte_path.suffix, len(eager_outputs)  # doctest: +SKIP
        ('.pte', 2)
    """
    seed_all(_EXPORT_SEED)
    detector = RFDETRNano(pretrain_weights=None)
    out_dir = tmp_path_factory.mktemp("executorch_coreml")
    pte_path = detector.export(output_dir=str(out_dir), format="executorch", backend="coreml", verbose=False)
    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return Path(pte_path), example, eager_reference_tensors(model, example)


def _delegate_call_count(pte_path: Path) -> int:
    """Return how many CoreML ``DelegateCall`` instructions the serialized ``.pte`` graph contains.

    Args:
        pte_path: Path to a serialized ExecuTorch program.

    Returns:
        Number of ``DelegateCall`` instructions across every chain in the first execution plan.

    Examples:
        Requires a real ``.pte`` artifact and the ``executorch`` package — not runnable standalone.

        >>> callable(_delegate_call_count)
        True
    """
    from executorch.exir._serialize import _deserialize_pte_binary

    plan = _deserialize_pte_binary(pte_path.read_bytes()).program.execution_plan[0]
    return sum(
        1
        for chain in plan.chains
        for instruction in chain.instructions
        if type(instruction.instr_args).__name__ == "DelegateCall"
    )


@executorch_coreml_only
@pytest.mark.integration
@pytest.mark.e2e_executorch
class TestExecuTorchCoreMLDelegate:
    """The CoreML delegate must take the whole graph and produce a ``.pte`` the runtime can execute."""

    def test_shipped_pte_has_no_undelegated_kernel_calls(
        self, nano_coreml_pte: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """The exported ``.pte`` must contain no portable kernel call: every op belongs to the CoreML delegate.

        Read from the artifact the exporter actually wrote, not from a re-lowering, so a regression anywhere in
        ``ExecuTorchExporter``'s own lowering path is caught too. A portable kernel call here would be an op the
        partitioner dropped, which then runs on ExecuTorch's CPU kernels and never reaches the Neural Engine.
        """
        pte_path, _, _ = nano_coreml_pte

        assert _portable_kernel_call_names(pte_path) == []

    def test_shipped_pte_has_exactly_one_coreml_delegate_call(
        self, nano_coreml_pte: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """The exported ``.pte`` must contain exactly one CoreML ``DelegateCall``: the graph is one delegate blob.

        The no-portable-kernel-calls test above only proves nothing fell back to portable CPU kernels; it does not prove
        the documented invariant that the partitioner produced a single CoreML delegate rather than several disjoint
        ones. Multiple delegate calls would mean the graph was split across separate Core ML subgraphs, each paying its
        own load/dispatch overhead and losing the "one native ``.mlpackage``" scheduling guarantee this module's
        docstring describes.
        """
        pte_path, _, _ = nano_coreml_pte

        assert _delegate_call_count(pte_path) == 1

    def test_runtime_outputs_match_eager_shapes(
        self, nano_coreml_pte: tuple[Path, torch.Tensor, list[torch.Tensor]]
    ) -> None:
        """The delegated ``.pte`` must load and return the eager model's output shapes, finite throughout.

        Values are not compared: the delegate runs in fp16, where raw RF-DETR outputs drift by ~1e0. Numeric parity
        for this graph is covered at fp32 by the XNNPACK suite in ``test_executorch_export.py``.

        Unlike the native ``.mlpackage`` ANE load test in ``test_coreml_ane.py``, this delegate-path test's
        pre-fix ``develop`` baseline was not independently re-verified for this PR.
        """
        from executorch.runtime import Runtime

        pte_path, example, eager_outputs = nano_coreml_pte
        method = Runtime.get().load_program(str(pte_path)).load_method("forward")

        outputs = [np.asarray(tensor) for tensor in method.execute([example])]

        assert [output.shape for output in outputs] == [tuple(tensor.shape) for tensor in eager_outputs]
        assert all(np.isfinite(output).all() for output in outputs)
