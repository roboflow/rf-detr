# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ExecuTorch ``backend="tensorrt"`` lowering and runtime on an NVIDIA GPU (``@pytest.mark.e2e_executorch_tensorrt``).

The unit tests in ``test_executorch_export.py`` mock Torch-TensorRT away. These export real models through it and run
the ``.pte`` with ExecuTorch's runtime and the Torch-TensorRT delegate, so they need the ``executorch-tensorrt`` extra
and a CUDA device.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from rfdetr import RFDETRNano
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export.imports import _IS_TORCH_TENSORRT_INSTALLED
from rfdetr.utilities.reproducibility import seed_all
from tests._markers import _requires_cuda
from tests.export.conftest import assert_prediction_roundtrip
from tests.export.executorch.test_executorch_export import _portable_kernel_call_names

# Same reason as test_executorch_coreml.py's `_EXPORT_SEED`: the module-scoped fixture runs before the function-scoped
# autouse seed reset.
_EXPORT_SEED = 0

_requires_executorch_tensorrt = pytest.mark.skipif(
    not _IS_EXECUTORCH_AVAILABLE or not _IS_TORCH_TENSORRT_INSTALLED,
    reason="needs the executorch-tensorrt extra",
)


@pytest.fixture(scope="module")
def nano_tensorrt_pte(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Lower an untrained RFDETRNano through the TensorRT delegate and return the ``.pte``.

    Examples:
        Skipped: a pytest fixture needing the executorch-tensorrt extra and a GPU, so it cannot run standalone.

        >>> nano_tensorrt_pte.name  # doctest: +SKIP
        'rfdetr-nano_tensorrt.pte'
    """
    seed_all(_EXPORT_SEED)
    detector = RFDETRNano(pretrain_weights=None)
    out_dir = tmp_path_factory.mktemp("executorch_tensorrt")
    return Path(detector.export(output_dir=str(out_dir), format="executorch", backend="tensorrt", verbose=False))


def _delegate_ids(pte_path: Path) -> list[str]:
    """Return the backend id of every delegate in the serialized ``.pte``, in program order.

    Args:
        pte_path: Path to a serialized ExecuTorch program.

    Returns:
        One backend id per delegate in the first execution plan, such as ``"TensorRTBackend"``.

    Examples:
        Requires a real ``.pte`` artifact and the ``executorch`` package, so it is not runnable standalone.

        >>> callable(_delegate_ids)
        True
    """
    from executorch.exir._serialize import _deserialize_pte_binary

    plan = _deserialize_pte_binary(pte_path.read_bytes()).program.execution_plan[0]
    return [delegate.id for delegate in plan.delegates]


@_requires_executorch_tensorrt
@_requires_cuda
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_executorch_tensorrt
class TestExecuTorchTensorRTDelegate:
    """The TensorRT delegate must take the whole graph and produce a ``.pte`` that predicts like the PyTorch model."""

    def test_shipped_pte_is_one_tensorrt_engine(self, nano_tensorrt_pte: Path) -> None:
        """The exported ``.pte`` holds a single TensorRT delegate, so no op of the model runs outside the engine."""
        assert _delegate_ids(nano_tensorrt_pte) == ["TensorRTBackend"]

    def test_shipped_pte_only_copies_outside_the_engine(self, nano_tensorrt_pte: Path) -> None:
        """The only kernel calls outside the engine move the input to the GPU and the two outputs back to the CPU.

        Any other kernel call would be a model op TensorRT did not convert, running on ExecuTorch's CPU kernels.
        """
        assert sorted(_portable_kernel_call_names(nano_tensorrt_pte)) == [
            "et_copy::_d2h_copy.out",
            "et_copy::_d2h_copy.out",
            "et_copy::_h2d_copy.out",
        ]

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_prediction_matches_native(self, task: str, tmp_path: Path) -> None:
        """``RFDETRInference`` on the TensorRT ``.pte`` returns the native boxes, scores, labels and task data."""
        assert_prediction_roundtrip("executorch", task, tmp_path, backend="tensorrt")
