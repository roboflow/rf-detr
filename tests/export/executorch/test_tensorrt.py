# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""ExecuTorch ``backend="tensorrt"`` lowering and runtime on an NVIDIA GPU (``@pytest.mark.e2e_executorch_tensorrt``).

The unit tests in ``test_export.py`` mock Torch-TensorRT away. These export real models through it and run the ``.pte``
with ExecuTorch's runtime and the Torch-TensorRT delegate, so they need the ``executorch-tensorrt`` extra and a CUDA
device.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from rfdetr import RFDETRNano
from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export.imports import _IS_TORCH_TENSORRT_EXECUTORCH_RUNTIME_INSTALLED, _IS_TORCH_TENSORRT_INSTALLED
from rfdetr.utilities.reproducibility import seed_all
from tests._markers import _requires_cuda
from tests.export.conftest import _prediction_model_for_task, assert_prediction_roundtrip
from tests.export.executorch.test_export import _portable_kernel_call_names

# Same reason as test_coreml.py's `_EXPORT_SEED`: the module-scoped fixture runs before the function-scoped
# autouse seed reset.
_EXPORT_SEED = 0

_requires_executorch_tensorrt = pytest.mark.skipif(
    not _IS_EXECUTORCH_AVAILABLE
    or not _IS_TORCH_TENSORRT_INSTALLED
    or not _IS_TORCH_TENSORRT_EXECUTORCH_RUNTIME_INSTALLED,
    reason="needs the executorch-tensorrt extra",
)

#: Skip on a GPU older than Ampere: Torch-TensorRT 2.15 pulls in TensorRT 11.4, which builds attention with
#: ``IAttention``, and that layer needs compute capability 8.0 or higher. On a Turing T4 (7.5) the engine build fails.
#: A host without CUDA is left to ``_requires_cuda``. The reason's opening words are what the CI job's skip allow-list
#: matches, so change both together.
_requires_ampere = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() < (8, 0),
    reason="TensorRT 11 attention needs compute capability 8.0 or newer",
)


#: Device-to-host copies the ``.pte`` of each task holds: one per model output (detection ``dets`` and ``labels``;
#: segmentation adds ``masks``; keypoints add ``keypoints``). Every ``.pte`` also holds one host-to-device input copy.
_EXPECTED_D2H_COPIES = {"detect": 2, "segment": 3, "keypoints": 3}


@pytest.fixture(scope="module", params=list(_EXPECTED_D2H_COPIES))
def tensorrt_pte(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> tuple[str, Path]:
    """Lower an untrained model of one task through the TensorRT delegate and return the task and its ``.pte``.

    Detection is the default ``RFDETRNano``; segmentation and keypoints are the small random models the prediction round
    trips use.
    """
    task = request.param
    seed_all(_EXPORT_SEED)
    detector = RFDETRNano(pretrain_weights=None) if task == "detect" else _prediction_model_for_task(task)
    out_dir = tmp_path_factory.mktemp(f"executorch_tensorrt_{task}")
    return task, Path(detector.export(output_dir=str(out_dir), format="executorch", backend="tensorrt", verbose=False))


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
@_requires_ampere
@pytest.mark.gpu
@pytest.mark.integration
@pytest.mark.e2e_executorch_tensorrt
class TestExecuTorchTensorRTDelegate:
    """The TensorRT delegate must take the whole graph and produce a ``.pte`` that predicts like the PyTorch model."""

    def test_shipped_pte_is_one_tensorrt_engine(self, tensorrt_pte: tuple[str, Path]) -> None:
        """The exported ``.pte`` of every task holds a single TensorRT delegate, so no op runs outside the engine."""
        _, pte = tensorrt_pte
        assert _delegate_ids(pte) == ["TensorRTBackend"]

    def test_shipped_pte_only_copies_outside_the_engine(self, tensorrt_pte: tuple[str, Path]) -> None:
        """The only kernel calls outside the engine move the input to the GPU and each output back to the CPU.

        Any other kernel call would be a model op TensorRT did not convert, running on ExecuTorch's CPU kernels. A task
        with more outputs holds more device-to-host copies.
        """
        task, pte = tensorrt_pte
        expected = ["et_copy::_d2h_copy.out"] * _EXPECTED_D2H_COPIES[task] + ["et_copy::_h2d_copy.out"]
        assert sorted(_portable_kernel_call_names(pte)) == sorted(expected)

    @pytest.mark.parametrize("task", ["detect", "segment", "keypoints"])
    def test_prediction_matches_native(self, task: str, tmp_path: Path) -> None:
        """``RFDETRInference`` on the TensorRT ``.pte`` returns the native boxes, scores, labels and task data."""
        assert_prediction_roundtrip("executorch", task, tmp_path, backend="tensorrt")
