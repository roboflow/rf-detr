# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Regression coverage for the coalesced GPU->CPU output transfer in ``predict()``.

The existing ``predict()`` tests (``test_predict.py``) use ``_DummyModel``, whose ``postprocess()`` builds result
tensors on CPU (``helpers.py``) — that path never exercises the CUDA branch added to batch the output transfers, so it
cannot catch an incomplete device->host read or a synchronization call targeting the wrong device. These tests require
CUDA and are skipped otherwise.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from .helpers import _BaseFakeRFDETR, _DummyModel

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]

_NUM_KEYPOINTS = 4
_MASK_SIZE = 4
_TORN_READ_FILL = 7.0


class _CudaDummyRFDETR(_BaseFakeRFDETR):
    """Weight-free RFDETR whose model lives on CUDA, for exercising the async transfer path.

    ``predict()`` always wraps its result into ``sv.KeyPoints`` (never ``sv.Detections``) whenever
    keypoints are present — a pre-existing, unrelated branch — so a dummy requesting masks *and*
    keypoints together can only ever be read back as a ``KeyPoints`` object, which has no ``mask``
    field. ``include_masks``/``include_keypoints`` default to the mutually exclusive shapes real
    segmentation vs. keypoint models actually produce; pass both ``True`` only when a test genuinely
    needs to exercise every optional transfer branch without reading the mask back afterward.

    Examples:
        >>> m = _CudaDummyRFDETR.__new__(_CudaDummyRFDETR)  # doctest: +SKIP
        >>> isinstance(m, RFDETR)  # doctest: +SKIP
        True
    """

    def __init__(
        self,
        *args: object,
        include_masks: bool = False,
        include_keypoints: bool = False,
        **kwargs: object,
    ) -> None:
        """Initialise with the optional-field combination this test instance should exercise."""
        self._include_masks = include_masks
        self._include_keypoints = include_keypoints
        super().__init__(*args, **kwargs)

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _DummyModel:
        """Return a CUDA-resident ``_DummyModel`` configured per this instance's constructor flags."""
        return _DummyModel(
            labels=[1, 2, 3],
            device="cuda:0",
            include_keypoints=self._include_keypoints,
            num_keypoints=_NUM_KEYPOINTS,
            include_masks=self._include_masks,
            mask_size=_MASK_SIZE,
            fill_value=_TORN_READ_FILL,
        )


class TestPredictCudaTransfer:
    """Verifies the batched CUDA->CPU transfer is complete, ordered, non-blocking, and correctly synchronized."""

    def test_output_values_are_complete_after_async_transfer(self) -> None:
        """Every ``Detections`` field — boxes, scores, labels, mask — is fully populated after the batched transfer.

        Uses a distinctive non-zero fill (7.0) across every field so a torn (partially completed) async device-to-host
        copy leaves detectable stale values instead of being masked by a coincidentally correct zero — the prior
        fixture's zero-containing boxes could not distinguish "never written" from "written as zero". Masks only,
        no keypoints: ``predict()`` always returns ``sv.KeyPoints`` (no ``mask`` field) whenever keypoints are
        present, so a combined masks+keypoints dummy could never validate the mask read-back.
        """
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None, include_masks=True)
        detections = model.predict(img, threshold=0.5)
        assert len(detections) == 3
        assert detections.confidence == pytest.approx([_TORN_READ_FILL] * 3)
        assert detections.class_id.tolist() == [1, 2, 3]
        assert np.allclose(detections.xyxy, [[_TORN_READ_FILL] * 4] * 3, atol=1e-6)
        assert detections.mask is not None
        assert detections.mask.all(), "mask transfer is incomplete or torn"

    def test_keypoint_precision_is_complete_after_async_transfer(self) -> None:
        """Every ``KeyPoints`` field — including keypoint precision — is fully populated after the batched transfer.

        Keypoints only, no masks: ``predict()`` returns ``sv.KeyPoints`` whenever keypoints are present, and
        ``KeyPoints`` has no ``mask`` field — the mask-completeness case above is covered separately.
        """
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None, include_keypoints=True)
        keypoints = model.predict(img, threshold=0.5)
        assert len(keypoints) == 3
        assert np.allclose(keypoints.xy, _TORN_READ_FILL, atol=1e-6)
        assert np.allclose(keypoints.keypoint_confidence, _TORN_READ_FILL, atol=1e-6)
        assert np.allclose(
            keypoints.data["keypoint_precision_cholesky"],
            _TORN_READ_FILL,
            atol=1e-6,
        )

    def test_synchronize_targets_the_result_tensors_own_stream_before_any_numpy_read(self) -> None:
        """The barrier must target the tensors' own device's current stream, and land before every `.numpy()` read.

        Occurrence alone is insufficient: a refactor that moves the sync after the first read would still
        pass a test that only checks the sync happened at all. This asserts the sync call index precedes
        every recorded `.numpy()` call index, and that the synchronized stream belongs to the result
        tensors' own device (cuda:0) rather than being left to default to `torch.cuda.current_device()` —
        the two can differ on a multi-GPU host.

        ``include_source_image=False`` is required here: capturing the source image from an
        already-CUDA tensor input does its own separate, already-correctly-synchronized blocking
        ``.cpu().numpy()`` call (see the ``predict()`` docstring's Note on this) before the
        output-transfer loop even starts. Left at the default, that unrelated call is the first
        ``.numpy()`` recorded and predates this test's barrier, which is a false positive for the
        assertion below, not a regression in the transfer path under test.
        """
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None, include_masks=True, include_keypoints=True)
        call_order: list[str] = []

        def _record_sync(*args: object, **kwargs: object) -> torch.cuda.Stream:
            call_order.append("sync")
            return real_current_stream(*args, **kwargs)

        real_current_stream = torch.cuda.current_stream
        real_numpy = torch.Tensor.numpy

        def _record_numpy(self: torch.Tensor) -> np.ndarray:
            call_order.append("numpy")
            return real_numpy(self)

        with (
            patch("torch.cuda.current_stream", side_effect=_record_sync) as stream_spy,
            patch.object(torch.Tensor, "numpy", _record_numpy, create=False),
        ):
            model.predict(img, threshold=0.5, include_source_image=False)

        assert stream_spy.call_count >= 1
        called_devices = [torch.device(c.args[0]) if c.args else None for c in stream_spy.call_args_list]
        assert torch.device("cuda:0") in called_devices, f"expected current_stream(cuda:0), got: {called_devices}"

        first_sync_index = call_order.index("sync")
        numpy_indices = [i for i, event in enumerate(call_order) if event == "numpy"]
        assert numpy_indices, "expected at least one .numpy() call"
        assert all(first_sync_index < i for i in numpy_indices), (
            f"synchronize must precede every .numpy() read; call order was: {call_order}"
        )

    def test_output_transfer_uses_non_blocking_copies(self) -> None:
        """`non_blocking=True` must actually reach the output-side `.to()` calls, not silently degrade to blocking.

        A regression that drops `non_blocking=True` from the output transfer would pass the value- and
        ordering-based tests above bit-identically — only the PR's entire perf gain would be lost, invisibly.
        This spies `torch.Tensor.to` (the same pattern used by `test_already_cuda_tensor_input_is_not_pinned`
        and `test_cuda_tensor_input_to_cpu_model_is_not_non_blocking` in `test_predict.py` for the input side)
        and asserts every CPU-bound `.to()` call for the result tensors was issued with `non_blocking=True`.
        """
        img = torch.rand(3, 28, 28, device="cuda:0")
        model = _CudaDummyRFDETR(pretrain_weights=None, include_masks=True, include_keypoints=True)
        real_to = torch.Tensor.to
        cpu_bound_calls: list[bool] = []

        def _spy_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            target = args[0] if args else kwargs.get("device")
            if self.is_cuda and str(target) == "cpu":
                cpu_bound_calls.append(bool(kwargs.get("non_blocking", False)))
            return real_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", _spy_to, create=False):
            model.predict(img, threshold=0.5)

        assert cpu_bound_calls, "expected at least one CUDA->CPU .to() call"
        assert all(cpu_bound_calls), (
            f"expected every CUDA->CPU transfer to use non_blocking=True, got: {cpu_bound_calls}"
        )

    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires a second CUDA device")
    def test_synchronize_distinguishes_result_device_from_process_current_device(self) -> None:
        """The sync must target `boxes.device`, not `torch.cuda.current_device()`, when the two differ.

        On a single-GPU runner both spellings resolve to the same device, so a regression to `torch.cuda.synchronize()`
        (device-current, not tensor-current) would be invisible. This pins the dummy model to `cuda:1` while leaving the
        process default device at `cuda:0`, so only a correct, tensor-scoped sync call is observed on `cuda:1`.
        """
        torch.cuda.set_device(0)
        img = torch.rand(3, 28, 28, device="cuda:1")

        class _SecondDeviceDummyRFDETR(_BaseFakeRFDETR):
            def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _DummyModel:
                return _DummyModel(labels=[1, 2, 3], device="cuda:1")

        model = _SecondDeviceDummyRFDETR(pretrain_weights=None)
        with patch("torch.cuda.current_stream", wraps=torch.cuda.current_stream) as spy:
            model.predict(img, threshold=0.5)
        called_devices = [torch.device(c.args[0]) if c.args else None for c in spy.call_args_list]
        assert torch.device("cuda:1") in called_devices, (
            f"expected sync on cuda:1 (tensors' own device), not the process current device; got: {called_devices}"
        )
