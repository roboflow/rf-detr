# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared test helpers for the inference test suite.

Plain classes and functions (not pytest fixtures) shared across multiple test modules to avoid verbatim duplication.
Import with a relative import::

    from .helpers import _BaseFakeRFDETR, _DummyModel, _DummyRFDETR
"""

from __future__ import annotations

from itertools import chain, repeat
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

from rfdetr.detr import RFDETR
from rfdetr.utilities.imports import _IS_OPENCV_INSTALLED

#: Gate for tests that import or patch ``cv2``; the package is optional and imported locally where used.
_requires_opencv = pytest.mark.skipif(not _IS_OPENCV_INSTALLED, reason="requires the optional opencv package")


class _BaseFakeRFDETR(RFDETR):
    """RFDETR test double that skips weight downloads and returns a minimal model config.

    Subclasses must override ``get_model`` to supply the model context appropriate for
    the scenario under test.

    Examples:
        This class is imported directly by test modules that need a weight-free RFDETR.
    """

    def maybe_download_pretrain_weights(self) -> None:
        """Skip weight download in tests."""
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        """Return a minimal config sufficient for most test scenarios."""
        return SimpleNamespace(num_channels=3)


def _scripted_capture(script: list[tuple[bool, Any] | BaseException]) -> MagicMock:
    """Fake an OpenCV capture whose ``read`` and ``grab`` consume one shared frame script.

    ``read()`` returns the next ``(success, frame)`` entry and ``grab()`` returns only its success flag, so a
    skipped frame advances the same position a real decoder would. Exception entries are raised. After the
    script ends, every call reports a failed read.

    Args:
        script: Decoder results in frame order, or exceptions to raise at that position.

    Returns:
        A capture mock with ``read`` and ``grab`` bound to the script.

    Examples:
        >>> capture = _scripted_capture([(True, "frame-0"), (True, "frame-1"), (True, "frame-2")])
        >>> capture.read()
        (True, 'frame-0')
        >>> capture.grab()
        True
        >>> capture.read()
        (True, 'frame-2')
        >>> capture.read()
        (False, None)
    """
    entries = chain(script, repeat((False, None)))

    def advance() -> tuple[bool, Any]:
        """Return the next scripted decoder result.

        Examples:
            >>> _scripted_capture([(True, 1)]).read()  # doctest: +SKIP
            # Bound to the enclosing script; exercised through the capture above.
        """
        entry = next(entries)
        if isinstance(entry, BaseException):
            raise entry
        return entry

    capture = MagicMock()
    capture.read.side_effect = advance
    capture.grab.side_effect = lambda: advance()[0]
    return capture


class _DummyModel:
    """Minimal model stub that returns deterministic postprocessed results.

    Examples:
        >>> m = _DummyModel(labels=[0, 1])
        >>> len(m._labels)
        2
        >>> scored = _DummyModel(labels=[0, 1], scores=[0.25, 0.75])
        >>> scored.postprocess(None, torch.tensor([[4, 4]]))[0]["scores"].tolist()
        [0.25, 0.75]
        >>> half = _DummyModel(labels=[0], scores=[0.5], score_dtype=torch.float16)
        >>> half.postprocess(None, torch.tensor([[4, 4]]))[0]["scores"].dtype
        torch.float16
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        labels: list[int] | None = None,
        include_keypoints: bool = False,
        num_keypoints: int = 17,
        device: torch.device | str = "cpu",
        include_masks: bool = False,
        mask_size: int = 4,
        fill_value: float | None = None,
        scores: list[float] | None = None,
        score_dtype: torch.dtype = torch.float32,
    ) -> None:
        """Initialise stub with optional class names, label list, device, mask flag, and keypoint flag.

        Args:
            class_names: Optional class-name list forwarded to consumers that read it.
            labels: Class ids returned per detection. Defaults to a single detection labelled ``1``.
            include_keypoints: When ``True``, ``postprocess`` also emits ``keypoints`` and
                ``keypoint_precision_cholesky``.
            num_keypoints: Keypoint count per detection when ``include_keypoints`` is set.
            device: Device every returned tensor is placed on. Defaults to CPU, matching every
                existing call site; pass ``"cuda:0"`` to exercise a CUDA-resident result.
            include_masks: When ``True``, ``postprocess`` also emits a boolean ``masks`` tensor
                shaped ``(N, 1, mask_size, mask_size)``, matching the real model's
                ``PostProcess`` output convention.
            mask_size: Height/width of the emitted mask when ``include_masks`` is set.
            fill_value: When set, overrides the default scores/boxes/keypoints/keypoint-precision
                fill with this single scalar — useful for detecting a torn (partially completed)
                async device-to-host copy, where a stale zero would otherwise be indistinguishable
                from a genuine one. ``None`` preserves the original fixed values every existing
                caller already asserts on.
            scores: Per-detection scores returned by ``postprocess``, one per entry of ``labels``. ``None`` keeps
                the single fill value for every detection, which is what existing callers rely on.
            score_dtype: Dtype of the ``scores`` tensor, e.g. ``torch.float16`` for a model run with
                ``inference(dtype=torch.float16)``. Only applies when ``scores`` is given.
        """
        self.device = torch.device(device)
        self.resolution = 28
        self.model = torch.nn.Identity().to(self.device)
        self.class_names = class_names
        self._labels = labels if labels is not None else [1]
        self._include_keypoints = include_keypoints
        self._num_keypoints = num_keypoints
        self._include_masks = include_masks
        self._mask_size = mask_size
        self._fill_value = fill_value
        self._scores = scores
        self._score_dtype = score_dtype
        # Captured once, independent of `self.device`: some tests reassign `self.device` after
        # construction to simulate a model "declared" on CUDA while every tensor op stays mocked
        # CPU-side — `postprocess` must keep allocating on the device it was actually built for.
        self._result_device = self.device

    def postprocess(
        self,
        predictions: Any,
        target_sizes: torch.Tensor,
        score_threshold: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        """Return fixed scores/boxes (and optional keypoints/masks) for every image in the batch."""
        batch = target_sizes.shape[0]
        n = len(self._labels)
        score_value = self._fill_value if self._fill_value is not None else 0.9
        box_value = [self._fill_value] * 4 if self._fill_value is not None else [0.0, 0.0, 1.0, 1.0]
        kp_value = self._fill_value if self._fill_value is not None else 0.5
        kpp_value = self._fill_value if self._fill_value is not None else 0.25
        device = self._result_device
        results = []
        for _ in range(batch):
            result: dict[str, torch.Tensor] = {
                "scores": torch.tensor(self._scores, device=device, dtype=self._score_dtype)
                if self._scores is not None
                else torch.full((n,), score_value, device=device),
                "labels": torch.tensor(self._labels, device=device),
                "boxes": torch.tensor([box_value] * n, device=device),
            }
            if self._include_keypoints:
                result["keypoints"] = torch.full(
                    (n, self._num_keypoints, 3), kp_value, dtype=torch.float32, device=device
                )
                result["keypoint_precision_cholesky"] = torch.full(
                    (n, self._num_keypoints, 3), kpp_value, dtype=torch.float32, device=device
                )
            if self._include_masks:
                result["masks"] = torch.ones((n, 1, self._mask_size, self._mask_size), dtype=torch.bool, device=device)
            results.append(result)
        return results


class _DummyRFDETR(RFDETR):
    """Weight-free RFDETR that delegates to ``_DummyModel`` for all inference.

    Examples:
        >>> m = _DummyRFDETR()
        >>> isinstance(m.model, _DummyModel)
        True
    """

    def maybe_download_pretrain_weights(self) -> None:
        """Skip weight download in tests."""
        return None

    def get_model_config(self, **kwargs: object) -> SimpleNamespace:
        """Return a minimal namespace with just ``num_channels``."""
        return SimpleNamespace(num_channels=3)

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _DummyModel:
        """Return a fresh ``_DummyModel`` instance."""
        return _DummyModel()
