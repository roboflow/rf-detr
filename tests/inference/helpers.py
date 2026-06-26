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

from types import SimpleNamespace
from typing import Any

import torch

from rfdetr.detr import RFDETR


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


class _DummyModel:
    """Minimal model stub that returns deterministic postprocessed results.

    Examples:
        >>> m = _DummyModel(labels=[0, 1])
        >>> len(m._labels)
        2
    """

    def __init__(
        self,
        class_names: list[str] | None = None,
        labels: list[int] | None = None,
        include_keypoints: bool = False,
        num_keypoints: int = 17,
    ) -> None:
        """Initialise stub with optional class names, label list, and keypoint flag."""
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = torch.nn.Identity()
        self.class_names = class_names
        self._labels = labels if labels is not None else [1]
        self._include_keypoints = include_keypoints
        self._num_keypoints = num_keypoints

    def postprocess(self, predictions: Any, target_sizes: torch.Tensor) -> list[dict[str, torch.Tensor]]:
        """Return fixed scores/boxes (and optional keypoints) for every image in the batch."""
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            result: dict[str, torch.Tensor] = {
                "scores": torch.tensor([0.9] * len(self._labels)),
                "labels": torch.tensor(self._labels),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]] * len(self._labels)),
            }
            if self._include_keypoints:
                result["keypoints"] = torch.full((len(self._labels), self._num_keypoints, 3), 0.5, dtype=torch.float32)
                result["keypoint_precision_cholesky"] = torch.full(
                    (len(self._labels), self._num_keypoints, 3), 0.25, dtype=torch.float32
                )
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

    def get_model(self, config: SimpleNamespace) -> _DummyModel:
        """Return a fresh ``_DummyModel`` instance."""
        return _DummyModel()
