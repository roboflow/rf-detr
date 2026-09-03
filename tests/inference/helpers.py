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


class _IdentityAcceptingReturnEmbeddings(torch.nn.Module):
    """``torch.nn.Identity``-like stub whose ``forward`` also accepts (and ignores) ``return_embeddings``.

    ``predict()`` always calls the unoptimized model with ``return_embeddings=<bool>``, mirroring the real
    ``LWDETR.forward`` signature. Plain ``torch.nn.Identity`` doesn't accept that kwarg, so test doubles standing
    in for the base model use this instead.

    Examples:
        >>> stub = _IdentityAcceptingReturnEmbeddings()
        >>> x = torch.zeros(1)
        >>> torch.equal(stub(x, return_embeddings=True), x)
        True
    """

    def __init__(self, hidden_dim: int = 4) -> None:
        """Initialise the stub, remembering ``hidden_dim`` for the synthetic embeddings tensor."""
        super().__init__()
        self.hidden_dim = hidden_dim
        self.last_return_embeddings: bool | None = None

    def forward(self, x: torch.Tensor, return_embeddings: bool = False) -> torch.Tensor:
        """Return the input unchanged, recording ``return_embeddings`` for later assertions."""
        self.last_return_embeddings = return_embeddings
        return x


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
        include_embeddings: bool = False,
        embedding_dim: int = 4,
    ) -> None:
        """Initialise stub with optional class names, label list, keypoint flag, and embeddings flag."""
        self.device = torch.device("cpu")
        self.resolution = 28
        self.model = _IdentityAcceptingReturnEmbeddings(hidden_dim=embedding_dim)
        self.class_names = class_names
        self._labels = labels if labels is not None else [1]
        self._include_keypoints = include_keypoints
        self._num_keypoints = num_keypoints
        self._include_embeddings = include_embeddings
        self._embedding_dim = embedding_dim

    def postprocess(
        self,
        predictions: Any,
        target_sizes: torch.Tensor,
        score_threshold: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
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
            if self._include_embeddings:
                # Identifiable per-label embeddings: row i is filled with value i, so tests can assert on content.
                result["embeddings"] = torch.stack(
                    [torch.full((self._embedding_dim,), float(i)) for i in range(len(self._labels))]
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

    def get_model(self, config: SimpleNamespace, *, trust_checkpoint: bool = False) -> _DummyModel:
        """Return a fresh ``_DummyModel`` instance."""
        return _DummyModel()
