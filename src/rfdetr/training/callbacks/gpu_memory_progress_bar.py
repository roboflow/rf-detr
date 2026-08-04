# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Progress bar variants that append peak GPU memory usage to the displayed metrics.

RF-DETR 1.5 reported ``torch.cuda.max_memory_allocated()`` in the training progress bar (``rfdetr.engine``). The
migration to PyTorch Lightning (PR #794) dropped ``engine.py`` entirely, along with that reporting, leaving the stock
``TQDMProgressBar`` / ``RichProgressBar`` with no GPU memory metric. These mixins restore it by overriding
``get_metrics``, the same extension point PTL itself uses for ``RichProgressBar``.

Scope: only ``trainer.fit()`` (training and its periodic in-training validation) renders ``max_mem``. A standalone
``RFDETR.evaluate()`` call builds its trainer with ``include_training_callbacks=False``, which still installs one of
these progress bars, but PTL's own ``TQDMProgressBar``/``RichProgressBar`` never call ``get_metrics()`` outside
``trainer.state.fn == "fit"`` (``tqdm_progress.py``'s ``on_validation_end``/``on_test_end`` and ``rich_progress.py``'s
``MetricsTextColumn.render`` all gate on it) — so no metric, not just ``max_mem``, reaches the progress bar during a
standalone ``validate()``/``test()`` run. That is a pre-existing PTL-wide constraint, not a gap in this mixin.
"""

from __future__ import annotations

from typing import Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

_BYTES_TO_MB = 1024.0 * 1024.0

_Metrics = dict[str, Union[int, str, float, dict[str, float]]]


def _is_cuda(device: torch.device) -> bool:
    """Return True if device is a CUDA device with an active CUDA context."""
    return (
        isinstance(device, torch.device)
        and device.type == "cuda"
        and torch.cuda.is_available()
        and torch.cuda.is_initialized()  # type: ignore[no-untyped-call]
    )


class _GpuMemoryMetricsMixin:
    """Adds a ``max_mem`` entry (peak allocated memory, in MB) when training on CUDA."""

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> _Metrics:
        """Return the progress bar metrics, with ``max_mem`` appended on CUDA.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``LightningModule`` being trained.

        Returns:
            The metrics dict produced by the base progress bar, plus ``max_mem`` when
            ``trainer.strategy.root_device`` is an active CUDA device.
        """
        items: _Metrics = super().get_metrics(trainer, pl_module)  # type: ignore[misc]
        device = trainer.strategy.root_device
        if _is_cuda(device):
            items["max_mem"] = f"{torch.cuda.max_memory_allocated(device) / _BYTES_TO_MB:.0f}MB"
        return items


class GpuMemoryTQDMProgressBar(_GpuMemoryMetricsMixin, TQDMProgressBar):
    """``TQDMProgressBar`` with peak GPU memory usage in the postfix."""


class GpuMemoryRichProgressBar(_GpuMemoryMetricsMixin, RichProgressBar):
    """``RichProgressBar`` with peak GPU memory usage in the metrics column."""
