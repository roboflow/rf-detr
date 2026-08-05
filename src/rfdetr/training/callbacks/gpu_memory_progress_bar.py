# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Progress bar variants that append peak CUDA memory usage to the displayed metrics.

RF-DETR 1.5 reported ``torch.cuda.max_memory_allocated()`` in the training progress bar (``rfdetr.engine``). The
migration to PyTorch Lightning (PR #794) dropped ``engine.py`` entirely, along with that reporting, leaving the stock
``TQDMProgressBar`` / ``RichProgressBar`` with no GPU memory metric. These mixins restore it by overriding
``get_metrics``, the same extension point PTL itself uses for ``RichProgressBar``.

Reported value: ``max_mem`` is the peak allocation of the **local process** on ``trainer.strategy.root_device``,
formatted as whole MB. Under DDP only rank 0 renders a progress bar, so the number shown is rank 0's own peak — not a
sum, max, or mean across the cluster. Other ranks may peak higher (uneven batch shapes) without that ever surfacing.
This is pre-#794 behaviour, kept unchanged; it is documented here only because the rank-local scope is not obvious from
the rendered label.

Peak reset: ``on_train_start`` calls ``torch.cuda.reset_peak_memory_stats()``, so each ``trainer.fit()`` reports its own
peak rather than a high-water mark inherited from earlier work in the same process (model construction, a prior
``fit()``, an earlier ``predict()``). This is a **deliberate deviation** from the pre-#794 baseline, which never reset
and therefore reported a cumulative process-lifetime peak. The hook fires on every rank (``ProgressBar.setup`` only
disables *rendering* off rank 0), so each rank restarts its own counter. The call mutates *process-wide* CUDA allocator
state, so anything else reading peak stats for that device — ``DeviceStatsMonitor``, a profiler, user code — sees the
counter restart at every fit.

Scope: only ``trainer.fit()`` (training and its periodic in-training validation) renders ``max_mem``. A standalone
``RFDETR.evaluate()`` call builds its trainer with ``include_training_callbacks=False``, which still installs one of
these progress bars, but PTL's own ``TQDMProgressBar``/``RichProgressBar`` surface no metrics outside ``trainer.state.fn
== "fit"``: ``tqdm_progress.py``'s ``on_validation_end`` calls ``get_metrics()`` only under that condition, and
``rich_progress.py``'s ``MetricsTextColumn.render`` returns empty text otherwise. Their ``on_test_end`` hooks need no
such gate — neither reads metrics at all, both only reset eval-loop bookkeeping. So no metric, not just ``max_mem``,
reaches the progress bar during a standalone ``validate()``/``test()`` run. That is a PTL-wide constraint predating this
mixin, not a gap in it.
"""

from __future__ import annotations

from typing import Union

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ProgressBar, RichProgressBar, TQDMProgressBar

# PTL re-exports `rank_zero_warn` here without `__all__` (an implicit re-export mypy --strict rejects), but this is the
# path PTL's own `progress_bar.py` imports and the indirection is deliberate upstream: importing it wires up
# `rank_zero_module.log`. Prefer it over reaching into `lightning_fabric`.
from pytorch_lightning.utilities.rank_zero import rank_zero_warn  # type: ignore[attr-defined]

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


class _GpuMemoryMetricsMixin(ProgressBar):
    """Adds a ``max_mem`` entry (peak allocated CUDA memory, in MB) when training on CUDA.

    Always mix in *before* a concrete PTL progress bar (see ``GpuMemoryTQDMProgressBar``). The ``ProgressBar`` base is
    declared only to pin the MRO: it makes ``super()`` statically resolvable, while at runtime every ``super()`` call
    below still lands on the concrete bar to the right of this mixin, never on ``ProgressBar`` itself.
    """

    def on_train_start(self, trainer: Trainer, pl_module: LightningModule) -> None:
        """Reset the CUDA peak-memory counter so ``max_mem`` measures this run alone.

        ``torch.cuda.reset_peak_memory_stats`` has no internal availability guard — it dispatches straight to
        ``torch._C._cuda_resetPeakMemoryStats``, which does not exist on a CPU-only build — so it is called behind the
        same ``_is_cuda`` predicate as the metric itself. The reset mutates process-wide CUDA allocator state and is a
        deliberate deviation from the pre-#794 baseline; see the module docstring.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``LightningModule`` being trained.
        """
        device = trainer.strategy.root_device
        if _is_cuda(device):
            torch.cuda.reset_peak_memory_stats(device)
        super().on_train_start(trainer, pl_module)

    def get_metrics(self, trainer: Trainer, pl_module: LightningModule) -> _Metrics:
        """Return the progress bar metrics, with ``max_mem`` appended on CUDA.

        An explicitly logged ``max_mem`` wins: if the base metrics already carry that key with a different value (a
        user's own ``self.log("max_mem", ..., prog_bar=True)``), the logged value is kept and a rank-zero warning is
        emitted, mirroring how ``ProgressBar.get_metrics`` reports name collisions.

        Args:
            trainer: The Lightning Trainer instance.
            pl_module: The ``LightningModule`` being trained.

        Returns:
            The metrics dict produced by the base progress bar, plus ``max_mem`` when
            ``trainer.strategy.root_device`` is an active CUDA device.
        """
        items: _Metrics = super().get_metrics(trainer, pl_module)
        device = trainer.strategy.root_device
        if _is_cuda(device):
            max_mem = f"{torch.cuda.max_memory_allocated(device) / _BYTES_TO_MB:.0f} MB"
            if "max_mem" in items and items["max_mem"] != max_mem:
                rank_zero_warn(
                    "The progress bar already tracks a metric with the name 'max_mem', and"
                    " `self.log('max_mem', ..., prog_bar=True)` takes precedence over the peak CUDA memory reported by"
                    f" `{type(self).__name__}`. If this is undesired, change the name of the logged metric."
                )
            items.setdefault("max_mem", max_mem)
        return items


class GpuMemoryTQDMProgressBar(_GpuMemoryMetricsMixin, TQDMProgressBar):
    """``TQDMProgressBar`` with peak CUDA memory usage in the postfix."""


class GpuMemoryRichProgressBar(_GpuMemoryMetricsMixin, RichProgressBar):
    """``RichProgressBar`` with peak CUDA memory usage in the metrics column."""
