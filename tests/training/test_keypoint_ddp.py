# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Exercise keypoint accumulation through real Lightning CPU/Gloo DDP."""

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor, nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.training.module_model import RFDETRModelModule


class _RankLossModel(nn.Module):
    """Supply rank-distinct linear gradients and one permanently unused parameter."""

    def __init__(self) -> None:
        """Initialize deterministic float64 parameters without model downloads."""
        super().__init__()
        self.weight = nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.unused = nn.Parameter(torch.zeros((), dtype=torch.float64))

    def forward(self, samples: Tensor, targets: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Return a linear loss with a distinct coefficient on each rank.

        Examples:
            >>> model = _RankLossModel()
            >>> model(torch.tensor(2), [])['pred_logits'].item()
            0.0
        """
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        return {"pred_logits": self.weight * (rank + 1) * samples.square()}


class _LinearCriterion(SetCriterion):
    """Keep production distributed box normalization with a simple linear numerator."""

    def __init__(self) -> None:
        """Configure a one-group criterion whose loss needs no matching."""
        super().__init__(1, HungarianMatcher(), {"loss_ce": 1.0}, 0.25, [])

    def forward(
        self, outputs: dict[str, Tensor], targets: list[dict[str, Tensor]], num_boxes: Tensor
    ) -> dict[str, Tensor]:
        """Normalize the supplied numerator with the production caller's denominator.

        Examples:
            >>> criterion = _LinearCriterion()
            >>> criterion({'pred_logits': torch.tensor(6.)}, [], torch.tensor(2.))
            {'loss_ce': tensor(3.)}
        """
        return {"loss_ce": outputs["pred_logits"] / num_boxes}


class _SGDModule(RFDETRModelModule):
    """Use plain SGD to make each real RF-DETR training-step update independently calculable."""

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Use unit learning rate and no scheduler for the analytic update oracle.

        Examples:
            Requires a constructed Lightning training module.
            >>> callable(_SGDModule.configure_optimizers)  # doctest: +SKIP
            True
        """
        return torch.optim.SGD(self.parameters(), lr=1.0)


def _count_reduction(state: dict[str, int], bucket: torch.distributed.GradBucket) -> torch.futures.Future[Tensor]:
    """Count actual gradient buckets while preserving DDP's mean reduction.

    Examples:
        Requires a live distributed process group and reducer-owned bucket.
        >>> callable(_count_reduction)  # doctest: +SKIP
        True
    """
    state["reductions"] += 1
    return allreduce_hook(None, bucket)


class _CheckWindows(Callback):
    """Assert reduction timing and effective-batch updates on both ranks after every batch."""

    def __init__(self, output_dir: Path, accumulation: int) -> None:
        """Keep per-process observations and the independent scalar reference state."""
        self.output_dir = output_dir
        self.accumulation = accumulation
        self.state = {"reductions": 0}
        self.expected = 0.0
        self.window: list[int] = []
        self.steps = 0
        self.observations: list[dict[str, Any]] = []

    def on_train_start(self, trainer: Trainer, pl_module: RFDETRModelModule) -> None:
        """Register directly on CPU DDP, which Lightning's CUDA-only hook option skips.

        Examples:
            Requires a Trainer with its real two-rank DDP wrapper initialized.
            >>> callable(_CheckWindows.on_train_start)  # doctest: +SKIP
            True
        """
        assert isinstance(trainer.strategy.model, DistributedDataParallel)
        trainer.strategy.model.register_comm_hook(self.state, _count_reduction)

    def on_train_batch_end(
        self, trainer: Trainer, pl_module: RFDETRModelModule, outputs: Any, batch: Any, batch_idx: int
    ) -> None:
        """Check intermediate silence and full/partial-window updates against global sums.

        Examples:
            Requires a completed distributed training batch on each rank.
            >>> callable(_CheckWindows.on_train_batch_end)  # doctest: +SKIP
            True
        """
        self.window.append(batch_idx + 1)
        closes_window = len(self.window) == self.accumulation or batch_idx == 2
        if closes_window:
            # Two ranks have coefficients 1 and 2; both see n boxes and numerator n**2.
            self.expected -= 3 * sum(n * n for n in self.window) / (2 * sum(self.window))
            self.window.clear()
            self.steps += 1
        assert self.state["reductions"] == self.steps, (batch_idx, self.state, self.steps)
        assert trainer.global_step == self.steps
        torch.testing.assert_close(
            pl_module.model.weight.detach(), torch.tensor(self.expected, dtype=torch.float64), rtol=1e-4, atol=1e-6
        )
        assert pl_module.model.unused.item() == 0.0
        self.observations.append({"batch": batch_idx, "reductions": self.state["reductions"], "weight": self.expected})
        if batch_idx == 2:
            (self.output_dir / f"rank-{trainer.global_rank}.json").write_text(json.dumps(self.observations))


@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_keypoint_ddp_reduces_only_at_window_boundaries(tmp_path: Path, accumulation: int) -> None:
    """Real Lightning backward must suppress intermediate reductions and flush a final partial window."""
    if not torch.distributed.is_available() or not torch.distributed.is_gloo_available():
        pytest.skip("PyTorch build lacks the Gloo backend")
    model_config = RFDETRSmallConfig(
        pretrain_weights=None, device="cpu", use_grouppose_keypoints=True, num_keypoints_per_class=[17]
    )
    train_config = TrainConfig(
        dataset_dir=str(tmp_path),
        output_dir=str(tmp_path),
        accelerator="cpu",
        grad_accum_steps=accumulation,
        multi_scale=False,
        compute_train_metrics=False,
        clip_max_norm=0.0,
        seed=42,
    )
    # Substitute expensive architecture construction only; the optimization and distributed paths stay real.
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_RankLossModel()),
        patch("rfdetr.training.module_model.build_criterion_from_config", return_value=(_LinearCriterion(), None)),
    ):
        module = _SGDModule(model_config, train_config)
    batches = [(torch.tensor(n), [{"labels": torch.zeros(n, dtype=torch.long)}]) for n in (1, 2, 3)]
    trainer = Trainer(
        accelerator="cpu",
        devices=2,
        precision="64-true",
        strategy=DDPStrategy(start_method="spawn", process_group_backend="gloo", find_unused_parameters=True),
        max_epochs=1,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        callbacks=[_CheckWindows(tmp_path, accumulation)],
        default_root_dir=str(tmp_path),
    )
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None, num_workers=0))
    rank_results = [json.loads((tmp_path / f"rank-{rank}.json").read_text()) for rank in range(2)]
    assert len(rank_results[0]) == 3
    assert rank_results[0] == rank_results[1]
