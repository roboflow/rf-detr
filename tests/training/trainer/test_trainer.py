# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for build_trainer() — callback stack and config coercion.

Exercise keypoint accumulation through real Lightning CPU/Gloo DDP.

Tests for resuming training from checkpoint.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar
from pytorch_lightning.strategies import DDPStrategy
from torch import Tensor, nn
from torch.distributed.algorithms.ddp_comm_hooks.default_hooks import allreduce_hook
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from rfdetr import RFDETRNano
from rfdetr.config import RFDETRSmallConfig, TrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.training import build_trainer
from rfdetr.training.callbacks import GPUMemoryRichProgressBar, GPUMemoryTQDMProgressBar
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.logger import get_logger
from tests.training.helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel


class TestProgressBarCallbacks:
    """build_trainer() must install the right progress bar callback for each mode.

    The installed callbacks are the ``GPUMemory*`` subclasses (see ``gpu_memory_progress_bar.py``), so membership is
    checked with ``isinstance`` against the base PTL classes rather than exact ``type(cb) in [...]`` matches.
    """

    @pytest.mark.parametrize(
        ("progress_bar", "installed_cls", "other_cls", "gpu_memory_cls"),
        [
            pytest.param("rich", RichProgressBar, TQDMProgressBar, GPUMemoryRichProgressBar, id="rich"),
            pytest.param("tqdm", TQDMProgressBar, RichProgressBar, GPUMemoryTQDMProgressBar, id="tqdm"),
        ],
    )
    def test_installs_matching_progress_bar(
        self, base_model_config, base_train_config, progress_bar, installed_cls, other_cls, gpu_memory_cls
    ):
        """progress_bar='rich'/'tqdm' installs the matching GPU-memory bar and none of the other kind."""
        trainer = build_trainer(base_train_config(progress_bar=progress_bar), base_model_config(), accelerator="cpu")
        assert any(isinstance(cb, installed_cls) for cb in trainer.callbacks)
        assert not any(isinstance(cb, other_cls) for cb in trainer.callbacks)
        assert any(isinstance(cb, gpu_memory_cls) for cb in trainer.callbacks)

    def test_progress_bar_refreshes_every_five_batches(self, base_model_config, base_train_config):
        """The installed progress bar callback should refresh every five batches."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, TQDMProgressBar))

        assert progress_bar.refresh_rate == 5

    def test_rich_bar_leaves_completed_epochs(self, base_model_config, base_train_config):
        """progress_bar='rich' must leave each completed epoch bar in the terminal history.

        Without ``leave=True``, ``RichProgressBar`` overwrites the previous epoch's bar in place, so a run that never
        improves ``checkpoint_best_regular.pth`` leaves no visible record of earlier epochs' metrics.
        """
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, RichProgressBar))

        assert progress_bar._leave is True

    def test_no_progress_bar_for_none(self, base_model_config, base_train_config):
        """progress_bar=None must not add any progress bar callback."""
        mc = base_model_config()
        tc = base_train_config(progress_bar=None)
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert not any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)

    def test_disables_pre_training_sanity_validation(self, base_model_config, base_train_config):
        """RF-DETR training should start directly with the first training epoch."""
        trainer = build_trainer(base_train_config(), base_model_config(), accelerator="cpu")

        assert trainer.num_sanity_val_steps == 0

    def test_num_sanity_val_steps_is_configurable(self, base_model_config, base_train_config):
        """TrainConfig.num_sanity_val_steps should reach the underlying Trainer unchanged."""
        tc = base_train_config(num_sanity_val_steps=2)
        trainer = build_trainer(tc, base_model_config(), accelerator="cpu")

        assert trainer.num_sanity_val_steps == 2

    def test_num_sanity_val_steps_kwarg_overrides_default(self, base_model_config, base_train_config):
        """A caller-supplied ``num_sanity_val_steps`` kwarg must survive the trainer_config.update() merge.

        build_trainer() disables sanity validation by default (see the previous test). A caller that explicitly wants
        sanity validation back must be able to re-enable it via ``**trainer_kwargs``, the same override mechanism used
        for other PTL-native flags like ``fast_dev_run``.
        """
        trainer = build_trainer(base_train_config(), base_model_config(), accelerator="cpu", num_sanity_val_steps=2)

        assert trainer.num_sanity_val_steps == 2


class TestRichProgressBarLoggerIntegration:
    """get_logger()'s handlers must track Rich's Live redirect during an actual training run.

    ``TestProgressBarCallbacks`` above only checks which callback/flag build_trainer() installs; it never runs
    ``Trainer.fit()``, so it can't catch a handler writing through a stale pre-fit ``sys.stdout`` instead of Rich's
    redirected one — the exact corruption ``_RedirectAwareStreamHandler`` exists to prevent (see
    ``rfdetr.utilities.logger``). This runs a real ``build_trainer() + trainer.fit(fast_dev_run=2)`` with
    ``progress_bar="rich"`` (same fixture pattern as ``TestProgressBarEndToEnd`` in
    ``tests/training/callbacks/test_gpu_memory_progress_bar_callback.py``, no real dataset or model weights) and logs
    mid-epoch from inside a callback hook, while Rich's ``Live`` display is genuinely active.
    """

    def test_logger_tracks_rich_live_redirect_during_fit(self, base_model_config, base_train_config, mocker) -> None:
        """A logger constructed before fit() (mirrors the real module-import call site) must still follow the redirect
        Rich installs for the duration of ``Trainer.fit()``, not the stream it saw at construction time.

        Rich's ``Live.start()`` only redirects ``sys.stdout`` when ``console.is_terminal`` is true (see
        ``rich/live.py``), which is false under pytest's non-tty capture — so without forcing it, this test would pass
        trivially (no redirect ever happens) regardless of whether the handler tracks it. ``force_terminal=True`` makes
        the run behave like the real terminal training session the bug actually occurs in.
        """
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich", use_ema=False, run_test=False)

        # Constructed before fit() starts, exactly like the real `logger = get_logger()` call sites which run at
        # module-import time, long before Rich's Live redirect is ever installed.
        logger = get_logger("rf-detr-test-rich-live-redirect")
        stdout_handler = next(h for h in logger.handlers if h.level == logging.DEBUG)
        pre_fit_stream = stdout_handler.stream

        seen_during_batch: list[tuple[object, object]] = []

        class _LogDuringBatch(Callback):
            """Log once at batch end and capture the active stdout stream."""

            def on_train_batch_end(
                self,
                trainer: Trainer,
                pl_module: LightningModule,
                outputs: Any,
                batch: Any,
                batch_idx: int,
            ) -> None:
                """Record the redirected stream after emitting a training-time log message."""
                logger.info("checkpoint-style message mid-epoch")
                seen_during_batch.append((stdout_handler.stream, sys.stdout))

        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
        )
        mocker.patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4))
        mocker.patch(
            "rfdetr.training.module_model.get_param_dict", side_effect=lambda args, model: _make_param_dicts(model)
        )
        module = RFDETRModelModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(tc, mc, accelerator="cpu", fast_dev_run=2)
        progress_bar_cb = next(cb for cb in trainer.callbacks if isinstance(cb, RichProgressBar))
        progress_bar_cb._console_kwargs = {"force_terminal": True}
        trainer.callbacks.append(_LogDuringBatch())
        trainer.fit(module, datamodule=datamodule)

        assert seen_during_batch, "on_train_batch_end never fired — fast_dev_run did not run a batch"
        handler_stream, live_stdout = seen_during_batch[0]
        assert live_stdout is not pre_fit_stream, (
            "RichProgressBar never redirected sys.stdout during fit — this test would pass even against the "
            "pre-fix plain logging.StreamHandler(sys.stdout), so it would not catch the regression."
        )
        assert handler_stream is live_stdout, (
            "the handler wrote through a stale stream instead of Rich's Live-redirected sys.stdout — this is the "
            "duplicated/garbled epoch bar bug _RedirectAwareStreamHandler exists to prevent."
        )


class TestCoerceLegacyProgressBar:
    """_coerce_legacy_progress_bar must normalise legacy bool values."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [(True, "tqdm"), (False, None), ("rich", "rich"), ("tqdm", "tqdm"), (None, None)],
    )
    def test_coerce(self, base_train_config, value, expected):
        """progress_bar field normalises legacy bool and passes through string/None."""
        tc = base_train_config(progress_bar=value)
        assert tc.progress_bar == expected


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


#: Skip a test node that runs DDP on the Gloo backend, which some PyTorch builds lack.
_requires_gloo = pytest.mark.skipif(
    not (torch.distributed.is_available() and torch.distributed.is_gloo_available()),
    reason="PyTorch build lacks the Gloo backend",
)


@pytest.mark.ddp
@_requires_gloo
@pytest.mark.parametrize("accumulation", [1, 2, 4])
def test_keypoint_ddp_reduces_only_at_window_boundaries(tmp_path: Path, accumulation: int) -> None:
    """Real Lightning backward must suppress intermediate reductions and flush a final partial window."""
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


@pytest.fixture
def mock_build_trainer(mocker):
    """Patch the PTL training stack and return the ``build_trainer`` mock, so ``RFDETR.train()`` stops at ``fit()``."""
    mocker.patch("rfdetr.training.RFDETRModelModule")
    mocker.patch("rfdetr.training.RFDETRDataModule")
    return mocker.patch("rfdetr.training.build_trainer")


@pytest.fixture
def rf_detr_caplog(caplog: pytest.LogCaptureFixture) -> pytest.LogCaptureFixture:
    """Capture "rf-detr" logger warnings in ``caplog``.

    ``get_logger()`` (``rfdetr.utilities.logger``) sets ``propagate = False`` on the "rf-detr" logger, while
    ``caplog``'s handler lives on the root logger, so propagation is re-enabled for the test (same pattern as
    ``tests/utilities/test_state_dict.py``).
    """
    rf_detr_logger = logging.getLogger("rf-detr")
    prev_propagate = rf_detr_logger.propagate
    rf_detr_logger.propagate = True
    with caplog.at_level(logging.WARNING, logger="rf-detr"):
        yield caplog
    rf_detr_logger.propagate = prev_propagate


def _train(tmp_path: Path, output_dir: Path, resume: Path | None = None) -> RFDETRNano:
    """Run ``RFDETRNano.train()`` for one epoch on CPU (PTL stack mocked by ``mock_build_trainer``) and return the
    model.

    Args:
        tmp_path: Pytest temporary directory, used as ``dataset_dir``.
        output_dir: Path passed as ``output_dir=``.
        resume: Optional checkpoint path passed as ``resume=``.

    Examples:
        >>> _train  # doctest: +SKIP
        Needs patched training dependencies and a pytest tmp_path fixture.
    """
    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.train(
        dataset_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        resume=None if resume is None else str(resume),
        device="cpu",
    )
    return model


def test_resume_with_completed_epochs_returns_early(tmp_path: Path, mock_build_trainer) -> None:
    """Training reaches trainer.fit() when resuming a run that already completed its epochs.

    In the legacy engine.py path, ``start_epoch=epochs`` caused the training loop to be skipped (``range(start_epoch,
    epochs)`` was empty), which triggered an ``UnboundLocalError`` when accessing ``test_stats``. In the PTL path,
    resuming happens automatically via ``ckpt_path`` and does not require a ``start_epoch`` kwarg.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)

    _train(tmp_path, output_dir)

    mock_build_trainer.return_value.fit.assert_called_once()


def test_resume_with_completed_epochs_skips_on_train_end_callback(tmp_path: Path, mock_build_trainer) -> None:
    """Old-style ``model.callbacks["on_train_end"]`` callbacks are not forwarded to PTL.

    In the legacy engine.py path they ran at the end of training (even when the loop was skipped). In the PTL path the
    callback dict on the model instance is not consulted; use PTL ``Callback`` objects via ``build_trainer()``.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    callback_calls = 0

    def _callback() -> None:
        nonlocal callback_calls
        callback_calls += 1

    model = RFDETRNano(pretrain_weights=None, num_classes=3, device="cpu")
    model.callbacks["on_train_end"].append(_callback)
    model.train(
        dataset_dir=str(tmp_path),
        epochs=1,
        batch_size=1,
        grad_accum_steps=1,
        output_dir=str(output_dir),
        device="cpu",
    )

    assert callback_calls == 0


def test_resume_with_callback_state_warns_it_resumes_correctly(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with a non-empty "callbacks" key gets the "will resume correctly" warning.

    Regression test for the RFDETR.train() resume warning: without peeking at the checkpoint's own "callbacks" key
    first, this branch and ``test_resume_without_callback_state_warns_it_restarts_cold`` would be indistinguishable and
    the warning would always claim full restoration.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)

    _train(tmp_path, output_dir, ckpt_path)

    assert any("matching configured callbacks" in record.message for record in rf_detr_caplog.records)
    assert not any("predates callback-state persistence" in record.message for record in rf_detr_caplog.records)


def test_resume_without_callback_state_warns_it_restarts_cold(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """A lightweight checkpoint with no "callbacks" key (or an empty one) warns everything restarts cold.

    Simulates a checkpoint written before ``BestModelCallback._build_checkpoint_payload`` persisted per-callback state
    (or a run where every registered callback had empty state): PTL's ``_call_callbacks_load_state_dict`` no-ops on a
    missing/empty "callbacks" key, silently skipping best-score/EMA/early-stopping restoration.
    """
    output_dir = tmp_path / "train_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}}, ckpt_path)

    _train(tmp_path, output_dir, ckpt_path)

    assert any("restart cold" in record.message for record in rf_detr_caplog.records)
    assert not any("matching configured callbacks" in record.message for record in rf_detr_caplog.records)


def test_resume_dirpath_mismatch_warns_best_score_restarts_fresh(
    tmp_path: Path, mock_build_trainer, rf_detr_caplog: pytest.LogCaptureFixture
) -> None:
    """Resuming with output_dir different from the checkpoint's own directory warns best-score tracking resets.

    Mirrors PTL's own ``ModelCheckpoint.load_state_dict()`` dirpath gate (callback-level regression test:
    ``tests/training/callbacks/test_best_model_callback.py::test_best_model_score_not_restored_when_output_dir_changes``);
    this is the config-level warning that surfaces the same gap to a user calling ``RFDETR.train(resume=...)``.
    """
    source_dir = tmp_path / "source_output"
    source_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = source_dir / "checkpoint_best_regular.pth"
    torch.save({"model": {}, "callbacks": {"BestModelCallback": {"best_model_score": 0.7}}}, ckpt_path)
    other_output_dir = tmp_path / "other_output_dir"
    other_output_dir.mkdir(parents=True, exist_ok=True)

    _train(tmp_path, other_output_dir, ckpt_path)

    assert any(
        "best-score tracking" in record.message and "restarts fresh" in record.message
        for record in rf_detr_caplog.records
    )
