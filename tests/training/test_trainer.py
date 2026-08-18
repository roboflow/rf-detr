# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for build_trainer() — callback stack and config coercion."""

from __future__ import annotations

import logging
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from pytorch_lightning import Callback, LightningModule, Trainer
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

from rfdetr.training import build_trainer
from rfdetr.training.callbacks import GPUMemoryRichProgressBar, GPUMemoryTQDMProgressBar
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.logger import get_logger

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

# ---------------------------------------------------------------------------
# TestProgressBarCallbacks — verifies the correct callback is installed
# ---------------------------------------------------------------------------


class TestProgressBarCallbacks:
    """build_trainer() must install the right progress bar callback for each mode.

    The installed callbacks are the ``GPUMemory*`` subclasses (see ``gpu_memory_progress_bar.py``), so membership is
    checked with ``isinstance`` against the base PTL classes rather than exact ``type(cb) in [...]`` matches.
    """

    def test_rich_progress_bar_installed_for_rich(self, base_model_config, base_train_config):
        """progress_bar='rich' must add a RichProgressBar and no TQDMProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, GPUMemoryRichProgressBar) for cb in trainer.callbacks)

    def test_tqdm_progress_bar_installed_for_tqdm(self, base_model_config, base_train_config):
        """progress_bar='tqdm' must add a TQDMProgressBar and no RichProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, GPUMemoryTQDMProgressBar) for cb in trainer.callbacks)

    def test_progress_bar_refresh_rate_is_five(self, base_model_config, base_train_config):
        """The installed progress bar callback should refresh every five batches."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, TQDMProgressBar))

        assert progress_bar.refresh_rate == 5

    def test_rich_progress_bar_leaves_completed_epoch_bars(self, base_model_config, base_train_config):
        """progress_bar='rich' must leave each completed epoch bar in the terminal history.

        Without ``leave=True``, ``RichProgressBar`` overwrites the previous epoch's bar in place, so a run that never
        improves ``checkpoint_best_regular.pth`` leaves no visible record of earlier epochs' metrics.
        """
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, RichProgressBar))

        assert progress_bar._leave is True

    def test_no_progress_bar_callback_for_none(self, base_model_config, base_train_config):
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

    def test_num_sanity_val_steps_kwarg_overrides_disabled_default(self, base_model_config, base_train_config):
        """A caller-supplied ``num_sanity_val_steps`` kwarg must survive the trainer_config.update() merge.

        build_trainer() disables sanity validation by default (see the previous test). A caller that explicitly wants
        sanity validation back must be able to re-enable it via ``**trainer_kwargs``, the same override mechanism used
        for other PTL-native flags like ``fast_dev_run``.
        """
        trainer = build_trainer(base_train_config(), base_model_config(), accelerator="cpu", num_sanity_val_steps=2)

        assert trainer.num_sanity_val_steps == 2


# ---------------------------------------------------------------------------
# TestRichProgressBarLoggerIntegration — the real regression _RedirectAwareStreamHandler
# guards against, reproduced with a genuine Trainer.fit() run
# ---------------------------------------------------------------------------


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

    def test_logger_call_during_rich_fit_tracks_the_live_redirected_stream(
        self, base_model_config, base_train_config
    ) -> None:
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

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4)),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
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


# ---------------------------------------------------------------------------
# TestCoerceLegacyProgressBar — backward-compat validator on TrainConfig
# ---------------------------------------------------------------------------


class TestCoerceLegacyProgressBar:
    """_coerce_legacy_progress_bar must normalise legacy bool values."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            pytest.param(True, "tqdm", id="True->tqdm"),
            pytest.param(False, None, id="False->None"),
            pytest.param("rich", "rich", id="rich_passthrough"),
            pytest.param("tqdm", "tqdm", id="tqdm_passthrough"),
            pytest.param(None, None, id="None_passthrough"),
        ],
    )
    def test_coerce(self, base_train_config, value, expected):
        """progress_bar field normalises legacy bool and passes through string/None."""
        tc = base_train_config(progress_bar=value)
        assert tc.progress_bar == expected
