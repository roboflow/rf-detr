# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for build_trainer() — callback stack and config coercion."""

import pytest
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

from rfdetr.training import build_trainer
from rfdetr.training.callbacks import GpuMemoryRichProgressBar, GpuMemoryTQDMProgressBar

# ---------------------------------------------------------------------------
# TestProgressBarCallbacks — verifies the correct callback is installed
# ---------------------------------------------------------------------------


class TestProgressBarCallbacks:
    """build_trainer() must install the right progress bar callback for each mode.

    The installed callbacks are the ``GpuMemory*`` subclasses (see ``gpu_memory_progress_bar.py``), so membership is
    checked with ``isinstance`` against the base PTL classes rather than exact ``type(cb) in [...]`` matches.
    """

    def test_rich_progress_bar_installed_for_rich(self, base_model_config, base_train_config):
        """progress_bar='rich' must add a RichProgressBar and no TQDMProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, GpuMemoryRichProgressBar) for cb in trainer.callbacks)

    def test_tqdm_progress_bar_installed_for_tqdm(self, base_model_config, base_train_config):
        """progress_bar='tqdm' must add a TQDMProgressBar and no RichProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert any(isinstance(cb, GpuMemoryTQDMProgressBar) for cb in trainer.callbacks)

    def test_progress_bar_refresh_rate_is_five(self, base_model_config, base_train_config):
        """The installed progress bar callback should refresh every five batches."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        progress_bar = next(cb for cb in trainer.callbacks if isinstance(cb, TQDMProgressBar))

        assert progress_bar.refresh_rate == 5

    def test_no_progress_bar_callback_for_none(self, base_model_config, base_train_config):
        """progress_bar=None must not add any progress bar callback."""
        mc = base_model_config()
        tc = base_train_config(progress_bar=None)
        trainer = build_trainer(tc, mc, accelerator="cpu")
        assert not any(isinstance(cb, RichProgressBar) for cb in trainer.callbacks)
        assert not any(isinstance(cb, TQDMProgressBar) for cb in trainer.callbacks)


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
