# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Unit tests for build_trainer() — callback stack and config coercion."""

import pytest
from pytorch_lightning.callbacks import RichProgressBar, TQDMProgressBar

from rfdetr.training import build_trainer

# ---------------------------------------------------------------------------
# TestProgressBarCallbacks — verifies the correct callback is installed
# ---------------------------------------------------------------------------


class TestProgressBarCallbacks:
    """build_trainer() must install the right progress bar callback for each mode."""

    def test_rich_progress_bar_installed_for_rich(self, base_model_config, base_train_config):
        """progress_bar='rich' must add RichProgressBar and not TQDMProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="rich")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert RichProgressBar in cb_types
        assert TQDMProgressBar not in cb_types

    def test_tqdm_progress_bar_installed_for_tqdm(self, base_model_config, base_train_config):
        """progress_bar='tqdm' must add TQDMProgressBar and not RichProgressBar."""
        mc = base_model_config()
        tc = base_train_config(progress_bar="tqdm")
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert TQDMProgressBar in cb_types
        assert RichProgressBar not in cb_types

    def test_no_progress_bar_callback_for_none(self, base_model_config, base_train_config):
        """progress_bar=None must not add any progress bar callback."""
        mc = base_model_config()
        tc = base_train_config(progress_bar=None)
        trainer = build_trainer(tc, mc, accelerator="cpu")
        cb_types = [type(cb) for cb in trainer.callbacks]
        assert RichProgressBar not in cb_types
        assert TQDMProgressBar not in cb_types


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
