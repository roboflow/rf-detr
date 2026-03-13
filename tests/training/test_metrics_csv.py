# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Integration tests: metrics.csv contains all columns used by plot_metrics().

Runs a minimal PTL training loop (1 epoch, 2 batches each) using mocked model
internals so no real dataset or GPU is required.  After training, reads the
CSVLogger output and asserts that every metric column that ``plot_metrics()``
needs is present and has at least one non-NaN value.

Also verifies that ``train/loss`` is logged at the same scale as ``val/loss``
(i.e. NOT divided by ``grad_accum_steps`` before logging).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.datamodule import RFDETRDataModule
from rfdetr.training.module import RFDETRModule

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------


def _fit_and_read_csv(mc: RFDETRBaseConfig, tc: TrainConfig, criterion=None) -> pd.DataFrame:
    """Run 1 epoch (2 train + 2 val batches) and return the resulting metrics.csv."""
    fake_criterion = criterion or _FakeCriterion()
    with (
        patch("rfdetr.training.module.build_model", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module.build_criterion_and_postprocessors",
            return_value=(fake_criterion, MagicMock(side_effect=_fake_postprocess)),
        ),
        patch("rfdetr.training.datamodule.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.training.module.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModule(mc, tc)
        datamodule = RFDETRDataModule(mc, tc)
        trainer = build_trainer(
            tc,
            mc,
            accelerator="cpu",
            max_epochs=1,
            limit_train_batches=2,
            limit_val_batches=2,
            log_every_n_steps=1,
        )
        trainer.fit(module, datamodule=datamodule)

    csv_path = Path(tc.output_dir) / "metrics.csv"
    assert csv_path.exists(), "CSVLogger must write metrics.csv to output_dir"
    return pd.read_csv(csv_path)


# ---------------------------------------------------------------------------
# Expected columns (must exist and have ≥1 non-NaN row after one epoch)
# ---------------------------------------------------------------------------

_REQUIRED_DETECTION = frozenset(
    {
        "train/loss",
        "val/loss",
        "val/mAP_50",
        "val/mAP_50_95",
        "val/mAR",
    }
)

_REQUIRED_DETECTION_EMA = _REQUIRED_DETECTION | frozenset(
    {
        "val/ema_mAP_50",
        "val/ema_mAP_50_95",
        "val/ema_mAR",
    }
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDetectionMetricsCSV:
    """metrics.csv contains all columns that plot_metrics() needs for detection."""

    def test_base_metrics_present_without_ema(self, base_model_config, base_train_config):
        """Without EMA all core val/* columns must appear in metrics.csv with non-NaN data."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)
        df = _fit_and_read_csv(mc, tc)

        missing = _REQUIRED_DETECTION - set(df.columns)
        assert not missing, f"Missing columns in metrics.csv: {sorted(missing)}"

        all_nan = {c for c in _REQUIRED_DETECTION if df[c].isna().all()}
        assert not all_nan, f"Columns with all-NaN values: {sorted(all_nan)}"

    def test_ema_metrics_present_with_ema_enabled(self, base_model_config, base_train_config):
        """With use_ema=True the ema_* aliases must also appear in metrics.csv."""
        mc = base_model_config()
        tc = base_train_config(use_ema=True, run_test=False)
        df = _fit_and_read_csv(mc, tc)

        missing = _REQUIRED_DETECTION_EMA - set(df.columns)
        assert not missing, f"Missing EMA columns in metrics.csv: {sorted(missing)}"

        all_nan = {c for c in _REQUIRED_DETECTION_EMA if df[c].isna().all()}
        assert not all_nan, f"EMA columns with all-NaN values: {sorted(all_nan)}"

    def test_train_loss_is_unscaled(self, base_model_config, base_train_config):
        """train/loss must be logged at the raw criterion scale, not divided by grad_accum_steps.

        With grad_accum_steps=4 the old code divided the logged value by 4,
        making train/loss ~4× smaller than val/loss.  After the fix the logged
        value equals the raw weighted criterion output so both losses are on the
        same scale.
        """
        FIXED_LOSS = 5.0
        GRAD_ACCUM = 4

        class _FixedCriterion:
            weight_dict = {"loss_ce": 1.0}

            def __call__(self, outputs, targets):
                # Loss is always FIXED_LOSS, connected to model params for gradient.
                dummy = outputs.get("dummy", torch.zeros(1))
                return {"loss_ce": dummy.mean() * 0 + FIXED_LOSS}

        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, grad_accum_steps=GRAD_ACCUM)
        df = _fit_and_read_csv(mc, tc, criterion=_FixedCriterion())

        logged = df["train/loss"].dropna().mean()
        expected_unscaled = FIXED_LOSS
        expected_if_divided = FIXED_LOSS / GRAD_ACCUM

        assert abs(logged - expected_unscaled) < abs(logged - expected_if_divided), (
            f"train/loss={logged:.4f} is closer to the grad-accum-divided value "
            f"({expected_if_divided:.4f}) than the raw criterion output "
            f"({expected_unscaled:.4f}). The division must have been removed."
        )
