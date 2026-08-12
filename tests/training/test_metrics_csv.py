# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Integration tests: metrics.csv contains all columns used by plot_metrics().

Runs a minimal PTL training loop (1 epoch, 2 batches each) using mocked model internals so no real dataset or GPU is
required.  After training, reads the CSVLogger output and asserts that every metric column that ``plot_metrics()`` needs
is present and has at least one non-NaN value.

Also verifies that ``train/loss`` is logged at the same scale as ``val/loss`` (i.e. NOT divided by ``grad_accum_steps``
before logging).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import torch

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.training import build_trainer
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule

from .helpers import _fake_postprocess, _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

# ---------------------------------------------------------------------------
# Helpers local to this module
# ---------------------------------------------------------------------------


def _fit_and_read_csv(mc: RFDETRBaseConfig, tc: TrainConfig, criterion=None) -> pd.DataFrame:
    """Run 1 epoch (2 train + 2 val batches) and return the resulting metrics.csv."""
    fake_criterion = criterion or _FakeCriterion()
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(fake_criterion, MagicMock(side_effect=_fake_postprocess)),
        ),
        patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
        patch(
            "rfdetr.training.module_model.get_param_dict",
            side_effect=lambda args, model: _make_param_dicts(model),
        ),
    ):
        module = RFDETRModelModule(mc, tc)
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
        "train/lr",
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
        """Train/loss must be logged at the raw criterion scale, not divided by grad_accum_steps.

        With grad_accum_steps=4 the old code divided the logged value by 4, making train/loss ~4× smaller than val/loss.
        After the fix the logged value equals the raw weighted criterion output so both losses are on the same scale.
        """
        fixed_loss_value = 5.0
        grad_accum_steps = 4

        class _FixedCriterion:
            weight_dict = {"loss_ce": 1.0}

            def num_boxes_for_targets(self, outputs, targets):
                dummy = outputs.get("dummy", torch.zeros(1))
                return torch.ones((), dtype=dummy.dtype, device=dummy.device)

            def __call__(self, outputs, targets, num_boxes=None):
                # Loss is always fixed_loss_value, connected to model params for gradient.
                dummy = outputs.get("dummy", torch.zeros(1))
                denominator = self.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
                return {"loss_ce": (dummy.mean() * 0 + fixed_loss_value) / denominator}

        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False, grad_accum_steps=grad_accum_steps)
        df = _fit_and_read_csv(mc, tc, criterion=_FixedCriterion())

        logged = df["train/loss"].dropna().mean()
        expected_unscaled = fixed_loss_value
        expected_if_divided = fixed_loss_value / grad_accum_steps

        assert abs(logged - expected_unscaled) < abs(logged - expected_if_divided), (
            f"train/loss={logged:.4f} is closer to the grad-accum-divided value "
            f"({expected_if_divided:.4f}) than the raw criterion output "
            f"({expected_unscaled:.4f}). The division must have been removed."
        )


class TestMetricsCSVResume:
    """metrics.csv must retain history across a resumed run.

    Regression test for #1321: resuming training builds a brand-new ``Trainer``/``CSVLogger`` pointed at the same
    ``output_dir``. Every row written before the resume must still be present afterward, not just the rows from the
    resumed epoch(s).
    """

    def _fit_one_run(
        self,
        mc: RFDETRBaseConfig,
        tc: TrainConfig,
        fake_criterion: _FakeCriterion,
        max_epochs: int,
        ckpt_path: str | None,
    ) -> None:
        """Run one PTL ``fit()`` call against mocked model internals, writing into ``tc.output_dir``.

        Args:
            mc: Model config for the run.
            tc: Train config for the run; ``tc.output_dir`` is where ``metrics.csv`` and checkpoints land.
            fake_criterion: Shared criterion instance so loss values stay comparable across resumed runs.
            max_epochs: Epoch count passed to ``build_trainer``.
            ckpt_path: Checkpoint to resume PTL's trainer state from, or ``None`` for a fresh run.

        Examples:
            >>> import contextlib
            >>> import io
            >>> from tempfile import TemporaryDirectory
            >>> with TemporaryDirectory() as d:
            ...     mc = RFDETRBaseConfig(pretrain_weights=None, device='cpu', num_classes=3)
            ...     tc = TrainConfig(
            ...         dataset_dir=str(Path(d) / 'ds'),
            ...         output_dir=str(Path(d) / 'out'),
            ...         epochs=1,
            ...         batch_size=2,
            ...         tensorboard=False,
            ...     )
            ...     with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            ...         TestMetricsCSVResume()._fit_one_run(mc, tc, _FakeCriterion(), max_epochs=1, ckpt_path=None)
            ...     (Path(tc.output_dir) / 'metrics.csv').exists()
            True
        """
        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, MagicMock(side_effect=_fake_postprocess)),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=20)),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = build_trainer(
                tc,
                mc,
                accelerator="cpu",
                max_epochs=max_epochs,
                limit_train_batches=2,
                limit_val_batches=2,
                log_every_n_steps=1,
            )
            trainer.fit(module, datamodule=datamodule, ckpt_path=ckpt_path)

    def test_history_preserved_across_resume(self, base_model_config, base_train_config):
        """A second build_trainer() call against the same output_dir must APPEND, not overwrite, metrics.csv."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)
        fake_criterion = _FakeCriterion()

        self._fit_one_run(mc, tc, fake_criterion, max_epochs=1, ckpt_path=None)

        csv_path = Path(tc.output_dir) / "metrics.csv"
        assert csv_path.exists(), "First run must have written metrics.csv"
        rows_before_resume = pd.read_csv(csv_path)
        assert not rows_before_resume.empty, "First run must have logged at least one row"

        last_ckpt = Path(tc.output_dir) / "last.ckpt"
        assert last_ckpt.exists(), "checkpoint_interval default (10) must still save a `last` checkpoint every epoch"

        tc.resume = str(last_ckpt)
        self._fit_one_run(mc, tc, fake_criterion, max_epochs=2, ckpt_path=str(last_ckpt))

        rows_after_resume = pd.read_csv(csv_path)
        assert len(rows_after_resume) > len(rows_before_resume), (
            "metrics.csv must grow across a resume, not shrink or reset"
        )
        assert set(rows_before_resume.columns).issubset(rows_after_resume.columns), (
            "Every pre-resume metric column must remain available after the resume"
        )
        pd.testing.assert_frame_equal(
            rows_before_resume.reset_index(drop=True),
            rows_after_resume.loc[: len(rows_before_resume) - 1, rows_before_resume.columns].reset_index(drop=True),
            check_dtype=False,
        )

    def test_reused_output_dir_with_empty_resume_resets_history(self, base_model_config, base_train_config):
        """A fresh run with an empty resume value reusing output_dir must reset metrics.csv, like resume=None."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)
        fake_criterion = _FakeCriterion()

        self._fit_one_run(mc, tc, fake_criterion, max_epochs=1, ckpt_path=None)

        csv_path = Path(tc.output_dir) / "metrics.csv"
        rows_from_first_run = pd.read_csv(csv_path)
        assert not rows_from_first_run.empty, "First run must have logged at least one row"

        tc.resume = ""
        assert tc.resume == "", "This case must exercise the public empty-string resume value"
        self._fit_one_run(mc, tc, fake_criterion, max_epochs=1, ckpt_path=None)

        rows_from_second_run = pd.read_csv(csv_path)
        assert len(rows_from_second_run) == len(rows_from_first_run), (
            f"Fresh run wrote {len(rows_from_second_run)} rows but the prior run wrote "
            f"{len(rows_from_first_run)} — resume='' must let CSVLogger reset metrics.csv, "
            "not silently append onto an unrelated prior run's history."
        )

    def test_reused_output_dir_without_resume_resets_history(self, base_model_config, base_train_config):
        """A fresh run (``resume=None``) reusing a prior run's output_dir must reset metrics.csv, not append."""
        mc = base_model_config()
        tc = base_train_config(use_ema=False, run_test=False)
        fake_criterion = _FakeCriterion()

        self._fit_one_run(mc, tc, fake_criterion, max_epochs=1, ckpt_path=None)

        csv_path = Path(tc.output_dir) / "metrics.csv"
        rows_from_first_run = pd.read_csv(csv_path)
        assert not rows_from_first_run.empty, "First run must have logged at least one row"

        assert tc.resume is None, "This case only applies to a fresh run, not a resumed one"
        self._fit_one_run(mc, tc, fake_criterion, max_epochs=1, ckpt_path=None)

        rows_from_second_run = pd.read_csv(csv_path)
        assert len(rows_from_second_run) == len(rows_from_first_run), (
            f"Fresh run wrote {len(rows_from_second_run)} rows but the prior run wrote "
            f"{len(rows_from_first_run)} — a fresh run (resume=None) reusing output_dir must let "
            "CSVLogger reset metrics.csv, not silently append onto an unrelated prior run's history."
        )
