# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for the EMA checkpoint tracking and weight reload of :class:`BestModelCallback`."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from pytorch_lightning import LightningModule

from rfdetr.training.callbacks.best_model import BestModelCallback
from rfdetr.training.callbacks.ema import RFDETREMACallback
from tests.training.callbacks.helpers import _make_best_model_pl_module as _make_pl_module
from tests.training.callbacks.helpers import _make_best_model_trainer as _make_trainer


class TestEvaluatesBaseModel:
    """Regular-track behaviour when validation does not evaluate the base model (``TrainConfig.eval_base_model``)."""

    def test_regular_checkpoint_skipped_when_base_model_unevaluated(self, tmp_path: Path) -> None:
        """No base-weights checkpoint may be saved while the primary key carries the EMA score.

        Under the default eval policy COCOEvalCallback mirrors the EMA score onto ``val/mAP_50_95``, so the key is
        present and the parent ModelCheckpoint would happily save the base weights against it — checkpointing one model
        on the other model's score. Only the EMA track is entitled to write this epoch.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert not (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_ema_checkpoint_written_when_base_model_unevaluated(self, tmp_path: Path) -> None:
        """Suppressing the regular track must not also suppress the EMA track.

        The EMA track is the only checkpoint tracking that runs under the default policy (#1285); disabling both would
        leave a run with no best checkpoint at all.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", evaluates_base_model=False)
        trainer = _make_trainer({"val/mAP_50_95": 0.9, "val/ema_mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_ema.pth").exists()

    def test_regular_checkpoint_written_when_base_model_evaluated(self, tmp_path: Path) -> None:
        """eval_base_model=True (and every use_ema=False run) keeps the regular track working as before.

        The suppression must be scoped to the configuration that actually stops evaluating the base model, not applied
        whenever EMA happens to be enabled.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=True)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert (tmp_path / "checkpoint_best_regular.pth").exists()

    def test_smoothing_ignores_mirrored_ema_score(self, tmp_path: Path) -> None:
        """The regular smoothing accumulator must not absorb EMA values it will never checkpoint against.

        ``_smoothed_regular`` is persisted in the callback state and compared against the raw EMA score in
        ``on_fit_end``; letting the mirrored EMA score feed it would corrupt that comparison on a later resume.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), evaluates_base_model=False, smooth_alpha=0.5)
        trainer = _make_trainer({"val/mAP_50_95": 0.9})

        cb.on_validation_end(trainer, _make_pl_module())

        assert cb._smoothed_regular == 0.0


# ---------------------------------------------------------------------------
# _best_ema state persistence across resume (#969)
# ---------------------------------------------------------------------------


class TestBestEmaStatePersistence:
    """Regression tests for _best_ema not surviving Lightning checkpoint resume.

    Before the fix, BestModelCallback did not override state_dict() / load_state_dict(), so _best_ema was never included
    in the Lightning callback state bundle.  On resume via trainer.fit(ckpt_path=...) the callback was reconstructed
    fresh with _best_ema=0.0, causing any positive post-resume EMA value to trivially overwrite checkpoint_best_ema.pth
    with inferior weights.

    Regression tests for GitHub issue #969.
    """

    def test_state_dict_includes_best_ema(self, tmp_path: Path) -> None:
        """state_dict() must include _best_ema so it survives Lightning checkpointing."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()

        # Drive _best_ema to 0.75 via a validation pass.
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75})
        cb.on_validation_end(trainer, pl_module)

        state = cb.state_dict()

        assert "_best_ema" in state, "_best_ema must be present in state_dict() output"
        assert state["_best_ema"] == pytest.approx(0.75)

    def test_load_state_dict_restores_best_ema(self, tmp_path: Path) -> None:
        """load_state_dict() must restore _best_ema from the persisted state."""
        # First callback: train to _best_ema=0.75.
        cb_first = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75})
        cb_first.on_validation_end(trainer, pl_module)
        saved_state = cb_first.state_dict()

        # Second callback: simulate a fresh resume by loading the saved state.
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb_resumed.load_state_dict(saved_state)

        assert cb_resumed._best_ema == pytest.approx(0.75), (
            "load_state_dict() must restore _best_ema; without fix it stays 0.0"
        )

    def test_resume_does_not_clobber_ema_checkpoint_with_inferior_weights(self, tmp_path: Path) -> None:
        """After resume, inferior post-resume EMA must not overwrite checkpoint_best_ema.pth.

        Without the fix: _best_ema resets to 0.0 on resume, so any positive EMA metric (0.5) trivially satisfies ema_val
        > _best_ema and overwrites the checkpoint saved pre-resume (0.75).
        """
        # --- Pre-resume phase: establish EMA best of 0.75 ---
        cb_pre = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        pl_module = _make_pl_module()
        trainer_pre = _make_trainer(
            {"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.75},
            current_epoch=5,
        )
        cb_pre.on_validation_end(trainer_pre, pl_module)

        ema_path = tmp_path / "checkpoint_best_ema.pth"
        assert ema_path.exists()
        baseline_epoch = torch.load(ema_path, map_location="cpu", weights_only=False)["epoch"]
        saved_state = cb_pre.state_dict()

        # --- Resume phase: fresh callback loaded from saved state ---
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb_resumed.load_state_dict(saved_state)

        # Post-resume validation reports EMA=0.5 — worse than pre-resume best (0.75).
        trainer_post = _make_trainer(
            {"val/mAP_50_95": 0.45, "val/ema_mAP_50_95": 0.5},
            current_epoch=7,
        )
        cb_resumed.on_validation_end(trainer_post, pl_module)

        assert torch.load(ema_path, map_location="cpu", weights_only=False)["epoch"] == baseline_epoch, (
            "checkpoint_best_ema.pth must not be overwritten by an inferior post-resume EMA value"
        )

    def test_on_fit_end_selects_ema_winner_after_resume(self, tmp_path: Path) -> None:
        """on_fit_end picks EMA winner correctly when _best_ema is properly restored.

        Without the fix: _best_ema=0.0 after resume, so regular (0.6) wins over the true EMA best (0.8) —
        checkpoint_best_total.pth is built from the wrong source.  The payload's ``best_total_source`` field records
        which source won, so EMA selection is asserted directly instead of via a saved-epoch proxy.
        """
        # Pre-resume epoch 1: regular best=0.6.
        cb_pre = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        pl_module = _make_pl_module()
        trainer_ep1 = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.5}, current_epoch=1)
        cb_pre.on_validation_end(trainer_ep1, pl_module)

        # Pre-resume epoch 3: EMA best=0.8 (better than epoch-1 EMA=0.5).
        trainer_ep3 = _make_trainer({"val/mAP_50_95": 0.55, "val/ema_mAP_50_95": 0.8}, current_epoch=3)
        cb_pre.on_validation_end(trainer_ep3, pl_module)

        saved_state = cb_pre.state_dict()

        # Resume: fresh callback, load state, run fit_end.
        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=False)
        cb_resumed.load_state_dict(saved_state)

        # fit_end uses _best_ema to decide EMA vs regular winner.
        # trainer_ep3 carries best_model_score=0.6 (regular) and _best_ema=0.8 (EMA wins).
        cb_resumed.on_fit_end(trainer_ep3, pl_module)

        total = tmp_path / "checkpoint_best_total.pth"
        assert total.exists()
        total_data = torch.load(total, map_location="cpu", weights_only=False)
        # best_total_source records the winning source directly.
        # If _best_ema was NOT restored (bug), regular wins → "regular".
        # If _best_ema IS restored (fix), EMA wins → "ema".
        source = total_data["best_total_source"]
        assert source == "ema", (
            "on_fit_end must select EMA (best=0.8) over regular (best=0.6); "
            f"got best_total_source={source!r} — _best_ema not restored from state_dict"
        )

    @pytest.mark.parametrize(
        ("mutate_state", "expected_best_ema"),
        [
            pytest.param(lambda state: state.pop("_best_ema"), 0.0, id="missing-key"),
            pytest.param(lambda state: state.__setitem__("_best_ema", 1), 1.0, id="int-coercion"),
            pytest.param(lambda state: state.__setitem__("_best_ema", "0.75"), 0.75, id="string-coercion"),
        ],
    )
    def test_load_state_dict_backward_compat(self, tmp_path: Path, mutate_state, expected_best_ema: float) -> None:
        """load_state_dict() keeps backward-compatible _best_ema restoration behavior."""
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        state = cb.state_dict()
        mutate_state(state)

        cb.load_state_dict(state)

        assert isinstance(cb._best_ema, float)
        assert cb._best_ema == expected_best_ema

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_load_state_dict_non_finite_values(self, bad_value) -> None:
        """load_state_dict() resets non-finite persisted _best_ema values to 0.0."""
        cb = BestModelCallback(output_dir=".")
        cb._best_ema = 999.0
        state = cb.state_dict()
        state["_best_ema"] = bad_value

        cb.load_state_dict(state)

        assert cb._best_ema == 0.0

    def test_state_dict_round_trip_preserves_all_three_smooth_fields(self, tmp_path: Path) -> None:
        """state_dict/load_state_dict round-trip preserves _best_ema, _smoothed_regular, and _best_raw_regular.

        Scenario: user trains with smooth_alpha=0.5, two validation epochs advance all three accumulators,
        then a fresh callback is restored from the persisted state — all three fields must survive unchanged.
        """
        cb = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", smooth_alpha=0.5)
        pl_module = _make_pl_module()

        # Epoch 0: raw=0.6 → smoothed=0.3, EMA=0.5 (saves EMA checkpoint)
        trainer0 = _make_trainer({"val/mAP_50_95": 0.6, "val/ema_mAP_50_95": 0.5}, current_epoch=0)
        trainer0.global_step = 1
        cb.on_validation_end(trainer0, pl_module)

        # Epoch 1: raw=0.4 → smoothed=0.35, EMA=0.7 > 0.5 (saves new EMA checkpoint); raw at smoothed-best=0.4
        trainer1 = _make_trainer({"val/mAP_50_95": 0.4, "val/ema_mAP_50_95": 0.7}, current_epoch=1)
        trainer1.global_step = 2
        cb.on_validation_end(trainer1, pl_module)

        state = cb.state_dict()

        cb_resumed = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", smooth_alpha=0.5)
        cb_resumed.load_state_dict(state)

        assert cb_resumed._best_ema == pytest.approx(cb._best_ema), "_best_ema must survive round-trip"
        assert cb_resumed._smoothed_regular == pytest.approx(cb._smoothed_regular), (
            "_smoothed_regular must survive round-trip"
        )
        assert cb_resumed._best_raw_regular == pytest.approx(cb._best_raw_regular), (
            "_best_raw_regular must survive round-trip"
        )

    def test_load_state_dict_does_not_mutate_caller_dict(self, tmp_path: Path) -> None:
        """load_state_dict must not pop or mutate the caller-supplied dict."""
        cb1 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb1._best_ema = 0.75
        original_sd = cb1.state_dict()
        saved = dict(original_sd)

        cb2 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb2.load_state_dict(original_sd)

        assert original_sd["_best_ema"] == 0.75
        assert original_sd == saved

    def test_state_dict_roundtrip_initial_zero(self, tmp_path: Path) -> None:
        """state_dict/load_state_dict round-trips the default _best_ema=0.0 correctly."""
        cb1 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        sd = cb1.state_dict()

        assert "_best_ema" in sd
        assert sd["_best_ema"] == 0.0

        cb2 = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95")
        cb2.load_state_dict(sd)

        assert cb2._best_ema == 0.0


class TestBestModelSmoothAlpha:
    """Verify ``smooth_alpha`` smooths the monitored metric before checkpoint comparison.

    Without smoothing, a single noisy spike (e.g. 0.9 surrounded by 0.3-0.4 values) locks the best checkpoint to that
    spike epoch.  With ``smooth_alpha=0.5`` the EMA of the metric is what the parent ModelCheckpoint compares, so the
    later, steadily-improving epochs can overtake the early spike.
    """

    @pytest.mark.parametrize(
        ("invalid_value", "exc_type"),
        [
            pytest.param(True, TypeError, id="bool-true"),
            pytest.param("0.5", TypeError, id="string"),
            pytest.param(-0.1, ValueError, id="negative"),
            pytest.param(1.0, ValueError, id="exactly-one"),
            pytest.param(1.5, ValueError, id="greater-than-one"),
            pytest.param(float("nan"), ValueError, id="nan"),
            pytest.param(float("inf"), ValueError, id="inf"),
        ],
    )
    def test_invalid_smooth_alpha_raises(
        self, tmp_path: Path, invalid_value: object, exc_type: type[Exception]
    ) -> None:
        """BestModelCallback raises TypeError for non-numeric and ValueError for out-of-range smooth_alpha."""
        with pytest.raises(exc_type):
            BestModelCallback(output_dir=str(tmp_path), smooth_alpha=invalid_value)  # type: ignore[arg-type]

    def test_smoothing_disabled_when_alpha_zero(self, tmp_path: Path) -> None:
        """With smooth_alpha=0.0 (default), the raw metric drives checkpoint selection unchanged."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.0)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.42}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        # best_model_score reflects the raw value because no smoothing was applied.
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.42)
        # The smoothing accumulator must stay at its zero default when smoothing is disabled.
        assert cb._smoothed_regular == 0.0

    def test_smoothing_uses_ema_of_raw_metric(self, tmp_path: Path) -> None:
        """With smooth_alpha=0.5, best_model_score reflects the smoothed EMA, not the raw value."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()

        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        # EMA: 0.5 * 0.0 + 0.5 * 0.8 = 0.4 — the smoothed value, not the raw 0.8.
        assert cb._smoothed_regular == pytest.approx(0.4)
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.4)

    def test_smoothing_prefers_steady_improvement_over_early_spike(self, tmp_path: Path) -> None:
        """A late, smoothed run that beats the smoothed early spike wins the best checkpoint."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()

        # Epoch 0: raw=0.8 → smoothed=0.4
        # Epoch 1: raw=0.3 → smoothed=0.35
        # Epoch 2: raw=0.4 → smoothed=0.375
        # Epoch 3: raw=0.6 → smoothed=0.4875  (overtakes the early spike's smoothed 0.4)
        raw_per_epoch = [0.8, 0.3, 0.4, 0.6]
        for epoch, value in enumerate(raw_per_epoch):
            trainer = _make_trainer({"val/mAP_50_95": value}, current_epoch=epoch)
            # Distinct global_step per epoch so ModelCheckpoint does not skip saves on
            # the second-and-later calls due to its same-step guard.
            trainer.global_step = epoch + 1
            cb.on_validation_end(trainer, pl_module)

        # The best smoothed value should be epoch-3's 0.4875, not epoch-0's 0.4.
        assert cb._smoothed_regular == pytest.approx(0.4875)
        assert cb.best_model_score is not None
        assert cb.best_model_score.item() == pytest.approx(0.4875)

    def test_callback_metrics_restored_after_super_call(self, tmp_path: Path) -> None:
        """trainer.callback_metrics[monitor] must hold the original raw tensor after on_validation_end returns."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        raw_tensor = trainer.callback_metrics["val/mAP_50_95"]

        cb.on_validation_end(trainer, pl_module)

        # The original tensor (raw value 0.8) must be back in callback_metrics so other
        # callbacks and metrics.csv see the unsmoothed value.
        assert trainer.callback_metrics["val/mAP_50_95"] is raw_tensor
        assert trainer.callback_metrics["val/mAP_50_95"].item() == pytest.approx(0.8)

    def test_state_dict_includes_smoothed_regular(self, tmp_path: Path) -> None:
        """state_dict() must include _smoothed_regular so resumed training continues from the correct EMA value."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb.on_validation_end(trainer, pl_module)

        state = cb.state_dict()

        assert "_smoothed_regular" in state
        assert state["_smoothed_regular"] == pytest.approx(0.4)

    def test_load_state_dict_restores_smoothed_regular(self, tmp_path: Path) -> None:
        """load_state_dict() must restore _smoothed_regular so the EMA continues from the persisted value."""
        cb_first = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        cb_first.on_validation_end(trainer, pl_module)
        saved_state = cb_first.state_dict()

        cb_resumed = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        cb_resumed.load_state_dict(saved_state)

        assert cb_resumed._smoothed_regular == pytest.approx(0.4)

    @pytest.mark.parametrize(
        "bad_value",
        [float("nan"), float("inf"), float("-inf")],
    )
    def test_load_state_dict_non_finite_smoothed_regular_resets_to_zero(self, tmp_path: Path, bad_value: float) -> None:
        """load_state_dict() resets non-finite persisted _smoothed_regular values to 0.0."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        state = cb.state_dict()
        state["_smoothed_regular"] = bad_value

        cb.load_state_dict(state)

        assert cb._smoothed_regular == 0.0

    def test_load_state_dict_smoothed_regular_missing_key_defaults_to_zero(self, tmp_path: Path) -> None:
        """load_state_dict() defaults _smoothed_regular to 0.0 when key is absent (backward compat)."""
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        cb._smoothed_regular = 0.42
        state = cb.state_dict()
        state.pop("_smoothed_regular")

        cb.load_state_dict(state)

        assert cb._smoothed_regular == 0.0

    def test_ema_accumulator_warms_up_during_skip_window(self, tmp_path: Path) -> None:
        """_smoothed_regular is pre-warmed by skip-window epochs so epoch 2 starts from a non-zero EMA.

        Scenario: skip_best_epochs=2, smooth_alpha=0.5, metric=0.5 each epoch.
        The EMA accumulator must update on epochs 0 and 1 (skip window) so that by epoch 2 it
        is already non-zero.  No checkpoint file should be written during the skip window.
        """
        skip_epochs = 2
        cb = BestModelCallback(
            output_dir=str(tmp_path), monitor_regular="val/mAP_50_95", smooth_alpha=0.5, skip_best_epochs=skip_epochs
        )
        pl_module = _make_pl_module()

        # Epochs 0 and 1 are inside the skip window.
        for epoch in range(skip_epochs):
            trainer_skip = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=epoch)
            trainer_skip.global_step = epoch + 1
            cb.on_validation_end(trainer_skip, pl_module)

        # No checkpoint must have been written during the skip window.
        assert not (tmp_path / "checkpoint_best_regular.pth").exists(), (
            "checkpoint must not be written during the skip window (epochs 0 and 1)"
        )

        # Epoch 2 is the first eligible epoch — accumulator must be pre-warmed.
        trainer_eligible = _make_trainer({"val/mAP_50_95": 0.5}, current_epoch=skip_epochs)
        trainer_eligible.global_step = skip_epochs + 1
        cb.on_validation_end(trainer_eligible, pl_module)

        assert cb._smoothed_regular > 0.0, (
            "_smoothed_regular must be > 0.0 at epoch 2 because the skip-window epochs warmed the EMA"
        )

    def test_callback_metrics_restored_when_super_raises(self, mocker: MockerFixture, tmp_path: Path) -> None:
        """trainer.callback_metrics[monitor] is restored in the finally block even when super() raises."""
        mocker.patch(
            "rfdetr.training.callbacks.best_model.ModelCheckpoint.on_validation_end",
            side_effect=RuntimeError("simulated failure"),
        )
        cb = BestModelCallback(output_dir=str(tmp_path), smooth_alpha=0.5)
        pl_module = _make_pl_module()
        trainer = _make_trainer({"val/mAP_50_95": 0.8}, current_epoch=0)
        raw_tensor = trainer.callback_metrics["val/mAP_50_95"]

        with pytest.raises(RuntimeError) as exc_info:
            cb.on_validation_end(trainer, pl_module)
        assert "simulated failure" in str(exc_info.value)

        assert trainer.callback_metrics["val/mAP_50_95"] is raw_tensor


class _TestStepLinearModule(LightningModule):
    """Real module with ``test_step`` and a tiny linear model for weight-identity assertions."""

    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.train_config = {"lr": 0.001}

    def test_step(self, batch: object, batch_idx: int) -> None:
        """No-op test step so BestModelCallback.on_fit_end runs trainer.test()."""


def _fill_module_weight(pl_module: LightningModule, value: float) -> None:
    """Set the single linear weight of *pl_module* to *value* in-place."""
    with torch.no_grad():
        pl_module.model.weight.fill_(value)


class TestOnFitEndEMASwapSuppression:
    """The fit-end test run must evaluate the best checkpoint, not the final EMA weights.

    Regression tests: ``BestModelCallback.on_fit_end`` loads ``checkpoint_best_total.pth`` into the module and then
    calls ``trainer.test()`` — but ``RFDETREMACallback.on_test_epoch_start`` used to swap in the final EMA weights,
    silently overwriting the just-loaded best weights for the whole test run.
    """

    @staticmethod
    def _build_fit_end_scenario(
        tmp_path: Path,
    ) -> tuple[BestModelCallback, RFDETREMACallback, LightningModule, MagicMock, list[torch.Tensor]]:
        """Arrange best=3.0 checkpoint, EMA=5.0 average model, live=7.0 weights, and a hook-simulating trainer."""
        from torch.optim.swa_utils import AveragedModel

        pl_module = _TestStepLinearModule()
        cb = BestModelCallback(output_dir=str(tmp_path), run_test=True)
        ema_cb = RFDETREMACallback()

        # Best checkpoint saved at weight 3.0.
        _fill_module_weight(pl_module, 3.0)
        trainer = _make_trainer({"val/mAP_50_95": 0.5})
        cb.on_validation_end(trainer, pl_module)

        # EMA average model captured at weight 5.0.
        _fill_module_weight(pl_module, 5.0)
        ema_cb._average_model = AveragedModel(
            model=pl_module,
            use_buffers=True,
            avg_fn=ema_cb._avg_fn,
        )
        ema_cb._average_model.eval()

        # Final live weights end training at 7.0.
        _fill_module_weight(pl_module, 7.0)

        trainer.callbacks = [ema_cb, cb]
        observed_weights: list[torch.Tensor] = []

        def _fake_test(module: object, datamodule: object = None, verbose: bool = False) -> None:
            # Simulate the PTL test loop invoking the EMA callback's test hooks.
            ema_cb.on_test_epoch_start(trainer, pl_module)
            observed_weights.append(pl_module.model.weight.detach().clone())
            ema_cb.on_test_epoch_end(trainer, pl_module)

        trainer.test = MagicMock(side_effect=_fake_test)
        return cb, ema_cb, pl_module, trainer, observed_weights

    def test_fit_end_test_runs_on_best_weights_not_ema(self, tmp_path: Path) -> None:
        """During the fit-end test run the module must hold the best checkpoint weights (3.0), not EMA (5.0)."""
        cb, _ema_cb, pl_module, trainer, observed_weights = self._build_fit_end_scenario(tmp_path)

        cb.on_fit_end(trainer, pl_module)

        assert observed_weights, "trainer.test() must have been invoked"
        expected = torch.full_like(observed_weights[0], 3.0)
        assert torch.allclose(observed_weights[0], expected), (
            f"test ran with weight {observed_weights[0].item():.1f}; expected best-checkpoint weight 3.0"
        )

    def test_ema_swap_suppression_cleared_after_fit_end(self, tmp_path: Path) -> None:
        """After on_fit_end the EMA callback must swap again for standalone trainer.test() runs."""
        cb, ema_cb, pl_module, trainer, _observed = self._build_fit_end_scenario(tmp_path)

        cb.on_fit_end(trainer, pl_module)

        assert ema_cb.suppress_test_swap is False

    def test_ema_swap_suppression_cleared_when_test_raises(self, tmp_path: Path) -> None:
        """Suppression must be lifted even when trainer.test() raises."""
        cb, ema_cb, pl_module, trainer, _observed = self._build_fit_end_scenario(tmp_path)
        trainer.test = MagicMock(side_effect=RuntimeError("boom"))

        with pytest.raises(RuntimeError) as exc_info:
            cb.on_fit_end(trainer, pl_module)
        assert "boom" in str(exc_info.value)

        assert ema_cb.suppress_test_swap is False


class _GuardedExtraStateLinear(torch.nn.Linear):
    """Represent a converted layer whose extra-state loader forbids deserialization."""

    def get_extra_state(self) -> torch.Tensor:
        """Return a serializable stand-in for FP8 scaling history.

        Examples:
            >>> _GuardedExtraStateLinear(1, 1).get_extra_state()
            tensor([1], dtype=torch.uint8)
        """
        return torch.ones(1, dtype=torch.uint8)

    def set_extra_state(self, state: object) -> None:
        """Reject loading history, as Transformer Engine does without pickle opt-in.

        Examples:
            >>> _GuardedExtraStateLinear(1, 1).set_extra_state(None)
            Traceback (most recent call last):
            ...
            RuntimeError: extra-state deserialization forbidden
        """
        raise RuntimeError("extra-state deserialization forbidden")


class TestFP8BestEMAWeightReload:
    """Exercise real checkpoint export and reload with converted-layer extra state."""

    @pytest.mark.parametrize(
        ("damage", "errors"),
        [
            pytest.param(None, None, id="intact-ema"),
            pytest.param("missing", ("Missing", "weight"), id="missing-weight"),
            pytest.param("unexpected", ("Unexpected", "spurious"), id="unexpected-key"),
            pytest.param("shape", ("size mismatch",), id="wrong-shape"),
        ],
    )
    def test_final_ema_reload(self, tmp_path: Path, damage: str | None, errors: tuple[str, ...] | None) -> None:
        """Only absent extra state is tolerated; all tensor/key corruption remains fatal."""
        module = _TestStepLinearModule()
        module.model = torch.nn.Sequential(_GuardedExtraStateLinear(1, 1, bias=False))
        with torch.no_grad():
            module.model[0].weight.fill_(3.0)
        ema = RFDETREMACallback()
        callback = BestModelCallback(output_dir=str(tmp_path), monitor_ema="val/ema_mAP_50_95", run_test=True)
        trainer = _make_trainer({"val/ema_mAP_50_95": 0.8}, callbacks=[ema, callback])
        ema.on_fit_start(trainer, module)
        callback.on_validation_end(trainer, module)
        path = tmp_path / "checkpoint_best_ema.pth"
        checkpoint = torch.load(path, weights_only=True)
        assert set(checkpoint["model"]) == {"0.weight"}
        if damage == "missing":
            del checkpoint["model"]["0.weight"]
        elif damage == "unexpected":
            checkpoint["model"]["spurious"] = torch.ones(1)
        elif damage == "shape":
            checkpoint["model"]["0.weight"] = torch.ones(2, 1)
        torch.save(checkpoint, path)
        with torch.no_grad():
            module.model[0].weight.fill_(7.0)

        if errors is not None:
            with pytest.raises(RuntimeError) as exc_info:
                callback.on_fit_end(trainer, module)
            assert all(part in str(exc_info.value) for part in errors)
            trainer.test.assert_not_called()
        else:
            callback.on_fit_end(trainer, module)
            trainer.test.assert_called_once_with(module, datamodule=trainer.datamodule, verbose=False)
            torch.testing.assert_close(module.model[0].weight, torch.full((1, 1), 3.0), rtol=1e-4, atol=1e-6)
