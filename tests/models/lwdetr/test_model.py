# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for the public ``RFDETR.evaluate()`` API (issue #1110).

The expensive end-to-end evaluation (build a real RFDETRNano, run ``trainer.test`` on the synthetic dataset) is run once
via a module-scoped fixture; the individual assertions each read from that single result so they stay fast and follow
the "one validation case per test" convention.

The split-dispatch and class-count-mismatch tests mock the PTL stack so they run without a forward pass.

Regression tests proving LWDETR forward contract survives joiner return-shape changes.

Tests that ``RFDETR`` normalizes each input channel with the statistics training uses.

Tests for which segmentation masks an eval-mode ``LWDETR`` forward computes.

Only ``SetCriterion`` reads the auxiliary-layer and encoder-proposal masks, so eval mode computes the last decoder
layer's masks alone unless ``all_layer_masks=True`` asks for the rest.
"""

import builtins
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture
from torch import Tensor

from rfdetr import RFDETR, RFDETRLarge, RFDETRNano
from rfdetr.config import TrainConfig
from rfdetr.datasets._aug_utils import cycle_channel_stats
from rfdetr.datasets.kornia_transforms import IMAGENET_MEAN, IMAGENET_STD
from rfdetr.detr import _device_move_lock
from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor


def _num_classes(dataset_dir: Path) -> int:
    """Return the COCO category count from a Roboflow-style dataset's train split.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as tmp:
        ...     root = Path(tmp)
        ...     train = root / "train"
        ...     train.mkdir()
        ...     _ = (train / "_annotations.coco.json").write_text('{"categories": [{}, {}]}')
        ...     _num_classes(root)
        2
    """
    annotations = json.loads((dataset_dir / "train" / "_annotations.coco.json").read_text())
    return len(annotations["categories"])


@pytest.fixture(scope="module")
def nano_model(synthetic_shape_dataset_dir: Path) -> RFDETRNano:
    """A CPU RFDETRNano sized to the synthetic dataset, built once for the module."""
    return RFDETRNano(
        pretrain_weights=None,
        num_classes=_num_classes(synthetic_shape_dataset_dir),
        device="cpu",
    )


@pytest.fixture(scope="module")
def evaluation_result(
    nano_model: RFDETRNano,
    synthetic_shape_dataset_dir: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> SimpleNamespace:
    """Run ``evaluate()`` on the test split once and capture inputs/outputs for assertions."""
    output_dir = tmp_path_factory.mktemp("evaluate_output")
    state_before = {key: value.detach().clone() for key, value in nano_model.model.model.state_dict().items()}
    num_classes_before = nano_model.model_config.num_classes

    metrics = nano_model.evaluate(
        dataset_dir=str(synthetic_shape_dataset_dir),
        split="test",
        device="cpu",
        output_dir=str(output_dir),
        batch_size=4,
        num_workers=0,
        tensorboard=False,
        log_per_class_metrics=True,
    )

    return SimpleNamespace(
        metrics=metrics,
        output_dir=output_dir,
        state_before=state_before,
        state_after=nano_model.model.model.state_dict(),
        num_classes_before=num_classes_before,
        model=nano_model,
    )


class TestEvaluateReturnValue:
    """``evaluate()`` returns the COCO metric dictionary produced by the eval callback."""

    def test_returns_dict(self, evaluation_result: SimpleNamespace) -> None:
        """The return value is a dictionary."""
        assert isinstance(evaluation_result.metrics, dict)

    @pytest.mark.parametrize("key", ["test/mAP_50_95", "test/F1"])
    def test_contains_metric(self, evaluation_result: SimpleNamespace, key: str) -> None:
        """The primary COCO mAP key and the F1-sweep key are present for the requested split."""
        assert key in evaluation_result.metrics

    def test_metric_values_are_finite_floats(self, evaluation_result: SimpleNamespace) -> None:
        """Every returned metric is a finite scalar."""
        values = [float(v) for v in evaluation_result.metrics.values()]
        assert all(torch.isfinite(torch.tensor(values)))


class TestEvaluateDoesNotMutateModel:
    """Evaluation must never re-initialize the detection head or change weights."""

    def test_num_classes_unchanged(self, evaluation_result: SimpleNamespace) -> None:
        """``model_config.num_classes`` is identical before and after evaluation."""
        assert evaluation_result.model.model_config.num_classes == evaluation_result.num_classes_before

    def test_weights_unchanged(self, evaluation_result: SimpleNamespace) -> None:
        """The underlying module weights are byte-for-byte identical after evaluation."""
        before, after = evaluation_result.state_before, evaluation_result.state_after
        assert before.keys() == after.keys()
        assert all(torch.equal(before[key], after[key]) for key in before)


class TestEvaluateClassNames:
    """Per-class metrics must be labelled by class name on a test-only run (regression).

    ``COCOEvalCallback`` previously resolved class names only in the fit-only ``on_fit_start`` hook, so a standalone
    ``trainer.test()`` produced per-class keys labelled by numeric id. ``evaluate()`` runs test-only, so the callback
    must resolve names in an test/validate hook too.
    """

    def test_per_class_keys_use_names(self, evaluation_result: SimpleNamespace) -> None:
        """At least one per-class metric key is labelled by a non-numeric class name."""
        per_class_suffixes = [k.split("/")[-1] for k in evaluation_result.metrics if k.startswith("test/AP/")]
        assert per_class_suffixes, "expected per-class AP metrics to be logged"
        assert any(not suffix.isdigit() for suffix in per_class_suffixes)


class TestEvaluateNoSideEffects:
    """Evaluation must not write checkpoints or training logs to the output directory."""

    def test_no_checkpoint_files(self, evaluation_result: SimpleNamespace) -> None:
        """No ``*.ckpt`` files are produced during evaluation."""
        assert not list(Path(evaluation_result.output_dir).rglob("*.ckpt"))

    def test_no_metrics_csv(self, evaluation_result: SimpleNamespace) -> None:
        """No CSVLogger ``metrics.csv`` is produced during evaluation."""
        assert not list(Path(evaluation_result.output_dir).rglob("metrics.csv"))


def _mock_trainer() -> Any:
    """Return a MagicMock standing in for the PTL ``Trainer`` returned by ``build_trainer``.

    ``test``/``validate`` return a one-element metrics list so ``evaluate()`` can index ``results[0]``.

    Examples:
        >>> trainer = _mock_trainer()
        >>> hasattr(trainer, "test") and hasattr(trainer, "validate")
        True
    """
    trainer = MagicMock()
    trainer.test.return_value = [{"test/mAP_50_95": 0.0}]
    trainer.validate.return_value = [{"val/mAP_50_95": 0.0}]
    return trainer


@pytest.fixture
def mocked_training(mocker: MockerFixture) -> SimpleNamespace:
    """Patch the PTL training stack that ``evaluate()`` imports lazily, so no forward pass runs.

    Exposes the mocked ``trainer`` plus the patched ``module``, ``datamodule`` and ``build_trainer`` objects.
    """
    trainer = _mock_trainer()
    return SimpleNamespace(
        trainer=trainer,
        module=mocker.patch("rfdetr.training.RFDETRModelModule"),
        datamodule=mocker.patch("rfdetr.training.RFDETRDataModule"),
        build_trainer=mocker.patch("rfdetr.training.build_trainer", return_value=trainer),
    )


class _LockTrackingLiveModel(torch.nn.Module):
    """Live-module stand-in that records the device-move lock state at every ``.to()`` call instead of moving.

    Parameter-free on purpose: the deferred restore in ``evaluate()``'s ``finally`` then finds nothing left to move,
    so the recorded calls are exactly the outbound transplant move this stand-in is here to observe.
    """

    def __init__(self) -> None:
        super().__init__()
        self.lock_held: list[bool] = []

    def to(self, *args: Any, **kwargs: Any) -> "_LockTrackingLiveModel":
        """Record whether the device-move lock was held, without performing any move."""
        self.lock_held.append(_device_move_lock(self).locked())
        return self


def test_evaluate_moves_the_live_model_under_the_device_move_lock(
    mocked_training: SimpleNamespace, tmp_path: Path
) -> None:
    """``evaluate()``'s move of the live weights to CPU for the transplant must hold that module's move lock.

    The transplant frees accelerator memory for the freshly built eval module by moving the *live* module — the one
    ``predict()`` runs on — to CPU. That is the same in-place parameter rewrite the deferred first-use move performs, so
    a thread reaching its first ``predict()`` must not be able to run it concurrently.
    """
    live_model = _LockTrackingLiveModel()
    mock_self = MagicMock()
    mock_self.model.model = live_model
    # Anything but "cpu": the transplant's CPU move is skipped for a model that already lives there.
    mock_self.model.device = torch.device("meta")

    RFDETR.evaluate(mock_self, dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))

    assert live_model.lock_held == [True], (
        f"expected one live-model move, taken under the module's move lock, got lock states {live_model.lock_held!r}"
    )


class TestEvaluateSplitDispatch:
    """``split`` selects ``trainer.test`` vs ``trainer.validate`` and validates the value."""

    @pytest.mark.parametrize(
        ("split", "called", "not_called"), [("test", "test", "validate"), ("val", "validate", "test")]
    )
    def test_split_routes_to_trainer_method(
        self,
        mocked_training: SimpleNamespace,
        nano_model: RFDETRNano,
        tmp_path: Path,
        split: str,
        called: str,
        not_called: str,
    ) -> None:
        """``split`` routes to ``trainer.test``/``trainer.validate`` and never passes ``ckpt_path`` (issue #1110).

        The #1110 fix hinges on ``evaluate()`` never passing ``ckpt_path`` to ``trainer.test``/``validate`` — PTL's
        loop-state-restore raises ``KeyError`` when ``ckpt_path`` points at a bare ``.pth``. Inspecting the actual call
        args (not just that the call happened) catches a future regression that reintroduces ``ckpt_path=...``.
        """
        trainer = mocked_training.trainer

        nano_model.evaluate(dataset_dir=str(tmp_path), split=split, output_dir=str(tmp_path / "o"))

        getattr(trainer, called).assert_called_once()
        getattr(trainer, not_called).assert_not_called()
        _, call_kwargs = getattr(trainer, called).call_args
        assert "ckpt_path" not in call_kwargs

    def test_invalid_split_raises(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """An unsupported split value raises ``ValueError``."""
        with pytest.raises(ValueError, match="split"):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="train")  # type: ignore[arg-type]


class TestEvaluateClassCountMismatch:
    """A dataset whose class count differs from the model warns instead of adapting the head."""

    def test_mismatch_warns_on_default_split(self, synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
        """Evaluating the default ``split="test"`` on a mismatched dataset emits a ``UserWarning`` (issue #1110).

        Regression for the class-mismatch warning being dead code on the default split: ``RFDETRDataModule.
        class_names`` previously inspected only ``_dataset_train``/``_dataset_val``, so ``setup("test")`` (which
        builds only ``_dataset_test``) always saw ``class_names is None`` and the warning never fired. This drives
        a real (unmocked) datamodule end to end so the fixed property is exercised, not a stub.
        """
        mismatched_model = RFDETRNano(pretrain_weights=None, num_classes=5, device="cpu")
        with pytest.warns(UserWarning, match="classes"):
            mismatched_model.evaluate(
                dataset_dir=str(synthetic_shape_dataset_dir),
                split="test",
                device="cpu",
                output_dir=str(tmp_path),
                batch_size=4,
                num_workers=0,
                tensorboard=False,
            )

    def test_no_warning_when_class_names_unresolvable(
        self, mocked_training: SimpleNamespace, nano_model: RFDETRNano, tmp_path: Path
    ) -> None:
        """No class-mismatch ``UserWarning`` is emitted when the datamodule cannot resolve class names (``None``)."""
        mocked_training.datamodule.return_value.class_names = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        assert not any("classes" in str(w.message) for w in caught)


class TestEvaluateImportGuard:
    """Evaluate() mirrors train()'s training-extras import guard (issue #1110)."""

    def test_missing_training_extra_raises_install_hint(
        self, nano_model: RFDETRNano, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing optional training dependency raises ImportError with the extras-install hint."""
        real_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rfdetr.training":
                raise ModuleNotFoundError("No module named 'pytorch_lightning'", name="pytorch_lightning")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        with pytest.raises(ImportError) as exc_info:
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        assert "rfdetr[train,loggers]" in str(exc_info.value)
        assert exc_info.value.__cause__ is not None

    def test_internal_training_module_import_error_preserved(
        self, nano_model: RFDETRNano, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A missing internal rfdetr.* training module keeps the original ModuleNotFoundError."""
        real_import = builtins.__import__

        def _mock_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "rfdetr.training":
                raise ModuleNotFoundError("No module named 'rfdetr.training'", name="rfdetr.training")
            return real_import(name, globals, locals, fromlist, level)

        monkeypatch.setattr(builtins, "__import__", _mock_import)
        with pytest.raises(ModuleNotFoundError) as exc_info:
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        assert "rfdetr.training" in str(exc_info.value)


class TestEvaluateTrainerBoundary:
    """Evaluate() drives build_trainer and maps its results at the trainer boundary."""

    def test_device_index_forwarded_and_eval_mode(
        self, mocked_training: SimpleNamespace, nano_model: RFDETRNano, tmp_path: Path
    ) -> None:
        """Device='cuda:1' maps to accelerator='gpu'/devices=[1] and the trainer is built in eval mode.

        The rebuilt module also skips upstream encoder weights: ``evaluate()`` transplants every weight into it, so a
        fetch would be wasted I/O.
        """
        nano_model.evaluate(dataset_dir=str(tmp_path), split="test", device="cuda:1", output_dir=str(tmp_path / "o"))

        _, build_kwargs = mocked_training.build_trainer.call_args
        assert build_kwargs == {"include_training_callbacks": False, "accelerator": "gpu", "devices": [1]}
        assert mocked_training.module.call_args.kwargs["load_encoder_weights"] is False

    def test_empty_results_returns_empty_dict(
        self, mocked_training: SimpleNamespace, nano_model: RFDETRNano, tmp_path: Path
    ) -> None:
        """When the trainer yields no metrics, evaluate() returns an empty dict."""
        mocked_training.trainer.test.return_value = []

        metrics = nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))

        assert metrics == {}


class TestEvaluateResolutionOverride:
    """A resolution override reconciles positional embeddings so the in-memory transplant still loads."""

    def test_resolution_override_evaluates(
        self, mocker: MockerFixture, synthetic_shape_dataset_dir: Path, tmp_path: Path
    ) -> None:
        """evaluate(resolution=...) interpolates PE, returns metrics, and leaves model_config unchanged.

        The PTL trainer is mocked so only the resolution-sensitive build + state-dict transplant run (the line that
        would raise on a PE-shape mismatch if interpolation were skipped).
        """
        model = RFDETRNano(
            pretrain_weights=None,
            num_classes=_num_classes(synthetic_shape_dataset_dir),
            device="cpu",
        )
        original_resolution = model.model_config.resolution
        original_pe = model.model_config.positional_encoding_size
        block_size = model.model_config.patch_size * model.model_config.num_windows
        new_resolution = block_size * (model.model_config.resolution // block_size + 1)
        mocker.patch("rfdetr.training.build_trainer", return_value=_mock_trainer())

        metrics = model.evaluate(
            dataset_dir=str(synthetic_shape_dataset_dir),
            split="test",
            resolution=new_resolution,
            device="cpu",
            output_dir=str(tmp_path / "o"),
            batch_size=4,
            num_workers=0,
            tensorboard=False,
        )

        assert "test/mAP_50_95" in metrics
        assert model.model_config.resolution == original_resolution
        assert model.model_config.positional_encoding_size == original_pe

    def test_resolution_override_resizes_real_batches(
        self, mocker: MockerFixture, synthetic_shape_dataset_dir: Path, tmp_path: Path
    ) -> None:
        """evaluate(resolution=...) must actually feed the model images resized to the override, repeatably.

        Unlike ``test_resolution_override_evaluates`` above, the PTL trainer is NOT mocked here: this runs a real
        ``trainer.test`` pass and inspects the pixel tensor shape ``RFDETRModelModule.test_step`` receives, which is
        what the datamodule (not the PE-interpolated model) actually controls. A datamodule built from the wrong config
        would silently keep evaluating at the original resolution while the model's positional embeddings were
        interpolated for the override, invalidating any resolution-vs-mAP comparison.

        Runs three sequential calls on the same model instance: a first override, a second and distinct override, then
        a call with no override at all. This checks that the fix isn't specific to the first transition and that it
        releases the datamodule back to the model's original resolution afterwards, not just to the previous override.
        """
        from rfdetr.training import RFDETRModelModule

        model = RFDETRNano(
            pretrain_weights=None,
            num_classes=_num_classes(synthetic_shape_dataset_dir),
            device="cpu",
        )
        original_resolution = model.model_config.resolution
        block_size = model.model_config.patch_size * model.model_config.num_windows
        first_override = block_size * (original_resolution // block_size + 1)
        second_override = block_size * (original_resolution // block_size + 2)

        observed_shapes: list[tuple[int, int]] = []
        original_test_step = RFDETRModelModule.test_step

        def recording_test_step(self: RFDETRModelModule, batch: Any, batch_idx: int) -> Any:
            """Spy on ``test_step`` and record the pixel shape of each batch it actually receives."""
            observed_shapes.append(tuple(batch[0].tensors.shape[-2:]))
            return original_test_step(self, batch, batch_idx)

        mocker.patch.object(RFDETRModelModule, "test_step", recording_test_step)

        def run_and_get_shape(resolution: int | None, output_dir: Path) -> tuple[int, int]:
            """Run one evaluate() call, spying on test_step, and return the single uniform batch shape it saw."""
            observed_shapes.clear()
            kwargs: dict[str, Any] = {"resolution": resolution} if resolution is not None else {}
            model.evaluate(
                dataset_dir=str(synthetic_shape_dataset_dir),
                split="test",
                device="cpu",
                output_dir=str(output_dir),
                batch_size=4,
                num_workers=0,
                tensorboard=False,
                **kwargs,
            )
            assert observed_shapes, "test_step was never called"
            shapes = set(observed_shapes)
            assert len(shapes) == 1, f"evaluate(resolution={resolution}) fed inconsistently shaped batches: {shapes}"
            return next(iter(shapes))

        assert run_and_get_shape(first_override, tmp_path / "o1") == (first_override, first_override), (
            f"evaluate(resolution={first_override}) did not resize batches to the override."
        )
        assert run_and_get_shape(second_override, tmp_path / "o2") == (second_override, second_override), (
            f"evaluate(resolution={second_override}) did not resize batches to the override on a second, distinct "
            "call — the datamodule fix must not be specific to the first transition."
        )
        assert run_and_get_shape(None, tmp_path / "o3") == (original_resolution, original_resolution), (
            "evaluate() without a resolution override did not return to the model's original resolution after "
            "two prior overrides — the datamodule must not stay pinned to the last override."
        )


def test_auto_batch_probe_not_invoked(
    mocked_training: SimpleNamespace, mocker: MockerFixture, nano_model: RFDETRNano, tmp_path: Path
) -> None:
    """``evaluate(batch_size="auto")`` skips the forward+backward prober and uses the default micro-batch.

    ``batch_size="auto"`` is a train-only feature; evaluate() must not run the training-style probe.
    """
    mock_probe = mocker.patch("rfdetr.training.auto_batch.resolve_auto_batch_config")

    nano_model.evaluate(dataset_dir=str(tmp_path), split="test", batch_size="auto", output_dir=str(tmp_path / "o"))

    mock_probe.assert_not_called()
    passed_config = mocked_training.build_trainer.call_args.args[0]
    assert passed_config.batch_size == TrainConfig.model_fields["batch_size"].default


def test_evaluate_fp8_auto_batch_uses_default_micro_batch(
    mocked_training: SimpleNamespace, mocker: MockerFixture, nano_model: RFDETRNano, tmp_path: Path
) -> None:
    """``evaluate(amp_dtype="fp8", batch_size="auto")`` must not raise before ``for_eval`` is consulted.

    ``TrainConfig.validate_fp8_batch_size`` rejects ``amp_dtype="fp8"`` with ``batch_size="auto"`` unconditionally at
    construction time, because the training-only probe never runs against a converted Transformer Engine model. That
    validator fires before ``_prepare_run_config``'s ``for_eval`` branch — which already skips the probe and would
    happily substitute the default micro-batch — is ever reached, so the combination must be resolved before
    ``TrainConfig`` is constructed, not after.
    """
    mock_probe = mocker.patch("rfdetr.training.auto_batch.resolve_auto_batch_config")

    nano_model.evaluate(
        dataset_dir=str(tmp_path), split="test", amp_dtype="fp8", batch_size="auto", output_dir=str(tmp_path / "o")
    )

    mock_probe.assert_not_called()
    passed_config = mocked_training.build_trainer.call_args.args[0]
    assert passed_config.amp_dtype == "fp8"
    assert passed_config.batch_size == TrainConfig.model_fields["batch_size"].default


def test_train_then_from_checkpoint_then_evaluate(synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
    """Train() writes a real checkpoint; from_checkpoint() reloads it; evaluate() runs without the PTL ``ckpt_path``
    ``KeyError``.

    End-to-end repro for issue #1110: evaluate() on a real trained-then-reloaded checkpoint must not raise. Issue
    #1110's reported path was ``trainer.test(ckpt_path=...)`` against a bare ``.pth`` raising a PTL loop- state-restore
    ``KeyError``. All other tests in this module build the state-dict transplant against an untrained in-memory model;
    this is the only one that exercises a real checkpoint round trip (train → save → reload as a new instance →
    evaluate), the exact case the issue asked for.

    ``resolution=224`` (the synthetic fixture's own image size, so no resize upscaling) shrinks the backbone forward
    pass to about a third of the default 384px FLOPs; the round-trip only needs finite, present metric keys, not a
    representative resolution. The 600s timeout (vs. the suite's 420s default) is headroom for this real train+
    checkpoint+reload+evaluate path on a slow CPU runner, not a promise this test needs that long normally.
    """
    output_dir = tmp_path / "train_output"
    model = RFDETRNano(
        pretrain_weights=None,
        num_classes=_num_classes(synthetic_shape_dataset_dir),
        device="cpu",
        resolution=224,
    )
    model.train(
        dataset_dir=str(synthetic_shape_dataset_dir),
        epochs=1,
        batch_size=4,
        grad_accum_steps=1,
        num_workers=0,
        output_dir=str(output_dir),
        device="cpu",
        tensorboard=False,
    )
    checkpoint_path = output_dir / "checkpoint_best_total.pth"
    assert checkpoint_path.exists(), "train() should have written checkpoint_best_total.pth"

    loaded_model = RFDETR.from_checkpoint(checkpoint_path)
    metrics = loaded_model.evaluate(
        dataset_dir=str(synthetic_shape_dataset_dir),
        split="test",
        device="cpu",
        output_dir=str(tmp_path / "eval_output"),
        batch_size=4,
        num_workers=0,
        tensorboard=False,
    )
    assert "test/mAP_50_95" in metrics
    assert all(torch.isfinite(torch.tensor(float(v))) for v in metrics.values())


class TestEvaluateLowPriorityGaps:
    """Real (non-mocked) split="val" metrics, and datamodule state across sequential evaluate() calls."""

    def test_val_split_returns_real_map_key(
        self, nano_model: RFDETRNano, synthetic_shape_dataset_dir: Path, tmp_path: Path
    ) -> None:
        """A real (unmocked) ``evaluate(split="val")`` run returns a genuine ``val/mAP_50_95`` key.

        Every other split="val"-adjacent assertion in this module mocks the PTL stack; this exercises a real
        ``trainer.validate`` pass end to end, mirroring the module-scoped ``evaluation_result`` fixture's split="test"
        coverage.
        """
        metrics = nano_model.evaluate(
            dataset_dir=str(synthetic_shape_dataset_dir),
            split="val",
            device="cpu",
            output_dir=str(tmp_path / "val_eval_output"),
            batch_size=4,
            num_workers=0,
            tensorboard=False,
        )
        assert "val/mAP_50_95" in metrics
        assert torch.isfinite(torch.tensor(float(metrics["val/mAP_50_95"])))

    def test_sequential_evaluate_calls_do_not_leak_state(
        self, synthetic_shape_dataset_dir: Path, tmp_path: Path
    ) -> None:
        """Two sequential ``evaluate()`` calls on the same instance return consistent metrics (no state leakage)."""
        model = RFDETRNano(
            pretrain_weights=None,
            num_classes=_num_classes(synthetic_shape_dataset_dir),
            device="cpu",
        )
        kwargs = dict(
            dataset_dir=str(synthetic_shape_dataset_dir),
            split="test",
            device="cpu",
            batch_size=4,
            num_workers=0,
            tensorboard=False,
        )
        first = model.evaluate(output_dir=str(tmp_path / "eval_1"), **kwargs)
        second = model.evaluate(output_dir=str(tmp_path / "eval_2"), **kwargs)
        assert first.keys() == second.keys()
        for key in first:
            assert first[key] == pytest.approx(second[key]), f"{key} differs between sequential evaluate() calls"


def test_evaluate_matches_manual_building_blocks_on_test_split(
    synthetic_shape_dataset_dir: Path, tmp_path: Path
) -> None:
    """``evaluate(split="test")`` and the manually-assembled PTL blocks return the same metrics.

    ``evaluate()`` is a convenience wrapper over the manual PTL building-block pattern documented in
    ``docs/learn/train/customization.md`` (``RFDETRModelModule`` + ``RFDETRDataModule`` + ``build_trainer`` +
    ``trainer.test``/``validate``, the same pattern exercised in ``tests/benchmarks/test_inference_coco.py::
    test_inference_ptl_predict``). Asserts the two paths return identical metrics for the same model and
    dataset, so the convenience wrapper cannot silently drift from the documented decomposed pattern.
    """
    from rfdetr.training import RFDETRDataModule, RFDETRModelModule, build_trainer

    num_classes = _num_classes(synthetic_shape_dataset_dir)

    # Path 1: the public evaluate() convenience API.
    eval_model = RFDETRNano(pretrain_weights=None, num_classes=num_classes, device="cpu")
    eval_metrics = eval_model.evaluate(
        dataset_dir=str(synthetic_shape_dataset_dir),
        split="test",
        device="cpu",
        output_dir=str(tmp_path / "evaluate_output"),
        batch_size=4,
        num_workers=0,
        tensorboard=False,
    )

    # Path 2: the manual decomposed PTL building blocks from docs/learn/train/customization.md. Weights are
    # synced from eval_model (evaluate() never mutates them -- TestEvaluateDoesNotMutateModel) so both paths
    # score the exact same model.
    tc = TrainConfig(
        dataset_file="roboflow",
        dataset_dir=str(synthetic_shape_dataset_dir),
        output_dir=str(tmp_path / "manual_output"),
        batch_size=4,
        num_workers=0,
        tensorboard=False,
        wandb=False,
        mlflow=False,
        clearml=False,
    )
    module = RFDETRModelModule(eval_model.model_config, tc)
    module.model.load_state_dict(eval_model.model.model.state_dict())
    module.model.eval()
    datamodule = RFDETRDataModule(eval_model.model_config, tc)
    trainer = build_trainer(tc, eval_model.model_config, accelerator="cpu", include_training_callbacks=False)
    (manual_metrics,) = trainer.test(module, datamodule)

    assert eval_metrics.keys() == manual_metrics.keys()
    for key in eval_metrics:
        assert eval_metrics[key] == pytest.approx(float(manual_metrics[key])), (
            f"{key}: evaluate()={eval_metrics[key]} manual={manual_metrics[key]}"
        )


def _get_patch_embed_projection(model) -> torch.nn.Conv2d:
    """Return the patch-embedding projection layer for an RF-DETR model.

    RFDETR wrappers are not nn.Module; the underlying PyTorch module lives at ``model.model.model``.  Walk
    named_modules() on that object.

    Args:
        model: Instantiated RF-DETR wrapper (RFDETRNano / RFDETRLarge).

    Returns:
        The convolution used to project image channels into patch embeddings.

    Raises:
        AssertionError: If the patch-embedding projection cannot be located.

    Examples:
        >>> model = RFDETRLarge(pretrain_weights=None, num_classes=1, device="cpu")
        >>> isinstance(_get_patch_embed_projection(model), torch.nn.Conv2d)
        True
    """
    # model.model → model context; model.model.model → nn.Module
    nn_model = model.model.model
    proj = nn_model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection
    if isinstance(proj, torch.nn.Conv2d):
        return proj

    # Fallback: scan named_modules on the underlying nn.Module
    for name, module in nn_model.named_modules():
        if "patch_embeddings" in name and "projection" in name and isinstance(module, torch.nn.Conv2d):
            return module

    msg = "Could not find patch embedding projection on model"
    raise AssertionError(msg)


@pytest.mark.parametrize("model_class", [RFDETRLarge, RFDETRNano])
@pytest.mark.parametrize("channels", [1, 4])
def test_patch_embed_in_channels_match_num_channels(model_class, channels: int) -> None:
    """A model built with ``num_channels=N`` projects N-channel input in its patch embedding."""
    model = model_class(
        num_channels=channels,
        device="cpu",
        pretrain_weights=None,
    )

    patch_embed_projection = _get_patch_embed_projection(model)

    assert patch_embed_projection.in_channels == channels


def test_default_detection_forward_accepts_backbone_tuple() -> None:
    """LWDETR should accept the backbone tuple and wrap a 4-D input as unpadded."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 4
    num_classes = 7

    features = [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]
    poss = [torch.zeros(batch_size, 4, 4, dtype=torch.bool)]

    backbone = MagicMock()
    backbone.return_value = (features, poss, None)

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer_out = (
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, hidden_dim),
    )
    transformer.return_value = transformer_out

    model = LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
    )

    images = torch.ones(batch_size, 3, 8, 8)
    outputs = model(images)

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
    nested = backbone.call_args.args[0]
    assert isinstance(nested, NestedTensor)
    assert nested.no_padding is True
    assert nested.tensors is images


class TestModelChannelStats:
    """``RFDETR.means`` and ``RFDETR.stds`` repeat the RGB statistics over the model's channels."""

    @pytest.mark.parametrize(
        ("num_channels", "expected_means", "expected_stds"),
        [
            pytest.param(1, [0.485], [0.229], id="grayscale"),
            pytest.param(3, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225], id="rgb"),
            pytest.param(4, [0.485, 0.456, 0.406, 0.485], [0.229, 0.224, 0.225, 0.229], id="four-channel"),
        ],
    )
    def test_stats_cover_each_channel(
        self,
        num_channels: int,
        expected_means: list[float],
        expected_stds: list[float],
        mocker: MockerFixture,
    ) -> None:
        """Extra channels reuse the RGB values in order, and the statistics stay lists."""
        # The statistics are set in ``__init__`` and do not depend on the network, so skip building it.
        mocker.patch.object(RFDETRNano, "get_model")

        model = RFDETRNano(num_channels=num_channels, pretrain_weights=None, device="cpu")

        # A tuple of the same values compares unequal to a list, so this also pins the list type.
        assert (model.means, model.stds) == (expected_means, expected_stds)

    @pytest.mark.parametrize("num_channels", [1, 3, 4])
    def test_stats_match_training_normalization(self, num_channels: int, mocker: MockerFixture) -> None:
        """Prediction normalizes every channel with the values the training pipelines compute."""
        mocker.patch.object(RFDETRNano, "get_model")

        model = RFDETRNano(num_channels=num_channels, pretrain_weights=None, device="cpu")

        training_stats = (
            list(cycle_channel_stats(IMAGENET_MEAN, num_channels)),
            list(cycle_channel_stats(IMAGENET_STD, num_channels)),
        )
        assert (model.means, model.stds) == training_stats


_BATCH, _HIDDEN, _QUERIES, _ENC_QUERIES, _CLASSES, _IMAGE, _LAYERS = 2, 8, 3, 4, 5, 8, 3


def _build_seg_lwdetr(with_decoder: bool = True) -> LWDETR:
    """Build a two-stage segmentation ``LWDETR`` whose backbone and transformer return fixed random tensors.

    The segmentation head is real, with one block per decoder layer as ``build_model`` creates it, so the masks it
    produces are the ones a released ``RFDETRSeg*`` model would compute from these hidden states.

    Args:
        with_decoder: When ``False`` the transformer returns no decoder hidden states, the decoder-less two-stage
            configuration in which the encoder proposals are the model's final predictions.

    Returns:
        The model, in training mode like any freshly built ``nn.Module``.

    Examples:
        >>> model = _build_seg_lwdetr()
        >>> len(model.segmentation_head.blocks)
        3
    """
    generator = torch.Generator().manual_seed(0)
    features = [
        NestedTensor(
            torch.randn(_BATCH, _HIDDEN, 4, 4, generator=generator),
            torch.zeros(_BATCH, 4, 4, dtype=torch.bool),
        )
    ]
    backbone = MagicMock(return_value=(features, [torch.zeros(_BATCH, _HIDDEN, 4, 4)], None))
    transformer = MagicMock()
    transformer.d_model = _HIDDEN
    hs = torch.randn(_LAYERS, _BATCH, _QUERIES, _HIDDEN, generator=generator)
    ref_unsigmoid = torch.rand(_LAYERS, _BATCH, _QUERIES, 4, generator=generator) * 0.4 + 0.3
    transformer.return_value = (
        hs if with_decoder else None,
        ref_unsigmoid if with_decoder else None,
        torch.randn(_BATCH, _ENC_QUERIES, _HIDDEN, generator=generator),  # hs_enc
        torch.rand(_BATCH, _ENC_QUERIES, 4, generator=generator) * 0.4 + 0.3,  # ref_enc
        torch.randn(_BATCH, _ENC_QUERIES, _CLASSES, generator=generator),  # encoder class logits
    )
    torch.manual_seed(0)
    return LWDETR(
        backbone=backbone,
        transformer=transformer,
        segmentation_head=SegmentationHead(in_dim=_HIDDEN, num_blocks=_LAYERS, bottleneck_ratio=1, downsample_ratio=1),
        num_classes=_CLASSES,
        num_queries=_QUERIES,
        aux_loss=True,
        group_detr=1,
        two_stage=True,
        bbox_reparam=False,
    )


def _images() -> Tensor:
    """Return a fixed input batch; the mocked backbone ignores its values, only the spatial size matters.

    Examples:
        >>> tuple(_images().shape)
        (2, 3, 8, 8)
    """
    return torch.zeros(_BATCH, 3, _IMAGE, _IMAGE)


class TestEvalForwardMasks:
    """Eval mode computes the last decoder layer's masks only; ``all_layer_masks=True`` returns the rest unchanged."""

    def test_eval_forward_skips_auxiliary_and_encoder_masks(self) -> None:
        """By default the auxiliary layers and encoder proposals carry no ``pred_masks`` in eval mode."""
        model = _build_seg_lwdetr().eval()

        with torch.no_grad():
            outputs = model(_images())

        assert outputs["pred_masks"].shape == (_BATCH, _QUERIES, _IMAGE, _IMAGE)
        assert len(outputs["aux_outputs"]) == _LAYERS - 1, "every auxiliary layer must keep its logits and boxes"
        for layer_idx, aux in enumerate(outputs["aux_outputs"]):
            assert "pred_logits" in aux and "pred_boxes" in aux, f"aux layer {layer_idx} lost its logits or boxes"
            assert "pred_masks" not in aux, f"aux layer {layer_idx} still computed dense mask logits in eval mode"
        assert "pred_logits" in outputs["enc_outputs"]
        assert "pred_masks" not in outputs["enc_outputs"], "encoder proposals still computed mask logits in eval mode"

    def test_final_masks_bit_exact_with_and_without_all_layer_masks(self) -> None:
        """Skipping the other layers leaves ``pred_masks`` identical, and ``all_layer_masks`` returns every layer's."""
        model = _build_seg_lwdetr().eval()

        with torch.no_grad():
            final_only = model(_images())
            all_layers = model(_images(), all_layer_masks=True)
            features = model.backbone.return_value[0][0].tensors
            head_masks = model.segmentation_head(features, model.transformer.return_value[0], (_IMAGE, _IMAGE))

        for key in ("pred_masks", "pred_logits", "pred_boxes"):
            assert torch.equal(final_only[key], all_layers[key]), f"{key} changed when auxiliary masks were skipped"
        for layer_idx, aux in enumerate(all_layers["aux_outputs"]):
            assert torch.equal(aux["pred_masks"], head_masks[layer_idx]), f"aux layer {layer_idx} masks differ"
        assert all_layers["enc_outputs"]["pred_masks"].shape == (_BATCH, _ENC_QUERIES, _IMAGE, _IMAGE)

    def test_decoderless_eval_still_returns_encoder_masks(self) -> None:
        """With no decoder the encoder proposals are the final predictions, so their masks are still computed."""
        model = _build_seg_lwdetr(with_decoder=False).eval()

        with torch.no_grad():
            outputs = model(_images())

        assert outputs["pred_masks"].shape == (_BATCH, _ENC_QUERIES, _IMAGE, _IMAGE)

    @pytest.mark.parametrize(
        ("all_layer_masks", "projections"),
        [(False, 1), (True, _LAYERS + 1)],
    )
    def test_eval_forward_projects_once_per_returned_mask_set(self, all_layer_masks: bool, projections: int) -> None:
        """Masks that eval mode leaves out are never computed: the head projects once per returned mask set."""
        model = _build_seg_lwdetr().eval()
        calls: list[None] = []
        model.segmentation_head.spatial_features_proj.register_forward_hook(lambda *_: calls.append(None))

        with torch.no_grad():
            model(_images(), all_layer_masks=all_layer_masks)

        assert len(calls) == projections

    def test_training_forward_keeps_every_layers_sparse_masks(self) -> None:
        """Training mode still gives the criterion sparse masks for every decoder layer and the encoder proposals."""
        model = _build_seg_lwdetr().train()

        outputs = model(_images())

        layers = [outputs, *outputs["aux_outputs"], outputs["enc_outputs"]]
        assert [isinstance(layer.get("pred_masks"), dict) for layer in layers] == [True] * (_LAYERS + 1)
