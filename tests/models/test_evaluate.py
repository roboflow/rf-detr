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
"""

import builtins
import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr import RFDETR, RFDETRNano


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

    def test_contains_map_metric(self, evaluation_result: SimpleNamespace) -> None:
        """The primary COCO mAP key is present for the requested split."""
        assert "test/mAP_50_95" in evaluation_result.metrics

    def test_contains_f1_metric(self, evaluation_result: SimpleNamespace) -> None:
        """The F1-sweep metric is present for the requested split."""
        assert "test/F1" in evaluation_result.metrics

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


class TestEvaluateSplitDispatch:
    """``split`` selects ``trainer.test`` vs ``trainer.validate`` and validates the value."""

    def test_split_test_calls_trainer_test(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """``split='test'`` routes to ``trainer.test`` and never passes ``ckpt_path`` (issue #1110 regression).

        The #1110 fix hinges on ``evaluate()`` never passing ``ckpt_path`` to ``trainer.test``/``validate`` — PTL's
        loop-state-restore raises ``KeyError`` when ``ckpt_path`` points at a bare ``.pth``. Inspecting the actual call
        args (not just that the call happened) catches a future regression that reintroduces ``ckpt_path=...``.
        """
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
        ):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        trainer.test.assert_called_once()
        trainer.validate.assert_not_called()
        _, test_kwargs = trainer.test.call_args
        assert "ckpt_path" not in test_kwargs

    def test_split_val_calls_trainer_validate(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """``split='val'`` routes to ``trainer.validate`` and never passes ``ckpt_path`` (issue #1110 regression)."""
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
        ):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="val", output_dir=str(tmp_path / "o"))
        trainer.validate.assert_called_once()
        trainer.test.assert_not_called()
        _, validate_kwargs = trainer.validate.call_args
        assert "ckpt_path" not in validate_kwargs

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

    def test_no_warning_when_class_names_unresolvable(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """No class-mismatch ``UserWarning`` is emitted when the datamodule cannot resolve class names (the ``None``
        branch)."""
        trainer = _mock_trainer()
        datamodule = MagicMock()
        datamodule.class_names = None
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule", return_value=datamodule),
            patch("rfdetr.training.build_trainer", return_value=trainer),
            warnings.catch_warnings(record=True) as caught,
        ):
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
        with pytest.raises(ImportError, match=r"rfdetr\[train,loggers\]") as exc_info:
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
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
        with pytest.raises(ModuleNotFoundError, match=r"rfdetr\.training"):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))


class TestEvaluateTrainerBoundary:
    """Evaluate() drives build_trainer and maps its results at the trainer boundary."""

    def test_device_index_forwarded_and_eval_mode(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """Device='cuda:1' maps to accelerator='gpu'/devices=[1] and the trainer is built in eval mode."""
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer) as mock_build,
        ):
            nano_model.evaluate(
                dataset_dir=str(tmp_path), split="test", device="cuda:1", output_dir=str(tmp_path / "o")
            )
        _, build_kwargs = mock_build.call_args
        assert build_kwargs == {"include_training_callbacks": False, "accelerator": "gpu", "devices": [1]}

    def test_empty_results_returns_empty_dict(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """When the trainer yields no metrics, evaluate() returns an empty dict."""
        trainer = MagicMock()
        trainer.test.return_value = []
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
        ):
            metrics = nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        assert metrics == {}


class TestEvaluateResolutionOverride:
    """A resolution override reconciles positional embeddings so the in-memory transplant still loads."""

    def test_resolution_override_evaluates(self, synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
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
        trainer = _mock_trainer()
        with patch("rfdetr.training.build_trainer", return_value=trainer):
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

    def test_resolution_override_resizes_real_batches(self, synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
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

        def run_and_get_shape(resolution: int | None, output_dir: Path) -> tuple[int, int]:
            """Run one evaluate() call, spying on test_step, and return the single uniform batch shape it saw."""
            observed_shapes.clear()
            kwargs: dict[str, Any] = {"resolution": resolution} if resolution is not None else {}
            with patch.object(RFDETRModelModule, "test_step", recording_test_step):
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


def test_auto_batch_probe_not_invoked(nano_model: RFDETRNano, tmp_path: Path) -> None:
    """``evaluate(batch_size="auto")`` skips the forward+backward prober and uses the default micro-batch.

    ``batch_size="auto"`` is a train-only feature; evaluate() must not run the training-style probe.
    """
    from rfdetr.config import TrainConfig

    trainer = _mock_trainer()
    with (
        patch("rfdetr.training.RFDETRModelModule"),
        patch("rfdetr.training.RFDETRDataModule"),
        patch("rfdetr.training.build_trainer", return_value=trainer) as mock_build_trainer,
        patch("rfdetr.training.auto_batch.resolve_auto_batch_config") as mock_probe,
    ):
        nano_model.evaluate(dataset_dir=str(tmp_path), split="test", batch_size="auto", output_dir=str(tmp_path / "o"))
    mock_probe.assert_not_called()
    passed_config = mock_build_trainer.call_args.args[0]
    assert passed_config.batch_size == TrainConfig.model_fields["batch_size"].default


def test_train_then_from_checkpoint_then_evaluate(synthetic_shape_dataset_dir: Path, tmp_path: Path) -> None:
    """Train() writes a real checkpoint; from_checkpoint() reloads it; evaluate() runs without the PTL ``ckpt_path``
    ``KeyError``.

    End-to-end repro for issue #1110: evaluate() on a real trained-then-reloaded checkpoint must not raise. Issue
    #1110's reported path was ``trainer.test(ckpt_path=...)`` against a bare ``.pth`` raising a PTL loop- state-restore
    ``KeyError``. All other tests in this module build the state-dict transplant against an untrained in-memory model;
    this is the only one that exercises a real checkpoint round trip (train → save → reload as a new instance →
    evaluate), the exact case the issue asked for.
    """
    output_dir = tmp_path / "train_output"
    model = RFDETRNano(
        pretrain_weights=None,
        num_classes=_num_classes(synthetic_shape_dataset_dir),
        device="cpu",
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
    test_inference_detection_ptl_predict``). Asserts the two paths return identical metrics for the same model and
    dataset, so the convenience wrapper cannot silently drift from the documented decomposed pattern.
    """
    from rfdetr.config import TrainConfig
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
