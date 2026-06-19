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
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

from rfdetr import RFDETRNano


def _num_classes(dataset_dir: Path) -> int:
    """Return the COCO category count from a Roboflow-style dataset's train split."""
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
    """
    trainer = MagicMock()
    trainer.test.return_value = [{"test/mAP_50_95": 0.0}]
    trainer.validate.return_value = [{"val/mAP_50_95": 0.0}]
    return trainer


class TestEvaluateSplitDispatch:
    """``split`` selects ``trainer.test`` vs ``trainer.validate`` and validates the value."""

    def test_split_test_calls_trainer_test(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """``split='test'`` routes to ``trainer.test``."""
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
        ):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))
        trainer.test.assert_called_once()
        trainer.validate.assert_not_called()

    def test_split_val_calls_trainer_validate(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """``split='val'`` routes to ``trainer.validate``."""
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
        ):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="val", output_dir=str(tmp_path / "o"))
        trainer.validate.assert_called_once()
        trainer.test.assert_not_called()

    def test_invalid_split_raises(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """An unsupported split value raises ``ValueError``."""
        with pytest.raises(ValueError, match="split"):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="train")  # type: ignore[arg-type]


class TestEvaluateKwargParity:
    """``evaluate()`` accepts and handles the same kwargs as ``train()`` (shared preprocessing)."""

    def test_shares_train_kwarg_handling(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """A deprecated train kwarg (``do_benchmark``) is absorbed and warned, proving shared handling."""
        trainer = _mock_trainer()
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule"),
            patch("rfdetr.training.build_trainer", return_value=trainer),
            pytest.warns(DeprecationWarning, match="do_benchmark"),
        ):
            nano_model.evaluate(
                dataset_dir=str(tmp_path), split="test", do_benchmark=True, output_dir=str(tmp_path / "o")
            )


class TestEvaluateClassCountMismatch:
    """A dataset whose class count differs from the model warns instead of adapting the head."""

    def test_mismatch_warns(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """Evaluating on a dataset with a different class count emits a ``UserWarning``."""
        trainer = _mock_trainer()
        datamodule = MagicMock()
        datamodule.class_names = ["only", "two"]  # model has 3 classes
        with (
            patch("rfdetr.training.RFDETRModelModule"),
            patch("rfdetr.training.RFDETRDataModule", return_value=datamodule),
            patch("rfdetr.training.build_trainer", return_value=trainer),
            pytest.warns(UserWarning, match="classes"),
        ):
            nano_model.evaluate(dataset_dir=str(tmp_path), split="test", output_dir=str(tmp_path / "o"))


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
        """evaluate(resolution=...) interpolates PE and returns metrics.

        Uses a fresh model because the resolution override mutates ``model_config`` in place; the PTL trainer is mocked
        so only the resolution-sensitive build + state-dict transplant run (the line that would raise on a PE-shape
        mismatch if interpolation were skipped).
        """
        model = RFDETRNano(
            pretrain_weights=None,
            num_classes=_num_classes(synthetic_shape_dataset_dir),
            device="cpu",
        )
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
