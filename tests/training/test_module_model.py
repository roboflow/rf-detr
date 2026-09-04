# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Comprehensive unit tests for RFDETRModelModule (LightningModule wrapper)."""

import random
import warnings
from types import SimpleNamespace
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from pytorch_lightning import Callback, Trainer
from torch import nn

from rfdetr.config import RFDETRBaseConfig, RFDETRSmallConfig, TrainConfig
from rfdetr.models.lwdetr import build_criterion_from_config, build_model_from_config
from rfdetr.models.weights import apply_lora, load_pretrain_weights
from rfdetr.training.callbacks.best_model import RFDETREarlyStopping
from rfdetr.training.module_data import RFDETRDataModule
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor

from .helpers import _fake_postprocess as _helpers_fake_postprocess
from .helpers import _FakeCriterion, _FakeDataset, _make_param_dicts, _TinyModel

# ---------------------------------------------------------------------------
# Private helpers — used by both module-level fixtures and class-level _setup_*
# methods (which cannot inject pytest fixtures directly).
# Only define a private helper when it is called from more than one site;
# single-use logic belongs directly in the fixture body.
# ---------------------------------------------------------------------------


def _base_model_config(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled.

    Examples:
        >>> config = _base_model_config(num_classes=7)
        >>> config.device, config.num_classes, config.pretrain_weights
        ('cpu', 7, None)
    """
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig suitable for unit tests.

    Examples:
        >>> config = _base_train_config(batch_size=4)
        >>> config.batch_size, config.dataset_dir.endswith("dataset"), config.output_dir.endswith("output")
        (4, True, True)
    """
    dataset_dir = str(tmp_path / "dataset") if tmp_path else "/nonexistent/dataset"
    output_dir = str(tmp_path / "output") if tmp_path else "/nonexistent/output"
    defaults = dict(
        dataset_dir=dataset_dir,
        output_dir=output_dir,
        epochs=10,
        lr=1e-4,
        lr_encoder=1.5e-4,
        batch_size=2,
        weight_decay=1e-4,
        warmup_epochs=1.0,
        drop_path=0.0,
        multi_scale=False,
        expanded_scales=False,
        do_random_resize_via_padding=False,
        grad_accum_steps=1,
        tensorboard=False,
    )
    defaults.update(overrides)
    return TrainConfig(**defaults)


def _fake_model():
    """Return a MagicMock that behaves enough like an LWDETR model.

    Examples:
        >>> model = _fake_model()
        >>> isinstance(next(model.parameters()), nn.Parameter)
        True
    """
    model = MagicMock(spec=nn.Module)
    real_param = nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = iter([real_param])
    model.named_parameters.return_value = iter([("weight", real_param)])
    model.update_drop_path = MagicMock()
    model.update_dropout = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    return model


def _fake_criterion():
    """Return a MagicMock criterion with a realistic weight_dict.

    Examples:
        >>> criterion = _fake_criterion()
        >>> sorted(criterion.weight_dict)
        ['loss_bbox', 'loss_ce', 'loss_giou']
    """
    criterion = MagicMock()
    criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    criterion.num_boxes_for_targets.return_value = torch.tensor(1.0)
    return criterion


def _fake_postprocess():
    """Return a callable MagicMock for postprocess.

    Examples:
        >>> import torch
        >>> postprocess = _fake_postprocess()
        >>> sorted(postprocess({}, torch.zeros(1, 2))[0])
        ['boxes', 'labels', 'scores']
    """
    return MagicMock(return_value=[{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1)}])


class _RecordingOptimizer(torch.optim.Optimizer):
    """Optimizer test double that records constructor defaults and kwargs."""

    def __init__(self, params, lr=1e-3, weight_decay=0.0, **kwargs):
        self.extra_kwargs = dict(kwargs)
        super().__init__(params, {"lr": lr, "weight_decay": weight_decay, **kwargs})

    def step(self, closure=None):
        """Run an optimizer step for the test double."""
        if closure is not None:
            return closure()
        return None


class _NoWeightDecayOptimizer(torch.optim.Optimizer):
    """Optimizer test double whose constructor does not accept ``weight_decay``."""

    def __init__(self, params, lr=1e-3, momentum=0.0):
        super().__init__(params, {"lr": lr, "momentum": momentum})

    def step(self, closure=None):
        """Run an optimizer step for the test double."""
        if closure is not None:
            return closure()
        return None


class _RaisingOptimizer:
    """Optimizer test double that always fails to construct."""

    def __init__(self, *args, **kwargs):
        raise TypeError("simulated optimizer construction failure")


class _StepHookOptimizer:
    """Optimizer double that fires ``on_before_optimizer_step`` from ``step()``, like ``LightningOptimizer``.

    Real ``LightningOptimizer.step()`` routes through ``Precision.optimizer_step``, which invokes the
    ``on_before_optimizer_step`` hook before running the wrapped optimizer's own ``step()``. This double reproduces just
    that ordering so tests can exercise the module's hook wiring without standing up the full Lightning
    precision/strategy stack.
    """

    def __init__(self, module, optimizer):
        self._module = module
        self._optimizer = optimizer

    @property
    def param_groups(self):
        """Proxy the wrapped optimizer's parameter groups."""
        return self._optimizer.param_groups

    def step(self, *args, **kwargs):
        """Fire the before-step hook, then delegate to the wrapped optimizer's ``step()``."""
        self._module.on_before_optimizer_step(self)
        self._optimizer.step(*args, **kwargs)

    def zero_grad(self, *args, **kwargs):
        """Delegate to the wrapped optimizer's ``zero_grad()``."""
        self._optimizer.zero_grad(*args, **kwargs)


def _build_module(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRModelModule with build_model_from_config and build_criterion_from_config mocked.

    Examples:
        >>> module, model, criterion, postprocess = _build_module()
        >>> module.model is model and module.criterion is criterion and module.postprocess is postprocess
        True
    """
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_criterion = _fake_criterion()
    fake_postprocess = _fake_postprocess()
    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(fake_criterion, fake_postprocess),
        ),
    ):
        from rfdetr.training.module_model import RFDETRModelModule

        module = RFDETRModelModule(mc, tc)
    return module, fake_model, fake_criterion, fake_postprocess


def test_keypoint_training_resets_gaussian_parameters_after_pretrained_load(tmp_path) -> None:
    """Keypoint finetuning should reset pretrained Gaussian precision rows after loading weights."""
    mc = _base_model_config(
        pretrain_weights="/fake/keypoint.pth",
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
    )
    tc = _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_model.reset_keypoint_gaussian_parameters = MagicMock()
    events: list[str] = []

    with (
        patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
        patch("rfdetr.training.module_model.load_pretrain_weights") as mock_load_pretrain_weights,
        patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_fake_criterion(), _fake_postprocess()),
        ),
    ):
        mock_load_pretrain_weights.side_effect = lambda *_args, **_kwargs: events.append("load")
        fake_model.reset_keypoint_gaussian_parameters.side_effect = lambda: events.append("reset")

        from rfdetr.training.module_model import RFDETRModelModule

        RFDETRModelModule(mc, tc)

    mock_load_pretrain_weights.assert_called_once_with(fake_model, mc)
    fake_model.reset_keypoint_gaussian_parameters.assert_called_once_with()
    assert events == ["load", "reset"]


def _make_batch(batch_size=2, channels=3, h=16, w=16):
    """Build a (NestedTensor, targets) tuple for testing.

    Examples:
        >>> samples, targets = _make_batch(batch_size=2, h=8, w=8)
        >>> samples.tensors.shape, len(targets)
        (torch.Size([2, 3, 8, 8]), 2)
    """
    tensors = torch.randn(batch_size, channels, h, w)
    mask = torch.zeros(batch_size, h, w, dtype=torch.bool)
    samples = NestedTensor(tensors, mask)
    targets = [
        {
            "boxes": torch.tensor([[0.5, 0.5, 0.1, 0.1]]),
            "labels": torch.tensor([1]),
            "image_id": torch.tensor(i),
            "orig_size": torch.tensor([h, w]),
        }
        for i in range(batch_size)
    ]
    return samples, targets


class TestMultiScaleBatchStart:
    """on_train_batch_start multi-scale resize picks a step-deterministic scale without clobbering global RNG."""

    def _build_multi_scale_module(self, tmp_path, global_step):
        """Return a module configured for multi-scale with a stubbed trainer at the given global step."""
        tc = _base_train_config(tmp_path, multi_scale=True, do_random_resize_via_padding=False)
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(global_step=global_step)
        return module

    def test_scale_choice_is_deterministic_per_step(self, tmp_path):
        """The same global step must resize the batch to the same scale regardless of batch contents."""
        module = self._build_multi_scale_module(tmp_path, global_step=7)

        batch_a = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_a, 0)
        size_a = tuple(batch_a[0].tensors.shape[-2:])

        batch_b = _make_batch(h=64, w=64)
        module.on_train_batch_start(batch_b, 0)
        size_b = tuple(batch_b[0].tensors.shape[-2:])

        assert size_a == size_b

    def test_does_not_perturb_global_rng(self, tmp_path):
        """Scale selection must use a step-local generator and leave the process-global RNG untouched."""
        module = self._build_multi_scale_module(tmp_path, global_step=3)

        random.seed(42)
        expected = [random.random() for _ in range(3)]

        random.seed(42)
        module.on_train_batch_start(_make_batch(h=64, w=64), 0)
        actual = [random.random() for _ in range(3)]

        assert actual == expected


class _ScalarLossModel(nn.Module):
    """Tiny model exposing one scalar parameter for gradient-scaling tests."""

    def __init__(self) -> None:
        super().__init__()
        self.value = nn.Parameter(torch.zeros(()))

    def forward(self, samples, targets=None):
        return {"loss_scale": self.value}


class _BoxNormalizedCriterion:
    """Criterion with controllable per-target loss numerators and box counts."""

    weight_dict = {"loss_ce": 1.0}
    supports_loss_normalizer_override: bool = True

    def num_boxes_for_targets(self, outputs, targets):
        return torch.as_tensor(
            sum(int(target["labels"].numel()) for target in targets),
            dtype=torch.float32,
            device=outputs["loss_scale"].device,
        ).clamp(min=1.0)

    def __call__(self, outputs, targets, num_boxes=None):
        denominator = self.num_boxes_for_targets(outputs, targets) if num_boxes is None else num_boxes
        numerator = outputs["loss_scale"] * sum(target["loss_numerator"] for target in targets)
        return {"loss_ce": numerator / denominator}


# ---------------------------------------------------------------------------
# Fixtures — inject common test infrastructure; prefer these over private
# helpers in test methods.  Class-level _setup_* helpers still use the private
# functions directly (they cannot inject fixtures themselves).
# ---------------------------------------------------------------------------


@pytest.fixture
def build_module(tmp_path):
    """Factory fixture — returns (module, fake_model, fake_criterion, fake_postprocess).

    build_model and build_criterion_and_postprocessors are mocked automatically. tmp_path is injected automatically so
    test methods do not need to declare it.
    """
    return lambda model_config=None, train_config=None: _build_module(model_config, train_config, tmp_path)


@pytest.fixture
def make_batch():
    """Factory fixture — call with optional batch_size/channels/h/w."""
    return _make_batch


class TestInit:
    """Tests for RFDETRModelModule.__init__ — covers attribute assignment and delegation to build_model() /
    build_criterion_and_postprocessors() when pretrain_weights is None."""

    @pytest.mark.parametrize(
        "model_config,expected_manual",
        [
            pytest.param(_base_model_config(use_grouppose_keypoints=False), False, id="detection"),
            pytest.param(_base_model_config(segmentation_head=True), False, id="segmentation"),
            pytest.param(
                _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
                True,
                id="keypoints",
            ),
        ],
    )
    def test_optimization_mode_per_model_type(self, build_module, model_config, expected_manual):
        """Only keypoint models need manual optimization for box-normalized accumulation; detection and segmentation
        keep Lightning's automatic optimization path."""
        module, _, _, _ = build_module(model_config=model_config)

        assert module._use_manual_optimization is expected_manual
        assert module.automatic_optimization is (not expected_manual)

    def test_model_is_set(self, build_module):
        """__init__ must assign the built model to module.model."""
        module, fake_model, _, _ = build_module()
        assert module.model is fake_model

    def test_criterion_is_set(self, build_module):
        """__init__ must assign the built criterion to module.criterion."""
        module, _, fake_criterion, _ = build_module()
        assert module.criterion is fake_criterion

    def test_postprocess_is_set(self, build_module):
        """__init__ must assign the built postprocessor to module.postprocess."""
        module, _, _, fake_pp = build_module()
        assert module.postprocess is fake_pp

    def test_configs_stored(self, base_model_config, base_train_config, build_module):
        """Both model and train configs must be stored for later access."""
        mc = base_model_config()
        tc = base_train_config()
        module, _, _, _ = build_module(model_config=mc, train_config=tc)
        assert module.model_config is mc
        assert module.train_config is tc

    def test_compile_disabled_when_multi_scale_enabled(self, tmp_path):
        """torch.compile is skipped when multi_scale=True (dynamic shapes)."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=True)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("rfdetr.training.module_model.torch.compile") as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_not_called()

    def test_compile_runs_when_enabled_and_static_shapes(self, tmp_path):
        """torch.compile runs when compile=True and multi_scale=False on CUDA."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m) as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()

    @patch("rfdetr.training.module_model.torch.compile")
    @patch("rfdetr.config.DEVICE", "cuda")
    def test_compile_disabled_when_train_accelerator_is_cpu(self, _mock_compile: MagicMock, tmp_path):
        """Compile stays disabled when training is explicitly forced to CPU."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False, accelerator="cpu")
        _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        _mock_compile.assert_not_called()


class TestLoadPretrainWeights:
    """Tests for _load_pretrain_weights() — covers checkpoint validation, detection-head reinitialization on class-count
    mismatch, query-embedding trimming, re-download on corruption, and class-name extraction from checkpoint
    metadata."""

    def _make_checkpoint(self, num_classes_in_ckpt=91, num_queries=300, group_detr=13):
        """Build a fake checkpoint dict."""
        total_queries = num_queries * group_detr
        return {
            "model": {
                "class_embed.weight": torch.randn(num_classes_in_ckpt, 256),
                "class_embed.bias": torch.randn(num_classes_in_ckpt),
                "refpoint_embed.weight": torch.randn(total_queries, 4),
                "query_feat.weight": torch.randn(total_queries, 256),
                "other_layer.weight": torch.randn(10, 10),
            }
        }

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_loads_checkpoint_successfully(self, mock_validate, mock_torch_load, base_model_config, build_module):
        """A valid checkpoint must be validated, loaded, and applied to the model."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        mock_validate.assert_called_once_with("/fake/weights.pth", strict=False)
        module.model.load_state_dict.assert_called_once()

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_class_count_mismatch_triggers_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head is expanded to checkpoint size, then trimmed back to config size."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        # First call: expand to checkpoint size so load_state_dict shapes match.
        # Second call: trim back to configured num_classes + 1 (background class).
        from unittest.mock import call

        fake_model.reinitialize_detection_head.assert_has_calls([call(91), call(6)])
        assert fake_model.reinitialize_detection_head.call_count == 2

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_class_count_match_does_not_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head must NOT be reinitialized when class counts match."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=6)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        fake_model.reinitialize_detection_head.assert_not_called()

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_query_embedding_trimmed_to_configured_count(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Oversized query embeddings in checkpoint must be trimmed to match config."""
        mc = base_model_config(num_classes=90)
        module, _, _, _ = build_module(model_config=mc)

        num_queries = getattr(module.model_config, "num_queries", 300)
        group_detr = getattr(module.model_config, "group_detr", 13)
        desired = num_queries * group_detr

        large_total = desired + 500
        checkpoint = {
            "model": {
                "class_embed.weight": torch.randn(91, 256),
                "class_embed.bias": torch.randn(91),
                "refpoint_embed.weight": torch.randn(large_total, 4),
                "query_feat.weight": torch.randn(large_total, 256),
            }
        }
        mock_torch_load.return_value = checkpoint

        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        assert checkpoint["model"]["refpoint_embed.weight"].shape[0] == desired
        assert checkpoint["model"]["query_feat.weight"].shape[0] == desired

    @patch("rfdetr.models.weights.os.path.isfile", return_value=True)
    @patch("rfdetr.models.weights.download_pretrain_weights")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_redownloads_on_load_failure(
        self, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """A corrupted checkpoint must trigger re-download and a second load attempt."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})

        load_calls = [0]

        def fake_safe_load(*args, **kwargs):
            load_calls[0] += 1
            if load_calls[0] == 1:
                raise RuntimeError("corrupted file")
            return checkpoint

        # Patch at the definition site in util.io (_safe_torch_load is a deferred import in
        # weights.py so it is not a module-level name there). MD5 validation is intentionally
        # kept on the retry (validate_md5=False was removed in favour of rejecting
        # hash-mismatched files rather than silently accepting them).
        with patch("rfdetr.utilities.io._safe_torch_load", side_effect=fake_safe_load):
            load_pretrain_weights(module.model, module.model_config)

        redownload_calls = [c for c in mock_download.call_args_list if c.kwargs.get("redownload") is True]
        assert len(redownload_calls) >= 1
        assert load_calls[0] == 2

    @patch("rfdetr.models.weights.os.path.isfile", return_value=False)
    @patch("rfdetr.models.weights.download_pretrain_weights")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    @patch("rfdetr.models.weights.torch.load")
    def test_download_before_load_when_weights_absent(
        self, mock_torch_load, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """download_pretrain_weights must be called before torch.load so a fresh environment (e.g. Colab) downloads
        weights automatically.

        Regression test: previously download was only called as an except-block fallback, but ModelWeights.from_filename
        received the absolute path and returned None, causing a silent no-op and a FileNotFoundError.
        """
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/content/rf-detr-base.pth"})
        load_pretrain_weights(module.model, module.model_config)

        # download_pretrain_weights must have been called at least once before any load
        assert mock_download.call_count >= 1
        first_call = mock_download.call_args_list[0]
        assert first_call.args[0] == "/content/rf-detr-base.pth"

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_seg_checkpoint_into_detection_model_raises(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Loading a segmentation checkpoint into a detection model must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=True, patch_size=12)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False}
        )

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_detection_checkpoint_into_seg_model_raises(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Loading a detection checkpoint into a segmentation model must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=16)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": True}
        )

        with pytest.raises(ValueError, match="segmentation head"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_patch_size_mismatch_raises(self, mock_validate, mock_torch_load, base_model_config, build_module):
        """Loading a checkpoint with a different patch_size must raise ValueError."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=12)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False, "patch_size": 16}
        )

        with pytest.raises(ValueError, match="patch_size"):
            load_pretrain_weights(module.model, module.model_config)

    @patch("rfdetr.models.weights.torch.load")
    @patch("rfdetr.models.weights.validate_pretrain_weights")
    def test_compatible_checkpoint_does_not_raise(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """A checkpoint matching segmentation_head and patch_size must load without error."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=False, patch_size=14, class_names=[])
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", "segmentation_head": False, "patch_size": 14}
        )

        # Should not raise.
        load_pretrain_weights(module.model, module.model_config)


class TestApplyLora:
    """Tests for _apply_lora() — verifies that PEFT LoraConfig is constructed with the correct target modules and that
    the backbone encoder is replaced in-place with the wrapped PEFT model."""

    def _build_module_with_backbone(self, tmp_path):
        """Build module with a mock backbone that exposes backbone[0].encoder."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path)

        fake_model = MagicMock()
        fake_encoder = MagicMock()
        fake_backbone_0 = MagicMock()
        fake_backbone_0.encoder = fake_encoder
        fake_model.backbone = MagicMock()
        fake_model.backbone.__getitem__ = MagicMock(return_value=fake_backbone_0)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_fake_criterion(), _fake_postprocess()),
            ),
        ):
            from rfdetr.training.module_model import RFDETRModelModule

            module = RFDETRModelModule(mc, tc)

        return module, fake_model, fake_backbone_0, fake_encoder

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    def test_calls_lora_config_with_correct_target_modules(self, mock_lora_cfg_class, mock_get_peft, tmp_path):
        """LoRA must target the expected attention and token projection modules."""
        module, _, _, _ = self._build_module_with_backbone(tmp_path)
        mock_get_peft.return_value = MagicMock()

        apply_lora(module.model)

        mock_lora_cfg_class.assert_called_once()
        target_modules = mock_lora_cfg_class.call_args.kwargs.get("target_modules")
        expected = ["q_proj", "v_proj", "k_proj", "qkv", "query", "key", "value", "cls_token", "register_tokens"]
        assert target_modules == expected

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    def test_replaces_encoder_with_peft_model(self, mock_lora_cfg_class, mock_get_peft, tmp_path):
        """The backbone encoder must be replaced in-place with the PEFT-wrapped model."""
        module, _, fake_backbone_0, fake_encoder = self._build_module_with_backbone(tmp_path)
        peft_wrapped = MagicMock()
        mock_get_peft.return_value = peft_wrapped

        apply_lora(module.model)

        assert mock_get_peft.call_args[0][0] is fake_encoder
        assert fake_backbone_0.encoder is peft_wrapped


class TestOnFitStart:
    """Tests for on_fit_start() seeding behavior."""

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_at_rank_zero(self, mock_seed, base_train_config, build_module):
        """Rank 0: seed_everything(seed + 0) == seed_everything(seed)."""
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        with patch.object(type(module), "global_rank", new_callable=PropertyMock, return_value=0):
            module.on_fit_start()

        mock_seed.assert_called_once_with(7, workers=True)

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_rank_offset(self, mock_seed, base_train_config, build_module):
        """Non-zero rank: seed_everything(seed + global_rank) must be called.

        Validates the rank-offset contract — each worker seeds with a unique value to prevent correlated data
        augmentation across DDP processes.
        """
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        with patch.object(type(module), "global_rank", new_callable=PropertyMock, return_value=2):
            module.on_fit_start()

        mock_seed.assert_called_once_with(9, workers=True)  # 7 + 2

    @patch("rfdetr.training.module_model.seed_everything")
    def test_seed_skipped_when_none(self, mock_seed, base_train_config, build_module):
        """No seed means on_fit_start should not call seed_everything."""
        tc = base_train_config(seed=None)
        module, _, _, _ = build_module(train_config=tc)

        module.on_fit_start()

        mock_seed.assert_not_called()


class TestOnTrainBatchStart:
    """Tests for on_train_batch_start() — covers multi-scale interpolation of NestedTensor inputs and verifies
    regularization scheduling is delegated to DropPathCallback."""

    def _setup_module(
        self,
        tmp_path,
        multi_scale=False,
        do_random_resize_via_padding=False,
    ):
        tc = _base_train_config(
            tmp_path,
            multi_scale=multi_scale,
            do_random_resize_via_padding=do_random_resize_via_padding,
        )
        module, fake_model, _, _ = _build_module(train_config=tc)

        trainer = MagicMock()
        trainer.global_step = 0
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        return module, fake_model

    def test_drop_path_not_applied_in_module_hook(self, tmp_path):
        """Drop-path scheduling must be handled by DropPathCallback, not module hook."""
        module, fake_model = self._setup_module(tmp_path)
        module._trainer.global_step = 1

        module.on_train_batch_start(_make_batch(), batch_idx=1)

        fake_model.update_drop_path.assert_not_called()

    def test_dropout_not_applied_in_module_hook(self, tmp_path):
        """Dropout scheduling must be handled by DropPathCallback, not module hook."""
        module, fake_model = self._setup_module(tmp_path)
        module._trainer.global_step = 2

        module.on_train_batch_start(_make_batch(), batch_idx=2)

        fake_model.update_dropout.assert_not_called()

    @pytest.mark.parametrize(
        "method_name",
        [
            pytest.param("update_drop_path", id="drop-path"),
            pytest.param("update_dropout", id="dropout"),
        ],
    )
    def test_update_not_called_when_schedule_is_none(self, method_name, tmp_path):
        """Without a schedule, neither update_drop_path nor update_dropout must be called."""
        module, fake_model = self._setup_module(tmp_path)

        module.on_train_batch_start(_make_batch(), batch_idx=0)

        getattr(fake_model, method_name).assert_not_called()

    def test_multi_scale_resize_mutates_nested_tensor(self, tmp_path):
        """Multi-scale training must resize the input tensor to a square resolution."""
        module, _ = self._setup_module(tmp_path, multi_scale=True, do_random_resize_via_padding=False)
        module._trainer.global_step = 0
        samples, targets = _make_batch(batch_size=2, h=16, w=16)

        module.on_train_batch_start((samples, targets), batch_idx=0)

        new_h, new_w = samples.tensors.shape[2], samples.tensors.shape[3]
        assert new_h == new_w, "Multi-scale should produce square outputs"

    def test_multi_scale_skipped_when_random_resize_via_padding(self, tmp_path):
        """Padding-based resize takes precedence, so multi-scale must be a no-op."""
        module, _ = self._setup_module(tmp_path, multi_scale=True, do_random_resize_via_padding=True)
        samples, targets = _make_batch(batch_size=2, h=16, w=16)
        original_shape = samples.tensors.shape

        module.on_train_batch_start((samples, targets), batch_idx=0)

        assert samples.tensors.shape == original_shape


class TestTrainingStep:
    """Tests for training_step() — covers weighted loss aggregation, per-loss logging under the train/ prefix, prog_bar
    visibility, scalar tensor output, and that losses absent from weight_dict are excluded from the total."""

    def _run_step(
        self,
        tmp_path,
        loss_dict=None,
        weight_dict=None,
        accumulate_grad_batches=1,
        model_config=None,
        train_log_on_step=False,
        compact_train_metrics=True,
    ):
        module, fake_model, fake_criterion, _ = _build_module(
            model_config=model_config,
            train_config=_base_train_config(
                tmp_path,
                grad_accum_steps=accumulate_grad_batches,
                train_log_on_step=train_log_on_step,
                compact_train_metrics=compact_train_metrics,
            ),
            tmp_path=tmp_path,
        )
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        # Provide a real optimizer so param_groups carries a real "lr" key.
        real_param = nn.Parameter(torch.randn(4))
        real_optimizer = torch.optim.SGD([real_param], lr=1e-3)
        module.optimizers = MagicMock(return_value=real_optimizer)
        module.manual_backward = MagicMock()
        module.lr_schedulers = MagicMock(return_value=None)
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        trainer.gradient_clip_val = 0.0
        trainer.gradient_clip_algorithm = "norm"
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module, samples, targets, fake_model, fake_criterion

    def test_returns_weighted_loss_sum(self, tmp_path):
        """Total loss must equal the sum of each loss multiplied by its weight."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "loss_bbox": torch.tensor(2.0), "loss_giou": torch.tensor(3.0)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0 + 10.0 + 6.0)

    def test_loss_backward_uses_box_normalizer_contract(self, tmp_path):
        """Backward loss for keypoint models is scaled by the criterion box normalizer (manual optimization owns
        accumulation), not by Lightning's ``accumulate_grad_batches``."""
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        keypoint_config = _base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=4,
            model_config=keypoint_config,
        )
        module.criterion.num_boxes_for_targets.return_value = torch.tensor(4.0)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)
        backward_loss = module.manual_backward.call_args.args[0]
        assert backward_loss.item() == pytest.approx(1.0)

    def test_detection_loss_uses_lightning_grad_accum_scaling(self, tmp_path):
        """Detection (automatic optimization) divides loss by ``trainer.accumulate_grad_batches`` so the returned loss
        matches the legacy non-manual training path."""
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict,
            weight_dict,
            accumulate_grad_batches=1,
        )
        module._trainer.accumulate_grad_batches = 4

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)
        module.manual_backward.assert_not_called()

    def _make_keypoint_module(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a keypoint module wired with ``_ScalarLossModel`` and ``_BoxNormalizedCriterion`` for accum tests."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        model = _ScalarLossModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
        module.model = model
        module.criterion = _BoxNormalizedCriterion()
        module.postprocess = MagicMock()
        module.log = MagicMock()
        module.log_dict = MagicMock()
        module.optimizers = MagicMock(return_value=optimizer)
        module.manual_backward = lambda loss: loss.backward()
        module.lr_schedulers = MagicMock(return_value=None)
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = num_training_batches
        trainer.gradient_clip_val = 0.0
        trainer.gradient_clip_algorithm = "norm"
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module, model

    @pytest.mark.parametrize(
        "grad_accum_steps,box_counts,loss_numerators,expected_value",
        [
            pytest.param(1, (4,), (8.0,), -2.0, id="ga1-single-microbatch"),
            pytest.param(2, (2, 6), (10.0, 6.0), -2.0, id="ga2-balanced"),
            pytest.param(3, (2, 4, 6), (4.0, 8.0, 12.0), -2.0, id="ga3-balanced"),
            pytest.param(4, (1, 1, 1, 1), (2.0, 2.0, 2.0, 2.0), -2.0, id="ga4-uniform"),
            pytest.param(2, (1, 99), (1.0, 99.0), -1.0, id="ga2-skewed-1-vs-99"),
            pytest.param(4, (1, 1, 1, 97), (1.0, 1.0, 1.0, 97.0), -1.0, id="ga4-skewed-1-1-1-97"),
        ],
    )
    def test_box_normalized_accumulation_matches_large_effective_batch(
        self, tmp_path, grad_accum_steps, box_counts, loss_numerators, expected_value
    ):
        """Accumulated gradients across ``grad_accum_steps`` microbatches must equal a single large batch normalized by
        total boxes, regardless of how lopsided the per-microbatch box counts are."""
        large_module, large_model = self._make_keypoint_module(tmp_path, grad_accum_steps=1, num_training_batches=1)
        accum_module, accum_model = self._make_keypoint_module(
            tmp_path, grad_accum_steps=grad_accum_steps, num_training_batches=grad_accum_steps
        )

        microbatch_targets = [
            {
                "labels": torch.ones(box_count, dtype=torch.int64),
                "loss_numerator": torch.tensor(loss_numerator),
                "orig_size": torch.tensor([16, 16]),
            }
            for box_count, loss_numerator in zip(box_counts, loss_numerators, strict=True)
        ]
        samples, _ = _make_batch(batch_size=2)

        large_module.training_step((samples, microbatch_targets), batch_idx=0)
        for batch_idx, target in enumerate(microbatch_targets):
            accum_module.training_step((samples, [target]), batch_idx=batch_idx)

        torch.testing.assert_close(accum_model.value, large_model.value)
        assert large_model.value.item() == pytest.approx(expected_value)

    def test_logs_live_train_loss_to_progress_bar(self, tmp_path):
        """Aggregate training loss must be logged every step as a progress-only metric."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        progress_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "loss"]
        assert len(progress_loss_calls) == 1
        assert progress_loss_calls[0].kwargs.get("prog_bar") is True
        assert progress_loss_calls[0].kwargs.get("logger") is False
        assert progress_loss_calls[0].kwargs.get("on_step") is True
        assert progress_loss_calls[0].kwargs.get("on_epoch") is False

    @pytest.mark.parametrize(
        "train_log_on_step",
        [
            pytest.param(False, id="epoch-only"),
            pytest.param(True, id="step-and-epoch"),
        ],
    )
    def test_logs_epoch_train_loss_to_progress_bar(self, tmp_path, train_log_on_step):
        """Canonical training loss stays epoch-aggregated, with on_step mirroring train_log_on_step."""
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=train_log_on_step)

        module.training_step((samples, targets), batch_idx=0)

        epoch_loss_calls = [call for call in module.log.call_args_list if call[0][0] == "train/loss"]
        assert len(epoch_loss_calls) == 1
        assert epoch_loss_calls[0].kwargs.get("prog_bar") is True
        assert epoch_loss_calls[0].kwargs.get("on_step") is train_log_on_step
        assert epoch_loss_calls[0].kwargs.get("on_epoch") is True

    @pytest.mark.parametrize(
        "train_log_on_step,expected_live_loss_calls",
        [
            pytest.param(False, 1, id="default-emits-live-loss"),
            pytest.param(True, 0, id="on-step-skips-live-loss"),
        ],
    )
    def test_live_loss_key_gated_on_train_log_on_step(self, tmp_path, train_log_on_step, expected_live_loss_calls):
        """Redundant live ``loss`` progress key is emitted only when train_log_on_step is False.

        With train_log_on_step=True the ``train/loss`` call forks into a live ``train/loss_step`` progress entry, so the
        separate ``loss`` scalar becomes redundant and is skipped.
        """
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=train_log_on_step)

        module.training_step((samples, targets), batch_idx=0)

        live_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "loss"]
        assert len(live_loss_calls) == expected_live_loss_calls

    def test_training_step_does_not_log_learning_rate(self, tmp_path):
        """Learning-rate metrics must wait for the optimizer-step boundary."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        lr_calls = [c for c in module.log.call_args_list if c[0][0] == "train/lr"]
        assert not lr_calls

    def test_logs_learning_rate_range_for_multiple_param_groups(self, tmp_path):
        """An automatic optimizer step logs first, minimum, and maximum rates with distinct visibility flags."""
        module, samples, targets, _, _ = self._run_step(tmp_path)
        first_param = nn.Parameter(torch.randn(4))
        second_param = nn.Parameter(torch.randn(4))
        third_param = nn.Parameter(torch.randn(4))
        module.optimizers.return_value = torch.optim.SGD(
            [
                {"params": [first_param], "lr": 0.1},
                {"params": [second_param], "lr": 0.05},
                {"params": [third_param], "lr": 0.2},
            ],
            lr=0.1,
        )

        module.on_before_optimizer_step(module.optimizers.return_value)

        expected_metrics = {
            "train/lr": (0.1, True),
            "train/lr_min": (0.05, False),
            "train/lr_max": (0.2, False),
        }
        for metric_name, (expected_value, expected_prog_bar) in expected_metrics.items():
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == 1
            assert metric_calls[0].args[1] == pytest.approx(expected_value)
            assert metric_calls[0].kwargs.get("prog_bar") is expected_prog_bar
            assert metric_calls[0].kwargs.get("on_step") is True
            assert metric_calls[0].kwargs.get("on_epoch") is False

    def test_compacts_auxiliary_loss_metrics(self, tmp_path):
        """Layer-suffixed loss metrics should be reduced to one aggregate per base loss term."""
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_bbox": torch.tensor(2.0),
            "loss_ce_0": torch.tensor(3.0),
            "loss_bbox_0": torch.tensor(4.0),
            "loss_ce_enc": torch.tensor(5.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={key: 1.0 for key in loss_dict},
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {"train/loss_ce", "train/loss_bbox", "train/loss_ce_aux", "train/loss_bbox_aux"}
        assert logged["train/loss_ce_aux"].item() == pytest.approx(8.0)
        assert logged["train/loss_bbox_aux"].item() == pytest.approx(4.0)

    def test_excludes_non_weighted_layer_suffixed_keys_from_aggregation(self, tmp_path):
        """A digit/enc-suffixed key absent from weight_dict must be logged individually, not aggregated.

        ``cardinality_error_0`` is a diagnostic count, never a weighted loss term, so weight_dict never contains it even
        though its name matches the layer-suffix pattern used for real auxiliary terms.
        """
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_ce_0": torch.tensor(2.0),
            "cardinality_error": torch.tensor(3.0),
            "cardinality_error_0": torch.tensor(4.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={"loss_ce": 1.0, "loss_ce_0": 1.0},
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {
            "train/loss_ce",
            "train/loss_ce_aux",
            "train/cardinality_error",
            "train/cardinality_error_0",
        }
        assert logged["train/cardinality_error"].item() == pytest.approx(3.0)
        assert logged["train/cardinality_error_0"].item() == pytest.approx(4.0)

    def test_logs_layer_suffixed_keys_individually_when_compact_metrics_disabled(self, tmp_path):
        """With compact_train_metrics=False, every per-layer key is logged individually, honoring on_step."""
        loss_dict = {
            "loss_ce": torch.tensor(1.0),
            "loss_ce_0": torch.tensor(2.0),
            "loss_ce_enc": torch.tensor(3.0),
        }
        module, samples, targets, _, _ = self._run_step(
            tmp_path,
            loss_dict=loss_dict,
            weight_dict={key: 1.0 for key in loss_dict},
            train_log_on_step=True,
            compact_train_metrics=False,
        )

        module.training_step((samples, targets), batch_idx=0)

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == {"train/loss_ce", "train/loss_ce_0", "train/loss_ce_enc"}
        assert module.log_dict.call_args.kwargs.get("on_step") is True

    def test_compacts_real_criterion_output_gated_by_weight_dict(self, tmp_path):
        """Aggregation of a real SetCriterion's key inventory must fold only weighted per-layer losses.

        ``test_compacts_auxiliary_loss_metrics`` above only exercises synthetic ``loss_ce``/``loss_bbox`` keys, whose
        weight_dict membership and layer-suffix pattern happen to coincide. RF-DETR's real criterion also emits
        ``cardinality_error`` with the same ``_<i>``/``_enc`` suffix pattern (see ``SetCriterion.forward``) despite
        never appearing in ``weight_dict`` (it is a logging-only diagnostic, not a backpropagated loss term). Suffix-
        only aggregation folds it into a spurious ``cardinality_error_aux`` key; correct aggregation must gate on
        ``weight_dict`` membership and leave it out. This uses the real ``RFDETRSmall`` model/criterion (this repo's
        default size) with a tiny synthetic batch so the assertion is checked against RF-DETR's actual key inventory
        instead of a hand-picked synthetic one.
        """
        mc = RFDETRSmallConfig(pretrain_weights=None, num_classes=3, device="cpu")
        tc = _base_train_config(tmp_path)
        real_model = build_model_from_config(mc, tc)
        real_criterion, real_postprocess = build_criterion_from_config(mc, tc)
        real_model.train()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=real_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(real_criterion, real_postprocess),
            ),
        ):
            module = RFDETRModelModule(mc, tc)

        samples, targets = _make_batch(batch_size=1, h=mc.resolution, w=mc.resolution)
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        # Spy on the criterion to capture its raw (pre-aggregation) per-key loss dict, so the expected
        # aggregate is derived from the same real forward pass training_step() uses internally rather
        # than a second, independently-random forward pass.
        raw_loss_dicts: list[dict[str, torch.Tensor]] = []
        original_forward = module.criterion.forward

        def _spy_forward(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            raw_loss_dicts.append(result)
            return result

        module.criterion.forward = _spy_forward

        module.training_step((samples, targets), batch_idx=0)

        raw_loss_dict = raw_loss_dicts[0]
        weight_dict = module.criterion.weight_dict
        expected_aux_sums: dict[str, torch.Tensor] = {}
        expected_base_keys: set[str] = set()
        for key, value in raw_loss_dict.items():
            base_name, separator, suffix = key.rpartition("_")
            is_layer_suffixed = bool(separator) and (suffix.isdigit() or suffix == "enc")
            if is_layer_suffixed and key in weight_dict:
                aggregate_name = f"train/{base_name}_aux"
                expected_aux_sums[aggregate_name] = expected_aux_sums.get(aggregate_name, torch.zeros(())) + value
            else:
                # Not layer-suffixed, or layer-suffixed but absent from weight_dict (e.g.
                # cardinality_error_0/_1/_enc — a diagnostic count, never a weighted loss term):
                # both cases fall through to individual logging, matching production's else branch.
                expected_base_keys.add(f"train/{key}")

        logged = module.log_dict.call_args.args[0]
        assert set(logged) == expected_base_keys | set(expected_aux_sums)
        assert "train/cardinality_error_aux" not in logged
        for aggregate_name, expected_value in expected_aux_sums.items():
            assert logged[aggregate_name].item() == pytest.approx(expected_value.item())

    def test_logs_loss_components_on_epoch_when_step_logging_is_enabled(self, tmp_path):
        """Per-step total loss logging must not re-enable per-step component metrics."""
        module, samples, targets, _, _ = self._run_step(tmp_path, train_log_on_step=True)

        module.training_step((samples, targets), batch_idx=0)

        assert module.log_dict.call_args.kwargs.get("on_step") is False
        assert module.log_dict.call_args.kwargs.get("on_epoch") is True

    def test_logs_learning_rates_for_manual_optimizer_steps_including_tail(self, tmp_path):
        """Manual accumulation logs rates once per completed or partial optimizer window, never twice.

        ``_step_optimizer`` must not log learning rates itself — the single emission site is the
        ``on_before_optimizer_step`` hook, which ``_StepHookOptimizer`` fires from ``step()`` the same way Lightning's
        real ``LightningOptimizer.step()`` does on both automatic and manual paths.
        """
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=2),
            tmp_path=tmp_path,
        )
        parameter = nn.Parameter(torch.randn(4))
        optimizer = _StepHookOptimizer(module, torch.optim.SGD([parameter], lr=0.1))
        trainer = MagicMock(num_training_batches=3, gradient_clip_val=0.0, gradient_clip_algorithm="norm")
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        module.log = MagicMock()

        for batch_idx in range(3):
            if module._should_step_optimizer(batch_idx):
                module._step_optimizer(optimizer)

        for metric_name in ("train/lr", "train/lr_min", "train/lr_max"):
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == 2

    def test_logs_learning_rates_at_grad_accum_steps_one(self, tmp_path):
        """With no accumulation (``grad_accum_steps=1``), every batch closes its own window and logs rates.

        ``grad_accum_steps=1`` is this repo's own recommended setting for multi-GPU keypoint training (see
        docs/learn/train/advanced.md's "Prefer grad_accum_steps=1 on multi-GPU for keypoints" note) — the
        ``grad_accum_steps=2`` case above never exercises the no-accumulation path where the modulo check in
        ``_should_step_optimizer`` is trivially true for every batch and the end-of-epoch tail fallback never engages
        (there is never a partial window to flush).
        """
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=1),
            tmp_path=tmp_path,
        )
        parameter = nn.Parameter(torch.randn(4))
        optimizer = _StepHookOptimizer(module, torch.optim.SGD([parameter], lr=0.1))
        trainer = MagicMock(num_training_batches=3, gradient_clip_val=0.0, gradient_clip_algorithm="norm")
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        module.log = MagicMock()

        for batch_idx in range(3):
            if module._should_step_optimizer(batch_idx):
                module._step_optimizer(optimizer)

        for metric_name in ("train/lr", "train/lr_min", "train/lr_max"):
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == 3

    def test_logs_learning_rates_for_infinite_dataset_skips_tail_fallback(self, tmp_path):
        """On an infinite/streaming dataset only modulo-boundary batches log rates; no spurious tail fallback fires.

        ``test_logs_learning_rates_for_manual_optimizer_steps_including_tail`` above only covers a finite
        ``num_training_batches=3``, where the third batch triggers ``_should_step_optimizer``'s end-of-epoch tail
        fallback and logs a second time. ``TestShouldStepOptimizer.test_infinite_dataset_uses_modulo_only`` covers
        the boolean return of ``_should_step_optimizer`` in isolation for ``num_training_batches=float("inf")``, but
        not that the LR-logging path downstream reflects it: with the same ``grad_accum_steps=2`` and 3 batches, an
        infinite dataset must log exactly once (only the modulo-boundary step at batch_idx=1), not twice, because
        ``batch_idx + 1 >= num_training_batches`` can never hold against infinity.
        """
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=2),
            tmp_path=tmp_path,
        )
        parameter = nn.Parameter(torch.randn(4))
        optimizer = _StepHookOptimizer(module, torch.optim.SGD([parameter], lr=0.1))
        trainer = MagicMock(num_training_batches=float("inf"), gradient_clip_val=0.0, gradient_clip_algorithm="norm")
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        module.log = MagicMock()

        for batch_idx in range(3):
            if module._should_step_optimizer(batch_idx):
                module._step_optimizer(optimizer)

        for metric_name in ("train/lr", "train/lr_min", "train/lr_max"):
            metric_calls = [call for call in module.log.call_args_list if call.args[0] == metric_name]
            assert len(metric_calls) == 1

    def test_logs_convergence_components_to_progress_bar(self, tmp_path):
        """Selected detection and keypoint losses should appear as compact progress-only metrics."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_nll": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        progress_names = {c[0][0] for c in module.log.call_args_list if c.kwargs.get("prog_bar") is True}
        assert {"loss_cls", "loss_box", "kp_l1", "kp_nll"}.issubset(progress_names)

    def test_logs_individual_main_losses_as_dict(self, tmp_path):
        """Each main decoder loss must be logged separately under the train/ prefix."""
        loss_dict = {"loss_ce": torch.tensor(0.5), "loss_bbox": torch.tensor(0.3)}
        weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        module.training_step((samples, targets), batch_idx=0)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args[0][0]
        assert "train/loss_ce" in logged
        assert "train/loss_bbox" in logged

    def test_returns_scalar_tensor(self, tmp_path):
        """Loss must be a 0-dim tensor so Lightning can call .backward() on it."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.dim() == 0

    def test_returns_detached_predictions_when_train_metrics_enabled(self, tmp_path):
        """compute_train_metrics=True should expose detached predictions without changing the Lightning loss key."""
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, fake_postprocess = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        model_output = {"pred_logits": torch.randn(2, 3, requires_grad=True)}
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        fake_postprocess.return_value = [{"boxes": torch.randn(1, 4, requires_grad=True)}]
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        result = module.training_step((samples, targets), batch_idx=0)

        assert isinstance(result, dict)
        assert result["loss"].dim() == 0
        assert result["results"][0]["boxes"].requires_grad is False
        assert result["targets"] is targets

    def test_ignores_losses_not_in_weight_dict(self, tmp_path):
        """Losses absent from weight_dict (e.g. cardinality_error) must not affect total."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "cardinality_error": torch.tensor(99.0)}
        weight_dict = {"loss_ce": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(2.0)

    def test_train_metrics_slices_to_group0_queries(self, tmp_path):
        """compute_train_metrics postprocess must receive only group-0 queries ([:num_queries]).

        Group DETR emits group_detr×num_queries outputs in train mode. Without the slice, postprocess top-k draws from
        all groups and OKS/mAP reads ~50× below true accuracy. Assert the received pred_logits has shape (B,
        num_queries, C).
        """
        nq = 10
        group_detr = 3
        batch_size = 2
        num_classes = 5
        mc = _base_model_config(num_classes=num_classes, num_queries=nq)
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        full_logits = torch.randn(batch_size, group_detr * nq, num_classes)
        model_output = {
            "pred_logits": full_logits,
            "pred_boxes": torch.randn(batch_size, group_detr * nq, 4),
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [
                {"boxes": torch.zeros(nq, 4), "scores": torch.ones(nq), "labels": torch.zeros(nq, dtype=torch.long)}
            ]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        samples, targets = _make_batch(batch_size=batch_size)

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_logits" in received
        assert received["pred_logits"].shape == (batch_size, nq, num_classes)
        torch.testing.assert_close(received["pred_logits"], full_logits[:, :nq])

    def test_train_metrics_skips_dict_pred_masks(self, tmp_path):
        """Dict-valued pred_masks (sparse_forward in train mode) must not crash training_step.

        In segmentation train mode lwdetr uses sparse_forward which returns pred_masks as a dict. PostProcess cannot
        handle a dict — it calls .shape[0] on it. The fix filters out non-tensor values so postprocess receives
        pred_masks=None (box path).
        """
        tc = _base_train_config(tmp_path, compute_train_metrics=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)

        model_output = {
            "pred_logits": torch.randn(2, 10, 5),
            "pred_boxes": torch.randn(2, 10, 4),
            "pred_masks": {"spatial_features": torch.randn(2, 256, 8, 8), "query_features": torch.randn(2, 10, 256)},
        }
        fake_model.return_value = model_output
        fake_criterion.return_value = {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        received: dict = {}

        def capture_postprocess(outputs, orig_sizes):
            received.update(outputs)
            return [{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1, dtype=torch.long)}]

        module.postprocess = capture_postprocess
        module.log = MagicMock()
        module.log_dict = MagicMock()
        real_param = nn.Parameter(torch.randn(4))
        module.optimizers = MagicMock(return_value=torch.optim.SGD([real_param], lr=1e-3))
        trainer = MagicMock()
        trainer.accumulate_grad_batches = 1
        trainer.num_training_batches = 1
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        samples, targets = _make_batch()

        module.training_step((samples, targets), batch_idx=0)

        assert "pred_masks" not in received


class TestOnBeforeOptimizerStepFiresOncePerAccumulationWindow:
    """``on_before_optimizer_step`` must fire once per optimizer update on the automatic-optimization path.

    ``TestTrainingStep.test_logs_learning_rate_range_for_multiple_param_groups`` above calls
    ``on_before_optimizer_step`` directly with a hand-built optimizer, which only verifies the payload it logs — it
    never drives Lightning's own automatic-optimization gradient-accumulation loop, so it cannot catch a regression
    where the hook fired once per microbatch instead of once per completed accumulation window. Detection/segmentation
    models use PTL's automatic optimization (unlike keypoint models' manual ``_should_step_optimizer`` path tested
    elsewhere in this file), so accumulation gating here is entirely internal to Lightning; the only way to exercise
    it faithfully is a real ``Trainer.fit()`` run with ``accumulate_grad_batches > 1``.
    """

    def test_fires_once_per_window_not_once_per_microbatch(self, tmp_path):
        """4 training batches at accumulate_grad_batches=2 must trigger exactly 2 optimizer-step hook calls."""
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, num_workers=0)

        tiny_model = _TinyModel()
        fake_criterion = _FakeCriterion()
        fake_postprocess = MagicMock(side_effect=_helpers_fake_postprocess)
        fake_dataset = _FakeDataset(length=20)

        class _CountOptimizerSteps(Callback):
            """Count how many times Lightning invokes the pre-optimizer-step hook."""

            def __init__(self) -> None:
                self.count = 0

            def on_before_optimizer_step(self, trainer, pl_module, optimizer) -> None:
                """Increment the call count for every hook invocation."""
                self.count += 1

        counter = _CountOptimizerSteps()

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=tiny_model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(fake_criterion, fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=fake_dataset),
            patch(
                "rfdetr.training.module_model.get_param_dict",
                side_effect=lambda args, model: _make_param_dicts(model),
            ),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = Trainer(
                fast_dev_run=4,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                accumulate_grad_batches=2,
                callbacks=[counter],
            )
            trainer.fit(module, datamodule)

        assert counter.count == 2


class TestShouldStepOptimizer:
    """Tests for ``_should_step_optimizer`` — covers the modulo path, the end-of-epoch fallback, and the iterable /
    infinite dataset case where ``trainer.num_training_batches`` is ``float('inf')``."""

    def _make_module_with_trainer(self, tmp_path, grad_accum_steps, num_training_batches):
        """Build a module with a stub trainer exposing ``num_training_batches`` for the test scenario."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=_base_train_config(tmp_path, grad_accum_steps=grad_accum_steps),
            tmp_path=tmp_path,
        )
        trainer = MagicMock()
        trainer.num_training_batches = num_training_batches
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module

    @pytest.mark.parametrize(
        "grad_accum_steps,num_training_batches,batch_idx,expected",
        [
            pytest.param(1, 10, 0, True, id="ga1-bidx0-steps-every-batch"),
            pytest.param(1, 10, 9, True, id="ga1-bidx9-steps-every-batch"),
            pytest.param(2, 10, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 10, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(2, 10, 2, False, id="ga2-bidx2-opens-new-window"),
            pytest.param(2, 10, 9, True, id="ga2-bidx9-closes-final-window"),
            pytest.param(4, 10, 7, True, id="ga4-bidx7-closes-second-window"),
            pytest.param(4, 10, 8, False, id="ga4-bidx8-opens-partial-window"),
            pytest.param(4, 10, 9, True, id="ga4-bidx9-final-batch-flushes-partial"),
            pytest.param(4, 11, 8, False, id="ga4-bidx8-of-11-mid-window"),
            pytest.param(4, 11, 10, True, id="ga4-bidx10-final-batch-flushes-partial"),
        ],
    )
    def test_finite_dataset_steps_at_window_close_and_epoch_end(
        self, tmp_path, grad_accum_steps, num_training_batches, batch_idx, expected
    ):
        """Optimizer steps when the accumulation window closes or when the epoch ends with a partial window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, num_training_batches)

        assert module._should_step_optimizer(batch_idx) is expected

    @pytest.mark.parametrize(
        "grad_accum_steps,batch_idx,expected",
        [
            pytest.param(2, 0, False, id="ga2-bidx0-mid-window"),
            pytest.param(2, 1, True, id="ga2-bidx1-closes-window"),
            pytest.param(4, 2, False, id="ga4-bidx2-mid-window"),
            pytest.param(4, 3, True, id="ga4-bidx3-closes-window"),
        ],
    )
    def test_infinite_dataset_uses_modulo_only(self, tmp_path, grad_accum_steps, batch_idx, expected):
        """Iterable datasets report ``num_training_batches=float('inf')``; only the modulo path can close the window."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps, float("inf"))

        assert module._should_step_optimizer(batch_idx) is expected

    def test_none_num_training_batches_uses_modulo_only(self, tmp_path):
        """If trainer.num_training_batches is None (very early in fit), only the modulo path can trigger a step."""
        module = self._make_module_with_trainer(tmp_path, grad_accum_steps=2, num_training_batches=None)

        assert module._should_step_optimizer(batch_idx=0) is False
        assert module._should_step_optimizer(batch_idx=1) is True


class TestOnTrainEpochStart:
    """Tests for ``on_train_epoch_start`` — must reset the accumulated box normalizer between epochs."""

    def test_reset_clears_stale_accumulator(self, tmp_path):
        """A stale normalizer from a previous epoch must not leak into the new epoch's first microbatch."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        module._accumulated_box_normalizer = torch.tensor(42.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_is_noop_for_detection_module(self, tmp_path):
        """Detection models never populate _accumulated_box_normalizer; reset must leave it None."""
        module, *_ = _build_module(tmp_path=tmp_path)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None

    def test_zeros_optimizer_grad_on_stale_accumulator(self, tmp_path):
        """When a partial window survived epoch end, optimizer gradients must be zeroed before reset."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        real_param = nn.Parameter(torch.randn(4))
        real_param.grad = torch.ones(4)
        optimizer = torch.optim.SGD([real_param], lr=1.0)
        module.optimizers = MagicMock(return_value=optimizer)
        module._accumulated_box_normalizer = torch.tensor(7.0)

        module.on_train_epoch_start()

        assert module._accumulated_box_normalizer is None
        assert real_param.grad is None or real_param.grad.abs().sum().item() == pytest.approx(0.0)


class TestRescaleAccumulatedGradients:
    """Direct contract tests for _rescale_accumulated_gradients."""

    def test_scales_all_parameter_grads_by_factor(self, tmp_path):
        """Calling _rescale with factor 0.5 must halve every parameter's .grad tensor."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(3, 5)
        # weight: [5, 3], bias: [5]
        nano_model.weight.grad = torch.full((5, 3), 4.0)
        nano_model.bias.grad = torch.full((5,), 8.0)
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(0.5))

        torch.testing.assert_close(nano_model.weight.grad, torch.full((5, 3), 2.0))
        torch.testing.assert_close(nano_model.bias.grad, torch.full((5,), 4.0))

    def test_scale_one_leaves_grads_unchanged(self, tmp_path):
        """Scale factor 1.0 must leave gradients exactly unchanged (identity)."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(2, 2)
        nano_model.weight.grad = torch.full((2, 2), 3.0)
        nano_model.bias.grad = torch.full((2,), 7.0)
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(1.0))

        torch.testing.assert_close(nano_model.weight.grad, torch.full((2, 2), 3.0))
        torch.testing.assert_close(nano_model.bias.grad, torch.full((2,), 7.0))

    def test_skips_params_with_no_grad(self, tmp_path):
        """Parameters without .grad must remain None after rescaling."""
        module, *_ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            tmp_path=tmp_path,
        )
        nano_model = nn.Linear(2, 2)
        # No backward pass — all grads are None
        module.model = nano_model

        module._rescale_accumulated_gradients(torch.tensor(0.5))

        assert nano_model.weight.grad is None
        assert nano_model.bias.grad is None


class TestValidationStep:
    """Tests for validation_step() — verifies output dict shape, postprocessor invocation with correct original sizes,
    and val/loss logging."""

    def _run_val_step(
        self,
        tmp_path,
        loss_dict: dict[str, torch.Tensor] | None = None,
        weight_dict: dict[str, float] | None = None,
    ):
        tc = _base_train_config(tmp_path, compute_val_loss=True)
        module, fake_model, fake_criterion, fake_pp = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        result = module.validation_step((samples, targets), batch_idx=0)
        return result, fake_pp, module

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("results", id="results-key"),
            pytest.param("targets", id="targets-key"),
        ],
    )
    def test_returns_dict_with_required_key(self, key, tmp_path):
        """Output dict must contain both 'results' and 'targets' for downstream metric computation."""
        result, _, _ = self._run_val_step(tmp_path)
        assert key in result

    def test_postprocess_called_with_orig_sizes(self, tmp_path):
        """Postprocessor must receive original image sizes to rescale predictions."""
        _result, fake_pp, _ = self._run_val_step(tmp_path)
        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (2, 2)

    def test_logs_val_loss(self, tmp_path):
        """Validation loss must be logged for monitoring and early stopping."""
        _, _, module = self._run_val_step(tmp_path)
        val_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "val/loss"]
        assert len(val_loss_calls) == 1

    def test_logs_val_keypoint_loss_components_once(self, tmp_path):
        """Validation should expose full keypoint losses without duplicate progress aliases."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_keypoints_l1": torch.tensor(0.4),
            "loss_keypoints_findable": torch.tensor(0.3),
            "loss_keypoints_visible": torch.tensor(0.2),
            "loss_keypoints_nll": torch.tensor(0.1),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        module.log_dict.assert_called_once()
        logged = module.log_dict.call_args.args[0]
        assert "val/loss_keypoints_l1" in logged
        assert "val/loss_keypoints_findable" in logged
        logged_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/kp_l1" not in logged_names
        assert "val/kp_find" not in logged_names
        assert "val/kp_vis" not in logged_names
        assert "val/kp_nll" not in logged_names

    def test_val_detection_loss_components_are_not_relogged_as_progress_aliases(self, tmp_path):
        """Validation component losses should be logged once under canonical ``val/loss_*`` names."""
        loss_dict = {
            "loss_ce": torch.tensor(0.5),
            "loss_bbox": torch.tensor(0.3),
            "loss_giou": torch.tensor(0.2),
        }
        weight_dict = {key: 1.0 for key in loss_dict}

        _, _, module = self._run_val_step(tmp_path, loss_dict=loss_dict, weight_dict=weight_dict)

        logged_loss_names = set(module.log_dict.call_args.args[0])
        direct_log_names = {c[0][0] for c in module.log.call_args_list}
        assert "val/loss_giou" in logged_loss_names
        assert "val/loss_giou" not in direct_log_names
        assert "val/giou" not in direct_log_names

    def test_can_disable_val_loss_computation(self, tmp_path):
        """compute_val_loss=False skips criterion call and val/loss logging."""
        tc = _base_train_config(tmp_path, compute_val_loss=False)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "val/loss" not in logged_keys
        assert "results" in result and "targets" in result

    def test_auto_val_loss_skips_criterion_without_a_loss_monitor(self, tmp_path):
        """compute_val_loss='auto' skips validation loss when no configured consumer monitors it."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        assert "results" in result and "targets" in result

    def test_auto_val_loss_keeps_criterion_for_callback_monitor(self, tmp_path):
        """compute_val_loss='auto' retains validation loss for a callback monitoring val/loss."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.trainer = SimpleNamespace(callbacks=[SimpleNamespace(monitor="val/loss")])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_criterion.assert_called_once_with({}, targets)
        assert any(call.args[0] == "val/loss" for call in module.log.call_args_list)

    def test_auto_val_loss_keeps_criterion_for_rfdetr_early_stopping(self, tmp_path):
        """compute_val_loss='auto' retains validation loss for a real RFDETREarlyStopping monitoring val/loss.

        ``RFDETREarlyStopping.monitor`` is always the synthetic ``__rfdetr_effective_map__`` key it injects itself, so
        the callback's real target only ever appears in ``_monitor_regular``. Stub callbacks carrying a plain
        ``monitor="val/loss"`` attribute exercise the generic half of the scan and would keep passing even if the half
        covering RF-DETR's own callbacks regressed.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[RFDETREarlyStopping(monitor_regular="val/loss")])

        assert module._should_compute_val_loss is True

    def test_auto_val_loss_detects_ema_monitor_attribute(self, tmp_path):
        """compute_val_loss='auto' retains validation loss for a callback consuming val/loss as its EMA monitor.

        RF-DETR's ``BestModelCallback`` / ``RFDETREarlyStopping`` keep their EMA-track metric key in ``_monitor_ema``
        rather than in the PTL-native ``monitor`` attribute, so a scan that inspects only ``monitor`` (and
        ``_monitor_regular``) would silently skip the loss those callbacks still read.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(
            callbacks=[SimpleNamespace(monitor="__rfdetr_effective_map__", _monitor_ema="val/loss")]
        )

        assert module._should_compute_val_loss is True

    def test_auto_val_loss_skips_criterion_for_empty_callback_list(self, tmp_path):
        """compute_val_loss='auto' resolves to skipping the loss when the attached trainer carries no callbacks.

        ``any()`` over an empty callback list is False, which is the intended answer, but nothing in the scan states
        it: an attached trainer with an empty ``callbacks`` list must resolve exactly like the unattached case rather
        than raising or falling back to computing the loss.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[])

        assert module._should_compute_val_loss is False

    def test_explicit_val_loss_disable_rejects_callback_monitor(self, tmp_path):
        """compute_val_loss=False rejects a callback that would consume val/loss."""
        tc = _base_train_config(tmp_path, compute_val_loss=False)
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[SimpleNamespace(monitor="val/loss")])

        with pytest.raises(ValueError, match="compute_val_loss=False is incompatible"):
            module.on_fit_start()

    def test_standalone_validate_run_skips_the_fit_start_rejection(self, tmp_path):
        """A standalone trainer.validate() with compute_val_loss=False runs despite a callback monitoring val/loss.

        The rejection lives in ``on_fit_start``, which PTL invokes only when ``trainer.state.fn`` is ``FITTING``; a bare
        ``validate()`` call sets it to ``VALIDATING`` and skips the hook entirely. The asymmetry is intentional (a
        validation-only run has no scheduler or early-stopping loop to starve), so this test documents the current
        behaviour instead of asserting a raise.
        """
        mc = _base_model_config()
        tc = _base_train_config(tmp_path, compute_val_loss=False, num_workers=0)

        class _ValLossMonitorCallback(Callback):
            """Callback declaring a val/loss monitor, as PTL-native checkpoint and early-stopping callbacks do."""

            monitor = "val/loss"

        fake_postprocess = MagicMock(side_effect=_helpers_fake_postprocess)

        with (
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_TinyModel()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(_FakeCriterion(), fake_postprocess),
            ),
            patch("rfdetr.training.module_data.build_dataset", return_value=_FakeDataset(length=4)),
        ):
            module = RFDETRModelModule(mc, tc)
            datamodule = RFDETRDataModule(mc, tc)
            trainer = Trainer(
                limit_val_batches=1,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                enable_checkpointing=False,
                logger=False,
                callbacks=[_ValLossMonitorCallback()],
            )
            trainer.validate(module, datamodule)

        # The validation batch reached postprocess, so the loop ran to completion instead of raising the guard.
        fake_postprocess.assert_called_once()

    def test_forwards_through_ema_model_not_base_by_default(self, tmp_path):
        """Validation must forward through the EMA-averaged model, not the base model, by default.

        Regression for #416: this single forward replaces the duplicate base+EMA pair COCOEvalCallback would otherwise
        run, and is the ~3-3.5%-of-epoch saving behind the eval_base_model default.
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        ema_model = MagicMock(name="ema_model", return_value={"ema": True})
        fake_ema_callback = SimpleNamespace(_average_model=SimpleNamespace(module=SimpleNamespace(model=ema_model)))
        module.trainer = SimpleNamespace(callbacks=[fake_ema_callback])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        ema_model.assert_called_once()
        fake_model.assert_not_called()

    def test_falls_back_to_base_model_when_ema_not_warmed_up(self, tmp_path):
        """Validation must fall back to the base model if the EMA callback has no averaged model yet.

        The very first validation can run before RFDETREMACallback has built its averaged model; forwarding through a
        missing EMA model would crash instead of degrading to the base weights.
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}

        fake_ema_callback = SimpleNamespace(_average_model=None)
        module.trainer = SimpleNamespace(callbacks=[fake_ema_callback])
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once()

    def test_eval_base_model_does_not_touch_trainer(self, tmp_path):
        """eval_base_model=True must not access self.trainer at all — it returns the base model unconditionally.

        validation_step has to stay usable on a module with no Trainer attached, as every other test in this class
        relies on; the opt-in path must short-circuit before any trainer lookup.
        """
        tc = _base_train_config(tmp_path, eval_base_model=True)
        module, fake_model, _, _ = _build_module(train_config=tc, tmp_path=tmp_path)

        assert module._resolve_eval_model() is fake_model

    def test_falls_back_to_base_model_when_trainer_unattached(self, tmp_path):
        """Validation must fall back to the base model — not raise — when the module isn't attached to a Trainer at all.

        ``LightningModule.trainer`` raises ``RuntimeError`` (not ``None``) when unattached, so a naive
        ``getattr(self.trainer, ...)`` would crash instead of falling back (regression: module never has a Trainer wired
        up in this test module, matching the direct/standalone validation_step-call scenario).
        """
        tc = _base_train_config(tmp_path, use_ema=True)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {"base": True}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()

        module.validation_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once()


class TestValidationLossSkipNotice:
    """on_validation_epoch_start() announces an 'auto' policy that resolved to skipping validation-loss computation.

    The consumer scan cannot see a human reading a ``val/loss`` curve from ``metrics.csv``, TensorBoard, or Weights &
    Biases, so the resolution is stated once instead of the curve vanishing silently.
    """

    def _build_module_with_trainer(self, tmp_path, **trainer_state):
        """Return an 'auto'-policy module whose stub trainer carries the given validation-loop state."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        state = dict(callbacks=[], sanity_checking=False, is_global_zero=True)
        state.update(trainer_state)
        module.trainer = SimpleNamespace(**state)
        return module

    @patch("rfdetr.training.module_model.logger")
    def test_notice_is_emitted_once_across_validation_epochs(self, mock_logger, tmp_path):
        """The skip notice is logged on the first validation epoch only, not once per epoch.

        Every validation epoch re-enters the hook, so an unguarded notice would repeat for the whole run and drown the
        per-epoch metric lines it sits next to.
        """
        module = self._build_module_with_trainer(tmp_path)

        module.on_validation_epoch_start()
        module.on_validation_epoch_start()

        mock_logger.info.assert_called_once()

    @patch("rfdetr.training.module_model.logger")
    def test_no_notice_when_a_callback_consumes_val_loss(self, mock_logger, tmp_path):
        """No notice is logged while a callback monitors val/loss, because the loss is still computed."""
        module = self._build_module_with_trainer(tmp_path, callbacks=[SimpleNamespace(monitor="val/loss")])

        module.on_validation_epoch_start()

        mock_logger.info.assert_not_called()

    @patch("rfdetr.training.module_model.logger")
    def test_sanity_check_pass_defers_the_notice(self, mock_logger, tmp_path):
        """The pre-training sanity-check validation pass does not consume the one-shot notice.

        Sanity checking runs before training starts and is invisible in most run logs; emitting the notice there would
        spend the single announcement on a pass the user is least likely to be watching.
        """
        module = self._build_module_with_trainer(tmp_path, sanity_checking=True)

        module.on_validation_epoch_start()

        mock_logger.info.assert_not_called()


class TestValidationLossPolicyCaching:
    """_should_compute_val_loss reuses the resolution that on_validation_epoch_start cached for the running epoch."""

    def test_per_batch_reads_reuse_the_epoch_resolution(self, tmp_path):
        """A callback attached mid-epoch does not change the policy the running validation epoch already resolved.

        The resolution walks every configured callback, so ``validation_step`` must not redo it per batch. Mutating
        ``trainer.callbacks`` after the epoch hook ran is the observable proxy: an unchanged answer proves the batch
        read came from the cached resolution rather than a fresh scan.
        """
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[], sanity_checking=False, is_global_zero=True)
        module.on_validation_epoch_start()

        module.trainer.callbacks.append(SimpleNamespace(monitor="val/loss"))

        assert module._should_compute_val_loss is False

    def test_resolution_is_refreshed_at_every_validation_epoch(self, tmp_path):
        """Each validation epoch re-resolves the policy, so a fit -> validate transition cannot serve a stale answer."""
        tc = _base_train_config(tmp_path, compute_val_loss="auto")
        module, *_ = _build_module(train_config=tc, tmp_path=tmp_path)
        module.trainer = SimpleNamespace(callbacks=[], sanity_checking=False, is_global_zero=True)
        module.on_validation_epoch_start()

        module.trainer.callbacks.append(SimpleNamespace(monitor="val/loss"))
        module.on_validation_epoch_start()

        assert module._should_compute_val_loss is True


class TestTestStep:
    """Tests for test_step() — verifies output dict shape, postprocessor invocation with correct original sizes, and
    test/loss logging.

    Mirrors :class:`TestValidationStep` since both steps share the same forward+postprocess logic and differ only in the
    logged metric prefix.
    """

    def _run_test_step(self, tmp_path):
        module, fake_model, fake_criterion, fake_pp = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
        result = module.test_step((samples, targets), batch_idx=0)
        return result, fake_pp, module

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("results", id="results-key"),
            pytest.param("targets", id="targets-key"),
        ],
    )
    def test_returns_dict_with_required_key(self, key, tmp_path):
        """Output dict must contain both 'results' and 'targets' for COCOEvalCallback."""
        result, _, _ = self._run_test_step(tmp_path)
        assert key in result

    def test_postprocess_called_with_orig_sizes(self, tmp_path):
        """Postprocessor must receive original image sizes to rescale predictions."""
        _result, fake_pp, _ = self._run_test_step(tmp_path)
        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (2, 2)

    def test_logs_test_loss(self, tmp_path):
        """Test loss must be logged under test/ prefix for monitoring."""
        _, _, module = self._run_test_step(tmp_path)
        test_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "test/loss"]
        assert len(test_loss_calls) == 1

    def test_model_called_with_samples_only(self, tmp_path):
        """Test step must pass only samples (not targets) to the model forward."""
        module, fake_model, fake_criterion, _ = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()

        module.test_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once_with(samples)

    def test_loss_prefix_differs_from_validation(self, tmp_path):
        """test_step must log 'test/loss', not 'val/loss', to keep metric namespaces separate."""
        _, _, module = self._run_test_step(tmp_path)
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "test/loss" in logged_keys
        assert "val/loss" not in logged_keys

    def test_can_disable_test_loss_computation(self, tmp_path):
        """compute_test_loss=False skips criterion call and test/loss logging."""
        tc = _base_train_config(tmp_path, compute_test_loss=False)
        module, fake_model, fake_criterion, _ = _build_module(train_config=tc, tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        module.log = MagicMock()

        result = module.test_step((samples, targets), batch_idx=0)

        fake_criterion.assert_not_called()
        logged_keys = [c[0][0] for c in module.log.call_args_list]
        assert "test/loss" not in logged_keys
        assert "results" in result and "targets" in result


class TestConfigureOptimizers:
    """Tests for configure_optimizers() — covers required output keys, AdamW optimizer type, step-interval scheduler, LR
    lambda warmup ramp, and step-decay behaviour before and after lr_drop."""

    def _setup_module(self, tmp_path, **train_overrides):
        tc = _base_train_config(tmp_path, **train_overrides)
        module, _, _, _ = _build_module(train_config=tc)

        trainer = MagicMock()
        trainer.estimated_stepping_batches = 1000
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)

        real_param = nn.Parameter(torch.randn(4, 4))
        param_dicts = [{"params": real_param, "lr": tc.lr}]
        return module, param_dicts

    @pytest.mark.parametrize(
        "key",
        [
            pytest.param("optimizer", id="optimizer-key"),
            pytest.param("lr_scheduler", id="lr-scheduler-key"),
        ],
    )
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_configure_optimizers_returns_required_key(self, mock_get_param_dict, key, tmp_path):
        """Lightning requires both 'optimizer' and 'lr_scheduler' keys in the returned config dict."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert key in module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_optimizer_is_adamw(self, mock_get_param_dict, tmp_path):
        """RF-DETR must use AdamW for its decoupled weight decay behavior."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert isinstance(module.configure_optimizers()["optimizer"], torch.optim.AdamW)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_adamw_optimizer_kwargs_forwarded(self, mock_get_param_dict, tmp_path):
        """optimizer_kwargs are forwarded to RF-DETR's default AdamW optimizer."""
        optimizer_kwargs = {"betas": (0.8, 0.95), "eps": 1e-7}
        module, param_dicts = self._setup_module(tmp_path, optimizer_kwargs=optimizer_kwargs)
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.defaults["betas"] == optimizer_kwargs["betas"]
        assert optimizer.defaults["eps"] == pytest.approx(optimizer_kwargs["eps"])

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_managed_optimizer_resolves_native_class(self, mock_get_param_dict, tmp_path):
        """Managed short names resolve to a native torch.optim class with lr/weight_decay injected."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        with patch(
            "rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer
        ) as mock_resolve:
            optimizer = module.configure_optimizers()["optimizer"]

        mock_resolve.assert_called_once_with("sgd")
        assert isinstance(optimizer, _RecordingOptimizer)
        assert optimizer.defaults["lr"] == pytest.approx(module.train_config.lr)
        assert optimizer.defaults["weight_decay"] == pytest.approx(module.train_config.weight_decay)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_managed_optimizer_preserves_rfdetr_param_groups(self, mock_get_param_dict, tmp_path):
        """Managed optimizers must receive RF-DETR param groups with layer-wise LR values."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        param_dicts[0]["lr"] = 2.5e-5
        mock_get_param_dict.return_value = param_dicts

        with patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer):
            optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.param_groups[0]["initial_lr"] == pytest.approx(2.5e-5)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_managed_optimizer_kwargs_forwarded(self, mock_get_param_dict, tmp_path):
        """optimizer_kwargs are forwarded to a managed optimizer constructor."""
        optimizer_kwargs = {"momentum": 0.9, "nesterov": True}
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd", optimizer_kwargs=optimizer_kwargs)
        mock_get_param_dict.return_value = param_dicts

        with patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer):
            optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.extra_kwargs == optimizer_kwargs

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_callable_optimizer_called_with_param_groups_only(self, mock_get_param_dict, tmp_path):
        """A non-reconstructable callable optimizer is invoked with the param groups and nothing else."""
        module, param_dicts = self._setup_module(tmp_path, optimizer=lambda params: _RecordingOptimizer(params))
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, _RecordingOptimizer)
        assert optimizer.extra_kwargs == {}

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_dotted_optimizer_built_from_kwargs_without_injection(self, mock_get_param_dict, tmp_path):
        """A dotted-path optimizer is built from optimizer_kwargs only, with no lr/weight_decay injection."""
        module, param_dicts = self._setup_module(
            tmp_path, optimizer="torch.optim.AdamW", optimizer_kwargs={"weight_decay": 0.5}
        )
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["weight_decay"] == pytest.approx(0.5)
        # Explicit mode must not inject the config lr as the optimizer-level default.
        assert optimizer.defaults["lr"] == pytest.approx(1e-3)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_uninstalled_import_path_optimizer_raises_value_error(self, mock_get_param_dict, tmp_path):
        """A dotted import path that cannot be imported surfaces as a configuration error at train start."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="pytorch_optimizer.Lion")
        mock_get_param_dict.return_value = param_dicts

        with (
            patch("rfdetr.training.module_model.importlib.import_module", side_effect=ImportError("No module")),
            pytest.raises(ValueError, match="Could not import optimizer"),
        ):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_real_import_path_optimizer_smoke(self, mock_get_param_dict, tmp_path):
        """A real pytorch-optimizer optimizer can be built via its import path when installed."""
        pytest.importorskip("pytorch_optimizer")
        module, param_dicts = self._setup_module(tmp_path, optimizer="pytorch_optimizer.Lion")
        mock_get_param_dict.return_value = param_dicts

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.__class__.__name__ == "Lion"

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_custom_optimizer_omits_weight_decay_when_unsupported(self, mock_get_param_dict, tmp_path):
        """weight_decay must not be injected into optimizers whose constructor does not accept it."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        with patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_NoWeightDecayOptimizer):
            optimizer = module.configure_optimizers()["optimizer"]

        assert isinstance(optimizer, _NoWeightDecayOptimizer)
        assert "weight_decay" not in optimizer.defaults

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    def test_fused_optimizer_warns_for_custom_optimizer_on_bf16(
        self,
        mock_bf16_supported,
        mock_cuda_available,
        mock_get_param_dict,
        tmp_path,
    ):
        """A custom optimizer on a fused-eligible BF16/CUDA run must warn that fused_optimizer is ignored."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = "bf16-mixed"

        with (
            patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RecordingOptimizer),
            patch("rfdetr.training.module_model.logger.warning") as mock_warning,
        ):
            module.configure_optimizers()

        assert any("fused_optimizer=True is ignored" in str(call.args[0]) for call in mock_warning.call_args_list)

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    def test_no_fused_warning_for_adamw_on_bf16(
        self,
        mock_bf16_supported,
        mock_cuda_available,
        mock_get_param_dict,
        tmp_path,
    ):
        """The built-in AdamW path uses fused and must not emit the fused-ignored warning."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = "bf16-mixed"

        with patch("rfdetr.training.module_model.logger.warning") as mock_warning:
            module.configure_optimizers()

        assert not any("fused_optimizer=True is ignored" in str(call.args[0]) for call in mock_warning.call_args_list)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_adamw_construction_error_is_rewrapped(self, mock_get_param_dict, tmp_path):
        """An unsupported AdamW kwarg must surface as an RF-DETR-specific initialization error."""
        module, param_dicts = self._setup_module(tmp_path, optimizer_kwargs={"nonexistent_adamw_kwarg": True})
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(TypeError, match="Failed to initialize optimizer 'adamw'"):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_pytorch_optimizer_construction_error_is_rewrapped(self, mock_get_param_dict, tmp_path):
        """A managed optimizer constructor failure must surface as an RF-DETR-specific initialization error."""
        module, param_dicts = self._setup_module(tmp_path, optimizer="sgd")
        mock_get_param_dict.return_value = param_dicts

        with (
            patch("rfdetr.training.module_model._resolve_native_optimizer", return_value=_RaisingOptimizer),
            pytest.raises(TypeError, match="Failed to initialize optimizer 'sgd'"),
        ):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_use_fused_optimizer_false_for_custom_optimizer(self, mock_cuda_available, mock_bf16_supported, tmp_path):
        """Fused AdamW lifecycle must not activate for custom optimizers, even on BF16/CUDA."""
        module, _ = self._setup_module(tmp_path, optimizer="sgd")
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_optimizer is False

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_use_fused_optimizer_true_for_adamw(self, mock_cuda_available, mock_bf16_supported, tmp_path):
        """Fused AdamW must activate for the built-in AdamW optimizer on BF16/CUDA."""
        module, _ = self._setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"

        assert module._use_fused_optimizer is True

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_scheduler_interval_is_step(self, mock_get_param_dict, tmp_path):
        """Scheduler must step per batch (not per epoch) for fine-grained warmup."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert module.configure_optimizers()["lr_scheduler"]["interval"] == "step"

    @pytest.mark.parametrize(
        "step, expected_behavior",
        [
            pytest.param(0, "warmup_start", id="warmup-start"),
            pytest.param(50, "warmup_mid", id="warmup-midpoint"),
        ],
    )
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_warmup_phase(self, mock_get_param_dict, step, expected_behavior, tmp_path):
        """LR lambda must produce a linear ramp during the warmup phase."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=1.0, epochs=10)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # steps_per_epoch=100, warmup_steps=100
        expected = float(step) / float(max(1, 100))
        assert lr_lambda(step) == pytest.approx(expected)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_step_decay_before_drop(self, mock_get_param_dict, tmp_path):
        """Before lr_drop epoch, the LR multiplier must remain at 1.0."""
        module, param_dicts = self._setup_module(
            tmp_path, warmup_epochs=0.0, epochs=10, lr_scheduler_kwargs={"lr_drop": 8}
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # lr_drop * steps_per_epoch = 8 * 100 = 800; step 500 < 800 → factor 1.0
        assert lr_lambda(500) == pytest.approx(1.0)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_step_decay_after_drop(self, mock_get_param_dict, tmp_path):
        """After lr_drop epoch, the LR multiplier must decay to 0.1."""
        module, param_dicts = self._setup_module(
            tmp_path, warmup_epochs=0.0, epochs=10, lr_scheduler_kwargs={"lr_drop": 8}
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # step 900 > 800 → factor 0.1
        assert lr_lambda(900) == pytest.approx(0.1)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_lr_lambda_cosine_reads_train_config_fields(self, mock_get_param_dict, tmp_path):
        """Cosine preset must read its floor from lr_scheduler_kwargs['min_factor']."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            epochs=10,
            lr_scheduler="cosine",
            lr_scheduler_kwargs={"min_factor": 0.2},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # At the final step, cosine schedule must end at lr_min_factor.
        assert lr_lambda(1000) == pytest.approx(0.2)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_explicit_dotted_scheduler_builds_from_kwargs(self, mock_get_param_dict, tmp_path):
        """An explicit dotted-path lr_scheduler is built from lr_scheduler_kwargs verbatim."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 30, "gamma": 0.1},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 30
        assert scheduler.gamma == pytest.approx(0.1)

    @patch("rfdetr.training.module_model._build_param_dicts")
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_per_parameter_lr_lambda_list_is_collapsed_onto_merged_groups(
        self, mock_get_param_dict, mock_build_param_dicts, tmp_path
    ):
        """A legacy per-parameter lr_lambda list is regrouped to one callback per merged group."""

        def shared_lambda(step: int) -> float:
            return 1.0

        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.LambdaLR",
            lr_scheduler_kwargs={"lr_lambda": [shared_lambda, shared_lambda]},
        )
        mock_get_param_dict.return_value = param_dicts
        lr = module.train_config.lr
        mock_build_param_dicts.return_value = [
            {"params": nn.Parameter(torch.randn(2)), "lr": lr},
            {"params": nn.Parameter(torch.randn(2)), "lr": lr},
        ]

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert scheduler.lr_lambdas == [shared_lambda]

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_managed_preset_builds_lambda_lr(self, mock_get_param_dict, tmp_path):
        """A managed preset still builds a LambdaLR at step interval."""
        module, param_dicts = self._setup_module(tmp_path, lr_scheduler="step")
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        config = module.configure_optimizers()["lr_scheduler"]

        assert isinstance(config["scheduler"], torch.optim.lr_scheduler.LambdaLR)
        assert config["interval"] == "step"

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_callable_scheduler_built_from_optimizer_only(self, mock_get_param_dict, tmp_path):
        """A non-reconstructable callable scheduler is invoked with the optimizer and nothing else."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.StepLR(optimizer, step_size=11),
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 11

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_explicit_scheduler_auto_wrapped_with_warmup(self, mock_get_param_dict, tmp_path):
        """With warmup_epochs>0 an explicit scheduler is wrapped in a SequentialLR linear warmup."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_epoch_interval_warmup_ramps_over_multiple_epochs(self, mock_get_param_dict, tmp_path):
        """An epoch-interval explicit scheduler gets a real warmup ramp sized in whole epochs (not optimizer steps)."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=3.0,
            epochs=10,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000  # steps_per_epoch = 100
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        # The ramp spans 3 epoch-steps (not 300 optimizer-steps) and actually ramps up (start_factor < 1.0).
        assert isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        warmup = scheduler._schedulers[0]
        assert warmup.total_iters == 3
        assert warmup.start_factor < 1.0

    @patch("rfdetr.training.module_model.logger")
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_epoch_interval_single_epoch_warmup_warns_and_skips(self, mock_get_param_dict, mock_logger, tmp_path):
        """A single-epoch epoch-interval warmup cannot ramp; it warns and is skipped, not silently flat."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            epochs=10,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]

        # No degenerate flat SequentialLR wrap; the no-op warmup is skipped loudly instead.
        assert not isinstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        mock_logger.warning.assert_called_once()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_explicit_scheduler_interval_propagates(self, mock_get_param_dict, tmp_path):
        """lr_scheduler_interval is forwarded to the Lightning scheduler config for explicit schedulers."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",
            lr_scheduler_kwargs={"step_size": 5},
            lr_scheduler_interval="epoch",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        assert module.configure_optimizers()["lr_scheduler"]["interval"] == "epoch"

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_reduce_on_plateau_sets_monitor_and_epoch_interval(self, mock_get_param_dict, tmp_path):
        """A plateau loss monitor makes automatic validation loss available each epoch."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
            lr_scheduler_monitor="val/loss",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        config = module.configure_optimizers()["lr_scheduler"]

        assert isinstance(config["scheduler"], torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert config["monitor"] == "val/loss"
        assert config["interval"] == "epoch"
        assert module._should_compute_val_loss is True

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_callable_plateau_scheduler_rejects_disabled_val_loss(self, mock_get_param_dict, tmp_path):
        """A closure-built ReduceLROnPlateau monitoring val/loss is rejected when compute_val_loss=False.

        ``TrainConfig.validate_explicit_val_loss_disable`` only string-compares ``lr_scheduler`` against the
        ReduceLROnPlateau dotted path, and a lambda closure — unlike a plain ``functools.partial``, which is desugared
        to that path — never reaches the comparison. The runtime check in ``configure_optimizers`` is therefore the only
        guard left once the concrete scheduler exists.
        """
        with warnings.catch_warnings():
            # A lambda lr_scheduler cannot round-trip through training_config.json; that reproducibility warning is
            # expected here and unrelated to the conflict under test.
            warnings.simplefilter("ignore", UserWarning)
            module, param_dicts = self._setup_module(
                tmp_path,
                warmup_epochs=0.0,
                lr_scheduler=lambda optimizer: torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5),
                lr_scheduler_monitor="val/loss",
                compute_val_loss=False,
            )
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(ValueError, match="incompatible with ReduceLROnPlateau"):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_uninstalled_scheduler_path_raises_value_error(self, mock_get_param_dict, tmp_path):
        """A dotted scheduler path that cannot be imported surfaces as a configuration error at train start."""
        module, param_dicts = self._setup_module(tmp_path, lr_scheduler="nonexistent_pkg.MyScheduler")
        mock_get_param_dict.return_value = param_dicts

        with (
            patch("rfdetr.training.module_model.importlib.import_module", side_effect=ImportError("No module")),
            pytest.raises(ValueError, match="Could not import lr_scheduler"),
        ):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_explicit_scheduler_construction_error_is_reraised(self, mock_get_param_dict, tmp_path):
        """A missing required scheduler kwarg surfaces as an RF-DETR configuration error, not a bare TypeError."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            lr_scheduler="torch.optim.lr_scheduler.StepLR",  # StepLR requires step_size
            lr_scheduler_kwargs={},
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        with pytest.raises(TypeError, match="Failed to initialize lr_scheduler"):
            module.configure_optimizers()

    @patch("rfdetr.training.module_model.logger")
    @patch("rfdetr.training.module_model.get_param_dict")
    def test_plateau_with_warmup_warns(self, mock_get_param_dict, mock_logger, tmp_path):
        """warmup_epochs>0 with ReduceLROnPlateau warns; a metric-driven scheduler cannot compose with a warmup ramp."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=1.0,
            lr_scheduler="torch.optim.lr_scheduler.ReduceLROnPlateau",
            lr_scheduler_monitor="val/loss",
        )
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        module.configure_optimizers()

        mock_logger.warning.assert_called_once()

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_fused_optimizer_disabled_when_precision_not_bf16(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        mock_get_param_dict,
        tmp_path,
    ):
        """Fused AdamW must be disabled when trainer precision is not bf16-mixed.

        On Ampere+ GPUs torch.cuda.is_bf16_supported() is True even when the trainer is configured for 32-true
        precision.  The old code always enabled fused AdamW based on GPU capability alone, crashing with ``params,
        grads, exp_avgs, and exp_avg_sqs must have same dtype, device, and layout`` when DDP gradient bucket views had
        non-matching strides. The fix checks ``trainer.precision`` before enabling fused.
        """
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        # Simulate trainer configured for full FP32 precision.
        module._trainer.precision = "32-true"

        optimizer = module.configure_optimizers()["optimizer"]

        assert not optimizer.defaults.get("fused")

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_fused_optimizer_enabled_when_precision_is_bf16_mixed(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        mock_get_param_dict,
        tmp_path,
    ):
        """Fused AdamW must be enabled when both GPU supports BF16 and trainer uses bf16-mixed.

        The fused path is beneficial (and safe) only when training precision is actually BF16: parameters, gradients,
        and optimizer state all stay in the same dtype/layout, satisfying the fused kernel requirements.
        """
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        # Simulate trainer configured for BF16 mixed precision.
        module._trainer.precision = "bf16-mixed"

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.defaults.get("fused") is True

    @patch("rfdetr.training.module_model.get_param_dict")
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_fused_optimizer_enabled_with_transformer_engine(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        mock_get_param_dict,
        tmp_path,
    ):
        """Default FP8 uses BF16 weights, so the built-in fused AdamW path must remain active."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts
        module._trainer.precision = "transformer-engine"

        optimizer = module.configure_optimizers()["optimizer"]

        assert optimizer.defaults.get("fused") is True

    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=False)
    def test_fused_optimizer_disabled_when_cuda_unavailable(self, mock_cuda_available, tmp_path):
        """_use_fused_optimizer must return False when CUDA is not available, regardless of precision."""
        module, _ = self._setup_module(tmp_path)
        module._trainer.precision = "bf16-mixed"

        assert not module._use_fused_optimizer

    @patch("rfdetr.training.module_model.get_param_dict")
    def test_total_steps_divided_by_grad_accum_for_keypoint_module(self, mock_get_param_dict, tmp_path):
        """Keypoint (manual-opt) path must divide estimated_stepping_batches by grad_accum_steps for LR scheduling.

        With microbatches=100, grad_accum_steps=4, epochs=1, warmup_epochs=0 the scheduler should span 25 optimizer
        steps (ceil(100/4)).  At step 24 (0-indexed last step) a cosine LR schedule should be nearly at lr_min_factor;
        if total_steps were mistakenly 100 the LR would still be near its peak at step 24.
        """
        import math

        grad_accum_steps = 4
        microbatches = 100
        lr_min_factor = 0.1
        tc = _base_train_config(
            tmp_path,
            grad_accum_steps=grad_accum_steps,
            warmup_epochs=0,
            epochs=1,
            lr_scheduler="cosine",
            lr_scheduler_kwargs={"min_factor": lr_min_factor},
        )
        module, _, _, _ = _build_module(
            model_config=_base_model_config(use_grouppose_keypoints=True, num_keypoints_per_class=[17]),
            train_config=tc,
        )
        trainer = MagicMock()
        trainer.estimated_stepping_batches = microbatches
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        real_param = nn.Parameter(torch.randn(4, 4))
        mock_get_param_dict.return_value = [{"params": real_param, "lr": tc.lr}]

        result = module.configure_optimizers()
        scheduler = result["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        expected_total_steps = max(1, math.ceil(microbatches / grad_accum_steps))  # 25
        # The cosine schedule reaches lr_min_factor exactly at step == total_steps (progress=1.0).
        # If total_steps were wrongly 100, lr at step 25 would still be ~0.87 (near peak).
        lr_at_decay_end = lr_lambda(expected_total_steps)
        assert lr_at_decay_end == pytest.approx(lr_min_factor, abs=1e-6)


class TestFusedOptimizerResumeStateNormalization:
    """Tests for resume-time fused AdamW state normalization."""

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_on_train_start_normalizes_restored_fused_optimizer_state(
        self,
        mock_cuda_available,
        mock_bf16_supported,
    ) -> None:
        """Resumed fused AdamW state tensors must match the live parameter layout before the first step."""
        module = RFDETRModelModule.__new__(RFDETRModelModule)
        module.model_config = SimpleNamespace(fused_optimizer=True)
        module.train_config = SimpleNamespace(optimizer="adamw")
        trainer = MagicMock()
        trainer.precision = "bf16-mixed"
        trainer.is_global_zero = True
        module._trainer = trainer
        module.optimizers = MagicMock()

        optimizer_param = nn.Parameter(torch.arange(6.0, dtype=torch.bfloat16).reshape(2, 3).t())
        optimizer = torch.optim.AdamW([optimizer_param], lr=1e-3)
        optimizer.state[optimizer_param] = {
            "exp_avg": torch.full((3, 2), 2.0, dtype=torch.float32),
            "exp_avg_sq": torch.full((3, 2), 3.0, dtype=torch.float64),
        }
        module.optimizers.return_value = optimizer

        with patch.object(type(module), "trainer", new_callable=PropertyMock) as trainer_prop:
            trainer_prop.return_value = trainer
            module.on_train_start()

        module.optimizers.assert_called_once_with(use_pl_optimizer=False)
        state = optimizer.state[optimizer_param]
        assert state["exp_avg"].dtype == optimizer_param.dtype
        assert state["exp_avg"].stride() == optimizer_param.stride()
        assert state["exp_avg_sq"].dtype == optimizer_param.dtype
        assert state["exp_avg_sq"].stride() == optimizer_param.stride()
        torch.testing.assert_close(state["exp_avg"], torch.full_like(optimizer_param, 2.0))
        torch.testing.assert_close(state["exp_avg_sq"], torch.full_like(optimizer_param, 3.0))

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_on_train_start_unwraps_optimizer_wrapper(
        self,
        mock_cuda_available,
        mock_bf16_supported,
    ) -> None:
        """Lightning-style optimizer wrappers must still be normalized on resume."""
        module = RFDETRModelModule.__new__(RFDETRModelModule)
        module.model_config = SimpleNamespace(fused_optimizer=True)
        module.train_config = SimpleNamespace(optimizer="adamw")
        trainer = MagicMock()
        trainer.precision = "bf16-mixed"
        trainer.is_global_zero = True
        module._trainer = trainer
        module.optimizers = MagicMock()

        optimizer_param = nn.Parameter(torch.arange(6.0, dtype=torch.bfloat16).reshape(2, 3).t())
        optimizer = torch.optim.AdamW([optimizer_param], lr=1e-3)
        optimizer.state[optimizer_param] = {
            "exp_avg": torch.full((3, 2), 4.0, dtype=torch.float32),
            "exp_avg_sq": torch.full((3, 2), 5.0, dtype=torch.float64),
        }
        module.optimizers.return_value = SimpleNamespace(optimizer=optimizer)

        with patch.object(type(module), "trainer", new_callable=PropertyMock) as trainer_prop:
            trainer_prop.return_value = trainer
            module.on_train_start()

        module.optimizers.assert_called_once_with(use_pl_optimizer=False)
        state = optimizer.state[optimizer_param]
        assert state["exp_avg"].dtype == optimizer_param.dtype
        assert state["exp_avg"].stride() == optimizer_param.stride()
        assert state["exp_avg_sq"].dtype == optimizer_param.dtype
        assert state["exp_avg_sq"].stride() == optimizer_param.stride()
        torch.testing.assert_close(state["exp_avg"], torch.full_like(optimizer_param, 4.0))
        torch.testing.assert_close(state["exp_avg_sq"], torch.full_like(optimizer_param, 5.0))

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_on_train_start_leaves_empty_optimizer_state_untouched(
        self,
        mock_cuda_available,
        mock_bf16_supported,
    ) -> None:
        """Fresh fused-optimizer runs with no restored state should remain a no-op."""
        module = RFDETRModelModule.__new__(RFDETRModelModule)
        module.model_config = SimpleNamespace(fused_optimizer=True)
        module.train_config = SimpleNamespace(optimizer="adamw")
        trainer = MagicMock()
        trainer.precision = "bf16-mixed"
        trainer.is_global_zero = True
        module._trainer = trainer
        module.optimizers = MagicMock()

        optimizer_param = nn.Parameter(torch.ones((2, 2), dtype=torch.bfloat16))
        optimizer = torch.optim.AdamW([optimizer_param], lr=1e-3)
        module.optimizers.return_value = optimizer

        with patch.object(type(module), "trainer", new_callable=PropertyMock) as trainer_prop:
            trainer_prop.return_value = trainer
            module.on_train_start()

        assert optimizer.state == {}


class TestClipGradients:
    """Tests for clip_gradients() — verifies precision gating mirrors configure_optimizers()."""

    def _setup_module(self, tmp_path, precision: str):
        tc = _base_train_config(tmp_path)
        module, _, _, _ = _build_module(train_config=tc)
        trainer = MagicMock()
        trainer.precision = precision
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        return module

    @pytest.mark.parametrize(
        "precision",
        [
            pytest.param("32-true", id="fp32"),
            pytest.param("16-mixed", id="fp16-mixed"),
        ],
    )
    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    def test_clip_gradients_delegates_to_super_when_not_bf16(
        self,
        mock_cuda_available,
        mock_bf16_supported,
        precision,
        tmp_path,
    ):
        """clip_gradients must delegate to super() when trainer precision is not a BF16 variant.

        On Ampere+ GPUs is_bf16_supported() is True regardless of actual precision. The method must check
        trainer.precision before choosing the fused path, mirroring the same gate in configure_optimizers() to prevent
        silent divergence.
        """
        module = self._setup_module(tmp_path, precision=precision)

        with patch.object(type(module).__bases__[0], "clip_gradients") as mock_super_clip:
            module.clip_gradients(MagicMock(), gradient_clip_val=0.1)

        mock_super_clip.assert_called_once()

    @patch("rfdetr.training.module_model.torch.cuda.is_bf16_supported", return_value=True)
    @patch("rfdetr.training.module_model.torch.cuda.is_available", return_value=True)
    @patch("rfdetr.training.module_model.torch.nn.utils.clip_grad_norm_")
    def test_clip_gradients_uses_clip_grad_norm_when_bf16_mixed(
        self,
        mock_clip_grad_norm,
        mock_cuda_available,
        mock_bf16_supported,
        tmp_path,
    ):
        """clip_gradients must call clip_grad_norm_ directly when precision is bf16-mixed.

        When fused AdamW is active (BF16, no GradScaler), the standard PTL AMP plugin refuses to clip gradients.
        clip_grad_norm_ is called directly instead, bypassing the scaler-aware path that would otherwise raise.
        """
        module = self._setup_module(tmp_path, precision="bf16-mixed")

        module.clip_gradients(MagicMock(), gradient_clip_val=0.5)

        mock_clip_grad_norm.assert_called_once()
        _, _call_kwargs = mock_clip_grad_norm.call_args
        # Positional arg[1] is max_norm
        assert mock_clip_grad_norm.call_args[0][1] == pytest.approx(0.5)


class TestPredictStep:
    """Tests for predict_step() — verifies that only samples (not targets) are passed to the model, that postprocess
    receives the correct original sizes, and that the postprocessor output is returned directly to the caller."""

    def test_calls_postprocess_with_orig_sizes(self, build_module):
        """Postprocessor must receive a (batch, 2) tensor of original image sizes."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch(batch_size=3)
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (3, 2)

    def test_returns_postprocess_output(self, build_module):
        """predict_step must return the postprocessor output directly to the caller."""
        module, fake_model, _, fake_pp = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}
        expected_output = [{"boxes": torch.zeros(1, 4)}]
        fake_pp.return_value = expected_output

        assert module.predict_step((samples, targets), batch_idx=0) is expected_output

    def test_model_called_with_samples_only(self, build_module):
        """Inference must pass only samples (not targets) to the model forward."""
        module, fake_model, _, _ = build_module()
        samples, targets = _make_batch()
        fake_model.return_value = {}

        module.predict_step((samples, targets), batch_idx=0)

        fake_model.assert_called_once_with(samples)

    def test_default_dataloader_idx_is_zero(self, build_module):
        """predict_step must work with the default dataloader_idx without errors."""
        module, fake_model, _, _ = build_module()
        fake_model.return_value = {}

        # Should not raise with default dataloader_idx.
        module.predict_step(_make_batch(), batch_idx=0)


class TestReinitializeDetectionHead:
    """Tests for reinitialize_detection_head() — verifies that the module delegates to the underlying model and that
    arbitrary class counts are forwarded unchanged."""

    def test_delegates_to_model(self, build_module):
        """Module must delegate head reinitialization to the underlying model."""
        module, fake_model, _, _ = build_module()

        module.reinitialize_detection_head(num_classes=42)

        fake_model.reinitialize_detection_head.assert_called_once_with(42)

    @pytest.mark.parametrize(
        "num_classes",
        [
            pytest.param(1, id="single-class"),
            pytest.param(80, id="coco-80"),
            pytest.param(365, id="objects365"),
        ],
    )
    def test_passes_various_class_counts(self, num_classes, build_module):
        """Arbitrary class counts must be forwarded to the underlying model unchanged."""
        module, fake_model, _, _ = build_module()

        module.reinitialize_detection_head(num_classes=num_classes)

        fake_model.reinitialize_detection_head.assert_called_once_with(num_classes)


class TestOnLoadCheckpoint:
    """Tests for on_load_checkpoint() — covers legacy .pth normalisation and positional-embedding interpolation for
    custom-resolution PTL checkpoints.

    Regression: issue #998 — resume with custom resolution crashed because
    on_load_checkpoint did not interpolate PE before PTL applied the state dict.
    """

    _PE_KEY = "model.backbone.0.encoder.encoder.embeddings.position_embeddings"

    def _make_ptl_checkpoint(self, pe_size_src: int, _pe_size_tgt: int, dim: int = 16) -> dict:
        """Build a minimal PTL checkpoint with mismatched PE shape.

        Args:
            pe_size_src: Source grid side length (checkpoint was saved with this PE).
            _pe_size_tgt: Target grid side length (model was built with this PE),
                accepted for test readability but intentionally unused here.
            dim: Embedding dimension (small value for fast tests).

        Returns:
            Checkpoint dict in PTL format with ``state_dict`` key.
        """
        n_src = pe_size_src * pe_size_src + 1  # +1 for class token
        return {
            "state_dict": {
                self._PE_KEY: torch.randn(1, n_src, dim),
                "model.other_layer.weight": torch.randn(4, 4),
            },
            "epoch": 44,
            "global_step": 1000,
        }

    def _make_legacy_pth_checkpoint(self, pe_size_src: int, dim: int = 16) -> dict:
        """Build a minimal legacy .pth checkpoint (no ``state_dict`` key).

        Args:
            pe_size_src: Source grid side length.
            dim: Embedding dimension.

        Returns:
            Checkpoint dict in legacy format with ``model`` key only.
        """
        n_src = pe_size_src * pe_size_src + 1
        pe_key_no_prefix = self._PE_KEY[len("model.") :]
        return {
            "model": {
                pe_key_no_prefix: torch.randn(1, n_src, dim),
                "other_layer.weight": torch.randn(4, 4),
            }
        }

    @pytest.mark.parametrize(
        "pe_src,pe_tgt",
        [
            pytest.param(36, 56, id="pe_interpolated_in_ptl_checkpoint"),
            pytest.param(36, 36, id="pe_unchanged_when_shapes_match"),
        ],
    )
    def test_ptl_checkpoint_pe_shape(self, pe_src, pe_tgt, build_module):
        """on_load_checkpoint must produce PE with tokens matching the model's positional_encoding_size.

        Regression for #998: resume from .ckpt with custom resolution crashed because PTL applied the checkpoint state
        dict before PE shapes were reconciled.
        """
        checkpoint = self._make_ptl_checkpoint(pe_size_src=pe_src, _pe_size_tgt=pe_tgt)

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        pe_after = checkpoint["state_dict"][self._PE_KEY]
        expected_tokens = pe_tgt * pe_tgt + 1
        assert pe_after.shape == (
            1,
            expected_tokens,
            16,
        ), f"PE should have {expected_tokens} tokens, got shape {tuple(pe_after.shape)}"

    def test_legacy_pth_normalised_and_pe_interpolated(self, build_module):
        """Legacy .pth checkpoint (no state_dict key) must be normalised and PE interpolated.

        on_load_checkpoint converts the raw "model" dict to PTL format and must also interpolate PE so that PTL's
        subsequent load_state_dict does not crash.
        """
        pe_src, pe_tgt = 36, 56
        checkpoint = self._make_legacy_pth_checkpoint(pe_size_src=pe_src)

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        assert "state_dict" in checkpoint, "Legacy checkpoint must be normalised to PTL format."
        pe_after = checkpoint["state_dict"][self._PE_KEY]
        expected_tokens = pe_tgt * pe_tgt + 1
        assert pe_after.shape == (1, expected_tokens, 16)

    def test_non_pe_tensors_not_modified(self, build_module):
        """on_load_checkpoint must not alter non-PE tensors in the state dict."""
        pe_src, pe_tgt = 36, 56
        checkpoint = self._make_ptl_checkpoint(pe_size_src=pe_src, _pe_size_tgt=pe_tgt)
        original_other = checkpoint["state_dict"]["model.other_layer.weight"].clone()

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=pe_tgt))
        module.on_load_checkpoint(checkpoint)

        assert torch.equal(checkpoint["state_dict"]["model.other_layer.weight"], original_other)

    def test_no_pe_keys_in_state_dict_is_noop(self, build_module):
        """on_load_checkpoint must not raise when state_dict contains no PE keys."""
        checkpoint = {
            "state_dict": {"model.some_layer.weight": torch.randn(4, 4)},
            "epoch": 1,
        }
        original_keys = set(checkpoint["state_dict"].keys())

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=36))
        module.on_load_checkpoint(checkpoint)

        assert set(checkpoint["state_dict"].keys()) == original_keys


class TestManualOptLRSchedulerStepping:
    """Manual-optimization (keypoint) path steps LR schedulers itself, honoring interval and plateau semantics."""

    def _module(self, tmp_path, **overrides):
        tc = _base_train_config(tmp_path, **overrides)
        module, _, _, _ = _build_module(train_config=tc)
        module.automatic_optimization = False
        return module

    def test_step_interval_scheduler_stepped_per_optimizer_step(self, tmp_path):
        """A step-interval scheduler is stepped by _step_lr_scheduler."""
        module = self._module(tmp_path)
        module._lr_scheduler_interval = "step"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            module._step_lr_scheduler()

        scheduler.step.assert_called_once_with()

    def test_epoch_interval_scheduler_not_stepped_per_optimizer_step(self, tmp_path):
        """An epoch-interval scheduler is not stepped by the per-optimizer-step hook."""
        module = self._module(tmp_path)
        module._lr_scheduler_interval = "epoch"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            module._step_lr_scheduler()

        scheduler.step.assert_not_called()

    def test_epoch_interval_scheduler_stepped_on_train_epoch_end(self, tmp_path):
        """on_train_epoch_end steps an epoch-interval scheduler on the manual path."""
        module = self._module(tmp_path)
        module._lr_scheduler_interval = "epoch"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.StepLR)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            module.on_train_epoch_end()

        scheduler.step.assert_called_once_with()

    def test_plateau_stepped_from_monitor_metric_on_validation_epoch_end(self, tmp_path):
        """on_validation_epoch_end steps ReduceLROnPlateau with the monitored metric."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.callback_metrics = {"val/loss": torch.tensor(1.23)}
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            module.on_validation_epoch_end()

        scheduler.step.assert_called_once()

    def test_plateau_raises_when_monitor_metric_missing(self, tmp_path):
        """on_validation_epoch_end fails loud (not silent warn) when the monitored metric is absent."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        trainer = MagicMock()
        trainer.sanity_checking = False
        trainer.callback_metrics = {}
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            with pytest.raises(RuntimeError, match="never be reduced"):
                module.on_validation_epoch_end()

        scheduler.step.assert_not_called()

    def test_plateau_not_stepped_during_sanity_check(self, tmp_path):
        """on_validation_epoch_end must not step plateau during Lightning's pre-training sanity check."""
        module = self._module(tmp_path)
        module._lr_scheduler_monitor = "val/loss"
        scheduler = MagicMock(spec=torch.optim.lr_scheduler.ReduceLROnPlateau)
        trainer = MagicMock()
        trainer.sanity_checking = True
        trainer.callback_metrics = {"val/loss": torch.tensor(1.23)}
        module._trainer = trainer
        type(module).trainer = property(lambda self: self._trainer)
        with patch.object(module, "lr_schedulers", return_value=scheduler):
            module.on_validation_epoch_end()

        scheduler.step.assert_not_called()
