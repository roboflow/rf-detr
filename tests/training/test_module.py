# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Comprehensive unit tests for RFDETRModule (LightningModule wrapper)."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRBaseConfig, TrainConfig
from rfdetr.utilities.tensors import NestedTensor

# ---------------------------------------------------------------------------
# Private helpers — used by both module-level fixtures and class-level _setup_*
# methods (which cannot inject pytest fixtures directly).
# Only define a private helper when it is called from more than one site;
# single-use logic belongs directly in the fixture body.
# ---------------------------------------------------------------------------


def _base_model_config(**overrides):
    """Return a minimal RFDETRBaseConfig with pretrain_weights disabled."""
    defaults = dict(pretrain_weights=None, device="cpu", num_classes=5)
    defaults.update(overrides)
    return RFDETRBaseConfig(**defaults)


def _base_train_config(tmp_path=None, **overrides):
    """Return a minimal TrainConfig suitable for unit tests."""
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
        lr_drop=8,
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
    """Return a MagicMock that behaves enough like an LWDETR model."""
    model = MagicMock(spec=nn.Module)
    real_param = nn.Parameter(torch.randn(4, 4))
    model.parameters.return_value = iter([real_param])
    model.named_parameters.return_value = iter([("weight", real_param)])
    model.update_drop_path = MagicMock()
    model.update_dropout = MagicMock()
    model.reinitialize_detection_head = MagicMock()
    return model


def _fake_criterion():
    """Return a MagicMock criterion with a realistic weight_dict."""
    criterion = MagicMock()
    criterion.weight_dict = {"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0}
    return criterion


def _fake_postprocess():
    """Return a callable MagicMock for postprocess."""
    return MagicMock(return_value=[{"boxes": torch.zeros(1, 4), "scores": torch.ones(1), "labels": torch.zeros(1)}])


def _build_module(model_config=None, train_config=None, tmp_path=None):
    """Construct RFDETRModule with build_model and build_criterion_and_postprocessors mocked."""
    mc = model_config or _base_model_config()
    tc = train_config or _base_train_config(tmp_path)
    fake_model = _fake_model()
    fake_criterion = _fake_criterion()
    fake_postprocess = _fake_postprocess()
    with (
        patch("rfdetr.training.module.build_model", return_value=fake_model),
        patch(
            "rfdetr.training.module.build_criterion_and_postprocessors", return_value=(fake_criterion, fake_postprocess)
        ),
    ):
        from rfdetr.training.module import RFDETRModule

        module = RFDETRModule(mc, tc)
    return module, fake_model, fake_criterion, fake_postprocess


def _make_batch(batch_size=2, channels=3, h=16, w=16):
    """Build a (NestedTensor, targets) tuple for testing."""
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


# ---------------------------------------------------------------------------
# Fixtures — inject common test infrastructure; prefer these over private
# helpers in test methods.  Class-level _setup_* helpers still use the private
# functions directly (they cannot inject fixtures themselves).
# ---------------------------------------------------------------------------


@pytest.fixture
def build_module(tmp_path):
    """Factory fixture — returns (module, fake_model, fake_criterion, fake_postprocess).

    build_model and build_criterion_and_postprocessors are mocked automatically.
    tmp_path is injected automatically so test methods do not need to declare it.
    """
    return lambda model_config=None, train_config=None: _build_module(model_config, train_config, tmp_path)


@pytest.fixture
def make_batch():
    """Factory fixture — call with optional batch_size/channels/h/w."""
    return _make_batch


class TestInit:
    """Tests for RFDETRModule.__init__ — covers attribute assignment and
    delegation to build_model() / build_criterion_and_postprocessors()
    when pretrain_weights is None."""

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
            patch("rfdetr.training.module.torch.compile") as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_not_called()

    def test_compile_runs_when_enabled_and_static_shapes(self, tmp_path):
        """torch.compile runs when compile=True and multi_scale=False on CUDA."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        with (
            patch("torch.cuda.is_available", return_value=True),
            patch("rfdetr.training.module.torch.compile", side_effect=lambda m, **_: m) as mock_compile,
        ):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()


class TestLoadPretrainWeights:
    """Tests for _load_pretrain_weights() — covers checkpoint validation, detection-head
    reinitialization on class-count mismatch, query-embedding trimming, re-download on
    corruption, and class-name extraction from checkpoint metadata."""

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

    @patch("rfdetr.training.module.torch.load")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_loads_checkpoint_successfully(self, mock_validate, mock_torch_load, base_model_config, build_module):
        """A valid checkpoint must be validated, loaded, and applied to the model."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/fake/weights.pth"
        module._load_pretrain_weights()

        mock_validate.assert_called_once_with("/fake/weights.pth", strict=False)
        module.model.load_state_dict.assert_called_once()

    @patch("rfdetr.training.module.torch.load")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_class_count_mismatch_triggers_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head is expanded to checkpoint size, then trimmed back to config size."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/fake/weights.pth"
        module._load_pretrain_weights()

        # First call: expand to checkpoint size so load_state_dict shapes match.
        # Second call: trim back to configured num_classes + 1 (background class).
        from unittest.mock import call

        fake_model.reinitialize_detection_head.assert_has_calls([call(91), call(6)])
        assert fake_model.reinitialize_detection_head.call_count == 2

    @patch("rfdetr.training.module.torch.load")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_class_count_match_does_not_reinitialize(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Detection head must NOT be reinitialized when class counts match."""
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=6)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/fake/weights.pth"
        module._load_pretrain_weights()

        fake_model.reinitialize_detection_head.assert_not_called()

    @patch("rfdetr.training.module.torch.load")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_query_embedding_trimmed_to_configured_count(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Oversized query embeddings in checkpoint must be trimmed to match config."""
        mc = base_model_config(num_classes=90)
        module, _, _, _ = build_module(model_config=mc)

        num_queries = module._args.num_queries
        group_detr = module._args.group_detr
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

        module._args.pretrain_weights = "/fake/weights.pth"
        module._load_pretrain_weights()

        assert checkpoint["model"]["refpoint_embed.weight"].shape[0] == desired
        assert checkpoint["model"]["query_feat.weight"].shape[0] == desired

    @patch("rfdetr.training.module.os.path.isfile", return_value=True)
    @patch("rfdetr.training.module.download_pretrain_weights")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_redownloads_on_load_failure(
        self, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """A corrupted checkpoint must trigger re-download and a second load attempt."""
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        module, _, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/fake/weights.pth"

        load_calls = [0]

        def fake_torch_load(*args, **kwargs):
            load_calls[0] += 1
            if load_calls[0] == 1:
                raise RuntimeError("corrupted file")
            return checkpoint

        with patch("rfdetr.training.module.torch.load", side_effect=fake_torch_load):
            module._load_pretrain_weights()

        # Verify a redownload with validate_md5=False was triggered after load failure.
        redownload_calls = [c for c in mock_download.call_args_list if c.kwargs.get("redownload") is True]
        assert len(redownload_calls) >= 1
        assert all(c.kwargs.get("validate_md5") is False for c in redownload_calls)
        assert load_calls[0] == 2

    @patch("rfdetr.training.module.os.path.isfile", return_value=False)
    @patch("rfdetr.training.module.download_pretrain_weights")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    @patch("rfdetr.training.module.torch.load")
    def test_download_before_load_when_weights_absent(
        self, mock_torch_load, mock_validate, mock_download, mock_isfile, base_model_config, build_module
    ):
        """download_pretrain_weights must be called before torch.load so a fresh
        environment (e.g. Colab) downloads weights automatically.

        Regression test: previously download was only called as an except-block
        fallback, but ModelWeights.from_filename received the absolute path and
        returned None, causing a silent no-op and a FileNotFoundError.
        """
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/content/rf-detr-base.pth"
        module._load_pretrain_weights()

        # download_pretrain_weights must have been called at least once before any load
        assert mock_download.call_count >= 1
        first_call = mock_download.call_args_list[0]
        assert first_call.args[0] == "/content/rf-detr-base.pth"

    @patch("rfdetr.training.module.torch.load")
    @patch("rfdetr.training.module.validate_pretrain_weights")
    def test_pretrain_class_names_stored_when_present(
        self, mock_validate, mock_torch_load, base_model_config, build_module
    ):
        """Class names from checkpoint args must be saved for transfer learning use."""
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(class_names=["cat", "dog"])
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module._args.pretrain_weights = "/fake/weights.pth"
        module._load_pretrain_weights()

        assert module._pretrain_class_names == ["cat", "dog"]


class TestApplyLora:
    """Tests for _apply_lora() — verifies that PEFT LoraConfig is constructed with the
    correct target modules and that the backbone encoder is replaced in-place with the
    wrapped PEFT model."""

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
            patch("rfdetr.training.module.build_model", return_value=fake_model),
            patch(
                "rfdetr.training.module.build_criterion_and_postprocessors",
                return_value=(_fake_criterion(), _fake_postprocess()),
            ),
        ):
            from rfdetr.training.module import RFDETRModule

            module = RFDETRModule(mc, tc)

        return module, fake_model, fake_backbone_0, fake_encoder

    @patch("peft.get_peft_model")
    @patch("peft.LoraConfig")
    def test_calls_lora_config_with_correct_target_modules(self, mock_lora_cfg_class, mock_get_peft, tmp_path):
        """LoRA must target the expected attention and token projection modules."""
        module, _, _, _ = self._build_module_with_backbone(tmp_path)
        mock_get_peft.return_value = MagicMock()

        module._apply_lora()

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

        module._apply_lora()

        assert mock_get_peft.call_args[0][0] is fake_encoder
        assert fake_backbone_0.encoder is peft_wrapped


class TestOnFitStart:
    """Tests for on_fit_start() seeding behavior."""

    @patch("rfdetr.training.module.seed_everything")
    def test_seed_applied_when_configured(self, mock_seed, base_train_config, build_module):
        """Configured seed should be applied at fit start."""
        tc = base_train_config(seed=7)
        module, _, _, _ = build_module(train_config=tc)

        module.on_fit_start()

        mock_seed.assert_called_once_with(7, workers=True)

    @patch("rfdetr.training.module.seed_everything")
    def test_seed_skipped_when_none(self, mock_seed, base_train_config, build_module):
        """No seed means on_fit_start should not call seed_everything."""
        tc = base_train_config(seed=None)
        module, _, _, _ = build_module(train_config=tc)

        module.on_fit_start()

        mock_seed.assert_not_called()


class TestOnTrainBatchStart:
    """Tests for on_train_batch_start() — covers multi-scale interpolation of
    NestedTensor inputs and verifies regularization scheduling is delegated to
    DropPathCallback."""

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


class TestTransferBatchToDevice:
    """Tests for transfer_batch_to_device() — verifies that NestedTensor samples and
    all target-dict tensors are correctly moved to the target device without unwrapping
    the NestedTensor into plain tensors."""

    def test_samples_transferred_to_target_device(self, build_module):
        """Both tensors and mask in NestedTensor must land on the target device."""
        module, _, _, _ = build_module()
        samples, targets = _make_batch()
        device = torch.device("cpu")

        result_samples, _ = module.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        assert result_samples.tensors.device == device
        assert result_samples.mask.device == device

    def test_targets_transferred_to_target_device(self, build_module):
        """All tensor values in every target dict must be moved to the target device."""
        module, _, _, _ = build_module()
        samples, targets = _make_batch()
        device = torch.device("cpu")

        _, result_targets = module.transfer_batch_to_device((samples, targets), device, dataloader_idx=0)

        for t in result_targets:
            for v in t.values():
                assert v.device == device

    def test_returns_tuple_of_correct_length(self, build_module):
        """Return value must be a (samples, targets) tuple to match batch contract."""
        module, _, _, _ = build_module()
        result = module.transfer_batch_to_device(_make_batch(), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_preserves_nested_tensor_type(self, build_module):
        """Device transfer must not unwrap NestedTensor into plain tensors."""
        module, _, _, _ = build_module()
        samples, targets = _make_batch()

        result_samples, _ = module.transfer_batch_to_device((samples, targets), torch.device("cpu"), dataloader_idx=0)

        assert isinstance(result_samples, NestedTensor)


class TestTrainingStep:
    """Tests for training_step() — covers weighted loss aggregation, per-loss logging
    under the train/ prefix, prog_bar visibility, scalar tensor output, and that losses
    absent from weight_dict are excluded from the total."""

    def _run_step(self, tmp_path, loss_dict=None, weight_dict=None, accumulate_grad_batches=1):
        module, fake_model, fake_criterion, _ = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = loss_dict or {"loss_ce": torch.tensor(1.0)}
        fake_criterion.weight_dict = weight_dict or {"loss_ce": 1.0}
        module.log = MagicMock()
        module.log_dict = MagicMock()
        trainer = MagicMock()
        trainer.accumulate_grad_batches = accumulate_grad_batches
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

    def test_loss_normalised_by_accum_steps(self, tmp_path):
        """Loss must be divided by accumulate_grad_batches to match legacy engine scaling."""
        loss_dict = {"loss_ce": torch.tensor(4.0)}
        weight_dict = {"loss_ce": 1.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict, accumulate_grad_batches=4)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(1.0)  # 4.0 / 4

    def test_logs_train_loss_to_prog_bar(self, tmp_path):
        """Aggregate training loss must be logged with prog_bar=True for visibility."""
        module, samples, targets, _, _ = self._run_step(tmp_path)

        module.training_step((samples, targets), batch_idx=0)

        train_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "train/loss"]
        assert len(train_loss_calls) == 1
        assert train_loss_calls[0].kwargs.get("prog_bar") is True

    def test_logs_individual_losses_as_dict(self, tmp_path):
        """Each component loss must be logged separately under train/ prefix."""
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

    def test_ignores_losses_not_in_weight_dict(self, tmp_path):
        """Losses absent from weight_dict (e.g. cardinality_error) must not affect total."""
        loss_dict = {"loss_ce": torch.tensor(1.0), "cardinality_error": torch.tensor(99.0)}
        weight_dict = {"loss_ce": 2.0}
        module, samples, targets, _, _ = self._run_step(tmp_path, loss_dict, weight_dict)

        loss = module.training_step((samples, targets), batch_idx=0)

        assert loss.item() == pytest.approx(2.0)


class TestValidationStep:
    """Tests for validation_step() — verifies output dict shape, postprocessor
    invocation with correct original sizes, and val/loss logging."""

    def _run_val_step(self, tmp_path):
        module, fake_model, fake_criterion, fake_pp = _build_module(tmp_path=tmp_path)
        samples, targets = _make_batch()
        fake_model.return_value = {}
        fake_criterion.return_value = {"loss_ce": torch.tensor(0.5)}
        fake_criterion.weight_dict = {"loss_ce": 1.0}
        module.log = MagicMock()
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
        result, fake_pp, _ = self._run_val_step(tmp_path)
        fake_pp.assert_called_once()
        orig_sizes = fake_pp.call_args[0][1]
        assert orig_sizes.shape == (2, 2)

    def test_logs_val_loss(self, tmp_path):
        """Validation loss must be logged for monitoring and early stopping."""
        _, _, module = self._run_val_step(tmp_path)
        val_loss_calls = [c for c in module.log.call_args_list if c[0][0] == "val/loss"]
        assert len(val_loss_calls) == 1

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


class TestTestStep:
    """Tests for test_step() — verifies output dict shape, postprocessor
    invocation with correct original sizes, and test/loss logging.

    Mirrors :class:`TestValidationStep` since both steps share the same
    forward+postprocess logic and differ only in the logged metric prefix.
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
        result, fake_pp, _ = self._run_test_step(tmp_path)
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
    """Tests for configure_optimizers() — covers required output keys, AdamW optimizer
    type, step-interval scheduler, LR lambda warmup ramp, and step-decay behaviour
    before and after lr_drop."""

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
    @patch("rfdetr.training.module.get_param_dict")
    def test_configure_optimizers_returns_required_key(self, mock_get_param_dict, key, tmp_path):
        """Lightning requires both 'optimizer' and 'lr_scheduler' keys in the returned config dict."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert key in module.configure_optimizers()

    @patch("rfdetr.training.module.get_param_dict")
    def test_optimizer_is_adamw(self, mock_get_param_dict, tmp_path):
        """RF-DETR must use AdamW for its decoupled weight decay behavior."""
        module, param_dicts = self._setup_module(tmp_path)
        mock_get_param_dict.return_value = param_dicts

        assert isinstance(module.configure_optimizers()["optimizer"], torch.optim.AdamW)

    @patch("rfdetr.training.module.get_param_dict")
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
    @patch("rfdetr.training.module.get_param_dict")
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

    @patch("rfdetr.training.module.get_param_dict")
    def test_lr_lambda_step_decay_before_drop(self, mock_get_param_dict, tmp_path):
        """Before lr_drop epoch, the LR multiplier must remain at 1.0."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=10, lr_drop=8)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # lr_drop * steps_per_epoch = 8 * 100 = 800; step 500 < 800 → factor 1.0
        assert lr_lambda(500) == pytest.approx(1.0)

    @patch("rfdetr.training.module.get_param_dict")
    def test_lr_lambda_step_decay_after_drop(self, mock_get_param_dict, tmp_path):
        """After lr_drop epoch, the LR multiplier must decay to 0.1."""
        module, param_dicts = self._setup_module(tmp_path, warmup_epochs=0.0, epochs=10, lr_drop=8)
        module._trainer.estimated_stepping_batches = 1000
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # step 900 > 800 → factor 0.1
        assert lr_lambda(900) == pytest.approx(0.1)

    @patch("rfdetr.training.module.get_param_dict")
    def test_lr_lambda_cosine_reads_train_config_fields(self, mock_get_param_dict, tmp_path):
        """Cosine scheduler must read lr_scheduler/lr_min_factor from TrainConfig."""
        module, param_dicts = self._setup_module(
            tmp_path,
            warmup_epochs=0.0,
            epochs=10,
            lr_scheduler="cosine",
            lr_min_factor=0.2,
        )
        module._trainer.estimated_stepping_batches = 1000
        # Guard against regressions that read deprecated Namespace fields.
        delattr(module._args, "lr_scheduler")
        delattr(module._args, "lr_min_factor")
        mock_get_param_dict.return_value = param_dicts

        scheduler = module.configure_optimizers()["lr_scheduler"]["scheduler"]
        lr_lambda = scheduler.lr_lambdas[0]

        # At the final step, cosine schedule must end at lr_min_factor.
        assert lr_lambda(1000) == pytest.approx(0.2)


class TestPredictStep:
    """Tests for predict_step() — verifies that only samples (not targets) are passed
    to the model, that postprocess receives the correct original sizes, and that the
    postprocessor output is returned directly to the caller."""

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
    """Tests for reinitialize_detection_head() — verifies that the module delegates
    to the underlying model and that arbitrary class counts are forwarded unchanged."""

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
