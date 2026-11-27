# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for RFDETRModelModule construction, pretrained/LoRA weight loading, head reinit and checkpoint restore."""

import logging
import pickle
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest
import torch
from torch import nn

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.models.weights import apply_lora, load_pretrain_weights
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.box_ops import pairwise_box_l1_cost
from tests._markers import _requires_cuda
from tests.training.helpers import (
    _base_model_config,
    _base_train_config,
    _build_module,
    _fake_criterion,
    _fake_model,
    _FakeCriterion,
    _make_batch,
)
from tests.training.helpers import _mock_postprocess as _fake_postprocess


def test_keypoint_training_resets_gaussian_parameters_after_pretrained_load(tmp_path, mocker) -> None:
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

    mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=fake_model)
    mock_load_pretrain_weights = mocker.patch("rfdetr.training.module_model.load_pretrain_weights")
    mocker.patch(
        "rfdetr.training.module_model.build_criterion_from_config",
        return_value=(_fake_criterion(), _fake_postprocess()),
    )
    mock_load_pretrain_weights.side_effect = lambda *_args, **_kwargs: events.append("load")
    fake_model.reset_keypoint_gaussian_parameters.side_effect = lambda: events.append("reset")

    from rfdetr.training.module_model import RFDETRModelModule

    RFDETRModelModule(mc, tc)

    mock_load_pretrain_weights.assert_called_once_with(fake_model, mc)
    fake_model.reset_keypoint_gaussian_parameters.assert_called_once_with()
    assert events == ["load", "reset"]


class _DynamicShapeModel(nn.Module):
    """Tiny model whose loss path depends on the multi-scale batch tensor."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))

    def forward(self, samples, targets=None):
        return {"dummy": samples.tensors.mean() * self.weight}

    def update_drop_path(self, *args, **kwargs) -> None:
        pass

    def update_dropout(self, *args, **kwargs) -> None:
        pass

    def reinitialize_detection_head(self, *args, **kwargs) -> None:
        pass


#: Skip a test node that replays Inductor CUDA graph trees.
_requires_cudagraph_trees = pytest.mark.skipif(
    not torch.cuda.is_available() or not hasattr(torch.compiler, "cudagraph_mark_step_begin"),
    reason="combined cudagraph regression requires CUDA graph-tree support",
)


class TestInit:
    """Tests for RFDETRModelModule.__init__.

    Covers attribute assignment and delegation to build_model() / build_criterion_and_postprocessors() when
    pretrain_weights is None.
    """

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
        """Only keypoint models need manual optimization for box-normalized accumulation.

        Detection and segmentation keep Lightning's automatic optimization path.
        """
        module, _, _, _ = build_module(model_config=model_config)

        assert module._use_manual_optimization is expected_manual
        assert module.automatic_optimization is (not expected_manual)

    @pytest.mark.parametrize(
        "attribute, built_index",
        [("model", 1), ("criterion", 2), ("postprocess", 3)],
    )
    def test_built_component_is_assigned(self, build_module, attribute, built_index):
        """__init__ assigns the built model, criterion and postprocessor to module.model/.criterion/.postprocess."""
        built = build_module()
        assert getattr(built[0], attribute) is built[built_index]

    def test_stores_configs(self, base_model_config, base_train_config, build_module):
        """Both model and train configs must be stored for later access."""
        mc = base_model_config()
        tc = base_train_config()
        module, _, _, _ = build_module(model_config=mc, train_config=tc)
        assert module.model_config is mc
        assert module.train_config is tc

    def test_compile_runs_when_multi_scale_enabled(self, mocker, tmp_path):
        """torch.compile still runs with multi_scale=True, which is the shipped default.

        ``dynamic=True`` is passed precisely so one graph covers every (H, W) the multi-scale pipeline produces, so
        multi-scale is not a reason to skip compilation on CUDA.
        """
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=True)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        _, _, criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs["dynamic"] is True
        criterion.enable_compiled_detection_losses.assert_not_called()

    def test_compile_does_not_enable_global_error_suppression(self, mocker, tmp_path: Path) -> None:
        """RF-DETR must not hide compiler failures or change unrelated models' fallback policy."""
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda model, **_: model)
        with torch._dynamo.config.patch(suppress_errors=False):
            _build_module(model_config=_base_model_config(compile=True), tmp_path=tmp_path)
            assert torch._dynamo.config.suppress_errors is False

    @pytest.mark.gpu
    @_requires_cuda
    def test_compiled_multi_scale_forward_backward_across_resolutions(self, tmp_path):
        """The real compiled CUDA module must backpropagate finite losses at two resized multi-scale resolutions."""
        torch.manual_seed(0)
        model_config = _base_model_config(compile=True, resolution=128)
        train_config = _base_train_config(tmp_path, multi_scale=True)
        model = _DynamicShapeModel().cuda()
        criterion = _FakeCriterion()

        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("rfdetr.training.module_model.build_model_from_config", return_value=model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(criterion, _fake_postprocess()),
            ),
        ):
            module = RFDETRModelModule(model_config, train_config)

        resized_shapes = set()
        for global_step in (0, 1):
            module.trainer = SimpleNamespace(global_step=global_step)
            samples, targets = _make_batch(h=128, w=128)
            samples.tensors = samples.tensors.cuda()
            samples.mask = samples.mask.cuda()

            module.on_train_batch_start((samples, targets), batch_idx=global_step)
            resized_shapes.add(tuple(samples.tensors.shape[-2:]))
            loss = criterion(module.model(samples, targets), targets)["loss_ce"]
            loss.backward()

            assert torch.isfinite(loss)
            assert model.weight.grad is not None
            assert torch.isfinite(model.weight.grad)
            model.zero_grad(set_to_none=True)

        assert len(resized_shapes) == 2

    @pytest.mark.gpu
    @_requires_cuda
    def test_static_compile_runs_compiled_batched_losses_through_training_step(self, tmp_path: Path) -> None:
        """A real RF-DETR Nano training step statically compiles the model and reaches the compiled loss graph.

        The shipped builders create the detector, the criterion (``group_detr=13``) and the matcher. Only the whole-
        model ``torch.compile`` is replaced by a recorder, because compiling the detector takes minutes; the loss graph
        is compiled for real, and the step's loss matches the per-layer path on the same weights.
        """
        torch.manual_seed(0)
        real_compile = torch.compile
        model_config = RFDETRNanoConfig(num_classes=5, pretrain_weights=None, compile=True, device="cuda")
        train_config = _base_train_config(tmp_path, multi_scale=False)

        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch(
                "rfdetr.training.module_model.torch.compile",
                side_effect=lambda target, **kwargs: (
                    target if isinstance(target, nn.Module) else real_compile(target, **kwargs)
                ),
            ) as compiler,
        ):
            module = RFDETRModelModule(model_config, train_config).cuda().train()

        model_compiles = [call for call in compiler.call_args_list if isinstance(call.args[0], nn.Module)]
        assert len(model_compiles) == 1
        assert model_compiles[0].kwargs["dynamic"] is False
        criterion = module.criterion
        assert type(criterion) is SetCriterion
        assert criterion._compile_batched_detection_losses is True

        resolution = model_config.resolution
        samples, _ = _make_batch(batch_size=2, h=resolution, w=resolution)
        samples.tensors = samples.tensors.cuda()
        samples.mask = samples.mask.cuda()
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2], [0.3, 0.6, 0.1, 0.2]][:count], device="cuda"),
                "labels": torch.tensor([1, 3][:count], device="cuda"),
            }
            for count in (2, 1)
        ]
        module.log = MagicMock()
        module.log_dict = MagicMock()
        trainer = SimpleNamespace(accumulate_grad_batches=1)
        with patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer):
            loss = module.training_step((samples, targets), batch_idx=0)
            loss.backward()
            with patch.object(criterion, "_can_batch_detection_losses", return_value=False):
                fallback_loss = module.training_step((samples, targets), batch_idx=0)

        assert criterion._compiled_batched_detection_losses is not None, "the compiled loss graph never ran"
        assert torch.isfinite(loss)
        gradients = [parameter.grad for parameter in module.model.parameters() if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert torch.allclose(loss.detach(), fallback_loss.detach(), rtol=1e-4, atol=1e-6)

    @pytest.mark.gpu
    @_requires_cudagraph_trees
    def test_compiled_cudagraph_trees_replay_two_optimizer_steps(self, mocker, tmp_path: Path) -> None:
        """Combined compilation and CUDA graph trees must preserve finite gradients across optimizer replays."""
        torch.manual_seed(0)
        model_config = _base_model_config(compile=True, cuda_graphs=True, resolution=128)
        train_config = _base_train_config(tmp_path, multi_scale=False)
        model = _DynamicShapeModel().cuda().train()
        criterion = _FakeCriterion()

        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("rfdetr.training.module_model.build_model_from_config", return_value=model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(criterion, _fake_postprocess()),
            ),
        ):
            module = RFDETRModelModule(model_config, train_config)

        assert module._inductor_cudagraphs is True
        optimizer = torch.optim.SGD(module.model.parameters(), lr=1e-3)
        trainer = SimpleNamespace(accumulate_grad_batches=1)
        module.log = MagicMock()
        module.log_dict = MagicMock()
        mocker.patch.object(type(module), "trainer", new_callable=PropertyMock, return_value=trainer)
        for step in range(2):
            samples, targets = _make_batch(h=128, w=128)
            samples.tensors = samples.tensors.cuda()
            samples.mask = samples.mask.cuda()
            loss = module.training_step((samples, targets), batch_idx=step)
            loss.backward()

            assert torch.isfinite(loss)
            assert model.weight.grad is not None
            assert torch.isfinite(model.weight.grad).all()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

    @pytest.mark.parametrize("accelerator", ["xla", "tpu"])
    def test_compile_disabled_on_xla_accelerator_even_with_static_shapes(self, mocker, accelerator, tmp_path):
        """XLA/TPU never compiles, and that no longer depends on multi_scale being set.

        This is the invariant the removed ``not multi_scale`` clause was documented as protecting; the accelerator check
        is what actually enforces it.
        """
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile")
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False, accelerator=accelerator)
        _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_not_called()

    def test_compile_disabled_when_device_is_not_cuda(self, mocker, tmp_path, caplog, monkeypatch):
        """A non-CUDA accelerator disables compilation regardless of multi_scale, with an explanatory notice.

        A caller who sets ``compile=True`` on CPU/XLA/TPU must still learn why nothing was compiled, the same way the
        old multi_scale-only notice used to inform CUDA callers.
        """
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=True)
        # get_logger() sets propagate=False on the "rf-detr" logger, so caplog's root-level
        # handler only sees its records while propagation is re-enabled.
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        mocker.patch("rfdetr.config.DEVICE", "cpu")
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile")
        with caplog.at_level(logging.INFO, logger="rf-detr"):
            _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_not_called()
        assert any("Disabling torch.compile" in record.getMessage() for record in caplog.records)

    def test_compile_runs_when_enabled_and_static_shapes(self, mocker, tmp_path):
        """torch.compile runs when compile=True and multi_scale=False on CUDA."""
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        _, _, criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs["dynamic"] is False
        criterion.enable_compiled_detection_losses.assert_called_once_with()

    @pytest.mark.parametrize(
        ("model_overrides", "train_overrides"),
        [
            pytest.param({}, {"square_resize_div_64": False}, id="aspect-ratio-resize"),
            pytest.param(
                {},
                {"aug_config": {"RandomCrop": {"height": 320, "width": 320, "p": 0.5}}},
                id="custom-shape-changing-augmentation",
            ),
            pytest.param({"cuda_graphs": True}, {}, id="inductor-cuda-graphs"),
            pytest.param({"segmentation_head": True}, {}, id="segmentation"),
            pytest.param({"use_grouppose_keypoints": True}, {}, id="keypoints"),
        ],
    )
    def test_compile_keeps_dynamic_shapes_when_batch_shape_can_vary(
        self, mocker, model_overrides: dict[str, Any], train_overrides: dict[str, Any], tmp_path: Path
    ) -> None:
        """A fixed ``multi_scale=False`` run still keeps the dynamic recipe outside the measured detection route.

        Aspect-preserving resize pads each batch to its own per-axis maximum, and custom augmentations may change image
        shape after resize; static graphs would recompile when those batch shapes vary. Inductor CUDA graph trees keep
        their established recipe, and segmentation and keypoint models were not measured. None of these routes enables
        the batched loss graph.
        """
        mc = _base_model_config(compile=True, **model_overrides)
        tc = _base_train_config(tmp_path, multi_scale=False, **train_overrides)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("torch._inductor.config", SimpleNamespace(triton=SimpleNamespace()))
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        _, _, criterion, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        mock_compile.assert_called_once()
        assert mock_compile.call_args.kwargs["dynamic"] is True
        criterion.enable_compiled_detection_losses.assert_not_called()

    def test_compile_tolerates_criterion_without_matcher(self, mocker, tmp_path: Path) -> None:
        """A criterion that exposes no ``matcher`` attribute must not break compiled construction.

        ``_FakeCriterion`` is what the GPU compile regressions inject; reading ``self.criterion.matcher`` unguarded
        raised ``AttributeError`` there before the ``getattr`` probe.
        """
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_fake_model())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(_FakeCriterion(), _fake_postprocess()),
        )
        compiler = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        module = RFDETRModelModule(_base_model_config(compile=True), _base_train_config(tmp_path))

        assert module._compile_active is True
        compiler.assert_called_once()

    @pytest.mark.parametrize("compile_enabled", [False, True])
    def test_matcher_l1_compilation_follows_model_flag(self, tmp_path: Path, compile_enabled: bool) -> None:
        """Compile only the real matcher's tensor cost, without changing other matcher instances."""
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=5,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
        )
        model = _fake_model()
        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("torch._inductor.config", SimpleNamespace(triton=SimpleNamespace(coalesce_tiling_analysis=True))),
            patch("rfdetr.training.module_model.build_model_from_config", return_value=model),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(criterion, _fake_postprocess()),
            ),
            patch.object(HungarianMatcher, "enable_compiled_l1_cost", autospec=True) as enabler,
            patch("rfdetr.training.module_model.torch.compile", side_effect=lambda value, **_: value) as compiler,
        ):
            module = RFDETRModelModule(
                _base_model_config(compile=compile_enabled, cuda_graphs=True), _base_train_config(tmp_path)
            )

        assert module.criterion.matcher is matcher
        assert type(matcher).forward is HungarianMatcher.forward
        if not compile_enabled:
            enabler.assert_not_called()
            compiler.assert_not_called()
            return
        # Construction compiles only the model; the matcher's cost compiles lazily on its first use, from the
        # same Inductor options the model got (the matcher itself forces CUDA graph replay off on top of them).
        compiler.assert_called_once()
        assert compiler.call_args.args == (model,)
        model_options = compiler.call_args.kwargs["options"]
        assert model_options == {"triton.coalesce_tiling_analysis": False, "triton.cudagraphs": True}
        enabler.assert_called_once_with(matcher, model_options)

    def test_compiled_matcher_stays_picklable_for_spawn_launchers(self, tmp_path: Path) -> None:
        """The module's matcher pickles before and after its compiled cost is built in this process.

        ``ddp_spawn``/``ddp_notebook`` pickle the trainer -- criterion and matcher included -- into every worker. The
        ``torch.compile`` double returns a local closure, which ``pickle`` rejects (negative control below), so this
        passes only when the matcher stores the compile request and builds the callable lazily per process.
        """

        def unpicklable_compile(fn, **_):
            def compiled(*args):
                return fn(*args)

            return compiled

        with pytest.raises((pickle.PicklingError, AttributeError)):
            pickle.dumps(unpicklable_compile(pairwise_box_l1_cost))

        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=5,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
        )
        # The double is live during construction too: an eager compile in __init__ would already leave the
        # unpicklable closure on the matcher and fail the first pickle below.
        with (
            patch("rfdetr.config.DEVICE", "cuda"),
            patch("rfdetr.training.module_model.build_model_from_config", return_value=_fake_model()),
            patch(
                "rfdetr.training.module_model.build_criterion_from_config",
                return_value=(criterion, _fake_postprocess()),
            ),
            patch(
                "rfdetr.training.module_model.torch.compile",
                side_effect=lambda value, **_: unpicklable_compile(value) if value is pairwise_box_l1_cost else value,
            ),
        ):
            module = RFDETRModelModule(_base_model_config(compile=True), _base_train_config(tmp_path))

        torch.manual_seed(0)
        outputs = {"pred_logits": torch.randn(2, 4, 5), "pred_boxes": torch.rand(2, 4, 4) * 0.4 + 0.3}
        targets = [
            {"labels": torch.tensor([1, 3]), "boxes": torch.rand(2, 4) * 0.4 + 0.3},
            {"labels": torch.tensor([0]), "boxes": torch.rand(1, 4) * 0.4 + 0.3},
        ]
        expected = HungarianMatcher()(outputs, targets)

        # Pickled before any use: the spawn launcher's moment.
        spawned_matcher = pickle.loads(pickle.dumps(module.criterion.matcher))
        with patch("rfdetr.training.module_model.torch.compile", side_effect=unpicklable_compile) as compiler:
            actual = module.criterion.matcher(outputs, targets)
            spawned = spawned_matcher(outputs, targets)
        assert compiler.call_count == 2, "each process builds its own compiled callable on first use"
        for (actual_i, actual_j), (expected_i, expected_j) in zip((*actual, *spawned), (*expected, *expected)):
            assert torch.equal(actual_i, expected_i)
            assert torch.equal(actual_j, expected_j)

        # Pickled after use: the compiled callable is a per-process cache, never state.
        pickle.dumps(module.criterion.matcher)

    @pytest.mark.parametrize("knob_supported", [True, False])
    def test_coalesce_tiling_knob_passed_only_when_torch_exposes_it(self, mocker, knob_supported, tmp_path):
        """The Inductor workaround reaches torch.compile only on a torch whose config exposes the knob.

        Older torch versions have no ``triton.coalesce_tiling_analysis``; passing it there raises
        ``RuntimeError("Unexpected optimization option ...")`` from ``_TorchCompileInductorWrapper.apply_options``.
        Compilation must still proceed in both cases, and the matcher's compile request must carry the same knob
        decision as the model's.
        """
        triton_config = SimpleNamespace(coalesce_tiling_analysis=True) if knob_supported else SimpleNamespace()
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        matcher = HungarianMatcher()
        criterion = SetCriterion(
            num_classes=5,
            matcher=matcher,
            weight_dict={"loss_ce": 1.0, "loss_bbox": 5.0, "loss_giou": 2.0},
            focal_alpha=0.25,
            losses=["labels", "boxes", "cardinality"],
        )
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("torch._inductor.config", SimpleNamespace(triton=triton_config))
        mocker.patch("rfdetr.training.module_model.build_model_from_config", return_value=_fake_model())
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config",
            return_value=(criterion, _fake_postprocess()),
        )
        enabler = mocker.patch.object(HungarianMatcher, "enable_compiled_l1_cost", autospec=True)
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        RFDETRModelModule(mc, tc)

        expected = {"triton.coalesce_tiling_analysis": False} if knob_supported else {}
        assert mock_compile.call_args.kwargs["options"] == (expected or None)
        enabler.assert_called_once_with(matcher, expected)

    def test_coalesce_tiling_knob_leaves_process_global_config_untouched(self, mocker, tmp_path):
        """The workaround must not disable coalesce tiling for unrelated compilations in the same process."""
        triton_config = SimpleNamespace(coalesce_tiling_analysis=True)
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("torch._inductor.config", SimpleNamespace(triton=triton_config))
        mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        assert triton_config.coalesce_tiling_analysis is True

    def test_cuda_graphs_with_compile_requests_inductor_cudagraphs(self, mocker, tmp_path):
        """``cuda_graphs=True`` together with ``compile=True`` turns on Inductor's CUDA graph trees.

        ``torch.compile(mode="reduce-overhead")`` cannot be used here because ``mode`` and ``options`` are mutually
        exclusive in ``torch.compile``; the equivalent option key is what must reach the compiler alongside the existing
        coalesce-tiling workaround.
        """
        triton_config = SimpleNamespace(coalesce_tiling_analysis=True)
        mc = _base_model_config(compile=True, cuda_graphs=True)
        tc = _base_train_config(tmp_path, multi_scale=False)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("torch._inductor.config", SimpleNamespace(triton=triton_config))
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        module, _, _, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        assert mock_compile.call_args.kwargs["options"] == {
            "triton.coalesce_tiling_analysis": False,
            "triton.cudagraphs": True,
        }
        assert module._inductor_cudagraphs is True

    @pytest.mark.parametrize(
        ("model_overrides", "train_overrides"),
        [
            pytest.param({}, {"grad_accum_steps": 2}, id="gradient-accumulation"),
            pytest.param({}, {"devices": 2}, id="multi-device"),
            pytest.param({}, {"num_nodes": 2}, id="multi-node"),
            pytest.param({}, {"amp_dtype": "fp8"}, id="fp8-needs-te-capture"),
            pytest.param({"segmentation_head": True}, {}, id="segmentation"),
            pytest.param({"use_grouppose_keypoints": True, "num_keypoints_per_class": [1]}, {}, id="keypoints"),
            pytest.param({"gradient_checkpointing": True}, {}, id="gradient-checkpointing"),
        ],
    )
    def test_cuda_graphs_with_compile_falls_back_to_plain_compile_outside_validated_scope(
        self, mocker, model_overrides, train_overrides, tmp_path, caplog, monkeypatch
    ):
        """Outside the single-GPU detection scope the combined request degrades to compile-only with a warning.

        Inductor CUDA graph trees allocate gradient outputs inside the graph pool, so a ``.grad`` adopted by the first
        microbatch is overwritten by the next replay — gradient accumulation cannot run on this path. Model modes and
        distributed layouts mirror the limits of the eager graph runner.
        """
        mc = _base_model_config(compile=True, cuda_graphs=True, **model_overrides)
        tc = _base_train_config(tmp_path, multi_scale=False, **train_overrides)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        mocker.patch("torch._inductor.config", SimpleNamespace(triton=SimpleNamespace()))
        mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile", side_effect=lambda m, **_: m)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            module, _, _, _ = _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)

        assert mock_compile.call_args.kwargs["options"] is None
        assert module._inductor_cudagraphs is False
        assert any("compile-only" in record.getMessage() for record in caplog.records)

    def test_compile_disabled_when_train_accelerator_is_cpu(self, mocker, tmp_path):
        """Compile stays disabled when training is explicitly forced to CPU."""
        mocker.patch("rfdetr.config.DEVICE", "cuda")
        _mock_compile = mocker.patch("rfdetr.training.module_model.torch.compile")
        mc = _base_model_config(compile=True)
        tc = _base_train_config(tmp_path, multi_scale=False, accelerator="cpu")
        _build_module(model_config=mc, train_config=tc, tmp_path=tmp_path)
        _mock_compile.assert_not_called()


class TestLoadPretrainWeights:
    """Tests for _load_pretrain_weights().

    Covers checkpoint validation, detection-head reinitialization on class-count mismatch, query-embedding trimming, re-
    download on corruption, and class-name extraction from checkpoint metadata.
    """

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

    def test_loads_valid_checkpoint(self, mocker, base_model_config, build_module):
        """A valid checkpoint must be validated, loaded, and applied to the model."""
        mock_validate = mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
        mc = base_model_config(num_classes=90)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        mock_validate.assert_called_once_with("/fake/weights.pth", strict=False)
        module.model.load_state_dict.assert_called_once()

    def test_class_count_mismatch_triggers_reinitialize(self, mocker, base_model_config, build_module):
        """Detection head is expanded to checkpoint size, then trimmed back to config size."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        # First call: expand to checkpoint size so load_state_dict shapes match.
        # Second call: trim back to configured num_classes + 1 (background class).
        fake_model.reinitialize_detection_head.assert_has_calls([mocker.call(91), mocker.call(6)])
        assert fake_model.reinitialize_detection_head.call_count == 2

    def test_class_count_match_does_not_reinitialize(self, mocker, base_model_config, build_module):
        """Detection head must NOT be reinitialized when class counts match."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
        mc = base_model_config(num_classes=5)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=6)
        mock_torch_load.return_value = checkpoint

        module, fake_model, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})
        load_pretrain_weights(module.model, module.model_config)

        fake_model.reinitialize_detection_head.assert_not_called()

    def test_query_embedding_trimmed_to_configured_count(self, mocker, base_model_config, build_module):
        """Oversized query embeddings in checkpoint must be trimmed to match config."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
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

    def test_load_failure_does_not_redownload_existing_weights(self, mocker, base_model_config, build_module):
        """A checkpoint read failure propagates without replacing the existing file."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_download = mocker.patch("rfdetr.models.weights.download_pretrain_weights")
        mocker.patch("rfdetr.models.weights.os.path.isfile", return_value=True)
        mc = base_model_config(num_classes=90)
        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(update={"pretrain_weights": "/fake/weights.pth"})

        mock_load = mocker.patch("rfdetr.utilities.io._safe_torch_load", side_effect=RuntimeError("corrupted file"))
        with pytest.raises(RuntimeError, match="corrupted file"):
            load_pretrain_weights(module.model, module.model_config)

        mock_download.assert_called_once_with("/fake/weights.pth")
        mock_load.assert_called_once()

    def test_download_before_load_when_weights_absent(self, mocker, base_model_config, build_module):
        """download_pretrain_weights is called before torch.load when weights are absent.

        A fresh environment (e.g. Colab) then downloads weights automatically.

        Regression test: previously download was only called as an except-block fallback, but ModelWeights.from_filename
        received the absolute path and returned None, causing a silent no-op and a FileNotFoundError.
        """
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_download = mocker.patch("rfdetr.models.weights.download_pretrain_weights")
        mocker.patch("rfdetr.models.weights.os.path.isfile", return_value=False)
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

    @pytest.mark.parametrize(
        "ckpt_segmentation_head, ckpt_patch_size, module_overrides, expected_match",
        [
            pytest.param(
                True, 12, {"segmentation_head": False}, "segmentation head", id="seg-checkpoint-into-detection-model"
            ),
            pytest.param(
                False, 16, {"segmentation_head": True}, "segmentation head", id="detection-checkpoint-into-seg-model"
            ),
            pytest.param(
                False,
                12,
                {"segmentation_head": False, "patch_size": 16},
                "patch_size",
                id="patch-size-mismatch",
            ),
        ],
    )
    def test_incompatible_checkpoint_raises(
        self,
        mocker,
        base_model_config,
        build_module,
        ckpt_segmentation_head: bool,
        ckpt_patch_size: int,
        module_overrides: dict[str, object],
        expected_match: str,
    ):
        """Loading a checkpoint whose segmentation_head/patch_size mismatch the model must raise ValueError."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
        mc = base_model_config(num_classes=90)
        ckpt_args = SimpleNamespace(segmentation_head=ckpt_segmentation_head, patch_size=ckpt_patch_size)
        checkpoint = self._make_checkpoint(num_classes_in_ckpt=91)
        checkpoint["args"] = ckpt_args
        mock_torch_load.return_value = checkpoint

        module, _, _, _ = build_module(model_config=mc)
        module.model_config = module.model_config.model_copy(
            update={"pretrain_weights": "/fake/weights.pth", **module_overrides}
        )

        with pytest.raises(ValueError) as exc_info:
            load_pretrain_weights(module.model, module.model_config)
        assert expected_match in str(exc_info.value)

    def test_compatible_checkpoint_does_not_raise(self, mocker, base_model_config, build_module):
        """A checkpoint matching segmentation_head and patch_size must load without error."""
        mocker.patch("rfdetr.models.weights.validate_pretrain_weights")
        mock_torch_load = mocker.patch("rfdetr.models.weights.torch.load")
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
    """Tests for _apply_lora().

    Verifies that PEFT LoraConfig is constructed with the correct target modules and that the backbone encoder is
    replaced in-place with the wrapped PEFT model.
    """

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

    def test_calls_lora_config_with_correct_target_modules(self, mocker, tmp_path):
        """LoRA must target the expected attention and token projection modules."""
        mock_lora_cfg_class = mocker.patch("peft.LoraConfig")
        mock_get_peft = mocker.patch("peft.get_peft_model")
        module, _, _, _ = self._build_module_with_backbone(tmp_path)
        mock_get_peft.return_value = MagicMock()

        apply_lora(module.model)

        mock_lora_cfg_class.assert_called_once()
        target_modules = mock_lora_cfg_class.call_args.kwargs.get("target_modules")
        expected = ["q_proj", "v_proj", "k_proj", "qkv", "query", "key", "value", "cls_token", "register_tokens"]
        assert target_modules == expected

    def test_replaces_encoder_with_peft_model(self, mocker, tmp_path):
        """The backbone encoder must be replaced in-place with the PEFT-wrapped model."""
        mocker.patch("peft.LoraConfig")
        mock_get_peft = mocker.patch("peft.get_peft_model")
        module, _, fake_backbone_0, fake_encoder = self._build_module_with_backbone(tmp_path)
        peft_wrapped = MagicMock()
        mock_get_peft.return_value = peft_wrapped

        apply_lora(module.model)

        assert mock_get_peft.call_args[0][0] is fake_encoder
        assert fake_backbone_0.encoder is peft_wrapped


class TestReinitializeDetectionHead:
    """Tests for reinitialize_detection_head().

    Verifies that the module delegates to the underlying model and that arbitrary class counts are forwarded unchanged.
    """

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
    """Tests for on_load_checkpoint().

    Covers legacy .pth normalisation and positional-embedding interpolation for custom-resolution PTL checkpoints.

    Regression: issue #998 — resume with custom resolution crashed because
    on_load_checkpoint did not interpolate PE before PTL applied the state dict.
    """

    _PE_KEY = "model.backbone.0.encoder.encoder.embeddings.position_embeddings"

    @pytest.mark.parametrize(
        ("optimizer_states", "expected_lr_encoder"),
        [
            pytest.param([{"state": {}, "param_groups": []}], 0.123, id="full-checkpoint"),
            pytest.param([], TrainConfig.model_fields["lr_encoder"].default, id="lightweight-checkpoint"),
        ],
    )
    def test_resolves_resumed_settings_without_a_trainer(
        self, optimizer_states: list[dict[str, Any]], expected_lr_encoder: float, tmp_path: Path
    ) -> None:
        """``load_from_checkpoint`` calls the hook before any trainer exists.

        Only a checkpoint carrying optimizer state supplies the settings left out (#1613); a lightweight one restarts
        the optimizer from the config.
        """
        module, *_ = _build_module(train_config=TrainConfig(dataset_dir=str(tmp_path)), tmp_path=tmp_path)

        module.on_load_checkpoint({"optimizer_states": optimizer_states, "args": {"lr_encoder": 0.123}})

        assert module.train_config.lr_encoder == expected_lr_encoder

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

    def test_state_dict_without_pe_or_extra_state_keys_is_unchanged(self, build_module):
        """on_load_checkpoint neither raises nor alters keys when state_dict has no PE and no `_extra_state` entries."""
        checkpoint = {
            "state_dict": {"model.some_layer.weight": torch.randn(4, 4)},
            "epoch": 1,
        }
        original_keys = set(checkpoint["state_dict"].keys())

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=36))
        module.on_load_checkpoint(checkpoint)

        assert set(checkpoint["state_dict"].keys()) == original_keys

    def test_extra_state_keys_stripped_from_state_dict(self, build_module):
        """on_load_checkpoint must remove `_extra_state` entries before PTL applies the state dict.

        Under FP8, the live module's checkpointed state_dict carries Transformer Engine `_extra_state` entries recording
        FP8 scaling history. `strict_loading=False` only tolerates their presence/absence after the fact — it does not
        stop `load_state_dict()` from calling `set_extra_state()` for a key present in both the checkpoint and the
        module, which Transformer Engine rejects on a pickle round-trip. The entries must therefore be excluded from
        `checkpoint["state_dict"]` here, before PTL ever applies it.
        """
        checkpoint = {
            "state_dict": {
                "model.some_layer.weight": torch.randn(4, 4),
                "model.some_layer._extra_state": b"fp8-scaling-history",
            },
            "epoch": 1,
        }

        module, _, _, _ = build_module(model_config=_base_model_config(positional_encoding_size=36))
        module.on_load_checkpoint(checkpoint)

        assert "model.some_layer._extra_state" not in checkpoint["state_dict"]
        assert "model.some_layer.weight" in checkpoint["state_dict"]


_STEM_KEY = "model.backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"


def _real_module(tmp_path: Path, **model_overrides: Any) -> RFDETRModelModule:
    """Build a real CPU Nano module without pretrained weights, applying *model_overrides* to its config.

    Examples:
        >>> _real_module(Path("."), num_channels=4)  # doctest: +SKIP
        RFDETRModelModule(...)
    """
    model_config = RFDETRNanoConfig(**{"pretrain_weights": None, "device": "cpu", **model_overrides})
    train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)
    return RFDETRModelModule(model_config, train_config)


class TestStemChannels:
    """A multi-channel module warns about a frozen stem and refuses a checkpoint with a narrower stem."""

    @pytest.mark.parametrize(
        ("num_channels", "freeze_encoder", "expected_warnings"),
        [(4, True, 1), (4, False, 0), (3, True, 0)],
    )
    def test_frozen_stem_warning(
        self, num_channels: int, freeze_encoder: bool, expected_warnings: int, tmp_path: Path, mocker
    ) -> None:
        """Only a frozen stem of a multi-channel model is reported, and the text makes no RGB-initialization claim."""
        warning = mocker.patch("rfdetr.training.module_model.logger.warning")

        _real_module(tmp_path, num_channels=num_channels, freeze_encoder=freeze_encoder)

        frozen = [call for call in warning.call_args_list if "patch embedding is frozen" in call.args[0]]
        assert len(frozen) == expected_warnings
        assert all("RGB" not in call.args[0] for call in frozen)

    def test_frozen_stem_warning_covers_a_widened_checkpoint(self, tmp_path: Path, mocker) -> None:
        """A checkpoint that already carries the wide stem loads trained filters, and the warning still fires once."""
        source = _real_module(tmp_path, num_channels=4)
        checkpoint = tmp_path / "wide.pth"
        torch.save({"model": source.model.state_dict()}, checkpoint)
        warning = mocker.patch("rfdetr.training.module_model.logger.warning")

        _real_module(tmp_path, num_channels=4, freeze_encoder=True, pretrain_weights=str(checkpoint))

        frozen = [call for call in warning.call_args_list if "patch embedding is frozen" in call.args[0]]
        assert len(frozen) == 1

    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_resume_refuses_a_three_channel_stem(self, num_channels: int, tmp_path: Path) -> None:
        """A checkpoint written before the stem was widened names the fresh-run workaround, not a size mismatch."""
        module = _real_module(tmp_path, num_channels=num_channels)
        narrow_stem = torch.zeros(8, 3, 2, 2)

        with pytest.raises(ValueError, match="pretrain_weights=<this checkpoint>"):
            module.on_load_checkpoint({"state_dict": {_STEM_KEY: narrow_stem}})

    def test_resume_accepts_a_matching_stem(self, tmp_path: Path) -> None:
        """A checkpoint saved by the same multi-channel module passes the stem check."""
        module = _real_module(tmp_path, num_channels=4)
        checkpoint = {"state_dict": {_STEM_KEY: module.state_dict()[_STEM_KEY]}}

        module.on_load_checkpoint(checkpoint)

        assert checkpoint["state_dict"][_STEM_KEY].shape[1] == 4

    def test_resume_of_a_three_channel_module_is_unchanged(self, tmp_path: Path) -> None:
        """A 3-channel module resumes a 3-channel checkpoint without the stem check intervening."""
        module = _real_module(tmp_path)

        module.on_load_checkpoint({"state_dict": {_STEM_KEY: module.state_dict()[_STEM_KEY]}})


class TestLoadEncoderWeights:
    """``load_encoder_weights`` maps onto the builder's ``force_no_pretrain`` default."""

    @pytest.mark.parametrize(("load_encoder_weights", "force_no_pretrain"), [(True, False), (False, True)])
    def test_flag_reaches_builder(self, load_encoder_weights: bool, force_no_pretrain: bool, tmp_path, mocker) -> None:
        """The module hands the inverted flag to ``build_model_from_config`` through its defaults."""
        model_config = RFDETRNanoConfig(pretrain_weights=None, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path))
        build = mocker.patch("rfdetr.training.module_model.build_model_from_config")
        mocker.patch(
            "rfdetr.training.module_model.build_criterion_from_config", return_value=(mocker.Mock(), mocker.Mock())
        )

        RFDETRModelModule(model_config, train_config, load_encoder_weights=load_encoder_weights)

        assert build.call_args.kwargs["defaults"].force_no_pretrain is force_no_pretrain
