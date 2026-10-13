# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""``validation_step`` and ``test_step`` get every layer's segmentation masks only when they compute a loss."""

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.matcher import HungarianMatcher
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.tensors import NestedTensor
from tests.models.test_lwdetr_segmentation_eval import _BATCH, _CLASSES, _IMAGE, _QUERIES, _build_seg_lwdetr

from .test_module_model import _base_model_config, _base_train_config, _build_module

_MASK_LOSS_WEIGHTS = {"loss_mask_ce": 1.0, "loss_mask_dice": 1.0}


def _seg_module(tmp_path: Path | None, **train_overrides: Any) -> RFDETRModelModule:
    """Build a module around a real tiny segmentation ``LWDETR`` and a real mask criterion; postprocess stays fake.

    Args:
        tmp_path: Output directory for the train config, or ``None``.
        **train_overrides: Train config fields to override.

    Returns:
        The module, with its model in eval mode.

    Examples:
        >>> module = _seg_module(None)
        >>> isinstance(module.criterion, SetCriterion)
        True
    """
    module, *_ = _build_module(
        model_config=_base_model_config(segmentation_head=True),
        train_config=_base_train_config(tmp_path, **train_overrides),
    )
    module.model = _build_seg_lwdetr().eval()
    module.criterion = SetCriterion(
        num_classes=_CLASSES,
        matcher=HungarianMatcher(),
        weight_dict=_MASK_LOSS_WEIGHTS,
        focal_alpha=0.25,
        losses=["masks"],
        group_detr=1,
        mask_point_sample_ratio=16,
    )
    module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
    module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]
    return module


def _seg_batch() -> tuple[NestedTensor, list[dict[str, Tensor]]]:
    """Build a batch matching ``_build_seg_lwdetr``'s input size, with one mask target per image.

    Returns:
        The padded images and one target dict per image.

    Examples:
        >>> samples, targets = _seg_batch()
        >>> tuple(samples.tensors.shape), sorted(targets[0])
        ((2, 3, 8, 8), ['boxes', 'labels', 'masks', 'orig_size'])
    """
    samples = NestedTensor(
        torch.zeros(_BATCH, 3, _IMAGE, _IMAGE), torch.zeros(_BATCH, _IMAGE, _IMAGE, dtype=torch.bool)
    )
    generator = torch.Generator().manual_seed(0)
    targets = [
        {
            "labels": torch.tensor([1]),
            "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
            "masks": (torch.rand(1, _IMAGE, _IMAGE, generator=generator) > 0.5).float(),
            "orig_size": torch.tensor([_IMAGE, _IMAGE]),
        }
        for _ in range(_BATCH)
    ]
    return samples, targets


class _ImagesOnlyDetector(torch.nn.Module):
    """A detection model stand-in whose forward takes the images alone, as a registered backbone's model may.

    Examples:
        >>> tuple(_ImagesOnlyDetector()(torch.zeros(1, 3, 8, 8))["pred_logits"].shape)
        (2, 3, 5)
    """

    def forward(self, samples: Any) -> dict[str, Tensor]:
        """Return fixed logits for a batch of ``_BATCH`` images, whatever the input."""
        return {"pred_logits": torch.zeros(_BATCH, _QUERIES, _CLASSES)}


class TestEvalStepSegmentationMasks:
    """The loss path keeps every layer's mask loss; the metric-only path skips the masks nothing reads."""

    @pytest.mark.parametrize(
        ("step", "overrides"),
        [
            pytest.param("validation_step", {"compute_val_loss": True}, id="validation"),
            pytest.param("test_step", {"compute_test_loss": True}, id="test"),
        ],
    )
    def test_loss_includes_every_layers_mask_loss(self, tmp_path: Path, step: str, overrides: dict[str, Any]) -> None:
        """The criterion gets the auxiliary-layer and encoder-proposal masks it computes the mask losses of."""
        module = _seg_module(tmp_path, **overrides)
        loss_dicts: list[dict[str, Tensor]] = []
        module.criterion.register_forward_hook(lambda _module, _inputs, output: loss_dicts.append(output))

        with torch.no_grad():
            getattr(module, step)(_seg_batch(), batch_idx=0)

        assert len(loss_dicts) == 1, f"{step} must run the criterion once when its loss is enabled"
        expected = {"loss_mask_ce", "loss_mask_ce_0", "loss_mask_ce_1", "loss_mask_ce_enc"}
        assert expected <= set(loss_dicts[0]), f"missing mask losses: {sorted(expected - set(loss_dicts[0]))}"

    @pytest.mark.parametrize(
        ("step", "overrides"),
        [
            pytest.param("validation_step", {"compute_val_loss": False}, id="validation"),
            pytest.param("test_step", {"compute_test_loss": False}, id="test"),
        ],
    )
    def test_step_without_loss_skips_auxiliary_masks(
        self, tmp_path: Path, step: str, overrides: dict[str, Any]
    ) -> None:
        """With the loss off, only the final layer's masks are computed and handed to postprocess."""
        module = _seg_module(tmp_path, **overrides)

        with torch.no_grad():
            getattr(module, step)(_seg_batch(), batch_idx=0)

        outputs = module.postprocess.call_args[0][0]
        assert outputs["pred_masks"].shape == (_BATCH, _QUERIES, _IMAGE, _IMAGE)
        assert all("pred_masks" not in aux for aux in outputs["aux_outputs"]), "aux masks computed without a loss"
        assert "pred_masks" not in outputs["enc_outputs"], "encoder masks computed without a loss"

    @pytest.mark.parametrize("segmentation_config", [False, True])
    def test_model_without_a_segmentation_head_is_called_with_the_images_alone(
        self, tmp_path: Path, segmentation_config: bool
    ) -> None:
        """A model without a segmentation head gets no ``all_layer_masks`` keyword with a loss, whatever the config.

        Detection and keypoint models are called as before, and so is a stand-in whose forward takes the images alone.
        """
        module, *_ = _build_module(
            model_config=_base_model_config(segmentation_head=segmentation_config),
            train_config=_base_train_config(tmp_path, compute_val_loss=True),
        )
        module.log = lambda *args, **kwargs: None  # type: ignore[method-assign]
        module.log_dict = lambda *args, **kwargs: None  # type: ignore[method-assign]
        module.model = _ImagesOnlyDetector()

        result = module.validation_step(_seg_batch(), batch_idx=0)

        assert len(result["results"]) == 1, "postprocess output must reach the callback"

    def test_criterion_on_default_eval_outputs_names_the_keyword(self, tmp_path: Path) -> None:
        """A mask loss on eval outputs without the auxiliary masks raises a ``ValueError`` that says how to get them."""
        module = _seg_module(tmp_path)
        samples, targets = _seg_batch()
        with torch.no_grad():
            outputs = module.model(samples)

        with pytest.raises(ValueError, match="all_layer_masks=True"):
            module.criterion(outputs, targets)

    def test_criterion_without_mask_targets_takes_default_eval_outputs(self, tmp_path: Path) -> None:
        """Targets without masks need no auxiliary masks, so default eval outputs go through every layer's loss."""
        module = _seg_module(tmp_path)
        module.criterion = SetCriterion(
            num_classes=_CLASSES,
            matcher=HungarianMatcher(),
            weight_dict={"loss_ce": 1.0},
            focal_alpha=0.25,
            losses=["labels"],
            group_detr=1,
        )
        samples, targets = _seg_batch()
        box_targets = [{key: value for key, value in target.items() if key != "masks"} for target in targets]
        with torch.no_grad():
            outputs = module.model(samples)

        assert "loss_ce_enc" in module.criterion(outputs, box_targets)
