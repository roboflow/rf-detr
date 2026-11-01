# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for which segmentation masks an eval-mode ``LWDETR`` forward computes.

Only ``SetCriterion`` reads the auxiliary-layer and encoder-proposal masks, so eval mode computes the last decoder
layer's masks alone unless ``all_layer_masks=True`` asks for the rest.
"""

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from rfdetr.models.heads.segmentation import SegmentationHead
from rfdetr.models.lwdetr import LWDETR
from rfdetr.utilities.tensors import NestedTensor

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
