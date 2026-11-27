# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Unit tests for GroupPose keypoint output wiring in LWDETR."""

from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor, nn

from rfdetr.models.criterion import SetCriterion
from rfdetr.models.heads import ConditionalQueryInitializer
from rfdetr.models.lwdetr import LWDETR
from rfdetr.models.transformer import Transformer
from rfdetr.utilities.tensors import NestedTensor

_HIDDEN_DIM = 8


def _build_feature_batch(batch_size: int, hidden_dim: int) -> list[NestedTensor]:
    """Build feature maps consumed by keypoint LWDETR tests.

    Examples:
        >>> len(_build_feature_batch(batch_size=1, hidden_dim=8))
        1
    """
    return [
        NestedTensor(
            torch.zeros(batch_size, hidden_dim, 4, 4),
            torch.zeros(batch_size, 4, 4, dtype=torch.bool),
        )
    ]


def _mock_backbone(batch_size: int, hidden_dim: int) -> MagicMock:
    """Return a mocked backbone that yields zero feature maps and positional encodings.

    Examples:
        >>> features, poss, _ = _mock_backbone(batch_size=1, hidden_dim=8)(None)
        >>> len(features), len(poss)
        (1, 1)
    """
    backbone = MagicMock()
    backbone.return_value = (
        _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim),
        [torch.zeros(batch_size, hidden_dim, 4, 4)],
        None,
    )
    return backbone


def _keypoint_transformer_outputs(
    batch_size: int, num_queries: int, hidden_dim: int, ref_unsigmoid: Tensor | None = None
) -> tuple[Tensor, ...]:
    """Build the zero-valued 17-keypoint tuple a GroupPose transformer returns to ``LWDETR``.

    Args:
        batch_size: Batch dimension of every tensor.
        num_queries: Number of object queries.
        hidden_dim: Transformer hidden size.
        ref_unsigmoid: Optional replacement for the decoder reference points, e.g. one that requires grad.

    Examples:
        >>> outputs = _keypoint_transformer_outputs(batch_size=2, num_queries=3, hidden_dim=8)
        >>> len(outputs), tuple(outputs[5].shape)
        (7, (2, 3, 17, 8))
    """
    return (
        torch.zeros(2, batch_size, num_queries, hidden_dim),  # hs
        torch.zeros(2, batch_size, num_queries, 4) if ref_unsigmoid is None else ref_unsigmoid,  # ref_unsigmoid
        torch.zeros(batch_size, num_queries, hidden_dim),  # hs_enc
        torch.zeros(batch_size, num_queries, 4),  # ref_enc
        torch.zeros(2, batch_size, num_queries, 17, hidden_dim),  # keypoint_hs
        torch.zeros(batch_size, num_queries, 17, 8),  # enc_kp_predictions
        torch.zeros(batch_size, num_queries, 17, hidden_dim),  # unused keypoint encoder hidden state
    )


class _DummyKeypointDecoder(nn.Module):
    """Minimal decoder surface needed for keypoint schema resizing."""

    def __init__(self, hidden_dim: int, num_keypoints_per_class: list[int]) -> None:
        super().__init__()
        self.num_keypoints_per_class = num_keypoints_per_class
        self.keypoint_pos_embed = nn.Parameter(torch.randn(sum(num_keypoints_per_class), hidden_dim))
        self.register_buffer(
            "keypoint_class_mask",
            torch.zeros(1 + sum(num_keypoints_per_class), 1 + sum(num_keypoints_per_class), dtype=torch.bool),
        )


class _DummyKeypointTransformer(nn.Module):
    """Minimal transformer surface needed for LWDETR construction and keypoint schema resizing."""

    def __init__(self, hidden_dim: int, num_keypoints_per_class: list[int]) -> None:
        super().__init__()
        self.d_model = hidden_dim
        self.num_keypoints_per_class = num_keypoints_per_class
        self.decoder = _DummyKeypointDecoder(hidden_dim, num_keypoints_per_class)
        self.keypoint_query_initializer = ConditionalQueryInitializer(hidden_dim, sum(num_keypoints_per_class))
        self.keypoint_query_initializer_enc = ConditionalQueryInitializer(hidden_dim, sum(num_keypoints_per_class))


def test_keypoint_forward_outputs() -> None:
    """GroupPose mode should expose keypoint tensors in model outputs."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 8
    num_classes = 6

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = _keypoint_transformer_outputs(batch_size, num_queries, hidden_dim)

    model = LWDETR(
        backbone=_mock_backbone(batch_size, hidden_dim),
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=True,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )

    outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
    assert outputs["pred_keypoints"].shape == (batch_size, num_queries, 17, 8)
    assert "keypoint_hidden_states" not in outputs
    assert "pred_keypoints" in outputs["aux_outputs"][0]
    assert "keypoint_hidden_states" not in outputs["aux_outputs"][0]


def test_nan_keypoint_delta_keeps_ref_gradient_finite() -> None:
    """A non-finite keypoint-embed output must not poison ``ref_unsigmoid``'s gradient.

    Regression: ``keypoints_xy = outputs_keypoints_delta[..., :2] * ref_wh + ref_xy`` combines the
    keypoint head's raw delta with the shared box reference point *before* ``compute_l1_keypoint_loss``
    ever sees the result. That function's own ``torch.where`` guards can zero the gradient flowing
    *into* this multiply, but not the multiply's own local backward (``d(a*b)/d(b) == a``): a
    non-finite delta still poisons ``ref_wh``'s (and therefore ``ref_unsigmoid``'s, shared with the box
    head) gradient via ``0.0 * nan == nan``. Fixed by ``nan_to_num``-ing ``outputs_keypoints_delta`` at
    the source, before the multiply.
    """
    batch_size = 2
    num_queries = 3
    hidden_dim = 8
    num_classes = 6

    ref_unsigmoid = torch.zeros(2, batch_size, num_queries, 4, requires_grad=True)
    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = _keypoint_transformer_outputs(batch_size, num_queries, hidden_dim, ref_unsigmoid)

    model = LWDETR(
        backbone=_mock_backbone(batch_size, hidden_dim),
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )

    def _inject_nan(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        poisoned = output.clone()
        with torch.no_grad():
            poisoned[0, 0, 0, 0] = float("nan")
        return poisoned

    model.keypoint_embed.register_forward_hook(_inject_nan)

    outputs = model(torch.ones(batch_size, 3, 8, 8))
    assert torch.isfinite(outputs["pred_keypoints"]).all(), (
        "the injected NaN must be sanitized before it reaches ref_wh/ref_xy composition, "
        f"got pred_keypoints={outputs['pred_keypoints']}"
    )
    outputs["pred_keypoints"].sum().backward()

    assert ref_unsigmoid.grad is not None
    assert torch.isfinite(ref_unsigmoid.grad).all(), (
        f"a single non-finite keypoint-embed output must not poison ref_unsigmoid's gradient, "
        f"got grad={ref_unsigmoid.grad}"
    )


@pytest.mark.parametrize(
    ("bbox_reparam", "nonfinite_channel", "nonfinite_delta"),
    [
        pytest.param(False, 0, float("nan"), id="sigmoid_box_nan_xy"),
        pytest.param(False, 0, float("inf"), id="sigmoid_box_positive_infinity_xy"),
        pytest.param(False, 0, float("-inf"), id="sigmoid_box_negative_infinity_xy"),
        pytest.param(True, 0, float("nan"), id="reparam_box_nan_xy"),
        pytest.param(True, 2, float("nan"), id="reparam_box_nan_findable"),
    ],
)
def test_two_stage_encoder_nonfinite_keypoint_delta_stays_finite(
    bbox_reparam: bool,
    nonfinite_channel: int,
    nonfinite_delta: float,
) -> None:
    """Encoder keypoint deltas must be sanitized before sharing the box reference composition.

    The default two-stage encoder combines a raw keypoint delta with the shared
    reference box before passing it to the criterion. A non-finite delta must become zero at that model
    boundary: finite delta channels remain unchanged, the corresponding coordinate stays at the reference
    center, and backward through the real keypoint criterion keeps both encoder-head gradients finite. The
    full prediction is sanitized here because encoder outputs also feed outer classification and matching
    consumers.
    """
    torch.manual_seed(0)
    hidden_dim = 8
    transformer = Transformer(
        d_model=hidden_dim,
        sa_nhead=2,
        ca_nhead=2,
        num_queries=2,
        num_decoder_layers=0,
        dim_feedforward=16,
        num_feature_levels=1,
        dec_n_points=1,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[2],
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 1)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    finite_delta = torch.tensor(
        [
            [
                [
                    [0.25, -0.50, 0.10, -0.20, 0.30, -0.40, 0.50, -0.60],
                    [0.75, 0.25, -0.10, 0.20, -0.30, 0.40, -0.50, 0.60],
                ],
                [
                    [-0.25, 0.50, 0.20, -0.10, 0.40, -0.30, 0.60, -0.50],
                    [0.50, -0.75, -0.20, 0.10, -0.40, 0.30, -0.60, 0.50],
                ],
            ]
        ]
    )

    def inject_nonfinite_delta(_module: nn.Module, _inputs: tuple, output: torch.Tensor) -> torch.Tensor:
        """Return known encoder deltas with one non-finite channel."""
        delta = output + (finite_delta.to(device=output.device, dtype=output.dtype) - output).detach()
        nonfinite_mask = torch.zeros_like(delta, dtype=torch.bool)
        nonfinite_mask[0, 0, 0, nonfinite_channel] = True
        return torch.where(nonfinite_mask, torch.full_like(delta, nonfinite_delta), delta)

    transformer.enc_out_keypoint_embed[0].register_forward_hook(inject_nonfinite_delta)
    outputs = transformer(
        [torch.randn(1, hidden_dim, 2, 2)],
        [torch.zeros(1, 2, 2, dtype=torch.bool)],
        [torch.zeros(1, hidden_dim, 2, 2)],
        torch.rand(2, 4),
        torch.randn(2, hidden_dim),
    )
    _, _, _, ref_enc, _, enc_keypoints, _, _ = outputs
    assert enc_keypoints is not None

    expected_delta = finite_delta.clone()
    expected_delta[0, 0, 0, nonfinite_channel] = 0.0
    expected_xy = expected_delta[..., :2] * ref_enc[..., 2:].unsqueeze(-2) + ref_enc[..., :2].unsqueeze(-2)
    expected_keypoints = torch.cat([expected_xy, expected_delta[..., 2:]], dim=-1)
    torch.testing.assert_close(enc_keypoints, expected_keypoints)

    criterion = SetCriterion(
        num_classes=1,
        matcher=MagicMock(return_value=[(torch.tensor([0]), torch.tensor([0]))]),
        weight_dict={},
        focal_alpha=0.25,
        losses=["keypoints"],
        num_keypoints_per_class=[2],
    )
    losses = criterion(
        {"pred_keypoints": enc_keypoints},
        [
            {
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.5, 0.5, 0.4, 0.4]]),
                "keypoints": torch.tensor([[[0.2, 0.3, 2.0], [0.4, 0.5, 2.0]]]),
            }
        ],
    )
    assert all(torch.isfinite(loss) for loss in losses.values())
    sum(losses.values()).backward()

    box_head_grad = transformer.enc_out_bbox_embed[0].weight.grad
    assert box_head_grad is not None
    assert torch.isfinite(box_head_grad).all()

    keypoint_head_grad = transformer.enc_out_keypoint_embed[0].layers[-1].weight.grad
    assert keypoint_head_grad is not None
    assert torch.isfinite(keypoint_head_grad).all()
    assert torch.count_nonzero(keypoint_head_grad) > 0


@pytest.fixture
def dummy_keypoint_model() -> LWDETR:
    """A 17-keypoint ``LWDETR`` wrapping a minimal dummy transformer, used by the schema-dependent tests."""
    return LWDETR(
        backbone=MagicMock(),
        transformer=_DummyKeypointTransformer(hidden_dim=_HIDDEN_DIM, num_keypoints_per_class=[17]),
        segmentation_head=None,
        num_classes=3,
        num_queries=2,
        aux_loss=False,
        group_detr=1,
        two_stage=True,
        lite_refpoint_refine=True,
        bbox_reparam=False,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[17],
        grouppose_keypoint_dim_downscale=1,
    )


def test_reinitialize_keypoint_head_resizes_schema_state(dummy_keypoint_model: LWDETR) -> None:
    """Keypoint schema reinit should resize masks and learned keypoint query embeddings."""
    model = dummy_keypoint_model
    transformer = model.transformer
    hidden_dim = _HIDDEN_DIM

    model.reinitialize_keypoint_head([2, 1])

    assert model.num_keypoints_per_class == [2, 1]
    assert model.get_num_keypoints_per_class() == [2, 1]
    assert model._kp_active_mask.shape == (2, 2)
    assert model._kp_active_mask.tolist() == [[True, True], [True, False]]
    assert transformer.num_keypoints_per_class == [2, 1]
    assert transformer.decoder.num_keypoints_per_class == [2, 1]
    assert transformer.decoder.keypoint_pos_embed.shape == (3, hidden_dim)
    assert transformer.decoder.keypoint_class_mask.shape == (4, 4)
    assert transformer.keypoint_query_initializer.queries.shape == (3, hidden_dim)
    assert transformer.keypoint_query_initializer_enc.queries.shape == (3, hidden_dim)


def test_reset_gaussian_parameters_zeroes_only_precision_rows(dummy_keypoint_model: LWDETR) -> None:
    """Gaussian reset should only zero precision-Cholesky output rows on decoder and encoder keypoint heads."""
    model = dummy_keypoint_model
    hidden_dim = _HIDDEN_DIM
    with torch.no_grad():
        model.keypoint_embed.layers[-1].weight.fill_(3.0)
        model.keypoint_embed.layers[-1].bias.fill_(4.0)
        model.transformer.enc_out_keypoint_embed[0].layers[-1].weight.fill_(5.0)
        model.transformer.enc_out_keypoint_embed[0].layers[-1].bias.fill_(6.0)

    model.reset_keypoint_gaussian_parameters()

    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[:4], torch.full((4, hidden_dim), 3.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[4:7], torch.zeros(3, hidden_dim))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].weight[7:], torch.full((1, hidden_dim), 3.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[:4], torch.full((4,), 4.0))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[4:7], torch.zeros(3))
    torch.testing.assert_close(model.keypoint_embed.layers[-1].bias[7:], torch.full((1,), 4.0))
    torch.testing.assert_close(
        model.transformer.enc_out_keypoint_embed[0].layers[-1].weight[4:7], torch.zeros(3, hidden_dim)
    )
    torch.testing.assert_close(model.transformer.enc_out_keypoint_embed[0].layers[-1].bias[4:7], torch.zeros(3))


def test_num_keypoints_recovered_from_checkpoint() -> None:
    """Checkpoint keypoint schema should be recoverable from `_kp_active_mask`."""
    state_dict = {"_kp_active_mask": torch.tensor([[True, True], [True, False]])}

    assert LWDETR.get_num_keypoints_per_class_from_checkpoint(state_dict) == [2, 1]


def test_default_detection_omits_keypoint_outputs() -> None:
    """Default detection mode should not expose keypoint outputs."""
    batch_size = 2
    num_queries = 3
    hidden_dim = 8
    num_classes = 6

    transformer = MagicMock()
    transformer.d_model = hidden_dim
    transformer.return_value = (
        torch.zeros(1, batch_size, num_queries, hidden_dim),
        torch.zeros(1, batch_size, num_queries, 4),
        torch.zeros(batch_size, num_queries, hidden_dim),
        torch.zeros(batch_size, num_queries, 4),
    )

    model = LWDETR(
        backbone=_mock_backbone(batch_size, hidden_dim),
        transformer=transformer,
        segmentation_head=None,
        num_classes=num_classes,
        num_queries=num_queries,
        aux_loss=False,
        group_detr=1,
        two_stage=False,
        lite_refpoint_refine=False,
        bbox_reparam=False,
        use_grouppose_keypoints=False,
        num_keypoints_per_class=[],
        grouppose_keypoint_dim_downscale=1,
    )

    outputs = model(torch.ones(batch_size, 3, 8, 8))

    assert outputs["pred_logits"].shape == (batch_size, num_queries, num_classes)
    assert outputs["pred_boxes"].shape == (batch_size, num_queries, 4)
    assert "pred_keypoints" not in outputs
    assert "keypoint_hidden_states" not in outputs


class TestForwardExportKeypointDecode:
    """``forward_export`` must decode keypoints exactly as the training-path ``forward`` does.

    ``forward_export`` decodes keypoint xy on a flattened ``(..., K * 2)`` layout instead of broadcasting the box
    reference over the keypoint axis, because onnx2tf cannot convert that rank-4 broadcast (#1514).
    """

    @pytest.mark.parametrize(
        "num_keypoints_per_class",
        [
            pytest.param([1], id="one_keypoint"),
            pytest.param([4], id="four_keypoints"),
            pytest.param([17], id="coco_person"),
            pytest.param([3, 2], id="two_classes_padded"),
        ],
    )
    def test_keypoints_match_forward(self, num_keypoints_per_class: list[int]) -> None:
        """For finite keypoint deltas, the flattened export decode equals the broadcast decode, per keypoint schema."""
        torch.manual_seed(0)
        batch_size = 2
        num_queries = 3
        hidden_dim = 8
        num_compact_keypoints = sum(num_keypoints_per_class)

        features = _build_feature_batch(batch_size=batch_size, hidden_dim=hidden_dim)
        poss = [torch.zeros(batch_size, hidden_dim, 4, 4)]
        hs = torch.randn(batch_size, num_queries, hidden_dim)
        ref_unsigmoid = torch.rand(batch_size, num_queries, 4)
        keypoint_hs = torch.randn(batch_size, num_queries, num_compact_keypoints, hidden_dim)
        hs_enc = torch.zeros(batch_size, num_queries, hidden_dim)
        ref_enc = torch.zeros(batch_size, num_queries, 4)
        enc_kp_predictions = torch.zeros(batch_size, num_queries, num_compact_keypoints, 8)

        backbone = MagicMock()
        transformer = MagicMock()
        transformer.d_model = hidden_dim
        model = LWDETR(
            backbone=backbone,
            transformer=transformer,
            segmentation_head=None,
            num_classes=len(num_keypoints_per_class) + 1,
            num_queries=num_queries,
            aux_loss=False,
            group_detr=1,
            two_stage=False,
            lite_refpoint_refine=False,
            bbox_reparam=True,
            use_grouppose_keypoints=True,
            num_keypoints_per_class=num_keypoints_per_class,
            grouppose_keypoint_dim_downscale=1,
        ).eval()
        # The keypoint head is zero-initialized; randomize it so the decode sees non-trivial deltas.
        for layer in model.keypoint_embed.layers:
            nn.init.normal_(layer.weight)
            nn.init.normal_(layer.bias)

        backbone.return_value = (features, poss, None)
        transformer.return_value = (
            hs.unsqueeze(0),
            ref_unsigmoid.unsqueeze(0),
            hs_enc,
            ref_enc,
            keypoint_hs.unsqueeze(0),
            enc_kp_predictions,
            torch.zeros_like(keypoint_hs),
        )
        with torch.no_grad():
            expected = model(torch.ones(batch_size, 3, 8, 8))["pred_keypoints"]

        backbone.return_value = ([features[0].tensors], None, poss, None)
        transformer.return_value = (
            hs,
            ref_unsigmoid,
            hs_enc,
            ref_enc,
            keypoint_hs,
            enc_kp_predictions,
            torch.zeros_like(keypoint_hs),
        )
        with torch.no_grad():
            _, _, actual = model.forward_export(torch.ones(batch_size, 3, 8, 8))

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
