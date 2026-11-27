# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the transformer's two-stage query selection: top-k gather, group selection and its batching eligibility,
keypoint two-stage, dynamic compile and the CUDA-graph spatial-shapes cache."""

import copy
from collections.abc import Callable

import pytest
import torch
from pytest_mock import MockerFixture
from torch import nn

from rfdetr._namespace import _namespace_from_configs
from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.models.lwdetr import build_model
from rfdetr.models.math import MLP
from rfdetr.models.transformer import (
    Transformer,
    _CastThenExpand,
    gen_encoder_output_proposals,
)
from rfdetr.utilities.tensors import NestedTensor
from tests._markers import _requires_cuda
from tests.models._transformer_support import single_level_two_stage_transformer


@pytest.fixture(autouse=True)
def _reset_random_seeds() -> None:
    """Ensure reproducible random state for every test."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)


@pytest.fixture
def gather_calls(mocker: MockerFixture) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Record ``(input, index)`` of every ``torch.gather`` call made while the test runs.

    This is a pytest fixture (it needs ``mocker``), so its doctest cannot run standalone.

    Examples:
        >>> gather_calls  # doctest: +SKIP
    """
    calls: list[tuple[torch.Tensor, torch.Tensor]] = []
    original_gather = torch.gather

    def _tracking_gather(input: torch.Tensor, dim: int, index: torch.Tensor, **kwargs: object) -> torch.Tensor:
        calls.append((input, index))
        return original_gather(input, dim, index, **kwargs)

    mocker.patch.object(torch, "gather", side_effect=_tracking_gather)
    return calls


def _assert_gather_indices_broadcast(calls: list[tuple[torch.Tensor, torch.Tensor]]) -> None:
    """Assert ``calls`` is non-empty and every gather index broadcasts its last dim via ``Tensor.expand``.

    ``Tensor.repeat``, ``tile``, ``expand(...).contiguous()`` and ``repeat_interleave`` all re-materialise the index
    into a nonzero-stride tensor, so one stride check catches every one of those regressions.

    Examples:
        >>> _assert_gather_indices_broadcast([(torch.zeros(2, 3, 4), torch.zeros(2, 3, 1).expand(2, 3, 4))])
    """
    assert calls, "expected torch.gather to be called during the two-stage top-k selection"
    for _, index in calls:
        assert index.stride(-1) == 0, (
            "every two-stage top-k gather index must broadcast its last dim via Tensor.expand (Transformer.forward "
            f"two-stage top-k gather); got a materialised index with nonzero last-dim stride {index.stride()} "
            f"for shape {tuple(index.shape)}"
        )


class _FixedTopkScores(nn.Module):
    """Returns pre-set per-position scores regardless of its input.

    Stubs ``enc_out_class_embed`` so the two-stage top-k selection in ``Transformer.forward`` picks known positions in a
    known, deliberately out-of-position-order rank.
    """

    def __init__(self, scores: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("scores", scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Ignore ``x`` and return the fixed scores."""
        return self.scores


@pytest.mark.parametrize(
    ("bbox_reparam", "finalize_boxes"),
    [pytest.param(False, torch.sigmoid, id="plain-sigmoid"), pytest.param(True, torch.clone, id="reparam-raw")],
)
def test_two_stage_topk_gather_selects_correct_rows_out_of_position_order(
    gather_calls, bbox_reparam: bool, finalize_boxes: Callable[[torch.Tensor], torch.Tensor]
) -> None:
    """The two-stage top-k gather must copy each selected proposal's exact memory row and box.

    ``torch.topk`` ranks proposals by score, not by position, so the selected indices are rarely in ascending position
    order. This pins that ``memory_ts``/``boxes_ts`` reproduce the source rows picked by an out-of-order, per-batch-row-
    distinct selection, and additionally asserts that every ``torch.gather`` index used by the two-stage selection is a
    broadcast view produced by ``Tensor.expand`` (its broadcast dim keeps stride 0), not a materialised copy.

    ``bbox_reparam=False`` adds the box delta to the unsigmoided proposals and ``Transformer.forward`` returns the
    sigmoid of the result; ``bbox_reparam=True`` scales a cx/cy delta by the proposal size, adds a log-space w/h delta
    and returns the boxes as they are. Neither computation shares code with the other, so each is covered.
    """
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries = 2, 16, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(batch_size, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(batch_size, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(batch_size, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        group_detr=1,
    )

    # Deliberately out-of-position-order and batch-row-distinct top-3 picks.
    scores = torch.full((batch_size, total_hw, 1), -100.0)
    scores[0, 17, 0], scores[0, 2, 0], scores[0, 9, 0] = 30.0, 20.0, 10.0
    scores[1, 5, 0], scores[1, 19, 0], scores[1, 0, 0] = 25.0, 15.0, 5.0
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(scores)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )

    _assert_gather_indices_broadcast(gather_calls)

    # Ground truth computed independently of the gather under test: the same flatten/proposal
    # machinery Transformer.forward uses internally, then plain (non-gather) row indexing.
    memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
    mask_flatten = torch.cat([m.flatten(1) for m in masks], 1)
    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, mask_flatten, spatial_shapes_hw, unsigmoid=not bbox_reparam
    )
    output_memory_gidx = transformer.enc_output_norm[0](transformer.enc_output[0](output_memory))
    coord_unselected = _bbox_from_delta(
        transformer.enc_out_bbox_embed[0](output_memory_gidx), output_proposals, bbox_reparam
    )
    chosen_idx = scores.squeeze(-1).topk(num_queries, dim=1).indices  # mirrors forward()'s torch.topk call

    assert torch.equal(chosen_idx, torch.tensor([[17, 2, 9], [5, 19, 0]]))  # sanity: out of position order
    expected_memory = torch.stack([output_memory_gidx[b, chosen_idx[b]] for b in range(batch_size)])
    # forward() returns boxes_ts.sigmoid() when bbox_reparam=False and boxes_ts as-is when bbox_reparam=True.
    expected_coord = finalize_boxes(torch.stack([coord_unselected[b, chosen_idx[b]] for b in range(batch_size)]))
    assert torch.equal(memory_ts, expected_memory)
    assert torch.equal(boxes_ts, expected_coord)


def test_two_stage_topk_gather_reads_pre_norm_memory(gather_calls) -> None:
    """The eval-path two-stage gather must read ``enc_output``'s output, never ``enc_output_norm``'s.

    Regression for an Apple Neural Engine compile failure: in an fp16 CoreML export, an ``enc_output_norm``
    output that feeds the ANE-resident class head and also crosses to the CPU-resident ``topk``-indexed gather
    makes the ANE compiler reject the whole program ("Invalid layer") when that norm still has its identity
    affine and the encoder token count is a multiple of 32. The model then runs entirely on CPU, or fails to
    load with ``ComputeUnit.ALL``. Trained checkpoints have a non-identity affine and are unaffected, so this
    reaches models exported before training — the export tests among them. Gathering the pre-norm rows and
    normalizing only the selected tokens is exact, because LayerNorm acts per token; the selected rows
    themselves are pinned by ``test_two_stage_topk_gather_selects_correct_rows_out_of_position_order``.
    """
    hidden_dim, num_queries, feature_size = 16, 3, 4
    num_tokens = feature_size * feature_size
    srcs = [torch.randn(1, hidden_dim, feature_size, feature_size)]
    masks = [torch.zeros(1, feature_size, feature_size, dtype=torch.bool)]
    pos_embeds = [torch.randn(1, hidden_dim, feature_size, feature_size)]
    transformer = single_level_two_stage_transformer(hidden_dim, num_queries)

    norm_outputs: list[torch.Tensor] = []
    transformer.enc_output_norm[0].register_forward_hook(lambda _m, _i, out: norm_outputs.append(out))
    transformer.eval()
    with torch.no_grad():
        transformer(srcs, masks, pos_embeds, torch.rand(num_queries, 4), torch.randn(num_queries, hidden_dim))

    full_length_norm_outputs = [out for out in norm_outputs if out.shape[1] == num_tokens]
    assert full_length_norm_outputs, "expected enc_output_norm to run over every encoder position"
    assert gather_calls, "expected torch.gather to be called during the two-stage top-k selection"
    norm_storages = {out.untyped_storage().data_ptr() for out in full_length_norm_outputs}
    # Compare storage, not identity, so a view of the norm output (reshape/slice/expand) is caught as well.
    assert not any(inp.untyped_storage().data_ptr() in norm_storages for inp, _ in gather_calls), (
        "the two-stage gather reads enc_output_norm's output; gather the pre-norm enc_output rows and "
        "normalize the selected tokens instead (fp16 CoreML ANE compile failure)"
    )


def _make_out_of_order_scores(total_hw: int, picks: list[int]) -> torch.Tensor:
    """Build batch=1 per-position class scores with `picks` as the strictly descending top-k winners.

    Args:
        total_hw: Total number of flattened spatial positions across all feature levels.
        picks: Position indices to rank first, second, third, ... in descending score order.

    Returns:
        Score tensor of shape (1, total_hw, 1); every position not in `picks` scores -100.0.

    Examples:
        >>> _make_out_of_order_scores(5, [3, 1]).squeeze(-1).tolist()
        [[-100.0, 20.0, -100.0, 30.0, -100.0]]
    """
    scores = torch.full((1, total_hw, 1), -100.0)
    picks_tensor = torch.tensor(picks)
    scores[0, picks_tensor, 0] = 30.0 - 10.0 * torch.arange(len(picks), dtype=torch.float32)
    return scores


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_topk_gather_broadcasts_across_groups_in_training_mode(gather_calls, bbox_reparam: bool) -> None:
    """With group_detr>1 in training mode, each group's gather index broadcasts via expand and its top-k rows match.

    Regression: test_two_stage_topk_gather_selects_correct_rows_out_of_position_order only exercises group_detr=1,
    where the ``group_detr = self.group_detr if self.training else 1`` guard in Transformer.forward degenerates to a
    single gather per stage. This pins the group_detr>1 branch, which loops the same two gathers once per extra group
    and concatenates the results along the query dimension. The module never calls .eval(): group_detr>1 only takes
    effect while nn.Module.training is True (its default).
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 3, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(1, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(1, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    assert transformer.training  # default nn.Module state; group_detr>1 only takes effect while training

    # Deliberately out-of-position-order, distinct picks per group.
    picks_per_group = [[17, 2, 9], [5, 19, 0], [11, 14, 3]]
    scores_per_group = [_make_out_of_order_scores(total_hw, picks) for picks in picks_per_group]
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(scores) for scores in scores_per_group])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )

    # Two gather calls (refpoint, memory) per group -- the only torch.gather call sites in the two-stage selection.
    assert len(gather_calls) == 2 * group_detr
    _assert_gather_indices_broadcast(gather_calls)

    # Ground truth computed independently of the gather under test, per group.
    memory = torch.cat([src.flatten(2).transpose(1, 2) for src in srcs], 1)
    mask_flatten = torch.cat([m.flatten(1) for m in masks], 1)
    output_memory, output_proposals = gen_encoder_output_proposals(
        memory, mask_flatten, spatial_shapes_hw, unsigmoid=not bbox_reparam
    )
    picks_tensors = [torch.tensor(picks) for picks in picks_per_group]
    output_memory_per_group = [
        transformer.enc_output_norm[g](transformer.enc_output[g](output_memory)) for g in range(group_detr)
    ]
    if bbox_reparam:
        coord_unselected_per_group = []
        for g in range(group_detr):
            delta = transformer.enc_out_bbox_embed[g](output_memory_per_group[g])
            coord_unselected_per_group.append(
                torch.cat(
                    [
                        delta[..., :2] * output_proposals[..., 2:] + output_proposals[..., :2],
                        delta[..., 2:].exp() * output_proposals[..., 2:],
                    ],
                    dim=-1,
                )
            )
    else:
        coord_unselected_per_group = [
            transformer.enc_out_bbox_embed[g](output_memory_per_group[g]) + output_proposals for g in range(group_detr)
        ]

    expected_memory = torch.cat(
        [output_memory_per_group[g][0, picks_tensors[g]] for g in range(group_detr)], dim=0
    ).unsqueeze(0)
    # forward() returns boxes_ts.sigmoid() only when bbox_reparam=False.
    expected_coord = torch.cat(
        [coord_unselected_per_group[g][0, picks_tensors[g]] for g in range(group_detr)], dim=0
    ).unsqueeze(0)
    if not bbox_reparam:
        expected_coord = expected_coord.sigmoid()
    torch.testing.assert_close(memory_ts, expected_memory, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(boxes_ts, expected_coord, atol=1e-5, rtol=1e-4)

    parameters = [parameter for module in transformer.enc_out_bbox_embed for parameter in module.parameters()]
    new_gradients = torch.autograd.grad(boxes_ts.sum(), [*srcs, *parameters], retain_graph=True)
    reference_gradients = torch.autograd.grad(expected_coord.sum(), [*srcs, *parameters])
    for new_gradient, reference_gradient in zip(new_gradients, reference_gradients, strict=True):
        torch.testing.assert_close(new_gradient, reference_gradient, atol=1e-5, rtol=1e-4)


def _build_two_stage_transformer_with_production_shaped_heads(
    hidden_dim: int,
    num_queries: int,
    group_detr: int,
    num_classes: int,
    bbox_reparam: bool,
    num_feature_levels: int = 2,
) -> Transformer:
    """Build a two-stage ``Transformer`` whose group heads are the concrete types LWDETR constructs.

    ``Transformer.__init__`` already builds ``enc_output``/``enc_output_norm`` as real
    ``nn.Linear``/``nn.LayerNorm``. This additionally assigns real ``nn.Linear``/``MLP`` instances to
    ``enc_out_class_embed``/``enc_out_bbox_embed`` (which start ``None`` and are set externally by
    ``LWDETR`` in production), matching the layout ``Transformer._two_stage_batching_eligible`` requires
    for the batched fast path.

    Args:
        hidden_dim: Model width.
        num_queries: Queries selected per group.
        group_detr: Number of independent groups.
        num_classes: Class-head output width.
        bbox_reparam: Whether the transformer uses the reparameterised box-delta path.
        num_feature_levels: Number of feature-map levels the decoder's deformable attention expects; the caller's
            ``srcs``/``masks``/``pos_embeds`` lists must have this many entries.

    Returns:
        A two-stage ``Transformer`` left in its default training mode.

    Examples:
        >>> t = _build_two_stage_transformer_with_production_shaped_heads(16, 3, 2, 5, False)
        >>> t._two_stage_batching_eligible()
        True
    """
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=num_feature_levels,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )
    return transformer


def test_two_stage_batching_eligible_true_for_production_shaped_two_stage_modules() -> None:
    """The batched fast path's eligibility guard must accept the exact layout LWDETR constructs."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=4, num_classes=5, bbox_reparam=False
    )
    assert transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_true_for_real_build_model_rfdetr_nano() -> None:
    """The fast path activates on RFDETRNano assembled through the real production constructor.

    ``_build_two_stage_transformer_with_production_shaped_heads`` manually assigns the head types
    ``LWDETR.__init__`` constructs; this test instead calls ``build_model()`` -- the same function
    ``RFDETRNano`` uses -- so the eligibility guard is proven against the actual shipped wiring,
    not a hand-built stand-in.
    """
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    model = build_model(ns)
    assert model.group_detr == 13
    assert model.transformer._two_stage_batching_eligible()


def test_two_stage_group_selection_class_logits_reuse_matches_recomputed_loop() -> None:
    """Reused batched-path class logits match the per-group recompute loop in ``enc_outputs["pred_logits"]``.

    ``LWDETR.forward`` reuses the batched path's gathered ``class_logits_all`` instead of the per-group
    ``enc_out_class_embed`` recompute loop. The head is a per-position ``nn.Linear``, so the two differ only in GEMM
    accumulation order (stacked-groups batched matmul vs. one matmul per group) and are compared within float32
    tolerance.

    Unlike the ``memory_ts``/``boxes_ts`` comparisons, the class head's parameters receive real gradient through the
    reused value. Their gradients are compared with the recompute loop's, because a same-path-only non-vacuity check
    would pass a gather that silently detached ``class_logits_all`` or indexed the wrong positions.
    """
    torch.manual_seed(0)
    ns = _namespace_from_configs(
        RFDETRNanoConfig(num_classes=7, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
    )
    model_fast = build_model(ns)
    assert model_fast.transformer._two_stage_batching_eligible()
    state = copy.deepcopy(model_fast.state_dict())

    model_loop = build_model(ns)
    model_loop.load_state_dict(state)
    model_loop.transformer._two_stage_batching_eligible = lambda: False
    assert not model_loop.transformer._two_stage_batching_eligible()

    model_fast.train()
    model_loop.train()

    samples = NestedTensor(torch.randn(1, 3, 256, 256), torch.zeros(1, 256, 256, dtype=torch.bool))
    out_fast = model_fast(samples)
    out_loop = model_loop(samples)

    torch.testing.assert_close(
        out_fast["enc_outputs"]["pred_logits"], out_loop["enc_outputs"]["pred_logits"], atol=1e-4, rtol=1e-4
    )

    fast_class_embed_parameters = [p for m in model_fast.transformer.enc_out_class_embed for p in m.parameters()]
    loop_class_embed_parameters = [p for m in model_loop.transformer.enc_out_class_embed for p in m.parameters()]
    fast_gradients = torch.autograd.grad(out_fast["enc_outputs"]["pred_logits"].sum(), fast_class_embed_parameters)
    loop_gradients = torch.autograd.grad(out_loop["enc_outputs"]["pred_logits"].sum(), loop_class_embed_parameters)
    for fast_gradient, loop_gradient in zip(fast_gradients, loop_gradients, strict=True):
        assert torch.isfinite(fast_gradient).all()
        assert fast_gradient.abs().sum() > 0
        torch.testing.assert_close(fast_gradient, loop_gradient, atol=1e-4, rtol=1e-4)


class _DoublingLinear(nn.Linear):
    """``nn.Linear`` whose overridden ``forward`` doubles the output, so a skipped call changes the result."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return twice the linear output."""
        return super().forward(x) * 2


@pytest.mark.parametrize(
    ("attribute", "make_module"),
    [
        pytest.param("enc_output", lambda: _DoublingLinear(16, 16), id="linear-subclass"),
        pytest.param("enc_output", lambda: nn.Linear(16, 16, bias=False), id="bias-free-linear"),
        pytest.param("enc_output_norm", lambda: nn.LayerNorm(16, elementwise_affine=False), id="affine-free-norm"),
        pytest.param("enc_output_norm", lambda: nn.LayerNorm((20, 16)), id="multi-axis-norm"),
    ],
)
def test_two_stage_batching_eligible_false_for_unsupported_group_module(
    attribute: str, make_module: Callable[[], nn.Module]
) -> None:
    """Group modules the fast path cannot stack or whose ``forward`` it would skip fall back to the generic loop.

    The fast path reads ``.weight``/``.bias`` directly and never calls the module, so a ``Linear`` subclass would lose
    its ``forward`` override if the guard accepted it. A bias-free ``Linear`` or an affine-free ``LayerNorm`` has
    ``None`` parameters that would raise inside ``torch.stack`` instead of falling back, and a multi-axis ``LayerNorm``
    needs the generic path even when every group has the same shape.
    """
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    setattr(transformer, attribute, nn.ModuleList([make_module() for _ in range(2)]))

    assert not transformer._two_stage_batching_eligible()


@pytest.mark.parametrize(
    "mismatch",
    [
        "group-count",
        "layer-norm-eps",
        "layer-norm-dtype",
        "class-head-shape",
        "class-head-dtype",
        "bbox-depth",
        "bbox-layer-shape",
    ],
)
def test_two_stage_batching_eligible_false_for_heterogeneous_group_modules(mismatch: str) -> None:
    """Groups that cannot share one stacked operation must fall back to their individual module calls."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )

    if mismatch == "group-count":
        transformer.enc_output = nn.ModuleList([transformer.enc_output[0]])
    elif mismatch == "layer-norm-eps":
        transformer.enc_output_norm[1].eps = 0.1
    elif mismatch == "layer-norm-dtype":
        transformer.enc_output_norm[1].double()
    elif mismatch == "class-head-shape":
        transformer.enc_out_class_embed[1] = nn.Linear(16, 6)
    elif mismatch == "class-head-dtype":
        transformer.enc_out_class_embed[1].double()
    elif mismatch == "bbox-depth":
        transformer.enc_out_bbox_embed[1] = MLP(16, 16, 4, num_layers=2)
    else:
        transformer.enc_out_bbox_embed[1] = MLP(16, 12, 4, num_layers=3)

    assert not transformer._two_stage_batching_eligible()


@pytest.mark.parametrize(
    ("submodule", "register"),
    [
        pytest.param("enc_output.0", "register_forward_hook", id="encoder-hook"),
        pytest.param("enc_output_norm.0", "register_forward_pre_hook", id="norm-hook"),
        pytest.param("enc_out_class_embed.0", "register_full_backward_hook", id="class-hook"),
        pytest.param("enc_out_bbox_embed.0", "register_full_backward_pre_hook", id="bbox-hook"),
        pytest.param("enc_out_bbox_embed.0.layers.0", "register_forward_hook", id="bbox-layer-hook"),
    ],
)
def test_two_stage_batching_eligible_false_for_hooked_group_module(
    submodule: str, register: str, mocker: MockerFixture
) -> None:
    """A hook on any module the fast path reads parameters from keeps the generic module-calling path."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    getattr(transformer.get_submodule(submodule), register)(mocker.Mock(return_value=None))

    assert not transformer._two_stage_batching_eligible()


@pytest.mark.parametrize(
    "attribute",
    [pytest.param("forward", id="instance-forward"), pytest.param("_compiled_call_impl", id="compiled-child")],
)
def test_two_stage_batching_eligible_false_for_instance_call_override(attribute: str, mocker: MockerFixture) -> None:
    """A per-instance ``forward`` or compiled call on a group module keeps the generic module-calling path."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    encoder = transformer.enc_output[0]
    setattr(encoder, attribute, mocker.Mock(side_effect=encoder.forward))

    assert not transformer._two_stage_batching_eligible()


def test_two_stage_batching_eligible_false_for_global_forward_hook(
    mocker: MockerFixture, request: pytest.FixtureRequest
) -> None:
    """A global module forward hook keeps the generic module-calling path."""
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
    )
    handle = nn.modules.module.register_module_forward_hook(mocker.Mock(return_value=None))
    request.addfinalizer(handle.remove)

    assert not transformer._two_stage_batching_eligible()


def test_two_stage_forward_preserves_hooked_child_module_call(mocker: MockerFixture) -> None:
    """The public forward route must execute child hooks through the generic fallback."""
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    hook = mocker.Mock(return_value=None)
    transformer.enc_output[0].register_forward_hook(hook)
    selection_mock = mocker.patch.object(
        transformer, "_two_stage_group_selection", side_effect=transformer._two_stage_group_selection
    )

    spatial_shapes_hw = [(4, 4), (2, 2)]
    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert hook.call_count == 1
    assert selection_mock.call_count == 0


def test_two_stage_group_selection_dispatches_through_forward(mocker: MockerFixture) -> None:
    """`Transformer.forward` must invoke the batched fast path exactly once when eligible."""
    hidden_dim, num_queries, group_detr = 16, 3, 4
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    assert transformer._two_stage_batching_eligible()

    selection_mock = mocker.patch.object(
        transformer, "_two_stage_group_selection", side_effect=transformer._two_stage_group_selection
    )

    spatial_shapes_hw = [(4, 4), (2, 2)]
    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert selection_mock.call_count == 1


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_group_selection_matches_generic_loop_forward_and_gradient(bbox_reparam: bool) -> None:
    """The batched fast path (group_detr>1) must reproduce the generic per-group loop's fp32 outputs and gradients.

    The same transformer runs twice from identical inputs: through the normal routing (the fast path, since
    ``_two_stage_batching_eligible()`` is True here) and with that guard forced to False. In fp32 every fast-path op
    (batched matmul, one shared-then-affine LayerNorm, batched topk/gather) is an algebraic restatement of the loop's,
    so forward and backward match to float32 rounding noise from the different reduction order. A tolerance replaces
    ``torch.equal`` because larger GEMMs may choose another accumulation order; see
    ``Transformer._two_stage_group_selection`` for the mixed-precision limitation.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 5, 4
    spatial_shapes_hw = [(4, 4), (2, 2)]

    srcs = [torch.randn(2, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=bbox_reparam
    )
    assert transformer._two_stage_batching_eligible()

    _, _, memory_fast, boxes_fast, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
    fast_loss = memory_fast.sum() + boxes_fast.sum()
    # enc_out_class_embed is excluded: its output only ranks torch.topk's (non-differentiable)
    # selection indices, so it structurally never receives a gradient from memory_ts/boxes_ts,
    # in the loop and the batched path alike.
    fast_modules = [*transformer.enc_output, *transformer.enc_output_norm, *transformer.enc_out_bbox_embed]
    fast_parameters = [parameter for module in fast_modules for parameter in module.parameters()]
    fast_gradients = torch.autograd.grad(fast_loss, [*srcs, *fast_parameters])

    transformer_loop = copy.deepcopy(transformer)
    transformer_loop._two_stage_batching_eligible = lambda: False
    srcs_loop = [src.detach().clone().requires_grad_(True) for src in srcs]
    _, _, memory_loop, boxes_loop, _ = transformer_loop(
        srcs_loop, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
    loop_loss = memory_loop.sum() + boxes_loop.sum()
    loop_modules = [
        *transformer_loop.enc_output,
        *transformer_loop.enc_output_norm,
        *transformer_loop.enc_out_bbox_embed,
    ]
    loop_parameters = [parameter for module in loop_modules for parameter in module.parameters()]
    loop_gradients = torch.autograd.grad(loop_loss, [*srcs_loop, *loop_parameters])

    # A batched GEMM can select a different accumulation order than separate calls, so compare
    # within float32 tolerance rather than requiring bit-for-bit equality.
    torch.testing.assert_close(memory_fast, memory_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(boxes_fast, boxes_loop, atol=1e-4, rtol=1e-4)
    for fast_gradient, loop_gradient in zip(fast_gradients, loop_gradients, strict=True):
        torch.testing.assert_close(fast_gradient, loop_gradient, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@_requires_cuda
def test_two_stage_group_selection_bf16_produces_finite_valid_selection_with_gradients() -> None:
    """Under bf16 autocast the batched fast path yields finite outputs and gradients reaching every group.

    The outputs need not be bit-identical to the generic loop.

    ``Transformer._two_stage_group_selection`` documents that ``torch.baddbmm``'s batched-GEMM kernel can
    accumulate in a different order than the loop's separate GEMM calls, so a near-tied ``torch.topk``
    ranking can legitimately pick a different, equally valid query under bf16. A hard numeric-parity bound
    would be flaky across GPU architectures and problem sizes depending on whether that specific run
    happens to hit a tie, so this test instead pins the invariant that must ALWAYS hold regardless of
    which tie-break wins: finite values and gradients reaching every parameter.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 32, 20, 13
    spatial_shapes_hw = [(16, 16), (8, 8)]  # matches num_feature_levels=2 in the shared builder
    device = "cuda"

    srcs = [torch.randn(2, hidden_dim, ht, wd, device=device, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool, device=device) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd, device=device) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4, device=device)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim, device=device)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=90, bbox_reparam=True
    ).to(device)
    assert transformer._two_stage_batching_eligible()

    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, _, memory_ts, boxes_ts, cls_ts = transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
        )
        loss = memory_ts.float().sum() + boxes_ts.float().sum() + cls_ts.float().sum()

    assert torch.isfinite(memory_ts).all()
    assert torch.isfinite(boxes_ts).all()
    # cls_ts is enc_out_class_embed's output gathered by the batched path (module-level docstring
    # above), the same value the class-logits-reuse mechanism forwards as enc_outputs["pred_logits"]
    # instead of recomputing -- the mechanism this benchmark's BF16 training step actually exercises.
    assert torch.isfinite(cls_ts).all()

    # Only modules _two_stage_group_selection uses for a differentiable output (the decoder gets no gradient from this
    # loss). cls_ts is part of the loss, so enc_out_class_embed is included: the first check to route gradient through
    # the reused gather.
    two_stage_modules = [
        *transformer.enc_output,
        *transformer.enc_output_norm,
        *transformer.enc_out_bbox_embed,
        *transformer.enc_out_class_embed,
    ]
    named_parameters = [
        (name, parameter)
        for module in two_stage_modules
        for name, parameter in module.named_parameters()
        if parameter.requires_grad
    ]
    inputs = [*srcs, *[parameter for _, parameter in named_parameters]]
    names = [f"srcs[{i}]" for i in range(len(srcs))] + [name for name, _ in named_parameters]
    gradients = torch.autograd.grad(loss, inputs)
    for name, gradient in zip(names, gradients, strict=True):
        assert torch.isfinite(gradient).all(), f"{name} gradient has non-finite entries"
    class_embed_parameter_ids = {id(p) for m in transformer.enc_out_class_embed for p in m.parameters()}
    class_embed_gradients = [
        gradient
        for (name, parameter), gradient in zip(named_parameters, gradients[len(srcs) :], strict=True)
        if id(parameter) in class_embed_parameter_ids
    ]
    assert class_embed_gradients, "enc_out_class_embed contributed no parameters to this check"
    assert any(gradient.abs().sum() > 0 for gradient in class_embed_gradients), (
        "enc_out_class_embed must receive real gradient through the reused cls_ts under bf16 autocast"
    )


@pytest.mark.gpu
@_requires_cuda
@pytest.mark.parametrize(
    "spatial_shapes_hw",
    [
        pytest.param([(8, 8), (4, 4)], id="2-level"),
        pytest.param([(8, 8)], id="1-level"),
    ],
)
def test_two_stage_group_selection_compiles_with_finite_gradients(spatial_shapes_hw: list[tuple[int, int]]) -> None:
    """The batched fast path must be compatible with torch.compile, at both 1 and 2 feature levels.

    Unlike ONNX/TorchScript export (which always forces `group_detr=1` and never reaches this code), `module_model.py`
    applies `torch.compile` directly to the training model, where `group_detr>1` and this new path are exactly what
    training exercises. This asserts compile-time and run-time compatibility (finite outputs and gradients through a
    compiled call, using the same `capture_scalar_outputs` config `module_model.py` itself sets before compiling) -- not
    a timing claim; a local single-GPU smoke run is not a substitute for this project's separately reported L4 step-time
    evidence. Production RF-DETR models (Base, Small, Nano, Medium, Large) use a single feature level, so the 1-level
    case exercises the `is_compiling()` stacked-scalar_tensor spatial_shapes branch at the level count training actually
    compiles most, not only the 2-level case the other real-compile coverage in this file used exclusively.
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 5, 4
    device = "cuda"

    srcs = [torch.randn(2, hidden_dim, ht, wd, device=device, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool, device=device) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd, device=device) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4, device=device)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim, device=device)

    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=True, num_feature_levels=len(spatial_shapes_hw)
    ).to(device)
    assert transformer._two_stage_batching_eligible()
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled_transformer = torch.compile(transformer, dynamic=True)

        _, _, memory_ts, boxes_ts, _ = compiled_transformer(
            srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
        )
        loss = memory_ts.sum() + boxes_ts.sum()
        parameters = [
            parameter
            for module in [*transformer.enc_output, *transformer.enc_output_norm, *transformer.enc_out_bbox_embed]
            for parameter in module.parameters()
        ]
        gradients = torch.autograd.grad(loss, [*srcs, *parameters])

    assert torch.isfinite(memory_ts).all()
    assert torch.isfinite(boxes_ts).all()
    for gradient in gradients:
        assert torch.isfinite(gradient).all()


def test_dynamic_compile_reuses_graph_across_resolutions() -> None:
    """A new input resolution must not compile a new Transformer graph under ``dynamic=True``.

    Multi-scale training feeds up to 11 resolutions through one ``torch.compile(dynamic=True)`` model. Building
    ``spatial_shapes`` with ``torch.as_tensor`` of the (H, W) pairs, or the encoder-proposal grid with
    ``torch.linspace(0, n - 1, n)``, specialises H and W to their traced values, so every resolution recompiled this
    frame until Dynamo's recompile limit (8) left the remaining resolutions eager. The ``eager`` backend exercises the
    Dynamo guards that decide recompilation without paying for Inductor, and ``capture_scalar_outputs`` mirrors the
    training compile setup in ``module_model.py``. The first-traced sizes avoid 0 and 1, which Dynamo would specialise
    to constants, and differ from the other input dimensions, which duck sizing would tie to a shared symbol. Counts are
    taken relative to the global Dynamo counter, and the first resolution must add a graph, so the test also fails if
    nothing gets compiled at all. Every resolution is non-square (H != W) and each level within a resolution swaps
    which axis is larger, so a [W, H] axis swap anywhere in the compiled ``spatial_shapes`` construction --
    undetectable by the graph-count assertion alone -- would still change the numeric output compared against the
    eager, uncompiled ``transformer`` (built with the default ``dropout=0.0``, so its forward is deterministic and
    directly comparable against the compiled call on the same inputs).
    """
    torch._dynamo.reset()
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=7, bbox_reparam=True
    )
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)
    graph_counts = []
    baseline = torch._dynamo.utils.counters["stats"]["unique_graphs"]
    with torch._dynamo.config.patch(capture_scalar_outputs=True):
        compiled_transformer = torch.compile(transformer, dynamic=True, backend="eager")
        for spatial_shapes_hw in ([(10, 14), (5, 7)], [(12, 20), (6, 10)], [(18, 14), (9, 7)]):
            srcs = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
            masks = [torch.zeros(2, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
            pos_embeds = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
            compiled_out = compiled_transformer(
                srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
            )
            graph_counts.append(torch._dynamo.utils.counters["stats"]["unique_graphs"] - baseline)
            uncompiled_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)
            torch.testing.assert_close(compiled_out, uncompiled_out)

    assert graph_counts[0] > 0 and graph_counts[1:] == [graph_counts[0]] * 2, f"graphs per resolution: {graph_counts}"


def test_is_exporting_branch_builds_spatial_shapes_with_as_tensor(mocker: MockerFixture) -> None:
    """``Transformer.forward`` must build ``spatial_shapes`` via ``torch.as_tensor`` while ``is_exporting()`` is True.

    ``torch.export`` (ExecuTorch) cannot trace ``torch._shape_as_tensor`` (the eager branch's op), and the
    ``is_compiling()`` branch stacks per-size ``torch.scalar_tensor`` calls to stay symbolic under ``torch.compile`` --
    neither is what ``torch.export`` needs. The ``is_exporting()`` branch exists to route around both by baking
    ``spatial_shapes`` from the concrete (H, W) pairs with a single ``torch.as_tensor`` call. This branch had zero test
    coverage before and after this PR. Mirrors the ``is_compiling()`` polyfill regression
    (``test_spatial_shapes_survives_dynamo_shape_as_tensor_polyfill``): monkeypatch the compile-state predicate, spy on
    both candidate tensor constructors to pin exactly which branch ran, and check the output still matches an
    unpatched eager run.
    """
    hidden_dim, num_queries, group_detr = 16, 3, 2
    transformer = _build_two_stage_transformer_with_production_shaped_heads(
        hidden_dim, num_queries, group_detr, num_classes=5, bbox_reparam=False
    )
    spatial_shapes_hw = [(6, 8), (3, 4)]
    srcs = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    masks = [torch.zeros(2, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    with torch.no_grad():
        eager_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    mocker.patch.object(torch.compiler, "is_exporting", create=True, return_value=True)
    as_tensor = mocker.patch.object(torch, "as_tensor", wraps=torch.as_tensor)
    scalar_tensor = mocker.patch.object(
        torch,
        "scalar_tensor",
        side_effect=AssertionError("is_exporting() branch must not build spatial_shapes with torch.scalar_tensor"),
    )

    with torch.no_grad():
        exporting_out = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert as_tensor.call_count == 1, "is_exporting() branch must build spatial_shapes with exactly one as_tensor call"
    assert scalar_tensor.call_count == 0
    torch.testing.assert_close(eager_out, exporting_out)


def _build_two_stage_transformer_with_keypoints(
    hidden_dim: int, num_queries: int, group_detr: int, num_classes: int, num_keypoints: int
) -> Transformer:
    """Build a decoder-free, two-stage ``Transformer`` with GroupPose keypoint heads enabled.

    ``num_decoder_layers=0`` isolates the encoder-side computation this PR touches (
    :meth:`Transformer._two_stage_group_selection` and the untouched per-group
    ``enc_out_keypoint_embed`` loop that consumes its output) from the decoder's own keypoint
    cross-attention plumbing, which this change does not modify.

    Args:
        hidden_dim: Model width.
        num_queries: Queries selected per group.
        group_detr: Number of independent groups.
        num_classes: Class-head output width.
        num_keypoints: Keypoints per instance.

    Returns:
        A decoder-free, two-stage, GroupPose-keypoint-enabled ``Transformer`` left in its default
        training mode.

    Examples:
        >>> t = _build_two_stage_transformer_with_keypoints(16, 3, 2, 5, 4)
        >>> t._two_stage_batching_eligible()
        True
    """
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=0,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=2,
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=group_detr,
        use_grouppose_keypoints=True,
        num_keypoints_per_class=[num_keypoints],
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in range(group_detr)]
    )
    return transformer


def test_two_stage_group_selection_matches_generic_loop_for_grouppose_keypoint_config() -> None:
    """The batched fast path must also match the generic loop for a GroupPose keypoint config.

    ``_two_stage_batching_eligible`` only inspects ``enc_output``/``enc_output_norm``/
    ``enc_out_class_embed``/``enc_out_bbox_embed`` -- the same concrete types ``LWDETR.__init__``
    constructs regardless of ``use_grouppose_keypoints`` -- so the fast path also activates for
    keypoint/pose models (``RFDETRKeypointPreviewConfig`` defaults to ``group_detr=13``, same as every
    detection config). ``enc_kp_predictions`` is derived from the two-stage ``memory_ts``/``boxes_ts``
    via ``keypoint_query_initializer_enc`` and the untouched per-group ``enc_out_keypoint_embed`` loop,
    so a divergence in ``memory_ts``/``boxes_ts`` between the two paths could still surface here even
    though it does not for a plain detection model.
    """
    torch.manual_seed(0)
    hidden_dim, num_queries, group_detr, num_keypoints = 16, 5, 4, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]

    srcs = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(2, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(2, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = _build_two_stage_transformer_with_keypoints(
        hidden_dim, num_queries, group_detr, num_classes=7, num_keypoints=num_keypoints
    )
    assert transformer._two_stage_batching_eligible()

    outputs_fast = transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    transformer_loop = copy.deepcopy(transformer)
    transformer_loop._two_stage_batching_eligible = lambda: False
    outputs_loop = transformer_loop(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    # return_values layout with num_decoder_layers=0: (hs=None, references=None, memory_ts, boxes_ts,
    # keypoint_hs=None, enc_kp_predictions, keypoint_memory_ts) -- see Transformer.forward's tail.
    memory_fast, boxes_fast, enc_kp_fast = outputs_fast[2], outputs_fast[3], outputs_fast[5]
    memory_loop, boxes_loop, enc_kp_loop = outputs_loop[2], outputs_loop[3], outputs_loop[5]

    assert enc_kp_fast is not None and enc_kp_loop is not None
    torch.testing.assert_close(memory_fast, memory_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(boxes_fast, boxes_loop, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(enc_kp_fast, enc_kp_loop, atol=1e-4, rtol=1e-4)


def test_two_stage_topk_gather_backward_routes_gradient_only_to_selected_rows() -> None:
    """Backward through the two-stage top-k gather routes gradient only to the selected source positions.

    The gather is a plain row copy, so every unselected position must see exactly zero gradient.

    Regression: the forward-value assertions in the sibling tests confirm memory_ts/boxes_ts *equal* the
    correct rows, but a broadcast-index bug that accidentally selected the wrong rows in a way that still
    passed those value checks (e.g. by coincidence on this fixture) would still corrupt exactly which
    upstream positions receive gradient during training. Every position in this fixture's 4x4+2x2
    spatial grid is a "valid" proposal (see gen_encoder_output_proposals's 0.01-0.99 validity window), so
    output_memory is an untouched pass-through of memory and gradient must localise per-row exactly.
    """
    torch.manual_seed(0)
    batch_size, hidden_dim, num_queries = 1, 16, 3
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)

    srcs = [torch.randn(batch_size, hidden_dim, ht, wd, requires_grad=True) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(batch_size, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(batch_size, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=False,
        group_detr=1,
    )

    picks = [17, 2, 9]
    transformer.enc_out_class_embed = nn.ModuleList([_FixedTopkScores(_make_out_of_order_scores(total_hw, picks))])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])

    _, _, memory_ts, boxes_ts, _ = transformer(
        srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None
    )
    (memory_ts.sum() + boxes_ts.sum()).backward()

    assert all(src.grad is not None for src in srcs)
    # Reproduce Transformer.forward's memory flatten order: cat(src.flatten(2).transpose(1, 2)).
    flattened_grads = torch.cat([src.grad.flatten(2).transpose(1, 2) for src in srcs], dim=1)  # (1, total_hw, dim)

    selected_mask = torch.zeros(total_hw, dtype=torch.bool)
    selected_mask[torch.tensor(picks)] = True

    assert (flattened_grads[0, selected_mask] != 0).any(dim=-1).all(), (
        "every selected row must receive nonzero gradient through the two-stage top-k gather"
    )
    assert torch.equal(flattened_grads[0, ~selected_mask], torch.zeros_like(flattened_grads[0, ~selected_mask])), (
        "every unselected row must receive exactly zero gradient through the two-stage top-k gather"
    )


@pytest.mark.gpu
@_requires_cuda
def test_two_stage_topk_gather_cuda_matches_cpu_for_out_of_position_order_indices() -> None:
    """On CUDA the two-stage top-k gather reproduces the CPU result bit-for-bit for out-of-order indices.

    The gather (``torch.gather`` with an index broadcast via ``Tensor.expand``) is an arithmetic-free row copy; the
    indices are those of ``test_two_stage_topk_gather_selects_correct_rows_out_of_position_order``. Like the parity twin
    for ``PostProcess._gather_and_scale_boxes`` (PR #1268), it isolates the bare gather+expand pattern, so surrounding
    Linear/LayerNorm CPU/CUDA floating-point differences cannot reach the comparison.
    """
    batch_size, hidden_dim, num_positions = 2, 16, 20
    source = torch.randn(batch_size, num_positions, hidden_dim)
    # Same out-of-position-order, per-batch-row-distinct picks as the CPU-only sibling test.
    topk_proposals = torch.tensor([[17, 2, 9], [5, 19, 0]])

    cpu_selected = torch.gather(source, 1, topk_proposals.unsqueeze(-1).expand(-1, -1, hidden_dim))
    cuda_selected = torch.gather(source.cuda(), 1, topk_proposals.cuda().unsqueeze(-1).expand(-1, -1, hidden_dim))

    assert torch.equal(cpu_selected, cuda_selected.cpu())


def _bbox_from_delta(delta: torch.Tensor, proposals: torch.Tensor, bbox_reparam: bool) -> torch.Tensor:
    """Reproduce Transformer.forward's two-stage box construction from a bbox-delta MLP output.

    Args:
        delta: Raw ``enc_out_bbox_embed`` output, shape ``(..., 4)``.
        proposals: Matching ``output_proposals`` rows, shape ``(..., 4)``.
        bbox_reparam: Selects the ``bbox_reparam`` branch (cx/cy/w/h reparam vs. plain unsigmoid add).

    Returns:
        Box tensor, shape ``(..., 4)``.

    Examples:
        >>> import torch
        >>> delta = torch.zeros(1, 1, 4)
        >>> proposals = torch.full((1, 1, 4), 0.5)
        >>> torch.equal(_bbox_from_delta(delta, proposals, bbox_reparam=False), proposals)
        True
    """
    if bbox_reparam:
        return torch.cat(
            [
                delta[..., :2] * proposals[..., 2:] + proposals[..., :2],
                delta[..., 2:].exp() * proposals[..., 2:],
            ],
            dim=-1,
        )
    return delta + proposals


@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_bbox_mlp_gather_order_matches_forward_and_gradient_for_real_three_layer_mlp(
    bbox_reparam: bool,
) -> None:
    """Gathering top-k rows before vs.

    after the bbox-delta MLP must match in forward value and gradient.     Uses the real production MLP (``MLP(d, d, 4,
    num_layers=3)``, as ``LWDETR.bbox_embed`` in ``lwdetr.py``), not the     single ``nn.Linear`` stand-in of the
    sibling gather tests, and covers gradient parity, which those check only     indirectly (nonzero/zero row pattern,
    not old-vs-new equality).

    The MLP has no cross-token mixing, so ``d(mlp(x)_i)/dx_j`` is zero for every ``j != i``: backward is as row-
    independent as forward, and gathering before or after the MLP must give identical gradients w.r.t. the shared input,
    not just identical boxes. A tolerance replaces ``torch.equal``: three chained matmuls are enough for the CPU BLAS
    reduction order to differ across platforms (bit-inexact on macOS/Accelerate CI, 1e-7 to 1e-5 for the value; bit-
    exact on Linux/OpenBLAS), because the platform's BLAS choice matters more than tensor size.
    """
    torch.manual_seed(0)
    bs, sum_hw, d, num_queries = 2, 20, 16, 3
    bbox_mlp = MLP(d, d, 4, num_layers=3)
    output_memory = torch.randn(bs, sum_hw, d, requires_grad=True)
    output_proposals = torch.rand(bs, sum_hw, 4) * 0.9 + 0.05
    topk_idx = torch.stack([torch.randperm(sum_hw)[:num_queries] for _ in range(bs)])

    # Old: MLP on every row (bs, sum_hw, d), gather the box afterwards.
    box_old_full = _bbox_from_delta(bbox_mlp(output_memory), output_proposals, bbox_reparam)
    box_old = torch.gather(box_old_full, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_old.sum().backward()
    grad_old = output_memory.grad.clone()
    output_memory.grad = None

    # New (this fix): gather the selected rows first, run the MLP only on those.
    tgt_new = torch.gather(output_memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
    proposals_g = torch.gather(output_proposals, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_new = _bbox_from_delta(bbox_mlp(tgt_new), proposals_g, bbox_reparam)
    box_new.sum().backward()
    grad_new = output_memory.grad.clone()

    torch.testing.assert_close(box_old, box_new, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(grad_old, grad_new, atol=1e-5, rtol=1e-4)


@pytest.mark.gpu
@_requires_cuda
@pytest.mark.parametrize("bbox_reparam", [False, True])
def test_two_stage_bbox_mlp_gather_order_matches_on_cuda_for_real_three_layer_mlp(bbox_reparam: bool) -> None:
    """CUDA twin of test_two_stage_bbox_mlp_gather_order_matches_forward_and_gradient_for_real_three_layer_mlp.

    Runs the real bbox MLP on CUDA, unlike the bare ``torch.gather`` isolated by
    ``test_two_stage_topk_gather_cuda_matches_cpu_for_out_of_position_order_indices``. It compares old order against
    new order on the same device, not CUDA against CPU, whose reduction order already differs for this MLP and would
    swamp the gather-order comparison. The two paths may use different kernels, so a small numerical tolerance applies,
    without mutating process-wide deterministic-algorithm state or requiring ``CUBLAS_WORKSPACE_CONFIG``.
    """
    torch.manual_seed(0)
    bs, sum_hw, d, num_queries = 2, 20, 16, 3
    bbox_mlp = MLP(d, d, 4, num_layers=3).cuda()
    output_memory = torch.randn(bs, sum_hw, d, device="cuda")
    output_proposals = torch.rand(bs, sum_hw, 4, device="cuda") * 0.9 + 0.05
    topk_idx = torch.stack([torch.randperm(sum_hw, device="cuda")[:num_queries] for _ in range(bs)])

    box_old_full = _bbox_from_delta(bbox_mlp(output_memory), output_proposals, bbox_reparam)
    box_old = torch.gather(box_old_full, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))

    tgt_new = torch.gather(output_memory, 1, topk_idx.unsqueeze(-1).expand(-1, -1, d))
    proposals_g = torch.gather(output_proposals, 1, topk_idx.unsqueeze(-1).expand(-1, -1, 4))
    box_new = _bbox_from_delta(bbox_mlp(tgt_new), proposals_g, bbox_reparam)

    torch.testing.assert_close(box_old, box_new, atol=1e-5, rtol=1e-4)


class _ShapeRecordingLinear(nn.Module):
    """Wraps a real ``nn.Linear`` and records the shape of every input it is called with.

    Lets a test assert how many rows the wrapped layer actually processed, independent of the values
    it produced (already covered by the ``test_two_stage_topk_gather_selects_correct_rows_*`` tests).

    Examples:
        >>> input_shapes = []
        >>> layer = _ShapeRecordingLinear(nn.Linear(2, 3), input_shapes)
        >>> layer(torch.zeros(1, 2)).shape
        torch.Size([1, 3])
        >>> input_shapes
        [torch.Size([1, 2])]
    """

    def __init__(self, inner: nn.Linear, input_shapes: list[torch.Size]) -> None:
        """Initialize the recording wrapper around a linear layer.

        Args:
            inner: Linear layer whose input shapes should be recorded.
            input_shapes: Mutable list receiving each input shape in call order.
        """
        super().__init__()
        self.inner = inner
        self._input_shapes = input_shapes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Record ``x.shape`` then delegate to the wrapped ``nn.Linear``."""
        self._input_shapes.append(x.shape)
        return self.inner(x)


@pytest.mark.parametrize("group_detr", [1, 3])
@pytest.mark.parametrize("bbox_reparam", [False, True])
@pytest.mark.parametrize("num_queries", [3, 20])
def test_two_stage_bbox_embed_only_runs_on_selected_rows_not_full_encoder_memory(
    group_detr: int, bbox_reparam: bool, num_queries: int
) -> None:
    """The bbox-delta MLP must only run on the rows torch.topk selects, not on every encoder position.

    ``enc_out_bbox_embed`` is pointwise, so it needs only the ``num_queries`` rows that survive ``torch.topk``.
    Regression: Transformer.forward used to run ``enc_out_bbox_embed[g_idx]`` on the full ``output_memory_gidx``
    (bs, sum_hw, d) for every one of ``group_detr`` groups and keep only the gathered rows. This pins the input shape
    it is called with to the post-topk size, per group; it checks how much work the MLP does, not what it produces.
    Row values are covered by ``test_two_stage_topk_gather_selects_correct_rows_out_of_position_order`` and
    ``test_two_stage_topk_gather_broadcasts_across_groups_in_training_mode``.

    ``bbox_reparam`` is parametrized because ``True`` (the production default per ``ModelConfig.bbox_reparam``) runs
    extra pointwise ops (``.exp()``, multiply, ``torch.concat``) on the MLP output that ``False`` does not.
    """
    torch.manual_seed(0)
    hidden_dim = 16
    spatial_shapes_hw = [(4, 4), (2, 2)]
    total_hw = sum(ht * wd for ht, wd in spatial_shapes_hw)
    assert total_hw >= num_queries  # also cover the topk == encoder-memory boundary

    srcs = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    masks = [torch.zeros(1, ht, wd, dtype=torch.bool) for ht, wd in spatial_shapes_hw]
    pos_embeds = [torch.randn(1, hidden_dim, ht, wd) for ht, wd in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries * group_detr, 4)
    query_feat = torch.randn(num_queries * group_detr, hidden_dim)

    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=bbox_reparam,
        group_detr=group_detr,
    )
    assert transformer.training  # default nn.Module state; group_detr>1 only takes effect while training

    num_classes = 5
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, num_classes) for _ in range(group_detr)])
    bbox_embed_input_shapes: list[torch.Size] = []
    transformer.enc_out_bbox_embed = nn.ModuleList(
        [_ShapeRecordingLinear(nn.Linear(hidden_dim, 4), bbox_embed_input_shapes) for _ in range(group_detr)]
    )

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat, cross_attn_srcs=None)

    assert len(bbox_embed_input_shapes) == group_detr
    for shape in bbox_embed_input_shapes:
        assert shape == torch.Size([1, num_queries, hidden_dim]), (
            "enc_out_bbox_embed must only run on the num_queries rows selected by torch.topk, not the "
            f"full sum(H*W)={total_hw} encoder positions (Transformer.forward two-stage top-k gather); "
            f"got input shape {tuple(shape)}"
        )


def _make_cuda_graph_transformer_inputs(
    spatial_shapes_hw: list[tuple[int, int]], batch_size: int = 2, hidden_dim: int = 16, num_queries: int = 3
) -> tuple[Transformer, list[torch.Tensor], list[torch.Tensor], list[torch.Tensor], torch.Tensor, torch.Tensor]:
    """Build a small two-stage Transformer and matching forward inputs.

    Examples:
        >>> transformer, srcs, *_ = _make_cuda_graph_transformer_inputs([(2, 2)])
        >>> len(srcs), transformer.num_feature_levels
        (1, 1)
    """
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=1,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=len(spatial_shapes_hw),
        dec_n_points=1,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 5)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])
    srcs = [torch.randn(batch_size, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    masks = [torch.zeros(batch_size, height, width, dtype=torch.bool) for height, width in spatial_shapes_hw]
    pos_embeds = [torch.randn(batch_size, hidden_dim, height, width) for height, width in spatial_shapes_hw]
    refpoint_embed = torch.rand(num_queries, 4)
    query_feat = torch.randn(num_queries, hidden_dim)
    return transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat


def test_cuda_graph_spatial_shapes_cache_reuses_tensor_per_device_and_resolution() -> None:
    """Repeated capture warmups reuse the exact device shape tensor."""
    transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    transformer.enable_cuda_graph_capture()

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)
    assert transformer._cuda_graph_spatial_shapes is not None
    assert len(transformer._cuda_graph_spatial_shapes) == 1
    cached = next(iter(transformer._cuda_graph_spatial_shapes.values()))

    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)
    assert next(iter(transformer._cuda_graph_spatial_shapes.values())) is cached


def test_cuda_graph_spatial_shapes_cache_keys_distinct_resolutions() -> None:
    """Each multi-scale resolution gets its own static shape tensor."""
    transformer, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    transformer.enable_cuda_graph_capture()
    transformer(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    _, srcs_b, masks_b, pos_embeds_b, _, _ = _make_cuda_graph_transformer_inputs([(6, 6), (3, 3)])
    transformer(srcs_b, masks_b, pos_embeds_b, refpoint_embed, query_feat)

    assert transformer._cuda_graph_spatial_shapes is not None
    assert {key[1] for key in transformer._cuda_graph_spatial_shapes} == {
        ((4, 4), (2, 2)),
        ((6, 6), (3, 3)),
    }


def test_cuda_graph_spatial_shapes_cache_preserves_forward_values() -> None:
    """The cache changes tensor construction, not model arithmetic."""
    torch.manual_seed(7)
    baseline, srcs, masks, pos_embeds, refpoint_embed, query_feat = _make_cuda_graph_transformer_inputs(
        [(4, 4), (2, 2)]
    )
    baseline_output = baseline(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    torch.manual_seed(7)
    cached, *_ = _make_cuda_graph_transformer_inputs([(4, 4), (2, 2)])
    cached.enable_cuda_graph_capture()
    cached_output = cached(srcs, masks, pos_embeds, refpoint_embed, query_feat)

    for expected, actual in zip(baseline_output, cached_output):
        if expected is None:
            assert actual is None
        else:
            assert torch.equal(actual, expected)


class TestTwoStageGroupSelection:
    """Two-stage group selection: cast-then-expand of the shared encoder memory."""

    @pytest.mark.parametrize("dtype", [pytest.param(torch.bfloat16, id="bf16"), pytest.param(torch.float16, id="fp16")])
    def test_cast_then_expand_matches_expand_then_cast_forward_and_backward(self, dtype: torch.dtype) -> None:
        """Casting once before the expand gives the expanded cast's values; the backward is the fp32 group sum."""
        memory = torch.randn(2, 20, 16, requires_grad=True)
        grad_out = torch.randn(13, 2, 20, 16).to(dtype)

        expected = memory.unsqueeze(0).expand(13, -1, -1, -1).to(dtype)
        expected.backward(grad_out)
        expected_grad = memory.grad
        memory.grad = None

        actual = _CastThenExpand.apply(memory, 13, dtype)
        actual.backward(grad_out)

        assert actual.dtype == dtype and actual.shape == (13, 2, 20, 16)
        assert torch.equal(actual, expected)
        assert expected_grad is not None and memory.grad is not None
        assert memory.grad.dtype == torch.float32
        torch.testing.assert_close(memory.grad, expected_grad, rtol=1e-6, atol=1e-6)

    @pytest.mark.gpu
    @_requires_cuda
    def test_cast_once_matches_expanded_cast_under_autocast(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Under bf16 autocast cast-once selection is bitwise the expand-then-cast path, with gradients equal to fp32
        rounding.

        The encoder memory's gradient is summed over groups in fp32 either way.
        """
        torch.manual_seed(0)
        ns = _namespace_from_configs(
            RFDETRNanoConfig(num_classes=80, pretrain_weights=None, device="cpu"), TrainConfig(dataset_dir="/tmp")
        )
        transformer = build_model(ns).transformer.cuda()
        transformer.train()
        assert transformer._two_stage_batching_eligible()
        memory = torch.randn(2, 60, transformer.d_model, device="cuda")
        proposals = torch.rand(2, 60, 4, device="cuda")

        def run(cast_once: bool) -> tuple[list[torch.Tensor], torch.Tensor]:
            if not cast_once:
                monkeypatch.setattr(
                    _CastThenExpand,
                    "apply",
                    lambda x, groups, dtype: x.unsqueeze(0).expand(groups, -1, -1, -1).to(dtype),
                )
            else:
                monkeypatch.undo()
            transformer.zero_grad(set_to_none=True)
            memory_in = memory.clone().requires_grad_(True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                outputs = transformer._two_stage_group_selection(memory_in, proposals, transformer.group_detr)
            outputs[2].float().sum().backward()
            assert memory_in.grad is not None
            return [o.detach() for o in outputs], memory_in.grad.clone()

        expected, expected_grad = run(cast_once=False)
        actual, actual_grad = run(cast_once=True)

        for expected_out, actual_out in zip(expected, actual, strict=True):
            assert torch.equal(actual_out, expected_out)
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    @pytest.mark.gpu
    @_requires_cuda
    def test_keeps_expand_then_cast_while_tracing(self, mocker: MockerFixture) -> None:
        """TorchScript tracing must not record the custom cast-then-expand autograd function.

        The first call is a positive control proving bf16 CUDA autocast normally takes the rewrite. The tracing call
        then requires the previous expand-then-cast graph, so this fails if the tracing predicate is absent from the
        guard.
        """
        transformer = _build_two_stage_transformer_with_production_shaped_heads(
            hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
        ).cuda()
        memory = torch.randn(2, 20, 16, device="cuda")
        proposals = torch.rand(2, 20, 4, device="cuda")
        cast_once_calls: list[object] = []
        real_apply = _CastThenExpand.apply
        mocker.patch.object(
            _CastThenExpand, "apply", side_effect=lambda *args: (cast_once_calls.append(args), real_apply(*args))[1]
        )

        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)
        assert len(cast_once_calls) == 1
        cast_once_calls.clear()

        mocker.patch("rfdetr.models.transformer._is_tracing", return_value=True)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)

        assert cast_once_calls == []

    @pytest.mark.usefixtures("forced_eager_cuda_gate")
    def test_routes_the_shared_memory_through_cast_then_expand(self, mocker: MockerFixture) -> None:
        """With the eager-CUDA gate open and bf16 autocast, the fp32 memory is cast once, then expanded per group.

        Runs on CPU by forcing the gate, so it asserts routing only: the spy hands back the fp32 expansion the plain
        path builds, because without real autocast the batched GEMM downstream needs fp32 operands.
        """
        transformer = _build_two_stage_transformer_with_production_shaped_heads(
            hidden_dim=16, num_queries=3, group_detr=2, num_classes=5, bbox_reparam=False
        )
        memory = torch.randn(2, 20, 16)
        proposals = torch.rand(2, 20, 4)
        cast_then_expand = mocker.patch.object(
            _CastThenExpand, "apply", side_effect=lambda x, groups, dtype: x.unsqueeze(0).expand(groups, *x.shape)
        )

        transformer._two_stage_group_selection(memory, proposals, transformer.group_detr)

        assert [call.args[1:] for call in cast_then_expand.call_args_list] == [(2, torch.bfloat16)]
