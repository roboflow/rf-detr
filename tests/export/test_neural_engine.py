# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Graph rewrites that keep an exported RF-DETR on the Apple Neural Engine (``coreml_neural_engine=True``)."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch
from torch import nn

from rfdetr import RFDETRNano
from rfdetr.export._coreml.exporter import CoreMLConfig, CoreMLExporter
from rfdetr.export._neural_engine import neural_engine_model, one_hot_top_rows, split_einsum_encoder
from rfdetr.models.backbone.dinov2_with_windowed_attn import (
    WindowedDinov2WithRegistersBackbone,
    WindowedDinov2WithRegistersConfig,
    WindowedDinov2WithRegistersEncoder,
)
from rfdetr.models.transformer import Transformer, select_top_rows
from rfdetr.utilities.reproducibility import seed_all
from tests.export.test_coreml_export import _make_export_graph


def _tiny_backbone(**config_overrides: Any) -> WindowedDinov2WithRegistersBackbone:
    """Build a two-layer, 32-channel windowed DINOv2 backbone, with *config_overrides* applied to its config.

    Examples:
        >>> backbone = _tiny_backbone(drop_path_rate=0.1)
        >>> len(backbone.encoder.layer), backbone.config.hidden_size, backbone.config.drop_path_rate
        (2, 32, 0.1)
    """
    config = WindowedDinov2WithRegistersConfig(
        image_size=32,
        patch_size=16,
        hidden_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        out_indices=[2],
        **config_overrides,
    )
    return WindowedDinov2WithRegistersBackbone(config)


def _perturbed_windowed_encoder(num_windows: int) -> WindowedDinov2WithRegistersEncoder:
    """Build a tiny eval-mode windowed encoder whose LayerNorm affines and layer scales are random, not identity.

    Layer 0 attends within each window and layer 1 across all windows, so both attention paths of the rebuild run.
    Fresh LayerNorms are 1/0 and fresh layer scales all ones, which would hide a wrong fold of either into the weights.

    Examples:
        >>> encoder = _perturbed_windowed_encoder(num_windows=2)
        >>> encoder.config.window_block_indexes, bool((encoder.layer[0].layer_scale1.lambda1 != 1).all())
        ([0], True)
    """
    encoder = _tiny_backbone(num_windows=num_windows, window_block_indexes=[0]).encoder
    with torch.no_grad():
        for layer in encoder.layer:
            for norm in (layer.norm1, layer.norm2):
                norm.weight.uniform_(0.5, 1.5)
                norm.bias.normal_(std=0.1)
            for layer_scale in (layer.layer_scale1, layer.layer_scale2):
                layer_scale.lambda1.uniform_(0.5, 1.5)
    return encoder.eval()


def _two_stage_transformer() -> Transformer:
    """Build a small eval-mode two-stage transformer, like the released detection models (``bbox_reparam=True``).

    Examples:
        >>> transformer = _two_stage_transformer()
        >>> transformer.two_stage, transformer.bbox_reparam, transformer.training
        (True, True, False)
    """
    hidden_dim, num_queries = 16, 5
    transformer = Transformer(
        d_model=hidden_dim,
        num_queries=num_queries,
        num_decoder_layers=2,
        sa_nhead=4,
        ca_nhead=4,
        num_feature_levels=1,
        dec_n_points=2,
        return_intermediate_dec=True,
        lite_refpoint_refine=True,
        two_stage=True,
        bbox_reparam=True,
        group_detr=1,
    )
    transformer.enc_out_class_embed = nn.ModuleList([nn.Linear(hidden_dim, 3)])
    transformer.enc_out_bbox_embed = nn.ModuleList([nn.Linear(hidden_dim, 4)])
    return transformer.eval()


class TestOneHotQuerySelection:
    """:func:`one_hot_top_rows` selects the rows ``torch.topk`` + ``gather`` would, without either op."""

    def test_transformer_outputs_match_top_k_gather(self) -> None:
        """Verify that one-hot row selection matches the default top-k gather outputs."""
        torch.manual_seed(0)
        transformer = _two_stage_transformer()
        hidden_dim, feature_size = 16, 6
        srcs = [torch.randn(1, hidden_dim, feature_size, feature_size)]
        masks = [torch.zeros(1, feature_size, feature_size, dtype=torch.bool)]
        pos_embeds = [torch.randn(1, hidden_dim, feature_size, feature_size)]
        query_inputs = (torch.rand(5, 4), torch.randn(5, hidden_dim))

        with torch.no_grad():
            reference = transformer(srcs, masks, pos_embeds, *query_inputs)
            transformer.select_top_rows = one_hot_top_rows
            candidate = transformer(srcs, masks, pos_embeds, *query_inputs)

        reference_tensors, _ = torch.utils._pytree.tree_flatten(reference)
        candidate_tensors, _ = torch.utils._pytree.tree_flatten(candidate)
        assert len(candidate_tensors) == len(reference_tensors)
        for expected, actual in zip(reference_tensors, candidate_tensors):
            assert (actual is None and expected is None) or torch.equal(actual, expected)

    def test_equal_scores_select_the_lower_index_first(self) -> None:
        """Tied scores are ranked by index, lowest first, like a stable descending sort."""
        scores = torch.tensor([[1.0, 3.0, 3.0, 2.0, 3.0]])
        rows = torch.arange(5.0).reshape(1, 5, 1)

        selected = one_hot_top_rows(scores, 4)(rows)

        assert selected.flatten().tolist() == [1.0, 2.0, 4.0, 3.0]

    def test_more_rows_than_tokens_is_refused(self) -> None:
        """Asking for more rows than there are tokens raises, as ``torch.topk`` does, instead of returning zero rows."""
        with pytest.raises(ValueError, match="cannot select 4 rows from 3 tokens"):
            one_hot_top_rows(torch.zeros(1, 3), 4)

    @pytest.mark.parametrize(
        "dtype", [pytest.param(torch.float32, id="float32"), pytest.param(torch.float16, id="float16")]
    )
    @pytest.mark.parametrize("num_levels", [1, 3, 1000])
    @pytest.mark.parametrize(
        ("batch", "num_tokens", "k"), [(1, 1, 1), (2, 9, 1), (2, 9, 4), (3, 9, 9), (2, 300, 7), (2, 300, 300)]
    )
    def test_selects_the_rows_of_a_stable_descending_sort(
        self, dtype: torch.dtype, num_levels: int, batch: int, num_tokens: int, k: int
    ) -> None:
        """One-hot selection picks the same rows, in the same order, as ``torch.sort(descending=True, stable=True)``.

        Scores drawn from 1, 3 or 1000 levels give all-equal, dense and sparse ties, so tie blocks straddle rank ``k``;
        float16 is the precision the Neural Engine runs. Each row holds its own index, so the check compares indices.
        """
        scores = torch.randint(0, num_levels, (batch, num_tokens)).to(dtype)
        rows = torch.arange(num_tokens, dtype=dtype).reshape(1, num_tokens, 1).expand(batch, -1, -1)
        expected = torch.sort(scores, dim=1, descending=True, stable=True).indices[:, :k, None].to(dtype)

        selected = one_hot_top_rows(scores, k)(rows)

        assert torch.equal(selected, expected)


class TestSplitEinsumEncoder:
    """:func:`split_einsum_encoder` rebuilds a windowed DINOv2 encoder with the same outputs."""

    @pytest.mark.parametrize("query_chunk", [256, 100])
    def test_backbone_features_match_the_original_encoder(self, query_chunk: int) -> None:
        """The RFDETRNano backbone returns the same features, to 1e-4, with its encoder rebuilt.

        At 384 px, windowed layers attend over 4 windows of 145 tokens and global layers over 580 tokens. A 256-token
        chunk splits only the global layers (194 + 194 + 192); a 100-token chunk splits both (73 + 72, and five chunks
        of 97 plus one of 95).
        """
        seed_all(0)
        dinov2 = RFDETRNano(pretrain_weights=None).model.model.backbone[0].encoder.eval()
        image = torch.randn(1, 3, 384, 384)
        with torch.no_grad():
            reference = dinov2(image)
            dinov2.encoder.encoder = split_einsum_encoder(dinov2.encoder.encoder, query_chunk=query_chunk)
            candidate = dinov2(image)

        assert len(candidate) == len(reference)
        for expected, actual in zip(reference, candidate):
            torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)

    @pytest.mark.parametrize("num_windows", [1, 2])
    def test_perturbed_parameters_match_the_original_encoder(self, num_windows: int) -> None:
        """With non-identity norms and layer scales, batch 2, the rebuild matches the original to 2e-5.

        The Nano parity test runs fresh parameters, where the folded query scale, layer scales and re-implemented
        LayerNorm affine are all invisible. A 3-token chunk leaves a remainder chunk in both attention paths.
        """
        encoder = _perturbed_windowed_encoder(num_windows)
        hidden_states = torch.randn(2 * num_windows**2, 5, 32)

        with torch.no_grad():
            expected = encoder(hidden_states, output_hidden_states=True)
            actual = split_einsum_encoder(encoder, query_chunk=3)(hidden_states, output_hidden_states=True)

        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=2e-5, rtol=2e-5)

    def test_stochastic_depth_encoder_matches_in_eval_mode(self) -> None:
        """An encoder built with stochastic depth rebuilds to the same outputs, since drop path is inert in eval.

        A model trained with ``drop_path > 0`` keeps that rate in its backbone config, and exporting it right after
        training must not be refused for a layer that does nothing at inference.
        """
        encoder = _tiny_backbone(drop_path_rate=0.1).encoder.eval()
        hidden_states = torch.randn(1, 5, 32)

        with torch.no_grad():
            expected = encoder(hidden_states, output_hidden_states=True)
            actual = split_einsum_encoder(encoder)(hidden_states, output_hidden_states=True)

        torch.testing.assert_close(actual.hidden_states, expected.hidden_states, atol=1e-5, rtol=1e-5)

    def test_swiglu_encoder_is_refused(self) -> None:
        """A SwiGLU backbone is refused with a message that names the flag to turn off.

        The rebuild copies plain-MLP layers only, so converting a SwiGLU encoder would drop its MLP weights.
        """
        encoder = _tiny_backbone(use_swiglu_ffn=True).encoder

        with pytest.raises(NotImplementedError, match="coreml_neural_engine=False"):
            split_einsum_encoder(encoder)


class TestNeuralEngineModel:
    """What :func:`neural_engine_model` rewrites in a model, and what it leaves alone."""

    @pytest.mark.parametrize(
        ("bbox_reparam", "num_queries", "expected_selection", "expect_warning"),
        [
            pytest.param(True, 2048, one_hot_top_rows, False, id="one-hot-at-the-float16-rank-limit"),
            pytest.param(False, 5, select_top_rows, True, id="topk-without-bbox-reparam"),
            pytest.param(True, 2049, select_top_rows, True, id="topk-above-the-float16-rank-limit"),
        ],
    )
    def test_query_selection_falls_back_to_topk_with_a_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        bbox_reparam: bool,
        num_queries: int,
        expected_selection: object,
        expect_warning: bool,
    ) -> None:
        """Only ``bbox_reparam=True`` with at most 2048 queries gets one-hot selection; the rest keep ``topk`` and warn.

        Unreparameterized proposals can be infinite, and ``0 * inf`` is NaN in the one-hot matmul; above 2048 queries a
        float16 rank is no longer exact. Both cases must keep the CPU ``topk`` and say so.
        """
        transformer = _two_stage_transformer()
        transformer.bbox_reparam, transformer.num_queries = bbox_reparam, num_queries
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            rewritten = neural_engine_model(transformer)

        assert rewritten.select_top_rows is expected_selection
        messages = [record.getMessage() for record in caplog.records]
        assert any("Keeping top-k query selection" in message for message in messages) is expect_warning

    @pytest.mark.parametrize(
        ("build_model", "expect_warning"),
        [
            pytest.param(_tiny_backbone, False, id="windowed-backbone"),
            pytest.param(_two_stage_transformer, True, id="no-windowed-encoder"),
        ],
    )
    def test_warns_when_no_encoder_is_rewritten(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
        build_model: Callable[[], nn.Module],
        expect_warning: bool,
    ) -> None:
        """A model without a windowed DINOv2 encoder keeps its attention and says so, like the ``topk`` fallback.

        Otherwise a backbone the rewrite does not recognize would export its slow fused attention without a trace.
        """
        model = build_model()
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)

        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            neural_engine_model(model)

        messages = [record.getMessage() for record in caplog.records]
        assert any("Keeping the backbone attention unchanged" in message for message in messages) is expect_warning

    def test_input_model_is_not_modified(self) -> None:
        """The rewrite works on a copy: the caller's model keeps its encoder and its ``topk`` selection.

        The same model object is used again after export, for prediction or a second export at a different setting.
        """
        backbone, transformer = _tiny_backbone(), _two_stage_transformer()

        neural_engine_model(nn.ModuleDict({"backbone": backbone, "transformer": transformer}))

        assert type(backbone.encoder) is WindowedDinov2WithRegistersEncoder
        assert transformer.select_top_rows is select_top_rows


class TestCoreMLExporterWiring:
    """``CoreMLExporter`` must trace the rewritten copy exactly when ``neural_engine=True``."""

    @pytest.mark.parametrize("neural_engine", [True, False])
    def test_traces_the_rewritten_model_only_when_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, neural_engine: bool
    ) -> None:
        """The flag decides whether ``torch.export`` sees the Neural Engine copy or the prepared model itself.

        Without coremltools the conversion cannot run, so the rewrite and the trace are replaced at the exporter's
        references and only the hand-off between them is checked.
        """
        model, rewritten = nn.Identity(), nn.Identity()
        rewrite = mock.Mock(return_value=rewritten)
        trace = mock.Mock()
        monkeypatch.setattr("rfdetr.export._coreml.exporter.neural_engine_model", rewrite)
        monkeypatch.setattr("torch.export.export", trace)
        exporter = CoreMLExporter(
            CoreMLConfig(output_dir=tmp_path, compute_precision="float16", neural_engine=neural_engine)
        )

        exporter._export_program(_make_export_graph(model))

        assert trace.call_args.args[0] is (rewritten if neural_engine else model)
