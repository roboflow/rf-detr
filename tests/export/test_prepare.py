# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the format-independent export preparation shared by every exporter.

``prepare_export_graph`` is the single producer of the graph all six formats consume, so a mistake here reaches every
format at once — which is exactly why the work was pulled out of ``RFDETR.export()`` in the first place.
"""

from __future__ import annotations

import types

import pytest
import torch

from rfdetr.export.prepare import ExportGraph, prepare_export_graph, resolve_output_names


class _DetectorStub(torch.nn.Module):
    """Minimal detector returning the output mapping the export preparation expects.

    Examples:
        >>> sorted(_DetectorStub()(torch.zeros(1, 3, 8, 8)))
        ['pred_boxes', 'pred_logits']
    """

    def forward(self, images: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return one box and one logit row per image, ignoring pixel content."""
        batch = images.shape[0]
        return {"pred_boxes": torch.zeros(batch, 1, 4), "pred_logits": torch.zeros(batch, 1, 2)}


def _make_model_config(
    *, segmentation_head: bool = False, use_grouppose_keypoints: bool = False, projector_scale: list[str] | None = None
) -> types.SimpleNamespace:
    """Build the subset of a model config the export preparation reads.

    Args:
        segmentation_head: Whether the model predicts masks.
        use_grouppose_keypoints: Whether the model predicts keypoints.
        projector_scale: Feature levels the backbone projects.

    Returns:
        A namespace satisfying the preparation's structural config protocol.

    Examples:
        >>> _make_model_config(segmentation_head=True).segmentation_head
        True
    """
    return types.SimpleNamespace(
        num_channels=3,
        projector_scale=projector_scale or ["P4"],
        segmentation_head=segmentation_head,
        use_grouppose_keypoints=use_grouppose_keypoints,
    )


class TestResolveOutputNames:
    """Naming the graph's outputs for the model's task."""

    @pytest.mark.parametrize(
        "config_kwargs, expected",
        [
            pytest.param({}, ["dets", "labels"], id="detection"),
            pytest.param({"segmentation_head": True}, ["dets", "labels", "masks"], id="segmentation"),
            pytest.param({"use_grouppose_keypoints": True}, ["dets", "labels", "keypoints"], id="keypoints"),
        ],
    )
    def test_full_detector_names_its_task_specific_output(self, config_kwargs: dict, expected: list[str]) -> None:
        """A full-detector export names ``dets``/``labels`` plus the output its task adds.

        Consumers of an exported model match outputs by name (ONNX/TFLite) or by position (CoreML/OpenVINO), so losing
        the third name silently mislabels masks or keypoints as something else — for every format at once.
        """
        assert resolve_output_names(_make_model_config(**config_kwargs), backbone_only=False, backbone=None) == expected

    def test_backbone_export_names_one_output_per_projector_scale(self) -> None:
        """A backbone-only export emits one feature name per configured projector scale."""
        config = _make_model_config(projector_scale=["P3", "P4", "P5"])
        backbone = types.SimpleNamespace(cross_attn_projector=None)

        names = resolve_output_names(config, backbone_only=True, backbone=backbone)

        assert names == ["features", "features_1", "features_2"]

    def test_cross_attention_levels_follow_the_primary_levels(self) -> None:
        """A backbone with a separate cross-attention projector appends those levels after the primary ones."""
        config = _make_model_config(projector_scale=["P4"])
        backbone = types.SimpleNamespace(cross_attn_projector=object())

        names = resolve_output_names(config, backbone_only=True, backbone=backbone)

        assert names == ["features", "cross_attn_features"]

    def test_backbone_export_without_a_backbone_is_rejected(self) -> None:
        """Asking for backbone-only names without the backbone is a programming error, not a silent empty list."""
        with pytest.raises(ValueError, match="backbone must be provided"):
            resolve_output_names(_make_model_config(), backbone_only=True, backbone=None)


class TestPrepareExportGraph:
    """Building the traceable graph every format is handed."""

    def test_returns_a_graph_matching_the_requested_shape(self) -> None:
        """The prepared graph carries the example input, names and shape the caller asked for."""
        graph = prepare_export_graph(_DetectorStub(), _make_model_config(), shape=(16, 16), device="cpu")

        assert isinstance(graph, ExportGraph)
        assert (graph.input_tensors.shape, graph.input_names, graph.output_names, graph.shape) == (
            torch.Size([1, 3, 16, 16]),
            ("input",),
            ("dets", "labels"),
            (16, 16),
        )

    def test_static_export_carries_no_dynamic_axes(self) -> None:
        """Without ``dynamic_batch`` the graph is fully static, which is what the fixed-shape formats require."""
        graph = prepare_export_graph(_DetectorStub(), _make_model_config(), shape=(16, 16), device="cpu")

        assert graph.dynamic_axes is None

    def test_dynamic_batch_marks_every_input_and_output(self) -> None:
        """``dynamic_batch`` marks axis 0 of every graph input and output, not just the input."""
        graph = prepare_export_graph(
            _DetectorStub(), _make_model_config(), shape=(16, 16), device="cpu", dynamic_batch=True
        )

        assert graph.dynamic_axes == {"input": {0: "batch"}, "dets": {0: "batch"}, "labels": {0: "batch"}}

    def test_graph_is_left_on_cpu(self) -> None:
        """Every backend traces on CPU, so the prepared example input must be there regardless of the run device."""
        graph = prepare_export_graph(_DetectorStub(), _make_model_config(), shape=(16, 16), device="cpu")

        assert graph.input_tensors.device.type == "cpu"

    @pytest.mark.skipif(torch.cuda.is_available(), reason="exercises the CUDA-requested-but-unavailable fallback")
    def test_falls_back_to_cpu_when_cuda_is_requested_but_absent(self) -> None:
        """Requesting CUDA on a machine without it falls back to CPU (with a warning) instead of raising.

        The fallback has to happen before the example input is allocated, not just before the sanity pass: allocating on
        an unavailable device surfaces as a bare ``Torch not compiled with CUDA enabled`` from deep inside torch, which
        tells the user nothing about the export they asked for.
        """
        graph = prepare_export_graph(_DetectorStub(), _make_model_config(), shape=(16, 16), device="cuda")

        assert graph.input_tensors.device.type == "cpu"
