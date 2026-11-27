# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the ONNX graph :meth:`rfdetr.detr.RFDETR.export` produces.

Use cases covered:
- Segmentation outputs must be present in both train/eval modes to avoid export crashes.
- Export must not change the original model's training state.
- Rectangular shapes, dynamic batch axes and backbone-only exports must reach the ONNX graph intact.
- The exported graph must avoid the ops ONNX Runtime's CoreML provider and onnx2tf reject.
- The ONNX artifact's filename must follow the variant/output-name/backbone rules users glob for.
"""

import inspect
import types
import warnings
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import pytest
import torch
from torch.jit import TracerWarning

from rfdetr import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano
from rfdetr import detr as _detr_module
from rfdetr.export._onnx.exporter import OnnxConfig, OnnxExporter
from rfdetr.export.prepare import ExportGraph
from rfdetr.models.backbone.dinov2 import DinoV2
from tests._markers import _requires_cuda, _requires_onnx, _requires_onnxruntime
from tests.export.helpers import _disable_metadata_publication, _make_export_model

if TYPE_CHECKING:
    import onnx


@pytest.fixture(autouse=True)
def _skip_metadata_publication_for_export_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep facade tests focused on conversion when their fake artifacts have no file bytes.

    The metadata round-trip tests use the real writer in ``test_metadata.py``. This fixture does not replace
    ``metadata_from_model``: fake RFDETR models still need a valid prediction contract.

    Examples:
        Requires pytest's monkeypatch lifecycle, so this example cannot run standalone:
        >>> _skip_metadata_publication_for_export_stubs(None)  # doctest: +SKIP
    """
    _disable_metadata_publication(monkeypatch)


@contextmanager
def ignore_tracer_warnings() -> Iterator[None]:
    """Suppress torch.jit.TracerWarning during export tests to reduce log spam.

    Examples:
        >>> with warnings.catch_warnings(record=True) as caught:
        ...     warnings.simplefilter("always")
        ...     with ignore_tracer_warnings():
        ...         warnings.warn("traced", TracerWarning)
        >>> caught
        []
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TracerWarning)
        yield


def _run_onnx_export(
    *,
    output_dir: str,
    model: torch.nn.Module,
    input_names: Sequence[str],
    input_tensors: torch.Tensor,
    output_names: Sequence[str],
    dynamic_axes: Mapping[str, Mapping[int, str]] | None,
    backbone_only: bool = False,
    verbose: bool = True,
    opset_version: int = 17,
    variant_name: str | None = None,
    output_name: str | None = None,
) -> Path:
    """Run ``OnnxExporter`` over a throwaway graph, the way ``RFDETR.export()`` does.

    Assembles the config and the prepared graph the exporter expects so a test can exercise the real conversion
    path without building an ``RFDETR``. The keyword names mirror what ``RFDETR.export()`` accepts, which is what
    the naming assertions below are actually about.

    Args:
        output_dir: Directory the artifact is written to.
        model: Module to trace.
        input_names: Graph input names.
        input_tensors: Example input to trace with.
        output_names: Graph output names.
        dynamic_axes: Dynamic-axis mapping, or ``None`` for a static graph.
        backbone_only: Whether the graph is a backbone-only export.
        verbose: Whether the exporter logs its progress.
        opset_version: ONNX opset to target.
        variant_name: Model variant identifier used to name the artifact.
        output_name: Full filename override, without extension.

    Returns:
        Path to the exported ``.onnx`` file.

    Examples:
        >>> from tempfile import TemporaryDirectory
        >>> with TemporaryDirectory() as directory:  # doctest: +ELLIPSIS
        ...     path = _run_onnx_export(
        ...         output_dir=directory,
        ...         model=torch.nn.Identity(),
        ...         input_names=["input"],
        ...         input_tensors=torch.zeros(1, 3, 8, 8),
        ...         output_names=["dets"],
        ...         dynamic_axes=None,
        ...         verbose=False,
        ...     )
        ...     path.name
        [...] [INFO] rf-detr - Successfully exported ONNX model to: ...inference_model.onnx
        'inference_model.onnx'
    """
    config = OnnxConfig(
        output_dir=Path(output_dir),
        output_name=output_name,
        variant_name=variant_name,
        backbone_only=backbone_only,
        dynamic_batch=dynamic_axes is not None,
        verbose=verbose,
        opset_version=opset_version,
    )
    graph = ExportGraph(
        model=model,
        input_tensors=input_tensors,
        input_names=tuple(input_names),
        output_names=tuple(output_names),
        dynamic_axes=dynamic_axes,
        shape=tuple(input_tensors.shape[-2:]),
        backbone_only=backbone_only,
    )
    return OnnxExporter(config)(graph)


def test_export_onnx_uses_legacy_exporter_when_dynamo_flag_exists(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`export_onnx` should pass `dynamo=False` when supported by torch.onnx.export."""
    captured_kwargs: dict = {}

    class _ToyModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    def _fake_onnx_export(*args, **kwargs) -> None:
        captured_kwargs.update(kwargs)

    monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

    _run_onnx_export(
        output_dir=str(tmp_path),
        model=_ToyModel(),
        input_names=["images"],
        input_tensors=torch.randn(1, 3, 8, 8),
        output_names=["dets"],
        dynamic_axes={},
        verbose=False,
    )

    has_dynamo_arg = "dynamo" in inspect.signature(torch.onnx.export).parameters
    assert ("dynamo" in captured_kwargs) == has_dynamo_arg
    if has_dynamo_arg:
        assert captured_kwargs["dynamo"] is False


@pytest.mark.gpu
@_requires_cuda
@_requires_onnx
def test_segmentation_model_exports(tmp_path: Path) -> None:
    """Integration test: exporting a segmentation model writes an ONNX file instead of crashing.

    This exercises the full export path to ensure no AttributeError occurs.
    """
    model = RFDETRSegNano()

    # This should not crash with "AttributeError: 'dict' object has no attribute 'shape'"
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), verbose=False)

    # Verify export produced output files
    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


@pytest.mark.gpu
@_requires_cuda
@_requires_onnx
def test_export_accepts_rectangular_shape_differing_from_resolution(tmp_path: Path) -> None:
    """Integration test: exporting with a valid rectangular shape writes an ONNX file instead of crashing.

    A mismatched shape forces the DINOv2 backbone to interpolate its position embeddings for the new grid size. This
    must not trace an antialiased bicubic resize (``aten::_upsample_bicubic2d_aa``), which has no ONNX opset-17
    symbolic function.
    """
    model = RFDETRNano()
    block_size = model.model_config.patch_size * model.model_config.num_windows
    native_resolution = model.model.resolution
    export_shape = (native_resolution, native_resolution + block_size)
    assert all(dimension % block_size == 0 for dimension in export_shape)
    assert export_shape != (native_resolution, native_resolution)

    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path), shape=export_shape, verbose=False)

    onnx_files = list(tmp_path.glob("*.onnx"))
    assert len(onnx_files) > 0, "Export should produce ONNX file(s)"


@_requires_onnx
@pytest.mark.integration
@pytest.mark.e2e_onnx
class TestExportedGraphAvoidsCoreMLRejectedOps:
    """A real whole-model export must avoid the ops ONNX Runtime's CoreML provider rejects — the unit tests in
    ``tests/models/transformer/test_onnx_two_stage.py`` and ``test_segmentation_head.py`` only check the ``Transformer``
    and ``SegmentationHead`` submodules in isolation, never a full detection or segmentation graph as
    ``RFDETR.export()`` actually produces it.

    ``pretrain_weights=None`` builds each model without downloading or loading a checkpoint, keeping this CPU-runnable
    and fast.
    """

    @staticmethod
    def _exported_graph(tmp_path: Path, model: object) -> "onnx.GraphProto":
        """Export ``model`` to ONNX under ``tmp_path`` and return its shape-inferred graph.

        Args:
            tmp_path: Directory to export into; must contain no other ``*.onnx`` file.
            model: An ``RFDETR*`` model instance with an ``export`` method.

        Returns:
            The exported model's graph, after :func:`onnx.shape_inference.infer_shapes`.

        Examples:
            Requires a full model export to a temp dir; exercised by the tests below instead.
            >>> TestExportedGraphAvoidsCoreMLRejectedOps._exported_graph(tmp_path, model)  # doctest: +SKIP
        """
        import onnx

        with ignore_tracer_warnings():
            model.export(output_dir=str(tmp_path), verbose=False)
        (onnx_path,) = tmp_path.glob("*.onnx")
        return onnx.shape_inference.infer_shapes(onnx.load(str(onnx_path))).graph

    @pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSegNano])
    def test_graph_has_no_rejected_ops(
        self, tmp_path: Path, model_class: type[RFDETRNano] | type[RFDETRSegNano]
    ) -> None:
        """A whole-model detection or segmentation graph holds no ``Einsum`` and no degenerate ``Concat`` node."""
        from tests.models.transformer.test_onnx_two_stage import (
            find_single_input_concat_nodes,
            find_zero_dim_float_concat_nodes,
        )

        graph = self._exported_graph(tmp_path, model_class(pretrain_weights=None))

        assert "Einsum" not in {node.op_type for node in graph.node}
        assert find_single_input_concat_nodes(graph) == []
        assert find_zero_dim_float_concat_nodes(graph) == []


def test_dinov2_export_uses_precomputed_positions_for_exact_rectangular_grid() -> None:
    """DINOv2 export must bypass interpolation only for its precomputed rectangular grid."""
    patch_size = 8
    export_shape = (32, 48)
    source_positions = torch.nn.Parameter(torch.randn(1, 17, 2))
    original_calls: list[tuple[int, int]] = []
    fallback_positions = torch.randn(1, 1, 2)

    def original_interpolate_pos_encoding(_embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        """Record fallback interpolation calls made by the exported backbone."""
        original_calls.append((height, width))
        return fallback_positions

    backbone = DinoV2.__new__(DinoV2)
    torch.nn.Module.__init__(backbone)
    backbone.shape = export_shape
    backbone._export = False
    backbone.encoder = types.SimpleNamespace(
        config=types.SimpleNamespace(patch_size=patch_size),
        embeddings=types.SimpleNamespace(
            position_embeddings=source_positions,
            interpolate_pos_encoding=original_interpolate_pos_encoding,
        ),
    )

    backbone.export()
    patch_count = (export_shape[0] // patch_size) * (export_shape[1] // patch_size)
    embeddings = torch.randn(1, patch_count + 1, 2)

    fixed_positions = backbone.encoder.embeddings.interpolate_pos_encoding(embeddings, *export_shape)
    assert fixed_positions is backbone.encoder.embeddings.position_embeddings
    assert original_calls == []

    transposed_positions = backbone.encoder.embeddings.interpolate_pos_encoding(
        embeddings, export_shape[1], export_shape[0]
    )
    assert transposed_positions is fallback_positions
    assert original_calls == [(export_shape[1], export_shape[0])]


@pytest.mark.gpu
@_requires_cuda
@_requires_onnx
def test_export_does_not_change_original_training_state(tmp_path: Path) -> None:
    """Verify that calling export() does not change the original model's train/eval state.

    This ensures that export() puts a deepcopy of the model in eval mode without mutating the underlying training model
    used by RF-DETR.
    """
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model), as in other tests
    torch_model = model.model.model.to("cuda")

    # Ensure the original model is in training mode
    torch_model.train()
    assert torch_model.training is True, "Precondition: original model should start in training mode"

    # Call export() on the high-level model; this should not change the original model's mode
    with ignore_tracer_warnings():
        model.export(output_dir=str(tmp_path))

    # After export, the original underlying model should still be in training mode
    assert torch_model.training is True, "export() should not change the original model's training state"


@pytest.mark.parametrize(
    "dynamic_batch, segmentation_head",
    [
        (True, False),
        (True, True),
        (False, False),
    ],
)
def test_rfdetr_export_dynamic_batch_forwards_dynamic_axes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    dynamic_batch: bool,
    segmentation_head: bool,
) -> None:
    """`dynamic_batch=True` forwards a batch-axis `dynamic_axes` dict to the converter; `False` forwards `None`."""
    model = _make_export_model(segmentation_head=segmentation_head)

    captured: dict = {}

    def _fake_make_infer_image(*_args, **_kwargs):
        return torch.zeros(1, 3, 14, 14)

    def _fake_convert(_self, graph):
        captured["dynamic_axes"] = graph.dynamic_axes
        return str(tmp_path / "inference_model.onnx")

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), dynamic_batch=dynamic_batch, shape=(14, 14))

    dynamic_axes = captured.get("dynamic_axes")
    if not dynamic_batch:
        assert dynamic_axes is None, f"expected None for static export, got {dynamic_axes!r}"
        return

    assert isinstance(dynamic_axes, dict), f"expected dict, got {dynamic_axes!r}"
    for name, axes in dynamic_axes.items():
        assert axes == {0: "batch"}, f"axis spec for {name!r} should be {{0: 'batch'}}, got {axes!r}"

    expected_names = {"input", "dets", "labels", "masks"} if segmentation_head else {"input", "dets", "labels"}
    assert set(dynamic_axes.keys()) == expected_names, f"expected keys {expected_names}, got {set(dynamic_axes.keys())}"


@pytest.mark.gpu
@_requires_cuda
@pytest.mark.parametrize("mode", ["train", "eval"])
def test_segmentation_outputs_present_in_train_and_eval(mode: Literal["train", "eval"]) -> None:
    """Use case: segmentation outputs are present in both train and eval modes."""
    model = RFDETRSegNano()

    # Access the underlying torch module (model.model.model)
    torch_model = model.model.model.to("cuda")

    # Use resolution compatible with model's patch size (312 for seg-nano)
    resolution = model.model.resolution
    dummy_input = torch.randn(1, 3, resolution, resolution, device="cuda")

    if mode == "train":
        torch_model.train()
    else:
        torch_model.eval()

    with torch.no_grad():
        output = torch_model(dummy_input)

    assert "pred_boxes" in output
    assert "pred_logits" in output
    assert "pred_masks" in output


def test_make_infer_image_keeps_rectangular_shape() -> None:
    """make_infer_image must produce a (B, C, H, W) tensor for non-square shapes.

    Regression test for the square-resize bug where ``Resize((shape[0], shape[0]))`` was used instead of
    ``Resize((shape[0], shape[1]))``, causing the output width to silently equal the height.
    """
    from rfdetr.export.prepare import make_infer_image

    h, w, b = 112, 224, 2
    tensor = make_infer_image(infer_dir=None, shape=(h, w), batch_size=b, device="cpu")
    assert tensor.shape == (b, 3, h, w), f"Expected shape ({b}, 3, {h}, {w}), got {tensor.shape}"


# ---------------------------------------------------------------------------
# ONNX export variant naming
# ---------------------------------------------------------------------------


class TestExportOnnxVariantNaming:
    """Verify that export_onnx uses variant_name in the output filename."""

    @pytest.mark.parametrize(
        "variant_name, backbone_only, output_name, output_names, expected_suffix",
        [
            ("rfdetr-medium", False, None, ["dets"], "rfdetr-medium.onnx"),
            ("rfdetr-nano", True, None, ["features"], "rfdetr-nano-backbone.onnx"),
            (None, False, None, ["dets"], "inference_model.onnx"),
            (None, True, None, ["features"], "backbone_model.onnx"),
            ("rfdetr-medium", False, "my-model", ["dets"], "my-model.onnx"),
            (None, True, "my-model", ["features"], "my-model-backbone.onnx"),
            pytest.param("", False, None, ["dets"], "inference_model.onnx", id="empty_variant_name"),
            ("foo/bar", False, None, ["dets"], "bar.onnx"),
            ("/tmp/x", False, None, ["dets"], "x.onnx"),
        ],
    )
    def test_artifact_filename(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        variant_name: str | None,
        backbone_only: bool,
        output_name: str | None,
        output_names: list[str],
        expected_suffix: str,
    ) -> None:
        """The ONNX artifact filename follows the variant/output-name/backbone-suffix naming rules.

        Covers variant_name alone, variant_name + backbone_only (appends '-backbone'), no or empty variant_name (falls
        back to the default name), path separators stripped from variant_name, output_name overriding variant_name, and
        output_name + backbone_only still appending the structural '-backbone' suffix.
        """
        captured: dict = {}

        def _fake_onnx_export(*args, **kwargs) -> None:
            captured["output_file"] = args[2]  # 3rd positional arg is output_file

        monkeypatch.setattr(torch.onnx, "export", _fake_onnx_export)

        _run_onnx_export(
            output_dir=str(tmp_path),
            model=torch.nn.Identity(),
            input_names=["input"],
            input_tensors=torch.randn(1, 3, 8, 8),
            output_names=output_names,
            dynamic_axes=None,
            backbone_only=backbone_only,
            verbose=False,
            variant_name=variant_name,
            output_name=output_name,
        )

        assert captured["output_file"].endswith(expected_suffix)

    @pytest.mark.parametrize(
        "size, output_name_kwarg, expected_variant_name, expected_output_name",
        [
            ("rfdetr-medium", None, "rfdetr-medium", None),
            (None, None, None, None),
            ("rfdetr-medium", "my-model", "rfdetr-medium", "my-model"),
        ],
    )
    def test_rfdetr_export_passes_variant_and_output_name(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
        size: str | None,
        output_name_kwarg: str | None,
        expected_variant_name: str | None,
        expected_output_name: str | None,
    ) -> None:
        """RFDETR.export() forwards self.size as variant_name, and any explicit output_name, to export_onnx.

        size=None must forward variant_name=None (base RFDETR has no size), and an explicit output_name kwarg must reach
        the exporter config alongside variant_name rather than replacing it there.
        """
        captured: dict = {}

        model = _make_export_model(size=size)

        def _fake_make_infer_image(*_args, **_kwargs):
            return torch.zeros(1, 3, 14, 14)

        def _fake_convert(self, _graph):
            captured["variant_name"] = self.config.variant_name
            captured["output_name"] = self.config.output_name
            return str(tmp_path / "inference_model.onnx")

        monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", _fake_make_infer_image)
        monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", _fake_convert)
        monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)

        export_kwargs = {"output_dir": str(tmp_path), "shape": (14, 14)}
        if output_name_kwarg is not None:
            export_kwargs["output_name"] = output_name_kwarg
        _detr_module.RFDETR.export(model, **export_kwargs)

        assert captured["variant_name"] == expected_variant_name
        assert captured["output_name"] == expected_output_name


@pytest.mark.gpu
@_requires_onnx
@_requires_onnxruntime
@pytest.mark.parametrize("model_class", [RFDETRNano, RFDETRSegNano, RFDETRKeypointPreview])
@pytest.mark.parametrize("projector_scale", [["P4"], ["P4", "P5"]])
@pytest.mark.parametrize("dynamic_batch", [False, True])
def test_backbone_only_exports_features_without_detector(
    tmp_path: Path,
    model_class: type[RFDETRNano] | type[RFDETRSegNano] | type[RFDETRKeypointPreview],
    projector_scale: list[str],
    dynamic_batch: bool,
) -> None:
    """The public backbone export runs independently of detector heads and preserves every feature level."""
    import onnxruntime as ort

    model = model_class(
        pretrain_weights=None,
        device="cpu",
        resolution=96,
        num_queries=4,
        num_select=4,
        num_classes=2,
        projector_scale=projector_scale,
    )
    core = model.model.model
    assert core is not None
    core.eval()
    batch = 2 if dynamic_batch else 1
    inputs = torch.linspace(-1, 1, batch * 3 * 96 * 192).reshape(batch, 3, 96, 192)
    backbone = core.backbone[0]
    with torch.no_grad():
        raw_features = backbone.encoder(inputs)
        expected = backbone.projector(raw_features)
        if backbone.cross_attn_projector is not None:
            expected.extend(backbone.cross_attn_projector(raw_features))
    with ignore_tracer_warnings():
        path = model.export(
            output_dir=str(tmp_path), backbone_only=True, shape=(96, 192), dynamic_batch=dynamic_batch, verbose=False
        )
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
    expected_names = ["features" if index == 0 else f"features_{index}" for index in range(len(projector_scale))]
    if backbone.cross_attn_projector is not None:
        expected_names.extend(
            "cross_attn_features" if index == 0 else f"cross_attn_features_{index}"
            for index in range(len(projector_scale))
        )
    assert [output.name for output in session.get_outputs()] == expected_names
    actual = session.run(None, {session.get_inputs()[0].name: inputs.numpy()})
    assert len(actual) == len(expected)
    for exported, reference in zip(actual, expected):
        np.testing.assert_allclose(exported, reference.numpy(), atol=1e-4, rtol=1e-4)
    assert not backbone._export
    assert model.model.model is core


def test_onnx_exporter_creates_a_missing_output_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The ONNX stage must create its own output directory.

    The composed TFLite and TensorRT exporters route their intermediate ONNX through ``OnnxExporter`` before creating
    their own output directory, so it cannot rely on a caller having made it.
    """
    monkeypatch.setattr(torch.onnx, "export", lambda *_args, **_kwargs: None)
    output_dir = tmp_path / "missing" / "nested"
    graph = ExportGraph(
        model=torch.nn.Identity(),
        input_tensors=torch.zeros(1, 2),
        input_names=("input",),
        output_names=("output",),
        dynamic_axes=None,
        shape=(2, 2),
        backbone_only=False,
    )

    OnnxExporter(OnnxConfig(output_dir=output_dir, verbose=False))(graph)

    assert output_dir.is_dir()


def _keypoint_axis_broadcasts(model: "onnx.ModelProto") -> list[str]:
    """Name every elementwise node that broadcasts one input over axis -2 only, the keypoint axis of the decode.

    That is the pattern onnx2tf mis-transposes: a ``(B, Q, 1, 2)`` box reference broadcast against ``(B, Q, K, 2)``
    keypoint deltas made the TFLite conversion of keypoint models fail (#1514). Shapes are right-aligned, so a
    rank-3 ``(Q, 1, 2)`` input counts too. A full broadcast such as ``(1, 1, 1, 2)``, which onnx2tf converts, does not.

    Args:
        model: The ONNX model; shapes are inferred before the scan.

    Returns:
        The names of the offending nodes, in graph order.

    Raises:
        ValueError: If a possible keypoint-axis broadcast has an unknown keypoint
            dimension, or no eligible node was checked.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> def _elementwise(ref_shape: list[int]) -> "onnx.ModelProto":
        ...     graph = helper.make_graph(
        ...         [helper.make_node("Mul", ["delta", "ref"], ["out"], name="decode")],
        ...         "g",
        ...         [
        ...             helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 4, 3, 2]),
        ...             helper.make_tensor_value_info("ref", TensorProto.FLOAT, ref_shape),
        ...         ],
        ...         [helper.make_tensor_value_info("out", TensorProto.FLOAT, None)],
        ...     )
        ...     return helper.make_model(graph)
        >>> _keypoint_axis_broadcasts(_elementwise([1, 4, 1, 2]))
        ['decode']
        >>> _keypoint_axis_broadcasts(_elementwise([4, 1, 2]))
        ['decode']
        >>> _keypoint_axis_broadcasts(_elementwise([1, 1, 1, 2]))
        []
        >>> _keypoint_axis_broadcasts(_elementwise([1, 4, 3, 2]))
        []
    """
    from onnx import shape_inference

    graph = shape_inference.infer_shapes(model).graph
    shapes = {
        value.name: [dim.dim_value if dim.HasField("dim_value") else None for dim in value.type.tensor_type.shape.dim]
        for value in [*graph.input, *graph.value_info, *graph.output]
    }
    shapes.update({initializer.name: list(initializer.dims) for initializer in graph.initializer})
    offenders = []
    checked = 0
    for node in graph.node:
        if node.op_type not in ("Mul", "Add", "Sub", "Div"):
            continue
        output_shape = shapes.get(node.output[0])
        if not output_shape or len(output_shape) != 4 or output_shape[-2] == 1:
            continue
        if output_shape[-2] is None:
            for name in node.input:
                input_shape = shapes.get(name) or []
                if (
                    2 < len(input_shape) <= 4
                    and input_shape[-2] == 1
                    and output_shape[-3] not in (None, 1)
                    and input_shape[-3] == output_shape[-3]
                ):
                    raise ValueError(
                        f"unknown keypoint axis at elementwise node {node.name!r}; shape check is incomplete"
                    )
            continue
        checked += 1
        for name in node.input:
            input_shape = shapes.get(name) or []
            if not 2 < len(input_shape) <= 4:
                continue
            if input_shape[-2] == 1 and output_shape[-3] not in (None, 1) and input_shape[-3] == output_shape[-3]:
                offenders.append(node.name)
                break
    if not checked:
        raise ValueError(
            "no rank-4 elementwise node with a known keypoint axis; shape inference found nothing to check"
        )
    return offenders


def _if_nodes(model: "onnx.ModelProto") -> list[str]:
    """Name every ``If`` node; onnx2tf's onnxsim pass cannot simplify one, and conversion then fails on ``Expand``.

    Args:
        model: The ONNX model.

    Returns:
        The names of the ``If`` nodes, in graph order.

    Examples:
        >>> from onnx import TensorProto, helper
        >>> branch = helper.make_graph([], "branch", [], [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])])
        >>> node = helper.make_node("If", ["cond"], ["y"], name="rank_check", then_branch=branch, else_branch=branch)
        >>> graph = helper.make_graph(
        ...     [node],
        ...     "g",
        ...     [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        ...     [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        ... )
        >>> _if_nodes(helper.make_model(graph))
        ['rank_check']
    """
    return [node.name for node in model.graph.node if node.op_type == "If"]


@_requires_onnx
class TestKeypointOnnxGraphAvoidsOnnx2tfBlockers:
    """The keypoint ONNX graph must avoid the constructs that broke its onnx2tf/TFLite conversion (#1514).

    A real onnx2tf conversion of a keypoint model takes over ten minutes on CPU, so these tests pin the graph constructs
    onnx2tf could not handle instead.
    """

    @pytest.fixture(scope="class")
    @staticmethod
    def keypoint_onnx(tmp_path_factory: pytest.TempPathFactory) -> "onnx.ModelProto":
        """Export a small, randomly initialized keypoint model to ONNX once and load the graph.

        Args:
            tmp_path_factory: Pytest's session-scoped temporary directory factory.

        Returns:
            The exported ONNX model, with a ``dets``/``labels``/``keypoints`` output contract.
        """
        import onnx

        model = RFDETRKeypointPreview(
            pretrain_weights=None,
            device="cpu",
            resolution=96,
            num_queries=4,
            num_classes=2,
            num_keypoints_per_class=[3],
        )
        with ignore_tracer_warnings():
            path = model.export(output_dir=str(tmp_path_factory.mktemp("keypoint_onnx")), verbose=False)
        return onnx.load(str(path))

    @pytest.mark.parametrize(
        "find_blockers",
        [
            pytest.param(_keypoint_axis_broadcasts, id="keypoint_axis_broadcast"),
            pytest.param(_if_nodes, id="if_node"),
        ],
    )
    def test_graph_has_no_onnx2tf_blockers(
        self, keypoint_onnx: "onnx.ModelProto", find_blockers: Callable[["onnx.ModelProto"], list[str]]
    ) -> None:
        """No node in the exported keypoint graph matches a construct onnx2tf fails to convert."""
        assert find_blockers(keypoint_onnx) == []

    def test_unknown_keypoint_axis_does_not_pass_after_checking_an_unrelated_node(self) -> None:
        """A symbolic keypoint dimension must not be hidden by an unrelated eligible elementwise node."""
        from onnx import TensorProto, helper

        checked_node = helper.make_node("Add", ["left", "right"], ["checked"], name="unrelated")
        keypoint_node = helper.make_node("Mul", ["delta", "reference"], ["keypoints"], name="decode")
        graph = helper.make_graph(
            [checked_node, keypoint_node],
            "keypoint_unknown_axis",
            [
                helper.make_tensor_value_info("left", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("right", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("delta", TensorProto.FLOAT, [1, 4, "keypoints", 2]),
                helper.make_tensor_value_info("reference", TensorProto.FLOAT, [1, 4, 1, 2]),
            ],
            [
                helper.make_tensor_value_info("checked", TensorProto.FLOAT, [1, 4, 3, 2]),
                helper.make_tensor_value_info("keypoints", TensorProto.FLOAT, [1, 4, "keypoints", 2]),
            ],
        )
        model = helper.make_model(graph)

        with pytest.raises(ValueError, match="unknown keypoint axis"):
            _keypoint_axis_broadcasts(model)
