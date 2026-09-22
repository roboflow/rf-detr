# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for native CoreML (``.mlpackage``) export.

Covers:
* ``CoreMLExporter`` — argument/dependency behaviour (``coremltools`` mocked where needed)
* ``format="coreml"`` wiring through ``RFDETR.export()``
* End-to-end convert + numerical parity (``@pytest.mark.e2e_coreml``, opt-in)

Parity inputs are spatially structured (gradient + checkerboard) and
``download_assets(ImageAssets.PEOPLE_WALKING)`` under ``e2e_coreml`` (no committed images).
Random Gaussian noise is intentionally avoided: it can hide export/runtime divergence that
only appears on correlated image structure.
"""

from __future__ import annotations

import contextlib
import os
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from PIL import Image
from supervision.assets import ImageAssets, download_assets

import rfdetr
from rfdetr.export._backend import _BackboneExport
from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE
from rfdetr.export._coreml.exporter import CoreMLConfig, CoreMLExporter, _check_coremltools_available
from rfdetr.export.prepare import ExportGraph
from rfdetr.utilities.reproducibility import seed_all
from tests.export.conftest import (
    _parity_input_from_image,
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)

coreml_only = pytest.mark.skipif(not _IS_COREMLTOOLS_AVAILABLE, reason="coremltools not installed")

# FLOAT32 CoreML convert matches eager to ~1e-5 on boxes/logits; masks need a bit more
# headroom (~8e-5 observed on SegNano; keypoints ~1e-5 observed on KeypointPreview). Bound stays
# well under structural-failure scale (>=1e-3).
_COREML_MAX_ABS_DIFF = 1e-4


def _coreml_parity_diffs(
    mlpackage_path: Path,
    pytorch_model: torch.nn.Module,
    example_input: torch.Tensor,
) -> list[float]:
    """Run *example_input* through eager export-mode PyTorch and CoreML; return per-output max-abs-diff.

    Export-mode ``forward`` can mutate its input, so each side gets a fresh clone. CoreML outputs are
    taken in ``MLModel`` spec order and paired with the eager tuple in the same order.

    Args:
        mlpackage_path: Path to the exported ``.mlpackage``.
        pytorch_model: Export-mode PyTorch module on CPU.
        example_input: ``(N, C, H, W)`` example tensor.

    Returns:
        One max-abs-diff per output tensor.

    Raises:
        AssertionError: If output counts or shapes disagree.
        ImportError: If ``coremltools`` is not installed.

    Examples:
        Requires a real exported ``.mlpackage`` and ``coremltools`` — not runnable standalone.
        See ``TestCoreMLEndToEnd`` for real invocations.

        >>> callable(_coreml_parity_diffs)
        True
    """
    import coremltools as ct

    eager_tensors = eager_reference_tensors(pytorch_model, example_input)

    # CPU_ONLY avoids ANE/GPU fp16 execution drift when validating FLOAT32 bundles.
    mlmodel = ct.models.MLModel(str(mlpackage_path), compute_units=ct.ComputeUnit.CPU_ONLY)
    spec = mlmodel.get_spec()
    input_name = spec.description.input[0].name
    output_names = [o.name for o in spec.description.output]
    prediction = mlmodel.predict({input_name: np.ascontiguousarray(example_input.detach().cpu().numpy())})
    coreml_tensors = [torch.from_numpy(np.asarray(prediction[name], dtype=np.float32)) for name in output_names]

    return max_abs_output_diffs(eager_tensors, coreml_tensors, check_shape=True, names=output_names)


def _validate_coreml_vs_pytorch(
    mlpackage_path: Path,
    pytorch_model: torch.nn.Module,
    example_input: torch.Tensor,
    *,
    output_labels: tuple[str, ...],
) -> None:
    """Compare every CoreML output to eager export-mode PyTorch and assert parity within tolerance.

    ``output_labels`` names the outputs the export is expected to yield, in order — ``("boxes", "logits")`` for
    detection, plus ``"masks"`` or ``"keypoints"`` as the third slot for segmentation and keypoint heads. Masks and
    keypoints share that slot and are both rank-4, so the count alone does not distinguish them; per-output shapes
    are asserted by :func:`_coreml_parity_diffs` against eager.

    Args:
        mlpackage_path: Path to the exported ``.mlpackage``.
        pytorch_model: Export-mode PyTorch module on CPU.
        example_input: ``(N, C, H, W)`` tensor used for both forwards.
        output_labels: Expected output names in export order; the count must match the exported outputs.

    Raises:
        AssertionError: When output count/shape disagrees or max-abs-diff exceeds tolerance.

    Examples:
        Stub the CoreML comparison so the validator can run without a real export.

        >>> with mock.patch(
        ...     f"{_validate_coreml_vs_pytorch.__module__}._coreml_parity_diffs",
        ...     return_value=[0.0, 0.0],
        ... ):
        ...     _validate_coreml_vs_pytorch(
        ...         Path("model.mlpackage"),
        ...         torch.nn.Identity(),
        ...         torch.zeros(1, 3, 1, 1),
        ...         output_labels=("boxes", "logits"),
        ...     )

        A mismatched output count is reported against the expected labels.

        >>> with mock.patch(
        ...     f"{_validate_coreml_vs_pytorch.__module__}._coreml_parity_diffs",
        ...     return_value=[0.0, 0.0],
        ... ):
        ...     _validate_coreml_vs_pytorch(
        ...         Path("model.mlpackage"),
        ...         torch.nn.Identity(),
        ...         torch.zeros(1, 3, 1, 1),
        ...         output_labels=("boxes", "logits", "masks"),
        ...     )
        Traceback (most recent call last):
            ...
        AssertionError: CoreML export must yield ('boxes', 'logits', 'masks'), got 2 outputs
        ...
    """
    diffs = _coreml_parity_diffs(mlpackage_path, pytorch_model, example_input)
    assert len(diffs) == len(output_labels), f"CoreML export must yield {output_labels}, got {len(diffs)} outputs"
    per_output = ", ".join(f"{label}={diff}" for label, diff in zip(output_labels, diffs))
    assert max(diffs) < _COREML_MAX_ABS_DIFF, (
        f"CoreML outputs diverge from PyTorch: max abs diff {max(diffs)} ({per_output}, bound={_COREML_MAX_ABS_DIFF})"
    )


def _mil_op_counts(spec: Any) -> Counter[str]:
    """Return per-type MIL operation counts in *spec*'s active ``main`` block, nested blocks included.

    A multiset rather than a set of types: type membership alone cannot see a shape-driven lowering route that
    reuses an op type both graphs already contain but emits far more of it. ``cond``/``while_loop`` carry their
    body in ``Operation.blocks``, so a top-level-only walk would silently stop reporting once such an op appears.

    Args:
        spec: A CoreML ``Model`` protobuf (``coremltools.utils.load_spec``) holding an ``mlProgram``.

    Returns:
        A count of every ``Operation.type`` reachable from the ``main`` function's active block specialization.
        Empty when *spec* holds no ``mlProgram``: protobuf message maps default-construct on lookup rather than
        raising, so callers must treat an empty result as "traversal found nothing", not as "the graphs agree".

    Examples:
        >>> from types import SimpleNamespace
        >>> def op(type_, *blocks):
        ...     return SimpleNamespace(type=type_, blocks=list(blocks))
        >>> main = SimpleNamespace(
        ...     opset="CoreML8",
        ...     block_specializations={
        ...         "CoreML8": SimpleNamespace(
        ...             operations=[op("const"), op("const"), op("cond", SimpleNamespace(operations=[op("linear")]))]
        ...         ),
        ...         "CoreML7": SimpleNamespace(operations=[op("matmul")]),
        ...     },
        ... )
        >>> sorted(_mil_op_counts(SimpleNamespace(mlProgram=SimpleNamespace(functions={"main": main}))).items())
        [('cond', 1), ('const', 2), ('linear', 1)]
    """
    main = spec.mlProgram.functions["main"]
    counts: Counter[str] = Counter()
    pending = list(main.block_specializations[main.opset].operations)
    while pending:
        operation = pending.pop()
        counts[operation.type] += 1
        pending.extend(nested_op for block in operation.blocks for nested_op in block.operations)
    return counts


def _mil_op_types(spec: Any) -> set[str]:
    """Return the distinct MIL operation types in *spec*'s active ``main`` block, nested blocks included.

    Args:
        spec: A CoreML ``Model`` protobuf (``coremltools.utils.load_spec``) holding an ``mlProgram``.

    Returns:
        Every ``Operation.type`` reachable from the ``main`` function's active block specialization (see
        :func:`_mil_op_counts` for the traversal and the empty-result caveat).

    Examples:
        >>> from types import SimpleNamespace
        >>> def op(type_, *blocks):
        ...     return SimpleNamespace(type=type_, blocks=list(blocks))
        >>> main = SimpleNamespace(
        ...     opset="CoreML8",
        ...     block_specializations={
        ...         "CoreML8": SimpleNamespace(
        ...             operations=[op("const"), op("cond", SimpleNamespace(operations=[op("linear")]))]
        ...         ),
        ...         "CoreML7": SimpleNamespace(operations=[op("matmul")]),
        ...     },
        ... )
        >>> sorted(_mil_op_types(SimpleNamespace(mlProgram=SimpleNamespace(functions={"main": main}))))
        ['cond', 'const', 'linear']
    """
    return set(_mil_op_counts(spec))


#: Max per-op-type count the shipped-query graph may exceed the value-checked (5-query) graph by before
#: ``test_default_query_count_lowers_through_no_unchecked_operation`` flags it. Ordinary constant folding at the
#: larger shipped shape can trim a handful of ``const``/``reshape`` nodes relative to the smaller checked shape;
#: this value is a conservative starting point, not an empirical measurement like ``_MIN_TWO_STAGE_RANK_MARGIN`` --
#: widen it if it proves noisy, narrow it if a real regression sneaks in under it.
_MIL_OP_COUNT_TOLERANCE = 2


def _mil_op_count_regressions(
    shipped: Counter[str], checked: Counter[str], *, tolerance: int
) -> dict[str, tuple[int, int]]:
    """Return MIL op types where *shipped* exceeds *checked* by more than *tolerance* occurrences.

    One-directional by design: an op type *shipped* reaches far more than *checked* (including one *checked*
    never reaches at all, i.e. a checked count of 0) is the gap worth flagging; the reverse -- a node folded away
    only at the larger shipped shape -- is benign, same rationale as :func:`_mil_op_types`.

    Args:
        shipped: Op-type counts from the shipped-query-count graph.
        checked: Op-type counts from the value-checked (5-query) graph.
        tolerance: Maximum benign per-type count difference before a type is reported.

    Returns:
        ``{op_type: (shipped_count, checked_count)}`` for every regression found; empty when none.

    Examples:
        >>> _mil_op_count_regressions(
        ...     Counter(const=5, slice_by_index=3), Counter(const=4, slice_by_index=0), tolerance=2
        ... )
        {'slice_by_index': (3, 0)}
        >>> _mil_op_count_regressions(Counter(const=5), Counter(const=4), tolerance=2)
        {}
    """
    return {
        op_type: (shipped_count, checked[op_type])
        for op_type, shipped_count in shipped.items()
        if shipped_count - checked[op_type] > tolerance
    }


# ---------------------------------------------------------------------------
# CoreMLExporter — unit / dependency behaviour
# ---------------------------------------------------------------------------


def _make_export_graph(model: torch.nn.Module, *, backbone_only: bool = False) -> ExportGraph:
    """Build a minimal prepared graph a :class:`CoreMLExporter` can be handed without a real detector.

    Args:
        model: Stand-in module the exporter traces; a plain ``nn.Module`` exposes no ``export`` method
            for the export-mode switch to call, so nothing about the real detector is needed here.
        backbone_only: Whether the graph stands in for a backbone-only export.

    Returns:
        An :class:`ExportGraph` wrapping *model* with a ``1x3x32x32`` example input.

    Examples:
        >>> _make_export_graph(torch.nn.Identity(), backbone_only=True).backbone_only
        True
    """
    return ExportGraph(
        model=model,
        input_tensors=torch.zeros(1, 3, 32, 32),
        input_names=("input",),
        output_names=("dets", "labels"),
        dynamic_axes=None,
        shape=(32, 32),
        backbone_only=backbone_only,
    )


class TestCheckCoremltoolsAvailable:
    """Tests for ``_check_coremltools_available``."""

    def test_returns_false_when_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing ``coremltools`` must return ``False``."""
        # _check_coremltools_available reads the module-level _IS_COREMLTOOLS_AVAILABLE flag
        # (computed once, at rfdetr.export._coreml import time) rather than importing
        # coremltools live, so the flag itself — as bound into converter's namespace by its
        # `from rfdetr.export._coreml import _IS_COREMLTOOLS_AVAILABLE` — is what must be patched.
        monkeypatch.setattr("rfdetr.export._coreml.exporter._IS_COREMLTOOLS_AVAILABLE", False)
        assert _check_coremltools_available(raise_error=False) is False

    @coreml_only
    def test_returns_true_when_installed(self) -> None:
        """Installed ``coremltools`` must return ``True``."""
        assert _check_coremltools_available() is True


class TestExportCoremlValidation:
    """Argument and dependency behaviour of ``CoreMLExporter`` (no real convert)."""

    def test_missing_coremltools_raises_import_error(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """``CoreMLExporter`` must surface the install hint when coremltools is absent."""
        monkeypatch.setattr(
            "rfdetr.export._coreml.exporter._check_coremltools_available",
            mock.Mock(side_effect=ImportError("pip install rfdetr[coreml]")),
        )
        exporter = CoreMLExporter(CoreMLConfig(output_dir=tmp_path))
        with pytest.raises(ImportError, match="rfdetr\\[coreml\\]"):
            exporter(_make_export_graph(torch.nn.Linear(1, 1)))


class TestExportCoremlBareDefaultNaming:
    """``variant_name=None`` + ``output_name=None`` combined with a non-default ``compute_precision`` (fp16).

    Real ``coremltools.convert``/``torch.export.export`` are mocked out so this stays a fast unit test — only the naming
    path (``resolve_export_stem`` -> precision-suffix branch) is under test, not conversion correctness (that is covered
    by ``TestCoreMLEndToEnd``, gated on a real ``coremltools`` install).
    """

    @coreml_only
    def test_bare_default_stem_with_non_default_fp16_precision(self, tmp_path: Path) -> None:
        """No variant_name/output_name (bare default stem) plus fp16 (non-default precision) must still produce
        ``inference_model_fp16.mlpackage`` — the precision suffix is not dropped just because the stem is generic."""
        mock_exported_program = mock.MagicMock()
        mock_exported_program.run_decompositions.return_value = mock_exported_program
        mock_mlmodel = mock.MagicMock()

        with (
            mock.patch("torch.export.export", return_value=mock_exported_program),
            mock.patch("rfdetr.export._coreml.exporter.unsupported_coreml_ops", return_value={}),
            mock.patch("coremltools.convert", return_value=mock_mlmodel),
        ):
            exporter = CoreMLExporter(
                CoreMLConfig(
                    output_dir=tmp_path,
                    variant_name=None,
                    output_name=None,
                    compute_precision="float16",
                    verbose=False,
                )
            )
            output_file = exporter(_make_export_graph(torch.nn.Identity()))

        assert output_file.name == "inference_model_fp16.mlpackage"
        mock_mlmodel.save.assert_called_once_with(str(output_file))

    @pytest.mark.parametrize(
        "variant_name, output_name, full_name, backbone_name",
        [
            ("rfdetr-nano", None, "rfdetr-nano_fp32.mlpackage", "rfdetr-nano_fp32-backbone.mlpackage"),
            ("rfdetr-nano", "custom", "custom.mlpackage", "custom-backbone.mlpackage"),
            (None, None, "inference_model_fp32.mlpackage", "backbone_model_fp32.mlpackage"),
        ],
    )
    def test_backbone_only_does_not_collide_with_full_detector_export(
        self,
        tmp_path: Path,
        variant_name: str | None,
        output_name: str | None,
        full_name: str,
        backbone_name: str,
    ) -> None:
        """Backbone and detector paths stay distinct with variant, custom, and default names."""
        coremltools = mock.MagicMock()
        full_model, backbone_model = mock.MagicMock(), mock.MagicMock()
        coremltools.convert.side_effect = [full_model, backbone_model]
        exported_program = mock.MagicMock()
        exported_program.run_decompositions.return_value = exported_program
        with (
            mock.patch.dict("sys.modules", {"coremltools": coremltools}),
            mock.patch("rfdetr.export._coreml.exporter._IS_COREMLTOOLS_AVAILABLE", True),
            mock.patch("torch.export.export", return_value=exported_program),
            mock.patch("rfdetr.export._coreml.exporter.unsupported_coreml_ops", return_value={}),
        ):
            config = CoreMLConfig(
                output_dir=tmp_path, variant_name=variant_name, output_name=output_name, verbose=False
            )
            full_out = CoreMLExporter(config)(_make_export_graph(torch.nn.Identity()))
            backbone_out = CoreMLExporter(config)(_make_export_graph(torch.nn.Identity(), backbone_only=True))
        assert full_out != backbone_out
        assert full_out.name == full_name
        assert backbone_out.name == backbone_name
        full_model.save.assert_called_once_with(str(full_out))
        backbone_model.save.assert_called_once_with(str(backbone_out))


class TestVariantNamePathSafety:
    """Regression coverage for the path-traversal mitigation ``CoreMLExporter`` applies to ``variant_name`` (via
    ``resolve_export_stem``: ``os.path.splitext(os.path.basename(variant_name))[0]``).

    Exercises the sanitization expression directly rather than through a full export: ``CoreMLExporter`` imports the
    real ``coremltools`` package immediately after the (mockable) availability check, so a full call still requires
    coremltools installed — this test covers the contract without that dependency.
    """

    @pytest.mark.parametrize(
        ("variant_name", "expected"),
        [
            pytest.param("../../etc/passwd", "passwd", id="forward-slash-traversal"),
            pytest.param("/absolute/path/rfdetr-nano", "rfdetr-nano", id="absolute-path"),
            pytest.param("rfdetr-nano.mlpackage", "rfdetr-nano", id="strips-extension"),
            pytest.param("rfdetr-nano", "rfdetr-nano", id="plain-name-unchanged"),
        ],
    )
    def test_sanitizes_directory_components(self, variant_name: str, expected: str) -> None:
        """``variant_name`` must be reduced to a bare filename stem, no directory components.

        Forward-slash paths only: ``os.path.basename`` splits on ``os.sep`` (platform-native), so a
        backslash-separated path is *not* sanitized on POSIX (only on Windows, where ``ntpath``
        treats both ``/`` and ``\\`` as separators) — that asymmetry is a real, pre-existing property
        of this mitigation, out of scope for this regression test to change.
        """
        import os

        sanitized = os.path.splitext(os.path.basename(variant_name))[0]
        assert sanitized == expected
        assert "/" not in sanitized
        assert ".." not in sanitized


class TestExportFormatParameter:
    """Tests for ``format="coreml"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock heavy export deps so ``RFDETR.export()`` stays fast."""
        self._tmp_path = tmp_path
        mlpackage = tmp_path / "inference_model.mlpackage"
        mlpackage.mkdir()

        self._mock_stack = contextlib.ExitStack()
        self._mock_export_onnx = self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export._onnx.exporter.OnnxExporter._convert",
                return_value=str(tmp_path / "inference_model.onnx"),
            )
        )
        self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export.prepare.make_infer_image",
                return_value=torch.zeros(1, 3, 560, 560),
            )
        )
        self._mock_coreml_convert = self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export._coreml.exporter.CoreMLExporter._convert",
                return_value=mlpackage,
            )
        )
        yield
        self._mock_stack.close()

    @staticmethod
    def _make_rfdetr(*, segmentation_head: bool = False, use_grouppose_keypoints: bool = False) -> Any:
        """Create a minimal RFDETR instance with mocked internals.

        Args:
            segmentation_head: Whether the mocked config reports a seg head.
            use_grouppose_keypoints: Whether the mocked config reports a keypoint head.

        Returns:
            An ``RFDETR`` built without ``__init__`` whose model and config are ``MagicMock`` stand-ins.

        Examples:
            >>> obj = TestExportFormatParameter._make_rfdetr(use_grouppose_keypoints=True)
            >>> obj.model_config.use_grouppose_keypoints, obj.model_config.segmentation_head
            (True, False)
        """
        from rfdetr.detr import RFDETR

        obj = RFDETR.__new__(RFDETR)
        obj.model = mock.MagicMock()
        obj.model.resolution = 560
        obj.model.device = "cpu"
        obj.model.model.to.return_value = obj.model.model
        obj.model_config = mock.MagicMock()
        obj.model_config.segmentation_head = segmentation_head
        obj.model_config.use_grouppose_keypoints = use_grouppose_keypoints
        obj.model_config.patch_size = 14
        obj.model_config.num_windows = 1
        obj.model_config.num_channels = 3
        return obj

    def test_coreml_format_dispatches_to_coreml_exporter_not_onnx(self) -> None:
        """``format="coreml"`` must dispatch to ``CoreMLExporter`` (not the ONNX one) and warn (experimental).

        Dispatch keys off ``format``, not off the head, so the default (detection) head stands for every task the format
        supports — a per-head parametrization would assert the identical thing three times. The head-specific part — the
        ``dets``/``labels``/``keypoints`` output contract — is asserted format-agnostically in
        ``tests/export/test_prepare.py``, not here: ``CoreMLExporter._convert`` is mocked out in this class.
        """
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match="experimental"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"))
        self._mock_coreml_convert.assert_called_once()
        self._mock_export_onnx.assert_not_called()

    def test_onnx_format_does_not_call_coreml_exporter(self) -> None:
        """``format="onnx"`` must not import/call the CoreML converter."""
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_coreml_convert.assert_not_called()

    def test_notes_warns_and_is_ignored(self) -> None:
        """``notes`` must warn for CoreML (no ONNX-style metadata slot) but still export."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match=r"`notes` is not forwarded to format='coreml'"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"), notes="hello")
        self._mock_coreml_convert.assert_called_once()

    def test_dynamic_batch_raises_before_converter(self) -> None:
        """``dynamic_batch=True`` is refused by ``RFDETR.export()`` before the converter is invoked."""
        obj = self._make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(format="coreml", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)
        self._mock_coreml_convert.assert_not_called()

    def test_invalid_format_raises_value_error(self) -> None:
        """Unknown ``format`` must raise ``ValueError`` listing supported formats.

        Facade-level: proves ``RFDETR.export()`` validates before any format dispatch, so one instance covers
        every format (this used to be copy-pasted per exporter test file; ``TestResolveExporter`` in
        ``test_registry.py`` separately covers the underlying guard, parametrized per registered format).
        """
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="Unsupported export format"):
            obj.export(format="bogus", output_dir=str(self._tmp_path / "out"))


# ---------------------------------------------------------------------------
# End-to-end (gated) — real convert + FLOAT32 CPU parity vs eager PyTorch
# ---------------------------------------------------------------------------


# (model class name, expected output labels) — all share the same fixture setup shape (export once,
# reuse for the mlpackage-written / structured-parity / supervision-image checks below), so the
# fixture and its consuming tests are parametrized over this pair instead of duplicated per variant.
_COREML_E2E_VARIANTS = [
    pytest.param(("RFDETRNano", ("boxes", "logits")), id="detection"),
    pytest.param(("RFDETRSegNano", ("boxes", "logits", "masks")), id="segmentation"),
    # Keypoints are preview-only and therefore XLarge at resolution 576 — several times the work of the
    # two Nano variants, which is why this variant is the slow one in the `e2e_coreml` job.
    pytest.param(("RFDETRKeypointPreview", ("boxes", "logits", "keypoints")), id="keypoint"),
]

# Raw-tensor parity on a two-stage detector is only well defined when the encoder's `torch.topk` ranking has
# no near-ties. Untrained weights put the ranking scores of neighbouring encoder tokens ~1e-6 apart across the
# default 300 selected queries, which is the size of legitimate fp32 rounding differences between eager and the
# CoreML CPU runtime. A near-tied pair then swaps places, two reference points trade queries while `query_feat`
# stays positional, and self-attention spreads that into every logit (0.5-1.5 abs diff). Which pairs flip moves
# with the seed, the input, the torch version, the compute unit and even runtime kernel fusion, which is what made
# this suite look flaky (and what the former `torch<2.12` pin on the `coreml` extra was wrongly attributed to).
# The top of the ranking is sparse, though. Measured on Apple M3 Pro with coremltools 9.0 over Nano, SegNano and
# KeypointPreview, seeds 0/3/7, both parity inputs and torch 2.11/2.12/2.14 (54 cases): CoreML moves a ranking
# score by at most 1e-5, while the smallest gap among the top 6 scores is 3.1e-4. Exporting 5 queries therefore
# keeps every output under the tight raw bound, and `_assert_well_conditioned` re-checks that precondition on
# every run instead of trusting it (weight init differs across torch versions, so the margins do too).
#: Queries the e2e parity exports select, small enough that their two-stage ranking is well separated.
_COREML_E2E_NUM_QUERIES = 5
#: Minimum eager gap between neighbouring top-ranked scores: 5x the worst swap (two scores drifting 1e-5 apart).
_MIN_TWO_STAGE_RANK_MARGIN = 1e-4
#: Seed the module-scoped e2e fixtures set themselves: they run before the autouse per-test ``reset_random_seeds``.
_COREML_EXPORT_SEED = 0


def _two_stage_rank_margin(model: torch.nn.Module, example_input: torch.Tensor) -> float:
    """Return the smallest gap between neighbouring scores that decide the model's two-stage ``torch.topk``.

    Only the top ``k + 1`` scores matter: a swap among them reorders the selected queries or changes which
    ones are selected, and a swap below them changes nothing.

    Args:
        model: Export-mode module whose forward makes exactly one ``torch.topk`` call over its last dimension.
        example_input: ``(N, C, H, W)`` input; every image in the batch is measured.

    Returns:
        The smallest gap between neighbouring scores among the top ``k + 1`` of any image.

    Raises:
        AssertionError: If the forward does not call ``torch.topk`` exactly once, or if the call does not rank
            along the scores tensor's last axis (the margin below is only meaningful there).

    Examples:
        >>> _two_stage_rank_margin(_TopkRanker(2), torch.tensor([[1.0, 0.5, 0.25, 0.0]]))
        0.25
    """
    with mock.patch("torch.topk", wraps=torch.topk) as topk, torch.no_grad():
        model(example_input.clone())
    assert topk.call_count == 1, f"expected one two-stage torch.topk call, got {topk.call_count}"
    assert len(topk.call_args.args) >= 2, (
        "expected the two-stage torch.topk call to pass scores and k positionally, got "
        f"args={topk.call_args.args!r} kwargs={topk.call_args.kwargs!r}"
    )
    scores, k = topk.call_args.args[:2]
    # `dim` may arrive as a third positional arg or (as the production call site does, a keyword on a
    # 2-D tensor) as a kwarg; either way, the margin below is only meaningful when it targets the last axis.
    dim = topk.call_args.kwargs.get("dim", topk.call_args.args[2] if len(topk.call_args.args) > 2 else -1)
    assert dim % scores.ndim == scores.ndim - 1, (
        "expected the two-stage torch.topk call to rank along the scores tensor's last axis, got "
        f"dim={dim!r} for a {scores.ndim}-D scores tensor"
    )
    top = scores.sort(dim=-1, descending=True).values[..., : k + 1]
    return float((top[..., :-1] - top[..., 1:]).min())


def _assert_well_conditioned(model: torch.nn.Module, example_input: torch.Tensor) -> None:
    """Fail with an explicit precondition message when the input's two-stage ranking has a near-tie.

    Args:
        model: Export-mode module whose forward makes exactly one ``torch.topk`` call.
        example_input: ``(N, C, H, W)`` parity input.

    Raises:
        AssertionError: If the ranking margin is below ``_MIN_TWO_STAGE_RANK_MARGIN``.

    Examples:
        >>> _assert_well_conditioned(_TopkRanker(1), torch.tensor([[1.0, 0.0]]))
        >>> _assert_well_conditioned(_TopkRanker(1), torch.tensor([[0.5, 0.5]]))
        Traceback (most recent call last):
            ...
        AssertionError: parity input is ill-conditioned: ...
    """
    margin = _two_stage_rank_margin(model, example_input)
    assert margin >= _MIN_TWO_STAGE_RANK_MARGIN, (
        f"parity input is ill-conditioned: two-stage topk scores are only {margin:.2e} apart "
        f"(< {_MIN_TWO_STAGE_RANK_MARGIN}), so fp32 rounding can swap selected queries and raw outputs cannot "
        "match; lower _COREML_E2E_NUM_QUERIES or change the parity input rather than loosening the parity bound"
    )


@pytest.fixture(scope="module")
def people_walking_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download supervision's ``PEOPLE_WALKING`` asset once, shared across CoreML e2e tests."""
    asset_dir = tmp_path_factory.mktemp("coreml_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        return Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module", params=_COREML_E2E_VARIANTS)
def coreml_export(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> tuple[str, Any, torch.Tensor, Path, tuple[str, ...]]:
    """Export RFDETRNano/RFDETRSegNano/RFDETRKeypointPreview to a ``.mlpackage`` once per variant for e2e tests.

    Exports ``_COREML_E2E_NUM_QUERIES`` queries so the two-stage ranking is well separated (see the module-level
    comment), and reseeds itself because it runs before the autouse per-test seed reset. The variant's class name
    comes back with it so ``coreml_default_queries_export`` can depend on this fixture and inherit its
    parametrization, rather than declaring the same ``params`` again and producing a cross product.

    Examples:
        Skipped: a pytest fixture, and a real ``coremltools`` conversion, so it cannot run standalone.

        >>> model_cls_name, model, example, mlpackage_path, output_labels = coreml_export  # doctest: +SKIP
        >>> model_cls_name, example.shape[0], mlpackage_path.suffix, output_labels  # doctest: +SKIP
        ('RFDETRNano', 1, '.mlpackage', ('boxes', 'logits'))
    """
    model_cls_name, output_labels = request.param
    model_cls = getattr(rfdetr, model_cls_name)
    out_dir = tmp_path_factory.mktemp(f"coreml_{model_cls_name.lower()}")
    seed_all(_COREML_EXPORT_SEED)
    detector = model_cls(pretrain_weights=None, num_queries=_COREML_E2E_NUM_QUERIES)
    mlpackage_path = detector.export(output_dir=str(out_dir), format="coreml", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model_cls_name, model, example, Path(mlpackage_path), output_labels


@pytest.fixture(scope="module")
def coreml_backbone_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
    """Export RFDETRNano's backbone and return its eager feature-map reference module.

    Uses the public ``backbone_only=True`` route so the CoreML runtime executes the same list-valued ``_BackboneExport``
    graph that users receive, rather than a mocked converter dispatch.
    """
    out_dir = tmp_path_factory.mktemp("coreml_backbone")
    seed_all(_COREML_EXPORT_SEED)
    detector = rfdetr.RFDETRNano(pretrain_weights=None)
    mlpackage_path = detector.export(output_dir=str(out_dir), format="coreml", backbone_only=True, verbose=False)
    backbone = detector.model.model.backbone[0].to("cpu").eval()
    reference_model = _BackboneExport(backbone)
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return reference_model, example, Path(mlpackage_path)


@pytest.fixture(scope="module")
def coreml_default_queries_export(
    coreml_export: tuple[str, Any, torch.Tensor, Path, tuple[str, ...]],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[torch.nn.Module, torch.Tensor, Path, Path, tuple[str, ...]]:
    """Export each e2e variant with its shipped query count, which ``coreml_export`` trades for a separated ranking.

    Those counts differ per variant (300 for detection, 100 for segmentation and keypoints), and the mask and keypoint
    output shapes follow them, so every shipped graph is converted and run rather than only the detection one. Depends
    on ``coreml_export`` for the variant *and* for its 5-query ``.mlpackage``, which the graph comparison below is
    measured against.

    Examples:
        Skipped: a pytest fixture, and a real ``coremltools`` conversion, so it cannot run standalone.

        >>> model, example, mlpackage_path, few_queries_path, output_labels = (
        ...     coreml_default_queries_export
        ... )  # doctest: +SKIP
        >>> mlpackage_path != few_queries_path, example.shape[0], output_labels  # doctest: +SKIP
        (True, 1, ('boxes', 'logits'))
    """
    model_cls_name, _, _, few_queries_path, output_labels = coreml_export
    out_dir = tmp_path_factory.mktemp(f"coreml_default_queries_{model_cls_name.lower()}")
    seed_all(_COREML_EXPORT_SEED)
    detector = getattr(rfdetr, model_cls_name)(pretrain_weights=None)
    mlpackage_path = detector.export(output_dir=str(out_dir), format="coreml", verbose=False)
    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return model, example, Path(mlpackage_path), few_queries_path, output_labels


@coreml_only
@pytest.mark.integration
@pytest.mark.e2e_coreml
class TestCoreMLEndToEnd:
    """Real CoreML export + FLOAT32 CPU numerical parity (``-m e2e_coreml``)."""

    def test_mlpackage_written(self, coreml_export: tuple[str, Any, torch.Tensor, Path, tuple[str, ...]]) -> None:
        """Export must write a non-empty ``.mlpackage`` directory/bundle, named with the resolved precision."""
        _, _, _, mlpackage_path, _ = coreml_export
        assert mlpackage_path.exists()
        # Default compute_precision resolves to FLOAT32 (see the exporter module docstring); the filename must
        # always encode it, since precision materially changes the artifact.
        assert mlpackage_path.stem.endswith("_fp32")
        assert mlpackage_path.suffix == ".mlpackage" or mlpackage_path.name.endswith(".mlpackage")

    def test_outputs_match_pytorch_structured(
        self, coreml_export: tuple[str, Any, torch.Tensor, Path, tuple[str, ...]]
    ) -> None:
        """CoreML output matches eager on structured (gradient+checkerboard) input."""
        _, model, example, mlpackage_path, output_labels = coreml_export
        _assert_well_conditioned(model, example)
        _validate_coreml_vs_pytorch(mlpackage_path, model, example, output_labels=output_labels)

    def test_outputs_match_pytorch_supervision_image(
        self,
        coreml_export: tuple[str, Any, torch.Tensor, Path, tuple[str, ...]],
        people_walking_image_path: Path,
    ) -> None:
        """CoreML output matches eager on ``ImageAssets.PEOPLE_WALKING``."""
        _, model, structured, mlpackage_path, output_labels = coreml_export
        example = _parity_input_from_image(people_walking_image_path, int(structured.shape[-1]))
        _assert_well_conditioned(model, example)
        _validate_coreml_vs_pytorch(mlpackage_path, model, example, output_labels=output_labels)

    def test_backbone_outputs_match_pytorch_structured(
        self, coreml_backbone_export: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """CoreML must run every backbone feature-map output from the public backbone-only export."""
        model, example, mlpackage_path = coreml_backbone_export
        assert "-backbone" in mlpackage_path.stem
        diffs = _coreml_parity_diffs(mlpackage_path, model, example)
        assert max(diffs) < _COREML_MAX_ABS_DIFF, (
            f"CoreML backbone outputs diverge from PyTorch: max abs diff {max(diffs)} (bound={_COREML_MAX_ABS_DIFF})"
        )

    def test_default_query_count_runs_with_eager_shapes(
        self, coreml_default_queries_export: tuple[torch.nn.Module, torch.Tensor, Path, Path, tuple[str, ...]]
    ) -> None:
        """The shipped-query-count export must run on CoreML with eager's output count, shapes and finite values.

        Values are deliberately not bounded here. At the shipped query count the untrained two-stage ranking always has
        near-ties, so a legitimate fp32 swap can move every output, and no value comparison is both tight and stable
        (post-processed scores drift up to ~1e-3 on a swap). The strict 1e-4 value parity is carried by the 5-query
        export; this test covers what the shipped graph adds on top of it: the shipped output shapes.
        """
        model, example, mlpackage_path, _, output_labels = coreml_default_queries_export
        diffs = _coreml_parity_diffs(mlpackage_path, model, example)
        assert len(diffs) == len(output_labels), f"CoreML export must yield {output_labels}, got {len(diffs)} outputs"
        assert all(np.isfinite(diffs)), f"CoreML produced non-finite outputs: max abs diffs {diffs}"

    def test_default_query_count_lowers_through_no_unchecked_operation(
        self, coreml_default_queries_export: tuple[torch.nn.Module, torch.Tensor, Path, Path, tuple[str, ...]]
    ) -> None:
        """The shipped-query graph must reach no MIL operation, nor far more of one, than the checked graph does.

        Deliberately a *structural* guard, not a numerical one, and it does not make the 5-query parity stand in for
        the shipped graph: the historical CoreML parity failures this suite was rewritten around happened with an
        identical op set on both sides, so bare op-type membership cannot discriminate drift on its own. Op *counts*
        (within ``_MIL_OP_COUNT_TOLERANCE``, since constant folding legitimately trims a handful of `const`/`reshape`
        nodes between shapes) additionally catch the shipped shapes reusing a familiar op type far more heavily —
        say, a live ``slice_by_index`` chain at 300 queries where 5 queries folded most of it away to ``const``.

        One-directional on purpose: an op type the shipped graph reaches far more of (including one the checked
        graph never reaches at all) is the gap; the reverse (a node fused away only at the larger shape) is benign
        and must not turn the macOS-only parity job red.
        """
        import coremltools as ct

        _, _, mlpackage_path, few_queries_path, _ = coreml_default_queries_export

        shipped_counts = _mil_op_counts(ct.utils.load_spec(str(mlpackage_path)))
        few_queries_counts = _mil_op_counts(ct.utils.load_spec(str(few_queries_path)))

        # Protobuf message maps default-construct on lookup, so a spec this traversal does not understand yields
        # an empty count on both sides and the regression check below would pass while inspecting nothing.
        assert shipped_counts, f"no MIL operations found in {mlpackage_path.name}: the spec traversal is wrong"
        regressions = _mil_op_count_regressions(shipped_counts, few_queries_counts, tolerance=_MIL_OP_COUNT_TOLERANCE)
        assert not regressions, (
            f"shipped-query graph reaches these MIL ops far more than the {_COREML_E2E_NUM_QUERIES}-query parity "
            f"graph does, beyond the folding tolerance of {_MIL_OP_COUNT_TOLERANCE} -- {{op: (shipped, checked)}}: "
            f"{regressions}"
        )


class TestCoreMLParityInputHelpers:
    """Unit checks for CoreML-local parity input builders (no coremltools required)."""

    def test_structured_parity_input_shape_and_determinism(self) -> None:
        """Structured tensor must be ``NCHW``, finite, and seed-independent-deterministic."""
        a = _structured_parity_input(1, 3, 64, 64)
        b = _structured_parity_input(1, 3, 64, 64)
        assert a.shape == (1, 3, 64, 64)
        assert torch.isfinite(a).all()
        assert torch.equal(a, b)
        # Spatially varying — not a constant fill.
        assert float(a.std()) > 1e-3

    def test_parity_input_from_image_loads_rgb(self, tmp_path: Path) -> None:
        """``_parity_input_from_image`` must normalize a local RGB file to ``1x3xHxW``."""
        image_path = tmp_path / "tiny.png"
        Image.new("RGB", (32, 24), color=(20, 40, 60)).save(image_path)
        tensor = _parity_input_from_image(image_path, 64)
        assert tensor.shape == (1, 3, 64, 64)
        assert torch.isfinite(tensor).all()


class _TopkRanker(torch.nn.Module):
    """Stand-in for a two-stage ranker: one ``torch.topk`` over the last dimension per configured ``k``.

    Every ``k`` re-ranks the same input rather than chaining, so the call *count* is what varies — that is what
    ``_two_stage_rank_margin``'s guard reads, and chaining would shrink the tensor out from under later ``k``s.

    Examples:
        >>> _TopkRanker(1)(torch.tensor([[0.25, 0.75]])).tolist()
        [[0.75]]
        >>> _TopkRanker()(torch.tensor([[0.25]])).tolist()
        [[0.25]]
    """

    def __init__(self, *ks: int) -> None:
        super().__init__()
        self.ks = ks

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rank ``x`` once per configured ``k`` and return the last ranking (``x`` itself when none)."""
        out = x
        for k in self.ks:
            out = torch.topk(x, k, dim=-1).values
        return out


class _RaisingRanker(torch.nn.Module):
    """Stand-in ranker whose forward calls ``torch.topk`` once, then always raises.

    Covers the restore-on-exception path of the ``mock.patch("torch.topk", ...)`` spy in
    :func:`_two_stage_rank_margin`: a model that blows up mid-forward, after its one ranking call, must not
    leave ``torch.topk`` patched for every test that runs after it.

    Examples:
        >>> _RaisingRanker()(torch.tensor([[0.9, 0.1]]))
        Traceback (most recent call last):
            ...
        RuntimeError: boom
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Call ``torch.topk`` once, then unconditionally raise ``RuntimeError("boom")``."""
        torch.topk(x, 1, dim=-1)
        raise RuntimeError("boom")


class TestTwoStageRankMargin:
    """``_two_stage_rank_margin`` must measure exactly the gaps that decide the two-stage selection."""

    @pytest.mark.parametrize(
        ("scores", "k", "expected"),
        [
            pytest.param([[0.9, 0.5, 0.49, 0.1]], 2, 0.01, id="gap-at-selection-boundary-counts"),
            pytest.param([[0.9, 0.5, 0.1, 0.1]], 1, 0.4, id="tie-below-boundary-ignored"),
            pytest.param([[0.1, 0.9, 0.5]], 1, 0.4, id="input-order-irrelevant"),
            pytest.param([[0.7, 0.7, 0.1]], 1, 0.0, id="exact-tie-inside-selection"),
            pytest.param([[0.9, 0.1], [0.5, 0.5]], 1, 0.0, id="tie-in-second-image"),
            pytest.param([[0.9, 0.9, 0.9, 0.1]], 2, 0.0, id="three-way-tie-at-selection-boundary"),
            pytest.param(
                [[0.9, 0.5, 0.1], [0.9, 0.85, 0.1], [0.9, 0.6, 0.1]],
                1,
                0.05,
                id="min-gap-in-middle-image-of-three",
            ),
        ],
    )
    def test_margin(self, scores: list[list[float]], k: int, expected: float) -> None:
        """The margin is the smallest neighbouring gap among the top ``k + 1`` scores of any image."""
        assert _two_stage_rank_margin(_TopkRanker(k), torch.tensor(scores)) == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize("ks", [pytest.param((), id="no-topk"), pytest.param((1, 1), id="two-topk")])
    def test_rejects_forward_without_exactly_one_topk(self, ks: tuple[int, ...]) -> None:
        """Measuring the wrong ranking would silently vouch for parity, so any other call count must fail."""
        with pytest.raises(AssertionError, match="expected one two-stage torch.topk call"):
            _two_stage_rank_margin(_TopkRanker(*ks), torch.tensor([[0.9, 0.1]]))

    def test_raises_on_k_equal_zero(self) -> None:
        """``k=0`` leaves no neighbouring pair to diff, so torch's own empty-tensor reduction error surfaces raw.

        Documents actual behaviour rather than a documented contract: ``_two_stage_rank_margin`` has no ``k=0``
        guard, so a caller hits ``Tensor.min()`` on an empty tensor instead of an actionable assertion.
        """
        with pytest.raises(RuntimeError, match=r"numel\(\) == 0"):
            _two_stage_rank_margin(_TopkRanker(0), torch.tensor([[0.9, 0.1]]))

    def test_restores_torch_topk(self) -> None:
        """The spy must not leak: ``torch.topk`` is the original function after a measurement."""
        original = torch.topk
        _two_stage_rank_margin(_TopkRanker(1), torch.tensor([[0.9, 0.1]]))
        assert torch.topk is original

    def test_restores_torch_topk_after_forward_raises(self) -> None:
        """The spy must not leak even when the model's forward raises after its one ``torch.topk`` call."""
        original = torch.topk
        with pytest.raises(RuntimeError, match="boom"):
            _two_stage_rank_margin(_RaisingRanker(), torch.tensor([[0.9, 0.1]]))
        assert torch.topk is original


class _ProtoMapDouble(dict):
    """Minimal double for a protobuf message-map: reads default-construct a missing key instead of raising.

    Protobuf map fields (e.g. ``ModelSpecification.mlProgram.functions``, ``Function.block_specializations``)
    return a fresh default-constructed message on ``some_map[missing_key]`` rather than raising ``KeyError`` --
    exactly the semantics :func:`_mil_op_types`'s docstring documents, and a plain ``dict`` (raises ``KeyError``)
    or ``SimpleNamespace`` (no ``__getitem__`` at all) cannot reproduce.

    Args:
        default_factory: Builds (and caches, matching protobuf's own auto-vivify-on-read behaviour) the value
            returned for a key not already present.

    Examples:
        >>> m = _ProtoMapDouble(lambda: "default")
        >>> m["missing"]
        'default'
        >>> m["missing"] is m["missing"]
        True
    """

    def __init__(self, default_factory: Any) -> None:
        super().__init__()
        self._default_factory = default_factory

    def __missing__(self, key: str) -> Any:
        """Default-construct, cache, and return the value for *key* instead of raising ``KeyError``."""
        value = self._default_factory()
        self[key] = value
        return value


class TestMilOpTypesMalformedSpec:
    """``_mil_op_types`` must not raise on a spec whose ``"main"`` function is entirely absent."""

    def test_returns_empty_set_on_default_constructed_main(self) -> None:
        """A spec with no ``"main"`` entry in ``functions`` must traverse to the empty set, never raise.

        Mirrors real protobuf behaviour end to end: ``functions["main"]`` default-constructs an empty
        ``Function`` (``opset=""``), and that function's ``block_specializations[""]`` in turn default-constructs
        an empty operation list -- no key anywhere actually exists, yet nothing raises.
        """
        block_specializations = _ProtoMapDouble(lambda: SimpleNamespace(operations=[]))
        functions = _ProtoMapDouble(lambda: SimpleNamespace(opset="", block_specializations=block_specializations))
        spec = SimpleNamespace(mlProgram=SimpleNamespace(functions=functions))

        assert _mil_op_types(spec) == set()


class TestE2EParityPrecondition:
    """The e2e query count must leave every exported variant's two-stage ranking well separated.

    Needs no ``coremltools``, so a model or initialisation change that breaks the precondition is caught by the regular
    CPU suite instead of first surfacing as a CoreML parity failure on the macOS-only ``e2e_coreml`` job.
    """

    @pytest.mark.parametrize("variant", _COREML_E2E_VARIANTS)
    def test_structured_input_is_well_conditioned(self, variant: tuple[str, tuple[str, ...]]) -> None:
        """Seeded exactly like ``coreml_export``, the structured parity input must pass the margin check."""
        model_cls_name, _ = variant
        seed_all(_COREML_EXPORT_SEED)
        detector = getattr(rfdetr, model_cls_name)(pretrain_weights=None, num_queries=_COREML_E2E_NUM_QUERIES)
        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        _assert_well_conditioned(model, _structured_parity_input(1, 3, resolution, resolution))
