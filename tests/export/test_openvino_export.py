# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for direct PyTorch -> OpenVINO IR (``.xml``/``.bin``) export.

Covers:
* ``export_openvino()`` — dependency-missing path, path-traversal sanitization, and the internal
  ``ModelWrapper`` dict-output mapping (``openvino.convert_model``/``save_model`` stubbed via
  ``sys.modules`` injection so these run without the real ``openvino`` package installed).
* ``OpenVINOInference.__init__`` — dependency-missing path.
* ``format="openvino"`` wiring through ``RFDETR.export()`` (heavy deps mocked, fast).
* A real end-to-end export + numerical parity check, gated behind ``pytest.importorskip("openvino")``
  so it only runs where the ``openvino`` package is installed.

This repository's CI/dev environment does not install ``openvino`` — the dependency-missing tests below
exercise the real (uninstalled) code path directly rather than mocking an ``ImportError``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

from rfdetr.export._openvino.exporter import export_openvino
from rfdetr.export._openvino.inference import OpenVINOInference
from tests.export.conftest import (
    _parity_input_from_image,
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)


def _infer_openvino_f32(xml_path: Path, input_array: NDArray[Any]) -> tuple[NDArray[Any], ...]:
    """Run *xml_path* through OpenVINO with execution precision pinned to float32.

    OpenVINO's ARM CPU plugin defaults to fp16 *execution* regardless of the IR's storage
    precision (``compress_to_fp16``) -- confirmed by measurement: pinning this hint dropped a
    backbone parity diff from 0.11 to 0.0059 on this repo's CI-equivalent macOS-ARM setup. The
    public :class:`~rfdetr.export._openvino.inference.OpenVINOInference` wrapper does not expose
    ``ov.Core``'s property passthrough, so parity assertions compile directly via raw ``ov.Core``
    here instead of growing the production wrapper's API for a test-only need.

    Args:
        xml_path: Path to an exported OpenVINO IR ``.xml`` file.
        input_array: C-contiguous float32 input, shaped to match the model's input layer.

    Returns:
        One NumPy array per model output, in declaration order.

    Examples:
        Requires a real exported ``.xml``/``.bin`` pair and ``openvino`` — not runnable standalone.
        See ``TestOpenVINOEndToEnd`` for real invocations.

        >>> callable(_infer_openvino_f32)
        True
    """
    import openvino as ov

    core = ov.Core()
    compiled = core.compile_model(core.read_model(xml_path), "CPU", {"INFERENCE_PRECISION_HINT": "f32"})
    request = compiled.create_infer_request()
    request.infer({compiled.input(0): input_array})
    return tuple(np.copy(request.get_output_tensor(i).data) for i in range(len(compiled.outputs)))


def _confident_query_diffs(
    eager_tensors: list[torch.Tensor],
    other_tensors: list[torch.Tensor],
    top_k: int = 10,
    sigmoid_indices: frozenset[int] = frozenset(),
) -> list[float]:
    """Max-abs-diff per output, restricted to the ``top_k`` highest-confidence queries.

    RF-DETR's two-stage query selection ranks ~300 raw encoder proposals by objectness and keeps
    all of them; on a real photo only a handful score as genuine detections (well-separated), and
    the rest are low-confidence background candidates whose objectness scores sit close enough
    together that ordinary cross-backend floating-point differences (different kernel
    implementations -- no backend gives a bitwise guarantee) flip their relative rank and swap
    which query slot each ends up in. That reordering is real and expected, not evidence the
    export is wrong: comparing all ~300 raw positions swamps the signal that actually matters (do
    the genuine detections match?) with unrelated background-candidate noise. Restricting to the
    confident queries -- which stay positionally aligned in measurement -- is what a parity check
    is actually for.

    Args:
        eager_tensors: Reference tensors from :func:`eager_reference_tensors`; ``eager_tensors[1]``
            must be the per-query class-logit tensor (``dets, labels[, ...]`` output order).
        other_tensors: Backend output tensors, same order and shapes as *eager_tensors*.
        top_k: Number of highest-confidence queries (by eager logit max) to compare.
        sigmoid_indices: Output indices to compare in sigmoid (probability) space instead of raw
            logit space -- e.g. segmentation mask logits span a much wider range than boxes/labels,
            so an equivalent raw-space tolerance would be too loose to catch real regressions.

    Returns:
        One max-abs-diff per output, computed over the ``top_k`` selected queries only.

    Examples:
        >>> boxes = torch.zeros(1, 3, 4)
        >>> labels = torch.tensor([[[0.1, 0.2], [5.0, 0.1], [0.0, 0.0]]])
        >>> other_boxes = boxes.clone()
        >>> other_labels = labels.clone()
        >>> other_boxes[0, 1] += 0.5  # perturb the only confident query (index 1)
        >>> diffs = _confident_query_diffs([boxes, labels], [other_boxes, other_labels], top_k=1)
        >>> round(diffs[0], 4)
        0.5
    """
    scores = eager_tensors[1][0].amax(dim=-1)
    top_indices = torch.topk(scores, top_k).indices
    diffs = []
    for index, (eager, other) in enumerate(zip(eager_tensors, other_tensors)):
        eager_selected = eager[0, top_indices]
        other_selected = other[0, top_indices].float()
        if index in sigmoid_indices:
            eager_selected, other_selected = eager_selected.sigmoid(), other_selected.sigmoid()
        diffs.append((eager_selected - other_selected).abs().max().item())
    return diffs


def _stub_openvino_module() -> types.ModuleType:
    """Build a minimal fake ``openvino`` module exposing ``convert_model``/``save_model``.

    Injected into ``sys.modules`` so ``export_openvino()``'s ``from openvino import convert_model,
    save_model`` succeeds without the real package installed, letting the naming/wrapping logic run
    end-to-end while the actual (heavy, unavailable) conversion is a no-op mock.

    Returns:
        A fresh fake module with ``convert_model`` and ``save_model`` as ``MagicMock`` attributes.

    Examples:
        >>> fake = _stub_openvino_module()
        >>> callable(fake.convert_model) and callable(fake.save_model)
        True
    """
    fake = types.ModuleType("openvino")
    fake.convert_model = mock.MagicMock(return_value=mock.MagicMock(name="ov_model"))
    fake.save_model = mock.MagicMock()
    return fake


# ---------------------------------------------------------------------------
# export_openvino() — dependency-missing path (real environment, no openvino installed)
# ---------------------------------------------------------------------------


class TestExportOpenvinoMissingDependency:
    """``export_openvino()``'s ``ImportError`` path, exercised for real (openvino not installed here)."""

    def test_raises_import_error(self, tmp_path: Path) -> None:
        """Missing ``openvino`` must surface an ``ImportError``, not any other exception type."""
        model = torch.nn.Identity()
        example = torch.zeros(1, 3, 32, 32)
        with pytest.raises(ImportError):
            export_openvino(model, example, str(tmp_path))

    def test_import_error_names_pip_install_hint(self, tmp_path: Path) -> None:
        """The raised ``ImportError`` must name the ``rfdetr[openvino]`` extra so users know how to fix it.

        ``_check_openvino_available()`` raises a new ``ImportError`` with the actionable install hint baked into the
        message itself (not merely logged separately), so a caller catching the exception — not just reading logs —
        still sees the fix.
        """
        model = torch.nn.Identity()
        example = torch.zeros(1, 3, 32, 32)
        with pytest.raises(ImportError, match="rfdetr\\[openvino\\]"):
            export_openvino(model, example, str(tmp_path))


class TestPublicInferenceFacade:
    """``rfdetr.export.inference`` — the public import path for session-tier inference wrappers.

    ``OpenVINOInference`` is re-exported there through a module ``__getattr__`` so that importing the
    facade never pulls in the optional ``openvino`` dependency. Both halves of that contract are
    pinned here: the re-export resolves to the same class object, and the import stays lazy.
    """

    def test_reexports_the_private_class(self) -> None:
        """The facade attribute must be the class defined in the private module, not a copy of it."""
        from rfdetr.export import inference as public_inference

        assert public_inference.OpenVINOInference is OpenVINOInference

    def test_unknown_attribute_raises_attribute_error(self) -> None:
        """A name outside ``__all__`` must raise ``AttributeError`` rather than import something unexpected."""
        from rfdetr.export import inference as public_inference

        with pytest.raises(AttributeError, match="NotARuntime"):
            _ = public_inference.NotARuntime

    def test_import_does_not_load_openvino(self) -> None:
        """Importing the facade must leave ``openvino`` out of ``sys.modules``.

        Runs in a fresh interpreter on purpose: other tests in this suite import ``openvino``, so an
        in-process check would pass for the wrong reason.
        """
        result = subprocess.run(
            [sys.executable, "-c", "import rfdetr.export.inference, sys; print('openvino' in sys.modules)"],
            check=True,
            text=True,
            capture_output=True,
        )
        assert result.stdout.strip() == "False"


class TestOpenVINOInferenceMissingDependency:
    """``OpenVINOInference.__init__``'s ``ImportError`` path.

    Uses ``monkeypatch`` on the shared ``_check_openvino_available`` choke point (see
    ``test_coreml_export.py::test_missing_coremltools_raises_import_error`` for the sibling
    pattern) instead of relying on ``openvino`` actually being absent from the environment —
    the previous version of this test passed only by environment happenstance and would start
    failing the moment ``openvino`` gets installed anywhere this suite runs.
    """

    def test_raises_import_error_before_file_check(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing ``openvino`` install raises ``ImportError`` even for a nonexistent model path.

        ``OpenVINOInference.__init__`` checks availability before checking ``model_path.exists()``, so a nonexistent
        path must still surface ``ImportError`` here (never ``FileNotFoundError`` — that branch is unreachable without
        openvino installed; see ``TestOpenVINOInferenceEndToEnd`` for the gated ``FileNotFoundError`` coverage).
        """
        monkeypatch.setattr(
            "rfdetr.export._openvino.inference._check_openvino_available",
            mock.Mock(side_effect=ImportError('pip install "rfdetr[openvino]"')),
        )
        with pytest.raises(ImportError):
            OpenVINOInference(tmp_path / "does-not-exist.xml")


class TestOpenVINOInferenceInputValidation:
    """``OpenVINOInference.infer()``'s dtype/contiguity boundary check (no real ``openvino`` needed).

    The check runs before ``self.infer_request`` is touched, so these tests construct an instance via ``__new__``
    (skipping ``__init__``'s real OpenVINO runtime setup) and call ``infer()`` directly.
    """

    @staticmethod
    def _make_inference() -> OpenVINOInference:
        """Build an ``OpenVINOInference`` with ``__init__`` skipped, for testing ``infer()`` in isolation.

        Examples:
            >>> inference = TestOpenVINOInferenceInputValidation._make_inference()
            >>> hasattr(inference, "infer")
            True
        """
        return OpenVINOInference.__new__(OpenVINOInference)

    def test_rejects_float64_input(self) -> None:
        """A ``float64`` array must raise ``ValueError`` instead of silently doubling the buffer size."""
        inference = self._make_inference()
        bad_input = np.zeros((1, 3, 32, 32), dtype=np.float64)
        with pytest.raises(ValueError, match="float32"):
            inference.infer(bad_input)

    def test_rejects_non_contiguous_input(self) -> None:
        """A non-contiguous ``float32`` view must raise ``ValueError`` instead of silently mis-decoding."""
        inference = self._make_inference()
        strided_input = np.zeros((1, 32, 32, 3), dtype=np.float32).transpose(0, 3, 1, 2)
        assert not strided_input.flags["C_CONTIGUOUS"]
        with pytest.raises(ValueError, match="contiguous"):
            inference.infer(strided_input)


# ---------------------------------------------------------------------------
# export_openvino() — naming, path safety, and ModelWrapper dict-output mapping
# (openvino.convert_model/save_model stubbed; no real openvino needed)
# ---------------------------------------------------------------------------


class TestExportOpenvinoNaming:
    """Output filename resolution, exercised with a stubbed ``openvino`` module."""

    @pytest.mark.parametrize(
        ("variant_name", "backbone_only", "expected_stem"),
        [
            pytest.param(None, False, "inference_model", id="bare-default-detector"),
            pytest.param(None, True, "backbone_model", id="bare-default-backbone"),
            pytest.param("rfdetr-nano", False, "rfdetr-nano", id="variant-detector"),
            pytest.param("rfdetr-nano", True, "rfdetr-nano-backbone", id="variant-backbone"),
        ],
    )
    def test_resolves_expected_stem(
        self, tmp_path: Path, variant_name: str | None, backbone_only: bool, expected_stem: str
    ) -> None:
        """Backbone and detector exports must resolve distinct, predictable ``.xml`` stems."""
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            output_xml = export_openvino(
                torch.nn.Identity(),
                torch.zeros(1, 3, 8, 8),
                str(tmp_path),
                backbone_only=backbone_only,
                variant_name=variant_name,
                verbose=False,
            )
        assert output_xml == str(tmp_path / f"{expected_stem}.xml")
        fake_ov.save_model.assert_called_once_with(mock.ANY, output_xml, compress_to_fp16=True)

    @pytest.mark.parametrize(
        ("variant_name", "expected"),
        [
            pytest.param("../../etc/passwd", "passwd", id="forward-slash-traversal"),
            pytest.param("/absolute/path/rfdetr-nano", "rfdetr-nano", id="absolute-path"),
            pytest.param("rfdetr-nano.xml", "rfdetr-nano", id="strips-extension"),
            pytest.param("rfdetr-nano", "rfdetr-nano", id="plain-name-unchanged"),
        ],
    )
    def test_sanitizes_variant_name_directory_components(
        self, tmp_path: Path, variant_name: str, expected: str
    ) -> None:
        """``variant_name`` must be reduced to a bare filename stem before building the output path.

        Regression coverage for the same ``os.path.splitext(os.path.basename(...))`` mitigation
        ``export_coreml`` applies (see ``tests/export/test_coreml_export.py::TestVariantNamePathSafety``);
        ``export_openvino`` guards its ``variant_name`` the identical way.
        """
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            output_xml = export_openvino(
                torch.nn.Identity(),
                torch.zeros(1, 3, 8, 8),
                str(tmp_path),
                variant_name=variant_name,
                verbose=False,
            )
        assert output_xml == str(tmp_path / f"{expected}.xml")
        assert ".." not in output_xml.removeprefix(str(tmp_path))


class TestExportOpenvinoPrecision:
    """``precision``'s ``compress_to_fp16`` forwarding, exercised with a stubbed ``openvino`` module.

    Previously only the ``precision=None`` default path was exercised (via ``test_resolves_expected_stem``'s
    ``compress_to_fp16=True`` assertion); explicit ``"float32"``/ ``"float16"`` and the invalid-value rejection had zero
    test coverage.
    """

    @pytest.mark.parametrize(
        ("precision", "expected_compress"),
        [
            pytest.param("float32", False, id="float32-disables-compression"),
            pytest.param("float16", True, id="float16-enables-compression"),
        ],
    )
    def test_precision_forwarded_as_compress_to_fp16(
        self, tmp_path: Path, precision: str, expected_compress: bool
    ) -> None:
        """``precision`` must map to ``save_model``'s ``compress_to_fp16`` flag, not silently default."""
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            export_openvino(
                torch.nn.Identity(), torch.zeros(1, 3, 8, 8), str(tmp_path), precision=precision, verbose=False
            )
        assert fake_ov.save_model.call_args.kwargs["compress_to_fp16"] is expected_compress

    def test_invalid_precision_raises_value_error(self, tmp_path: Path) -> None:
        """An unrecognized ``precision`` value must raise ``ValueError`` before any conversion is attempted."""
        fake_ov = _stub_openvino_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            with pytest.raises(ValueError, match="precision must be"):
                export_openvino(
                    torch.nn.Identity(), torch.zeros(1, 3, 8, 8), str(tmp_path), precision="int8", verbose=False
                )
        fake_ov.convert_model.assert_not_called()


def _stub_openvino_runtime_module() -> tuple[types.ModuleType, mock.MagicMock]:
    """Build a fake ``openvino`` module with a mocked ``Core`` for ``OpenVINOInference`` unit tests.

    Returns:
        A ``(module, core_instance)`` pair where ``module.Core()`` returns ``core_instance``, and
        ``core_instance.compile_model(...)`` returns a compiled-model mock with one output.

    Examples:
        >>> fake_module, fake_core = _stub_openvino_runtime_module()
        >>> fake_module.Core() is fake_core
        True
    """
    fake = types.ModuleType("openvino")
    core = mock.MagicMock(name="core")
    compiled_model = mock.MagicMock(name="compiled_model")
    compiled_model.outputs = [mock.MagicMock(name="output_0")]
    core.read_model.return_value = mock.MagicMock(name="ov_model")
    core.compile_model.return_value = compiled_model
    fake.Core = mock.MagicMock(return_value=core)
    return fake, core


class TestOpenVINOInferenceDeviceAndCache:
    """``device``/``cache_dir`` forwarding in ``OpenVINOInference.__init__``, previously untested.

    Both parameters were only ever exercised at their defaults (``device="AUTO"``, ``cache_dir=None``) inside the CI-
    gated end-to-end class; an explicit non-default value had zero coverage anywhere.
    """

    def test_device_forwarded_to_compile_model(self, tmp_path: Path) -> None:
        """A non-default ``device`` must reach ``core.compile_model(model, device)`` unchanged."""
        xml_path = tmp_path / "m.xml"
        xml_path.write_bytes(b"<xml/>")
        fake_ov, core = _stub_openvino_runtime_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            OpenVINOInference(xml_path, device="GPU")
        core.compile_model.assert_called_once_with(core.read_model.return_value, "GPU")

    def test_cache_dir_sets_property_before_compile(self, tmp_path: Path) -> None:
        """A non-``None`` ``cache_dir`` must set ``CACHE_DIR`` before ``compile_model`` is called.

        OpenVINO requires ``CACHE_DIR`` set before compilation to reuse compiled kernels across process starts; setting
        it after would silently skip caching for this compilation.
        """
        xml_path = tmp_path / "m.xml"
        xml_path.write_bytes(b"<xml/>")
        fake_ov, core = _stub_openvino_runtime_module()
        call_order: list[str] = []
        core.set_property.side_effect = lambda *_a, **_kw: call_order.append("set_property")
        core.compile_model.side_effect = lambda *_a, **_kw: (
            call_order.append("compile_model") or mock.MagicMock(outputs=[mock.MagicMock()])
        )
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            OpenVINOInference(xml_path, cache_dir=str(tmp_path / "cache"))
        core.set_property.assert_called_once_with({"CACHE_DIR": str(tmp_path / "cache")})
        assert call_order == ["set_property", "compile_model"]

    def test_cache_dir_none_skips_set_property(self, tmp_path: Path) -> None:
        """The default ``cache_dir=None`` must not call ``set_property`` at all."""
        xml_path = tmp_path / "m.xml"
        xml_path.write_bytes(b"<xml/>")
        fake_ov, core = _stub_openvino_runtime_module()
        with mock.patch.dict(sys.modules, {"openvino": fake_ov}):
            OpenVINOInference(xml_path)
        core.set_property.assert_not_called()


class TestModelWrapper:
    """``ModelWrapper`` (module-scope, importable in isolation) normalizes export-mode output to a tuple.

    The wrapped model is expected to already be in export mode (``forward_export``), which returns a tuple (full
    detector) or a plain list (:class:`rfdetr.export._backend._BackboneExport`) — never a dict. A dict output means the
    caller forgot the mode-switch, which is a caller bug the wrapper must surface loudly rather than silently reshape.
    """

    def test_tuple_output_passes_through_unchanged(self) -> None:
        """A tuple output (full-detector ``forward_export``) must pass through as the same tuple."""
        from rfdetr.export._openvino.exporter import ModelWrapper

        dets, labels = torch.full((1, 4), 1.0), torch.full((1, 2), 2.0)

        class _TupleOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return (dets, labels)

        wrapper = ModelWrapper(_TupleOutputModel())
        out_dets, out_labels = wrapper(torch.zeros(1, 3, 8, 8))
        assert torch.equal(out_dets, dets)
        assert torch.equal(out_labels, labels)

    def test_three_tuple_output_passes_through_unchanged(self) -> None:
        """A 3-tuple output (segmentation ``masks`` or keypoint ``keypoints`` as the 3rd element) passes through.

        ``forward_export`` returns a 3-tuple for segmentation (``dets, labels, masks``) and keypoint (``dets, labels,
        keypoints``) models, distinct from the 2-tuple plain-detection case already covered above; only the tuple length
        differed and had never been exercised for this wrapper.
        """
        from rfdetr.export._openvino.exporter import ModelWrapper

        dets, labels, third = torch.full((1, 4), 1.0), torch.full((1, 2), 2.0), torch.full((1, 3, 4, 4), 3.0)

        class _ThreeTupleOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
                return (dets, labels, third)

        wrapper = ModelWrapper(_ThreeTupleOutputModel())
        output = wrapper(torch.zeros(1, 3, 8, 8))
        assert len(output) == 3
        assert torch.equal(output[0], dets)
        assert torch.equal(output[1], labels)
        assert torch.equal(output[2], third)

    def test_list_output_converted_to_tuple(self) -> None:
        """A list output (:class:`_BackboneExport`'s backbone-only graph) must convert to a tuple.

        The backbone-only export path returns a plain ``list[Tensor]`` of feature maps rather than a tuple;
        ``convert_model`` requires a tuple, so the wrapper must coerce it rather than pass the list through as-is.
        """
        from rfdetr.export._openvino.exporter import ModelWrapper

        features = [torch.full((1, 3, 4, 4), 5.0), torch.full((1, 6, 2, 2), 6.0)]

        class _ListOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
                return features

        wrapper = ModelWrapper(_ListOutputModel())
        output = wrapper(torch.zeros(1, 3, 8, 8))
        assert isinstance(output, tuple)
        assert torch.equal(output[0], features[0])
        assert torch.equal(output[1], features[1])

    def test_dict_output_raises_not_implemented(self) -> None:
        """A dict output must raise ``NotImplementedError`` naming the mode-switch fix, never reshape silently.

        A dict reaching this wrapper means the caller forgot to call ``model.export()`` first (``forward_export`` always
        returns a tuple/list); the old behaviour silently mapped and sometimes dropped dict keys, which this test guards
        against regressing to.
        """
        from rfdetr.export._openvino.exporter import ModelWrapper

        class _DictOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
                return {"pred_boxes": torch.zeros(1, 4), "pred_logits": torch.zeros(1, 2)}

        wrapper = ModelWrapper(_DictOutputModel())
        with pytest.raises(NotImplementedError, match="model.export()"):
            wrapper(torch.zeros(1, 3, 8, 8))

    def test_unsupported_output_type_raises_type_error(self) -> None:
        """A model returning neither tuple, list, nor dict must raise ``TypeError``, not fail obscurely later."""
        from rfdetr.export._openvino.exporter import ModelWrapper

        class _TensorOutputModel(torch.nn.Module):
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return torch.zeros(1, 4)

        wrapper = ModelWrapper(_TensorOutputModel())
        with pytest.raises(TypeError, match="Unsupported model output type"):
            wrapper(torch.zeros(1, 3, 8, 8))


# ---------------------------------------------------------------------------
# format="openvino" wiring through RFDETR.export() (heavy deps mocked)
# ---------------------------------------------------------------------------


class TestExportFormatParameter:
    """Tests for ``format="openvino"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock heavy export deps so ``RFDETR.export()`` reaches the format dispatch without real work."""
        self._tmp_path = tmp_path
        xml_out = tmp_path / "inference_model.xml"
        xml_out.write_bytes(b"<xml/>")

        self._mock_stack = mock.patch.multiple(
            "rfdetr.export.main",
            make_infer_image=mock.DEFAULT,
            export_onnx=mock.DEFAULT,
        )
        mocks = self._mock_stack.start()
        mocks["make_infer_image"].return_value = torch.zeros(1, 3, 560, 560)
        self._mock_make_infer_image = mocks["make_infer_image"]
        self._mock_export_onnx = mocks["export_onnx"]
        self._mock_export_onnx.return_value = str(tmp_path / "inference_model.onnx")

        self._mock_export_openvino = mock.patch(
            "rfdetr.export._openvino.exporter.export_openvino", return_value=str(xml_out)
        ).start()

        yield

        mock.patch.stopall()

    @staticmethod
    def _make_rfdetr(*, segmentation_head: bool = False, use_grouppose_keypoints: bool = False) -> Any:
        """Create a minimal RFDETR instance with mocked internals (mirrors the CoreML/ExecuTorch suites)."""
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
        obj.size = "rfdetr-nano"
        return obj

    @pytest.mark.parametrize(
        "segmentation_head",
        [pytest.param(False, id="detection"), pytest.param(True, id="segmentation")],
    )
    def test_openvino_format_dispatches_to_export_openvino_not_onnx(self, segmentation_head: bool) -> None:
        """``format="openvino"`` must dispatch to ``export_openvino`` (not ``export_onnx``)."""
        obj = self._make_rfdetr(segmentation_head=segmentation_head)
        output_path = obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_called_once()
        self._mock_export_onnx.assert_not_called()
        assert output_path.suffix == ".xml"

    def test_onnx_format_does_not_call_export_openvino(self) -> None:
        """``format="onnx"`` must not import/call the OpenVINO converter."""
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_not_called()

    def test_variant_name_forwarded_to_converter(self) -> None:
        """The model's ``size`` attribute must be forwarded as ``variant_name``."""
        obj = self._make_rfdetr()
        obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_openvino.call_args.kwargs["variant_name"] == "rfdetr-nano"

    def test_output_name_forwarded_to_converter(self) -> None:
        """An explicit ``output_name`` must be forwarded verbatim, overriding the variant-based name."""
        obj = self._make_rfdetr()
        obj.export(format="openvino", output_dir=str(self._tmp_path / "out"), output_name="my-model")
        assert self._mock_export_openvino.call_args.kwargs["output_name"] == "my-model"

    def test_dynamic_batch_raises_not_implemented(self) -> None:
        """``dynamic_batch=True`` must raise ``NotImplementedError``, matching CoreML/ExecuTorch's fixed-shape guard.

        Regression guard: OpenVINO IR bakes a fixed input shape, so a silently-ignored
        ``dynamic_batch=True`` would produce a fixed-shape model with no feedback to the caller.
        """
        obj = self._make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(format="openvino", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)

    def test_dynamic_batch_raises_before_forward_pass(self) -> None:
        """``dynamic_batch=True`` must be rejected before ``make_infer_image`` runs the expensive forward pass.

        Regression guard: the rejection must be hoisted next to the ExecuTorch/CoreML fail-fast
        checks, matching their "reject before paying for a full DINOv2 forward" contract instead
        of living deep in the OpenVINO-specific dispatch path where it only fires after tracing.
        """
        obj = self._make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(format="openvino", output_dir=str(self._tmp_path / "out"), dynamic_batch=True)
        self._mock_make_infer_image.assert_not_called()

    def test_notes_warns_and_is_dropped(self) -> None:
        """A non-``None`` ``notes`` value must emit a ``UserWarning`` naming the missing metadata slot.

        Regression guard: OpenVINO IR has no ONNX-style metadata slot; silently dropping ``notes``
        would leave callers believing their metadata was embedded when it was not.
        """
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match="notes"):
            obj.export(format="openvino", output_dir=str(self._tmp_path / "out"), notes="some metadata")

    def test_invalid_format_raises_value_error(self) -> None:
        """Unknown ``format`` must raise ``ValueError`` listing supported formats, not reach the converter."""
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="Unsupported export format"):
            obj.export(format="bogus", output_dir=str(self._tmp_path / "out"))
        self._mock_export_openvino.assert_not_called()


class TestExportOpenvinoMissingDependencyViaPublicAPI:
    """``RFDETR.export(format="openvino")`` surfaces ``ImportError`` (not the registry ``ValueError``)."""

    @pytest.fixture(autouse=True)
    def _patch_light_export_deps(self, tmp_path: Path) -> Any:
        """Mock only ``make_infer_image``; the OpenVINO converter itself must run for real."""
        self._tmp_path = tmp_path
        self._mock_stack = mock.patch("rfdetr.export.main.make_infer_image", return_value=torch.zeros(1, 3, 560, 560))
        self._mock_stack.start()
        yield
        self._mock_stack.stop()

    @staticmethod
    def _make_rfdetr() -> Any:
        """Create a minimal RFDETR instance with mocked internals."""
        from rfdetr.detr import RFDETR

        obj = RFDETR.__new__(RFDETR)
        obj.model = mock.MagicMock()
        obj.model.resolution = 560
        obj.model.device = "cpu"
        obj.model.model.to.return_value = obj.model.model
        obj.model_config = mock.MagicMock()
        obj.model_config.segmentation_head = False
        obj.model_config.use_grouppose_keypoints = False
        obj.model_config.patch_size = 14
        obj.model_config.num_windows = 1
        obj.model_config.num_channels = 3
        return obj

    def test_raises_import_error_not_registry_value_error(self) -> None:
        """Missing ``openvino`` must surface ``ImportError`` — the format itself must be accepted.

        Regression guard for the format-dispatch registry: ``format="openvino"`` must reach the real
        ``export_openvino()`` call (and fail there, on the missing dependency) rather than being rejected
        upfront by ``_resolve_export_backend``'s ``_EXPORT_FORMATS`` membership check.
        """
        obj = self._make_rfdetr()
        with pytest.raises(ImportError):
            obj.export(format="openvino", output_dir=str(self._tmp_path / "out"))


# ---------------------------------------------------------------------------
# End-to-end (gated) — real convert + parity vs eager PyTorch
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def people_walking_image_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download supervision's ``PEOPLE_WALKING`` asset once, shared across OpenVINO e2e tests.

    The detection/segmentation/keypoint parity tests need a real photo, not the structured gradient+checkerboard input:
    on a real trained model, a synthetic background-only image produces no confident detections, so every one of the
    ~300 two-stage query-selection candidates sits at a similarly low objectness score. That makes their relative order
    tip over ordinary cross-backend floating-point noise, which then dominates a positional comparison with unrelated
    background-candidate reordering (see ``_confident_query_diffs``). A real photo gives genuine, well-separated
    detections whose positions stay stable.
    """
    asset_dir = tmp_path_factory.mktemp("openvino_assets")
    cwd = Path.cwd()
    os.chdir(asset_dir)
    try:
        from supervision.assets import ImageAssets, download_assets

        return Path(download_assets(ImageAssets.PEOPLE_WALKING)).resolve()
    finally:
        os.chdir(cwd)


@pytest.fixture(scope="module")
def openvino_detection_export(
    tmp_path_factory: pytest.TempPathFactory, people_walking_image_path: Path
) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRNano to OpenVINO IR once, shared across the gated detection e2e tests."""
    pytest.importorskip("openvino")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("openvino_nano")
    detector = rfdetr.RFDETRNano()
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _parity_input_from_image(people_walking_image_path, resolution)
    return model, example, Path(xml_path)


@pytest.fixture(scope="module")
def openvino_segmentation_export(
    tmp_path_factory: pytest.TempPathFactory, people_walking_image_path: Path
) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRSegNano to OpenVINO IR once, shared across the gated segmentation e2e test.

    Regression coverage for M-5: the gated OpenVINO e2e suite previously covered detection and
    backbone-only exports only, so ``ModelWrapper``'s 3-tuple (``dets, labels, masks``) path was
    never exercised end-to-end through a real ``convert_model`` call.
    """
    pytest.importorskip("openvino")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("openvino_seg_nano")
    detector = rfdetr.RFDETRSegNano()
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _parity_input_from_image(people_walking_image_path, resolution)
    return model, example, Path(xml_path)


@pytest.fixture(scope="module")
def openvino_keypoint_export(
    tmp_path_factory: pytest.TempPathFactory, people_walking_image_path: Path
) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRKeypointPreview to OpenVINO IR once, shared across the gated keypoint e2e test.

    Regression coverage for M-5: the gated OpenVINO e2e suite previously covered detection and
    backbone-only exports only, so ``ModelWrapper``'s 3-tuple (``dets, labels, keypoints``) path was
    never exercised end-to-end through a real ``convert_model`` call.
    """
    pytest.importorskip("openvino")
    import rfdetr

    out_dir = tmp_path_factory.mktemp("openvino_keypoint")
    detector = rfdetr.RFDETRKeypointPreview()
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    resolution = int(detector.model.resolution)
    example = _parity_input_from_image(people_walking_image_path, resolution)
    return model, example, Path(xml_path)


@pytest.fixture(scope="module")
def openvino_backbone_export(tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
    """Export RFDETRNano's backbone-only OpenVINO IR once, shared across the gated backbone e2e test."""
    pytest.importorskip("openvino")
    import rfdetr
    from rfdetr.export._backend import _BackboneExport

    out_dir = tmp_path_factory.mktemp("openvino_backbone")
    detector = rfdetr.RFDETRNano(pretrain_weights=None)
    xml_path = detector.export(output_dir=str(out_dir), format="openvino", backbone_only=True, verbose=False)
    backbone = detector.model.model.backbone[0].to("cpu").eval()
    reference_model = _BackboneExport(backbone)
    resolution = int(detector.model.resolution)
    example = _structured_parity_input(1, 3, resolution, resolution)
    return reference_model, example, Path(xml_path)


@pytest.mark.e2e_openvino
class TestOpenVINOEndToEnd:
    """Real OpenVINO IR export + CPU numerical parity (``-m e2e_openvino``, requires ``openvino`` installed)."""

    def test_xml_and_bin_written(self, openvino_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """Export must write both the ``.xml`` graph and its companion ``.bin`` weights file."""
        _, _, xml_path = openvino_detection_export
        assert xml_path.exists()
        assert xml_path.suffix == ".xml"
        assert xml_path.with_suffix(".bin").exists()

    def test_detection_outputs_match_pytorch(self, openvino_detection_export: tuple[Any, torch.Tensor, Path]) -> None:
        """OpenVINO detection output (boxes, logits) must match eager PyTorch on confident detections.

        Compared only over the top-10 highest-confidence queries -- see ``_confident_query_diffs``
        for why raw, unfiltered two-stage query-selection output is the wrong comparison here.
        Measured maxima on this fixture: box ~2e-5, label ~0.01; tolerances give ~10x headroom.
        """
        model, example, xml_path = openvino_detection_export
        eager_tensors = eager_reference_tensors(model, example)
        ov_tensors = [torch.from_numpy(output) for output in _infer_openvino_f32(xml_path, example.numpy())]

        assert len(ov_tensors) == 2, f"detection export must yield (boxes, logits), got {len(ov_tensors)} outputs"
        box_diff, label_diff = _confident_query_diffs(eager_tensors, ov_tensors)
        assert box_diff < 1e-3, f"OpenVINO detection boxes diverge from PyTorch: max abs diff {box_diff}"
        assert label_diff < 0.1, f"OpenVINO detection logits diverge from PyTorch: max abs diff {label_diff}"

    def test_segmentation_outputs_match_pytorch(
        self, openvino_segmentation_export: tuple[Any, torch.Tensor, Path]
    ) -> None:
        """OpenVINO segmentation output (boxes, logits, masks) must match eager PyTorch on confident detections.

        Exercises ``ModelWrapper``'s 3-tuple pass-through path (see
        ``TestModelWrapper::test_three_tuple_output_passes_through_unchanged`` for the unit-level
        version) through a real ``convert_model`` + IR-inference round trip. Compared only over the
        top-10 highest-confidence queries (see ``_confident_query_diffs``); masks are compared in
        sigmoid (probability) space since mask logits span a much wider range than boxes/labels, so
        an equivalent raw-space tolerance would be too loose to catch real regressions. Measured
        maxima on this fixture: box ~2e-4, label ~0.08, mask (sigmoid) ~0.026.
        """
        model, example, xml_path = openvino_segmentation_export
        eager_tensors = eager_reference_tensors(model, example)
        ov_tensors = [torch.from_numpy(output) for output in _infer_openvino_f32(xml_path, example.numpy())]

        assert len(ov_tensors) == 3, f"segmentation export must yield (boxes, logits, masks), got {len(ov_tensors)}"
        box_diff, label_diff, mask_diff = _confident_query_diffs(eager_tensors, ov_tensors, sigmoid_indices={2})
        assert box_diff < 1e-3, f"OpenVINO segmentation boxes diverge from PyTorch: max abs diff {box_diff}"
        assert label_diff < 0.1, f"OpenVINO segmentation logits diverge from PyTorch: max abs diff {label_diff}"
        assert mask_diff < 0.05, f"OpenVINO segmentation masks diverge from PyTorch (sigmoid space): {mask_diff}"

    def test_keypoint_outputs_match_pytorch(self, openvino_keypoint_export: tuple[Any, torch.Tensor, Path]) -> None:
        """OpenVINO keypoint output (boxes, logits, keypoints) must match eager PyTorch on confident detections.

        Exercises ``ModelWrapper``'s 3-tuple pass-through path (see
        ``TestModelWrapper::test_three_tuple_output_passes_through_unchanged`` for the unit-level
        version) through a real ``convert_model`` + IR-inference round trip. Compared only over the
        top-10 highest-confidence queries (see ``_confident_query_diffs``). Measured maxima on this
        fixture: box ~4e-5, label ~0.008, keypoints (raw, scale up to ~20) ~0.053.
        """
        model, example, xml_path = openvino_keypoint_export
        eager_tensors = eager_reference_tensors(model, example)
        ov_tensors = [torch.from_numpy(output) for output in _infer_openvino_f32(xml_path, example.numpy())]

        assert len(ov_tensors) == 3, f"keypoint export must yield (boxes, logits, keypoints), got {len(ov_tensors)}"
        box_diff, label_diff, keypoint_diff = _confident_query_diffs(eager_tensors, ov_tensors)
        assert box_diff < 1e-3, f"OpenVINO keypoint boxes diverge from PyTorch: max abs diff {box_diff}"
        assert label_diff < 0.1, f"OpenVINO keypoint logits diverge from PyTorch: max abs diff {label_diff}"
        assert keypoint_diff < 0.1, f"OpenVINO keypoints diverge from PyTorch: max abs diff {keypoint_diff}"

    def test_backbone_outputs_match_pytorch(
        self, openvino_backbone_export: tuple[torch.nn.Module, torch.Tensor, Path]
    ) -> None:
        """OpenVINO must run every backbone feature-map output from the public backbone-only export.

        No two-stage query selection is involved here (the backbone has no ``topk``), so the structured
        gradient+checkerboard input and a plain positional comparison are fine -- unlike the
        detection/segmentation/keypoint tests above. The tolerance is looser than a typical kernel-rounding bound
        because OpenVINO's own float32-pinned CPU execution of this backbone measures ~0.006 max abs diff against eager
        PyTorch (measured on this fixture, ~10x CoreML's equivalent <1e-4 for the same module) -- a real, currently-
        unexplained precision gap specific to this backbone's OpenVINO conversion, tracked separately from this PR.
        """
        model, example, xml_path = openvino_backbone_export
        assert "backbone" in xml_path.stem
        eager_tensors = eager_reference_tensors(model, example)
        ov_tensors = [torch.from_numpy(output) for output in _infer_openvino_f32(xml_path, example.numpy())]

        diffs = max_abs_output_diffs(eager_tensors, ov_tensors, check_shape=True)
        assert max(diffs) < 1e-2, f"OpenVINO backbone outputs diverge from PyTorch: max abs diff {max(diffs)}"


class TestOpenVINOInferenceEndToEnd:
    """Real ``OpenVINOInference`` behaviour gated behind an actual ``openvino`` install."""

    def test_missing_model_file_raises_file_not_found(self, tmp_path: Path) -> None:
        """A nonexistent ``.xml`` path must raise ``FileNotFoundError`` once ``openvino`` is importable.

        Unreachable without ``openvino`` installed (see ``TestOpenVINOInferenceMissingDependency`` for the ungated
        ``ImportError`` coverage of the same constructor when the import itself fails first).
        """
        pytest.importorskip("openvino")
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            OpenVINOInference(tmp_path / "does-not-exist.xml")
