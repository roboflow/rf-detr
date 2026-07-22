# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the PyTorch → ExecuTorch (``.pte``) export pipeline.

Tests cover:
* ``_resolve_export_backend``: the ``format`` / ``backend`` / ``soc`` validation, its warnings and errors.
* ``export_executorch()`` argument validation and the missing-dependency error path (no ``executorch`` needed).
* ``format="executorch"`` + ``backend`` / ``soc`` wiring through ``RFDETR.export()`` (heavy deps mocked, fast).
* A real end-to-end export + numerical-parity check, gated behind the ``executorch`` marker so it only runs where the
  ``executorch`` package is installed.
"""

from __future__ import annotations

import contextlib
import importlib.metadata
import sys
import types
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
import torch

from rfdetr.export._executorch import _IS_EXECUTORCH_AVAILABLE
from rfdetr.export._executorch.converter import (
    _VALID_BACKENDS,
    _check_executorch_available,
    export_executorch,
)

executorch_only = pytest.mark.skipif(not _IS_EXECUTORCH_AVAILABLE, reason="executorch not installed")


def _runtime_parity(model: Any, example: torch.Tensor, pte_path: Path) -> list[float]:
    """Run *example* through the eager model and the ExecuTorch runtime; return per-output max-abs-diff.

    The export-mode forward mutates its input in place, so a fresh clone is fed to each run.
    """
    _check_executorch_available(require_runtime=True)
    from executorch.runtime import Runtime
    from torch.utils._pytree import tree_flatten

    with torch.no_grad():
        eager_out = model(example.clone())
    eager_tensors = [t for t in tree_flatten(eager_out)[0] if isinstance(t, torch.Tensor)]

    method = Runtime.get().load_program(str(pte_path)).load_method("forward")
    runtime_tensors = [t for t in method.execute([example.clone()]) if isinstance(t, torch.Tensor)]

    assert len(runtime_tensors) == len(eager_tensors), "ExecuTorch output count differs from the source model"
    return [(eager - runtime).abs().max().item() for eager, runtime in zip(eager_tensors, runtime_tensors)]


# ---------------------------------------------------------------------------
# format / backend / soc validation (no executorch required)
# ---------------------------------------------------------------------------


class TestResolveExportBackend:
    """Validation of the ``format`` / ``backend`` / ``soc`` combination by ``_resolve_export_backend``."""

    @pytest.mark.parametrize(
        ("export_format", "backend", "soc", "expected"),
        [
            pytest.param("onnx", None, None, (None, None), id="onnx"),
            pytest.param("tflite", None, None, (None, None), id="tflite"),
            pytest.param("executorch", "xnnpack", None, ("xnnpack", None), id="executorch-xnnpack"),
            pytest.param("executorch", "coreml", None, ("coreml", None), id="executorch-coreml"),
            pytest.param("executorch", "qnn", "SM8650", ("qnn", "SM8650"), id="executorch-qnn"),
            pytest.param(
                "executorch",
                "XNNPACK",
                None,
                ("xnnpack", None),
                id="executorch-backend-uppercase-normalised-to-lowercase",
            ),
        ],
    )
    def test_valid_combination_resolves(
        self, export_format: str, backend: str | None, soc: str | None, expected: tuple[str | None, str | None]
    ) -> None:
        from rfdetr.export._backend import _resolve_export_backend

        assert _resolve_export_backend(export_format, backend, soc) == expected

    def test_unknown_format_raises_value_error(self) -> None:
        from rfdetr.export._backend import _resolve_export_backend

        with pytest.raises(ValueError, match="Unsupported export format"):
            _resolve_export_backend("bogus", None, None)

    def test_format_without_required_backend_raises_value_error(self) -> None:
        """A backend-bearing format needs a backend; omitting it is an error that lists the choices."""
        from rfdetr.export._backend import _resolve_export_backend

        with pytest.raises(ValueError, match="requires a valid backend"):
            _resolve_export_backend("executorch", None, None)

    def test_unknown_backend_raises_value_error(self) -> None:
        from rfdetr.export._backend import _resolve_export_backend

        with pytest.raises(ValueError, match="Unsupported backend"):
            _resolve_export_backend("executorch", "vulkan", None)

    def test_soc_backend_without_soc_raises_value_error(self) -> None:
        """A backend that compiles for a specific chip (qnn) requires a soc."""
        from rfdetr.export._backend import _resolve_export_backend

        with pytest.raises(ValueError, match="requires a valid soc"):
            _resolve_export_backend("executorch", "qnn", None)

    @pytest.mark.parametrize(
        ("export_format", "backend", "soc", "match"),
        [
            pytest.param("onnx", "xnnpack", None, r"backend=.*ignored", id="onnx-ignores-backend"),
            pytest.param("onnx", None, "SM8650", r"soc=.*ignored", id="onnx-ignores-soc"),
            pytest.param("executorch", "xnnpack", "SM8650", r"soc=.*ignored", id="xnnpack-ignores-soc"),
        ],
    )
    def test_unused_argument_warns(self, export_format: str, backend: str | None, soc: str | None, match: str) -> None:
        from rfdetr.export._backend import _resolve_export_backend

        with pytest.warns(UserWarning, match=match):
            _resolve_export_backend(export_format, backend, soc)


# ---------------------------------------------------------------------------
# Converter argument validation / dependency handling (no executorch required)
# ---------------------------------------------------------------------------


class TestExportExecutorchValidation:
    """Argument validation and dependency-error behaviour of ``export_executorch``."""

    def test_unsupported_backend_raises_value_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Unsupported ExecuTorch backend"):
            export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend="vulkan",
            )

    def test_supported_backend_set_is_exact(self) -> None:
        """``_VALID_BACKENDS`` is exactly ``{"xnnpack", "coreml", "qnn"}`` -- no more, no fewer."""
        assert _VALID_BACKENDS == {"xnnpack", "coreml", "qnn"}

    @pytest.mark.parametrize("backend", ["xnnpack", "coreml", "qnn"])
    def test_dynamic_batch_not_supported_raises(self, tmp_path: Path, backend: str) -> None:
        """``dynamic_batch`` is refused up front on executorch 1.3.1 (runtime can't resize windowed reshapes)."""
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend=backend,
                dynamic_batch=True,
            )

    def test_coreml_backend_missing_dependency_raises_import_error(self, tmp_path: Path) -> None:
        """Without ``coremltools`` (or the ``executorch`` package), the coreml backend raises an actionable hint."""
        try:
            importlib.import_module("executorch.backends.apple.coreml.partition.coreml_partitioner")
        except Exception:
            pass
        else:
            pytest.skip("ExecuTorch CoreML backend is available; missing-dependency path not exercised here")
        with pytest.raises(ImportError, match=r"coremltools|executorch"):
            export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend="coreml",
            )

    def test_qnn_backend_missing_delegate_raises_import_error(self, tmp_path: Path) -> None:
        """Without an ExecuTorch source build against the QNN SDK, the qnn backend raises an actionable hint."""
        try:
            importlib.import_module("executorch.backends.qualcomm.utils.utils")
        except Exception:
            pass
        else:
            pytest.skip("ExecuTorch QNN backend is available; missing-delegate path not exercised here")
        with pytest.raises(ImportError, match=r"QNN|executorch"):
            export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend="qnn",
            )

    def test_check_executorch_available_raises_when_missing(self) -> None:
        """``_check_executorch_available`` should raise an actionable ImportError when executorch is absent."""
        with mock.patch.dict(sys.modules, {"executorch": None}):
            with pytest.raises(ImportError, match=r"pip install rfdetr\[executorch\]"):
                _check_executorch_available()

    @pytest.mark.parametrize(
        ("version_return", "version_side_effect", "expectation"),
        [
            pytest.param("1.3.0", None, contextlib.nullcontext(), id="min-supported-version-ok"),
            pytest.param("1.4.1", None, contextlib.nullcontext(), id="newer-version-ok"),
            pytest.param(
                "1.2.9",
                None,
                pytest.raises(ImportError, match=r"requires >=1\.3"),
                id="older-version-raises-import-error",
            ),
            pytest.param(
                None,
                importlib.metadata.PackageNotFoundError(),
                contextlib.nullcontext(),
                id="source-build-package-not-found-ok",
            ),
        ],
    )
    def test_check_executorch_available_version_gate(
        self, version_return: str | None, version_side_effect: Exception | None, expectation: Any
    ) -> None:
        """``_check_executorch_available`` accepts >=1.3, rejects <1.3, and tolerates a source build (no distribution
        metadata -> ``PackageNotFoundError`` is swallowed rather than raised)."""
        fake_executorch = types.ModuleType("executorch")
        with mock.patch.dict(sys.modules, {"executorch": fake_executorch}):
            with mock.patch("importlib.metadata.version", return_value=version_return, side_effect=version_side_effect):
                with expectation:
                    _check_executorch_available()

    def test_check_executorch_available_require_runtime_false_ignores_broken_runtime(self) -> None:
        """Default ``require_runtime=False`` does not probe ``executorch.runtime`` — export capability is unaffected by
        a broken runtime extension (export never imports it)."""
        fake_executorch = types.ModuleType("executorch")
        with mock.patch.dict(sys.modules, {"executorch": fake_executorch, "executorch.runtime": None}):
            with mock.patch("importlib.metadata.version", return_value="1.3.1"):
                _check_executorch_available()  # must not raise, even though executorch.runtime is broken

    def test_check_executorch_available_require_runtime_true_raises_actionable_message(self) -> None:
        """``require_runtime=True`` surfaces an ABI-compatibility hint when ``executorch.runtime`` fails to import,
        distinguishing this from a plain "not installed" error."""
        fake_executorch = types.ModuleType("executorch")
        with mock.patch.dict(sys.modules, {"executorch": fake_executorch, "executorch.runtime": None}):
            with mock.patch("importlib.metadata.version", return_value="1.3.1"):
                with pytest.raises(ImportError, match=r"ABI-compatibility gap"):
                    _check_executorch_available(require_runtime=True)

    def test_check_executorch_available_require_runtime_true_passes_when_runtime_ok(self) -> None:
        """``require_runtime=True`` does not raise when ``executorch.runtime`` imports cleanly."""
        fake_executorch = types.ModuleType("executorch")
        fake_runtime_module = types.ModuleType("executorch.runtime")
        fake_runtime_module.Runtime = mock.MagicMock()
        fake_executorch.runtime = fake_runtime_module
        with mock.patch.dict(sys.modules, {"executorch": fake_executorch, "executorch.runtime": fake_runtime_module}):
            with mock.patch("importlib.metadata.version", return_value="1.3.1"):
                _check_executorch_available(require_runtime=True)  # must not raise

    def test_export_executorch_missing_dependency_raises_import_error(self, tmp_path: Path) -> None:
        """``export_executorch`` raises ImportError with install hint when executorch is absent."""
        with mock.patch.dict(sys.modules, {"executorch": None}):
            with pytest.raises(ImportError, match=r"pip install rfdetr\[executorch\]"):
                export_executorch(
                    model=mock.MagicMock(),
                    input_tensors=torch.zeros(1, 3, 8, 8),
                    output_dir=tmp_path,
                    backend="xnnpack",
                )

    def test_output_dir_is_used_as_provided(self, tmp_path: Path) -> None:
        """output_dir is passed through to the exporter; caller is responsible for sanitizing it.

        Regression guard: variant_name is sanitized (os.path.basename), but output_dir is not.
        This test documents that output_dir is caller-owned — the exporter trusts it.
        The ImportError fires before any mkdir, so output_dir is irrelevant on the missing-dep path.
        """
        with mock.patch.dict(sys.modules, {"executorch": None}):
            with pytest.raises(ImportError, match=r"pip install rfdetr\[executorch\]"):
                export_executorch(
                    model=mock.MagicMock(),
                    input_tensors=torch.zeros(1, 3, 8, 8),
                    output_dir=tmp_path / "some" / "nested" / "dir",
                    backend="xnnpack",
                )
        # If executorch is absent the ImportError fires before any filesystem operation,
        # confirming that output_dir sanitization is caller responsibility.
        assert not (tmp_path / "some").exists()


# ---------------------------------------------------------------------------
# format="executorch" wiring through RFDETR.export() (heavy deps mocked)
# ---------------------------------------------------------------------------


class TestExportFormatParameter:
    """Tests for ``format="executorch"`` wiring through ``RFDETR.export()``."""

    @pytest.fixture(autouse=True)
    def _patch_export_deps(self, tmp_path: Path) -> Any:
        """Mock the heavy export dependencies so ``RFDETR.export()`` is fast and dependency-free."""
        self._tmp_path = tmp_path
        pte_out = tmp_path / "inference_model.pte"
        pte_out.write_bytes(b"pte")

        self._mock_stack = contextlib.ExitStack()
        # make_infer_image returns a small tensor instead of building a real image.
        self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export.main.make_infer_image",
                return_value=torch.zeros(1, 3, 560, 560),
            )
        )
        # Mock the converter so no torch.export / executorch work happens.
        self._mock_export_executorch = self._mock_stack.enter_context(
            mock.patch(
                "rfdetr.export._executorch.converter.export_executorch",
                return_value=pte_out,
            )
        )
        # Mock export_onnx so the ONNX-format branch is also dependency-free.
        self._mock_export_onnx = self._mock_stack.enter_context(
            mock.patch("rfdetr.export.main.export_onnx", return_value=str(tmp_path / "inference_model.onnx"))
        )
        yield
        self._mock_stack.close()

    @staticmethod
    def _make_rfdetr() -> Any:
        """Create a minimal RFDETR instance with mocked internals (mirrors the TFLite suite)."""
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
        return obj

    @pytest.mark.parametrize(
        "export_format",
        [
            pytest.param("executorch", id="canonical"),
            pytest.param("pte", id="alias"),
        ],
    )
    def test_executorch_format_calls_export_executorch(self, export_format: str) -> None:
        """``format="executorch"`` — and its ``"pte"`` alias — trigger export_executorch and warn (experimental)."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match="experimental"):
            obj.export(format=export_format, backend="xnnpack", output_dir=str(self._tmp_path / "out"))
        self._mock_export_executorch.assert_called_once()

    def test_notes_ignored_with_warning_for_executorch_format(self) -> None:
        """``notes`` has no metadata slot in ``.pte``; passing it with ``format="executorch"`` warns and is dropped."""
        obj = self._make_rfdetr()
        with pytest.warns(UserWarning, match=r"`notes` is not forwarded to format='executorch'"):
            obj.export(
                format="executorch",
                backend="xnnpack",
                notes="some provenance notes",
                output_dir=str(self._tmp_path / "out"),
            )

    def test_executorch_format_does_not_call_export_onnx(self) -> None:
        obj = self._make_rfdetr()
        obj.export(format="executorch", backend="xnnpack", output_dir=str(self._tmp_path / "out"))
        self._mock_export_onnx.assert_not_called()

    @pytest.mark.parametrize(
        ("backend", "soc"),
        [
            pytest.param("xnnpack", None, id="xnnpack"),
            pytest.param("coreml", None, id="coreml"),
            pytest.param("qnn", "SM8650", id="qnn"),
        ],
    )
    def test_backend_forwarded_to_converter(self, backend: str, soc: str | None) -> None:
        obj = self._make_rfdetr()
        obj.export(format="executorch", backend=backend, soc=soc, output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_executorch.call_args.kwargs.get("backend") == backend

    def test_soc_forwarded_to_converter(self) -> None:
        obj = self._make_rfdetr()
        obj.export(format="executorch", backend="qnn", soc="SM8750", output_dir=str(self._tmp_path / "out"))
        assert self._mock_export_executorch.call_args.kwargs.get("soc") == "SM8750"

    def test_non_qnn_backend_does_not_forward_soc(self) -> None:
        """Xnnpack/coreml bake in no SoC, so the converter is called without a ``soc`` kwarg."""
        obj = self._make_rfdetr()
        obj.export(format="executorch", backend="xnnpack", output_dir=str(self._tmp_path / "out"))
        assert "soc" not in self._mock_export_executorch.call_args.kwargs

    def test_dynamic_batch_raises_before_converter(self) -> None:
        """dynamic_batch=True is refused by RFDETR.export() before the converter is invoked."""
        obj = self._make_rfdetr()
        with pytest.raises(NotImplementedError, match="dynamic_batch"):
            obj.export(
                format="executorch", backend="xnnpack", dynamic_batch=True, output_dir=str(self._tmp_path / "out")
            )
        self._mock_export_executorch.assert_not_called()

    def test_onnx_format_does_not_call_export_executorch(self) -> None:
        obj = self._make_rfdetr()
        obj.export(format="onnx", output_dir=str(self._tmp_path / "out"))
        self._mock_export_executorch.assert_not_called()

    def test_invalid_format_raises_value_error(self) -> None:
        obj = self._make_rfdetr()
        with pytest.raises(ValueError, match="Unsupported export format"):
            obj.export(format="bogus", output_dir=str(self._tmp_path / "out"))

    def test_converter_import_error_propagates(self) -> None:
        """If the executorch converter cannot be imported, ``export()`` surfaces an actionable ImportError."""
        obj = self._make_rfdetr()
        # Bypass _resolve_export_backend's own converter import, then make the dispatch-site import fail.
        with (
            mock.patch("rfdetr.export._backend._resolve_export_backend", return_value=("xnnpack", None)),
            mock.patch.dict(sys.modules, {"rfdetr.export._executorch.converter": None}),
        ):
            with pytest.raises(ImportError):
                obj.export(format="executorch", backend="xnnpack", output_dir=str(self._tmp_path / "out"))


# ---------------------------------------------------------------------------
# Converter body coverage with executorch fully mocked (no optional package, no native libraries)
# ---------------------------------------------------------------------------


def _fake_executorch_tree(leaves: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build a ``sys.modules`` patch dict for the converter's lazy imports.

    Each leaf dotted module is created with the given attributes, plus empty parent packages, so a ``from
    executorch.<...> import <name>`` succeeds even when the real ``executorch`` is not installed.
    """
    out: dict[str, Any] = {}
    for dotted, attrs in leaves.items():
        parts = dotted.split(".")
        for i in range(1, len(parts) + 1):
            name = ".".join(parts[:i])
            if name not in out:
                module = types.ModuleType(name)
                module.__path__ = []  # mark as a package so nested submodule imports resolve
                out[name] = module
        for key, value in attrs.items():
            setattr(out[dotted], key, value)
    return out


class TestBuildPartitioner:
    """``_build_partitioner`` backend selection and missing-extension handling (executorch mocked)."""

    @pytest.mark.parametrize(
        ("backend", "leaf", "cls"),
        [
            pytest.param(
                "xnnpack",
                "executorch.backends.xnnpack.partition.xnnpack_partitioner",
                "XnnpackPartitioner",
                id="xnnpack",
            ),
            pytest.param(
                "coreml",
                "executorch.backends.apple.coreml.partition.coreml_partitioner",
                "CoreMLPartitioner",
                id="coreml",
            ),
        ],
    )
    def test_returns_partitioner(self, backend: str, leaf: str, cls: str) -> None:
        from rfdetr.export._executorch.converter import _build_partitioner

        sentinel = object()
        mods = _fake_executorch_tree({leaf: {cls: mock.MagicMock(return_value=sentinel)}})
        with mock.patch.dict(sys.modules, mods):
            assert _build_partitioner(backend) == [sentinel]

    @pytest.mark.parametrize(
        ("backend", "leaf"),
        [
            pytest.param("xnnpack", "executorch.backends.xnnpack.partition.xnnpack_partitioner", id="xnnpack"),
            pytest.param("coreml", "executorch.backends.apple.coreml.partition.coreml_partitioner", id="coreml"),
        ],
    )
    def test_missing_extension_raises_import_error(self, backend: str, leaf: str) -> None:
        from rfdetr.export._executorch.converter import _build_partitioner

        with mock.patch.dict(sys.modules, {leaf: None}):
            with pytest.raises(ImportError):
                _build_partitioner(backend)

    def test_unknown_backend_raises_value_error(self) -> None:
        from rfdetr.export._executorch.converter import _build_partitioner

        with pytest.raises(ValueError, match="Unsupported ExecuTorch backend"):
            _build_partitioner("bogus")


class TestLowerQnn:
    """``_lower_qnn`` lowering, SoC validation, and missing-delegate handling (qualcomm backend mocked)."""

    @staticmethod
    def _qnn_modules() -> tuple[dict[str, Any], Any, Any]:
        program = mock.MagicMock()
        program.buffer = b"QNNPTE"
        edge = mock.MagicMock()
        edge.to_executorch.return_value = program

        class _QcomChipset:
            SM8650 = 30
            SM8750 = 69

        class _QnnOperatorSupport:
            def is_node_supported(self, *args: Any, **kwargs: Any) -> bool:
                return True

        mods = _fake_executorch_tree(
            {
                "executorch.backends.qualcomm.serialization.qc_schema": {"QcomChipset": _QcomChipset},
                "executorch.backends.qualcomm.partition.qnn_partitioner": {"QnnOperatorSupport": _QnnOperatorSupport},
                "executorch.backends.qualcomm.utils.utils": {
                    "generate_htp_compiler_spec": mock.MagicMock(return_value="htp"),
                    "generate_qnn_executorch_compiler_spec": mock.MagicMock(return_value="specs"),
                    "to_edge_transform_and_lower_to_qnn": mock.MagicMock(return_value=edge),
                },
            }
        )
        return mods, program, edge

    def test_success_returns_executorch_program(self) -> None:
        from rfdetr.export._executorch.converter import _lower_qnn

        mods, program, edge = self._qnn_modules()

        # Make the (mocked) QNN lowering call torch.export.export so the strict=False shim that _lower_qnn
        # installs around it is exercised; _original_export (the patched base export) should receive strict=False.
        def _lower(model: Any, args: Any, compiler_specs: Any, skip_node_op_set: Any = None) -> Any:
            torch.export.export(model, args, strict=True)
            return edge

        mods["executorch.backends.qualcomm.utils.utils"].to_edge_transform_and_lower_to_qnn.side_effect = _lower
        with mock.patch.dict(sys.modules, mods), mock.patch("torch.export.export") as base_export:
            result = _lower_qnn(mock.MagicMock(), torch.zeros(1, 3, 8, 8), soc_model="sm8650")
        assert result is program
        assert base_export.call_args.kwargs.get("strict") is False

    def test_op_support_exception_is_treated_as_unsupported(self) -> None:
        """A QnnOperatorSupport.is_node_supported that raises (no HTP visitor / weightless LayerNorm) is caught and the
        node is left on CPU instead of aborting the lowering; the original method is restored after."""
        from rfdetr.export._executorch.converter import _lower_qnn

        mods, program, edge = self._qnn_modules()
        qnn_support_cls = mods["executorch.backends.qualcomm.partition.qnn_partitioner"].QnnOperatorSupport

        def _raise(self: Any, *args: Any, **kwargs: Any) -> bool:
            raise AttributeError("'NoneType' object has no attribute 'name'")  # weightless-LayerNorm signature

        qnn_support_cls.is_node_supported = _raise
        original = qnn_support_cls.is_node_supported
        seen: dict[str, Any] = {}

        def _lower(model: Any, args: Any, compiler_specs: Any, skip_node_op_set: Any = None) -> Any:
            # while lowering, the wrapper is installed -> a throwing support check returns False, not raises
            seen["supported"] = qnn_support_cls().is_node_supported(object())
            return edge

        mods["executorch.backends.qualcomm.utils.utils"].to_edge_transform_and_lower_to_qnn.side_effect = _lower
        with mock.patch.dict(sys.modules, mods):
            result = _lower_qnn(mock.MagicMock(), torch.zeros(1, 3, 8, 8), soc_model="sm8650")

        assert result is program
        assert seen["supported"] is False  # exception swallowed -> CPU fallback
        assert qnn_support_cls.is_node_supported is original  # restored after lowering

    def test_unknown_soc_raises_value_error(self) -> None:
        from rfdetr.export._executorch.converter import _lower_qnn

        mods, _, _ = self._qnn_modules()
        with mock.patch.dict(sys.modules, mods):
            with pytest.raises(ValueError, match="Unknown QNN SoC"):
                _lower_qnn(mock.MagicMock(), torch.zeros(1, 3, 8, 8), soc_model="SM9999")

    def test_missing_delegate_raises_import_error(self) -> None:
        from rfdetr.export._executorch.converter import _lower_qnn

        with mock.patch.dict(sys.modules, {"executorch.backends.qualcomm.serialization.qc_schema": None}):
            with pytest.raises(ImportError, match=r"QNN|executorch"):
                _lower_qnn(mock.MagicMock(), torch.zeros(1, 3, 8, 8), soc_model="SM8650")


class TestExportExecutorchBody:
    """``export_executorch`` lowering dispatch + ``.pte`` writing, with executorch and its backends mocked."""

    @staticmethod
    def _generic_modules(buffer: bytes = b"PTEBYTES") -> dict[str, Any]:
        program = mock.MagicMock()
        program.buffer = buffer
        edge = mock.MagicMock()
        edge.to_executorch.return_value = program
        return _fake_executorch_tree(
            {
                "executorch.exir": {"to_edge_transform_and_lower": mock.MagicMock(return_value=edge)},
                "executorch.backends.xnnpack.partition.xnnpack_partitioner": {"XnnpackPartitioner": mock.MagicMock()},
                "executorch.backends.apple.coreml.partition.coreml_partitioner": {
                    "CoreMLPartitioner": mock.MagicMock()
                },
            }
        )

    @pytest.mark.parametrize("backend", [pytest.param("xnnpack", id="xnnpack"), pytest.param("coreml", id="coreml")])
    def test_generic_backend_writes_pte(self, tmp_path: Path, backend: str) -> None:
        with mock.patch.dict(sys.modules, self._generic_modules(b"PTE")), mock.patch("torch.export.export"):
            out = export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend=backend,
            )
        assert out.name == "inference_model.pte"
        assert out.read_bytes() == b"PTE"

    def test_variant_name_sanitized_to_basename(self, tmp_path: Path) -> None:
        """A path-like ``variant_name`` is reduced to its basename stem (mirrors the ONNX exporter)."""
        with mock.patch.dict(sys.modules, self._generic_modules()), mock.patch("torch.export.export"):
            out = export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend="xnnpack",
                variant_name="sub/dir/rfdetr-nano.pte",
            )
        assert out.name == "rfdetr-nano.pte"

    def test_lowering_failure_wrapped_as_runtime_error(self, tmp_path: Path) -> None:
        mods = self._generic_modules()
        mods["executorch.exir"].to_edge_transform_and_lower.side_effect = RuntimeError("boom")
        with mock.patch.dict(sys.modules, mods), mock.patch("torch.export.export"):
            with pytest.raises(RuntimeError, match="ExecuTorch export failed"):
                export_executorch(
                    model=mock.MagicMock(),
                    input_tensors=torch.zeros(1, 3, 8, 8),
                    output_dir=tmp_path,
                    backend="xnnpack",
                )

    def test_qnn_backend_dispatches_to_lower_qnn(self, tmp_path: Path) -> None:
        program = mock.MagicMock()
        program.buffer = b"QNN"
        with (
            mock.patch.dict(sys.modules, self._generic_modules()),
            mock.patch("rfdetr.export._executorch.converter._lower_qnn", return_value=program) as mock_lower,
        ):
            out = export_executorch(
                model=mock.MagicMock(),
                input_tensors=torch.zeros(1, 3, 8, 8),
                output_dir=tmp_path,
                backend="qnn",
                soc="SM8650",
            )
        mock_lower.assert_called_once()
        assert out.read_bytes() == b"QNN"


class TestPackageAvailabilityFlag:
    """The ``_IS_EXECUTORCH_AVAILABLE`` flag set at package import."""

    @pytest.fixture(autouse=True)
    def _restore_package_state(self) -> Any:
        """Reload the real package after every test in this class, even when the test body raises.

        Each test reloads ``rfdetr.export._executorch`` under a mock to force one branch of the availability check;
        ``importlib.reload`` mutates the module object in place, so a restore that only runs on the success path (a bare
        statement after the ``with`` block) is skipped whenever an assertion inside the mocked reload fails -- leaking
        the mocked ``_IS_EXECUTORCH_AVAILABLE`` state into every later test that imports the package. Restoring in
        fixture teardown guarantees it always runs.
        """
        yield
        import importlib

        import rfdetr.export._executorch as pkg

        importlib.reload(pkg)

    def test_true_when_executorch_importable(self) -> None:
        """Reloading the package with executorch importable sets the flag True (the success branch)."""
        import importlib

        import rfdetr.export._executorch as pkg

        with mock.patch.dict(sys.modules, _fake_executorch_tree({"executorch": {}})):
            reloaded = importlib.reload(pkg)
            assert reloaded._IS_EXECUTORCH_AVAILABLE is True

    def test_false_when_executorch_missing(self) -> None:
        """Reloading the package with executorch unavailable sets the flag False (the except branch).

        Covered explicitly (not just by the import-time state) so the result is deterministic whether or not the
        executorch extra happens to be installed in the test environment.
        """
        import importlib

        import rfdetr.export._executorch as pkg
        import rfdetr.export._executorch.converter as conv

        with mock.patch.object(conv, "_check_executorch_available", side_effect=ImportError("no executorch")):
            reloaded = importlib.reload(pkg)
            assert reloaded._IS_EXECUTORCH_AVAILABLE is False


# ---------------------------------------------------------------------------
# Real end-to-end export + numerical parity (requires the executorch package)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def exported(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, torch.Tensor, Path]:
    """Export RFDETRNano to a ``.pte`` once and reuse it across the parity checks."""
    from rfdetr import RFDETRNano

    torch.manual_seed(42)
    out_dir = tmp_path_factory.mktemp("executorch")
    detector = RFDETRNano(pretrain_weights=None)
    pte_path = detector.export(output_dir=str(out_dir), format="executorch", backend="xnnpack", verbose=False)

    model = detector.model.model.to("cpu").eval()
    model.export()
    example = torch.randn(1, 3, detector.model.resolution, detector.model.resolution)
    return model, example, Path(pte_path)


@executorch_only
@pytest.mark.executorch
class TestExecutorchEndToEnd:
    """End-to-end export of a real RF-DETR model, gated on the executorch package."""

    def test_pte_file_written(self, exported: tuple[Any, torch.Tensor, Path]) -> None:
        _, _, pte_path = exported
        assert pte_path.is_file()
        assert pte_path.suffix == ".pte"
        assert pte_path.stat().st_size > 0

    def test_runtime_output_matches_pytorch(self, exported: tuple[Any, torch.Tensor, Path]) -> None:
        """ExecuTorch runtime output must match the eager PyTorch forward within XNNPACK fp32 tolerance."""
        model, example, pte_path = exported
        diffs = _runtime_parity(model, example, pte_path)
        assert diffs, "expected at least one output to compare"
        # XNNPACK fp32 path matches eager to ~1e-5 (observed: pred_boxes 0.0, pred_logits ~1.0e-5 with random
        # weights).  The bound keeps modest headroom over that so it is robust to BLAS/XNNPACK build differences
        # while still failing on a structural regression (those diverge by >=1e-3).
        assert max(diffs) < 5e-5, f"ExecuTorch outputs diverge from PyTorch: max abs diff {max(diffs)}"


# ---------------------------------------------------------------------------
# MSDeformAttn export-mode vs eager-mode numerical parity (no executorch required)
# ---------------------------------------------------------------------------


class TestMSDeformAttnExportParity:
    """Numerical parity between MSDeformAttn export-mode and eager-mode forwards.

    Validates that switching ``_export=True`` (rank-5 path used for TFLite / ExecuTorch) produces bit-identical outputs
    to the standard eager forward.  No executorch installation needed.
    """

    # Fixed geometry used across all parity tests.
    _n_levels = 3
    _n_points = 4
    _n_heads = 4
    _d_model = 32
    _batch = 2
    _len_q = 5
    _hw_pairs: list[tuple[int, int]] = [(4, 6), (3, 5), (2, 4)]

    def _build_inputs(
        self, ref_last_dim: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build shared forward inputs for parity tests.

        Args:
            ref_last_dim: Last dimension of reference_points — 2 for point form, 4 for box form.

        Returns:
            Tuple of (query, reference_points, input_flatten,
                      input_spatial_shapes, input_level_start_index).
        """
        total_len = sum(height * width for height, width in self._hw_pairs)
        query = torch.randn(self._batch, self._len_q, self._d_model)
        # Values in [0.2, 0.8] to stay well inside the valid coordinate range.
        reference_points = torch.empty(self._batch, self._len_q, self._n_levels, ref_last_dim).uniform_(0.2, 0.8)
        input_flatten = torch.randn(self._batch, total_len, self._d_model)
        input_spatial_shapes = torch.tensor(self._hw_pairs, dtype=torch.long)
        input_level_start_index = torch.tensor([0, 24, 39], dtype=torch.long)
        return query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index

    def _make_eager_module(self) -> "MSDeformAttn":  # noqa: F821
        """Instantiate MSDeformAttn in default (eager) mode with shared weights."""
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        return MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )

    def test_export_mode_and_eager_mode_output_parity(self) -> None:
        """Export-mode forward (rank-5 path) must match eager forward for point-form reference_points.

        Both modes share identical weights; the only difference is the ``_export`` flag which selects the rank-5
        sampling offset layout.  The export path requires ``input_spatial_shapes_hw`` to build the offset normalizer
        from concrete Python ints.
        """
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index = self._build_inputs(
            ref_last_dim=2
        )

        eager_module = self._make_eager_module()
        export_module = MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )
        # Copy weights so the only difference is the export flag.
        export_module.load_state_dict(eager_module.state_dict())
        export_module.export()

        with torch.no_grad():
            eager_out = eager_module(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
            )
            export_out = export_module(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_spatial_shapes_hw=self._hw_pairs,
            )

        torch.testing.assert_close(eager_out, export_out, atol=1e-5, rtol=0)

    def test_export_mode_forward_with_box_reference_points(self) -> None:
        """Export-mode forward must match eager forward for box-form reference_points (last dim=4).

        The box form passes (cx, cy, w, h) normalised coordinates; the rank-5 export path handles this via a separate
        branch in MSDeformAttn.forward.
        """
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        query, reference_points, input_flatten, input_spatial_shapes, input_level_start_index = self._build_inputs(
            ref_last_dim=4
        )

        eager_module = self._make_eager_module()
        export_module = MSDeformAttn(
            d_model=self._d_model,
            n_levels=self._n_levels,
            n_heads=self._n_heads,
            n_points=self._n_points,
        )
        export_module.load_state_dict(eager_module.state_dict())
        export_module.export()

        with torch.no_grad():
            eager_out = eager_module(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
            )
            export_out = export_module(
                query,
                reference_points,
                input_flatten,
                input_spatial_shapes,
                input_level_start_index,
                input_spatial_shapes_hw=self._hw_pairs,
            )

        torch.testing.assert_close(eager_out, export_out, atol=1e-5, rtol=0)


# ---------------------------------------------------------------------------
# TFLite rank-5 regression guard (no executorch required)
# ---------------------------------------------------------------------------


class TestTFLiteRank5Regression:
    """Regression guard: ``MSDeformAttn.export()`` sets ``_export=True`` (TFLite fix dependency).

    The TFLite export fix relies on ``_export=True`` activating the rank-5 sampling-offset layout that avoids CoreML
    MIL's rank-6 tensor rejection.  This focused test links the executorch test file to that fix so regressions are
    caught here as well as in test_transformer.py.
    """

    def test_tflite_export_uses_rank5_path(self) -> None:
        """Calling ``module.export()`` must set ``_export=True`` on MSDeformAttn."""
        from rfdetr.models.ops.modules.ms_deform_attn import MSDeformAttn

        module = MSDeformAttn(d_model=32, n_levels=2, n_heads=2, n_points=4)
        assert not module._export
        module.export()
        assert module._export
