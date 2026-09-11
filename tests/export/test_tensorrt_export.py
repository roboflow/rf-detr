# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the in-process TensorRT engine builder (`build_engine`).

The unit tests monkeypatch the polygraphy entry points so they run without TensorRT, a GPU, or `polygraphy` installed.
The end-to-end class (``@pytest.mark.e2e_tensorrt``, GPU + ``rfdetr[tensorrt]``, opt-in) builds a real engine from an
exported RF-DETR ONNX and checks runtime parity — mirroring the CoreML and ExecuTorch export suites.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from rfdetr.export import _tensorrt as tensorrt_export
from rfdetr.export._tensorrt import _IS_FP16_CASTER_AVAILABLE, _IS_TENSORRT_AVAILABLE
from tests.export.conftest import (
    _structured_parity_input,
    eager_reference_tensors,
    max_abs_output_diffs,
)

tensorrt_only = pytest.mark.skipif(not _IS_TENSORRT_AVAILABLE, reason="tensorrt not installed")
fp16_caster_only = pytest.mark.skipif(not _IS_FP16_CASTER_AVAILABLE, reason="onnx/onnxconverter-common not installed")

if _IS_FP16_CASTER_AVAILABLE:
    import onnx
    from onnx import helper

# A class-level skipif does not cover a module-level doctest, so gate the one live helper doctest on
# the same packages its body needs. Without this it raises NameError wherever the caster is absent.
__doctest_requires__ = {"_float32_model_with_cast": ["onnx", "onnxconverter_common"]}

# A FP32 TensorRT engine still fuses/reorders kernels relative to eager PyTorch, so it diverges more
# than the XNNPACK CPU path (~1e-5). The bound tolerates kernel-level numerical differences while still
# failing on a structural regression (outputs collapse by >=1e-1). Recalibrate once real GPU numbers are
# observed in the tensorrt-parity CI job.
_TENSORRT_MAX_ABS_DIFF = 1e-2


def _patch_polygraphy_chain(monkeypatch: pytest.MonkeyPatch, capture: str = "config") -> dict:
    """Stub the polygraphy build chain and return a dict capturing either CreateConfig kwargs or build args.

    Examples:
        Cannot be called directly — it requires a live ``pytest.MonkeyPatch`` instance supplied by
        pytest's fixture machinery. See ``TestBuildEngineDryRun`` for real invocations.

        >>> callable(_patch_polygraphy_chain)  # doctest: +SKIP
        True
    """
    config_kwargs: dict = {}
    monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
    monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config")
    monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: "engine")
    monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
    if capture == "build":
        build_args: dict = {}

        def _engine_from_network(network, config):
            build_args["network"] = network
            build_args["config"] = config
            return "engine"

        monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
        return build_args
    return config_kwargs


def _fake_tensorrt(version: str, *, has_fp16_flag: bool) -> types.ModuleType:
    """Build a stand-in ``tensorrt`` module reporting *version* and optionally lacking ``BuilderFlag.FP16``.

    Lets the version-dependent branches in ``build_engine`` be exercised on a machine with no TensorRT
    at all, including the TensorRT 11 shape where the flag was removed from the API.

    Args:
        version: Value to expose as ``tensorrt.__version__``.
        has_fp16_flag: Whether ``BuilderFlag`` should carry an ``FP16`` member.

    Returns:
        A module object suitable for ``monkeypatch.setitem(sys.modules, "tensorrt", ...)``.

    Examples:
        >>> module = _fake_tensorrt("11.2.1.2", has_fp16_flag=False)
        >>> module.__version__
        '11.2.1.2'
        >>> hasattr(module.BuilderFlag, "FP16")
        False
        >>> hasattr(_fake_tensorrt("10.16.1.11", has_fp16_flag=True).BuilderFlag, "FP16")
        True
    """
    module = types.ModuleType("tensorrt")
    module.__version__ = version

    class BuilderFlag:
        INT8 = 0

    if has_fp16_flag:
        BuilderFlag.FP16 = 1
    module.BuilderFlag = BuilderFlag
    return module


def _unexpected_cast(onnx_path: str) -> str:
    """Stand in for ``_cast_onnx_to_fp16`` on paths that must never cast, failing loudly if called.

    Args:
        onnx_path: Path the caller tried to cast.

    Raises:
        AssertionError: Always.

    Examples:
        >>> _unexpected_cast("/tmp/model.onnx")
        Traceback (most recent call last):
        AssertionError: _cast_onnx_to_fp16 must not be called for /tmp/model.onnx
    """
    raise AssertionError(f"_cast_onnx_to_fp16 must not be called for {onnx_path}")


def _float32_model_with_cast() -> "onnx.ModelProto":
    """Build a tiny float32 model that starts with an explicit ``Cast(to=FLOAT)``, as RF-DETR exports do.

    Returns:
        A valid float32 ONNX model with one ``Cast`` and one ``Conv``.

    Examples:
        >>> model = _float32_model_with_cast()
        >>> [node.op_type for node in model.graph.node]
        ['Cast', 'Conv']
    """
    weight = np.zeros((2, 3, 3, 3), dtype=np.float32)
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["input"], ["casted"], to=onnx.TensorProto.FLOAT, name="leading_cast"),
            helper.make_node("Conv", ["casted", "weight"], ["output"], pads=[1, 1, 1, 1], name="conv"),
        ],
        "tiny",
        [helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 3, 8, 8])],
        [helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, [1, 2, 8, 8])],
        [helper.make_tensor("weight", onnx.TensorProto.FLOAT, weight.shape, weight.tobytes(), raw=True)],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model


class TestBuildEngineDryRun:
    """Dry-run derives the ``.trt`` path without touching the polygraphy build chain."""

    @pytest.mark.parametrize(
        ("onnx_path", "expected_engine"),
        [
            pytest.param("/output/rfdetr.onnx", "/output/rfdetr_fp16.trt", id="plain-path"),
            pytest.param("/path with spaces/model.onnx", "/path with spaces/model_fp16.trt", id="path-with-spaces"),
            pytest.param("/model;rm -rf /.onnx", "/model;rm -rf /.onnx_fp16.trt", id="shell-metachar"),
            pytest.param(
                "/data/my.onnx.backup/model.onnx",
                "/data/my.onnx.backup/model_fp16.trt",
                id="earlier-onnx-in-dir",
            ),
            pytest.param(
                "/output/model_v1.onnx.old.onnx",
                "/output/model_v1.onnx.old_fp16.trt",
                id="double-onnx-in-filename",
            ),
            pytest.param(
                "/output/model_without_extension",
                "/output/model_without_extension_fp16.trt",
                id="no-onnx-extension",
            ),
        ],
    )
    def test_derives_trt_path(self, onnx_path: str, expected_engine: str) -> None:
        """Only the final suffix is swapped to ``_fp16.trt``; earlier ``.onnx`` segments are never corrupted."""
        result = tensorrt_export.build_engine(onnx_path, dry_run=True)

        assert result == expected_engine

    def test_does_not_build(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Dry-run must return the engine path without invoking the polygraphy build chain."""
        called: list[str] = []
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda *a, **k: called.append("built"))

        result = tensorrt_export.build_engine("/tmp/model.onnx", dry_run=True)

        assert result == "/tmp/model_fp16.trt"
        assert not called, "dry_run must not invoke the polygraphy build chain"

    def test_output_name_overrides_and_suppresses_precision_suffix(self) -> None:
        """``output_name`` names the engine verbatim, in the ONNX's directory, with no ``_fp16``/``_fp32`` suffix."""
        result = tensorrt_export.build_engine("/output/rfdetr-medium.onnx", dry_run=True, output_name="my-engine")

        assert result == "/output/my-engine.trt"

    def test_output_name_preserves_windows_directory_separators(self) -> None:
        """A Windows-style ``onnx_path`` keeps its backslash directory prefix verbatim (no ``os.sep`` rewrite)."""
        result = tensorrt_export.build_engine(r"C:\out\m.onnx", dry_run=True, output_name="my-engine")

        assert result == r"C:\out\my-engine.trt"


class TestBuildEngineDependencyGuard:
    """A missing polygraphy/tensorrt install raises an actionable ImportError."""

    def test_missing_polygraphy_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing polygraphy/tensorrt install must raise an actionable ImportError."""
        monkeypatch.setattr(tensorrt_export, "engine_from_network", None)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            tensorrt_export.build_engine("/tmp/model.onnx")


class TestBuildEngineWiring:
    """``build_engine`` wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""

    @pytest.mark.parametrize("fp16", [pytest.param(True, id="fp16"), pytest.param(False, id="fp32")])
    def test_invokes_polygraphy_and_saves_trt(self, monkeypatch: pytest.MonkeyPatch, fp16: bool) -> None:
        """build_engine wires ONNX -> config -> engine -> save and returns the ``.trt`` path."""
        config_kwargs: dict = {}
        build_args: dict = {}
        saved: dict = {}

        # Pin a weakly typed TensorRT: this asserts the builder-flag wiring, and without the pin the
        # assertions would flip on a host that really has TensorRT >= 11 installed.
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
        monkeypatch.setattr(
            tensorrt_export, "CreateConfig", lambda **kwargs: config_kwargs.update(kwargs) or "config-sentinel"
        )

        def _engine_from_network(network, config):
            build_args["network"] = network
            build_args["config"] = config
            return "engine-sentinel"

        def _save_engine(engine, path):
            saved["engine"] = engine
            saved["path"] = path

        monkeypatch.setattr(tensorrt_export, "engine_from_network", _engine_from_network)
        monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)

        result = tensorrt_export.build_engine("/tmp/model.onnx", fp16=fp16)
        expected_path = f"/tmp/model_{'fp16' if fp16 else 'fp32'}.trt"

        assert result == expected_path
        assert config_kwargs == {"fp16": fp16}
        assert build_args == {"network": ("network", "/tmp/model.onnx"), "config": "config-sentinel"}
        assert saved == {"engine": "engine-sentinel", "path": expected_path}


@pytest.fixture
def fp16_cast_graph(tmp_path: Path) -> "onnx.GraphProto":
    """Graph of a tiny float32 model after a real round-trip through ``_cast_onnx_to_fp16``."""
    source = tmp_path / "tiny.onnx"
    onnx.save(_float32_model_with_cast(), source)
    return onnx.load(tensorrt_export._cast_onnx_to_fp16(str(source))).graph


class TestTensorRTMajor:
    """``_tensorrt_major`` reads the leading integer off a TensorRT version string."""

    @pytest.mark.parametrize(
        ("version", "expected"),
        [
            ("11.2.1.2", 11),
            ("10.16.1.11", 10),
            ("8.6.1", 8),
            ("12", 12),
            ("unknown", None),
            pytest.param("", None, id="empty"),
        ],
    )
    def test_parses_major(self, version: str, expected: int | None) -> None:
        """A numeric leading component is returned as an int; anything else yields None."""
        assert tensorrt_export._tensorrt_major(version) == expected


class TestBuildEngineWeaklyTyped:
    """TensorRT < 11 still exposes the FP16 builder flag: that path must stay exactly as it was."""

    @pytest.mark.parametrize("version", ["10.16.1.11", "9.0.0", "8.6.1"])
    def test_sets_the_builder_flag(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """Weak typing takes precision from the flag, so it must still be requested."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(version, has_fp16_flag=True))

        tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert config_kwargs == {"fp16": True}

    def test_builds_from_the_original_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No graph rewriting on a weakly typed build — the float32 ONNX is handed over untouched."""
        build_args = _patch_polygraphy_chain(monkeypatch, capture="build")
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert build_args["network"] == ("network", "/tmp/model.onnx")

    def test_engine_name_reports_fp16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Naming is unchanged from before the strong-typing branch existed."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))

        assert tensorrt_export.build_engine("/tmp/model.onnx", fp16=True) == "/tmp/model_fp16.trt"

    def test_fp32_request_is_unchanged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``fp16=False`` on a weakly typed build behaves as it always has."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=True))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert tensorrt_export.build_engine("/tmp/model.onnx", fp16=False) == "/tmp/model_fp32.trt"
        assert config_kwargs == {"fp16": False}


class TestBuildEngineLeanWheelFallback:
    """A *weakly typed* TensorRT lacking the FP16 flag is a lean wheel: fall back to FP32."""

    @pytest.mark.parametrize("version", ["10.16.1.11", "8.6.1", "unknown"])
    def test_downgrades_to_fp32(self, monkeypatch: pytest.MonkeyPatch, version: str) -> None:
        """Without a graph-level alternative on TensorRT < 11, an FP32 engine beats failing the export."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt(version, has_fp16_flag=False))

        result = tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert result == "/tmp/model_fp32.trt"
        assert config_kwargs == {"fp16": False}

    def test_does_not_cast_the_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The FP32 fallback must not invoke the fp16 graph caster."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("10.16.1.11", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert tensorrt_export.build_engine("/tmp/model.onnx", fp16=True) == "/tmp/model_fp32.trt"


class TestBuildEngineStrongTyping:
    """On TensorRT >= 11 the FP16 flag is gone by design; precision comes from the ONNX graph."""

    def test_builds_from_the_cast_graph(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The network must be loaded from the cast graph, not the original float32 one."""
        build_args = _patch_polygraphy_chain(monkeypatch, capture="build")
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: "/tmp/model.fp16.onnx")

        tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert build_args["network"] == ("network", "/tmp/model.fp16.onnx")

    def test_engine_name_reports_fp16(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The engine really is FP16, so the filename must say so rather than ``_fp32``."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: "/tmp/model.fp16.onnx")

        assert tensorrt_export.build_engine("/tmp/model.onnx", fp16=True) == "/tmp/model_fp16.trt"

    def test_builder_flag_is_not_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Polygraphy aborts if asked for a flag TensorRT 11 removed, so the config must request FP32."""
        config_kwargs = _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: "/tmp/model.fp16.onnx")

        tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert config_kwargs == {"fp16": False}

    def test_fp32_request_does_not_cast(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An explicit ``fp16=False`` must take the plain FP32 path with no graph rewriting."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", _unexpected_cast)

        assert tensorrt_export.build_engine("/tmp/model.onnx", fp16=False) == "/tmp/model_fp32.trt"

    def test_raises_when_caster_unavailable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without the caster there is no way to honour the request, so fail loudly."""
        _patch_polygraphy_chain(monkeypatch)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"tensorrt<11"):
            tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

    def test_never_returns_fp32_for_an_fp16_request(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Regression guard for #1453: a silent FP32 engine is reported as an FP16 latency."""
        build_calls: list[object] = []
        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(
            tensorrt_export, "engine_from_network", lambda network, config: build_calls.append(network) or "engine"
        )
        monkeypatch.setattr(tensorrt_export, "save_engine", lambda engine, path: None)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError):
            tensorrt_export.build_engine("/tmp/model.onnx", fp16=True)

        assert not build_calls, "an FP16 request must not fall through to an FP32 build"


class TestBuildEngineCastArtifactCleanup:
    """The cast graph is a build intermediate, so ``.trt`` stays the only file the export leaves."""

    def _build_with_cast(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, *, save_fails: bool) -> Path:
        """Run ``build_engine`` on the strongly typed path against a real cast file on disk.

        Args:
            monkeypatch: Fixture used to stub the polygraphy chain and the fake ``tensorrt``.
            tmp_path: Directory the stand-in cast graph is written to.
            save_fails: Whether ``save_engine`` should raise, simulating a failed build.

        Returns:
            Path the stand-in cast graph was written to, for an existence assertion.

        Examples:
            Needs live ``monkeypatch`` and ``tmp_path`` fixtures, so it cannot run standalone.

            >>> TestBuildEngineCastArtifactCleanup()._build_with_cast(mp, tmp, save_fails=False)
            ... # doctest: +SKIP
        """
        cast_path = tmp_path / "model.fp16-abcd1234.onnx"
        cast_path.write_bytes(b"cast-graph")

        monkeypatch.setattr(tensorrt_export, "network_from_onnx_path", lambda path: ("network", path))
        monkeypatch.setattr(tensorrt_export, "CreateConfig", lambda **kwargs: "config")
        monkeypatch.setattr(tensorrt_export, "engine_from_network", lambda network, config: "engine")

        def _save_engine(engine, path):
            if save_fails:
                raise RuntimeError("builder ran out of workspace")

        monkeypatch.setattr(tensorrt_export, "save_engine", _save_engine)
        monkeypatch.setitem(sys.modules, "tensorrt", _fake_tensorrt("11.2.1.2", has_fp16_flag=False))
        monkeypatch.setattr(tensorrt_export, "_cast_onnx_to_fp16", lambda path: str(cast_path))
        return cast_path

    def test_cast_graph_is_removed_after_a_successful_build(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Users are told the export writes a ``.trt``; a stray ``.fp16.onnx`` beside it is a surprise."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=False)

        tensorrt_export.build_engine(str(tmp_path / "model.onnx"), fp16=True)

        assert not cast_path.exists()

    def test_cast_graph_is_removed_after_a_failed_build(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """A half-finished build must not leave the intermediate behind either."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=True)

        with pytest.raises(RuntimeError):
            tensorrt_export.build_engine(str(tmp_path / "model.onnx"), fp16=True)

        assert not cast_path.exists()

    def test_missing_cast_graph_does_not_mask_the_build_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Cleanup of an already-absent file must not raise over the real failure."""
        cast_path = self._build_with_cast(monkeypatch, tmp_path, save_fails=True)
        cast_path.unlink()

        with pytest.raises(RuntimeError, match="workspace"):
            tensorrt_export.build_engine(str(tmp_path / "model.onnx"), fp16=True)


@fp16_caster_only
class TestCastOnnxToFp16:
    """Graph-level fp16 conversion: weights become FP16 while graph I/O stays FP32."""

    def test_weights_become_fp16(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """Every float initializer must end up FP16, or the engine is not really FP16."""
        dtypes = {initializer.data_type for initializer in fp16_cast_graph.initializer}

        assert dtypes == {onnx.TensorProto.FLOAT16}

    @pytest.mark.parametrize("collection", ["input", "output"])
    def test_graph_io_stays_fp32(self, fp16_cast_graph: onnx.GraphProto, collection: str) -> None:
        """Callers feed and read float32 on a weakly typed FP16 engine; keep that contract."""
        dtypes = {value.type.tensor_type.elem_type for value in getattr(fp16_cast_graph, collection)}

        assert dtypes == {onnx.TensorProto.FLOAT}

    def test_preexisting_float_casts_are_retargeted(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """A leftover ``Cast(to=FLOAT)`` feeding an FP16 consumer makes TensorRT reject the graph."""
        body = [node for node in fp16_cast_graph.node if not node.name.startswith("Cast_")]
        targets = [
            attribute.i
            for node in body
            if node.op_type == "Cast"
            for attribute in node.attribute
            if attribute.name == "to"
        ]

        assert onnx.TensorProto.FLOAT not in targets

    def test_model_is_valid(self, fp16_cast_graph: onnx.GraphProto) -> None:
        """The rewritten graph must still pass ONNX's own checker."""
        onnx.checker.check_model(helper.make_model(fp16_cast_graph, opset_imports=[helper.make_opsetid("", 17)]))

    def test_raises_without_caster(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The ImportError must name both remedies: install the extra, or pin an older TensorRT."""
        monkeypatch.setattr(tensorrt_export, "_IS_FP16_CASTER_AVAILABLE", False)

        with pytest.raises(ImportError, match=r"rfdetr\[tensorrt\]"):
            tensorrt_export._cast_onnx_to_fp16("/tmp/model.onnx")

    def test_does_not_clobber_a_same_named_file(self, tmp_path: Path) -> None:
        """``build_engine`` deletes what this returns, so it must never claim a file it did not create."""
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)
        squatter = tmp_path / "tiny.fp16.onnx"
        squatter.write_bytes(b"not ours")

        tensorrt_export._cast_onnx_to_fp16(str(source))

        assert squatter.read_bytes() == b"not ours"

    def test_concurrent_casts_get_separate_files(self, tmp_path: Path) -> None:
        """Two builds from one source model must not hand each other's graph to the parser."""
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)

        first = tensorrt_export._cast_onnx_to_fp16(str(source))
        second = tensorrt_export._cast_onnx_to_fp16(str(source))

        assert first != second

    def test_cast_lands_beside_the_source_model(self, tmp_path: Path) -> None:
        """The intermediate rivals the model in size, and /tmp is often a size-capped tmpfs."""
        source = tmp_path / "tiny.onnx"
        onnx.save(_float32_model_with_cast(), source)

        cast_path = tensorrt_export._cast_onnx_to_fp16(str(source))

        assert Path(cast_path).parent == tmp_path


@tensorrt_only
@pytest.mark.gpu
@pytest.mark.e2e_tensorrt
class TestTensorRTEndToEnd:
    """Real ONNX -> TensorRT engine build + runtime parity on GPU (requires ``rfdetr[tensorrt]`` and CUDA)."""

    @pytest.fixture(scope="class")
    def trt_engine(self, tmp_path_factory: pytest.TempPathFactory) -> tuple[torch.nn.Module, torch.Tensor, Path]:
        """Export RFDETRNano to ONNX, build a FP32 ``.trt`` engine, and reuse it across the parity checks."""
        from rfdetr import RFDETRNano
        from rfdetr.export._tensorrt import build_engine

        torch.manual_seed(42)
        out_dir = tmp_path_factory.mktemp("tensorrt")
        detector = RFDETRNano(pretrain_weights=None)
        onnx_path = detector.export(output_dir=str(out_dir), format="onnx", verbose=False)
        engine_path = build_engine(str(onnx_path), fp16=False, verbose=False)

        model = detector.model.model.to("cpu").eval()
        model.export()
        resolution = int(detector.model.resolution)
        example = _structured_parity_input(1, 3, resolution, resolution)
        return model, example, Path(engine_path)

    def test_engine_file_written(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """build_engine must produce a non-empty ``.trt`` engine from the exported ONNX."""
        _, _, engine_path = trt_engine
        assert engine_path.is_file()
        assert engine_path.suffix == ".trt"
        assert engine_path.stat().st_size > 0

    def test_runtime_output_matches_pytorch(self, trt_engine: tuple[torch.nn.Module, torch.Tensor, Path]) -> None:
        """The TensorRT engine's outputs (dets, labels) must match eager PyTorch within FP32 tolerance."""
        import numpy as np
        from polygraphy.backend.common import BytesFromPath
        from polygraphy.backend.trt import EngineFromBytes, TrtRunner

        model, example, engine_path = trt_engine
        eager_tensors = eager_reference_tensors(model, example)

        feed = {"input": np.ascontiguousarray(example.detach().cpu().numpy())}
        load_engine = EngineFromBytes(BytesFromPath(str(engine_path)))
        with TrtRunner(load_engine) as runner:
            outputs = runner.infer(feed_dict=feed)
        output_names = ["dets", "labels"]
        trt_tensors = [torch.from_numpy(np.asarray(outputs[name], dtype=np.float32)) for name in output_names]

        diffs = max_abs_output_diffs(eager_tensors, trt_tensors, check_shape=True, names=output_names)
        assert max(diffs) < _TENSORRT_MAX_ABS_DIFF, (
            f"TensorRT outputs diverge from PyTorch: max abs diff {max(diffs)} "
            f"(dets={diffs[0]}, labels={diffs[1]}, bound={_TENSORRT_MAX_ABS_DIFF})"
        )
