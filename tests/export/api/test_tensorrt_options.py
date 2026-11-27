# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the TensorRT options of :meth:`rfdetr.detr.RFDETR.export`.

Use cases covered:
- ``format="tensorrt"`` builds the engine from the intermediate ONNX file and forwards every TensorRT option.
- A TensorRT-only option used with another format, or without its prerequisite, must warn instead of silently no-op.
- An invalid TensorRT option or a missing ``tensorrt`` package must be refused before any model work.
- Export must restore the model's device even when a converter raises, and move it under the device-move lock.
- A custom TensorRT engine name keeps the backbone marker of a backbone-only export.
"""

import sys
import types
import warnings
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
from pytest_mock import MockerFixture

from rfdetr import detr as _detr_module
from rfdetr.export._tensorrt.exporter import TensorRTExporter, _BuiltEngine
from tests.export.helpers import (
    _DeviceTrackingCoreModel,
    _disable_metadata_publication,
    _make_export_model,
    _make_mock_infer_tensor,
    _make_tensorrt_export_model,
)


@pytest.fixture(autouse=True)
def _skip_metadata_publication_for_export_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep facade tests focused on conversion when their fake artifacts have no file bytes.

    The metadata round-trip tests use the real writer in ``tests/export/core/test_metadata.py``.
    This fixture does not replace ``metadata_from_model``: fake RFDETR models still need a valid prediction contract.

    Examples:
        Requires pytest's monkeypatch lifecycle, so this example cannot run standalone:
        >>> _skip_metadata_publication_for_export_stubs(None)  # doctest: +SKIP
    """
    _disable_metadata_publication(monkeypatch)


@pytest.fixture
def onnx_output(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> str:
    """Stub the example input, ONNX conversion and model copy of ``RFDETR.export()``; return the stubbed ONNX path.

    Examples:
        Requires pytest's monkeypatch and tmp_path lifecycles, so this example cannot run standalone:
        >>> onnx_output(None, None)  # doctest: +SKIP
    """
    path = str(tmp_path / "inference_model.onnx")
    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: path)
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    return path


@pytest.fixture
def tensorrt_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Report TensorRT and Polygraphy as installed so a stubbed engine build is not refused on a CPU host.

    Examples:
        Requires pytest's monkeypatch lifecycle, so this example cannot run standalone:
        >>> tensorrt_installed(None)  # doctest: +SKIP
    """
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE", True)


class _LockTrackingCoreModel(_DeviceTrackingCoreModel):
    """`_DeviceTrackingCoreModel` variant that also records whether the module's move lock was held per `.to()`."""

    def __init__(self) -> None:
        super().__init__()
        self.lock_held: list[bool] = []

    def to(self, device, *args, **kwargs):
        """Record the device-move lock state, then track the target device as the base class does."""
        self.lock_held.append(_detr_module._device_move_lock(self).locked())
        return super().to(device, *args, **kwargs)


def _built_engine(
    tmp_path: Path, *, name: str = "inference_model.trt", fp16: bool = True, engine_facts: dict | None = None
) -> _BuiltEngine:
    """Build the ``_BuiltEngine`` a stubbed TensorRT ``_build`` returns, rooted in ``tmp_path``.

    Examples:
        >>> from pathlib import Path
        >>> _built_engine(Path("out")).path.endswith("inference_model.trt")
        True
        >>> _built_engine(Path("out"), name="other.trt", fp16=False).fp16
        False
    """
    return _BuiltEngine(path=str(tmp_path / name), fp16=fp16, engine_facts=engine_facts)


@pytest.mark.parametrize("export_format", ["tensorrt", "trt"])
@pytest.mark.usefixtures("tensorrt_installed")
def test_rfdetr_export_tensorrt_builds_the_engine_from_the_onnx_path(
    mocker: MockerFixture, tmp_path: Path, onnx_output: str, export_format: str
) -> None:
    """`RFDETR.export(format="tensorrt")` — and its `"trt"` alias — must build the engine once from the ONNX path.

    Covers the public-API wrapper directly (as opposed to the CLI `main()` path, which
    `TestCliExportMain.test_tensorrt_flag_calls_build_engine` already covers).
    """
    model = _make_tensorrt_export_model()
    mock_build = mocker.patch.object(TensorRTExporter, "_build", return_value=_built_engine(tmp_path))

    result = _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format=export_format, shape=(14, 14))

    mock_build.assert_called_once()
    assert mock_build.call_args.args == (onnx_output,), (
        f"ONNX path must be passed positionally, got {mock_build.call_args.args!r}"
    )
    assert str(result) == str(tmp_path / "inference_model.trt")


@pytest.mark.usefixtures("onnx_output")
def test_rfdetr_export_warns_when_max_batch_size_used_without_tensorrt(tmp_path: Path) -> None:
    """`max_batch_size` outside `format="tensorrt"` is ignored and must warn rather than silently no-op.

    `format="onnx"` with `dynamic_batch=True` accepts a dynamic batch axis but has no optimization-profile concept for
    `max_batch_size` to tune, so passing it there previously vanished with no signal at all.
    """
    model = _make_export_model()

    with pytest.warns(UserWarning) as record:
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="onnx", dynamic_batch=True, max_batch_size=0, shape=(14, 14)
        )

    assert any("`max_batch_size`" in str(w.message) and "ignored" in str(w.message) for w in record)


@pytest.mark.usefixtures("onnx_output", "tensorrt_installed")
def test_rfdetr_export_warns_when_max_batch_size_used_without_dynamic_batch(
    mocker: MockerFixture, tmp_path: Path
) -> None:
    """`max_batch_size` on a static (`dynamic_batch=False`) TensorRT export is ignored and must warn.

    A static export builds the engine with the exact call it always made -- there is no optimization profile for
    `max_batch_size` to bound -- so even an invalid value passed there is inert and should not fail validation.
    """
    model = _make_tensorrt_export_model()
    mocker.patch.object(TensorRTExporter, "_build", return_value=_built_engine(tmp_path))

    with pytest.warns(UserWarning) as record:
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", dynamic_batch=False, max_batch_size=0, shape=(14, 14)
        )

    assert any("`max_batch_size`" in str(w.message) and "ignored" in str(w.message) for w in record)


@pytest.mark.usefixtures("onnx_output")
def test_rfdetr_export_warns_when_trt_timing_cache_used_for_another_format(tmp_path: Path) -> None:
    """`trt_timing_cache` outside `format="tensorrt"` is ignored and must warn rather than silently no-op.

    Only TensorRT times kernels, so on `format="onnx"` there is no cache to load or save and the file is never touched.
    """
    model = _make_tensorrt_export_model()
    cache = tmp_path / "engine.cache"

    with pytest.warns(UserWarning) as record:
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="onnx", trt_timing_cache=str(cache), shape=(14, 14)
        )

    assert any("`trt_timing_cache`" in str(w.message) and "ignored" in str(w.message) for w in record)
    assert not cache.exists()


@pytest.mark.usefixtures("onnx_output", "tensorrt_installed")
def test_rfdetr_export_tensorrt_forwards_trt_timing_cache(mocker: MockerFixture, tmp_path: Path) -> None:
    """`RFDETR.export(format="tensorrt", trt_timing_cache=...)` puts the path on the exporter's config, unwarned."""
    model = _make_tensorrt_export_model()
    cache = tmp_path / "engine.cache"
    build_engine = mocker.patch.object(TensorRTExporter, "_build", autospec=True, return_value=_built_engine(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", trt_timing_cache=cache, shape=(14, 14)
        )

    assert build_engine.call_args.args[0].config.timing_cache == cache


def test_rfdetr_export_tensorrt_without_tensorrt_fails_before_any_model_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Without ``tensorrt`` installed, ``format="tensorrt"`` reports it before any work on the model starts.

    A host with polygraphy but not ``tensorrt`` used to write the whole ONNX file first and only then fail inside
    polygraphy. Preparing the export graph is the more expensive of the two stages that used to precede the refusal — a
    full forward pass through the model — so both are guarded here.
    """
    model = _make_tensorrt_export_model()

    monkeypatch.setattr(
        "rfdetr.export.prepare.prepare_export_graph",
        lambda *_a, **_kw: pytest.fail("the export graph was prepared before the missing TensorRT was reported"),
    )
    monkeypatch.setattr(
        "rfdetr.export._onnx.exporter.OnnxExporter._convert",
        lambda *_a, **_kw: pytest.fail("the ONNX export ran before the missing TensorRT was reported"),
    )
    monkeypatch.setattr("rfdetr.detr.deepcopy", lambda x: x)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", False)

    with pytest.raises(ImportError) as exc_info:
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", shape=(14, 14))

    assert "rfdetr[tensorrt]" in str(exc_info.value)


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("trt_hardware_compatibility", "ampere_plus"),
        ("trt_version_compatible", True),
        ("trt_version_compatible", "yes"),
        ("trt_metadata", True),
    ],
)
@pytest.mark.usefixtures("onnx_output")
def test_rfdetr_export_warns_when_a_tensorrt_option_is_used_without_tensorrt(
    tmp_path: Path, keyword: str, value: object
) -> None:
    """A TensorRT-only option outside `format="tensorrt"` is ignored and must warn rather than silently no-op."""
    model = _make_tensorrt_export_model()

    with pytest.warns(UserWarning) as record:
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="onnx", shape=(14, 14), **{keyword: value})

    assert any(f"`{keyword}`" in str(w.message) and "ignored" in str(w.message) for w in record)


@pytest.mark.usefixtures("onnx_output")
def test_rfdetr_export_does_not_warn_for_the_default_tensorrt_settings(tmp_path: Path) -> None:
    """Passing the defaults explicitly (`None` and `False`) is not a request, so it never warns on another format.

    Only a request that would have had an effect is worth a warning; for `trt_metadata=False` too, the explicit default
    is not ignored in any way the caller needs to hear about.
    """
    model = _make_tensorrt_export_model()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _detr_module.RFDETR.export(
            model,
            output_dir=str(tmp_path),
            format="onnx",
            shape=(14, 14),
            trt_hardware_compatibility=None,
            trt_version_compatible=False,
            trt_metadata=False,
        )


@pytest.mark.usefixtures("onnx_output", "tensorrt_installed")
def test_rfdetr_export_tensorrt_forwards_the_portability_options(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """`RFDETR.export(format="tensorrt", trt_...=...)` puts both settings on the exporter's configuration, unwarned."""
    model = _make_tensorrt_export_model()

    monkeypatch.setattr("rfdetr.export._tensorrt.exporter.ctypes.CDLL", lambda name: None)
    # With CDLL faked, the real lean package would import without its libraries and stay broken in sys.modules.
    monkeypatch.setitem(sys.modules, "tensorrt_lean_libs", types.ModuleType("tensorrt_lean_libs"))
    # The lean-runtime check reads the version of the installed TensorRT, which CPU CI does not have.
    fake_tensorrt = types.ModuleType("tensorrt")
    fake_tensorrt.__version__ = "11.3.0.99"
    fake_tensorrt.HardwareCompatibilityLevel = types.SimpleNamespace(SAME_COMPUTE_CAPABILITY=object())
    monkeypatch.setitem(sys.modules, "tensorrt", fake_tensorrt)

    build_engine = mocker.patch.object(TensorRTExporter, "_build", autospec=True, return_value=_built_engine(tmp_path))

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _detr_module.RFDETR.export(
            model,
            output_dir=str(tmp_path),
            format="tensorrt",
            shape=(14, 14),
            trt_hardware_compatibility="same_compute_capability",
            trt_version_compatible=True,
        )

    config = build_engine.call_args.args[0].config
    assert (config.hardware_compatibility, config.version_compatible) == ("same_compute_capability", True)


@pytest.mark.parametrize("fp16", [True, False])
@pytest.mark.usefixtures("onnx_output", "tensorrt_installed")
def test_rfdetr_export_tensorrt_forwards_fp16(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fp16: bool) -> None:
    """`RFDETR.export(format="tensorrt", fp16=...)` must reach the exporter that builds the engine.

    Passing ``fp16=False`` lets callers build an FP32 engine on TensorRT builds that lack the FP16 builder flag, so the
    flag has to survive the trip through the configuration rather than silently reverting to the default.
    """
    model = _make_tensorrt_export_model()
    captured: dict = {}

    def _fake_build(self, *_args, **_kwargs) -> _BuiltEngine:
        captured["fp16"] = self.config.fp16
        return _built_engine(tmp_path, fp16=self.config.fp16)

    monkeypatch.setattr("rfdetr.export._tensorrt.exporter.TensorRTExporter._build", _fake_build)

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", fp16=fp16, shape=(14, 14))

    assert captured["fp16"] == fp16


def test_rfdetr_export_tensorrt_failure_restores_device(
    mocker: MockerFixture, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """An engine build failure must still restore the live model to its original device.

    Regression test for the try/finally around the CPU-move .. TensorRT-conversion span in `RFDETR.export()` — a build
    failure previously (pre-merge) could strand the model on CPU.
    """
    # Not "cpu" (the staging move), or the assertion would pass even without the `finally` restore; and a valid
    # `torch.device()` string, or `prepare_export_graph()` would raise before the mocked `_build` is ever reached.
    original_device = "meta"
    model = _make_tensorrt_export_model(device=original_device)
    onnx_output = str(tmp_path / "inference_model.onnx")
    mock_build = mocker.patch.object(TensorRTExporter, "_build", side_effect=RuntimeError("engine build failed"))

    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr("rfdetr.export._onnx.exporter.OnnxExporter._convert", lambda *_a, **_kw: onnx_output)
    # No deepcopy patch: the exported copy must stay distinct from `self.model.model`, whose `.to()` calls are tracked.
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    monkeypatch.setattr("rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE", True)

    with pytest.raises(RuntimeError, match="engine build failed"):
        _detr_module.RFDETR.export(model, output_dir=str(tmp_path), format="tensorrt", shape=(14, 14))

    # The error must come from `_build`, not an earlier failure, or the restore assertion passes for the wrong reason.
    mock_build.assert_called_once()

    core_model = model.model.model
    assert core_model.to_calls == ["cpu", original_device], (
        f"expected exactly one staging move to 'cpu' then one restore to {original_device!r} even though "
        f"the build raised, got device move sequence {core_model.to_calls!r}"
    )


def test_rfdetr_export_moves_the_live_model_under_the_device_move_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Both live-model moves in `export()` — the CPU staging move and the `finally` restore — must hold the lock.

    `export()` moves the very module `predict()` runs on, and `nn.Module.to()` rewrites parameter storage in place. A
    thread reaching its first `predict()` mid-export would otherwise rewrite the same tensors concurrently — the race
    that left a cold segmentation model with silently corrupted weights. The deepcopy taken between the two moves is a
    private object no other thread can see, so its own `.to()` is deliberately not covered here.
    """
    core_model = _LockTrackingCoreModel()
    original_device = "meta"
    model = _make_export_model(resolution=14, patch_size=14, num_windows=1, device=original_device)
    model.model.model = core_model

    # Mock infer tensor (as in the TensorRT tests above): a real one cannot be copied off the "meta" device.
    monkeypatch.setattr("rfdetr.export.prepare.make_infer_image", lambda *_a, **_kw: _make_mock_infer_tensor())
    monkeypatch.setattr(
        "rfdetr.export._onnx.exporter.OnnxExporter._convert",
        lambda *_a, **_kw: str(tmp_path / "inference_model.onnx"),
    )
    # No `deepcopy` patch: the export copy must stay a distinct object so only the live model's moves are recorded.

    _detr_module.RFDETR.export(model, output_dir=str(tmp_path), shape=(14, 14))

    assert core_model.to_calls == ["cpu", original_device], (
        f"precondition: expected the staging move and the restore, got {core_model.to_calls!r}"
    )
    assert core_model.lock_held == [True, True], (
        f"both live-model moves must run under the module's device-move lock, got lock states {core_model.lock_held!r}"
    )


@pytest.mark.parametrize("value", ["yes", 1, None])
def test_rfdetr_export_tensorrt_rejects_a_non_bool_trt_metadata_before_any_model_work(
    mocker: MockerFixture, tmp_path: Path, value: object
) -> None:
    """`RFDETR.export(format="tensorrt", trt_metadata=<not a bool>)` must raise instead of writing a file by accident.

    A truthy non-`bool` such as `"yes"` or `1` would switch the description on silently. The exporter is constructed and
    validated before the model is moved, copied or traced, so none of that work may have happened when it raises.
    """
    model = _make_tensorrt_export_model()
    prepare = mocker.patch("rfdetr.export.prepare.prepare_export_graph")

    with pytest.raises(ValueError, match="trt_metadata"):
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", trt_metadata=value, shape=(14, 14)
        )

    prepare.assert_not_called()
    assert model.model.model.to_calls == []


@pytest.mark.usefixtures("onnx_output", "tensorrt_installed")
def test_rfdetr_export_tensorrt_forwards_trt_metadata(mocker: MockerFixture, tmp_path: Path) -> None:
    """`RFDETR.export(format="tensorrt", trt_metadata=True)` sets it on the exporter's configuration, unwarned."""
    model = _make_tensorrt_export_model()
    mocker.patch.object(TensorRTExporter, "_write_metadata", return_value=None)
    engine = _built_engine(tmp_path, engine_facts={"size": 0, "sha256": ""})
    build = mocker.patch.object(TensorRTExporter, "_build", autospec=True, return_value=engine)

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", trt_metadata=True, shape=(14, 14)
        )

    assert build.call_args.args[0].config.metadata is True


@pytest.mark.parametrize(
    "settings, message",
    [
        pytest.param({}, "max_batch_size", id="missing-max-batch-size"),
        pytest.param(
            {"batch_size": 8, "max_batch_size": 4}, "1 <= batch_size <= max_batch_size", id="batch-size-over-max"
        ),
    ],
)
def test_rfdetr_export_tensorrt_dynamic_batch_rejects_invalid_bounds(
    tmp_path: Path, settings: dict[str, int], message: str
) -> None:
    """`RFDETR.export(format="tensorrt", dynamic_batch=True)` needs `max_batch_size`, and `batch_size` may not exceed
    it.

    Covers the public facade users call (as opposed to `TestDynamicBatchConfig` in
    `tests/export/tensorrt/test_build_engine.py`, which only exercises `TensorRTConfig`). The exporter is constructed
    and validated before any ONNX conversion starts, and `batch_size > max_batch_size` cannot form a valid optimization
    profile, so no conversion stubbing is needed.
    """
    model = _make_tensorrt_export_model()

    with pytest.raises(ValueError) as exc_info:
        _detr_module.RFDETR.export(
            model, output_dir=str(tmp_path), format="tensorrt", dynamic_batch=True, shape=(14, 14), **settings
        )

    assert message in str(exc_info.value)


@pytest.mark.parametrize("backbone_only", [False, True])
def test_public_export_preserves_backbone_marker_in_custom_tensorrt_name(
    mocker: MockerFixture, tmp_path: Path, backbone_only: bool
) -> None:
    """TensorRT receives the ONNX backbone marker when the caller supplies a custom name."""
    obj = _detr_module.RFDETR.__new__(_detr_module.RFDETR)
    obj.model = MagicMock()
    obj.model.resolution = 32
    obj.model.device = "cpu"
    obj.model.class_names = ["object"]
    obj.model.args = types.SimpleNamespace(num_classes=1, num_keypoints_per_class=[])
    obj.model.postprocess.num_select = 1
    obj.model.postprocess.trace_alpha = 0.2
    obj.model.postprocess.upsample_masks_to_image_size = True
    obj.model.model.to.return_value = obj.model.model
    obj.means = [0.485, 0.456, 0.406]
    obj.stds = [0.229, 0.224, 0.225]
    backbone = torch.nn.Identity()
    backbone.export = MagicMock()
    backbone.forward_export = MagicMock(return_value=([torch.zeros(1, 4, 2, 2)], None, None))
    backbone.cross_attn_projector = None
    obj.model.model.backbone = torch.nn.ModuleList([backbone])
    obj.model_config = types.SimpleNamespace(
        segmentation_head=False,
        use_grouppose_keypoints=False,
        patch_size=16,
        num_windows=1,
        num_channels=3,
        num_classes=1,
        num_keypoints_per_class=[],
        projector_scale=["P4"],
    )
    stem = "custom-backbone" if backbone_only else "custom"
    onnx_path = tmp_path / f"{stem}.onnx"
    mocker.patch("rfdetr.export.prepare.make_infer_image", return_value=torch.zeros(1, 3, 32, 32))
    mocker.patch("rfdetr.export._onnx.exporter.OnnxExporter._convert", return_value=onnx_path)
    mocker.patch("rfdetr.export._tensorrt.exporter._IS_TENSORRT_AVAILABLE", True)
    mocker.patch("rfdetr.export._tensorrt.exporter._IS_POLYGRAPHY_AVAILABLE", True)
    build = mocker.patch.object(TensorRTExporter, "_build", return_value=_built_engine(tmp_path, name=f"{stem}.trt"))

    obj.export(format="tensorrt", backbone_only=backbone_only, output_name="custom", output_dir=str(tmp_path))

    assert build.call_args.kwargs["output_name"] == stem
