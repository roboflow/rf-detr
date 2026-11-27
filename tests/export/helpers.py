# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared doubles for the ``RFDETR.export()`` facade test modules.

The ONNX, TensorRT-option, facade-validation and metadata test modules in ``tests/export/`` drive
``RFDETR.export()`` against a fake model instead of a real network. These builders create that fake and the stand-ins
for its example input, so every module exercises the same prediction contract::

    from tests.export.helpers import _make_export_model, _make_tensorrt_export_model

The facade modules also share ``_disable_metadata_publication``, which each calls from its own autouse fixture so the
stubbed artifacts are never published; the metadata module keeps the real writer.
"""

import types
from unittest.mock import MagicMock

import pytest
import torch


def _disable_metadata_publication(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the exporter's metadata writer with a no-op until ``monkeypatch`` is undone.

    Facade tests stub the conversion, so their artifact paths hold no file bytes to hash or annotate. Each facade test
    module calls this from its own autouse fixture; ``tests/export/core/test_metadata.py`` keeps the real writer.

    Args:
        monkeypatch: Patcher whose undo restores the real writer.

    Examples:
        >>> import rfdetr.export.base as export_base
        >>> with pytest.MonkeyPatch.context() as patcher:
        ...     _disable_metadata_publication(patcher)
        ...     export_base.write_metadata("missing.onnx", None) is None
        True
        >>> export_base.write_metadata.__name__
        'write_metadata'
    """
    monkeypatch.setattr("rfdetr.export.base.write_metadata", lambda *_args, **_kwargs: None)


class _DummyCoreModel:
    """Minimal torch.nn.Module stub shared across export tests.

    Avoids real forward passes; returns synthetic detection (and optionally segmentation) outputs matching the shapes
    expected by RFDETR.export().

    Examples:
        >>> sorted(_DummyCoreModel(segmentation_head=True)())
        ['pred_boxes', 'pred_logits', 'pred_masks']
    """

    def __init__(self, *, segmentation_head: bool = False) -> None:
        self._segmentation_head = segmentation_head

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def cpu(self):
        return self

    def modules(self):
        return iter(())

    def __call__(self, *_args, **_kwargs):
        out = {"pred_boxes": torch.zeros(1, 1, 4), "pred_logits": torch.zeros(1, 1, 2)}
        if self._segmentation_head:
            out["pred_masks"] = torch.zeros(1, 1, 2, 2)
        return out


def _make_export_model(
    *,
    core_model: _DummyCoreModel | None = None,
    segmentation_head: bool = False,
    device: str = "cpu",
    resolution: int = 14,
    patch_size: int = 14,
    num_windows: int = 1,
    size: str | None = None,
) -> types.SimpleNamespace:
    """Build an RFDETR export double with a valid detection or segmentation prediction contract.

    Examples:
        >>> model = _make_export_model(segmentation_head=True)
        >>> model.class_names, model.model.postprocess.num_select
        (['object'], 1)
    """
    return types.SimpleNamespace(
        model=types.SimpleNamespace(
            model=core_model or _DummyCoreModel(segmentation_head=segmentation_head),
            device=device,
            resolution=resolution,
            postprocess=types.SimpleNamespace(
                num_select=1,
                trace_alpha=0.2,
                upsample_masks_to_image_size=True,
            ),
        ),
        model_config=types.SimpleNamespace(
            segmentation_head=segmentation_head,
            use_grouppose_keypoints=False,
            num_channels=3,
            num_classes=1,
            num_keypoints_per_class=[],
            patch_size=patch_size,
            num_windows=num_windows,
        ),
        class_names=["object"],
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        size=size,
    )


class _DeviceTrackingCoreModel(_DummyCoreModel):
    """`_DummyCoreModel` variant that records every `.to()` call's target device.

    Examples:
        >>> core = _DeviceTrackingCoreModel()
        >>> core.to("cuda").to("cpu") is core, core.to_calls
        (True, ['cuda', 'cpu'])
    """

    def __init__(self) -> None:
        super().__init__()
        self.to_calls: list[str] = []

    def to(self, device, *_args, **_kwargs):
        self.to_calls.append(device)
        return self


def _make_tensorrt_export_model(*, device: str = "cpu") -> types.SimpleNamespace:
    """Build the minimal `self`-like fake `RFDETR.export()` needs for the format="tensorrt" branch.

    Examples:
        >>> m = _make_tensorrt_export_model()
        >>> m.model.device
        'cpu'
        >>> m = _make_tensorrt_export_model(device="cuda")
        >>> m.model.device
        'cuda'
    """
    return _make_export_model(core_model=_DeviceTrackingCoreModel(), device=device)


def _make_mock_infer_tensor() -> MagicMock:
    """Mock tensor standing in for `make_infer_image()`'s return value — avoids real-device `.to()`/`.cpu()`.

    Examples:
        >>> t = _make_mock_infer_tensor()
        >>> t.to("cpu") is t
        True
        >>> t.cpu() is t
        True
    """
    mock_tensor = MagicMock()
    mock_tensor.to.return_value = mock_tensor
    mock_tensor.cpu.return_value = mock_tensor
    return mock_tensor
