# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Native sources for the public inference facade."""

from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from supervision import Detections

from rfdetr.detr import RFDETR
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.inference import RFDETRInference
from rfdetr.variants import RFDETRNano


@pytest.fixture
def nano_model() -> RFDETRNano:
    """Build a small CPU model with random weights.

    Examples:
        The pytest runner must create this fixture before a test can use it.

        >>> nano_model()  # doctest: +SKIP
    """
    return RFDETRNano(
        pretrain_weights=None,
        device="cpu",
        resolution=64,
        num_queries=4,
        num_select=4,
        num_classes=2,
    )


class TestLiveModelDevicePolicy:
    """Borrowed models always keep their own device policy."""

    @pytest.mark.parametrize("device", ["cpu", "not_a_device", pytest.param(torch.device("cpu"), id="torch-device")])
    def test_explicit_device_is_rejected(self, nano_model: RFDETRNano, device: str | torch.device) -> None:
        """Even a matching device must be configured on the native source itself."""
        with pytest.raises(ValueError, match="live model.*device"):
            RFDETRInference(nano_model, device=device)

    @pytest.mark.parametrize("source_device", ["cuda", "cuda:0"])
    def test_explicit_device_is_rejected_before_and_after_device_resolution(
        self, nano_model: RFDETRNano, source_device: str
    ) -> None:
        """An explicit borrowed-model request has the same outcome before the deferred move."""
        nano_model.model.device = torch.device(source_device)
        with pytest.raises(ValueError, match="live model.*device"):
            RFDETRInference(nano_model, device="cuda:0")


class TestLiveModelSource:
    """Exercise live model behavior through the inference API."""

    def test_live_model_prediction_uses_current_model_and_labels(
        self,
        nano_model: RFDETRNano,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Facade prediction and labels follow changes to the borrowed native model."""
        native_model = nano_model
        native_model.model.class_names = ["cat", "dog"]
        model = native_model.model.model
        assert model is not None
        facade = RFDETRInference(native_model)
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        fail_on_placement = Mock(side_effect=AssertionError("Reading facade properties must not place the model."))

        with monkeypatch.context() as property_patch:
            property_patch.setattr("rfdetr.detr._move_model_context_to_device", fail_on_placement)
            assert facade.class_names == ["cat", "dog"]
            assert facade.runtime_info == {"backend": "pytorch", "device": "cpu"}

        expected = native_model.predict(image, threshold=0.0)
        actual = facade.predict(image, threshold=0.0)
        assert isinstance(expected, Detections)
        assert isinstance(actual, Detections)
        assert expected.confidence is not None
        assert actual.confidence is not None
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=0, rtol=0)
        np.testing.assert_allclose(actual.confidence, expected.confidence, atol=0, rtol=0)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)
        np.testing.assert_array_equal(actual.metadata["source_image"], image)

        native_model.model.class_names = ["owl", "fox"]
        assert facade.class_names == ["owl", "fox"]
        with torch.no_grad():
            model.class_embed.bias[0] = -100
            model.class_embed.bias[1] = 100

        updated_expected = native_model.predict(image, threshold=0.0)
        updated_actual = facade.predict(image, threshold=0.0)
        assert isinstance(updated_expected, Detections)
        assert isinstance(updated_actual, Detections)
        np.testing.assert_allclose(updated_actual.xyxy, updated_expected.xyxy, atol=0, rtol=0)
        np.testing.assert_array_equal(updated_actual.class_id, updated_expected.class_id)
        assert np.all(updated_actual.class_id == 1)

    def test_live_model_properties_do_not_change_device(self, nano_model: RFDETRNano) -> None:
        """An explicit device mismatch does not mutate the borrowed native model."""
        facade = RFDETRInference(nano_model)

        with pytest.raises(ValueError, match="device"):
            RFDETRInference(nano_model, device="cuda")

        assert facade.runtime_info["device"] == "cpu"
        assert nano_model.model.device == torch.device("cpu")


class TestCheckpointSource:
    """Exercise checkpoint behavior through the inference API."""

    @pytest.mark.parametrize("suffix", [".pt", ".pth"])
    def test_native_checkpoint_path_loads_and_predicts(
        self,
        nano_model: RFDETRNano,
        tmp_path: Path,
        suffix: str,
    ) -> None:
        """The facade reloads native checkpoint weights through the standard checkpoint loader."""
        native_model = nano_model
        native_model.model.class_names = ["cat", "dog"]
        model = native_model.model.model
        assert model is not None
        checkpoint = tmp_path / f"model{suffix}"
        torch.save(
            {
                "model": model.state_dict(),
                "model_config": native_model.model_config.model_dump(),
                "model_name": "RFDETRNano",
                "args": {"class_names": ["cat", "dog"], "num_classes": 2},
            },
            checkpoint,
        )

        facade = RFDETRInference(checkpoint, device="cpu")
        image = np.random.default_rng(44).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        expected = native_model.predict(image, threshold=0.0)
        actual = facade.predict(image, threshold=0.0)

        assert facade.class_names == ["cat", "dog"]
        assert isinstance(expected, Detections)
        assert isinstance(actual, Detections)
        assert expected.confidence is not None
        assert actual.confidence is not None
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=0, rtol=0)
        np.testing.assert_allclose(actual.confidence, expected.confidence, atol=0, rtol=0)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)

    def test_checkpoint_loader_omits_auto_device_and_forwards_trust(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Automatic checkpoint loading leaves device selection to the native loader."""
        loaded_model = SimpleNamespace(model=SimpleNamespace(device=torch.device("cpu")))
        loader = Mock(return_value=loaded_model)
        monkeypatch.setattr(RFDETR, "from_checkpoint", classmethod(lambda cls, path, **kwargs: loader(path, **kwargs)))
        checkpoint = tmp_path / "model.ckpt"

        RFDETRInference(checkpoint, trust_checkpoint=True)

        loader.assert_called_once_with(checkpoint, trust_checkpoint=True)

    def test_checkpoint_path_rejects_explicit_metadata(self, tmp_path: Path) -> None:
        """Native checkpoint paths do not accept exported inference metadata."""
        with pytest.raises(ValueError, match="metadata"):
            RFDETRInference(tmp_path / "model.pt", metadata={"task": "detect"})


@pytest.fixture
def stopped_runtime_load(monkeypatch: pytest.MonkeyPatch) -> tuple[ExportMetadata, Mock]:
    """Stop an exported-artifact load at the runtime loader, returning the metadata it reads and the loader mock.

    Examples:
        The pytest runner must create this fixture before a test can use it.

        >>> stopped_runtime_load()  # doctest: +SKIP
    """
    contract = ExportMetadata(
        format="tensorrt",
        task="detect",
        input_shape=(1, 3, 8, 8),
        outputs={"pred_boxes": "dets", "pred_logits": "labels"},
        means=[0.485, 0.456, 0.406],
        stds=[0.229, 0.224, 0.225],
        class_names=["object"],
        num_classes=1,
        num_select=1,
        trace_alpha=0.2,
        patch_size=1,
        num_windows=1,
    )
    load_runtime = Mock(side_effect=RuntimeError("stop after runtime load"))
    # Patch the module the facade's call-time import resolves (sys.modules), not a parent-package attribute: a
    # sys.modules restore in another test can leave the two pointing at different module objects.
    context_module = importlib.import_module("rfdetr.export._runtime.context")
    monkeypatch.setattr(context_module, "read_metadata", Mock(return_value=contract))
    monkeypatch.setattr(context_module, "load_runtime", load_runtime)
    return contract, load_runtime


class TestExportedDeviceArgument:
    """The device argument reaches an exported artifact's loader as a string."""

    def test_torch_device_is_passed_as_string(
        self, stopped_runtime_load: tuple[ExportMetadata, Mock], tmp_path: Path
    ) -> None:
        """A torch.device, which RFDETR accepts, reaches the format loader as the string every loader compares.

        Loaders call string methods such as startswith on the device, so passing the torch.device through would fail
        with AttributeError instead of selecting the device.
        """
        contract, load_runtime = stopped_runtime_load
        artifact = tmp_path / "model.engine"

        with pytest.raises(RuntimeError, match="stop after runtime load"):
            RFDETRInference(artifact, device=torch.device("cuda", 1))

        load_runtime.assert_called_once_with(artifact, contract, device="cuda:1", options=None)


class TestRuntimeOptionsRouting:
    """Runtime options reach an exported artifact's loader and are refused for native sources."""

    def test_live_model_rejects_runtime_options(self, nano_model: RFDETRNano) -> None:
        """A live model has no exported runtime to configure, so even an empty mapping is refused."""
        with pytest.raises(ValueError, match="runtime_options is only valid for exported artifacts"):
            RFDETRInference(nano_model, runtime_options={})

    def test_checkpoint_path_rejects_runtime_options_before_loading(self, tmp_path: Path) -> None:
        """The refusal comes before the checkpoint is read: the path does not exist, yet no FileNotFoundError."""
        with pytest.raises(ValueError, match="runtime_options is only valid for exported artifacts"):
            RFDETRInference(tmp_path / "model.pt", runtime_options={"verbose": True})

    def test_exported_artifact_forwards_runtime_options(
        self, stopped_runtime_load: tuple[ExportMetadata, Mock], tmp_path: Path
    ) -> None:
        """The facade passes its runtime options, unchanged, through the context builder to the runtime loader."""
        contract, load_runtime = stopped_runtime_load
        artifact = tmp_path / "model.engine"

        with pytest.raises(RuntimeError, match="stop after runtime load"):
            RFDETRInference(artifact, runtime_options={"verbose": True})

        load_runtime.assert_called_once_with(artifact, contract, device="auto", options={"verbose": True})


class TestOptimizedNativeSource:
    """Exercise optimization behavior through the inference API."""

    @pytest.mark.parametrize("inplace", [False, True])
    def test_facade_prediction_uses_model_after_optimization(self, nano_model: RFDETRNano, inplace: bool) -> None:
        """A borrowed facade uses the native model's current optimized execution path."""
        facade = RFDETRInference(nano_model)
        image = np.random.default_rng(43).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        nano_model.inference(compile=False, inplace=inplace)

        expected = nano_model.predict(image, threshold=0.0)
        actual = facade.predict(image, threshold=0.0)
        assert isinstance(expected, Detections)
        assert isinstance(actual, Detections)
        assert expected.confidence is not None
        assert actual.confidence is not None
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=0, rtol=0)
        np.testing.assert_allclose(actual.confidence, expected.confidence, atol=0, rtol=0)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)
