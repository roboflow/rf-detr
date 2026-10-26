# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Native sources for the public inference facade."""

from __future__ import annotations

import importlib
import logging
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from PIL import Image
from supervision import Detections, KeyPoints

from rfdetr.detr import RFDETR
from rfdetr.export._runtime.metadata import ExportMetadata
from rfdetr.inference import RFDETRInference
from rfdetr.variants import RFDETRKeypointPreview, RFDETRNano, RFDETRSegNano


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


class TestInferenceSnapshot:
    """The factory owns an independent predictor and leaves its trainable source unchanged."""

    def test_factory_preserves_source_weights_modes_and_optimization(self, nano_model: RFDETRNano) -> None:
        """Casting the snapshot must not change source tensors, module modes, or optimization state."""
        source = nano_model.model.model
        source.train()
        source.backbone.eval()
        modes = [module.training for module in source.modules()]
        weights = {name: value.clone() for name, value in source.state_dict().items()}
        source_flags = {name: value for name, value in vars(nano_model).items() if "optimized" in name}

        predictor = nano_model.inference(compile=False, dtype="float64")

        assert isinstance(predictor, RFDETRInference)
        assert nano_model.model.model is source
        assert nano_model.model.inference_model is None
        assert [module.training for module in source.modules()] == modes
        assert {name: value for name, value in vars(nano_model).items() if "optimized" in name} == source_flags
        for name, value in source.state_dict().items():
            torch.testing.assert_close(value, weights[name], rtol=0, atol=0)
        assert next(source.parameters()).dtype == torch.float32
        assert next(source.parameters()).device == torch.device("cpu")

    def test_snapshot_predictions_and_metadata_survive_source_changes(self, nano_model: RFDETRNano) -> None:
        """Later weight, label, normalization and config changes must not affect the predictor."""
        nano_model.model.class_names = ["cat", "dog"]
        image = np.random.default_rng(42).integers(0, 256, (64, 64, 3), dtype=np.uint8)
        expected = nano_model.predict(image, threshold=0.0)
        predictor = nano_model.inference(compile=False)
        assert isinstance(predictor, RFDETRInference)
        initial = predictor.predict(image, threshold=0.0)
        np.testing.assert_allclose(initial.xyxy, expected.xyxy, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(initial.confidence, expected.confidence, rtol=1e-4, atol=1e-6)
        np.testing.assert_array_equal(initial.class_id, expected.class_id)

        with torch.no_grad():
            nano_model.model.model.class_embed.bias.fill_(-100)
            nano_model.model.model.class_embed.bias[1] = 100
        nano_model.model.class_names[0] = "changed"
        nano_model.model.args.num_classes = 20
        nano_model.model.resolution = 128
        nano_model.model_config.num_channels = 1
        nano_model.means[0] = 10.0
        nano_model.stds[0] = 10.0
        nano_model.model.model.train()

        actual = predictor.predict(image, threshold=0.0)
        np.testing.assert_array_equal(actual.xyxy, initial.xyxy)
        np.testing.assert_array_equal(actual.confidence, initial.confidence)
        np.testing.assert_array_equal(actual.class_id, initial.class_id)
        assert predictor.class_names == ["cat", "dog"]
        assert predictor.runtime_info["input_shape"] == (None, 3, 64, 64)
        assert nano_model.model.model.training is True

    def test_invalid_request_does_not_place_a_cold_source(self, nano_model: RFDETRNano) -> None:
        """Validation must happen before attempting the configured accelerator move."""
        nano_model.model.device = torch.device("cuda")

        with pytest.raises(ValueError, match="compile_backend"):
            nano_model.inference(compile_backend="unknown")  # type: ignore[arg-type]

        assert next(nano_model.model.model.parameters()).device == torch.device("cpu")
        assert nano_model.model.device == torch.device("cuda")

    @pytest.mark.parametrize(
        "model_type",
        [pytest.param(RFDETRSegNano, id="segmentation"), pytest.param(RFDETRKeypointPreview, id="keypoints")],
    )
    def test_snapshot_preserves_task_outputs(self, model_type: type[RFDETR]) -> None:
        """Segmentation masks and keypoint coordinates survive snapshot export and decoding."""
        native = model_type(pretrain_weights=None, device="cpu", resolution=96, num_queries=4, num_select=4)
        image = np.random.default_rng(43).integers(0, 256, (96, 96, 3), dtype=np.uint8)
        expected = native.predict(image, threshold=0.0)

        predictor = native.inference(compile=False)
        assert isinstance(predictor, RFDETRInference)
        actual = predictor.predict(image, threshold=0.0)

        assert type(actual) is type(expected)
        if isinstance(actual, KeyPoints):
            np.testing.assert_allclose(actual.xy, expected.xy, rtol=1e-4, atol=1e-6)
            np.testing.assert_allclose(actual.keypoint_confidence, expected.keypoint_confidence, rtol=1e-4, atol=1e-6)
        else:
            np.testing.assert_allclose(actual.xyxy, expected.xyxy, rtol=1e-4, atol=1e-6)
            np.testing.assert_array_equal(actual.mask, expected.mask)


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
            assert facade.runtime_info == {
                "backend": "pytorch",
                "device": "cpu",
                "task": "detect",
                "input_shape": (None, 3, None, None),
                "default_shape": (64, 64),
                "batch_size": None,
                "max_batch_size": None,
            }

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

    def test_checkpoint_optimization_recipe_predicts(
        self,
        nano_model: RFDETRNano,
        tmp_path: Path,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The documented owned-checkpoint warning and native optimization recipe run with real weights."""
        checkpoint = tmp_path / "model.pth"
        torch.save(
            {
                "model": nano_model.model.model.state_dict(),
                "model_config": nano_model.model_config.model_dump(),
                "model_name": "RFDETRNano",
                "args": {"class_names": ["cat", "dog"], "num_classes": 2},
            },
            checkpoint,
        )
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            unoptimized = RFDETRInference(checkpoint, device="cpu").predict(image, threshold=0.0)
        assert "RFDETR.from_checkpoint(" in caplog.text

        native_model = RFDETR.from_checkpoint(checkpoint, device="cpu")
        facade = native_model.inference(compile=False)
        actual = facade.predict(image, threshold=0.0)

        assert isinstance(actual, Detections)
        assert isinstance(unoptimized, Detections)
        np.testing.assert_allclose(actual.xyxy, unoptimized.xyxy, rtol=1e-4, atol=1e-6)
        np.testing.assert_allclose(actual.confidence, unoptimized.confidence, rtol=1e-4, atol=1e-6)
        assert facade.runtime_info["input_shape"] == (None, 3, 64, 64)
        assert facade.runtime_info["batch_size"] is None


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

    def test_snapshot_matches_prediction_through_a_borrowed_facade(self, nano_model: RFDETRNano) -> None:
        """Snapshot creation leaves a borrowed facade on the source model's flexible eager path."""
        borrowed = RFDETRInference(nano_model)
        image = np.random.default_rng(43).integers(0, 256, (64, 64, 3), dtype=np.uint8)

        facade = nano_model.inference(compile=False)

        expected = borrowed.predict(image, threshold=0.0)
        actual = facade.predict(image, threshold=0.0)
        assert isinstance(expected, Detections)
        assert isinstance(actual, Detections)
        assert expected.confidence is not None
        assert actual.confidence is not None
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=1e-6, rtol=1e-4)
        np.testing.assert_allclose(actual.confidence, expected.confidence, atol=1e-6, rtol=1e-4)
        np.testing.assert_array_equal(actual.class_id, expected.class_id)

        assert facade.runtime_info["input_shape"] == (None, 3, 64, 64)
        assert facade.runtime_info["batch_size"] is None

        assert borrowed.runtime_info["input_shape"] == (None, 3, None, None)


class TestFacadeContracts:
    """Public facade arguments and inspection work before inference starts."""

    @pytest.mark.parametrize("keyword", ["threhsold", "confidence"])
    def test_unknown_prediction_keyword_fails_before_placement(
        self, nano_model: RFDETRNano, monkeypatch: pytest.MonkeyPatch, keyword: str
    ) -> None:
        """A typo must fail before a borrowed model changes device or executes."""
        facade = RFDETRInference(nano_model)
        placement = Mock(side_effect=AssertionError("Unknown options must fail before placement."))
        monkeypatch.setattr("rfdetr.detr._move_model_context_to_device", placement)

        with pytest.raises(TypeError, match="unexpected keyword argument"):
            facade.predict(np.zeros((64, 64, 3), dtype=np.uint8), **{keyword: 0.9})

        placement.assert_not_called()

    def test_native_warning_gives_a_callable_optimization_recipe(
        self, nano_model: RFDETRNano, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The warning names the native instance and checkpoint loading path, not a missing facade method."""
        monkeypatch.setattr(logging.getLogger("rf-detr"), "propagate", True)
        facade = RFDETRInference(nano_model)
        with caplog.at_level(logging.WARNING, logger="rf-detr"):
            facade.predict(np.zeros((64, 64, 3), dtype=np.uint8))

        assert "native_model.inference(" in caplog.text
        assert "RFDETR.from_checkpoint(" in caplog.text
        assert "predictor = native_model.inference(" in caplog.text
        assert "predictor.predict(" in caplog.text
        assert not hasattr(facade, "inference")

    def test_compiled_constraints_match_the_executable_native_batch(self, nano_model: RFDETRNano) -> None:
        """A real compiled model reports its fixed batch and spatial dimensions."""
        facade = nano_model.inference(batch_size=2)

        assert facade.runtime_info["input_shape"] == (2, 3, 64, 64)
        assert facade.runtime_info["batch_size"] == 2
        images = [np.zeros((64, 64, 3), dtype=np.uint8)] * 2
        predictions = facade.predict(images)
        assert isinstance(predictions, list)
        assert len(predictions) == 2

    def test_compiled_facade_batch_hint_creates_a_replacement_predictor(self, nano_model: RFDETRNano) -> None:
        """The batch error directs users to a new predictor while the previous snapshot stays fixed."""
        facade = nano_model.inference(batch_size=2)
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"native_model.inference\(batch_size="):
            facade.predict(image)

        replacement = nano_model.inference(batch_size=1)
        result = replacement.predict(image)
        assert isinstance(result, Detections)
        assert replacement.runtime_info["batch_size"] == 1
        assert facade.runtime_info["batch_size"] == 2

    def test_snapshot_shape_hint_targets_the_flexible_source_model(self, nano_model: RFDETRNano) -> None:
        """The source keeps flexible shapes even when an owned snapshot fixes its resolution."""
        facade = nano_model.inference(compile=False)
        image = np.zeros((64, 64, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match=r"native_model.predict\("):
            facade.predict(image, shape=(128, 128))

        result = nano_model.predict(image, shape=(128, 128))
        assert isinstance(result, Detections)
        assert facade.runtime_info["input_shape"] == (None, 3, 64, 64)

    def test_runtime_information_is_an_independent_copy(self, nano_model: RFDETRNano) -> None:
        """Mutating inspection values cannot alter subsequent inspection or prediction configuration."""
        facade = RFDETRInference(nano_model)
        info = facade.runtime_info
        info["input_shape"] = (1, 1, 1, 1)
        info["task"] = "segment"

        assert facade.runtime_info["input_shape"] == (None, 3, None, None)
        assert facade.runtime_info["task"] == "detect"

    @pytest.mark.parametrize(
        ("model_type", "task"),
        [
            pytest.param(RFDETRSegNano, "segment", id="segmentation"),
            pytest.param(RFDETRKeypointPreview, "keypoints", id="keypoints"),
        ],
    )
    def test_runtime_information_reports_native_task(self, model_type: type[RFDETR], task: str) -> None:
        """Inspection identifies actual segmentation and keypoint variants without inference."""
        native = model_type(pretrain_weights=None, device="cpu", resolution=64, num_queries=4, num_select=4)
        facade = RFDETRInference(native)

        assert facade.runtime_info["task"] == task
        assert facade.runtime_info["input_shape"] == (None, 3, None, None)

    def test_native_pathlike_batch_uses_the_shared_image_boundary(self, nano_model: RFDETRNano, tmp_path: Path) -> None:
        """Both public native entry points accept the Path objects produced by directory traversal."""
        path = tmp_path / "image.png"
        image = np.full((64, 64, 3), 127, dtype=np.uint8)
        Image.fromarray(image).save(path)
        native_results = nano_model.predict([path])
        facade_results = RFDETRInference(nano_model).predict((path,))

        assert isinstance(native_results, list)
        assert isinstance(facade_results, list)
        np.testing.assert_array_equal(facade_results[0].xyxy, native_results[0].xyxy)
        np.testing.assert_array_equal(facade_results[0].metadata["source_image"], image)
