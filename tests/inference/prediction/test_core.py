# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
import math
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import PIL.Image
import pytest
import supervision as sv
import torch

from rfdetr import RFDETRInference, RFDETRNano, RFDETRSegNano
from rfdetr._prediction import core
from rfdetr.detr import RFDETR
from rfdetr.models.postprocess import PostProcess
from rfdetr.utilities.keypoints import precision_cholesky_to_pixel_covariance

from ..helpers import _DummyModel, _DummyRFDETR


class TestPredictEmptyInputs:
    """Empty image containers fail consistently across native and exported prediction."""

    @pytest.mark.parametrize("images", [pytest.param([], id="list"), pytest.param((), id="tuple")])
    def test_native_rejects_empty_images(self, images: list[Any] | tuple[Any, ...]) -> None:
        """Native prediction rejects empty containers before stacking tensors."""
        model = _DummyRFDETR()

        with pytest.raises(ValueError, match="^At least one image is required\\.$"):
            model.predict(images)

    @pytest.mark.parametrize("images", [pytest.param([], id="list"), pytest.param((), id="tuple")])
    def test_dynamic_batch_rejects_empty_images(self, images: list[Any] | tuple[Any, ...]) -> None:
        """Dynamic-batch prediction rejects empty containers without mentioning None."""
        model = _DummyRFDETR()
        context = replace(model._prediction_context(), fixed_shape=True, batch_size=None)

        with pytest.raises(ValueError, match="^At least one image is required\\.$") as exc_info:
            core.predict(lambda: context, images)

        assert "None" not in str(exc_info.value)


class TestPredictReturnTypes:
    """``RFDETR.predict()`` API contract tests using synthetic images.

    Quality is not assessed here — see ``tests/benchmarks/test_inference_coco.py``.
    """

    def test_detection_returns_sv_detections(self) -> None:
        """Detection model returns a list of ``sv.Detections``."""
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
        model = RFDETRNano(pretrain_weights=None)
        detections = model.predict([img, img], threshold=0.3)
        assert isinstance(detections, list), "predict() must return a list for multiple inputs"
        assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"

    def test_segmentation_returns_sv_detections_with_masks(self) -> None:
        """Segmentation model returns ``sv.Detections`` with the mask field always set."""
        img = PIL.Image.new("RGB", (640, 640), color=(128, 128, 128))
        model = RFDETRSegNano(pretrain_weights=None)
        detections = model.predict([img, img], threshold=0.3)
        assert isinstance(detections, list), "predict() must return a list for multiple inputs"
        assert all(isinstance(d, sv.Detections) for d in detections), "Each result must be sv.Detections"
        assert all(d.mask is not None for d in detections), (
            "Segmentation predict() must always set the mask field, even when no objects are detected"
        )

    def test_keypoint_single_and_batch_return_sv_keypoints(self) -> None:
        """Keypoint model returns one KeyPoints for one image and list[KeyPoints] for multiple images."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1], include_keypoints=True)

        single = model.predict(img)
        batch = model.predict([img, img])

        assert isinstance(single, sv.KeyPoints)
        assert isinstance(batch, list)
        assert all(isinstance(result, sv.KeyPoints) for result in batch)

    @pytest.mark.parametrize("use_facade", [False, True])
    def test_generator_input_returns_one_result_per_image(self, use_facade: bool) -> None:
        """An iterable that is not a list or tuple is read into a list and predicted like one.

        Callers often hold images in a generator or another lazy iterable; both public APIs must treat it as a
        collection, not as one opaque image.
        """
        native = _DummyRFDETR()
        predictor = RFDETRInference(native) if use_facade else native
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))

        results = predictor.predict(image for image in (img, img, img))

        assert isinstance(results, list)
        assert len(results) == 3

    def test_native_bchw_tensor_returns_one_result_per_image(self) -> None:
        """A BCHW tensor passed to the native model is predicted as a batch of B images."""
        results = _DummyRFDETR().predict(torch.zeros((2, 3, 48, 64)))

        assert isinstance(results, list)
        assert len(results) == 2

    @pytest.mark.parametrize("option", ["stream", "batch", "vid_stride", "stream_buffer"])
    def test_native_rejects_source_options(self, option: str) -> None:
        """The native model refuses source options instead of ignoring them with a deprecation warning.

        Source expansion and streaming belong to ``RFDETRInference``; silently ignoring ``stream=True`` would return
        eager results where the caller expected a generator.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))

        with pytest.raises(TypeError, match=rf"does not accept {option}: .*RFDETRInference\(model\)"):
            _DummyRFDETR().predict(img, **{option: 1})

    @pytest.mark.parametrize("input_kind", ["pil", "path"])
    def test_native_single_channel_images_use_pillow_luminance(self, tmp_path: Path, input_kind: str) -> None:
        """Native prediction shares grayscale conversion and source metadata with exported prediction."""
        model = _DummyRFDETR()
        model.model_config.num_channels = 1
        model.means = model.means[:1]
        model.stds = model.stds[:1]
        image = PIL.Image.new("RGB", (29, 17), (17, 127, 231))
        path = tmp_path / "image.png"
        image.save(path)
        expected = np.array(image.convert("L"))

        result = model.predict(image if input_kind == "pil" else path)

        assert result.metadata["source_image"].shape == (17, 29)
        assert result.metadata["source_image"].dtype == np.uint8
        np.testing.assert_array_equal(result.metadata["source_image"], expected)

    def test_native_unknown_keyword_keeps_legacy_behavior(self) -> None:
        """Unknown native keywords warn while preserving the prediction result."""
        model = _DummyRFDETR()
        image = np.zeros((28, 28, 3), dtype=np.uint8)

        expected = model.predict(image)
        with pytest.warns(FutureWarning, match="legacy_option"):
            actual = model.predict(image, legacy_option=True)

        assert actual == expected

    def test_native_unknown_keyword_warning_points_to_caller(self) -> None:
        """The unknown-keyword warning identifies the calling test file."""
        model = _DummyRFDETR()
        image = np.zeros((28, 28, 3), dtype=np.uint8)

        with pytest.warns(FutureWarning, match="legacy_option") as recorded:
            model.predict(image, legacy_option=True)

        assert Path(recorded[0].filename).resolve() == Path(__file__).resolve()

    def test_native_unknown_keyword_warning_lists_sorted_names(self) -> None:
        """The deprecation warning lists ignored keywords in sorted order and states the removal version."""
        model = _DummyRFDETR()
        image = np.zeros((28, 28, 3), dtype=np.uint8)

        with pytest.warns(FutureWarning, match="alpha, zeta") as recorded:
            model.predict(image, zeta=True, alpha=True)

        assert str(recorded[0].message) == (
            "Passing unknown keyword arguments to RFDETR.predict is deprecated since v1.12.0: "
            "alpha, zeta. They are ignored and will raise TypeError in v2.0.0."
        )

    def test_shared_predict_rejects_unknown_keyword(self) -> None:
        """The shared predictor rejects unknown keywords before executing inference."""
        model = _DummyRFDETR()
        image = np.zeros((28, 28, 3), dtype=np.uint8)

        with pytest.raises(TypeError, match="legacy_option"):
            core.predict(model._prediction_context, image, legacy_option=True)


class _TupleOutputModelContext:
    """Model context whose forward returns a 3-tuple, mirroring ``forward_export()`` after ``inference()``.

    Regression fixture for GitHub #1208: ``predict()`` mislabels the tuple's third element as ``pred_masks`` instead of
    ``pred_keypoints`` because it reads a nonexistent ``model.model_config`` attribute instead of ``model.args``.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.resolution = 28
        self.class_names = ["object"]
        self.args = SimpleNamespace(use_grouppose_keypoints=True, num_keypoints_per_class=[17])
        self.model = torch.nn.Identity()
        self.inference_model = self._forward
        self.captured_predictions: dict[str, torch.Tensor] | None = None
        self.captured_score_threshold: float | None = None

    def _forward(self, batch_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch = batch_tensor.shape[0]
        boxes = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]] * batch)
        logits = torch.full((batch, 1, 1), 10.0)
        keypoints = torch.full((batch, 1, 17, 3), 0.5)
        return boxes, logits, keypoints

    def postprocess(
        self,
        predictions: dict[str, torch.Tensor],
        target_sizes: torch.Tensor,
        score_threshold: float | None = None,
    ) -> list[dict[str, torch.Tensor]]:
        self.captured_predictions = predictions
        self.captured_score_threshold = score_threshold
        batch = target_sizes.shape[0]
        results = []
        for _ in range(batch):
            result: dict[str, torch.Tensor] = {
                "scores": torch.tensor([0.9]),
                "labels": torch.tensor([0]),
                "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            }
            if "pred_keypoints" in predictions:
                result["keypoints"] = torch.full((1, 17, 3), 0.5)
            results.append(result)
        return results


def _make_optimized_keypoint_model() -> tuple[RFDETR, _TupleOutputModelContext]:
    """Build a ``_DummyRFDETR`` wired to look like it already ran ``inference()``.

    Examples:
        >>> model, stub = _make_optimized_keypoint_model()
        >>> model._is_optimized_for_inference
        True
        >>> isinstance(stub, _TupleOutputModelContext)
        True
    """
    model = _DummyRFDETR()
    stub = _TupleOutputModelContext()
    model.model = stub
    model._is_optimized_for_inference = True
    model._optimized_resolution = stub.resolution
    model._optimized_has_been_compiled = False
    model._optimized_dtype = torch.float32
    return model, stub


class TestPredictOptimizedInferenceKeypoints:
    """Regression tests for GitHub #1208: inference() breaks keypoint predict()."""

    def test_tuple_output_labels_third_slot_as_keypoints_not_masks(self) -> None:
        """The 3rd tuple slot must be labeled pred_keypoints, not pred_masks, when use_grouppose_keypoints=True."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model, stub = _make_optimized_keypoint_model()

        model.predict(img)

        assert stub.captured_predictions is not None
        assert "pred_keypoints" in stub.captured_predictions
        assert "pred_masks" not in stub.captured_predictions

    def test_optimized_keypoint_model_predict_returns_sv_keypoints(self) -> None:
        """Predict() must return sv.KeyPoints, not sv.Detections, for an optimized keypoint model."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model, _stub = _make_optimized_keypoint_model()

        result = model.predict(img)

        assert isinstance(result, sv.KeyPoints), (
            f"expected sv.KeyPoints for optimized keypoint model, got {type(result)}"
        )

    def test_predict_forwards_threshold_to_postprocess(self) -> None:
        """Predict must pass its public threshold to post-processing before mask work begins."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model, stub = _make_optimized_keypoint_model()

        model.predict(img, threshold=0.31)

        assert stub.captured_score_threshold == 0.31

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.bfloat16, id="bfloat16"),
            pytest.param(torch.float32, id="float32"),
            pytest.param(torch.float64, id="float64"),
        ],
    )
    @pytest.mark.parametrize("num_detections", [1, 2])
    def test_low_findability_retains_finite_keypoint_detection(self, dtype: torch.dtype, num_detections: int) -> None:
        """Finite tuple predictions must survive trace fusion and public score filtering."""
        model, stub = _make_optimized_keypoint_model()
        model._optimized_dtype = dtype
        stub.args.num_keypoints_per_class = [2]
        stub.postprocess = PostProcess(num_select=num_detections, num_keypoints_per_class=[2])
        keypoints = torch.zeros(1, num_detections, 2, 8, dtype=dtype)
        keypoints[..., :2] = 0.5
        keypoints[..., 2] = -20.0
        stub.inference_model = MagicMock(
            return_value=(
                torch.full((1, num_detections, 4), 0.5, dtype=dtype),
                torch.full((1, num_detections, 1), 4.0, dtype=dtype),
                keypoints,
            )
        )
        image = PIL.Image.new("RGB", (64, 48))

        result = model.predict(image, threshold=0.3)

        object_probability = 1.0 / (1.0 + math.exp(-4.0))
        fused_probability = object_probability * 2.0**-stub.postprocess.trace_alpha
        expected_score = fused_probability / (1.0 + fused_probability)
        assert isinstance(result, sv.KeyPoints)
        assert len(result) == num_detections
        np.testing.assert_allclose(result.detection_confidence, expected_score, rtol=0.005)
        np.testing.assert_array_equal(result.xy, np.tile([[[32.0, 24.0], [32.0, 24.0]]], (num_detections, 1, 1)))


class TestPredictSourceData:
    """Verify ``predict()`` source metadata behavior."""

    def test_source_image_included_by_default(self) -> None:
        """source_image remains included by default for API compatibility."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img)
        assert "source_image" in detections.metadata
        assert isinstance(detections.metadata["source_image"], np.ndarray)
        assert detections.metadata["source_image"].shape == (48, 64, 3)
        assert np.array_equal(detections.data["source_shape"], np.array([[48, 64]]))

    def test_source_image_included_by_default_tensor(self) -> None:
        """Tensor input keeps source_image by default for API compatibility."""
        tensor = torch.rand(3, 48, 64)
        model = _DummyRFDETR()
        detections = model.predict(tensor)
        assert "source_image" in detections.metadata
        assert isinstance(detections.metadata["source_image"], np.ndarray)
        assert detections.metadata["source_image"].dtype == np.uint8
        assert detections.metadata["source_image"].shape == (48, 64, 3)
        assert np.array_equal(detections.data["source_shape"], np.array([[48, 64]]))

    def test_source_image_can_be_disabled(self) -> None:
        """include_source_image=False omits source_image for memory-sensitive paths."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img, include_source_image=False)
        assert "source_image" not in detections.metadata
        assert np.array_equal(detections.data["source_shape"], np.array([[48, 64]]))

    def test_source_image_from_pil(self) -> None:
        """PIL input stores the original image as a numpy array."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img, include_source_image=True)
        assert "source_image" in detections.metadata
        assert isinstance(detections.metadata["source_image"], np.ndarray)
        assert detections.metadata["source_image"].shape == (48, 64, 3)

    def test_source_shape_from_pil(self) -> None:
        """PIL input stores source_shape as a per-detection numpy array."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img)
        assert "source_shape" in detections.data
        assert isinstance(detections.data["source_shape"], np.ndarray)
        assert detections.data["source_shape"].dtype == np.int64
        assert detections.data["source_shape"].shape == (len(detections), 2)
        assert np.array_equal(detections.data["source_shape"][0], [48, 64])

    def test_source_image_from_tensor(self) -> None:
        """Tensor input stores the original image as a uint8 numpy array."""
        tensor = torch.rand(3, 48, 64)
        model = _DummyRFDETR()
        detections = model.predict(tensor, include_source_image=True)
        assert "source_image" in detections.metadata
        assert isinstance(detections.metadata["source_image"], np.ndarray)
        assert detections.metadata["source_image"].dtype == np.uint8
        assert detections.metadata["source_image"].shape == (48, 64, 3)

    @pytest.mark.gpu
    @pytest.mark.parametrize(
        ("dtype", "shape", "expected_transfer_dtype"),
        [
            pytest.param(torch.float16, (3, 48, 64), torch.uint8, id="float16"),
            pytest.param(torch.float32, (3, 48, 64), torch.uint8, id="float32"),
            pytest.param(torch.float64, (3, 48, 64), torch.uint8, id="float64"),
            pytest.param(torch.float32, (3, 1, 64), torch.float32, id="degenerate_height"),
            pytest.param(torch.float32, (3, 48, 1), torch.float32, id="degenerate_width"),
        ],
    )
    def test_cuda_source_image_transfers_exact_bytes(
        self,
        dtype: torch.dtype,
        shape: tuple[int, int, int],
        expected_transfer_dtype: torch.dtype,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The default public path transfers uint8 storage without changing source metadata."""
        values = torch.rand(shape, generator=torch.Generator().manual_seed(20260821))
        boundary_count = min(256, values.numel())
        values.view(-1)[:boundary_count] = torch.arange(boundary_count, dtype=torch.float32) / 255
        tensor = values.to(device="cuda", dtype=dtype)
        expected = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        expected_source_shape = (shape[1], shape[2], shape[0])
        transferred_dtypes: list[torch.dtype] = []
        original_cpu = torch.Tensor.cpu

        def record_source_transfer(image: torch.Tensor) -> torch.Tensor:
            """Record the source-image transfer dtype before delegating to PyTorch.

            Examples:
                This closure requires CUDA and is covered by the enclosing GPU test:
                >>> record_source_transfer(torch.zeros(3, 48, 64, device="cuda"))  # doctest: +SKIP
            """
            if image.device.type == "cuda" and tuple(image.shape) == expected_source_shape:
                transferred_dtypes.append(image.dtype)
            return original_cpu(image)

        monkeypatch.setattr(torch.Tensor, "cpu", record_source_transfer)

        detections = _DummyRFDETR().predict(tensor)
        actual = detections.metadata["source_image"]

        assert transferred_dtypes == [expected_transfer_dtype]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.strides == expected.strides
        assert actual.flags.owndata == expected.flags.owndata
        assert actual.flags.writeable == expected.flags.writeable
        assert actual.tobytes() == expected.tobytes()

    @pytest.mark.gpu
    def test_cuda_source_image_transfers_exact_bytes_strided_input(self) -> None:
        """The fast path matches the previous NumPy path byte-for-byte for a non-contiguous CUDA tensor."""
        values = torch.rand((3, 96, 128), generator=torch.Generator().manual_seed(20260821))
        tensor = values.to(device="cuda", dtype=torch.float32)[:, ::2, ::2]
        assert not tensor.is_contiguous()
        expected = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

        detections = _DummyRFDETR().predict(tensor)
        actual = detections.metadata["source_image"]

        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.strides == expected.strides
        assert actual.flags.owndata == expected.flags.owndata
        assert actual.flags.writeable == expected.flags.writeable
        assert actual.tobytes() == expected.tobytes()

    @pytest.mark.gpu
    def test_cuda_source_image_nan_conversion_emits_no_warning(self) -> None:
        """The on-device cast does not emit NumPy's incidental invalid-cast RuntimeWarning that the old CPU cast did."""
        tensor = torch.full((3, 4, 5), torch.nan, device="cuda", dtype=torch.float32)
        expected = np.zeros((4, 5, 3), dtype=np.uint8)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            detections = _DummyRFDETR().predict(tensor)
        assert caught == []
        assert "source_image" in detections.metadata
        actual = detections.metadata["source_image"]
        assert actual.shape == expected.shape
        assert actual.dtype == expected.dtype
        assert actual.tobytes() == expected.tobytes()

    def test_tensor_with_negative_values_raises(self) -> None:
        """Tensor with negative pixel values raises ValueError."""
        tensor = torch.full((3, 48, 64), -0.1)
        model = _DummyRFDETR()
        with pytest.raises(ValueError, match="below 0"):
            model.predict(tensor)

    def test_source_image_batch(self) -> None:
        """Batch predict stores a source_image per detection."""
        img1 = PIL.Image.new("RGB", (64, 48), color=(100, 100, 100))
        img2 = PIL.Image.new("RGB", (32, 24), color=(200, 200, 200))
        model = _DummyRFDETR()
        detections_list = model.predict([img1, img2], include_source_image=True)
        assert isinstance(detections_list, list)
        assert detections_list[0].metadata["source_image"].shape == (48, 64, 3)
        assert detections_list[1].metadata["source_image"].shape == (24, 32, 3)
        assert np.array_equal(detections_list[0].data["source_shape"], np.array([[48, 64]]))
        assert np.array_equal(detections_list[1].data["source_shape"], np.array([[24, 32]]))

    def test_source_shape_survives_detections_iteration(self) -> None:
        """Iterating sv.Detections must not raise TypeError and must yield correct values.

        Regression test for https://github.com/roboflow/rf-detr/issues/963. supervision's Detections.__iter__ calls
        get_data_item() on every data value, which requires array-like types — storing source_shape as a Python tuple
        raised TypeError: Unsupported data type for key 'source_shape': <class 'tuple'>.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        detections = model.predict(img)

        # sv.Detections.__iter__ yields (xyxy, mask, confidence, class_id, tracker_id, data)
        iterated = list(detections)
        assert len(iterated) == len(detections)
        # Each iterated element's data dict must contain a 1-D [h, w] array
        for det_tuple in iterated:
            data = det_tuple[-1]
            assert np.array_equal(data["source_shape"], [48, 64])

    def test_source_image_survives_boolean_index(self) -> None:
        """Boolean-mask indexing must not raise IndexError when source_image is present.

        Regression test for https://github.com/roboflow/rf-detr/issues/968. source_image was stored as (H, W, C) in
        detections.data; supervision's __getitem__ tried to index it with a per-detection boolean mask, raising
        IndexError because H != N.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1])  # 2 detections
        detections = model.predict(img)  # include_source_image=True by default

        # Boolean-mask filtering — the pattern from issue #968
        mask = detections.confidence > 0.5
        filtered = detections[mask]
        assert len(filtered) == int(mask.sum())
        # source_image must survive the index operation unchanged (not dropped, not sliced)
        assert "source_image" in filtered.metadata
        assert filtered.metadata["source_image"].shape == (48, 64, 3)

    def test_source_image_survives_class_id_boolean_index(self) -> None:
        """Boolean index on class_id must not raise IndexError — exact issue #968 pattern.

        The reporter used ``detections.class_id == 1`` to filter by class, producing a partial boolean mask (1 of 2
        detections).  This is the primary reproduction path from the original bug report.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1])  # class_id 0 and 1
        detections = model.predict(img)

        # Exact pattern from issue #968: filter by class_id
        mask = detections.class_id == 1  # partial mask — 1 of 2 detections
        filtered = detections[mask]
        assert len(filtered) == 1
        assert "source_image" in filtered.metadata
        assert filtered.metadata["source_image"].shape == (48, 64, 3)

    def test_source_image_survives_integer_index(self) -> None:
        """Integer indexing must pass metadata["source_image"] through unchanged."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1])  # 2 detections
        detections = model.predict(img)

        single = detections[0]
        assert "source_image" in single.metadata
        assert single.metadata["source_image"].shape == (48, 64, 3)

    def test_predict_keypoints_return_supervision_keypoints(self) -> None:
        """Keypoint predictions return ``sv.KeyPoints`` after threshold filtering."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1], include_keypoints=True)

        key_points = model.predict(img)

        assert isinstance(key_points, sv.KeyPoints)
        assert key_points.xy.shape == (2, 17, 2)
        assert np.allclose(key_points.xy, 0.5)
        assert np.allclose(key_points.keypoint_confidence, 0.5)
        np.testing.assert_array_equal(key_points.visible, np.full((2, 17), True))
        np.testing.assert_array_equal(key_points.class_id, np.array([0, 1]))
        np.testing.assert_allclose(key_points.data["xyxy"], np.array([[0, 0, 1, 1], [0, 0, 1, 1]], dtype=np.float32))
        np.testing.assert_allclose(key_points.detection_confidence, np.array([0.9, 0.9], dtype=np.float32))
        assert "keypoint_precision_cholesky" in key_points.data
        keypoint_precision = key_points.data["keypoint_precision_cholesky"]
        assert isinstance(keypoint_precision, np.ndarray)
        assert keypoint_precision.shape == (2, 17, 3)
        assert np.allclose(keypoint_precision, 0.25)
        assert "covariance" in key_points.data
        np.testing.assert_allclose(
            key_points.data["covariance"],
            precision_cholesky_to_pixel_covariance(
                precision_cholesky=keypoint_precision, source_shape=key_points.data["source_shape"]
            ),
            rtol=1e-4,
            atol=1e-6,
        )

    def test_predict_keypoints_empty_threshold_return_supervision_keypoints(self) -> None:
        """Keypoint predictions remain ``sv.KeyPoints`` when all detections are filtered."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1], include_keypoints=True)

        key_points = model.predict(img, threshold=1.1)

        assert isinstance(key_points, sv.KeyPoints)
        assert len(key_points) == 0
        assert key_points.xy.shape == (0, 17, 2)
        assert key_points.keypoint_confidence.shape == (0, 17)
        assert key_points.detection_confidence.shape == (0,)
        assert key_points.visible.shape == (0, 17)
        np.testing.assert_allclose(key_points.data["xyxy"], np.empty((0, 4), dtype=np.float32))
        assert key_points.as_detections().is_empty()

    def test_predict_non_keypoint_no_keypoints_key_in_data(self) -> None:
        """Non-keypoint predictions do not attach keypoint fields."""
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()

        detections = model.predict(img)

        assert "keypoints" not in detections.data
        assert not hasattr(detections, "keypoints")

    def test_source_shape_survives_detections_indexing(self) -> None:
        """Integer and boolean-mask indexing of sv.Detections must work correctly.

        Regression test for https://github.com/roboflow/rf-detr/issues/963. MeanAveragePrecision.compute() uses
        __getitem__ (not just __iter__) on Detections objects — both paths go through get_data_item() and would have
        crashed on the old tuple format.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1])  # 2 detections
        detections = model.predict(img)

        # Integer indexing: detections[i] returns a Detections with 1 element
        single = detections[0]
        assert np.array_equal(single.data["source_shape"], np.array([[48, 64]]))

        # Boolean-mask indexing: used by supervision metrics to filter detections
        mask = detections.confidence > 0.5
        filtered = detections[mask]
        assert filtered.data["source_shape"].shape == (int(mask.sum()), 2)
        assert np.all(filtered.data["source_shape"] == np.array([48, 64]))

    def test_source_shape_correct_for_zero_detections(self) -> None:
        """source_shape must have shape (0, 2) when threshold filters all detections.

        Regression test for https://github.com/roboflow/rf-detr/issues/963. The zero-detection path must not raise and
        must produce an empty array, not a scalar or a (1, 2) array.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        # confidence=0.9 < 1.1 → all detections filtered
        detections = model.predict(img, threshold=1.1)
        assert "source_shape" in detections.data
        assert isinstance(detections.data["source_shape"], np.ndarray)
        assert detections.data["source_shape"].shape == (0, 2)

    def test_source_shape_correct_for_multiple_detections(self) -> None:
        """source_shape must have shape (N, 2) for N detections, each row [height, width].

        Regression test for https://github.com/roboflow/rf-detr/issues/963.
        """
        img = PIL.Image.new("RGB", (64, 48), color=(128, 128, 128))
        model = _DummyRFDETR()
        model.model = _DummyModel(labels=[0, 1])  # 2 detections
        detections = model.predict(img)
        assert "source_shape" in detections.data
        assert isinstance(detections.data["source_shape"], np.ndarray)
        assert detections.data["source_shape"].shape == (2, 2)
        assert np.all(detections.data["source_shape"] == np.array([48, 64]))


class TestPredictClassNameData:
    """Verify that ``predict()`` populates ``data["class_name"]`` in the returned Detections.

    class IDs are always 0-indexed (COCO category IDs are remapped during training); including the class name string in
    ``data`` lets callers read the class directly without a separate lookup into ``model.class_names``.
    """

    def _make_model_with_class_names(self, class_names: list[str], labels: list[int]) -> _DummyRFDETR:
        """Return a _DummyRFDETR whose inner model carries custom class_names and returns given labels."""
        model = _DummyRFDETR()
        model.model = _DummyModel(class_names=class_names, labels=labels)
        return model

    def test_class_name_key_present_in_detections_data(self) -> None:
        """Predict() must include 'class_name' in detections.data when class_names is set."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert "class_name" in detections.data, "data['class_name'] must be present"

    def test_class_name_values_match_class_id(self) -> None:
        """class_name at each position must equal class_names[class_id]."""
        model = self._make_model_with_class_names(["cat", "dog", "bird"], labels=[0, 1, 2])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        np.testing.assert_array_equal(
            detections.data["class_name"],
            np.array(["cat", "dog", "bird"]),
            err_msg="class_name must match class_names[class_id] for each detection",
        )

    def test_class_name_with_remapped_coco_dataset(self) -> None:
        """Simulates a single-class COCO dataset where category_id=1 is remapped to label=0.

        After training with remap_category_ids=True, the model outputs class_id=0 for the first class.  class_name must
        correctly map 0 → the first class name.
        """
        # Single-class model: category_id=1 was remapped to label=0 during training.
        model = self._make_model_with_class_names(["myclass"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.class_id[0] == 0, "class_id must be 0 (0-indexed)"
        assert detections.data["class_name"][0] == "myclass", (
            "class_name must be 'myclass' even though the original COCO category_id was 1"
        )

    def test_class_name_falls_back_to_coco_when_no_custom_names(self) -> None:
        """Without custom class_names, class_name maps class_id via COCO_CLASS_NAMES."""
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        # _DummyModel with no custom class_names; labels=[1] → COCO_CLASS_NAMES[1]
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert "class_name" in detections.data
        assert detections.data["class_name"][0] == COCO_CLASS_NAMES[1], (
            "class_name must fall back to COCO_CLASS_NAMES[class_id]"
        )

    def test_class_name_empty_array_when_no_detections(self) -> None:
        """When threshold filters all detections, data['class_name'] must be an empty array."""
        model = self._make_model_with_class_names(["cat"], labels=[0])
        img = PIL.Image.new("RGB", (28, 28))
        # threshold=1.1 filters out all detections (confidence=0.9 < 1.1)
        detections = model.predict(img, threshold=1.1)
        assert "class_name" in detections.data
        assert len(detections.data["class_name"]) == 0, "class_name must be empty when no detections pass threshold"
        assert detections.data["class_name"].dtype == object, (
            "class_name dtype must be object even when the array is empty (not float64)"
        )

    def test_class_name_out_of_bounds_class_id_returns_empty_string(self) -> None:
        """A class_id >= len(class_names) must map to an empty string (no IndexError)."""
        # class_names has 2 entries but labels includes out-of-bounds id=5
        model = self._make_model_with_class_names(["cat", "dog"], labels=[5])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "", "Out-of-bounds class_id must produce empty string"

    def test_class_name_negative_class_id_returns_empty_string(self) -> None:
        """A negative class_id must map to an empty string (bounds check: 0 <= cid)."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[-1])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "", "Negative class_id must produce empty string"

    def test_class_name_populated_for_each_image_in_batch(self) -> None:
        """class_name must be correctly populated for every Detections in a batch prediction."""
        model = self._make_model_with_class_names(["cat", "dog"], labels=[0, 1])
        img1 = PIL.Image.new("RGB", (28, 28))
        img2 = PIL.Image.new("RGB", (28, 28))
        results = model.predict([img1, img2])
        assert isinstance(results, list), "batch predict must return a list"
        assert len(results) == 2, "one Detections per input image"
        for idx, det in enumerate(results):
            assert "class_name" in det.data, f"image {idx}: class_name must be present"
            assert list(det.data["class_name"]) == ["cat", "dog"], (
                f"image {idx}: class_name must match class_names[class_id]"
            )

    def test_background_class_id_maps_to_background_label(self) -> None:
        """DETR's background/no-object class (class_id == n) must map to '__background__'.

        RF-DETR internally allocates num_classes + 1 outputs; the extra class at index n is the background/no-object
        class. Returning it as '__background__' is unambiguous, whereas the previous empty string was indistinguishable
        from a genuine OOB error.

        Regression / contract test for https://github.com/roboflow/rf-detr/pull/966 post-merge issue reported by
        @Alarmod.
        """
        # class_names has 2 entries (n=2); background class is label index 2
        model = self._make_model_with_class_names(["cat", "dog"], labels=[2])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "__background__", (
            "Background class (class_id == num_classes) must map to '__background__', not empty string"
        )

    def test_background_class_id_does_not_emit_oob_warning(self) -> None:
        """Predicting the background class must not emit an out-of-range warning.

        The background class (class_id == num_classes) is expected DETR behaviour, not a model error. Warning on it
        misleads users into thinking something is wrong.

        Uses _warned_once state (not caplog) because the RF-DETR logger has propagate=False, which prevents caplog from
        capturing records via the root-logger handler.

        Regression / contract test for https://github.com/roboflow/rf-detr/pull/966 post-merge issue reported by
        @Alarmod.
        """
        from rfdetr.utilities.logger import get_logger

        # Reset warning_once state so this test is not affected by earlier tests that may
        # have already triggered the same message template, masking a reintroduced warning.
        logger = get_logger()
        logger._warned_once.clear()

        model = self._make_model_with_class_names(["cat", "dog"], labels=[2])
        img = PIL.Image.new("RGB", (28, 28))
        model.predict(img)
        unmapped_warnings = [msg for msg in logger._warned_once if "unmapped class_id" in msg]
        assert not unmapped_warnings, "Background class must not trigger unmapped-class-id warning"

    def test_truly_oob_class_id_still_maps_to_empty_string_and_warns(self) -> None:
        """A class_id strictly above num_classes still maps to empty string AND emits a warning.

        class_id == n is background (no warning); class_id > n is truly unexpected — must produce '' AND trigger the
        out-of-range warning so the caller knows something is wrong.

        Uses _warned_once state (not caplog) because the RF-DETR logger has propagate=False, which prevents caplog from
        capturing records via the root-logger handler.
        """
        from rfdetr.utilities.logger import get_logger

        # Reset warning_once state so this test is not affected by deduplication from earlier tests.
        logger = get_logger()
        logger._warned_once.clear()

        # n=2, background is class_id=2; class_id=5 is truly OOB (> n)
        model = self._make_model_with_class_names(["cat", "dog"], labels=[5])
        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)
        assert detections.data["class_name"][0] == "", "Truly OOB class_id (> num_classes) must produce empty string"
        unmapped_warnings = [msg for msg in logger._warned_once if "unmapped class_id" in msg]
        assert unmapped_warnings, "Truly OOB class_id (> num_classes) must trigger an unmapped-class-id warning"

    @pytest.mark.parametrize(
        ("class_id", "expected_name"),
        [
            pytest.param(18, "dog", id="coco_id_18_dog"),
            pytest.param(27, "backpack", id="coco_id_27_backpack"),
            pytest.param(3, "car", id="coco_id_3_car"),
        ],
    )
    def test_coco_pretrained_sparse_id_mapping(self, class_id: int, expected_name: str) -> None:
        """Pretrained COCO models use raw COCO category IDs (1-indexed, with gaps) as class_ids.

        When num_classes=90 and class_names has 80 entries, class_id 18 must resolve to 'dog' (COCO category 18), not
        'sheep' (COCO_CLASS_NAMES[18] via 0-indexed lookup).

        Regression test for
        https://github.com/roboflow/rf-detr/issues/988.
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        coco_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[class_id])
        coco_model.args = SimpleNamespace(num_classes=90)
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == expected_name, (
            f"class_id={class_id} must map to '{expected_name}', got '{detections.data['class_name'][0]}'"
        )

    def test_coco_pretrained_dataset_file_roboflow(self) -> None:
        """Pretrained COCO weights packaged as dataset_file='roboflow' must still use sparse-ID mapping.

        RF-DETR pretrained checkpoints (e.g. RFDETRSegSmall) can have dataset_file='roboflow' even though they were
        trained on COCO. The fix must not depend on dataset_file value.

        Regression test for
        https://github.com/roboflow/rf-detr/issues/988
        (post-revert follow-up).
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        coco_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[18])
        coco_model.args = SimpleNamespace(num_classes=90, dataset_file="roboflow")
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == "dog", (
            f"dataset_file='roboflow' COCO pretrained: class_id=18 must map to 'dog', "
            f"got '{detections.data['class_name'][0]}'"
        )

    def test_finetuned_coco_names_uses_direct_indexing(self) -> None:
        """Fine-tuned 80-class model with COCO names must use direct 0-indexed lookup, not sparse remap.

        When num_classes == len(COCO_CLASS_NAMES) (not strictly greater), the COCO sparse-ID branch must NOT activate.
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES

        coco_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[18])
        coco_model.args = SimpleNamespace(num_classes=80, dataset_file="coco")
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == COCO_CLASS_NAMES[18], (
            f"Fine-tuned 80-class model must use direct indexing; got '{detections.data['class_name'][0]}'"
        )

    def test_custom_names_high_num_classes_no_coco_remap(self) -> None:
        """Custom class_names with num_classes>80 must NOT activate sparse COCO remap.

        Guard: a custom model with num_classes=90 but non-COCO class_names must use
        direct 0-indexed mapping (class_names != COCO_CLASS_NAMES fails the guard).
        """
        custom_names = [f"custom_{i}" for i in range(80)]
        coco_model = _DummyModel(class_names=custom_names, labels=[18])
        coco_model.args = SimpleNamespace(num_classes=90)
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == "custom_18", (
            f"Custom class names must use direct indexing; got '{detections.data['class_name'][0]}'"
        )

    def test_coco_names_without_model_args_fires_warning(self) -> None:
        """Predict() must warn when COCO class_names present but model has no 'args' attribute.

        Without args, num_logit_slots falls back to n so _is_coco_pretrained stays False. The warning is the caller's
        only signal that sparse COCO-ID mapping cannot activate, which may cause wrong class names for pretrained COCO
        checkpoints loaded without args.
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
        from rfdetr.utilities.logger import get_logger

        logger = get_logger()
        logger._warned_once.clear()

        no_args_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[0])
        # Do NOT set no_args_model.args — this is the scenario under test.
        model = _DummyRFDETR()
        model.model = no_args_model

        img = PIL.Image.new("RGB", (28, 28))
        model.predict(img)

        coco_warnings = [msg for msg in logger._warned_once if "COCO sparse-ID mapping cannot activate" in msg]
        assert coco_warnings, (
            "predict() must emit a warning when class_names matches COCO_CLASS_NAMES "
            "but model has no 'args' attribute (sparse-ID mapping cannot activate)"
        )

    def test_non_coco_names_without_model_args_no_warning_uses_direct_index(self) -> None:
        """No warning and direct indexing for non-COCO class_names when model has no 'args'.

        When model has no 'args' AND class_names != COCO_CLASS_NAMES, neither the COCO warning nor sparse-ID mapping
        activates. class_id maps directly to class_names[class_id].
        """
        from rfdetr.utilities.logger import get_logger

        logger = get_logger()
        logger._warned_once.clear()

        no_args_model = _DummyModel(class_names=["cat", "dog"], labels=[0])
        # Do NOT set no_args_model.args.
        model = _DummyRFDETR()
        model.model = no_args_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        coco_warnings = [msg for msg in logger._warned_once if "COCO" in msg]
        assert not coco_warnings, "Non-COCO class_names with no args must not emit a COCO warning"
        assert detections.data["class_name"][0] == "cat", (
            f"Direct-index mapping: class_id=0 must map to 'cat', got '{detections.data['class_name'][0]}'"
        )

    def test_coco_pretrained_oob_gap_class_id_maps_to_empty_string_and_warns(self) -> None:
        """COCO category gap ID 12 must produce empty string and OOB warning in pretrained branch.

        COCO skips category ID 12 (gap between fire hydrant=11 and stop sign=13). A pretrained model emitting cid=12 has
        no mapping in _class_id_to_name and must trigger the out-of-range warning even in the COCO-pretrained branch.
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
        from rfdetr.utilities.logger import get_logger

        logger = get_logger()
        logger._warned_once.clear()

        coco_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[12])
        coco_model.args = SimpleNamespace(num_classes=90)
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == "", "COCO gap ID 12 (no such category) must produce empty string"
        unmapped_warnings = [msg for msg in logger._warned_once if "unmapped class_id" in msg]
        assert unmapped_warnings, "COCO gap ID 12 must trigger an unmapped-class-id warning"

    def test_sparse_localized_mapping_overrides_background_sentinel(self) -> None:
        """An explicit foreground name wins over the background sentinel in a sparse localized layout."""
        model = self._make_model_with_class_names(["chat", "brosse à dents"], labels=[90])
        model.model.args = SimpleNamespace(num_classes=90)
        context = model._prediction_context()
        context.class_id_to_name.clear()
        context.class_id_to_name.update({1: "chat", 90: "brosse à dents"})

        img = PIL.Image.new("RGB", (28, 28))
        detections = core.predict(lambda: context, img)

        assert detections.data["class_name"][0] == "brosse à dents", (
            "An explicit class_id_to_name entry must override the background sentinel"
        )

    def test_coco_pretrained_class_id_90_maps_to_toothbrush_not_background(self) -> None:
        """COCO class ID 90 ('toothbrush') must not be mislabelled '__background__' in pretrained branch.

        For COCO-pretrained models num_logit_slots==90, which is also a valid COCO category (toothbrush). Background is
        implicit (below threshold), not a sentinel label. The background sentinel check must be scoped to fine-tuned
        models only.

        Regression test for HIGH-1 finding in /review of PR #1051.
        """
        from rfdetr.assets.coco_classes import COCO_CLASS_NAMES
        from rfdetr.utilities.logger import get_logger

        logger = get_logger()
        logger._warned_once.clear()

        coco_model = _DummyModel(class_names=list(COCO_CLASS_NAMES), labels=[90])
        coco_model.args = SimpleNamespace(num_classes=90)
        model = _DummyRFDETR()
        model.model = coco_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert detections.data["class_name"][0] == "toothbrush", (
            f"COCO pretrained: class_id=90 must map to 'toothbrush', got '{detections.data['class_name'][0]}'"
        )
        unmapped_warnings = [msg for msg in logger._warned_once if "unmapped class_id" in msg]
        assert not unmapped_warnings, "class_id=90 (valid COCO category) must not trigger unmapped-class-id warning"


class TestPredictKeypointClassNameMapping:
    """class_name mapping for keypoint and detection models (issue #1150).

    Active-first keypoint models use normal 0-based class IDs. Legacy background-first checkpoints use slot 0 as
    background and start real classes at slot 1; that path must keep class-name mapping compatible.
    """

    @pytest.mark.parametrize(
        "class_names,labels,num_kp_per_class,expected_class_name",
        [
            # Background-first schema (regression for https://github.com/roboflow/rf-detr/issues/1150)
            pytest.param(["person"], [1], [0, 17], "person", id="bg-first-slot-1-maps-to-person"),
            pytest.param(["person"], [0], [0, 17], "__background__", id="bg-first-slot-0-maps-to-background"),
            pytest.param(
                ["person", "bicycle"], [1], [0, 17, 4], "person", id="bg-first-multi-class-slot-1-maps-to-person"
            ),
            pytest.param(
                ["person", "bicycle"], [2], [0, 17, 4], "bicycle", id="bg-first-multi-class-slot-2-maps-to-bicycle"
            ),
            # Active-first schemas (no leading zero) — fallback path must stay correct
            pytest.param(["person"], [0], [25], "person", id="active-first-single-class-slot-0-maps-to-person"),
            pytest.param(
                ["person"], [1], [25], "__background__", id="active-first-single-class-slot-1-maps-to-background"
            ),
            pytest.param(
                ["person", "bicycle"], [0], [17, 4], "person", id="active-first-multi-class-slot-0-maps-to-person"
            ),
        ],
    )
    def test_keypoint_class_name_mapping(
        self,
        class_names: list[str],
        labels: list[int],
        num_kp_per_class: list[int],
        expected_class_name: str,
    ) -> None:
        """class_name resolved correctly for background-first and active-first keypoint schemas."""
        kp_model = _DummyModel(class_names=class_names, labels=labels, include_keypoints=True)
        kp_model.args = SimpleNamespace(num_classes=len(class_names), num_keypoints_per_class=num_kp_per_class)
        model = _DummyRFDETR()
        model.model = kp_model

        img = PIL.Image.new("RGB", (28, 28))
        key_points = model.predict(img)

        assert isinstance(key_points, sv.KeyPoints)
        assert key_points.data["class_name"][0] == expected_class_name, (
            f"schema={num_kp_per_class}, class_id={labels[0]}: "
            f"expected '{expected_class_name}', got '{key_points.data['class_name'][0]}'"
        )

    def test_detection_model_class_names_unaffected_by_keypoint_branch(self) -> None:
        """Detection models (no num_keypoints_per_class) map class_id via 0-based index, unchanged by fix."""
        det_model = _DummyModel(class_names=["cat"], labels=[0], include_keypoints=False)
        det_model.args = SimpleNamespace(num_classes=1)
        model = _DummyRFDETR()
        model.model = det_model

        img = PIL.Image.new("RGB", (28, 28))
        detections = model.predict(img)

        assert isinstance(detections, sv.Detections)
        assert detections.data["class_name"][0] == "cat", (
            f"Detection model: class_id=0 must map to 'cat', got '{detections.data['class_name'][0]}'"
        )


# ---------------------------------------------------------------------------
# Fix F — non-RGB PIL / file-path inputs are auto-converted to RGB
# ---------------------------------------------------------------------------


class TestPredictInputTypeReturnShape:
    """``predict()`` return shape is governed by input type, not runtime batch length."""

    def test_single_element_list_returns_list(self) -> None:
        """A one-element list input returns a list, not a bare ``Detections``."""
        img = PIL.Image.new("RGB", (28, 28), (128, 128, 128))
        model = _DummyRFDETR()
        result = model.predict([img])
        assert isinstance(result, list), "list input must always return a list"
        assert len(result) == 1

    def test_bare_image_returns_detections(self) -> None:
        """A single (non-list) image returns a bare ``Detections``."""
        img = PIL.Image.new("RGB", (28, 28), (128, 128, 128))
        model = _DummyRFDETR()
        result = model.predict(img)
        assert isinstance(result, sv.Detections)
