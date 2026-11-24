# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""End-to-end tests for a model built with a ``(height, width)`` resolution."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F  # noqa: N812

from rfdetr import RFDETR, RFDETRNano
from rfdetr.utilities.box_ops import box_cxcywh_to_xyxy
from rfdetr.utilities.state_dict import strip_checkpoint

_PE_KEY = "backbone.0.encoder.encoder.embeddings.position_embeddings"
_RESOLUTION = (320, 448)


def _build(resolution: int | tuple[int, int], pretrain_weights: str | None = None) -> RFDETRNano:
    """Return a one-class Nano model on the CPU, randomly initialized unless ``pretrain_weights`` is given."""
    torch.manual_seed(0)
    return RFDETRNano(pretrain_weights=pretrain_weights, num_classes=1, device="cpu", resolution=resolution)


def _image(height: int = 90, width: int = 150) -> np.ndarray:
    """Return a fixed random ``(height, width, 3)`` uint8 image."""
    return (np.random.default_rng(0).random((height, width, 3)) * 255).astype(np.uint8)


@pytest.fixture(scope="module")
def non_square_model() -> RFDETRNano:
    """A Nano model at 320x448, shared by the tests that only read it."""
    return _build(_RESOLUTION)


class TestNonSquareModel:
    """A model built at ``(height, width)`` runs, predicts and exports its defaults at that shape."""

    def test_config_and_context_carry_height_width(self, non_square_model: RFDETRNano) -> None:
        """The pair reaches the config, the model context and the prediction defaults in ``(height, width)`` order."""
        assert non_square_model.model_config.resolution == _RESOLUTION
        assert non_square_model.model_config.positional_encoding_size == (20, 28)
        assert non_square_model.model.resolution == _RESOLUTION
        assert non_square_model._prediction_context().default_shape == _RESOLUTION
        assert non_square_model._runtime_info()["default_shape"] == _RESOLUTION

    def test_position_embeddings_lie_on_the_non_square_grid(self, non_square_model: RFDETRNano) -> None:
        """The backbone stores one position embedding per patch of the 20x28 grid."""
        embeddings = non_square_model.model.model.backbone[0].encoder.encoder.embeddings

        assert embeddings.patch_grid_size == (20, 28)
        assert embeddings.position_embeddings.shape[1] == 20 * 28 + 1

    def test_forward_uses_the_stored_position_embeddings(
        self, non_square_model: RFDETRNano, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A forward at the native shape adds the stored table: nothing is interpolated."""
        monkeypatch.setattr(
            F, "interpolate", lambda *args, **kwargs: pytest.fail("position embeddings were interpolated")
        )
        embeddings = non_square_model.model.model.backbone[0].encoder.encoder.embeddings
        tokens = torch.zeros(1, 20 * 28 + 1, embeddings.position_embeddings.shape[-1])

        assert embeddings.interpolate_pos_encoding(tokens, *_RESOLUTION) is embeddings.position_embeddings

    def test_forward_returns_one_prediction_per_query(self, non_square_model: RFDETRNano) -> None:
        """The network runs on a ``(height, width)`` batch."""
        network = non_square_model.model.model.eval()

        with torch.no_grad():
            outputs = network(torch.rand(2, 3, *_RESOLUTION))

        assert outputs["pred_boxes"].shape == (2, non_square_model.model_config.num_queries, 4)
        assert torch.isfinite(outputs["pred_boxes"]).all()

    def test_predict_returns_boxes_in_source_image_pixels(self, non_square_model: RFDETRNano) -> None:
        """``predict()`` resizes to ``(height, width)`` and scales boxes back by the source width and height."""
        image = _image()
        network = non_square_model.model.model.eval()
        pixels = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        pixels = F.interpolate(pixels[None], size=_RESOLUTION, mode="bilinear", align_corners=False, antialias=True)
        mean = torch.tensor(non_square_model.means).view(1, 3, 1, 1)
        std = torch.tensor(non_square_model.stds).view(1, 3, 1, 1)
        with torch.no_grad():
            outputs = network((pixels - mean) / std)
        scores = outputs["pred_logits"].sigmoid().flatten(1)
        best = scores.argmax(dim=1) // outputs["pred_logits"].shape[2]
        expected = box_cxcywh_to_xyxy(outputs["pred_boxes"][0, best]) * torch.tensor([150.0, 90.0, 150.0, 90.0])

        detections = non_square_model.predict(image, threshold=0.0)

        top = int(np.argmax(detections.confidence))
        np.testing.assert_allclose(detections.xyxy[top], expected[0].numpy(), rtol=1e-3, atol=0.5)

    def test_inference_predictor_matches_predict(self, non_square_model: RFDETRNano) -> None:
        """An eager predictor from ``inference(compile=False)`` returns the boxes ``predict()`` returns."""
        image = _image()
        expected = non_square_model.predict(image, threshold=0.0)

        predictor = non_square_model.inference(compile=False)
        detections = predictor.predict(image, threshold=0.0)

        assert predictor.runtime_info["input_shape"] == (None, 3, *_RESOLUTION)
        np.testing.assert_allclose(detections.xyxy, expected.xyxy, rtol=1e-4, atol=1e-2)

    def test_inference_predictor_rejects_the_transposed_shape(self, non_square_model: RFDETRNano) -> None:
        """A predictor fixed at 320x448 refuses a 448x320 input instead of running it."""
        predictor = non_square_model.inference(compile=False)

        with pytest.raises(ValueError, match="optimized for resolution 320x448, but got 448x320"):
            predictor.predict(_image(), threshold=0.0, shape=(448, 320))

    def test_shift_tta_runs_each_pass_on_a_height_width_canvas(self, non_square_model: RFDETRNano) -> None:
        """``shift_tta`` pads the 320x448 input by one 32-pixel block per side pair and keeps boxes in the image."""
        network = non_square_model.model.model
        shapes: list[tuple[int, int]] = []
        hook = network.register_forward_pre_hook(lambda module, args: shapes.append(tuple(args[0].shape[-2:])))

        try:
            detections = non_square_model.predict(_image(), threshold=0.0, shift_tta=2)
        finally:
            hook.remove()

        assert shapes == [(352, 480), (352, 480)]
        assert len(detections) > 0
        assert np.isfinite(detections.xyxy).all()
        assert detections.xyxy[:, [0, 2]].max() <= 150 + 1e-3
        assert detections.xyxy[:, [1, 3]].max() <= 90 + 1e-3

    def test_traced_predictor_runs_at_height_width(self) -> None:
        """A traced predictor is traced with a ``(height, width)`` example and runs at that shape."""
        model = _build(_RESOLUTION)

        predictor = model.inference(compile=True, compile_backend="torchscript", batch_size=1)
        detections = predictor.predict(_image(), threshold=0.0)

        assert predictor.runtime_info["input_shape"] == (1, 3, *_RESOLUTION)
        assert detections.xyxy.shape[1] == 4


class TestTransposedResolution:
    """``(height, width)`` and ``(width, height)`` are different models."""

    def test_transposed_model_has_the_transposed_grid(self) -> None:
        """A 448x320 model has a 28x20 grid, which has the token count of 20x28 and another layout."""
        model = _build((448, 320))
        embeddings = model.model.model.backbone[0].encoder.encoder.embeddings

        assert model.model_config.positional_encoding_size == (28, 20)
        assert embeddings.patch_grid_size == (28, 20)
        assert model._prediction_context().default_shape == (448, 320)

    def test_transposed_input_is_interpolated_not_served_from_the_stored_table(
        self, non_square_model: RFDETRNano
    ) -> None:
        """The 20x28 model interpolates its table for a 448x320 input although the token counts match."""
        embeddings = non_square_model.model.model.backbone[0].encoder.encoder.embeddings
        tokens = torch.zeros(1, 28 * 20 + 1, embeddings.position_embeddings.shape[-1])

        position = embeddings.interpolate_pos_encoding(tokens, 448, 320)

        assert position is not embeddings.position_embeddings
        assert not torch.equal(position, embeddings.position_embeddings)

    def test_weights_load_into_the_transposed_model_through_interpolation(self, tmp_path: Path) -> None:
        """A 320x448 checkpoint that records its grid is resized, not copied, into a 448x320 model."""
        source = _build(_RESOLUTION)
        source_pe = source.model.model.state_dict()[_PE_KEY]
        checkpoint_path = tmp_path / "non_square.pth"
        torch.save(
            {
                "model": source.model.model.state_dict(),
                "args": {"class_names": ["object"]},
                "model_config": source.model_config.model_dump(),
            },
            checkpoint_path,
        )

        loaded = _build((448, 320), pretrain_weights=str(checkpoint_path))

        loaded_pe = loaded.model.model.state_dict()[_PE_KEY]
        patches = source_pe[:, 1:].reshape(1, 20, 28, -1).permute(0, 3, 1, 2)
        expected = F.interpolate(patches, size=(28, 20), mode="bicubic", align_corners=False, antialias=True)
        torch.testing.assert_close(loaded_pe[:, 1:], expected.permute(0, 2, 3, 1).reshape(1, 560, -1))
        assert not torch.equal(loaded_pe, source_pe)


class TestFromCheckpointNonSquare:
    """A checkpoint of a non-square model reloads as the model it was saved from."""

    def test_from_checkpoint_restores_the_resolution_and_the_predictions(self, tmp_path: Path) -> None:
        """The pair survives the checkpoint's ``model_config`` and ``strip_checkpoint``; boxes are unchanged."""
        model = _build(_RESOLUTION)
        path = tmp_path / "checkpoint_best_total.pth"
        # The payload BestModelCallback writes before on_fit_end strips it into checkpoint_best_total.pth.
        torch.save(
            {
                "model": model.model.model.state_dict(),
                "args": {"class_names": ["object"]},
                "model_name": "RFDETRNano",
                "model_config": model.model_config.model_dump(),
                "optimizer_states": [],
            },
            path,
        )
        strip_checkpoint(path, extra_metadata={"best_total_source": "ema"})

        loaded = RFDETR.from_checkpoint(path, device="cpu")

        assert loaded.model_config.resolution == _RESOLUTION
        assert loaded.model_config.positional_encoding_size == (20, 28)
        assert torch.equal(loaded.model.model.state_dict()[_PE_KEY], model.model.model.state_dict()[_PE_KEY])
        expected = model.predict(_image(), threshold=0.0)
        actual = loaded.predict(_image(), threshold=0.0)
        np.testing.assert_allclose(actual.xyxy, expected.xyxy, atol=1e-4)


class TestSquareCheckpointIntoNonSquareModel:
    """Weights trained at a square resolution load into a non-square model."""

    def test_square_state_dict_is_interpolated_to_the_non_square_grid(self, tmp_path: Path) -> None:
        """The 24x24 position embeddings of a default Nano are resized to 20x28; other weights load unchanged."""
        square = _build(384)
        square_state = square.model.model.state_dict()
        assert square_state[_PE_KEY].shape[1] == 24 * 24 + 1
        checkpoint_path = tmp_path / "square.pth"
        torch.save({"model": square_state, "args": {"class_names": ["object"]}}, checkpoint_path)

        loaded = _build(_RESOLUTION, pretrain_weights=str(checkpoint_path))

        loaded_state = loaded.model.model.state_dict()
        patches = square_state[_PE_KEY][:, 1:].reshape(1, 24, 24, -1).permute(0, 3, 1, 2)
        expected = F.interpolate(patches, size=(20, 28), mode="bicubic", align_corners=False, antialias=True)
        torch.testing.assert_close(loaded_state[_PE_KEY][:, 1:], expected.permute(0, 2, 3, 1).reshape(1, 560, -1))
        torch.testing.assert_close(loaded_state[_PE_KEY][:, 0], square_state[_PE_KEY][:, 0])
        projection_key = "backbone.0.encoder.encoder.embeddings.patch_embeddings.projection.weight"
        assert torch.equal(loaded_state[projection_key], square_state[projection_key])
        assert loaded.predict(_image(), threshold=0.0).xyxy.shape[1] == 4


class TestEvaluateNonSquare:
    """``evaluate()`` runs a non-square model, at its own shape or at an overriding pair."""

    @pytest.mark.parametrize(
        "override",
        [pytest.param({}, id="model-resolution"), pytest.param({"resolution": (448, 320)}, id="transposed-override")],
    )
    def test_evaluate_returns_metrics_and_leaves_the_model_unchanged(
        self, synthetic_shape_dataset_dir: Path, tmp_path: Path, override: dict[str, tuple[int, int]]
    ) -> None:
        """Evaluation builds its copy on the requested grid; the live model keeps its resolution and weights."""
        annotations = json.loads((synthetic_shape_dataset_dir / "train" / "_annotations.coco.json").read_text())
        torch.manual_seed(0)
        model = RFDETRNano(
            pretrain_weights=None, num_classes=len(annotations["categories"]), device="cpu", resolution=_RESOLUTION
        )
        pe_before = model.model.model.state_dict()[_PE_KEY].clone()

        metrics = model.evaluate(
            dataset_dir=str(synthetic_shape_dataset_dir),
            split="test",
            device="cpu",
            output_dir=str(tmp_path),
            batch_size=4,
            num_workers=0,
            tensorboard=False,
            **override,
        )

        assert "test/mAP_50_95" in metrics
        assert model.model_config.resolution == _RESOLUTION
        assert model.model_config.positional_encoding_size == (20, 28)
        assert model.model.resolution == _RESOLUTION
        assert torch.equal(model.model.model.state_dict()[_PE_KEY], pe_before)


class TestSquareResolutionUnchanged:
    """An ``int`` resolution keeps its ``int`` form everywhere it is stored."""

    def test_int_resolution_stays_int(self) -> None:
        """Config, context and position-embedding size stay ints, and the defaults are the square pair."""
        model = _build(320)

        assert type(model.model_config.resolution) is int
        assert type(model.model_config.positional_encoding_size) is int
        assert type(model.model.resolution) is int
        assert model._prediction_context().default_shape == (320, 320)
        assert model._runtime_info()["input_shape"] == (None, 3, None, None)

    def test_square_pair_builds_the_same_weights_as_the_int(self) -> None:
        """``resolution=(320, 320)`` initializes the same weights as ``resolution=320`` for one seed."""
        from_int = _build(320).model.model.state_dict()
        from_pair = _build((320, 320)).model.model.state_dict()

        assert from_int.keys() == from_pair.keys()
        for name, expected in from_int.items():
            assert torch.equal(from_pair[name], expected), name
