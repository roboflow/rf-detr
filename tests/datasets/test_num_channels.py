# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests that the dataset pipeline supplies the channel count the model is configured for (#1727)."""

import json
import random
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, PropertyMock

import numpy as np
import pytest
import torch
from PIL import Image
from pytest_mock import MockerFixture

from rfdetr.config import RFDETRNanoConfig, TrainConfig
from rfdetr.datasets import transforms
from rfdetr.datasets._torchvision import Resize
from rfdetr.datasets.coco import CocoDetection, make_coco_transforms, make_coco_transforms_square_div_64
from rfdetr.datasets.io_utils import decode_image, decode_image_bytes, decode_pil_image, decode_pil_image_bytes
from rfdetr.datasets.save_grids import denormalize_for_display
from rfdetr.datasets.transforms import Normalize
from rfdetr.datasets.webdataset.load import WebDatasetDetection
from rfdetr.datasets.webdataset.pack import pack_coco_to_shards
from rfdetr.datasets.yolo import YoloDetection
from rfdetr.training.module_model import RFDETRModelModule
from rfdetr.utilities.imports import _IS_WEBDATASET_INSTALLED
from rfdetr.utilities.tensors import NestedTensor

_requires_webdataset = pytest.mark.skipif(not _IS_WEBDATASET_INSTALLED, reason="webdataset not installed")

# Reading shards needs the optional ``webdataset`` package, so its helper doctest runs only when it is installed.
__doctest_requires__ = {"_webdataset_sample": ["webdataset"]}

_Decoder = Callable[[Path, int], np.ndarray]


def _pixels(mode: str) -> np.ndarray:
    """Return an 8 x 10 test image in *mode* whose last plane is zero on the left half.

    Examples:
        >>> _pixels("RGBA").shape, _pixels("L").shape
        ((8, 10, 4), (8, 10))
    """
    bands = len(Image.new(mode, (1, 1)).getbands())
    pixels = np.random.default_rng(0).integers(1, 255, (8, 10, bands), dtype=np.uint8)
    pixels[:, :5, -1] = 0
    return pixels.squeeze()


def _write_image(folder: Path, mode: str, name: str = "sample.png") -> Path:
    """Write :func:`_pixels` for *mode* in *folder*, in the format *name* ends with, and return its path.

    Examples:
        >>> import tempfile
        >>> Image.open(_write_image(Path(tempfile.mkdtemp()), "RGBA")).mode
        'RGBA'
    """
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    Image.fromarray(_pixels(mode), mode).save(path)
    return path


def _write_coco(folder: Path, mode: str) -> tuple[Path, Path]:
    """Write a one-image COCO split in *mode* and return its ``(image_dir, annotation_file)``.

    Examples:
        >>> import tempfile
        >>> image_dir, annotations = _write_coco(Path(tempfile.mkdtemp()), "RGB")
        >>> [path.name for path in image_dir.iterdir()]
        ['sample.png']
    """
    image_dir = folder / "images"
    _write_image(image_dir, mode)
    annotations = folder / "annotations.json"
    payload = {
        "images": [{"id": 1, "file_name": "sample.png", "width": 10, "height": 8}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 1, 4, 4], "area": 16, "iscrowd": 0}],
        "categories": [{"id": 1, "name": "square"}],
    }
    annotations.write_text(json.dumps(payload), encoding="utf-8")
    return image_dir, annotations


def _coco_sample(folder: Path, mode: str, num_channels: int) -> Image.Image:
    """Return the first image of a one-image ``CocoDetection`` written in *mode*.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     image = _coco_sample(Path(tempfile.mkdtemp()), "RGBA", 4)
        >>> image.mode
        'RGBA'
    """
    image_dir, annotations = _write_coco(folder, mode)
    return CocoDetection(image_dir, annotations, transforms=None, num_channels=num_channels)[0][0]


def _yolo_sample(folder: Path, mode: str, num_channels: int) -> Image.Image:
    """Return the first image of a one-image ``YoloDetection`` written in *mode*.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     image = _yolo_sample(Path(tempfile.mkdtemp()), "RGBA", 4)
        >>> image.mode
        'RGBA'
    """
    _write_image(folder / "images", mode)
    (folder / "labels").mkdir()
    (folder / "labels" / "sample.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    data_file = folder / "data.yaml"
    data_file.write_text("names:\n  - square\n", encoding="utf-8")
    dataset = YoloDetection(
        str(folder / "images"), str(folder / "labels"), str(data_file), transforms=None, num_channels=num_channels
    )
    return dataset[0][0]


def _webdataset_sample(folder: Path, mode: str, num_channels: int) -> Image.Image:
    """Return the first image of a one-image WebDataset split packed from a COCO split in *mode*.

    Examples:
        >>> import contextlib, io, tempfile
        >>> with contextlib.redirect_stdout(io.StringIO()):
        ...     image = _webdataset_sample(Path(tempfile.mkdtemp()), "RGBA", 4)
        >>> image.mode
        'RGBA'
    """
    image_dir, annotations = _write_coco(folder, mode)
    pack_coco_to_shards(image_dir, annotations, folder / "shards", split="train")
    dataset = WebDatasetDetection(folder / "shards", "train", transforms=None, num_channels=num_channels)
    return next(iter(dataset))[0]


_DECODERS = [
    pytest.param(lambda path, n: decode_image(path, num_channels=n)[0], id="decode_image"),
    pytest.param(lambda path, n: decode_image_bytes(path.read_bytes(), num_channels=n)[0], id="decode_image_bytes"),
    pytest.param(lambda path, n: np.asarray(decode_pil_image(path, num_channels=n)[0]), id="decode_pil_image"),
    pytest.param(
        lambda path, n: np.asarray(decode_pil_image_bytes(path.read_bytes(), num_channels=n)[0]),
        id="decode_pil_image_bytes",
    ),
]
_SAMPLES = [
    pytest.param(_coco_sample, id="coco"),
    pytest.param(_yolo_sample, id="yolo"),
    pytest.param(_webdataset_sample, id="webdataset", marks=_requires_webdataset),
]


class TestDecoders:
    """Every decoder returns the channels the model takes, and never invents one."""

    @pytest.mark.parametrize("decode", _DECODERS)
    def test_four_channels_keep_the_stored_fourth_plane(self, decode: _Decoder, tmp_path: Path) -> None:
        """An RGBA file reaches the caller with all four planes, including RGB where the fourth is zero."""
        path = _write_image(tmp_path, "RGBA")

        np.testing.assert_array_equal(decode(path, 4), _pixels("RGBA"))

    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA"])
    @pytest.mark.parametrize("decode", _DECODERS)
    def test_one_channel_is_grayscale(self, decode: _Decoder, mode: str, tmp_path: Path) -> None:
        """A colour or alpha-carrying file is converted to one luminance plane, as prediction converts its input."""
        path = _write_image(tmp_path, mode)

        np.testing.assert_array_equal(decode(path, 1), np.asarray(Image.open(path).convert("L")))

    @pytest.mark.parametrize("decode", _DECODERS)
    def test_cmyk_file_converts_to_rgb_and_grayscale(self, decode: _Decoder, tmp_path: Path) -> None:
        """A CMYK file is a colour file, so the 3- and 1-channel models read it as Pillow converts it."""
        path = _write_image(tmp_path, "CMYK", "sample.tiff")

        np.testing.assert_array_equal(decode(path, 3), np.asarray(Image.open(path).convert("RGB")))
        np.testing.assert_array_equal(decode(path, 1), np.asarray(Image.open(path).convert("L")))

    @pytest.mark.parametrize("decode", _DECODERS)
    def test_la_file_supplies_its_alpha_as_fourth_plane(self, decode: _Decoder, tmp_path: Path) -> None:
        """A grayscale file with alpha stores a real fourth plane: luminance in RGB, alpha kept as stored."""
        path = _write_image(tmp_path, "LA")
        luminance, alpha = _pixels("LA")[..., 0], _pixels("LA")[..., 1]

        np.testing.assert_array_equal(decode(path, 4), np.dstack([luminance, luminance, luminance, alpha]))

    @pytest.mark.parametrize(
        ("mode", "name"),
        [("RGB", "sample.png"), ("L", "sample.png"), ("CMYK", "sample.tiff"), ("RGB", "sample.jpg")],
    )
    @pytest.mark.parametrize("decode", _DECODERS)
    def test_file_without_fourth_channel_is_refused_for_four_channels(
        self, decode: _Decoder, mode: str, name: str, tmp_path: Path
    ) -> None:
        """A file without a fourth channel raises instead of training on three channels or a made-up fourth."""
        path = _write_image(tmp_path, mode, name)

        with pytest.raises(ValueError, match="has no fourth channel"):
            decode(path, 4)

    def test_error_names_the_file(self, tmp_path: Path) -> None:
        """The refusal says which file lacks the channel."""
        path = _write_image(tmp_path, "RGB")

        with pytest.raises(ValueError, match="sample.png"):
            decode_pil_image(path, num_channels=4)

    def test_palette_transparency_counts_as_fourth_channel(self, tmp_path: Path) -> None:
        """A palette PNG with transparency stores an alpha value per pixel, so it is accepted."""
        path = tmp_path / "palette.png"
        Image.fromarray(_pixels("RGBA"), "RGBA").convert("P").save(path, transparency=0)

        image, _ = decode_pil_image(path, num_channels=4)

        assert image.mode == "RGBA"

    @pytest.mark.parametrize("num_channels", [2, 5])
    def test_channel_count_files_cannot_supply_is_refused(self, num_channels: int, tmp_path: Path) -> None:
        """Only 1, 3 and 4 channels can come from an image file."""
        path = _write_image(tmp_path, "RGBA")

        with pytest.raises(ValueError, match="cannot be read from image files"):
            decode_pil_image(path, num_channels=num_channels)

    def test_raised_pixel_limit_keeps_four_channels(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Pillow worker process, used for a raised pixel limit, applies the same conversion."""
        path = _write_image(tmp_path, "RGBA")
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)

        image, _ = decode_pil_image(path, max_image_pixels=5000, num_channels=4)

        np.testing.assert_array_equal(np.asarray(image), _pixels("RGBA"))

    def test_raised_pixel_limit_refuses_rgb_for_four_channels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The worker's refusal comes back as the same ``ValueError``."""
        path = _write_image(tmp_path, "RGB")
        monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 50)

        with pytest.raises(ValueError, match="has no fourth channel"):
            decode_pil_image(path, max_image_pixels=5000, num_channels=4)


class TestDatasetFormats:
    """COCO, YOLO and WebDataset datasets load images for the configured channel count."""

    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_four_channel_dataset_returns_rgba(self, sample: Callable[..., Image.Image], tmp_path: Path) -> None:
        """The stored fourth plane reaches the transform pipeline."""
        np.testing.assert_array_equal(np.asarray(sample(tmp_path, "RGBA", 4)), _pixels("RGBA"))

    @pytest.mark.parametrize("mode", ["RGB", "RGBA", "LA"])
    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_one_channel_dataset_returns_grayscale(
        self, sample: Callable[..., Image.Image], mode: str, tmp_path: Path
    ) -> None:
        """A grayscale model gets one plane from colour and alpha-carrying files."""
        assert sample(tmp_path, mode, 1).mode == "L"

    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_four_channel_dataset_keeps_la_alpha(self, sample: Callable[..., Image.Image], tmp_path: Path) -> None:
        """A grayscale-with-alpha file reaches a 4-channel model with its alpha as the fourth plane."""
        image = np.asarray(sample(tmp_path, "LA", 4))

        np.testing.assert_array_equal(image[..., 3], _pixels("LA")[..., 1])

    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_default_dataset_returns_rgb(self, sample: Callable[..., Image.Image], tmp_path: Path) -> None:
        """A 3-channel model keeps getting RGB from RGBA files."""
        assert sample(tmp_path, "RGBA", 3).mode == "RGB"

    @pytest.mark.parametrize("mode", ["RGB", "L"])
    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_files_without_fourth_channel_are_refused_for_four_channels(
        self, sample: Callable[..., Image.Image], mode: str, tmp_path: Path
    ) -> None:
        """A 4-channel model on RGB or grayscale files fails loudly at the first sample."""
        with pytest.raises(ValueError, match="has no fourth channel"):
            sample(tmp_path, mode, 4)

    @pytest.mark.parametrize("sample", _SAMPLES)
    def test_unsupported_channel_count_is_refused_at_build(
        self, sample: Callable[..., Image.Image], tmp_path: Path
    ) -> None:
        """A count image files cannot supply fails when the dataset is built."""
        with pytest.raises(ValueError, match="num_channels=5 cannot be read from image files"):
            sample(tmp_path, "RGBA", 5)


class TestNormalize:
    """The training ``Normalize`` handles any channel count inside the one transform that also converts boxes."""

    def test_fourth_channel_reuses_the_first_statistics(self) -> None:
        """Training normalizes the fourth channel with the value prediction uses for it."""
        image, _ = Normalize()(torch.full((4, 2, 2), 0.5), None)

        torch.testing.assert_close(image[3], image[0])

    def test_single_channel_uses_the_first_statistics(self) -> None:
        """A grayscale image is normalized with the first mean and standard deviation."""
        image, _ = Normalize()(torch.full((1, 2, 2), 0.485), None)

        torch.testing.assert_close(image, torch.zeros(1, 2, 2), atol=1e-6, rtol=0)

    def test_normalizer_for_another_channel_count_is_built_once(self, mocker: MockerFixture) -> None:
        """Every 4-channel sample reuses one normalizer."""
        normalize = Normalize()
        build = mocker.spy(transforms, "_TVNormalize")

        normalize(torch.zeros(4, 8, 8))
        normalize(torch.zeros(4, 8, 8))

        assert build.call_count == 1

    def test_boxes_are_still_converted_for_four_channels(self) -> None:
        """The box conversion to relative cxcywh happens for a 4-channel image too."""
        target = {"boxes": torch.tensor([[0.0, 0.0, 4.0, 2.0]])}

        _, normalized = Normalize()(torch.zeros(4, 4, 8), target)

        torch.testing.assert_close(normalized["boxes"], torch.tensor([[0.25, 0.25, 0.5, 0.5]]))


class TestResize:
    """``Resize`` treats the fourth plane as data, not as transparency."""

    def test_color_survives_where_the_fourth_plane_is_zero(self) -> None:
        """Pillow's own RGBA resize would blank the color planes there."""
        pixels = np.full((8, 8, 4), 200, dtype=np.uint8)
        pixels[..., 3] = 0

        resized, _ = Resize((4, 4))(Image.fromarray(pixels, "RGBA"), None)

        np.testing.assert_array_equal(np.asarray(resized)[..., :3], np.full((4, 4, 3), 200, dtype=np.uint8))

    @pytest.mark.parametrize(
        "build",
        [
            pytest.param(make_coco_transforms, id="aspect"),
            pytest.param(make_coco_transforms_square_div_64, id="square"),
        ],
    )
    @pytest.mark.parametrize("image_set", ["val", "test", "train"])
    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_evaluation_pipeline_keeps_color_where_the_fourth_plane_is_zero(
        self, build: Callable[..., Any], image_set: str, seed: int
    ) -> None:
        """A zero in the fourth plane, a natural "no data" value, leaves the RGB intact, augmented or not."""
        random.seed(seed)
        torch.manual_seed(seed)
        pixels = np.full((60, 80, 4), 200, dtype=np.uint8)
        pixels[20:40, 20:60, 3] = 0
        target = {"boxes": torch.tensor([[20.0, 20.0, 60.0, 40.0]]), "labels": torch.tensor([1])}

        image, _ = build(image_set, 128, aug_config=None)(Image.fromarray(pixels, "RGBA"), target)

        expected = image[:3, :1, :1].expand_as(image[:3])
        assert image.shape[0] == 4
        torch.testing.assert_close(image[:3], expected)

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_train_pipeline_keeps_one_channel_for_grayscale(self, seed: int) -> None:
        """A grayscale image leaves the augmenting train pipeline as one normalized plane."""
        random.seed(seed)
        torch.manual_seed(seed)
        pixels = np.full((60, 80), 200, dtype=np.uint8)
        target = {"boxes": torch.tensor([[20.0, 20.0, 60.0, 40.0]]), "labels": torch.tensor([1])}

        image, _ = make_coco_transforms("train", 128, aug_config=None)(Image.fromarray(pixels, "L"), target)

        assert image.shape[0] == 1
        assert torch.isfinite(image).all()


class TestTrainingStep:
    """A model configured for another channel count trains on batches that carry it."""

    @pytest.mark.parametrize("num_channels", [1, 4])
    def test_training_step_returns_finite_loss_with_gradient_in_the_stem(
        self, num_channels: int, tmp_path: Path, mocker: MockerFixture
    ) -> None:
        """A Nano model fed a 1- or 4-channel batch produces a loss that reaches the widened stem."""
        torch.manual_seed(0)
        model_config = RFDETRNanoConfig(num_channels=num_channels, pretrain_weights=None, device="cpu")
        train_config = TrainConfig(dataset_dir=str(tmp_path), output_dir=str(tmp_path), tensorboard=False)
        module = RFDETRModelModule(model_config, train_config).train()
        module.log = MagicMock()
        module.log_dict = MagicMock()
        mocker.patch.object(
            type(module), "trainer", new_callable=PropertyMock, return_value=SimpleNamespace(accumulate_grad_batches=1)
        )
        resolution = model_config.resolution
        samples = NestedTensor(
            torch.randn(2, num_channels, resolution, resolution), torch.zeros(2, resolution, resolution, dtype=torch.bool)
        )
        targets = [
            {
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.2]]),
                "labels": torch.tensor([1]),
                "image_id": torch.tensor(index),
                "orig_size": torch.tensor([resolution, resolution]),
            }
            for index in range(2)
        ]

        loss = module.training_step((samples, targets), batch_idx=0)
        loss.backward()

        stem = module.model.backbone[0].encoder.encoder.embeddings.patch_embeddings.projection
        assert torch.isfinite(loss)
        assert stem.weight.shape[1] == num_channels
        assert stem.weight.grad is not None
        assert stem.weight.grad.abs().sum() > 0


class TestDenormalizeForDisplay:
    """Sample grids render an RGB picture for any channel count."""

    @pytest.mark.parametrize("channels", [1, 3, 4])
    def test_returns_rgb_picture(self, channels: int) -> None:
        """One, three and four channel images all become ``(H, W, 3)`` pixels."""
        assert denormalize_for_display(torch.zeros(channels, 2, 5)).shape == (2, 5, 3)

    def test_inverts_training_normalization(self) -> None:
        """A normalized 4-channel image comes back as its first three planes."""
        image = torch.rand(4, 3, 3)
        normalized, _ = Normalize()(image, None)

        picture = denormalize_for_display(normalized)

        np.testing.assert_allclose(picture, image[:3].permute(1, 2, 0).numpy() * 255, atol=1.0)
