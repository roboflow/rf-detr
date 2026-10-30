# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for how ``predict()`` prepares images: conversion, range checks, resize and device transfer."""

import struct
import zlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import PIL.Image
import pytest
import requests
import supervision as sv
import torch
import torchvision.transforms.functional as F  # noqa: N812

from rfdetr import RFDETRInference
from rfdetr._prediction import core
from rfdetr.detr import RFDETR

from ..helpers import _DummyRFDETR, _FakeResponse, _png_bytes, _record_model_inputs


class TestPredictImagePinning:
    """``predict()`` should pin CPU-resident image tensors before an accelerator transfer.

    A pageable-memory ``.to(device)`` copy onto CUDA is meaningfully slower than a pinned-memory one because the CUDA
    driver has to pin the source buffer itself first. But ``Tensor.pin_memory()`` only accepts CPU tensors: a caller who
    already placed the input tensor on the model's own accelerator (a legitimate use of the tensor-input path, e.g. to
    skip a host round-trip) must never have that tensor routed through ``pin_memory()``, and a CPU-only target device
    gets no benefit from pinning at all.
    """

    def test_cpu_tensor_inputs_transfer_pinned_buffers_to_cuda_non_blocking(self) -> None:
        """Every CPU input transfers its own pinned buffer to CUDA and reaches the model boundary.

        The pinning call returns a different tensor, so this catches the regression where ``predict()`` pins an input
        but transfers the original pageable tensor instead. CUDA allocation is intercepted so this dispatch contract
        also runs in CPU-only CI; the separate value-parity test exercises the real CUDA transfer.
        """
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda", 0)
        source_tensors = [torch.zeros(3, 48, 64), torch.ones(3, 48, 64)]
        pinned_tensors = [
            torch.full_like(source_tensors[0], 0.25),
            torch.full_like(source_tensors[1], 0.75),
        ]
        transferred_tensors = [
            torch.full_like(source_tensors[0], 0.125),
            torch.full_like(source_tensors[1], 0.875),
        ]
        real_to = torch.Tensor.to
        real_tensor = torch.tensor
        pinned_inputs: list[torch.Tensor] = []
        cuda_transfer_sources: list[torch.Tensor] = []
        model_inputs: list[torch.Tensor] = []

        def pin_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Return the distinct pseudo-pinned buffer paired with a source tensor.

            Examples:
                This test-local closure requires its enclosing tensors.
                >>> pin_spy(source_tensors[0])  # doctest: +SKIP
            """
            source_index = next(
                (index for index, source_tensor in enumerate(source_tensors) if self is source_tensor),
                None,
            )
            assert source_index is not None, "pin_memory() must be called on an original input tensor"
            pinned_inputs.append(self)
            return pinned_tensors[source_index]

        def to_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Return a distinct pseudo-CUDA buffer for each pinned transfer source.

            Examples:
                This test-local closure requires its enclosing tensors.
                >>> to_spy(pinned_tensors[0], torch.device("cuda"), non_blocking=True)  # doctest: +SKIP
            """
            if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
                cuda_transfer_sources.append(self)
                transfer_index = next(
                    (index for index, pinned_tensor in enumerate(pinned_tensors) if self is pinned_tensor),
                    None,
                )
                if transfer_index is not None:
                    assert kwargs.get("non_blocking") is True
                    return transferred_tensors[transfer_index]
                return self
            return real_to(self, *args, **kwargs)

        def tensor_spy(data: object, **kwargs: object) -> torch.Tensor:
            """Redirect CUDA target-size construction to CPU for the simulated transfer.

            Examples:
                This test-local closure requires the captured real tensor constructor.
                >>> tensor_spy([[48, 64]], device=torch.device("cuda"))  # doctest: +SKIP
            """
            device = kwargs.get("device")
            if isinstance(device, torch.device) and device.type == "cuda":
                kwargs["device"] = torch.device("cpu")
            return real_tensor(data, **kwargs)

        model_module = model.model.model
        assert model_module is not None
        with model_module.register_forward_pre_hook(_record_model_inputs(model_inputs)):
            with (
                patch.object(torch.Tensor, "pin_memory", pin_spy),
                patch.object(torch.Tensor, "to", to_spy),
                patch.object(torch, "tensor", tensor_spy),
            ):
                model.predict(source_tensors)

        assert len(pinned_inputs) == len(source_tensors)
        assert all(pinned_input is source_tensor for pinned_input, source_tensor in zip(pinned_inputs, source_tensors))
        assert len(cuda_transfer_sources) == len(pinned_tensors)
        assert all(
            cuda_transfer_source is pinned_tensor
            for cuda_transfer_source, pinned_tensor in zip(cuda_transfer_sources, pinned_tensors)
        )
        assert len(model_inputs) == 1
        captured_batch = model_inputs[0].cpu()
        expected_batch = torch.stack(
            [
                (
                    torch.full((3, model.model.resolution, model.model.resolution), value)
                    - torch.tensor(model.means)[:, None, None]
                )
                / torch.tensor(model.stds)[:, None, None]
                for value in (0.125, 0.875)
            ]
        )
        torch.testing.assert_close(captured_batch, expected_batch)

    @pytest.mark.gpu
    def test_pinned_non_blocking_transfer_preserves_values(self) -> None:
        """Pinning + non_blocking must not change the transferred values versus a plain, blocking .to()."""
        torch.manual_seed(0)
        source = torch.rand(3, 48, 64)
        device = torch.device("cuda", 0)

        plain = source.to(device)
        pinned = source.pin_memory().to(device, non_blocking=True)
        torch.cuda.synchronize()

        assert torch.equal(plain, pinned)

    @pytest.mark.gpu
    def test_already_cuda_tensor_input_is_not_pinned(self) -> None:
        """A tensor the caller already placed on the accelerator must never be routed through pin_memory()."""
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda", 0)
        tensor = torch.rand(3, 48, 64, device="cuda")
        pin_spy = MagicMock(side_effect=AssertionError("pin_memory() must not be called on a CUDA tensor"))

        with patch.object(torch.Tensor, "pin_memory", pin_spy):
            model.predict(tensor)  # must not raise: pin_memory() on a CUDA tensor would itself raise RuntimeError

        pin_spy.assert_not_called()

    def test_cpu_target_device_does_not_pin(self) -> None:
        """A CPU-only target device gets no benefit from pinning, so it must be skipped entirely."""
        model = _DummyRFDETR()  # _DummyModel defaults to a CPU device
        tensor = torch.rand(3, 48, 64)
        pin_spy = MagicMock(side_effect=AssertionError("pin_memory() must not be called for a CPU target device"))

        with patch.object(torch.Tensor, "pin_memory", pin_spy):
            model.predict(tensor)

        pin_spy.assert_not_called()

    @pytest.mark.gpu
    def test_cuda_tensor_input_to_cpu_model_is_not_non_blocking(self) -> None:
        """A tensor already on CUDA moving to a CPU-device model must use a blocking transfer.

        ``non_blocking=True`` only pays off, and is only safe without an explicit sync, when the destination is CUDA —
        matching ``transfer_batch_to_device()`` in ``training/module_data.py``. Here the ``.to()`` call allocates a
        fresh, unpinned CPU tensor as its destination, so an async D2H copy could leave that tensor holding an in-flight
        (partially written) result if read before the copy stream drains.
        """
        model = _DummyRFDETR()  # _DummyModel defaults to a CPU device
        tensor = torch.rand(3, 48, 64, device="cuda")
        real_to = torch.Tensor.to
        captured: list[bool] = []

        def to_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            if self is tensor:
                captured.append(bool(kwargs.get("non_blocking", False)))
            return real_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", to_spy):
            model.predict(tensor)

        assert captured, "expected predict() to move the image tensor with .to()"
        assert not any(captured), "CUDA tensor -> CPU-model transfer must not set non_blocking=True"


class TestPredictUint8Conversion:
    """The fused uint8 path must retain torchvision's exact conversion semantics."""

    @pytest.mark.parametrize(
        ("mode", "image_format", "save_options"),
        [
            pytest.param("RGB", "JPEG", {}, id="jpeg-rgb"),
            pytest.param("L", "JPEG", {}, id="jpeg-grayscale"),
            pytest.param("CMYK", "JPEG", {}, id="jpeg-cmyk"),
            pytest.param("RGBA", "PNG", {}, id="png-rgba"),
            pytest.param("P", "PNG", {}, id="png-palette"),
            pytest.param("1", "PNG", {}, id="png-1-bit"),
            # ``bits`` makes Pillow write a 2- or 4-bit palette; the expected pixels are read back from that file.
            pytest.param("P", "PNG", {"bits": 2}, id="png-2-bit-palette"),
            pytest.param("P", "PNG", {"bits": 4}, id="png-4-bit-palette"),
            pytest.param("P", "PNG", {"transparency": 0}, id="png-palette-trns"),
            pytest.param("LA", "PNG", {}, id="png-grayscale-alpha"),
        ],
    )
    def test_local_image_decoder_matches_pillow(
        self, tmp_path: Path, mode: str, image_format: str, save_options: dict[str, int]
    ) -> None:
        """The fast local-file decoder must produce Pillow's exact RGB bytes for supported formats."""
        image_path = tmp_path / f"image.{image_format.lower()}"
        rgb = ((np.arange(17 * 29 * 3).reshape(17, 29, 3) * 37 + 13) % 256).astype(np.uint8)
        image = PIL.Image.fromarray(rgb).convert(mode)
        if mode == "RGBA":
            alpha = ((np.arange(17 * 29).reshape(17, 29) * 19 + 7) % 256).astype(np.uint8)
            image.putalpha(PIL.Image.fromarray(alpha))
        if mode == "CMYK":
            # ``convert("CMYK")`` always writes K=0, which never exercises the black channel of the CMYK-to-RGB step.
            black = ((np.arange(17 * 29).reshape(17, 29) * 19 + 7) % 256).astype(np.uint8)
            image = PIL.Image.merge("CMYK", (*image.split()[:3], PIL.Image.fromarray(black)))
        image.save(image_path, format=image_format, **save_options)
        with PIL.Image.open(image_path) as image:
            expected = np.array(image.convert("RGB"))

        decoded = core._decode_local_image(str(image_path))

        assert decoded is not None
        assert decoded.dtype is torch.uint8
        np.testing.assert_array_equal(decoded.permute(1, 2, 0).numpy(), expected)

    @pytest.mark.parametrize("filename", ["PHOTO.JPG", "PHOTO.JPEG", "PHOTO.PNG"])
    def test_local_image_decoder_accepts_upper_case_extensions(self, tmp_path: Path, filename: str) -> None:
        """Cameras and Windows tools write upper-case extensions, and those files must not lose the native route."""
        image_path = tmp_path / filename
        rgb = ((np.arange(17 * 29 * 3).reshape(17, 29, 3) * 37 + 13) % 256).astype(np.uint8)
        PIL.Image.fromarray(rgb).save(image_path)
        with PIL.Image.open(image_path) as image:
            expected = np.array(image.convert("RGB"))

        decoded = core._decode_local_image(str(image_path))

        assert decoded is not None
        np.testing.assert_array_equal(decoded.permute(1, 2, 0).numpy(), expected)

    def test_local_image_decoder_ignores_exif_orientation_like_pillow(self, tmp_path: Path) -> None:
        """Pillow's conversion never rotates by EXIF orientation, so the native route must not either."""
        image_path = tmp_path / "rotated.jpg"
        rgb = ((np.arange(17 * 29 * 3).reshape(17, 29, 3) * 37 + 13) % 256).astype(np.uint8)
        exif = PIL.Image.Exif()
        exif[0x0112] = 6  # Orientation: a viewer that honours it would rotate this 29x17 image to 17x29.
        PIL.Image.fromarray(rgb).save(image_path, format="JPEG", exif=exif)
        with PIL.Image.open(image_path) as image:
            assert image.getexif()[0x0112] == 6
            expected = np.array(image.convert("RGB"))

        decoded = core._decode_local_image(str(image_path))

        assert decoded is not None
        assert expected.shape == (17, 29, 3)
        np.testing.assert_array_equal(decoded.permute(1, 2, 0).numpy(), expected)

    @pytest.mark.parametrize("image_format", ["TIFF", "PNG"])
    def test_local_image_decoder_returns_none_for_pillow_fallback(self, tmp_path: Path, image_format: str) -> None:
        """Inputs outside the measured 8-bit JPEG/PNG envelope must keep the Pillow path."""
        image_path = tmp_path / f"image.{image_format.lower()}"
        if image_format == "PNG":
            image = PIL.Image.fromarray(np.arange(29 * 17, dtype=np.uint16).reshape(17, 29))
        else:
            image = PIL.Image.new("RGB", (29, 17), color=(1, 127, 255))
        image.save(image_path, format=image_format)

        assert core._decode_local_image(str(image_path)) is None

    @pytest.mark.parametrize(
        ("color_type", "channels", "pillow_mode"),
        [
            pytest.param(0, 1, "I;16", id="grayscale"),
            pytest.param(2, 3, "RGB", id="rgb"),
            pytest.param(6, 4, "RGBA", id="rgba"),
        ],
    )
    def test_local_image_decoder_skips_the_native_decode_of_a_16_bit_png(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, color_type: int, channels: int, pillow_mode: str
    ) -> None:
        """Every 16-bit PNG is recognised from its header, although Pillow reports most of them as 8-bit modes."""
        image_path = tmp_path / "image.png"
        width, height = 29, 17
        # Pillow cannot write a 16-bit RGB or RGBA PNG, so assemble one: signature, IHDR, unfiltered rows, IEND.
        rows = b"".join(b"\x00" + bytes(range(width * channels * 2)) for _ in range(height))
        chunks = (
            (b"IHDR", struct.pack(">IIBBBBB", width, height, 16, color_type, 0, 0, 0)),
            (b"IDAT", zlib.compress(rows)),
            (b"IEND", b""),
        )
        image_path.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            + b"".join(
                struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
                for kind, data in chunks
            )
        )
        with PIL.Image.open(image_path) as header:
            # An 8-bit-looking mode is the reason the bit depth is read from the file rather than from ``header.mode``.
            assert header.mode == pillow_mode
        decode = MagicMock()
        monkeypatch.setattr(core, "decode_image", decode)

        assert core._decode_local_image(str(image_path)) is None
        decode.assert_not_called()

    def test_local_image_decoder_rejects_a_decode_that_is_not_uint8(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Whatever the header checks let through, a decode that is not ``uint8`` must fall back to Pillow."""
        image_path = tmp_path / "image.jpg"
        PIL.Image.new("RGB", (29, 17)).save(image_path)
        decode = MagicMock(return_value=torch.zeros(3, 17, 29, dtype=torch.float32))
        monkeypatch.setattr(core, "decode_image", decode)

        assert core._decode_local_image(str(image_path)) is None
        decode.assert_called_once()

    def test_predict_local_file_uses_pillow_without_torchvision_image_decode(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Once torchvision drops its deprecated image codecs, a local file must still predict through Pillow."""
        image_path = tmp_path / "image.png"
        expected = np.full((17, 29, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)
        decode = MagicMock()
        monkeypatch.setattr(core, "_IS_TORCHVISION_IMAGE_DECODE_AVAILABLE", False)
        monkeypatch.setattr(core, "decode_image", decode)

        detections = _DummyRFDETR().predict(str(image_path), include_source_image=True)

        decode.assert_not_called()
        np.testing.assert_array_equal(detections.metadata["source_image"], expected)

    @pytest.mark.parametrize(
        ("filename", "image_format", "animated"),
        [
            pytest.param("animated.png", "PNG", True, id="animated-png"),
            pytest.param("mislabelled.png", "GIF", False, id="gif-named-png"),
        ],
    )
    def test_local_image_decoder_keeps_pillow_for_content_behind_a_supported_name(
        self, tmp_path: Path, filename: str, image_format: str, animated: bool
    ) -> None:
        """A ``.png`` name is not enough: animated PNG and other formats behind it keep the Pillow path."""
        image_path = tmp_path / filename
        frames = [PIL.Image.new("RGB", (29, 17), color=(40 * index, 127, 255)) for index in range(3 if animated else 1)]
        frames[0].save(image_path, format=image_format, save_all=animated, append_images=frames[1:])
        with PIL.Image.open(image_path) as header:
            assert header.format == image_format
            assert getattr(header, "n_frames", 1) == len(frames)

        assert core._decode_local_image(str(image_path)) is None

    @pytest.mark.parametrize("image_format", ["JPEG", "PNG"])
    def test_predict_truncated_local_file_raises_pillow_error(self, tmp_path: Path, image_format: str) -> None:
        """A file the native decoder rejects must still fail with Pillow's error, not a decoder ``RuntimeError``."""
        image_path = tmp_path / f"truncated.{image_format.lower()}"
        rgb = ((np.arange(61 * 83 * 3).reshape(61, 83, 3) * 37 + 13) % 256).astype(np.uint8)
        PIL.Image.fromarray(rgb).save(image_path, format=image_format)
        image_path.write_bytes(image_path.read_bytes()[: image_path.stat().st_size // 2])

        with pytest.raises(OSError, match="truncated"):
            _DummyRFDETR().predict(str(image_path), include_source_image=False)

    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    def test_predict_local_file_preserves_decompression_bomb_guard(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, image_format: str
    ) -> None:
        """The native route must not bypass Pillow's established oversized-image rejection in ``predict()``."""
        image_path = tmp_path / f"image.{image_format.lower()}"
        PIL.Image.new("RGB", (10, 10)).save(image_path, format=image_format)
        # 100 pixels exceed twice the limit, the point where Pillow raises instead of warning.
        monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 40)

        with pytest.raises(PIL.Image.DecompressionBombError):
            _DummyRFDETR().predict(str(image_path), include_source_image=False)

    def test_predict_local_file_in_the_decompression_bomb_warning_band_still_predicts(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An image between Pillow's limit and twice that limit must warn as Pillow does and still be predicted."""
        image_path = tmp_path / "image.png"
        expected = np.full((10, 10, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)
        # 100 pixels exceed the limit but not twice the limit, so Pillow warns rather than raises.
        monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 60)

        with pytest.warns(PIL.Image.DecompressionBombWarning):
            detections = _DummyRFDETR().predict(str(image_path), include_source_image=True)

        np.testing.assert_array_equal(detections.metadata["source_image"], expected)

    def test_local_image_decoder_decodes_the_bytes_pillow_validated(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A file replaced after its first read must reach neither Pillow's check nor the native decoder."""
        image_path = tmp_path / "image.png"
        expected = np.full((10, 10, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)
        oversized_path = tmp_path / "oversized.png"
        PIL.Image.new("RGB", (30, 30)).save(oversized_path)
        # 100 pixels pass the guard, while 900 exceed twice the limit and would be rejected if Pillow saw them.
        monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 400)

        def replace_file_after(function: Any) -> Any:
            """Wrap ``function`` so the first call to finish swaps the file at ``image_path`` for the oversized one.

            Examples:
                This test-local closure requires the enclosing paths.
                >>> replace_file_after(len)  # doctest: +SKIP
            """

            def wrapper(*args: Any, **kwargs: Any) -> Any:
                """Call ``function``, then replace the file if that has not happened yet.

                Examples:
                    This test-local closure requires the enclosing function and paths.
                    >>> wrapper(len, [])  # doctest: +SKIP
                """
                result = function(*args, **kwargs)
                if oversized_path.exists():
                    oversized_path.replace(image_path)
                return result

            return wrapper

        # Whichever of the two reads comes first, the second one sees a different file if it touches the path.
        monkeypatch.setattr(PIL.Image, "open", replace_file_after(PIL.Image.open))
        monkeypatch.setattr(Path, "read_bytes", replace_file_after(Path.read_bytes))

        decoded = core._decode_local_image(str(image_path))

        assert not oversized_path.exists(), "the replacement never happened, so the race was not exercised"
        assert decoded is not None
        np.testing.assert_array_equal(decoded.permute(1, 2, 0).numpy(), expected)

    def test_predict_unidentified_local_file_raises_pillow_error_naming_the_path(self, tmp_path: Path) -> None:
        """A ``.png`` that is not an image must fail with Pillow's error for that path, not for an in-memory buffer."""
        image_path = tmp_path / "notes.png"
        image_path.write_bytes(b"not an image")

        with pytest.raises(PIL.UnidentifiedImageError) as pillow_error:
            PIL.Image.open(str(image_path))

        with pytest.raises(type(pillow_error.value)) as predict_error:
            _DummyRFDETR().predict(str(image_path), include_source_image=False)

        assert str(predict_error.value) == str(pillow_error.value)

    @pytest.mark.parametrize("kind", ["missing", "empty", "directory"])
    def test_predict_unreadable_local_path_raises_the_error_pillow_raises(self, tmp_path: Path, kind: str) -> None:
        """A path the native route cannot read must fail with the exception Pillow raises for that same path."""
        image_path = tmp_path / f"{kind}.png"
        if kind == "empty":
            image_path.write_bytes(b"")
        elif kind == "directory":
            image_path.mkdir()
        with pytest.raises(OSError) as pillow_error:
            PIL.Image.open(str(image_path))

        with pytest.raises(type(pillow_error.value)) as predict_error:
            _DummyRFDETR().predict(str(image_path), include_source_image=False)

        assert str(predict_error.value) == str(pillow_error.value)

    @pytest.mark.parametrize(
        "error",
        [
            pytest.param(AttributeError("image operators are not loaded"), id="missing-image-ops"),
            pytest.param(DeprecationWarning("image codecs are deprecated"), id="deprecation-warning-as-error"),
        ],
    )
    def test_predict_local_file_falls_back_to_pillow_when_native_decode_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
    ) -> None:
        """A native decode failing with neither ``OSError`` nor ``RuntimeError`` must still leave the file to Pillow.

        torch raises ``AttributeError`` when torchvision's image operators were not loaded, and ``-W error`` turns the
        deprecation warning torchvision 0.29+ emits on every decode into a raised ``DeprecationWarning``.
        """
        image_path = tmp_path / "image.png"
        expected = np.full((17, 29, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)
        monkeypatch.setattr(core, "decode_image", MagicMock(side_effect=error))

        detections = _DummyRFDETR().predict(str(image_path), include_source_image=True)

        assert isinstance(detections, sv.Detections)
        np.testing.assert_array_equal(detections.metadata["source_image"], expected)

    @pytest.mark.filterwarnings("error:The image decoding and encoding capabilities of TorchVision:DeprecationWarning")
    def test_predict_local_file_succeeds_when_torchvision_deprecation_is_an_error(self, tmp_path: Path) -> None:
        """A suite that turns torchvision's codec deprecation warning into an error must still predict a local file.

        The filter matches only that warning's message, so on torchvision releases that do not emit it the native decode
        runs and the same result is expected.
        """
        image_path = tmp_path / "image.png"
        expected = np.full((17, 29, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)

        detections = _DummyRFDETR().predict(str(image_path), include_source_image=True)

        assert isinstance(detections, sv.Detections)
        np.testing.assert_array_equal(detections.metadata["source_image"], expected)

    def test_predict_mixed_inputs_keep_each_source_image_in_input_order(self, tmp_path: Path) -> None:
        """A natively decoded file among other input types must not reorder or swap their source images.

        The call mixes every route a string or image can take, each input with its own size and colour, so a result
        attached to the wrong input changes the shape or the pixels.
        """
        native_path = tmp_path / "native.png"
        native_pixels = np.full((11, 21, 3), (10, 20, 30), dtype=np.uint8)
        PIL.Image.fromarray(native_pixels).save(native_path)
        pil_image = PIL.Image.new("RGB", (22, 12), color=(40, 50, 60))
        array = np.full((13, 23, 3), (70, 80, 90), dtype=np.uint8)
        # Ones convert to 255 exactly in the source image, whatever the float rounding of other values.
        tensor = torch.ones(3, 14, 24)
        fallback_path = tmp_path / "fallback.gif"
        PIL.Image.new("RGB", (25, 15), color=(255, 0, 0)).save(fallback_path)
        with PIL.Image.open(fallback_path) as fallback_image:
            fallback_pixels = np.array(fallback_image.convert("RGB"))
        expected = [
            native_pixels,
            np.array(pil_image),
            array,
            np.full((14, 24, 3), 255, dtype=np.uint8),
            fallback_pixels,
        ]

        detections = _DummyRFDETR().predict(
            [str(native_path), pil_image, array, tensor, str(fallback_path)], include_source_image=True
        )

        sources = [result.metadata["source_image"] for result in detections]
        assert [source.shape for source in sources] == [pixels.shape for pixels in expected]
        assert all(np.array_equal(source, pixels) for source, pixels in zip(sources, expected, strict=True))

    @pytest.mark.parametrize("include_source_image", [False, True])
    def test_predict_local_file_uses_fast_decoder(self, tmp_path: Path, include_source_image: bool) -> None:
        """A supported local file must reach public prediction without a second Pillow decode."""
        image_path = tmp_path / "image.png"
        expected = np.full((17, 29, 3), (1, 127, 255), dtype=np.uint8)
        PIL.Image.fromarray(expected).save(image_path)
        model = _DummyRFDETR()

        with (
            patch("rfdetr._prediction.core.decode_image", wraps=core.decode_image) as decode_spy,
            # Pillow may read the header, but must not decode the pixels the native decoder already returned.
            patch("PIL.ImageFile.ImageFile.load", side_effect=AssertionError("Pillow decoded the pixels")),
        ):
            detections = model.predict(str(image_path), include_source_image=include_source_image)

        decode_spy.assert_called_once()
        assert isinstance(decode_spy.call_args.args[0], torch.Tensor)

        if include_source_image:
            source_image = detections.metadata["source_image"]
            np.testing.assert_array_equal(source_image, expected)
            assert source_image.flags.writeable
            assert source_image.flags.owndata
        else:
            assert "source_image" not in detections.metadata

    @pytest.mark.parametrize("include_source_image", [False, True])
    @pytest.mark.parametrize("image_format", ["PNG", "JPEG"])
    def test_predict_local_file_delivers_the_same_model_batch_as_a_pil_image(
        self, tmp_path: Path, image_format: str, include_source_image: bool
    ) -> None:
        """The model must receive the same normalized batch from a local file as from the PIL image Pillow opens."""
        image_path = tmp_path / f"image.{image_format.lower()}"
        # Every pixel and channel differs, so a flip, a channel swap or an off-by-one changes the batch.
        rgb = ((np.arange(17 * 29 * 3).reshape(17, 29, 3) * 37 + 13) % 256).astype(np.uint8)
        PIL.Image.fromarray(rgb).save(image_path, format=image_format)
        model = _DummyRFDETR()
        model_module = model.model.model
        assert model_module is not None
        model_inputs: list[torch.Tensor] = []

        with model_module.register_forward_pre_hook(_record_model_inputs(model_inputs)):
            model.predict(str(image_path), include_source_image=include_source_image)
            with PIL.Image.open(image_path) as pillow_image:
                model.predict(pillow_image, include_source_image=include_source_image)

        assert len(model_inputs) == 2
        torch.testing.assert_close(model_inputs[0], model_inputs[1])

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
    @pytest.mark.parametrize("grayscale", [False, True])
    def test_converter_is_bit_exact_for_supported_default_dtypes(self, dtype: torch.dtype, grayscale: bool) -> None:
        """Layout/dtype fusion and in-place division produce the same raw floating-point bits."""
        rng = np.random.default_rng(20260821)
        image = rng.integers(0, 256, size=(17, 29, 3), dtype=np.uint8)[:, ::2]
        if grayscale:
            image = image[:, :, 0]
        previous_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            expected = F.to_tensor(image)
            actual = core._uint8_chw_to_float(core._uint8_image_to_chw_view(image), torch.tensor(255, dtype=dtype))
        finally:
            torch.set_default_dtype(previous_dtype)

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.stride() == expected.stride()
        assert actual.contiguous().view(torch.uint8).equal(expected.contiguous().view(torch.uint8))

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(PIL.Image.new("RGB", (29, 17), color=(1, 127, 255)), id="pil_rgb"),
            pytest.param(np.full((17, 29, 3), 127, dtype=np.uint8), id="numpy_uint8"),
        ],
    )
    @pytest.mark.parametrize("include_source_image", [False, True])
    def test_predict_uses_fused_path_for_uint8_images(self, image: object, include_source_image: bool) -> None:
        """PIL and valid uint8 NumPy inputs do not fall back to torchvision's allocating path."""
        model = _DummyRFDETR()
        to_tensor_spy = MagicMock(side_effect=AssertionError("uint8 input unexpectedly used F.to_tensor"))

        with patch("rfdetr._prediction.core.F.to_tensor", to_tensor_spy):
            model.predict(image, include_source_image=include_source_image)

        to_tensor_spy.assert_not_called()

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(np.full((17, 29), 127, dtype=np.uint8), id="numpy_uint8_grayscale"),
            pytest.param(np.full((17, 29, 1), 127, dtype=np.uint8), id="numpy_uint8_single_channel_hwc"),
        ],
    )
    def test_predict_uses_fused_path_for_single_channel_uint8_images(self, image: np.ndarray[Any, Any]) -> None:
        """One-channel uint8 images use the fused converter at the public ``predict()`` boundary."""
        model = _DummyRFDETR()
        model.model_config.num_channels = 1
        model.means = model.means[:1]
        model.stds = model.stds[:1]
        to_tensor_spy = MagicMock(side_effect=AssertionError("uint8 input unexpectedly used F.to_tensor"))

        with (
            patch(
                "rfdetr._prediction.core._uint8_image_to_chw_view", wraps=core._uint8_image_to_chw_view
            ) as converter_spy,
            patch("rfdetr._prediction.core.F.to_tensor", to_tensor_spy),
        ):
            model.predict(image)

        converter_spy.assert_called_once_with(image)
        to_tensor_spy.assert_not_called()

    def test_converter_matches_torchvision_for_contiguous_single_channel_hwc(self) -> None:
        """A freshly-allocated (not sliced) ``(H, W, 1)`` array exercises ``num_channels=1`` models
        (``ModelConfig.num_channels``, ``config.py``).

        Unlike a channel-sliced view, this array is already C-contiguous on input, which is exactly when torchvision's
        own ``to_tensor`` leaves the size-1 leading dimension's stride un-normalized -- so equivalence is checked on
        every dimension that actually has memory-layout meaning (size > 1), plus dtype/shape/contiguity/raw bits.
        """
        rng = np.random.default_rng(20260821)
        image = rng.integers(0, 256, size=(17, 29, 1), dtype=np.uint8)

        expected = F.to_tensor(image)
        actual = core._uint8_chw_to_float(core._uint8_image_to_chw_view(image), torch.tensor(255.0))

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.is_contiguous() == expected.is_contiguous()
        assert actual.stride()[1:] == expected.stride()[1:]
        assert actual.contiguous().view(torch.uint8).equal(expected.contiguous().view(torch.uint8))

    def test_predict_reuses_pil_source_array_for_conversion(self) -> None:
        """Default PIL prediction converts the same NumPy allocation retained as source metadata."""
        model = _DummyRFDETR()
        image = PIL.Image.new("RGB", (29, 17), color=(1, 127, 255))

        with patch(
            "rfdetr._prediction.core._uint8_image_to_chw_view", wraps=core._uint8_image_to_chw_view
        ) as converter_spy:
            detections = model.predict(image)

        converter_spy.assert_called_once()
        assert converter_spy.call_args.args[0] is detections.metadata["source_image"]

    def test_predict_keeps_original_readonly_numpy_as_tensor_source(self) -> None:
        """NumPy conversion retains the caller's storage, so it still triggers ``torch.from_numpy``'s not-writable
        ``UserWarning`` exactly like ``F.to_tensor`` would on the same array."""
        model = _DummyRFDETR()
        image = np.full((17, 29, 3), 127, dtype=np.uint8)
        image.flags.writeable = False

        with (
            patch(
                "rfdetr._prediction.core._uint8_image_to_chw_view", wraps=core._uint8_image_to_chw_view
            ) as converter_spy,
            pytest.warns(UserWarning, match="not writable"),
        ):
            detections = model.predict(image)

        converter_spy.assert_called_once()
        assert converter_spy.call_args.args[0] is image
        assert detections.metadata["source_image"] is not image
        assert detections.metadata["source_image"].flags.writeable

    def test_deferred_widening_is_bit_exact_for_every_byte_value(self) -> None:
        """The 0-dim divisor keeps IEEE-754 division, which a Python scalar does not on CUDA.

        ``Tensor.div_(255)`` is evaluated on CUDA as a multiplication by ``1/255``; that rounds differently from the
        CPU's division for just under half of all byte values (126 of 256). Since ``predict()`` now widens after the
        transfer, the divisor must be a 0-dim tensor for the accelerator result to stay byte-for-byte equal to
        torchvision's host-side conversion.
        """
        image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        chw = core._uint8_image_to_chw_view(image)

        assert core._uint8_chw_to_float(chw, torch.tensor(255.0)).equal(F.to_tensor(image))

    @pytest.mark.gpu
    def test_deferred_widening_is_bit_exact_on_real_cuda(self) -> None:
        """The reciprocal-multiplication gap this fix avoids only manifests on real CUDA hardware.

        ``test_deferred_widening_is_bit_exact_for_every_byte_value`` above proves the divisor swap is correct on
        CPU, where scalar and tensor division already agree bit-for-bit; it cannot exercise the CUDA-specific
        ``Tensor.div_(255)`` reciprocal path this fix replaces. This test runs the same view/pin/transfer/widen
        sequence ``predict()`` uses on a real CUDA device and compares it against the host-computed reference.
        """
        image = np.arange(256, dtype=np.uint8).reshape(16, 16, 1).repeat(3, axis=2)
        expected = F.to_tensor(image)

        chw = core._uint8_image_to_chw_view(image).pin_memory().to("cuda", non_blocking=True)
        scale = torch.tensor(255, device=chw.device, dtype=torch.get_default_dtype())
        actual = core._uint8_chw_to_float(chw, scale).cpu()

        assert actual.equal(expected)

    def test_one_divisor_serves_every_image_of_a_multi_image_call(self) -> None:
        """`predict()` builds the divisor once per call and reuses it for the rest of the batch.

        The in-place division has to consume the freshly widened copy, not the shared divisor, or the second and later
        images of a multi-image call would be scaled by an already-divided value.
        """
        scale = torch.tensor(255, dtype=torch.get_default_dtype())
        dark = np.full((4, 5, 3), 51, dtype=np.uint8)
        bright = np.full((4, 5, 3), 255, dtype=np.uint8)

        first = core._uint8_chw_to_float(core._uint8_image_to_chw_view(dark), scale)
        second = core._uint8_chw_to_float(core._uint8_image_to_chw_view(bright), scale)

        assert first.equal(F.to_tensor(dark))
        assert second.equal(F.to_tensor(bright))
        assert scale.equal(torch.tensor(255, dtype=torch.get_default_dtype()))

    def test_uint8_images_cross_the_device_boundary_unwidened(self) -> None:
        """The uint8 CHW view is pinned, then that distinct buffer crosses the CUDA boundary."""
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda", 0)
        real_to = torch.Tensor.to
        real_tensor = torch.tensor
        real_converter = core._uint8_image_to_chw_view
        chw_views: list[torch.Tensor] = []
        pinned_sources: list[torch.Tensor] = []
        pinned_tensors: list[torch.Tensor] = []
        transfer_sources: list[torch.Tensor] = []
        transfer_non_blocking: list[bool] = []
        transferred_dtypes: list[torch.dtype] = []

        def converter_spy(image: np.ndarray[Any, Any]) -> torch.Tensor:
            """Capture the uint8 CHW view that ``predict()`` prepares for transfer.

            Examples:
                This test-local closure requires its enclosing capture list.
                >>> converter_spy(np.zeros((1, 1, 3), dtype=np.uint8))  # doctest: +SKIP
            """
            chw_view = real_converter(image)
            chw_views.append(chw_view)
            return chw_view

        def pin_memory_spy(self: torch.Tensor) -> torch.Tensor:
            """Return a distinct CPU buffer, simulating ``pin_memory()`` without CUDA hardware.

            Examples:
                This test-local closure requires its enclosing capture list.
                >>> pin_memory_spy(torch.zeros(3))  # doctest: +SKIP
            """
            pinned_sources.append(self)
            pinned_tensor = self.clone()
            pinned_tensors.append(pinned_tensor)
            return pinned_tensor

        def to_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Record the dtype of each simulated CUDA transfer and keep the result on CPU.

            Examples:
                This test-local closure requires its enclosing capture list.
                >>> to_spy(torch.zeros(3), torch.device("cuda"))  # doctest: +SKIP
            """
            if args and isinstance(args[0], torch.device) and args[0].type == "cuda":
                transfer_sources.append(self)
                transfer_non_blocking.append(bool(kwargs.get("non_blocking", False)))
                transferred_dtypes.append(self.dtype)
                return self
            return real_to(self, *args, **kwargs)

        def tensor_spy(data: object, **kwargs: object) -> torch.Tensor:
            """Redirect CUDA target-size construction to CPU for the simulated transfer.

            Examples:
                This test-local closure requires the captured real tensor constructor.
                >>> tensor_spy([[48, 64]], device=torch.device("cuda"))  # doctest: +SKIP
            """
            device = kwargs.get("device")
            if isinstance(device, torch.device) and device.type == "cuda":
                kwargs["device"] = torch.device("cpu")
            return real_tensor(data, **kwargs)

        with (
            patch("rfdetr._prediction.core._uint8_image_to_chw_view", converter_spy),
            patch.object(torch.Tensor, "pin_memory", pin_memory_spy),
            patch.object(torch.Tensor, "to", to_spy),
            patch.object(torch, "tensor", tensor_spy),
        ):
            model.predict(np.full((17, 29, 3), 127, dtype=np.uint8))

        assert len(chw_views) == 1
        assert chw_views[0].dtype == torch.uint8
        assert len(pinned_sources) == 1
        assert pinned_sources[0] is chw_views[0]
        assert len(pinned_tensors) == 1
        assert pinned_tensors[0] is not chw_views[0]
        assert len(transfer_sources) == 1
        assert transfer_sources[0] is pinned_tensors[0]
        assert transfer_non_blocking == [True]
        assert transferred_dtypes == [torch.uint8]

    def test_predict_keeps_torchvision_fallback_for_float_numpy(self) -> None:
        """Non-uint8 NumPy inputs retain torchvision's no-scaling conversion semantics."""
        model = _DummyRFDETR()
        image = np.full((17, 29, 3), 0.5, dtype=np.float32)

        with patch("rfdetr._prediction.core.F.to_tensor", wraps=F.to_tensor) as to_tensor_spy:
            model.predict(image)

        to_tensor_spy.assert_called_once_with(image)

    def test_predict_keeps_torchvision_fallback_for_invalid_uint8_numpy_rank(self) -> None:
        """A uint8 NumPy input outside the documented 2-D/3-D shape keeps torchvision's validation error."""
        model = _DummyRFDETR()
        image = np.zeros((1, 3, 17, 29), dtype=np.uint8)

        with (
            patch("rfdetr._prediction.core.F.to_tensor", wraps=F.to_tensor) as to_tensor_spy,
            pytest.raises(ValueError, match="2/3 dimensional"),
        ):
            model.predict(image)

        to_tensor_spy.assert_called_once_with(image)


class TestPredictNegativeStrideNumpy:
    """``predict()`` must accept NumPy views with negative strides, such as the documented ``frame[:, :, ::-1]``.

    ``torch.from_numpy`` rejects negative strides, and both the fused uint8 conversion and the ``F.to_tensor`` fallback
    go through it. A flipped view must reach the model as exactly the tensor its contiguous copy produces.
    """

    @staticmethod
    def _predict_batch(
        model: RFDETR, image: np.ndarray[Any, Any], include_source_image: bool
    ) -> tuple[torch.Tensor, sv.Detections]:
        """Run ``predict()`` once and return the normalized batch that reached the model, plus the detections.

        Args:
            model: Weight-free model whose ``model.model.model`` is an ``nn.Module``.
            image: The single NumPy image handed to ``predict()``.
            include_source_image: Forwarded to ``predict()``.

        Returns:
            The captured ``(1, C, H, W)`` batch and the returned detections.

        Examples:
            >>> batch, detections = TestPredictNegativeStrideNumpy._predict_batch(
            ...     _DummyRFDETR(), np.zeros((17, 29, 3), dtype=np.uint8), include_source_image=True
            ... )
            >>> tuple(batch.shape[:2]), detections.metadata["source_image"].shape
            ((1, 3), (17, 29, 3))
        """
        batches: list[torch.Tensor] = []

        model_module = model.model.model
        assert model_module is not None
        with model_module.register_forward_pre_hook(_record_model_inputs(batches)):
            detections = model.predict(image, include_source_image=include_source_image)

        assert len(batches) == 1, f"expected one forward call, got {len(batches)}"
        return batches[0], detections

    @pytest.mark.parametrize("include_source_image", [True, False])
    @pytest.mark.parametrize("dtype", [pytest.param(np.uint8, id="uint8"), pytest.param(np.float32, id="float32")])
    def test_negative_stride_view_matches_contiguous_copy(
        self, dtype: type[np.generic], include_source_image: bool
    ) -> None:
        """A negative-stride view feeds the model the same batch, and returns the same source image, as its copy."""
        pixels = np.random.default_rng(20260925).integers(0, 256, size=(17, 29, 3), dtype=np.uint8)
        image = pixels if dtype is np.uint8 else (pixels / 255).astype(dtype)
        view = image[:, :, ::-1]  # the BGR-to-RGB idiom from docs/learn/pretrained.md
        assert any(stride < 0 for stride in view.strides), f"test setup: expected a negative stride, got {view.strides}"
        model = _DummyRFDETR()

        unflipped, _ = self._predict_batch(model, image, include_source_image)
        expected, expected_detections = self._predict_batch(model, np.ascontiguousarray(view), include_source_image)
        actual, actual_detections = self._predict_batch(model, view, include_source_image)

        assert not torch.equal(expected, unflipped), "test setup: the flip must change the model input"
        assert torch.equal(actual, expected), (
            "negative-stride view must reach the model exactly like its contiguous copy"
        )
        if include_source_image:
            np.testing.assert_array_equal(
                actual_detections.metadata["source_image"],
                expected_detections.metadata["source_image"],
                err_msg="source_image must hold the flipped pixels",
            )

    @pytest.mark.parametrize("include_source_image", [True, False])
    @pytest.mark.parametrize("dtype", [pytest.param(np.uint8, id="uint8"), pytest.param(np.float32, id="float32")])
    @pytest.mark.parametrize(
        ("shape", "flip"),
        [
            pytest.param((17, 29, 3), lambda image: image[::-1], id="rows"),
            pytest.param((17, 29, 3), lambda image: image[:, ::-1], id="columns"),
            pytest.param((17, 29, 3), lambda image: image[..., ::-1], id="channels"),
            pytest.param((17, 29, 3), lambda image: image[::-1, ::-1, ::-1], id="all-axes"),
            pytest.param((17, 29, 3), lambda image: image[::-2, ::2], id="mixed-step"),
            # NumPy flags a length-1 axis as contiguous whatever its stride, so ``np.ascontiguousarray`` leaves these.
            pytest.param((1, 29, 3), lambda image: image[::-1], id="size-one-rows"),
            pytest.param((17, 1, 3), lambda image: image[:, ::-1], id="size-one-columns"),
        ],
    )
    def test_flipped_view_matches_its_copy_for_every_axis(
        self,
        dtype: type[np.generic],
        include_source_image: bool,
        shape: tuple[int, int, int],
        flip: Callable[[np.ndarray[Any, Any]], np.ndarray[Any, Any]],
    ) -> None:
        """Flipping any axis, including a length-1 one, reaches the model as exactly its contiguous copy."""
        pixels = np.random.default_rng(20261002).integers(0, 256, size=shape, dtype=np.uint8)
        image = pixels if dtype is np.uint8 else (pixels / 255).astype(dtype)
        view = flip(image)
        assert any(stride < 0 for stride in view.strides), f"test setup: expected a negative stride, got {view.strides}"
        model = _DummyRFDETR()

        expected, expected_detections = self._predict_batch(model, view.copy(), include_source_image)
        actual, actual_detections = self._predict_batch(model, view, include_source_image)

        assert torch.equal(actual, expected), "flipped view must reach the model exactly like its contiguous copy"
        if include_source_image:
            np.testing.assert_array_equal(
                actual_detections.metadata["source_image"],
                expected_detections.metadata["source_image"],
                err_msg="source_image must hold the flipped pixels",
            )

    def test_source_image_of_a_flipped_view_is_an_independent_contiguous_copy(self) -> None:
        """``source_image`` holds the flipped pixels in its own C-contiguous storage, not a view of the caller's
        array."""
        pixels = np.random.default_rng(20261002).integers(0, 256, size=(17, 29, 3), dtype=np.uint8)
        view = pixels[:, :, ::-1]

        _, detections = self._predict_batch(_DummyRFDETR(), view, include_source_image=True)

        source_image = detections.metadata["source_image"]
        assert source_image.flags.c_contiguous
        np.testing.assert_array_equal(source_image, view)
        assert not np.shares_memory(source_image, pixels)

    def test_uint8_view_converts_the_retained_source_image(self) -> None:
        """``source_image`` is already a positive-stride copy of a uint8 view, so conversion reuses it instead of
        copying the frame a second time."""
        model = _DummyRFDETR()
        view = np.full((17, 29, 3), (1, 127, 255), dtype=np.uint8)[:, :, ::-1]

        with patch(
            "rfdetr._prediction.core._uint8_image_to_chw_view", wraps=core._uint8_image_to_chw_view
        ) as converter_spy:
            detections = model.predict(view)

        converter_spy.assert_called_once()
        assert converter_spy.call_args.args[0] is detections.metadata["source_image"]

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(np.zeros((17, 58, 3), dtype=np.uint8)[:, ::2], id="uint8_positive_step"),
            pytest.param(np.zeros((17, 58, 3), dtype=np.float32)[:, ::2], id="float32_positive_step"),
        ],
    )
    def test_non_negative_stride_views_are_converted_without_a_copy(self, image: np.ndarray[Any, Any]) -> None:
        """Positive-step views still hand the caller's own array to the tensor conversion."""
        model = _DummyRFDETR()

        with (
            patch(
                "rfdetr._prediction.core._uint8_image_to_chw_view", wraps=core._uint8_image_to_chw_view
            ) as converter_spy,
            patch("rfdetr._prediction.core.F.to_tensor", wraps=F.to_tensor) as to_tensor_spy,
        ):
            model.predict(image)

        converted = [call.args[0] for call in converter_spy.call_args_list + to_tensor_spy.call_args_list]
        assert len(converted) == 1, f"expected one tensor conversion, got {len(converted)}"
        assert converted[0] is image, "a view torch.from_numpy accepts must not be copied before conversion"


class TestPredictPixelRangeValidation:
    """``predict()`` must still reject out-of-[0, 1]-range tensor inputs, now that the range check is deferred (see the
    ``pending_checks`` comment in ``detr.py``) instead of raised inline, per image, inside the conversion loop."""

    def test_raises_for_pixel_value_above_one(self) -> None:
        """A tensor with any pixel above 1 still raises after the range check moved off the hot path."""
        model = _DummyRFDETR()
        img = torch.zeros(3, 8, 8)
        img[0, 0, 0] = 1.5

        with pytest.raises(ValueError, match="pixel values above 1"):
            model.predict(img)

    def test_valid_images_do_not_raise(self) -> None:
        """A batch of in-range tensors must not raise, deferred check included."""
        model = _DummyRFDETR()
        img = torch.full((3, 8, 8), 0.5)

        model.predict([img, img])  # must not raise

    @pytest.mark.parametrize(
        "image",
        [
            pytest.param(PIL.Image.new("F", (8, 8), color=10_000.0), id="pil"),
            pytest.param(np.full((8, 8, 3), 255, dtype=np.uint8), id="uint8_numpy"),
        ],
    )
    def test_known_valid_converted_image_skips_range_scans(self, image: PIL.Image.Image | np.ndarray[Any, Any]) -> None:
        """PIL conversion and uint8 NumPy scaling already guarantee values in [0, 1]."""
        model = _DummyRFDETR()

        # ``torchvision.normalize`` has its own unrelated ``std.any()`` guard, so replace it to isolate the image-range
        # scans exercised by this test.
        with (
            patch("rfdetr._prediction.core.F.normalize", return_value=torch.zeros(1, 3, 28, 28)),
            patch.object(torch.Tensor, "any", side_effect=AssertionError("unexpected range scan")),
        ):
            model.predict(image, include_source_image=False)

    def test_known_valid_url_skips_range_scans(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A successful URL fetch becomes a PIL image and must bypass range scans."""
        model = _DummyRFDETR()

        def fake_get(url: str, **kwargs: object) -> _FakeResponse:
            """Return a valid image response for the URL conversion boundary.

            Examples:
                >>> fake_get("https://example.com/image.png").status_code
                200
            """
            return _FakeResponse(content=_png_bytes(), status_code=200)

        monkeypatch.setattr(requests, "get", fake_get)
        with (
            patch("rfdetr._prediction.core.F.normalize", return_value=torch.zeros(1, 3, 28, 28)),
            patch.object(torch.Tensor, "any", side_effect=AssertionError("unexpected range scan")),
        ):
            model.predict("https://example.com/image.png", include_source_image=False)

    def test_mixed_known_valid_and_float_numpy_images_still_check_each_image(self) -> None:
        """A known-valid first image must not suppress a later float image's range error."""
        model = _DummyRFDETR()
        known_valid = PIL.Image.new("RGB", (8, 8), color=(255, 255, 255))
        out_of_range = np.full((8, 8, 3), 1.5, dtype=np.float32)

        with pytest.raises(ValueError, match="pixel values above 1"):
            model.predict([known_valid, out_of_range], include_source_image=False)

    def test_known_valid_file_path_skips_range_scans(self, tmp_path: Path) -> None:
        """A local file path decodes to known-valid ``uint8`` pixels, so it skips the range scan too."""
        img_path = tmp_path / "known_valid.png"
        PIL.Image.new("RGB", (8, 8), color=(255, 255, 255)).save(str(img_path))
        model = _DummyRFDETR()

        with (
            patch("rfdetr._prediction.core.F.normalize", return_value=torch.zeros(1, 3, 28, 28)),
            patch.object(torch.Tensor, "any", side_effect=AssertionError("unexpected range scan")),
        ):
            model.predict(str(img_path), include_source_image=False)

    def test_mixed_local_file_and_float_numpy_images_still_check_each_image(self, tmp_path: Path) -> None:
        """A natively decoded file must not suppress a later float image's range error."""
        img_path = tmp_path / "known_valid.png"
        PIL.Image.new("RGB", (8, 8), color=(255, 255, 255)).save(str(img_path))
        out_of_range = np.full((8, 8, 3), 1.5, dtype=np.float32)

        with pytest.raises(ValueError, match="pixel values above 1"):
            _DummyRFDETR().predict([str(img_path), out_of_range], include_source_image=False)

    def test_known_valid_image_skips_range_scans_with_source_image_capture(self) -> None:
        """The scan skip must still apply under ``include_source_image=True``, the public default."""
        model = _DummyRFDETR()
        image = PIL.Image.new("RGB", (8, 8), color=(255, 255, 255))

        with (
            patch("rfdetr._prediction.core.F.normalize", return_value=torch.zeros(1, 3, 28, 28)),
            patch.object(torch.Tensor, "any", side_effect=AssertionError("unexpected range scan")),
        ):
            model.predict(image)

    @pytest.mark.parametrize(
        ("value", "expected_message"),
        [
            pytest.param(1.5, "pixel values above 1", id="above_one"),
            pytest.param(-0.5, "pixel values below 0", id="below_zero"),
        ],
    )
    def test_float_numpy_image_still_checks_range(self, value: float, expected_message: str) -> None:
        """Floating NumPy input can preserve values outside [0, 1] on either bound, so it still needs validation."""
        model = _DummyRFDETR()
        image = np.full((8, 8, 3), value, dtype=np.float32)

        with pytest.raises(ValueError, match=expected_message):
            model.predict(image, include_source_image=False)

    @pytest.mark.parametrize(
        ("first_image_violation", "second_image_violation", "expected_message"),
        [
            pytest.param(1.5, -0.5, "pixel values above 1", id="first_image_above_wins"),
            pytest.param(-0.5, 1.5, "pixel values below 0", id="first_image_below_wins"),
        ],
    )
    def test_first_offending_image_in_input_order_determines_which_message_raises(
        self, first_image_violation: float, second_image_violation: float, expected_message: str
    ) -> None:
        """Matches the original inline check's per-image precedence: image 0 is fully checked (both conditions) before
        image 1 is considered at all, so image 0's own violation always raises first -- distinguishing this from a buggy
        "condition-first" implementation that would scan every image's "above" violations before any "below" ones, which
        would raise "above 1" from image 1 in the second case below instead of "below 0" from image 0."""
        first_image = torch.zeros(3, 8, 8)
        first_image[0, 0, 0] = first_image_violation
        second_image = torch.zeros(3, 8, 8)
        second_image[0, 0, 0] = second_image_violation
        model = _DummyRFDETR()

        with pytest.raises(ValueError, match=expected_message):
            model.predict([first_image, second_image])

    def test_all_images_are_processed_before_a_later_raise(self) -> None:
        """Regression pin for the deferred-check design: unlike the original per-image inline raise, which stopped
        converting/transferring images the moment an earlier one failed, every image now has its conversion and transfer
        queued before any range-check result is forced to a Python bool -- so a later, valid image is still fully
        processed even though an earlier image ultimately makes the whole call raise."""
        invalid_first = torch.zeros(3, 8, 8)
        invalid_first[0, 0, 0] = 1.5
        valid_second = torch.full((3, 8, 8), 0.5)
        model = _DummyRFDETR()

        real_to = torch.Tensor.to
        transferred: list[bool] = []

        def to_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Record whether ``valid_second`` reached its ``.to()`` transfer.

            Examples:
                This test-local closure requires its enclosing tensors.
                >>> to_spy(valid_second, torch.device("cuda"))  # doctest: +SKIP
            """
            if self is valid_second:
                transferred.append(True)
            return real_to(self, *args, **kwargs)

        with patch.object(torch.Tensor, "to", to_spy), pytest.raises(ValueError, match="pixel values above 1"):
            model.predict([invalid_first, valid_second])

        assert transferred, "the later, valid image must still be transferred before the earlier image's error raises"

    @pytest.mark.gpu
    def test_mixed_cpu_and_cuda_tensor_images_are_each_checked_on_their_own_device(self) -> None:
        """Pins the reason ``pending_checks`` keeps one check per image instead of ``torch.stack``-ing them:

        images in a single ``predict()`` call are not guaranteed to share a device, and stacking a CPU-resident and a
        CUDA-resident check result would raise ``RuntimeError`` instead of validating each on its own device.
        """
        model = _DummyRFDETR()
        model.model.device = torch.device("cuda", 0)
        cpu_valid = torch.full((3, 8, 8), 0.5)
        cuda_invalid = torch.full((3, 8, 8), 0.5, device="cuda")
        cuda_invalid[0, 0, 0] = 1.5

        with pytest.raises(ValueError, match="pixel values above 1"):
            model.predict([cpu_valid, cuda_invalid])

        # And the same mix with no violation must not raise.
        cuda_valid = torch.full((3, 8, 8), 0.5, device="cuda")
        model.predict([cpu_valid, cuda_valid])

    def test_range_violation_raises_before_shape_violation_on_the_same_image(self) -> None:
        """Matches the original inline check's within-image precedence: the original code raised the range error before
        ever reaching the shape check for that same image.

        Both checks are now deferred past the conversion loop, so this pins that the deferred range check still runs
        first in the post-loop raise order -- a tensor that is invalid on both axes at once must not surface the shape
        message instead.
        """
        model = _DummyRFDETR()
        wrong_channels_and_out_of_range = torch.zeros(4, 8, 8)
        wrong_channels_and_out_of_range[0, 0, 0] = 1.5

        with pytest.raises(ValueError, match="pixel values above 1"):
            model.predict(wrong_channels_and_out_of_range)

    def test_non_3d_tensor_raises_shape_error_not_an_unpacking_crash(self) -> None:
        """A 2D tensor (no channel dimension at all, not just the wrong channel count) must still raise
        the clear "Invalid tensor image shape" ``ValueError`` -- not an internal ``ValueError: not
        enough values to unpack`` from ``h, w = img_tensor.shape[1:]`` unconditionally unpacking a
        shape that was never checked to have 3 dims in the first place, now that the shape-check raise
        is deferred past that unpacking instead of happening inline before it.
        """
        model = _DummyRFDETR()
        no_channel_dim = torch.zeros(4, 8)  # (H, W)-shaped, not (C, H, W)

        with pytest.raises(ValueError, match="Invalid tensor image shape"):
            model.predict(no_channel_dim, include_source_image=False)

    def test_non_3d_tensor_raises_shape_error_with_default_source_image(self) -> None:
        """Malformed tensor ranks must validate before default source-image conversion."""
        model = _DummyRFDETR()
        no_channel_dim = torch.zeros(4, 8)

        with pytest.raises(ValueError, match="Invalid tensor image shape"):
            model.predict(no_channel_dim)


class TestPredictShape:
    """Verify that ``predict(shape=...)`` controls the resize target.

    Regression tests for https://github.com/roboflow/rf-detr/issues/682.
    """

    def test_predict_uses_resolution_when_no_shape_provided(self) -> None:
        """Without ``shape=``, resize uses ``(resolution, resolution)``."""
        from unittest.mock import patch

        import torchvision.transforms.functional as F  # noqa: N812

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img)

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [28, 28], f"Expected resize to model resolution (28, 28), got {resize_size}"

    def test_predict_uses_provided_rectangular_shape(self) -> None:
        # Regression test for #682
        from unittest.mock import patch

        import torchvision.transforms.functional as F  # noqa: N812

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=(378, 672))

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [378, 672], (
            f"Expected resize to user-provided shape (378, 672), got {resize_size}. "
            "predict() must honour the shape parameter instead of falling back "
            "to (resolution, resolution)."
        )

    def test_predict_shape_square_override(self) -> None:
        # Regression test for #682 — square shape different from model resolution.
        from unittest.mock import patch

        import torchvision.transforms.functional as F  # noqa: N812

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=(56, 56))

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [56, 56], (
            f"Expected resize to user-provided shape (56, 56), got {resize_size}. "
            "predict() must honour the shape parameter even for square sizes "
            "that differ from the model's default resolution."
        )

    @pytest.mark.parametrize(
        "int_shape",
        [
            pytest.param((np.int64(378), np.int64(672)), id="numpy_int64"),
            pytest.param((np.int32(378), np.int32(672)), id="numpy_int32"),
            pytest.param((torch.tensor(378), torch.tensor(672)), id="torch_scalar"),
        ],
    )
    def test_predict_shape_accepts_integer_like_types(self, int_shape: tuple) -> None:
        """Predict() accepts integer-like types (numpy, torch) via the __index__ protocol."""
        from unittest.mock import patch

        import torchvision.transforms.functional as F  # noqa: N812

        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, shape=int_shape)  # type: ignore[arg-type]

        resize_size = list(mock_resize.call_args[0][1])
        assert resize_size == [378, 672], f"predict() must accept integer-like shape types, got resize {resize_size}"

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((378, 671), id="width_not_div_14"),  # 671 % 14 != 0
            pytest.param((371, 672), id="height_not_div_14"),  # 371 % 14 != 0
        ],
    )
    def test_predict_shape_not_divisible_by_14_raises(self, bad_shape: tuple[int, int]) -> None:
        """Predict() must reject shapes with dimensions not divisible by 14."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="divisible by 14"):
            model.predict(img, shape=bad_shape)

    @pytest.mark.parametrize(
        "bad_shape",
        [
            pytest.param((378.0, 672.0), id="float_dims"),
            pytest.param((378,), id="wrong_arity_one_element"),
            pytest.param((378, 672, 3), id="wrong_arity_three_elements"),
            pytest.param((0, 56), id="zero_height"),
            pytest.param((-14, 56), id="negative_height"),
            pytest.param((56, 0), id="zero_width"),
            pytest.param((56, -14), id="negative_width"),
            pytest.param((True, 56), id="bool_height"),
            pytest.param((56, False), id="bool_width"),
        ],
    )
    def test_predict_shape_invalid_raises(self, bad_shape: tuple[int | float | bool, ...]) -> None:
        """Predict() must raise ValueError for invalid shape values."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="shape"):
            model.predict(img, shape=bad_shape)  # type: ignore[arg-type]


class TestPredictResizeMatchesTrainingInterpolation:
    """Verify ``predict()`` resize matches the training-time interpolation.

    Regression test for https://github.com/roboflow/rf-detr/issues/1203.

    Training/validation uses Albumentations ``Resize`` (cv2 bilinear, no
    antialiasing). torchvision's ``F.resize`` defaults to antialiased
    bilinear, which drifts bbox/confidence values relative to the
    pretrained checkpoints' reference preprocessing. ``predict()`` defaults to
    ``antialias=False`` to preserve that behaviour and export parity, and
    ``antialias=True`` opts in for checkpoints trained with antialiased resizing.
    """

    @pytest.mark.parametrize("use_facade", [False, True])
    @pytest.mark.parametrize(
        ("predict_kwargs", "expected_antialias"),
        [
            pytest.param({}, False, id="default"),
            pytest.param({"antialias": False}, False, id="explicit-off"),
            pytest.param({"antialias": True}, True, id="opt-in"),
        ],
    )
    def test_predict_resize_antialias_flag(
        self, predict_kwargs: dict[str, bool], expected_antialias: bool, use_facade: bool
    ) -> None:
        """``predict()`` passes the requested ``antialias`` flag to ``F.resize``; the default is ``False``."""
        native_model = _DummyRFDETR()
        model = RFDETRInference(native_model) if use_facade else native_model
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            model.predict(img, **predict_kwargs)

        assert mock_resize.call_args.kwargs.get("antialias") is expected_antialias, (
            "predict() must resize with antialias=False by default (the #1203 contract: the antialias-free "
            "bilinear resize, cv2.INTER_LINEAR, used during training and by exported models) and honour "
            "an explicit antialias request."
        )

    @pytest.mark.parametrize("antialias", [False, True])
    def test_streamed_predict_resize_antialias_flag(self, antialias: bool) -> None:
        """Streamed facade prediction resizes each lazy batch with the requested ``antialias`` flag.

        The lazy path predicts batches after the call returns, so it must carry the flag into every batch rather than
        read it once from the call that created the generator.
        """
        predictor = RFDETRInference(_DummyRFDETR())
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))

        with patch("rfdetr._prediction.core.F.resize", wraps=F.resize) as mock_resize:
            list(predictor.predict([img], stream=True, antialias=antialias))

        assert mock_resize.call_args.kwargs.get("antialias") is antialias

    def test_predict_antialias_changes_resize_and_matches_reference(self) -> None:
        """``antialias=True`` feeds the model an antialiased resize that differs from the default one."""
        model = _DummyRFDETR()
        resolution = model.model.resolution
        # A one-pixel checkerboard several times larger than the resolution, where antialiasing visibly averages.
        board = ((np.indices((resolution * 4, resolution * 4)).sum(axis=0) % 2) * 255).astype(np.uint8)
        images = [
            PIL.Image.fromarray(np.stack([board] * 3, axis=-1)),
            PIL.Image.fromarray(np.stack([board[:, : resolution * 3]] * 3, axis=-1)),
        ]
        model_inputs: list[torch.Tensor] = []

        model_module = model.model.model
        assert model_module is not None
        with model_module.register_forward_pre_hook(_record_model_inputs(model_inputs)):
            model.predict(images)
            model.predict(images, antialias=True)

        default_batch, antialiased_batch = model_inputs
        reference = torch.stack(
            [F.resize(F.to_tensor(image), [resolution, resolution], antialias=True) for image in images]
        )
        reference = F.normalize(reference, model.means, model.stds)
        assert not torch.allclose(default_batch, antialiased_batch, atol=1e-3)
        torch.testing.assert_close(antialiased_batch, reference)


class TestPredictMpsAntialiasFallback:
    """``predict()`` resize placement for a model whose device is MPS.

    No MPS hardware is needed: transfers to ``mps`` are recorded and simulated on the CPU, and the torch-version
    predicate that gates the fallback is patched, so both branches run on any installed torch.
    """

    @pytest.mark.parametrize(
        ("antialias", "mps_lacks_kernel", "expected_events", "expected_transfer_shapes"),
        [
            pytest.param(
                True,
                True,
                ["resize", "resize", "mps-transfer", "mps-tensor"],
                [(2, 3, 28, 28)],
                id="antialias-old-torch-resizes-on-cpu",
            ),
            pytest.param(
                True,
                False,
                ["mps-transfer", "mps-transfer", "resize", "resize", "mps-tensor"],
                [(3, 80, 100), (3, 48, 64)],
                id="antialias-new-torch-resizes-on-mps",
            ),
            pytest.param(
                False,
                True,
                ["mps-transfer", "mps-transfer", "resize", "resize", "mps-tensor"],
                [(3, 80, 100), (3, 48, 64)],
                id="no-antialias-resizes-on-mps",
            ),
        ],
    )
    def test_mps_resize_placement(
        self,
        antialias: bool,
        mps_lacks_kernel: bool,
        expected_events: list[str],
        expected_transfer_shapes: list[tuple[int, ...]],
    ) -> None:
        """Only antialiasing on an MPS-kernel-less torch resizes on CPU, then moves one batch to MPS."""
        model = _DummyRFDETR()
        model.model.device = torch.device("mps")
        # Two sizes so a per-image transfer cannot pass for a single batch transfer.
        images = [
            PIL.Image.new("RGB", (100, 80), color=(64, 64, 64)),
            PIL.Image.new("RGB", (64, 48), color=(32, 32, 32)),
        ]
        real_to = torch.Tensor.to
        real_tensor = torch.tensor
        real_resize = F.resize
        resize_antialias: list[object] = []
        transfer_shapes: list[tuple[int, ...]] = []
        events: list[str] = []

        def to_spy(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Record a transfer to ``mps`` and keep the tensor on the CPU.

            Examples:
                This test-local closure requires the captured real ``Tensor.to``.
                >>> to_spy(torch.zeros(1, 3, 4, 4), torch.device("mps"))  # doctest: +SKIP
            """
            target = args[0] if args else kwargs.get("device")
            if isinstance(target, torch.device) and target.type == "mps":
                transfer_shapes.append(tuple(self.shape))
                events.append("mps-transfer")
                return self
            return real_to(self, *args, **kwargs)

        def tensor_spy(data: object, **kwargs: object) -> torch.Tensor:
            """Record a tensor constructed on ``mps`` and build it on the CPU instead.

            Examples:
                This test-local closure requires the captured real ``torch.tensor``.
                >>> tensor_spy([[48, 64]], device=torch.device("mps"))  # doctest: +SKIP
            """
            device = kwargs.get("device")
            if isinstance(device, torch.device) and device.type == "mps":
                events.append("mps-tensor")
                kwargs["device"] = torch.device("cpu")
            return real_tensor(data, **kwargs)

        def resize_spy(tensor: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
            """Record each resize and the ``antialias`` flag it received, then run the real resize.

            Examples:
                This test-local closure requires the captured real ``F.resize``.
                >>> resize_spy(torch.zeros(3, 8, 8), [4, 4], antialias=False).shape  # doctest: +SKIP
                torch.Size([3, 4, 4])
            """
            events.append("resize")
            resize_antialias.append(kwargs.get("antialias"))
            return real_resize(tensor, *args, **kwargs)

        with (
            patch("rfdetr.detr._move_model_context_to_device"),
            patch("rfdetr._prediction.core._mps_lacks_antialiased_resize", return_value=mps_lacks_kernel),
            patch.object(torch.Tensor, "to", to_spy),
            patch.object(torch, "tensor", tensor_spy),
            patch("rfdetr._prediction.core.F.resize", side_effect=resize_spy),
        ):
            model.predict(images, antialias=antialias)

        assert resize_antialias == [antialias, antialias]
        assert events == expected_events
        assert transfer_shapes == expected_transfer_shapes


class TestPredictPatchSize:
    """Predict() patch_size resolution and validation tests."""

    def _make_model_with_config(self, patch_size: int, num_windows: int) -> _DummyRFDETR:
        """Return a _DummyRFDETR whose model_config carries patch_size and num_windows."""
        model = _DummyRFDETR()
        model.model_config = SimpleNamespace(patch_size=patch_size, num_windows=num_windows, num_channels=3)
        return model

    def test_predict_defaults_patch_size_from_model_config(self) -> None:
        """Predict() reads patch_size from model_config when not provided by the caller."""
        # patch_size=16, num_windows=2 → block_size=32; shape=(64,64) is valid
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        # Should not raise — 64 % 32 == 0
        model.predict(img, shape=(64, 64))

    def test_predict_shape_must_be_divisible_by_block_size(self) -> None:
        """Predict() rejects shapes not divisible by patch_size * num_windows."""
        # patch_size=16, num_windows=2 → block_size=32; shape (48, 64) fails (48%32==16)
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="divisible by 32"):
            model.predict(img, shape=(48, 64))

    @pytest.mark.parametrize("bad_patch_size", [0, -1, True, False])
    def test_predict_invalid_patch_size_raises(self, bad_patch_size: int) -> None:
        """Predict() must raise ValueError when patch_size is not a positive integer."""
        model = _DummyRFDETR()
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="patch_size must be a positive integer"):
            model.predict(img, patch_size=bad_patch_size)  # type: ignore[arg-type]

    def test_predict_patch_size_mismatch_raises(self) -> None:
        """Predict() must raise ValueError when caller's patch_size != model_config.patch_size."""
        # model has patch_size=16; passing patch_size=14 should raise immediately
        model = self._make_model_with_config(patch_size=16, num_windows=1)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="does not match"):
            model.predict(img, shape=(16, 16), patch_size=14)

    def test_predict_explicit_patch_size_matching_config_succeeds(self) -> None:
        """predict(patch_size=X) must succeed when X matches model_config.patch_size."""
        # patch_size=16, num_windows=2 → block_size=32; shape=(64,64) is valid
        model = self._make_model_with_config(patch_size=16, num_windows=2)
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        # Should not raise — patch_size matches config, 64 % 32 == 0
        model.predict(img, shape=(64, 64), patch_size=16)

    @pytest.mark.parametrize("bad_num_windows", [0, -1, True])
    def test_predict_invalid_num_windows_raises(self, bad_num_windows: int) -> None:
        """Predict() must raise ValueError when model_config.num_windows is not a positive integer."""
        model = self._make_model_with_config(patch_size=14, num_windows=1)
        model.model_config.num_windows = bad_num_windows
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="num_windows must be a positive integer"):
            model.predict(img, shape=(14, 14))

    def test_predict_default_resolution_not_divisible_by_block_size_raises(self) -> None:
        """Predict() with shape=None must raise ValueError when model.resolution % block_size != 0."""
        # patch_size=14, num_windows=1 → block_size=14; set resolution=25 (not divisible)
        model = self._make_model_with_config(patch_size=14, num_windows=1)
        model.model.resolution = 25
        img = PIL.Image.new("RGB", (100, 80), color=(64, 64, 64))
        with pytest.raises(ValueError, match="default resolution"):
            model.predict(img)


class TestPredictNonRGBAutoConvert:
    """Predict() must silently convert non-RGB PIL images and file paths to RGB.

    Standard detector contract: callers pass images in any PIL mode (L, LA,
    RGBA, P …) and expect detection results, not opaque tensor-shape errors.
    Tensor inputs with wrong channel count are still the caller's error.
    """

    @pytest.mark.parametrize(
        "pil_mode",
        [
            pytest.param("L", id="grayscale-L"),
            pytest.param("LA", id="grayscale-with-alpha-LA"),
            pytest.param("RGBA", id="rgba"),
            pytest.param("P", id="palette-P"),
            pytest.param("CMYK", id="cmyk"),
        ],
    )
    def test_non_rgb_pil_image_succeeds(self, pil_mode: str) -> None:
        """PIL images in any mode are auto-converted to RGB and return sv.Detections."""
        import supervision as sv

        img = PIL.Image.new(pil_mode, (28, 28))
        model = _DummyRFDETR()
        detections = model.predict(img)
        assert isinstance(detections, sv.Detections)

    def test_grayscale_file_path_succeeds(self, tmp_path) -> None:
        """Grayscale image opened from a file path is auto-converted and returns sv.Detections."""
        import supervision as sv

        img_path = tmp_path / "gray.png"
        PIL.Image.new("L", (28, 28)).save(str(img_path))
        model = _DummyRFDETR()
        detections = model.predict(str(img_path))
        assert isinstance(detections, sv.Detections)

    def test_wrong_channel_tensor_still_raises(self) -> None:
        """Tensor inputs with wrong channel count must still raise ValueError with helpful message."""
        import torch

        # 1-channel tensor — caller is responsible for correct shape
        tensor = torch.rand(1, 28, 28)
        model = _DummyRFDETR()
        with pytest.raises(ValueError, match="PIL Image or a file path"):
            model.predict(tensor)
