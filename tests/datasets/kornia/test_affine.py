# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the Kornia affine transforms: pixel translation, batch alignment, scalar parameters, shift-scale-rotate.

All tests in this module are CPU-compatible — Kornia operates on CPU tensors identically to GPU tensors, and the
integer-pixel translation indexes pixels directly, so no ``@pytest.mark.gpu`` is needed.
"""

from typing import Any

import numpy as np
import pytest
import torch

from rfdetr.datasets import kornia_transforms
from rfdetr.datasets.kornia_transforms import (
    _make_affine,
    build_kornia_pipeline,
)
from tests._markers import _requires_albumentations, _requires_kornia
from tests.datasets.helpers import _logged_messages, _only_child


def _affine_ranges(transform: Any) -> dict[str, tuple[float, float]]:
    """Read the resolved degrees/translate/scale off a Kornia ``RandomAffine``.

    Like :func:`_sharpness_sampler_range`, this reads a private Kornia detail because no public
    equivalent exists: ``RandomAffine(...).flags`` carries only the resampling options (verified
    against the installed Kornia version), so the geometric ranges are reachable only through the
    parameter generator. A future Kornia release renaming ``_param_generator`` fails here with an
    actionable message rather than a raw ``AttributeError`` inside a test body.

    Examples:
        Requires the optional Kornia dependency, so the examples are skipped where it is unavailable.

        >>> config = {"ShiftScaleRotate": {"p": 1.0}}
        >>> transform = TestShiftScaleRotateFactory()._only_affine(config)  # doctest: +SKIP
        >>> len(_affine_ranges(transform))  # doctest: +SKIP
        3
    """
    generator = getattr(transform, "_param_generator", None)
    if generator is None:
        pytest.fail(
            "Kornia's RandomAffine no longer exposes `_param_generator` (no public alternative "
            "exists for the resolved degrees/translate/scale). Update this helper to match "
            "Kornia's new internal parameter-generator shape."
        )
    resolved = {}
    for key in ("degrees", "translate", "scale"):
        value = getattr(generator, key, None)
        if value is None:
            pytest.fail(
                f"Kornia's RandomAffine parameter generator no longer carries {key!r}. "
                "Update this helper to match Kornia's new internal shape."
            )
        resolved[key] = tuple(float(v) for v in value)
    return resolved


@_requires_kornia
class TestAffinePixelTranslation:
    """Pixel translation must move image and training targets together."""

    def test_fixed_translation_moves_image_boxes_and_masks(self) -> None:
        """The GPU-backend config must not silently discard ``translate_px``."""
        config = {"Affine": {"translate_px": {"x": (4, 4), "y": (0, 0)}, "rotate": 0.0, "scale": 1.0, "p": 1.0}}
        pipeline = build_kornia_pipeline(config, resolution=64, with_masks=True)
        image = torch.arange(64 * 80, dtype=torch.float32).reshape(1, 1, 64, 80) / (64 * 80)
        boxes = torch.tensor([[[30.0, 30.0, 34.0, 34.0]]])
        masks = torch.zeros(1, 2, 64, 80)
        masks[0, 0, 30:34, 30:34] = 1.0
        masks[0, 1, :, 70:] = 1.0

        image_out, boxes_out, masks_out = pipeline(image, boxes, masks)

        expected_image = torch.zeros_like(image)
        expected_image[:, :, :, 4:] = image[:, :, :, :-4]
        expected_masks = torch.zeros_like(masks)
        expected_masks[:, :, :, 4:] = masks[:, :, :, :-4]
        torch.testing.assert_close(image_out, expected_image, rtol=0, atol=1e-5)
        torch.testing.assert_close(boxes_out, torch.tensor([[[34.0, 30.0, 38.0, 34.0]]]), rtol=0, atol=1e-5)
        torch.testing.assert_close(masks_out, expected_masks, rtol=0, atol=0)

    def test_fixed_vertical_translation_moves_image_boxes_and_masks(self) -> None:
        """A pure y-axis shift must move pixels, boxes, and masks down together.

        ``apply_transform`` indexes rows with ``offsets[:, 1:2]`` and columns with ``offsets[:, 0:1]``
        (``_kornia_pixel_affine.py``), while boxes and masks are carried through Kornia's own matrix-based affine
        transform built from the same ``translations`` parameter. A sign or axis mix-up confined to the y-component is
        invisible when every other case in this module fixes ``y=0``, because ``-0`` and ``+0`` index identically;
        isolate the vertical axis here with ``x=0`` so such a divergence cannot hide behind a nonzero horizontal offset.
        """
        config = {"Affine": {"translate_px": {"x": (0, 0), "y": (3, 3)}, "rotate": 0.0, "scale": 1.0, "p": 1.0}}
        pipeline = build_kornia_pipeline(config, resolution=64, with_masks=True)
        image = torch.arange(80 * 64, dtype=torch.float32).reshape(1, 1, 80, 64) / (80 * 64)
        boxes = torch.tensor([[[30.0, 30.0, 34.0, 34.0]]])
        masks = torch.zeros(1, 2, 80, 64)
        masks[0, 0, 30:34, 30:34] = 1.0
        masks[0, 1, 70:, :] = 1.0

        image_out, boxes_out, masks_out = pipeline(image, boxes, masks)

        expected_image = torch.zeros_like(image)
        expected_image[:, :, 3:, :] = image[:, :, :-3, :]
        expected_masks = torch.zeros_like(masks)
        expected_masks[:, :, 3:, :] = masks[:, :, :-3, :]
        torch.testing.assert_close(image_out, expected_image, rtol=0, atol=1e-5)
        torch.testing.assert_close(boxes_out, torch.tensor([[[30.0, 33.0, 34.0, 37.0]]]), rtol=0, atol=1e-5)
        torch.testing.assert_close(masks_out, expected_masks, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "configured,expected",
        [
            pytest.param(3, ((3, 3), (3, 3)), id="scalar-both-axes"),
            pytest.param((-2, 4), ((-2, 4), (-2, 4)), id="range-both-axes"),
            pytest.param({"x": (-2, 4)}, ((-2, 4), (0, 0)), id="x-only"),
            pytest.param({"y": -3}, ((0, 0), (-3, -3)), id="y-only"),
        ],
    )
    def test_integer_parameter_ranges(self, configured: Any, expected: tuple[tuple[int, int], tuple[int, int]]) -> None:
        """The public builder preserves signed, inclusive per-axis pixel ranges."""
        pipeline = build_kornia_pipeline({"Affine": {"translate_px": configured, "p": 1.0}}, resolution=32)
        transform = next(iter(pipeline.children()))
        torch.manual_seed(0)
        sampled = transform.generate_parameters((512, 1, 16, 20))["translations"]
        assert sampled.shape == (512, 2)
        assert sampled[:, 0].min().item() == expected[0][0]
        assert sampled[:, 0].max().item() == expected[0][1]
        assert sampled[:, 1].min().item() == expected[1][0]
        assert sampled[:, 1].max().item() == expected[1][1]
        assert torch.equal(sampled, sampled.round())

    def test_zero_probability_preserves_inputs(self) -> None:
        """``p=0`` keeps the historical no-augmentation behavior."""
        pipeline = build_kornia_pipeline({"Affine": {"translate_px": 4, "p": 0.0}}, resolution=32, with_masks=True)
        image = torch.rand(2, 1, 24, 32)
        boxes = torch.tensor([[[4.0, 5.0, 8.0, 9.0]], [[10.0, 8.0, 16.0, 15.0]]])
        masks = torch.zeros(2, 1, 24, 32)
        masks[:, :, 5:9, 4:8] = 1.0
        image_out, boxes_out, masks_out = pipeline(image, boxes, masks)
        torch.testing.assert_close(image_out, image, rtol=0, atol=0)
        torch.testing.assert_close(boxes_out, boxes, rtol=0, atol=0)
        torch.testing.assert_close(masks_out, masks, rtol=0, atol=0)

    def test_partial_probability_keeps_each_image_and_target_aligned(self) -> None:
        """Kornia's per-image selection must not mix selected and unselected targets."""
        pipeline = build_kornia_pipeline({"Affine": {"translate_px": {"x": 2}, "p": 0.5}}, resolution=12)
        image = torch.zeros(16, 1, 8, 12)
        image[:, :, 3, 3] = 1.0
        boxes = torch.tensor([[[3.0, 3.0, 4.0, 4.0]]]).expand(16, -1, -1).clone()
        torch.manual_seed(3)

        image_out, boxes_out = pipeline(image, boxes)

        selected = boxes_out[:, 0, 0] > 3.5
        assert selected.any() and (~selected).any()
        expected_x = torch.where(selected, 5, 3)
        for batch_index in range(16):
            assert image_out[batch_index, 0, 3, expected_x[batch_index]] == 1
        torch.testing.assert_close(boxes_out[:, 0, 0], expected_x.float(), rtol=0, atol=0)

    def test_no_boxes_still_moves_image_and_padding_mask(self) -> None:
        """A background-only batch does not depend on a populated box tensor."""
        pipeline = build_kornia_pipeline(
            {"Affine": {"translate_px": {"x": -2}, "p": 1.0}}, resolution=8, with_masks=True
        )
        image = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
        boxes = torch.empty(1, 0, 4)
        padding = torch.zeros(1, 1, 1, 8)
        padding[:, :, :, 6:] = 1.0

        image_out, boxes_out, padding_out = pipeline(image, boxes, padding)

        torch.testing.assert_close(image_out, torch.tensor([[[[2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 0.0, 0.0]]]]))
        assert boxes_out.shape == (1, 0, 4)
        expected_padding = torch.zeros_like(padding)
        expected_padding[:, :, :, 4:6] = 1.0
        torch.testing.assert_close(padding_out, expected_padding, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "border_mode,interpolation",
        [
            pytest.param(0, 1, id="constant-linear"),
            pytest.param(1, 1, id="replicate-linear"),
            pytest.param(0, 0, id="constant-nearest"),
        ],
    )
    @_requires_albumentations
    def test_fixed_translation_matches_albumentations_geometry(self, border_mode: int, interpolation: int) -> None:
        """Image, box, and mask geometry matches the CPU backend on a non-square input."""
        import albumentations

        config = {
            "translate_px": {"x": (4, 4), "y": (0, 0)},
            "rotate": 0.0,
            "scale": 1.0,
            "shear": 0.0,
            "border_mode": border_mode,
            "interpolation": interpolation,
            "p": 1.0,
        }
        image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        object_mask = np.zeros((7, 9), dtype=np.uint8)
        object_mask[2:5, 1:4] = 1
        padding_mask = np.zeros((7, 9), dtype=np.uint8)
        padding_mask[:, 7:] = 1
        cpu = albumentations.Compose(
            [albumentations.Affine(**config)], bbox_params=albumentations.BboxParams(format="pascal_voc")
        )(image=image, bboxes=[(1, 2, 4, 5)], masks=[object_mask, padding_mask])
        pipeline = build_kornia_pipeline({"Affine": config}, resolution=7, with_masks=True)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        boxes = torch.tensor([[[1.0, 2.0, 4.0, 5.0]]])
        masks = torch.from_numpy(np.stack((object_mask, padding_mask))).float().unsqueeze(0)

        image_out, boxes_out, masks_out = pipeline(image_tensor, boxes, masks)

        cpu_image = torch.from_numpy(cpu["image"].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        cpu_masks = torch.from_numpy(np.stack(cpu["masks"]).copy()).float().unsqueeze(0)
        torch.testing.assert_close(image_out, cpu_image, rtol=0, atol=1e-5)
        torch.testing.assert_close(boxes_out, torch.tensor([cpu["bboxes"]]), rtol=0, atol=1e-5)
        torch.testing.assert_close(masks_out, cpu_masks, rtol=0, atol=0)

    @pytest.mark.parametrize(
        "translation,other,diagnostic",
        [
            pytest.param(True, {}, "not booleans", id="boolean"),
            pytest.param(1.5, {}, "whole pixels.*got 1.5", id="fractional-pixel"),
            pytest.param((0, 2.5), {}, "whole pixels.*got 2.5", id="fractional-range-bound"),
            pytest.param({"x": 0.5}, {}, "whole pixels.*got 0.5", id="fractional-axis"),
            pytest.param((3, -3), {}, "minimum", id="reversed-range"),
            pytest.param((1, 2, 3), {}, "two-element", id="three-element-range"),
            pytest.param({"z": 4}, {}, "keys", id="wrong-axis"),
            pytest.param(4, {"translate_percent": 0.1}, "either", id="two-translation-units"),
            pytest.param(4, {"rotate": 10}, "got rotate=10", id="nonzero-rotate"),
            pytest.param(4, {"rotate": (-5, 5)}, r"got rotate=\(-5, 5\)", id="nonzero-rotate-range"),
            pytest.param(4, {"shear": 5}, "got shear=5", id="nonzero-shear"),
            pytest.param(4, {"scale": 1.2}, "got scale=1.2", id="nonunit-scalar-scale"),
            pytest.param(4, {"interpolation": 5}, "got interpolation=5", id="unknown-interpolation"),
            pytest.param(4, {"mask_interpolation": 7}, "got mask_interpolation=7", id="unknown-mask-interpolation"),
            pytest.param(4, {"fill_mask": 1}, "requires fill_mask=0", id="unsupported-nonzero-fill-mask"),
            pytest.param(4, {"border_mode": 2}, "got border_mode=2", id="unsupported-reflect-mode"),
            pytest.param(4, {"border_mode": 3}, "got border_mode=3", id="unsupported-wrap-mode"),
            pytest.param(4, {"border_mode": 4}, "got border_mode=4", id="reflected-box-duplication"),
            pytest.param(4, {"fill": 1}, "fill", id="unsupported-nonzero-fill"),
            pytest.param(4, {"fit_output": True}, "fit_output", id="output-shape-change"),
            pytest.param(4, {"scale": {"x": (0.8, 1.2)}}, "scale=1", id="unsupported-axis-scale"),
            pytest.param(4, {"scale": [0.8, 1.2]}, "scale=1", id="nonunit-list-scale"),
            pytest.param(4, {"shear": {"x": 0}}, "shear=0", id="axis-shear-missing-y-defaults-to-one-degree"),
            pytest.param(4, {"fill": np.array([0, 0, 1])}, "fill=0", id="nonzero-array-fill"),
            pytest.param(2**63 - 1, {}, "64-bit", id="int64-overflow"),
            pytest.param(4, {"cval": 1}, "cval=1; use fill instead", id="albumentations-1x-cval"),
            pytest.param(4, {"cval_mask": 1}, "cval_mask=1; use fill_mask instead", id="albumentations-1x-cval-mask"),
            pytest.param(4, {"mode": 2}, "mode=2; use border_mode instead", id="albumentations-1x-mode"),
        ],
    )
    def test_unsupported_pixel_affine_config_fails_at_build(
        self, translation: Any, other: dict[str, Any], diagnostic: str
    ) -> None:
        """Unsupported settings fail before training instead of changing silently."""
        with pytest.raises(ValueError, match=diagnostic):
            build_kornia_pipeline({"Affine": {"translate_px": translation, **other}}, resolution=32)

    @pytest.mark.parametrize(
        "translation,expected",
        [
            pytest.param(4.0, ((4, 4), (4, 4)), id="integral-float"),
            pytest.param(np.int64(3), ((3, 3), (3, 3)), id="numpy-integer"),
            pytest.param({"x": 3.0}, ((3, 3), (0, 0)), id="integral-float-axis"),
            pytest.param((-2.0, np.int32(4)), ((-2, 4), (-2, 4)), id="mixed-integral-range"),
        ],
    )
    def test_integral_offsets_resolve_to_int_bounds(
        self, translation: Any, expected: tuple[tuple[int, int], tuple[int, int]]
    ) -> None:
        """Integral offsets of any numeric type build and resolve to plain ``int`` bounds.

        Albumentations accepts ``4.0`` and NumPy integers for ``translate_px``, so a config that trains on the CPU
        backend must build here too. The bounds must come out as Python ``int`` because ``torch.randint`` rejects a
        float bound at the first forward pass, long after the pipeline was built.
        """
        pipeline = build_kornia_pipeline({"Affine": {"translate_px": translation, "p": 1.0}}, resolution=32)

        bounds = next(iter(pipeline.children())).translation_bounds
        assert bounds == expected
        assert all(type(bound) is int for axis in bounds for bound in axis)

    @pytest.mark.parametrize(
        "options",
        [
            pytest.param({"scale": {"x": 1.0, "y": 1.0}}, id="unit-axis-scale"),
            pytest.param({"scale": {"x": (1.0, 1.0)}}, id="unit-axis-scale-missing-y"),
            pytest.param({"shear": {"x": 0, "y": 0}}, id="zero-axis-shear"),
            pytest.param({"interpolation": 2}, id="cubic-interpolation"),
            pytest.param({"interpolation": 3}, id="area-interpolation"),
            pytest.param({"interpolation": 4}, id="lanczos-interpolation"),
            pytest.param({"mask_interpolation": 1}, id="linear-mask-interpolation"),
            pytest.param({"keep_ratio": True}, id="keep-ratio"),
            pytest.param({"balanced_scale": True}, id="balanced-scale"),
            pytest.param({"rotate_method": "ellipse"}, id="ellipse-rotate-method"),
            pytest.param({"fill": np.zeros(3)}, id="zero-array-fill"),
            pytest.param({"border_mode": 1, "fill": 5}, id="replicate-ignores-fill"),
            pytest.param({"border_mode": 1, "fill_mask": 1}, id="replicate-ignores-fill-mask"),
        ],
    )
    def test_options_without_effect_on_a_pixel_shift_build(self, options: dict[str, Any]) -> None:
        """Albumentations options that cannot change a whole-pixel shift build instead of raising.

        Each option only matters for scaling, rotation, sub-pixel sampling or a constant border; Albumentations accepts
        all of them and returns the same pure shift, so refusing them on Kornia would fail a CPU-valid config on GPU.
        """
        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        pipeline = build_kornia_pipeline({"Affine": {"translate_px": 4, **options, "p": 1.0}}, resolution=32)

        transform = next(iter(pipeline.children()))
        assert isinstance(transform, PixelTranslatedAffine)
        assert transform.translation_bounds == ((4, 4), (4, 4))

    @pytest.mark.parametrize(
        "scale", [pytest.param((1.0, 1.0), id="unit-tuple"), pytest.param([1.0, 1.0], id="unit-list"), 1]
    )
    def test_unit_scale_variants_build_for_translate_px(self, scale: Any) -> None:
        """A semantically unit scale must build regardless of its container type.

        ``scale != (1.0, 1.0)`` previously rejected a list such as ``[1.0, 1.0]`` because Python never treats a list as
        equal to a tuple, even with identical elements, so only the tuple form built successfully. Each parametrized
        value here represents the same no-op scale.
        """
        build_kornia_pipeline({"Affine": {"translate_px": 4, "scale": scale, "p": 1.0}}, resolution=32)


#: Non-square frame shared by the per-image alignment tests, so an x/y axis mix-up cannot cancel out.
_ALIGN_HEIGHT, _ALIGN_WIDTH = 6, 8


#: Per-image ``(tx, ty)`` pixel offsets for a four-image batch. ``_ALIGN_WIDTH=8`` and ``_ALIGN_HEIGHT=6`` make the
#: ``fully-off-frame-*`` cases satisfy ``|t| >= extent`` on the shifted axis, and each case mixes signs across images
#: so a batch-order or per-image mix-up cannot pass.
_ALIGNMENT_OFFSETS = [
    pytest.param([(2, 1), (-3, -2), (0, 0), (4, -1)], id="in-frame-mixed-signs"),
    pytest.param([(0, -1), (0, -3), (1, -2), (-1, -5)], id="negative-y"),
    pytest.param([(6, 0), (-6, 0), (0, 4), (0, -4)], id="partly-off-frame"),
    pytest.param([(8, 0), (-8, 0), (0, 6), (0, -6)], id="fully-off-frame-at-extent"),
    pytest.param([(20, 3), (-9, -2), (2, 25), (-8, -7)], id="fully-off-frame-beyond-extent"),
]


__doctest_requires__ = {"_translated_batch": ["kornia"]}


def _numpy_shift(array: np.ndarray, tx: int, ty: int, border_mode: int) -> np.ndarray:
    """Shift a ``(C, H, W)`` array right by ``tx`` and down by ``ty`` pixels, filling from the border.

    This is the reference for pixel translation: pad by the largest offset (constant zeros for ``border_mode=0``,
    replicated edge for ``border_mode=1``), then slice the shifted window. It shares no indexing code with the
    implementation under test.

    Args:
        array: Channel-first array to shift.
        tx: Horizontal shift in pixels; positive moves content right.
        ty: Vertical shift in pixels; positive moves content down.
        border_mode: ``0`` fills uncovered pixels with zeros, ``1`` replicates the nearest edge pixel.

    Returns:
        The shifted array with the same shape as ``array``.

    Examples:
        >>> row = np.array([[[1, 2, 3, 4]]])
        >>> _numpy_shift(row, 1, 0, 0).tolist()
        [[[0, 1, 2, 3]]]
        >>> _numpy_shift(row, 1, 0, 1).tolist()
        [[[1, 1, 2, 3]]]
        >>> _numpy_shift(row, -9, 0, 0).tolist()
        [[[0, 0, 0, 0]]]
    """
    _, height, width = array.shape
    pad = max(abs(tx), abs(ty))
    padded = np.pad(array, ((0, 0), (pad, pad), (pad, pad)), mode="constant" if border_mode == 0 else "edge")
    return padded[:, pad - ty : pad - ty + height, pad - tx : pad - tx + width]


def _translated_batch(images: torch.Tensor, border_mode: int, offsets: list[tuple[int, int]]) -> tuple[Any, Any, Any]:
    """Build a pixel-translation transform and fix its sampled per-image offsets.

    The transform comes from the public builder, so ``border_mode`` goes through the same mapping as a real config.
    ``generate_parameters`` supplies the full Kornia parameter set for the batch; only ``translations`` is then
    overwritten with ``offsets`` so every case is deterministic and RNG-free while covering offsets (such as
    ``|t| >= W``) that a random draw would rarely produce.

    Args:
        images: Batch the transform will be applied to; only its shape is used here.
        border_mode: Albumentations border mode, ``0`` constant or ``1`` replicate.
        offsets: One ``(tx, ty)`` pixel offset per image in ``images``.

    Returns:
        The transform, its parameter dict, and the per-image affine matrices for ``images``.

    Examples:
        >>> images = torch.zeros(2, 1, 6, 8)
        >>> transform, params, matrix = _translated_batch(images, 1, [(2, -1), (-3, 0)])
        >>> params["translations"].tolist()
        [[2.0, -1.0], [-3.0, 0.0]]
        >>> matrix.shape
        torch.Size([2, 3, 3])
    """
    config = {"Affine": {"translate_px": {"x": (-1, 1), "y": (-1, 1)}, "border_mode": border_mode, "p": 1.0}}
    transform = next(iter(build_kornia_pipeline(config, resolution=_ALIGN_WIDTH, with_masks=True).children()))
    params = transform.generate_parameters(tuple(images.shape))
    params["translations"] = torch.tensor(offsets, dtype=params["translations"].dtype)
    matrix = transform.compute_transformation(images, params, transform.flags)
    return transform, params, matrix


@_requires_kornia
class TestAffinePixelTranslationBatchAlignment:
    """Every image, box, and mask in a batch must move by its own sampled offset, matching a NumPy oracle.

    Border modes ``0`` and ``1`` are constant and replicate fill (Albumentations numbering).
    """

    @pytest.mark.parametrize("border_mode", [0, 1])
    @pytest.mark.parametrize("offsets", _ALIGNMENT_OFFSETS)
    def test_images_match_numpy_shift_per_image(self, offsets: list[tuple[int, int]], border_mode: int) -> None:
        """Each image in a batch is shifted by its own offset, including offsets at or past the frame extent.

        Distinct non-zero pixel values make a wrong source pixel, a swapped axis, or a zero fill that should have been a
        replicated edge (and the reverse) all change the output.
        """
        batch = len(offsets)
        images = torch.arange(1, batch * 3 * _ALIGN_HEIGHT * _ALIGN_WIDTH + 1, dtype=torch.float32).reshape(
            batch, 3, _ALIGN_HEIGHT, _ALIGN_WIDTH
        )
        transform, params, matrix = _translated_batch(images, border_mode, offsets)

        out = transform.apply_transform(images, params, transform.flags, transform=matrix)

        expected = np.stack(
            [_numpy_shift(image, tx, ty, border_mode) for image, (tx, ty) in zip(images.numpy(), offsets)]
        )
        torch.testing.assert_close(out, torch.from_numpy(expected), rtol=0, atol=0)

    @pytest.mark.parametrize("border_mode", [0, 1])
    @pytest.mark.parametrize("offsets", _ALIGNMENT_OFFSETS)
    def test_masks_match_numpy_shift_per_image(self, offsets: list[tuple[int, int]], border_mode: int) -> None:
        """Instance and padding masks move exactly like the image: whole pixels, no interpolation, same border mode.

        Masks are binary, so any interpolation would produce fractional values and any offset disagreement with the
        image would show up as a misaligned mask against the same NumPy oracle.
        """
        batch = len(offsets)
        masks = ((torch.arange(batch * 2 * _ALIGN_HEIGHT * _ALIGN_WIDTH) * 7) % 5 < 2).float()
        masks = masks.reshape(batch, 2, _ALIGN_HEIGHT, _ALIGN_WIDTH)
        transform, params, matrix = _translated_batch(masks, border_mode, offsets)

        out = transform.apply_transform_mask(masks, params, dict(transform.flags), transform=matrix)

        expected = np.stack([_numpy_shift(mask, tx, ty, border_mode) for mask, (tx, ty) in zip(masks.numpy(), offsets)])
        torch.testing.assert_close(out, torch.from_numpy(expected), rtol=0, atol=0)

    @pytest.mark.parametrize("offsets", _ALIGNMENT_OFFSETS)
    def test_boxes_translate_by_per_image_offset(self, offsets: list[tuple[int, int]]) -> None:
        """Boxes shift by their image's offset, so they stay aligned with its pixels even when they leave the frame.

        Clipping or dropping boxes that end up outside the frame is a later stage (see the data-module tests); this
        transform must move all four coordinates unclamped, which is what keeps partly and fully off-frame boxes
        consistent with the shifted image.
        """
        from kornia.geometry.boxes import Boxes

        batch = len(offsets)
        boxes = torch.tensor([[1.0, 1.0, 4.0, 3.0], [5.0, 2.0, 8.0, 5.0]]).expand(batch, -1, -1).clone()
        images = torch.zeros(batch, 1, _ALIGN_HEIGHT, _ALIGN_WIDTH)
        transform, params, matrix = _translated_batch(images, 0, offsets)

        out = transform.apply_transform_box(
            Boxes.from_tensor(boxes, mode="xyxy"), params, transform.flags, transform=matrix
        ).to_tensor(mode="xyxy")

        shifts = torch.tensor([[tx, ty, tx, ty] for tx, ty in offsets], dtype=torch.float32)[:, None, :]
        torch.testing.assert_close(out, boxes + shifts, rtol=0, atol=0)

    @pytest.mark.parametrize("border_mode", [0, 1])
    @pytest.mark.parametrize(
        "offset",
        [
            pytest.param((3, 2), id="positive-both-axes"),
            pytest.param((-2, -3), id="negative-both-axes"),
            pytest.param((-2, 2), id="mixed-signs"),
        ],
    )
    @_requires_albumentations
    def test_in_frame_translation_matches_albumentations(self, offset: tuple[int, int], border_mode: int) -> None:
        """Image, box, and mask geometry matches the CPU backend for in-frame shifts on both axes.

        The box stays fully inside the frame after every shift here: Albumentations clips and filters boxes that leave
        the frame, so parity is only defined for in-frame shifts. Offsets at or past the frame extent are covered
        against the NumPy oracle above, never against Albumentations.
        """
        import albumentations

        tx, ty = offset
        config = {
            "translate_px": {"x": (tx, tx), "y": (ty, ty)},
            "rotate": 0.0,
            "scale": 1.0,
            "shear": 0.0,
            "border_mode": border_mode,
            "interpolation": 1,
            "p": 1.0,
        }
        image = np.arange(7 * 9 * 3, dtype=np.uint8).reshape(7, 9, 3)
        object_mask = np.zeros((7, 9), dtype=np.uint8)
        object_mask[3:5, 3:6] = 1
        padding_mask = np.zeros((7, 9), dtype=np.uint8)
        padding_mask[:, 7:] = 1
        cpu = albumentations.Compose(
            [albumentations.Affine(**config)], bbox_params=albumentations.BboxParams(format="pascal_voc")
        )(image=image, bboxes=[(3, 3, 6, 5)], masks=[object_mask, padding_mask])
        pipeline = build_kornia_pipeline({"Affine": config}, resolution=7, with_masks=True)
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        boxes = torch.tensor([[[3.0, 3.0, 6.0, 5.0]]])
        masks = torch.from_numpy(np.stack((object_mask, padding_mask))).float().unsqueeze(0)

        image_out, boxes_out, masks_out = pipeline(image_tensor, boxes, masks)

        cpu_image = torch.from_numpy(cpu["image"].copy()).permute(2, 0, 1).float().unsqueeze(0) / 255
        cpu_masks = torch.from_numpy(np.stack(cpu["masks"]).copy()).float().unsqueeze(0)
        torch.testing.assert_close(image_out, cpu_image, rtol=0, atol=1e-5)
        torch.testing.assert_close(boxes_out, torch.tensor([cpu["bboxes"]]), rtol=0, atol=1e-5)
        torch.testing.assert_close(masks_out, cpu_masks, rtol=0, atol=0)


@_requires_kornia
class TestAffineScalarParameters:
    """Scalar ``Affine`` ranges must not lose an axis or fail in the Kornia backend."""

    def test_scalar_translate_percent_moves_both_axes(self) -> None:
        """A scalar translation must not silently leave the horizontal axis fixed."""
        torch.manual_seed(0)
        pipeline = build_kornia_pipeline(
            {"Affine": {"translate_percent": 0.1, "rotate": (0.0, 0.0), "p": 1.0}}, resolution=64
        )
        image = torch.zeros(1, 1, 64, 64)
        image[0, 0, 32, 32] = 1.0
        boxes = torch.tensor([[[31.0, 31.0, 33.0, 33.0]]])
        horizontal, vertical = [], []

        for _ in range(128):
            moved, _ = pipeline(image, boxes)
            flat_index = int(moved[0, 0].flatten().argmax().item())
            horizontal.append(flat_index % 64 - 32)
            vertical.append(flat_index // 64 - 32)

        assert min(horizontal) < 0 < max(horizontal), horizontal
        assert min(vertical) < 0 < max(vertical), vertical

    @_requires_albumentations
    def test_scalar_translate_percent_warns_about_the_cpu_distribution(self, mocker) -> None:
        """A fixed positive CPU scalar must not silently become signed GPU sampling."""
        import albumentations

        cpu_affine = albumentations.Affine(translate_percent=0.1)
        warn = mocker.patch.object(kornia_transforms.logger, "warning")
        gpu_affine = kornia_transforms._make_affine({"translate_percent": 0.1, "p": 1.0})

        assert cpu_affine.translate_percent == {"x": (0.1, 0.1), "y": (0.1, 0.1)}
        assert tuple(float(value) for value in gpu_affine._param_generator.translate) == pytest.approx(
            (0.1, 0.1), abs=1e-6
        )
        messages = _logged_messages(warn)
        assert any("scalar" in message and "different distribution" in message for message in messages), messages

    def test_scalar_scale_executes_public_pipeline(self) -> None:
        """A scalar scale must build and run instead of failing inside Kornia."""
        pipeline = build_kornia_pipeline({"Affine": {"scale": 1.1, "rotate": (0.0, 0.0), "p": 1.0}}, resolution=64)
        image = torch.rand(1, 3, 64, 64)
        boxes = torch.tensor([[[8.0, 8.0, 24.0, 24.0]]])

        image_out, boxes_out = pipeline(image, boxes)

        assert image_out.shape == image.shape
        assert boxes_out.shape == boxes.shape
        assert not torch.allclose(boxes_out, boxes)

    def test_unset_rotate_samples_no_rotation(self) -> None:
        """An unset ``rotate`` means no rotation, as in Albumentations' ``Affine``."""
        ranges = _affine_ranges(_make_affine({"translate_percent": (-0.1, 0.1), "scale": (0.8, 1.2), "p": 1.0}))

        assert ranges["degrees"] == pytest.approx((0.0, 0.0), abs=1e-6)

    def test_scale_only_config_leaves_the_image_unrotated(self) -> None:
        """A scale-only config with a unit scale changes nothing, so it cannot have rotated."""
        torch.manual_seed(0)
        transform = _make_affine({"scale": (1.0, 1.0), "p": 1.0})
        image = torch.rand(1, 3, 32, 32)

        result = transform(image)

        assert torch.allclose(result, image, atol=1e-5)

    def test_existing_pairs_keep_their_ranges(self) -> None:
        """The scalar fix must not change existing ordered-pair behaviour."""
        ranges = _affine_ranges(_make_affine({"translate_percent": (-0.05, 0.2), "scale": (0.8, 1.2), "p": 1.0}))

        assert ranges["translate"] == pytest.approx((0.2, 0.2), abs=1e-6)
        assert ranges["scale"] == pytest.approx((0.8, 1.2), abs=1e-6)

    @pytest.mark.parametrize(
        "params,key,expected",
        [
            pytest.param(
                {"translate_percent": 0.1, "scale": (0.9, 1.1), "rotate": (0.0, 0.0), "p": 1.0},
                "translate",
                (0.1, 0.1),
                id="translate",
            ),
            pytest.param(
                {"translate_percent": (-0.05, 0.05), "scale": 1.1, "rotate": (0.0, 0.0), "p": 1.0},
                "scale",
                (1.1, 1.1),
                id="scale",
            ),
        ],
    )
    def test_scalar_parameters_resolve_to_the_symmetric_range(
        self, params: dict[str, Any], key: str, expected: tuple[float, float]
    ) -> None:
        """A scalar parameter must resolve to the exact symmetric range Kornia receives, not just execute."""
        ranges = _affine_ranges(_make_affine(params))

        assert ranges[key] == pytest.approx(expected, abs=1e-6)

    @pytest.mark.parametrize(
        "parameter,value",
        [
            pytest.param("translate_percent", ("-0.1", "0.1"), id="translate-pair"),
            pytest.param("scale", ("0.8", "1.2"), id="scale-pair"),
            pytest.param("translate_percent", "0.1", id="translate-scalar"),
            pytest.param("scale", "1.1", id="scale-scalar"),
        ],
    )
    def test_string_values_keep_their_existing_rejection(self, parameter: str, value: Any) -> None:
        """Normalising numeric scalars must not accept string values."""
        with pytest.raises((TypeError, ValueError)):
            _make_affine({parameter: value, "p": 1.0})


@_requires_kornia
class TestShiftScaleRotateFactory:
    """`ShiftScaleRotate` on the Kornia backend (issue #1252).

    Albumentations deprecates this name in favour of `Affine`, but the CPU path still accepts it, so a config using it
    trained on a CPU box and raised on a GPU box. It maps onto the same `RandomAffine` that `Affine` uses, which is what
    makes it safe: no resolution change, so the padding-mask constraint that keeps the crops unsupported does not apply.

    The limits are *not* pass-through, which is why this needs its own builder rather than an alias.
    """

    def _only_affine(self, aug_config: dict[str, dict[str, Any]]) -> Any:
        """Build a pipeline and return its sole ``RandomAffine`` transform.

        Examples:
            Requires the optional Kornia dependency, so the examples are skipped where it is unavailable.

            >>> config = {"ShiftScaleRotate": {"p": 1.0}}
            >>> transform = TestShiftScaleRotateFactory()._only_affine(config)  # doctest: +SKIP
            >>> transform.__class__.__name__  # doctest: +SKIP
            'RandomAffine'
        """
        return _only_child(build_kornia_pipeline(aug_config, 560), "RandomAffine")

    def test_scale_limit_is_a_delta_biased_by_one(self) -> None:
        """The mapping that a plain alias to Affine would get wrong.

        Albumentations documents `scale_limit` as "biased by 1": it samples from `(1 + low, 1 + high)`, so `0.1` means a
        scale between 0.9 and 1.1. Kornia's `scale` is the absolute multiplier, so forwarding `0.1` unchanged would ask
        it to shrink the image to between a tenth of its size and nothing at all.
        """
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"scale_limit": 0.1, "p": 1.0}}))
        assert ranges["scale"] == pytest.approx((0.9, 1.1), abs=1e-4)

    def test_scale_limit_as_asymmetric_pair(self) -> None:
        """A pair is also a delta, so it pivots the same way rather than passing through."""
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"scale_limit": (-0.2, 0.5), "p": 1.0}}))
        assert ranges["scale"] == pytest.approx((0.8, 1.5), abs=1e-4)

    def test_scalar_limits_expand_symmetrically(self) -> None:
        """`rotate_limit=30` means (-30, 30), not the degenerate (30, 30) `_as_range` would give."""
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"rotate_limit": 30, "p": 1.0}}))
        assert ranges["degrees"] == pytest.approx((-30.0, 30.0), abs=1e-4)

    def test_defaults_match_albumentations(self) -> None:
        """An empty config resolves to Albumentations' own documented defaults."""
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"p": 1.0}}))
        assert ranges["degrees"] == pytest.approx((-45.0, 45.0), abs=1e-4)
        assert ranges["translate"] == pytest.approx((0.0625, 0.0625), abs=1e-4)
        assert ranges["scale"] == pytest.approx((0.9, 1.1), abs=1e-4)

    def test_per_axis_shift_limits(self) -> None:
        """`shift_limit_x`/`shift_limit_y` override the shared limit; Kornia takes them as (tx, ty)."""
        ranges = _affine_ranges(
            self._only_affine({"ShiftScaleRotate": {"shift_limit_x": 0.2, "shift_limit_y": 0.05, "p": 1.0}})
        )
        assert ranges["translate"] == pytest.approx((0.2, 0.05), abs=1e-4)

    @pytest.mark.parametrize("key", ["shift_limit_x", "shift_limit_y"])
    def test_none_per_axis_shift_limit_uses_the_shared_limit(self, key: str) -> None:
        """Albumentations accepts ``None`` as an axis-level fallback to ``shift_limit``."""
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"shift_limit": 0.2, key: None, "p": 1.0}}))
        assert ranges["translate"] == pytest.approx((0.2, 0.2), abs=1e-4)

    def test_asymmetric_shift_limit_warns_about_the_symmetric_kornia_approximation(self, mocker) -> None:
        """A one-sided CPU range must not silently become a different GPU distribution."""
        warn = mocker.patch.object(kornia_transforms.logger, "warning")
        ranges = _affine_ranges(self._only_affine({"ShiftScaleRotate": {"shift_limit_x": (0.1, 0.2), "p": 1.0}}))

        assert ranges["translate"] == pytest.approx((0.2, 0.0625), abs=1e-4)
        messages = _logged_messages(warn)
        assert any("asymmetric" in message and "shift_limit_x" in message for message in messages), messages

    def test_plain_config_does_not_warn(self, mocker) -> None:
        """Every parameter here maps, so an ordinary config must stay quiet."""
        warn = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms.build_kornia_pipeline(
            {"ShiftScaleRotate": {"shift_limit": 0.1, "scale_limit": 0.2, "rotate_limit": 15, "p": 1.0}}, 560
        )

        assert not warn.called, _logged_messages(warn)


@_requires_kornia
class TestResolutionPreservingGeometry:
    """Perspective and ShiftScaleRotate keep the output resolution and move boxes with the image (issue #1252)."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"Perspective": {"scale": 0.3, "p": 1.0}}, id="perspective"),
            pytest.param({"ShiftScaleRotate": {"rotate_limit": 30, "p": 1.0}}, id="shift-scale-rotate"),
        ],
    )
    def test_output_keeps_the_input_resolution(self, config) -> None:
        """Image height and width survive the transform, which is what makes it safe where the crops are not."""
        img = torch.rand(2, 3, 64, 64)
        boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0]], [[4.0, 4.0, 20.0, 20.0]]])

        img_out, _ = build_kornia_pipeline(config, 560)(img, boxes)

        assert img_out.shape[-2:] == img.shape[-2:]

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"Perspective": {"scale": 0.4, "p": 1.0}}, id="perspective"),
            pytest.param({"ShiftScaleRotate": {"rotate_limit": 45, "p": 1.0}}, id="shift-scale-rotate"),
        ],
    )
    def test_boxes_follow_the_transform(self, config) -> None:
        """A geometric transform that left the boxes where they were would silently mislabel every image."""
        img = torch.rand(1, 3, 64, 64)
        boxes = torch.tensor([[[8.0, 8.0, 40.0, 40.0]]])

        _, boxes_out = build_kornia_pipeline(config, 560)(img, boxes)

        assert not torch.allclose(boxes_out, boxes), "boxes must move with the image"

    @pytest.mark.parametrize(
        ("name", "key", "value"),
        [
            ("Perspective", "fit_output", True),
            ("Perspective", "interpolation", 1),
            ("Perspective", "mask_interpolation", 0),
            ("Perspective", "border_mode", 0),
            ("Perspective", "fill", 0),
            ("Perspective", "fill_mask", 0),
            ("ShiftScaleRotate", "interpolation", 1),
            ("ShiftScaleRotate", "border_mode", 0),
            ("ShiftScaleRotate", "mask_interpolation", 0),
            ("ShiftScaleRotate", "fill", 0),
            ("ShiftScaleRotate", "fill_mask", 0),
            ("ShiftScaleRotate", "rotate_method", "ellipse"),
        ],
    )
    def test_unmappable_options_are_reported_not_silently_dropped(self, mocker, name, key, value) -> None:
        """Options Kornia's random warp cannot take (RandomPerspective: only distortion_scale and p) must warn."""
        warn = mocker.patch.object(kornia_transforms.logger, "warning")
        build_kornia_pipeline({name: {key: value}}, 560)

        messages = _logged_messages(warn)
        assert any("ignores" in m and key in m for m in messages), messages


@_requires_kornia
class TestPixelTranslatedAffineInverse:
    """``inverse()`` must undo the integer shift on images, masks, and boxes alike."""

    def test_round_trip_restores_image_masks_and_boxes(self) -> None:
        """A forward shift followed by ``inverse()`` returns every target to its original position.

        Kornia's inverse path hands ``apply_transform`` the inverted matrix, which this transform ignores in favour of
        its sampled offsets, while boxes are inverted through that matrix. Images and masks must therefore be shifted
        back too, or they drift away from the boxes. The shift moves right by 3 and up by 2 so both axes and both signs
        are exercised; the bands the forward pass pushed out of frame were zero-filled and stay zero after the inverse.
        """
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((3, 3), (-2, -2)), p=1.0, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input", "bbox_xyxy", "mask"])
        image = torch.arange(1, 8 * 10 + 1, dtype=torch.float32).reshape(1, 1, 8, 10)
        boxes = torch.tensor([[[4.0, 3.0, 6.0, 5.0]]])
        masks = torch.zeros(1, 1, 8, 10)
        masks[0, 0, 3:5, 4:6] = 1.0

        restored_image, restored_boxes, restored_masks = pipeline.inverse(*pipeline(image, boxes, masks))

        expected_image = torch.zeros_like(image)
        expected_image[:, :, 2:, :7] = image[:, :, 2:, :7]
        torch.testing.assert_close(restored_image, expected_image, rtol=0, atol=0)
        torch.testing.assert_close(restored_masks, masks, rtol=0, atol=0)
        torch.testing.assert_close(restored_boxes, boxes, rtol=0, atol=1e-5)

    @pytest.mark.parametrize("p", [0.0, 0.5, 1.0])
    def test_round_trip_restores_each_image_with_its_own_offset(self, p: float) -> None:
        """Inverse images and masks stay aligned with boxes for empty, partial, and full selections."""
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((-2, 2), (-1, 1)), p=p, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input", "bbox_xyxy", "mask"])
        image = torch.arange(1, 16 * 6 * 8 + 1, dtype=torch.float32).reshape(16, 1, 6, 8)
        boxes = torch.tensor([[[2.0, 2.0, 4.0, 4.0]]]).repeat(16, 1, 1)
        masks = torch.zeros_like(image)
        masks[:, :, 2:4, 2:4] = 1.0

        shifted_image, shifted_boxes, shifted_masks = pipeline(image, boxes, masks)
        restored_image, restored_boxes, restored_masks = pipeline.inverse(shifted_image, shifted_boxes, shifted_masks)

        selected = shifted_boxes[:, 0, 0] != boxes[:, 0, 0]
        expected_image = torch.zeros_like(image)
        # Box displacements independently identify the original pixels that survived each forward shift.
        offsets = (shifted_boxes[:, 0, :2] - boxes[:, 0, :2]).to(torch.long)
        for row, (dx, dy) in enumerate(offsets.tolist()):
            rows = slice(max(0, -dy), min(6, 6 - dy))
            columns = slice(max(0, -dx), min(8, 8 - dx))
            expected_image[row, :, rows, columns] = image[row, :, rows, columns]
        if p == 0.5:
            assert selected.any() and (~selected).any()
            assert offsets.unique(dim=0).shape[0] > 2
        torch.testing.assert_close(restored_image, expected_image, rtol=0, atol=0)
        torch.testing.assert_close(restored_masks, masks, rtol=0, atol=0)
        torch.testing.assert_close(restored_boxes, boxes, rtol=0, atol=1e-5)


@_requires_kornia
class TestPixelTranslatedAffineSameOnBatch:
    """``same_on_batch=True`` must give every image in a batch one shared shift."""

    def test_container_flag_shifts_every_image_identically(self) -> None:
        """A container-level ``same_on_batch=True`` shifts a batch of identical images to identical outputs.

        ``AugmentationSequential(same_on_batch=True)`` sets the flag on each child, and Kornia's own generators then
        draw one value and repeat it across the batch. The integer offsets replace Kornia's translation draw, so they
        must honour the flag too; with 49 possible offsets per image, independent draws make 16 outputs differ.
        """
        from kornia.augmentation import AugmentationSequential

        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((-3, 3), (-3, 3)), p=1.0, padding_mode="zeros")
        pipeline = AugmentationSequential(transform, data_keys=["input"], same_on_batch=True)
        image = torch.arange(1, 8 * 8 + 1, dtype=torch.float32).reshape(1, 1, 8, 8).repeat(16, 1, 1, 1)

        shifted = pipeline(image)

        torch.testing.assert_close(shifted, shifted[:1].expand_as(shifted), rtol=0, atol=0)

    def test_empty_selection_returns_empty_offsets(self) -> None:
        """A shared draw still yields no offset rows when no image in the batch was selected.

        With ``p < 1`` Kornia may select zero images and then asks for parameters of an empty batch; expanding the one
        shared draw must produce an empty ``(0, 2)`` offset tensor rather than a stray row.
        """
        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((-3, 3), (-3, 3)), p=0.5, padding_mode="zeros")
        transform.same_on_batch = True

        translations = transform.generate_parameters((0, 1, 8, 8))["translations"]

        assert translations.shape == (0, 2)


@_requires_kornia
class TestPixelTranslatedAffineMaskDtype:
    """``apply_transform`` must return masks in the dtype they arrived in."""

    @pytest.mark.parametrize(
        "dtype",
        [
            pytest.param(torch.bool, id="bool"),
            pytest.param(torch.uint8, id="uint8"),
            pytest.param(torch.float16, id="float16"),
            pytest.param(torch.float32, id="float32"),
        ],
    )
    def test_shift_preserves_dtype_and_values(self, dtype: torch.dtype) -> None:
        """A shift by 1 right and 2 down keeps the mask dtype and moves each set pixel by exactly that offset.

        Masks reach the transform as bool, integer or float tensors, and the zero-padding path multiplies the gathered
        values in place by a boolean validity mask, which must neither promote nor reject any of them. The pixel in row
        0 would be smeared into the two rows the shift exposes if that masking were skipped.
        """
        from rfdetr.datasets._kornia_pixel_affine import PixelTranslatedAffine

        transform = PixelTranslatedAffine(((1, 1), (2, 2)), p=1.0, padding_mode="zeros")
        mask = torch.zeros(1, 1, 4, 5, dtype=dtype)
        mask[0, 0, 0, 3] = 1
        mask[0, 0, 1, 1] = 1

        shifted = transform.apply_transform(mask, {"translations": torch.tensor([[1.0, 2.0]])}, transform.flags)

        expected = torch.zeros(1, 1, 4, 5, dtype=dtype)
        expected[0, 0, 2, 4] = 1
        expected[0, 0, 3, 2] = 1
        assert shifted.dtype == dtype
        assert torch.equal(shifted, expected)
