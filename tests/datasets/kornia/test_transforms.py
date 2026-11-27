# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the Kornia GPU augmentation pipeline builder: factories, warnings, defaults and backend resolution.

All tests in this module are CPU-compatible — Kornia operates on CPU tensors identically to GPU tensors, so no
``@pytest.mark.gpu`` is needed.
"""

from typing import Any

import pytest
import torch

from rfdetr.config import AugmentationBackend
from rfdetr.datasets import kornia_transforms
from rfdetr.datasets._torchvision import RandomHorizontalFlip
from rfdetr.datasets.aug_configs import (
    AUG_AERIAL,
    AUG_AGGRESSIVE,
    AUG_CONSERVATIVE,
    AUG_INDUSTRIAL,
)
from rfdetr.datasets.coco import make_coco_transforms
from rfdetr.datasets.kornia_transforms import (
    build_kornia_pipeline,
    resolve_augmentation_backend,
    resolve_backend_for_build,
    split_aug_config_for_gpu,
)
from rfdetr.datasets.transforms import AlbumentationsWrapper, Normalize
from tests._markers import _requires_albumentations, _requires_kornia
from tests.datasets.helpers import _logged_messages, _only_child


def _sharpness_sampler_range(transform: torch.nn.Module) -> tuple[float, float]:
    """Read the sampled ``sharpness`` range off a Kornia ``RandomSharpness`` transform.

    This intentionally reads a private Kornia implementation detail (``_param_generator.sampler_dict``) because no
    public equivalent exists: ``RandomSharpness(...).flags`` is empty (verified against the installed Kornia
    version), so the sampled range isn't reachable through any public attribute. If a future Kornia release
    renames or removes ``_param_generator``/``sampler_dict``, this raises a clear, actionable failure instead of a
    raw ``AttributeError``/``KeyError`` deep inside the test body.

    Args:
        transform: A ``kornia.augmentation.RandomSharpness`` instance (or equivalent) built with a sampled range.

    Returns:
        The sampled ``(low, high)`` bounds as floats.
    """
    try:
        sampler = transform._param_generator.sampler_dict["sharpness"]
        return float(sampler.low), float(sampler.high)
    except (AttributeError, KeyError) as exc:
        pytest.fail(
            "Kornia's RandomSharpness no longer exposes the sampled `sharpness` range via the private "
            f"`_param_generator.sampler_dict['sharpness']` path (no public alternative exists): {exc!r}. Update "
            "this helper to match Kornia's new internal parameter-generator shape."
        )


@_requires_kornia
class TestBuildKorniaPipeline:
    """build_kornia_pipeline returns a valid pipeline for every preset and rejects unknown transform keys."""

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param(AUG_CONSERVATIVE, id="conservative"),
            pytest.param(AUG_AGGRESSIVE, id="aggressive"),
            pytest.param(AUG_AERIAL, id="aerial"),
            pytest.param(AUG_INDUSTRIAL, id="industrial"),
        ],
    )
    def test_each_preset_config(self, config):
        """Each named preset builds a pipeline without errors."""
        assert build_kornia_pipeline(config, 560) is not None, "a preset must build a non-None pipeline"

    @pytest.mark.parametrize(
        ("config", "bad_key"),
        [
            pytest.param({"FooBarTransform": {"p": 0.5}}, "FooBarTransform", id="unknown-only"),
            pytest.param(
                {"HorizontalFlip": {"p": 0.5}, "BogusTransform": {"p": 0.3}}, "BogusTransform", id="known-plus-unknown"
            ),
            # Deliberately out of scope: albumentations shifts additively, Kornia scales multiplicatively.
            pytest.param(
                {"HueSaturationValue": {"hue_shift_limit": 20}}, "HueSaturationValue", id="hue-saturation-value"
            ),
        ],
    )
    def test_unsupported_key_raises_value_error_naming_it(self, config, bad_key):
        """A transform key Kornia cannot build raises ValueError that names the offending key."""
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline(config, 560)

        assert bad_key in str(exc_info.value)

    def test_empty_config_returns_pipeline(self):
        """An empty config dict returns a valid (no-op) pipeline, not None."""
        assert build_kornia_pipeline({}, 560) is not None, "Empty config must still return a pipeline object"

    def test_to_gray_builds_random_grayscale(self):
        """``ToGray`` maps onto ``K.RandomGrayscale`` (issue #1227)."""
        pipeline = build_kornia_pipeline({"ToGray": {"p": 0.5}}, 560)
        transform_names = [child.__class__.__name__ for child in pipeline.children()]
        assert "RandomGrayscale" in transform_names

    def test_to_gray_keeps_three_channels_and_greys(self):
        """``ToGray`` matches Albumentations: grayscale content, still three channels."""
        pipeline = build_kornia_pipeline({"ToGray": {"p": 1.0}}, 560)
        image = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
        out, _ = pipeline(image, boxes)

        assert out.shape == image.shape, "ToGray must preserve the three-channel shape"
        # A greyscale image has identical values across the channel axis.
        assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5)
        assert torch.allclose(out[:, 1], out[:, 2], atol=1e-5)

    @_requires_albumentations
    def test_to_gray_accepted_by_both_backends(self):
        """The same config is readable by the Albumentations backend too (issue #1227).

        ``ToGray`` resolved on Albumentations via ``getattr`` long before it was a Kornia built-in, so a config that
        worked on one backend raised on the other.
        """
        config = {"ToGray": {"p": 0.5}}
        assert build_kornia_pipeline(config, 560) is not None

        wrappers = AlbumentationsWrapper.from_config(config)
        assert len(wrappers) == 1, (
            "from_config(strict=False) silently drops unresolved transforms, so length must be checked"
        )
        built_names = [t.__class__.__name__ for t in wrappers[0].transform.transforms]
        assert "ToGray" in built_names, f"expected a ToGray transform, got {built_names}"

    def test_to_gray_defaults_p_to_point_five_when_omitted(self):
        """Omitting p resolves to 0.5, matching Albumentations (not Kornia's native 0.1 default)."""
        pipeline = build_kornia_pipeline({"ToGray": {}}, 560)
        to_gray = next(child for child in pipeline.children() if child.__class__.__name__ == "RandomGrayscale")
        assert to_gray.p == pytest.approx(0.5)

    def test_to_gray_p_zero_is_a_no_op(self):
        """p=0.0 never applies: forward pass returns the input unchanged."""
        pipeline = build_kornia_pipeline({"ToGray": {"p": 0.0}}, 560)
        image = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
        out, _ = pipeline(image, boxes)

        assert torch.equal(out, image), "p=0.0 must never apply ToGray"

    def test_to_gray_ignores_method_and_num_output_channels_on_kornia(self):
        """method/num_output_channels have no Kornia equivalent and are currently ignored there.

        Pins the divergence documented on ``_make_to_gray``: Albumentations honors both, Kornia always uses BT.601
        weights and returns 3 channels. If this is ever fixed, this test should be updated to assert the new (parity)
        behavior instead.
        """
        pipeline = build_kornia_pipeline({"ToGray": {"method": "max", "num_output_channels": 1, "p": 1.0}}, 560)
        image = torch.rand(1, 3, 32, 32)
        boxes = torch.tensor([[[0.0, 0.0, 10.0, 10.0]]])
        out, _ = pipeline(image, boxes)

        assert out.shape == image.shape, "num_output_channels=1 is currently ignored -- output stays 3-channel"
        assert torch.allclose(out[:, 0], out[:, 1], atol=1e-5), "method='max' is currently ignored on Kornia"

    @pytest.mark.parametrize(
        "config",
        [
            pytest.param({"GaussianBlur": {"blur_limit": (3, 7), "p": 0.5}}, id="blur_limit-pair"),
            pytest.param({"GaussianBlur": {"sigma": 1.5, "p": 0.5}}, id="sigma-scalar"),
            pytest.param({"GaussianBlur": {"sigma": (1.5,), "p": 0.5}}, id="sigma-1elem-seq"),
            pytest.param({"GaussNoise": {"std_range": 0.05, "p": 0.5}}, id="std_range-scalar"),
        ],
    )
    def test_scalar_or_pair_range_params_build(self, config):
        """Range params accept a scalar or a pair, as Albumentations does.

        ``_make_rotate`` already accepts either form for ``limit``; these builders used to raise a bare ``TypeError``
        from inside Kornia on a config that is valid for the CPU path.
        """
        assert build_kornia_pipeline(config, 560) is not None

    @pytest.mark.parametrize(
        ("blur_limit", "expected"),
        [
            pytest.param((3, 7), (7, 7), id="odd-upper-bound"),
            pytest.param((3, 6), (7, 7), id="even-upper-bound-rounds-up"),
        ],
    )
    def test_blur_limit_pair_uses_upper_bound(self, blur_limit, expected):
        """A ``(min, max)`` ``blur_limit`` resolves to its upper bound, rounded up to an odd kernel size."""
        pipeline = build_kornia_pipeline({"GaussianBlur": {"blur_limit": blur_limit, "p": 1.0}}, 560)
        transform = next(iter(pipeline.children()))
        assert transform.flags["kernel_size"] == expected

    def test_scalar_std_range_is_used_verbatim(self, mocker) -> None:
        """A scalar ``std_range`` is used as-is, with no per-sample-drift warning emitted."""
        mock_warning = mocker.patch.object(kornia_transforms.logger, "warning")
        pipeline = kornia_transforms.build_kornia_pipeline({"GaussNoise": {"std_range": 0.05, "p": 1.0}}, 560)

        transform = next(iter(pipeline.children()))
        image = torch.ones(1, 3, 16, 16)
        torch.manual_seed(0)
        expected = image + torch.randn_like(image) * 0.05
        torch.manual_seed(0)
        torch.testing.assert_close(transform(image), expected, rtol=1e-4, atol=1e-6)
        mock_warning.assert_not_called()

    def test_hflip_disabled_for_keypoint_pipeline(self, mocker):
        """Keypoint-mode Kornia augmentation drops hflip transforms with a warning."""
        config = {"HorizontalFlip": {"p": 0.5}, "VerticalFlip": {"p": 0.5}}
        warning = mocker.patch.object(kornia_transforms.logger, "warning")

        pipeline = kornia_transforms.build_kornia_pipeline(config, 560, include_keypoints=True)

        transform_names = [child.__class__.__name__ for child in pipeline.children()]
        assert "RandomHorizontalFlip" not in transform_names
        assert "RandomVerticalFlip" in transform_names
        assert warning.called
        assert "HorizontalFlip" in str(warning.call_args)
        image = torch.zeros(1, 3, 16, 16)
        boxes = torch.tensor([[[1.0, 1.0, 8.0, 8.0]]])
        assert len(pipeline(image, boxes)) == 2

    def test_keypoint_pipeline_rejects_unpaired_flip_index(self) -> None:
        """A malformed joint-pair list fails before the first training batch."""
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, 16, include_keypoints=True, keypoint_flip_pairs=[0])

        assert "even number" in str(exc_info.value)

    # --- pixel-level transforms added for issue #1252 -------------------

    @pytest.mark.parametrize(
        ("name", "params", "expected"),
        [
            pytest.param("Blur", {"blur_limit": 5}, "RandomBoxBlur", id="blur"),
            pytest.param("Sharpen", {"alpha": (0.2, 0.5)}, "RandomSharpness", id="sharpen"),
            pytest.param("Equalize", {}, "RandomEqualize", id="equalize"),
            pytest.param("CLAHE", {"clip_limit": 4.0}, "RandomClahe", id="clahe"),
        ],
    )
    def test_pixel_transforms_map_to_kornia(self, name, params, expected):
        """Each documented pixel-level name builds its Kornia counterpart (issue #1252)."""
        pipeline = build_kornia_pipeline({name: params}, 560)
        assert expected in [child.__class__.__name__ for child in pipeline.children()]

    @pytest.mark.parametrize(
        ("name", "params"),
        [
            pytest.param("Blur", {"blur_limit": (3, 7)}, id="blur"),
            pytest.param("Sharpen", {"alpha": (0.2, 0.5)}, id="sharpen"),
            pytest.param("Equalize", {"p": 0.5}, id="equalize"),
            pytest.param("CLAHE", {"clip_limit": 4.0, "tile_grid_size": (8, 8)}, id="clahe"),
        ],
    )
    @_requires_albumentations
    def test_pixel_transforms_accepted_by_both_backends(self, name, params):
        """The same config builds on either backend, which is the gap issue #1252 reports."""
        config = {name: params}
        assert build_kornia_pipeline(config, 560) is not None

        wrappers = AlbumentationsWrapper.from_config(config)
        assert len(wrappers) == 1, (
            "from_config(strict=False) silently drops unresolved transforms, so length must be checked"
        )
        built = [t.__class__.__name__ for t in wrappers[0].transform.transforms]
        assert name in built, f"expected {name}, got {built}"

    @pytest.mark.parametrize(("blur_limit", "expected"), [(5, 5), (4, 5), ((3, 7), 7), ((3, 6), 7), (2, 3)])
    def test_blur_kernel_is_odd_and_at_least_three(self, blur_limit, expected):
        """Blur resolves its kernel the same way GaussianBlur does: odd, >= 3, upper bound of a pair."""
        pipeline = build_kornia_pipeline({"Blur": {"blur_limit": blur_limit}}, 560)
        transform = next(iter(pipeline.children()))
        assert transform.flags["kernel_size"] == (expected, expected)

    @pytest.mark.parametrize(
        ("config", "expected_message"),
        [
            # A non-degenerate pair the user chose explicitly collapses to one kernel, so it must say so.
            pytest.param({"Blur": {"blur_limit": (3, 5)}}, "Blur", id="blur-fixed-kernel"),
            # Lightness has no Kornia equivalent, so dropping it must be announced.
            pytest.param({"Sharpen": {"lightness": (0.5, 1.0)}}, "lightness", id="sharpen-ignored-lightness"),
            # mode/by_channels/mask are albumentations-only.
            pytest.param({"Equalize": {"by_channels": False}}, "by_channels", id="equalize-ignored-by_channels"),
        ],
    )
    def test_dropped_or_collapsed_options_warn_with_the_option_name(self, mocker, config, expected_message):
        """Building a config that silently collapses a range or drops an option must warn, naming what changed."""
        warning = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms.build_kornia_pipeline(config, 560)
        assert warning.called
        assert expected_message in str(warning.call_args)

    def test_blur_degenerate_pair_does_not_warn(self, mocker):
        """(5, 5) loses nothing, so it should stay quiet."""
        warning = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms.build_kornia_pipeline({"Blur": {"blur_limit": (5, 5)}}, 560)
        warning.assert_not_called()

    def test_blur_library_default_pair_logs_at_debug_not_warning(self, mocker):
        """(3, 7) is Albumentations' own Blur default, an expected divergence, so it must stay off WARNING."""
        warning = mocker.patch.object(kornia_transforms.logger, "warning")
        debug = mocker.patch.object(kornia_transforms.logger, "debug")

        kornia_transforms.build_kornia_pipeline({"Blur": {"blur_limit": (3, 7)}}, 560)

        warning.assert_not_called()
        assert debug.called
        assert "Blur" in str(debug.call_args)

    @pytest.mark.parametrize(
        "blur_limit",
        [
            pytest.param([], id="empty-sequence"),
            pytest.param((1, 2, 3), id="three-element-sequence"),
        ],
    )
    def test_blur_kernel_rejects_invalid_sequence_length(self, blur_limit):
        """A sequence that is neither a scalar nor a (min, max) pair must raise, not silently misresolve."""
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline({"Blur": {"blur_limit": blur_limit}}, 560)

        assert "Kernel size parameter must be" in str(exc_info.value)

    def test_sharpen_shifts_alpha_to_kornias_one_pivoted_sharpness_range(self):
        """Albumentations' alpha (0=no-op) is shifted to Kornia's sharpness (1.0=no-op): sharpness = 1.0 + alpha."""
        pipeline = build_kornia_pipeline({"Sharpen": {"alpha": (0.1, 0.4)}}, 560)
        transform = next(iter(pipeline.children()))
        # Kornia keeps sampled ranges on the parameter generator rather than in `flags`; see
        # `_sharpness_sampler_range` for why this reads a private attribute.
        assert _sharpness_sampler_range(transform) == pytest.approx((1.1, 1.4))

    def test_sharpen_default_alpha_sharpens_not_blurs(self):
        """Regression guard: at the default alpha=(0.2, 0.5), Sharpen must raise edge energy, not lower it.

        Kornia's ``sharpness`` is pivoted at 1.0 (0=blur, 1=no-op, >1=sharpen), Albumentations' ``alpha`` at 0. Passing
        ``alpha`` through unshifted (the pre-fix bug) gives (0.2, 0.5), below the no-op point, so a step edge blurs. The
        fixed mapping resolves to ``sharpness=(1.2, 1.5)``, which sharpens.
        """
        size = 16
        img = torch.full((1, 3, size, size), 0.3)
        img[:, :, :, size // 2 :] = 0.7  # a single sharp step edge down the middle column
        boxes = torch.tensor([[[0.0, 0.0, float(size), float(size)]]], dtype=torch.float32)

        pipeline = build_kornia_pipeline({"Sharpen": {"p": 1.0}}, 560)
        img_out, _ = pipeline(img, boxes)

        def edge_energy(x: torch.Tensor) -> float:
            # Exclude the outer 2-pixel ring: Kornia's sharpness leaves border pixels unchanged (see
            # kornia.enhance.adjust.sharpness), so including them would dilute the interior sharpening signal.
            interior = x[:, :, 2:-2, 2:-2]
            return (torch.diff(interior, dim=-1).abs().mean() + torch.diff(interior, dim=-2).abs().mean()).item()

        assert edge_energy(img_out) > edge_energy(img), (
            "Sharpen at the default alpha=(0.2, 0.5) must increase edge energy (sharpen); an unchanged or lower "
            "value means the pivot-point bug regressed (sharpness range fell back to (0.2, 0.5), which blurs)."
        )

    def test_clahe_maps_both_parameters(self):
        """clip_limit and tile_grid_size map straight onto Kornia's clip_limit and grid_size."""
        pipeline = build_kornia_pipeline({"CLAHE": {"clip_limit": (2.0, 6.0), "tile_grid_size": (4, 4)}}, 560)
        transform = next(iter(pipeline.children()))
        assert tuple(transform.flags["grid_size"]) == (4, 4)
        # Unlike Sharpen's sharpness range, RandomClahe exposes its range on a plain public
        # `clip_limit` attribute (set directly from the constructor arg), so no private access needed.
        assert tuple(transform.clip_limit) == pytest.approx((2.0, 6.0))

    @pytest.mark.parametrize(
        "configured,expected",
        [
            pytest.param(None, (1.0, 4.0), id="default"),
            pytest.param(4.0, (1.0, 4.0), id="scalar-default-value"),
            pytest.param(2.0, (1.0, 2.0), id="scalar"),
            pytest.param((1.0, 4.0), (1.0, 4.0), id="pair"),
            pytest.param((2.0, 6.0), (2.0, 6.0), id="pair-non-default"),
        ],
    )
    def test_clahe_scalar_clip_limit_is_a_range_not_a_fixed_value(
        self, configured: float | tuple[float, float] | None, expected: tuple[float, float]
    ) -> None:
        """Albumentations reads a scalar clip_limit as (1, v), so the GPU path must too.

        Passing it through `_as_range` produced the degenerate (v, v), which pins every sample to maximum contrast
        enhancement while the CPU path varies it. 4.0 is the default on both sides, so that divergence applied with no
        user config at all.
        """
        params = {} if configured is None else {"clip_limit": configured}
        pipeline = build_kornia_pipeline({"CLAHE": params}, 560)
        transform = next(iter(pipeline.children()))

        assert tuple(transform.clip_limit) == pytest.approx(expected)

    @pytest.mark.parametrize(
        "configured",
        [
            4.0,
            2.0,
            pytest.param((2.0, 6.0), id="pair"),
            pytest.param([1.0, 4.0], id="list-pair"),
        ],
    )
    @_requires_albumentations
    def test_clahe_clip_limit_matches_albumentations(
        self, configured: float | tuple[float, float] | list[float]
    ) -> None:
        """The contract stated directly: same config, same range on both backends."""
        import albumentations

        pipeline = build_kornia_pipeline({"CLAHE": {"clip_limit": configured}}, 560)
        transform = next(iter(pipeline.children()))
        cpu = albumentations.CLAHE(clip_limit=configured)

        assert tuple(transform.clip_limit) == pytest.approx(tuple(cpu.clip_limit)), (
            f"backends disagree for clip_limit={configured!r}"
        )

    @pytest.mark.parametrize(
        "configured",
        [
            pytest.param([4.0], id="one-list"),
            pytest.param((4.0,), id="one-tuple"),
            pytest.param((1.0, 2.0, 3.0), id="three"),
        ],
    )
    def test_clahe_rejects_sequences_that_albumentations_rejects(
        self, configured: tuple[float, ...] | list[float]
    ) -> None:
        """A one-element sequence is not a scalar.

        Albumentations validates `clip_limit` as a float or an exact 2-tuple and raises on `[4.0]`. Reading it as a
        scalar here would accept a config the CPU backend refuses, which is the divergence this helper exists to remove.
        """
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline({"CLAHE": {"clip_limit": configured}}, 560)

        assert "2-element" in str(exc_info.value)


@_requires_kornia
class TestRotateFactory:
    """Rotate factory translates limit (scalar or tuple) to K.RandomRotation(degrees=...)."""

    @pytest.mark.parametrize(
        ("limit", "expected"),
        [
            pytest.param(45, (-45.0, 45.0), id="scalar-is-symmetric"),
            pytest.param((90, 90), (90.0, 90.0), id="tuple-is-verbatim"),
        ],
    )
    def test_limit_maps_to_degrees(self, limit, expected):
        """A scalar ``limit`` becomes ``degrees=(-limit, limit)``; a pair is passed through."""
        pipeline = build_kornia_pipeline({"Rotate": {"limit": limit, "p": 1.0}}, 560)

        degrees = _only_child(pipeline, "RandomRotation").flags["degrees"]
        assert (float(degrees[0]), float(degrees[1])) == pytest.approx(expected, abs=0.1)

    def test_flags_include_degrees(self):
        """Rotate factory keeps a legacy degrees entry in Kornia flags for compatibility."""
        pipeline = build_kornia_pipeline({"Rotate": {"limit": 30, "p": 1.0}}, 560)

        assert _only_child(pipeline, "RandomRotation").flags["degrees"] == (-30, 30)


class TestGpuPostprocessFlag:
    """gpu_postprocess flag controls whether aug + normalize appear in CPU pipeline."""

    def test_gpu_postprocess_true_omits_aug_and_normalize_from_train(self):
        """gpu_postprocess=True: train pipeline has no CPU augmentation or Normalize."""
        pipeline_gpu = make_coco_transforms("train", 560, gpu_postprocess=True)
        pipeline_cpu = make_coco_transforms("train", 560, gpu_postprocess=False)

        steps_gpu = pipeline_gpu.transforms
        steps_cpu = pipeline_cpu.transforms

        normalize_gpu = [s for s in steps_gpu if isinstance(s, Normalize)]
        assert len(normalize_gpu) == 0, "gpu_postprocess=True must omit Normalize from train pipeline"

        assert not any(isinstance(s, RandomHorizontalFlip) for s in steps_gpu)
        assert any(isinstance(s, RandomHorizontalFlip) for s in steps_cpu)

    def test_gpu_postprocess_false_includes_aug_and_normalize_from_train(self):
        """gpu_postprocess=False (default): train pipeline includes Normalize."""
        pipeline = make_coco_transforms("train", 560, gpu_postprocess=False)
        steps = pipeline.transforms

        normalize_steps = [s for s in steps if isinstance(s, Normalize)]
        assert len(normalize_steps) > 0, "gpu_postprocess=False must include Normalize in train pipeline"

    def test_val_path_unaffected_by_gpu_postprocess(self):
        """Val pipeline is unchanged regardless of gpu_postprocess value."""
        pipeline_default = make_coco_transforms("val", 560, gpu_postprocess=False)
        pipeline_gpu = make_coco_transforms("val", 560, gpu_postprocess=True)

        # Both should have Normalize (val is never stripped)
        norm_default = [s for s in pipeline_default.transforms if isinstance(s, Normalize)]
        norm_gpu = [s for s in pipeline_gpu.transforms if isinstance(s, Normalize)]

        assert len(norm_default) > 0, "Val pipeline (default) must include Normalize"
        assert len(norm_gpu) > 0, "Val pipeline (gpu_postprocess=True) must include Normalize"

        # Same number of pipeline steps
        assert len(pipeline_default.transforms) == len(pipeline_gpu.transforms), (
            "Val pipeline step count must be identical regardless of gpu_postprocess"
        )


@_requires_kornia
class TestGaussianBlurMinKernel:
    """_make_gaussian_blur enforces kernel_size >= 3 regardless of blur_limit."""

    @pytest.mark.parametrize("blur_limit", [1, 2])
    def test_small_blur_limit_produces_valid_kernel(self, blur_limit):
        """blur_limit below 3 must be clamped so the resulting kernel_size >= 3 (blur_limit=1 once gave (3, 1))."""
        pipeline = build_kornia_pipeline({"GaussianBlur": {"blur_limit": blur_limit, "p": 1.0}}, 560)

        ks = _only_child(pipeline, "RandomGaussianBlur").flags["kernel_size"]
        assert int(ks[0]) >= 3, f"kernel_size[0]={int(ks[0])} must be >= 3"
        assert int(ks[1]) >= 3, f"kernel_size[1]={int(ks[1])} must be >= 3"

    def test_blur_limit_3_unchanged(self):
        """blur_limit=3 (default) passes through without modification."""
        pipeline = build_kornia_pipeline({"GaussianBlur": {"blur_limit": 3, "p": 1.0}}, 560)

        ks = _only_child(pipeline, "RandomGaussianBlur").flags["kernel_size"]
        assert (int(ks[0]), int(ks[1])) == (3, 3)


@_requires_kornia
class TestKorniaPipelineForwardPass:
    """build_kornia_pipeline output passes through without shape/dtype errors."""

    def test_preserves_image_shape_and_box_layout(self):
        """Pipeline output images have same shape as input; boxes shape is [B, N, 4]."""
        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=64)

        batch_size, channels, image_height, image_width = 2, 3, 64, 64
        img = torch.rand(batch_size, channels, image_height, image_width)
        boxes = torch.tensor([[[0.0, 0.0, 32.0, 32.0]], [[10.0, 10.0, 50.0, 50.0]]], dtype=torch.float32)

        img_out, boxes_out = pipeline(img, boxes)

        assert img_out.shape == (batch_size, channels, image_height, image_width), (
            f"Image shape changed: {img_out.shape}"
        )
        assert img_out.dtype == torch.float32
        assert boxes_out.shape == (batch_size, 1, 4), f"Boxes shape wrong: {boxes_out.shape}"

    def test_handles_empty_boxes(self):
        """Pipeline handles a batch where N_max=0 (no boxes) without error."""
        pipeline = build_kornia_pipeline({"HorizontalFlip": {"p": 1.0}}, resolution=32)

        batch_size, channels, image_height, image_width = 2, 3, 32, 32
        img = torch.rand(batch_size, channels, image_height, image_width)
        # [B, 0, 4] — no boxes
        boxes = torch.zeros(batch_size, 0, 4, dtype=torch.float32)

        img_out, boxes_out = pipeline(img, boxes)

        assert img_out.shape == (batch_size, channels, image_height, image_width)
        assert boxes_out.shape == (batch_size, 0, 4)


@_requires_kornia
class TestGaussNoiseStdRangeWarning:
    """_make_gauss_noise warns when the configured std range is non-degenerate (GPU uses a fixed upper-bound std)."""

    @pytest.mark.parametrize(
        ("config", "expected_calls"),
        [
            pytest.param({"std_range": (0.01, 0.05), "p": 0.5}, 1, id="unequal-range-warns"),
            pytest.param({"std_range": (0.05, 0.05), "p": 0.5}, 0, id="degenerate-range-is-silent"),
            # The albumentations default range is non-degenerate, so omitting std_range still warns.
            pytest.param({"p": 0.3}, 1, id="default-range-warns"),
        ],
    )
    def test_warns_only_for_non_degenerate_std_range(self, mocker, config, expected_calls):
        """A non-degenerate std_range emits one divergence warning at build time; an equal-bound range stays silent."""
        mock_warning = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms._make_gauss_noise(config)

        assert mock_warning.call_count == expected_calls


@_requires_kornia
class TestToGrayDroppedParamsWarning:
    """_make_to_gray warns when passed method/num_output_channels, which have no Kornia equivalent."""

    @pytest.mark.parametrize(
        ("config", "expected_calls"),
        [
            pytest.param({"method": "max", "p": 0.5}, 1, id="method"),
            pytest.param({"num_output_channels": 1, "p": 0.5}, 1, id="num_output_channels"),
            # Only p set matches the CPU path's default behavior, so nothing is dropped.
            pytest.param({"p": 0.5}, 0, id="p-only"),
        ],
    )
    def test_warns_only_when_dropped_params_are_set(self, mocker, config, expected_calls):
        """_make_to_gray warns once per build exactly when method/num_output_channels diverge from Kornia's default."""
        mock_warning = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms._make_to_gray(config)

        assert mock_warning.call_count == expected_calls


class TestResolveAugmentationBackend:
    """resolve_augmentation_backend maps backend strings to concrete AugmentationBackend members."""

    @pytest.mark.parametrize("value", ["auto", "cpu"])
    def test_falls_back_to_tv_when_no_optional_packages(self, value: str, mocker) -> None:
        """'cpu'/'auto' resolve to torchvision when neither Albumentations nor Kornia is installed."""
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is AugmentationBackend.TV)

        assert resolve_augmentation_backend(value, has_cuda=False) == AugmentationBackend.TV

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            pytest.param("kornia", "kornia", id="kornia_passthrough"),
            pytest.param("gpu", "kornia", id="gpu_alias_to_kornia"),
            pytest.param("torchvision", "torchvision", id="torchvision_passthrough"),
            pytest.param("tv", "torchvision", id="tv_alias_to_torchvision"),
        ],
    )
    def test_explicit_backend_passthrough_regardless_of_cuda(self, value: str, expected: str) -> None:
        """Explicit concrete backends and their legacy aliases ('gpu', 'tv') resolve regardless of CUDA."""
        assert resolve_augmentation_backend(value, has_cuda=False) == AugmentationBackend(expected)

    def test_albu_passthrough_when_installed(self, mocker) -> None:
        """Explicit legacy 'albu' resolves to ALBU when Albumentations is installed."""
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: True)
        assert resolve_augmentation_backend("albu", has_cuda=False) == AugmentationBackend.ALBU

    def test_albu_missing_raises_import_error(self, mocker) -> None:
        """Explicit 'albu' fails fast with an install hint when Albumentations is not installed."""
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.ALBU)

        with pytest.raises(ImportError) as exc_info:
            resolve_augmentation_backend("albu", has_cuda=False)

        assert "rfdetr[augment]" in str(exc_info.value)


class TestResolveBackendForBuild:
    """resolve_backend_for_build combines the GPU readiness fail-fast with backend resolution."""

    def test_resolves_concrete_backend_without_cuda(self) -> None:
        """A concrete non-GPU backend resolves without requiring a CUDA device."""
        assert resolve_backend_for_build("torchvision", has_cuda=False) == AugmentationBackend.TV

    def test_gpu_without_cuda_raises_runtime_error(self) -> None:
        """An explicit GPU request fails fast when no CUDA device is available."""
        with pytest.raises(RuntimeError) as exc_info:
            resolve_backend_for_build("gpu", has_cuda=False)

        assert "CUDA" in str(exc_info.value)

    def test_gpu_without_kornia_raises_import_error(self, mocker) -> None:
        """An explicit GPU request with CUDA but no Kornia fails fast with an install hint."""
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.KORNIA)

        with pytest.raises(ImportError) as exc_info:
            resolve_backend_for_build("gpu", has_cuda=True)

        assert "rfdetr[augment]" in str(exc_info.value)


class TestSplitAugConfigForGpu:
    """split_aug_config_for_gpu routes geometric entries to the CPU pre-resize step (#1769).

    A rotation, shear or perspective applied on the GPU after the square resize turns rigidly in the network input, a
    shape no camera produces, so those entries run on the CPU before the resize while flips and pixel-level transforms
    stay on the Kornia GPU pipeline.
    """

    def test_rotate_perspective_and_shift_scale_rotate_run_before_resize(self) -> None:
        """Geometry that displaces corners always runs before the resize."""
        config = {
            "Rotate": {"limit": 45, "p": 0.5},
            "Perspective": {"p": 0.5},
            "ShiftScaleRotate": {"rotate_limit": 15, "p": 0.5},
            "HorizontalFlip": {"p": 0.5},
        }

        pre_resize, gpu = split_aug_config_for_gpu(config)

        assert set(pre_resize) == {"Rotate", "Perspective", "ShiftScaleRotate"}
        assert set(gpu) == {"HorizontalFlip"}

    @pytest.mark.parametrize(
        "params",
        [
            {"rotate": (-15, 15), "p": 1.0},
            {"shear": (-5, 5), "p": 1.0},
        ],
    )
    def test_rotating_or_shearing_affine_runs_before_resize(self, params: dict[str, Any]) -> None:
        """An Affine that rotates or shears runs before the resize."""
        config = {"Affine": params}

        assert split_aug_config_for_gpu(config) == (config, {})

    @pytest.mark.parametrize(
        "params",
        [
            {"scale": (0.9, 1.1), "rotate": (0.0, 0.0), "p": 1.0},
            {"translate_percent": (-0.05, 0.05), "p": 1.0},
        ],
    )
    def test_pure_scale_or_percent_translation_affine_stays_on_gpu(self, params: dict[str, Any]) -> None:
        """An Affine with only scale or a fractional translation stays on the GPU."""
        config = {"Affine": params}

        assert split_aug_config_for_gpu(config) == ({}, config)

    def test_translate_px_affine_runs_before_resize(self) -> None:
        """A whole-pixel translate_px offset is in source-image pixels, so it runs before the resize.

        On the GPU path the same offset would be applied after the resize, in resized-image pixels — for an 800-pixel-
        wide image resized to 576, a fixed 100-pixel shift would be 72 output pixels on the CPU path but 100 on the GPU
        path.
        """
        config = {"Affine": {"translate_px": 4, "p": 1.0}}

        assert split_aug_config_for_gpu(config) == (config, {})

    def test_flip_before_rotation_keeps_configured_order(self) -> None:
        """A flip configured before a rotation stays before it on the GPU path.

        The split is a prefix cut, not a partition by key: a partition would run the rotation before the resize and the
        flip after it, reversing the tilt relative to the CPU path, where flips do not commute with rotations.
        """
        config = {"HorizontalFlip": {"p": 1.0}, "Rotate": {"limit": 30, "p": 1.0}}

        assert split_aug_config_for_gpu(config) == (config, {})

    def test_entries_before_last_geometric_entry_run_before_resize(self) -> None:
        """The prefix runs through the last geometric entry, in configured order."""
        config = {
            "ColorJitter": {"p": 0.5},
            "Rotate": {"limit": 30, "p": 1.0},
            "HorizontalFlip": {"p": 0.5},
        }

        pre_resize, gpu = split_aug_config_for_gpu(config)

        assert list(pre_resize) == ["ColorJitter", "Rotate"]
        assert list(gpu) == ["HorizontalFlip"]

    def test_unknown_key_after_rotation_stays_on_gpu(self) -> None:
        """An unknown key inside the prefix stays on the GPU so the pipeline still raises for it."""
        config = {"Rotate": {"limit": 30, "p": 1.0}, "NoSuchTransform": {"p": 1.0}}

        pre_resize, gpu = split_aug_config_for_gpu(config)

        assert set(pre_resize) == {"Rotate"}
        assert set(gpu) == {"NoSuchTransform"}

    def test_reporter_config_splits_rotation_from_flip_and_pixel_transforms(self) -> None:
        """The reporter's aug_config keeps only the flip and brightness/contrast on the GPU."""
        config = {
            "Affine": {"rotate": (-15, 15), "scale": (0.9, 1.1), "translate_percent": (-0.05, 0.05), "p": 1.0},
            "HorizontalFlip": {"p": 0.5},
            "RandomBrightnessContrast": {"p": 0.3},
        }

        pre_resize, gpu = split_aug_config_for_gpu(config)

        assert set(pre_resize) == {"Affine"}
        assert set(gpu) == {"HorizontalFlip", "RandomBrightnessContrast"}

    def test_nothing_moves_when_no_entry_rotates(self) -> None:
        """Without geometric entries the GPU config is returned unchanged."""
        config = {"HorizontalFlip": {"p": 0.5}}

        pre_resize, gpu = split_aug_config_for_gpu(config)

        assert pre_resize == {}
        assert gpu is config

    def test_unknown_keys_stay_on_gpu(self) -> None:
        """Unknown keys stay on the GPU so build_kornia_pipeline still raises for them."""
        pre_resize, gpu = split_aug_config_for_gpu({"NoSuchTransform": {"p": 1.0}})

        assert pre_resize == {}


@_requires_kornia
class TestPerspectiveFactory:
    """`Perspective` on the Kornia backend (issue #1252).

    Perspective preserves output resolution, and the DataModule carries the batch padding mask through the same Kornia
    sequence. Transforms that resize output remain unsupported because this path only transports fixed-size batches.
    """

    @pytest.mark.parametrize(
        "scale",
        [pytest.param((0.05, 0.2), id="range"), 0.2],
    )
    def test_distribution_divergence_is_always_reported(self, mocker, scale) -> None:
        """The GPU path draws uniformly where the CPU path draws a half-normal, so every config diverges.

        This holds for a scalar too: albumentations reads ``0.2`` as ``sigma`` in ``(0, 0.2)`` and samples
        ``abs(N(0, sigma))``, so even an "exact" request is not the same distribution Kornia produces.
        """
        warn = mocker.patch.object(kornia_transforms.logger, "warning")
        kornia_transforms.build_kornia_pipeline({"Perspective": {"scale": scale}}, 560)

        messages = _logged_messages(warn)
        assert any("Perspective" in m and "abs(N(0, sigma))" in m for m in messages), messages

    def test_keep_size_false_is_refused_not_ignored(self) -> None:
        """keep_size=False changes the output resolution, which this pipeline cannot express."""
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline({"Perspective": {"keep_size": False}}, 560)

        assert "keep_size=False" in str(exc_info.value)

    def test_padding_mask_follows_the_perspective_warp(self) -> None:
        """Batch padding stays an auxiliary mask under the same Perspective parameters as image and boxes."""
        torch.manual_seed(7)
        image = torch.zeros(1, 3, 64, 64)
        image[:, :, :48, :48] = 1.0
        boxes = torch.tensor([[[4.0, 4.0, 44.0, 44.0]]])
        instance_mask = torch.zeros(1, 1, 64, 64)
        instance_mask[:, :, 12:36, 12:36] = 1.0
        padding_mask = torch.ones(1, 1, 64, 64)
        padding_mask[:, :, :48, :48] = 0.0
        auxiliary_masks = torch.cat((instance_mask, padding_mask), dim=1)

        pipeline = build_kornia_pipeline({"Perspective": {"scale": 0.4, "p": 1.0}}, 560, with_masks=True)
        image_aug, boxes_aug, auxiliary_masks_aug = pipeline(image, boxes, auxiliary_masks)
        instance_mask_aug = auxiliary_masks_aug[:, :1]
        padding_mask_aug = auxiliary_masks_aug[:, 1:]

        assert image_aug.shape == image.shape
        assert boxes_aug.shape == boxes.shape
        assert padding_mask_aug.shape == padding_mask.shape
        assert instance_mask_aug.shape == instance_mask.shape
        assert not torch.equal(padding_mask_aug, padding_mask)
        assert not torch.equal(instance_mask_aug, instance_mask)
        bright_pixels = image_aug[:, 0] > 0.99
        assert not padding_mask_aug[:, 0].to(torch.bool)[bright_pixels].any()

    @pytest.mark.parametrize("name", ["RandomCrop", "CenterCrop", "RandomResizedCrop"])
    def test_crop_names_from_1252_remain_unsupported(self, name) -> None:
        """Guard for the reason Perspective ships alone: the crops resize, so they are still rejected."""
        with pytest.raises(ValueError) as exc_info:
            build_kornia_pipeline({name: {"height": 32, "width": 32}}, 560)

        assert "Unknown augmentation key" in str(exc_info.value)


@_requires_kornia
class TestGaussianDefaultsMatchAlbumentations:
    """Unspecified Gaussian defaults must align to Albumentations bounds while documenting known sampling
    differences."""

    @_requires_albumentations
    def test_blur_sigma_default_matches_albumentations(self):
        """An unspecified sigma must equal albumentations' GaussianBlur sigma_limit default."""
        import albumentations

        cpu_default = tuple(albumentations.GaussianBlur().sigma_limit)
        transform = kornia_transforms._make_gaussian_blur({"blur_limit": 3, "p": 0.3})
        gpu_default = tuple(float(v) for v in transform._param_generator.sigma)

        assert gpu_default == pytest.approx(cpu_default), (
            f"unspecified GaussianBlur sigma is {gpu_default} on the GPU path but "
            f"{cpu_default} on the CPU path, so the same config uses different default sigma bounds"
        )

    def test_blur_uses_kornia_default_when_albu_backend_is_unavailable(self, mocker):
        """A GPU-only install keeps Kornia's historic GaussianBlur default."""
        mocker.patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.ALBU)

        transform = kornia_transforms._make_gaussian_blur({"blur_limit": 3, "p": 0.3})

        assert tuple(float(value) for value in transform._param_generator.sigma) == pytest.approx((0.1, 2.0))

    def test_blur_keeps_explicit_sigma(self):
        """An explicit sigma remains authoritative over backend availability."""
        transform = kornia_transforms._make_gaussian_blur({"blur_limit": 3, "sigma": (1.25, 1.5), "p": 0.3})

        assert tuple(float(value) for value in transform._param_generator.sigma) == pytest.approx((1.25, 1.5))

    @_requires_albumentations
    def test_noise_std_default_matches_albumentations(self) -> None:
        """An unspecified std_range must equal albumentations' GaussNoise std_range default."""
        import albumentations

        cpu_default = tuple(albumentations.GaussNoise().std_range)
        # Kornia takes one std (the range's upper bound), so the default must land on the albumentations upper bound.
        transform = kornia_transforms._make_gauss_noise({"p": 1.0})
        image = torch.ones(1, 3, 16, 16)
        torch.manual_seed(0)
        expected = image + torch.randn_like(image) * cpu_default[1]
        torch.manual_seed(0)

        torch.testing.assert_close(transform(image), expected, rtol=1e-4, atol=1e-6)
