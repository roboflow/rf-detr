# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Kornia-based GPU augmentation pipeline for RF-DETR training.

This module provides GPU-side augmentation as an alternative to the CPU-based Albumentations pipeline.  All transforms
run on the device where the batch already resides (typically CUDA), avoiding a CPU-GPU round-trip per sample.

Supports detection (boxes only) and segmentation (boxes + instance masks).

Usage::

    from rfdetr.datasets.kornia_transforms import (
        build_kornia_pipeline,
        build_normalize,
        collate_boxes,
        collate_masks,
        unpack_boxes,
    )

    # Detection:
    pipeline = build_kornia_pipeline(aug_config, resolution=560)
    normalize = build_normalize()
    boxes_padded, valid = collate_boxes(targets, device)
    img_aug, boxes_aug = pipeline(img, boxes_padded)
    img_aug = normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, H, W)

    # Segmentation (Phase 2):
    pipeline = build_kornia_pipeline(aug_config, resolution=560, with_masks=True)
    normalize = build_normalize()
    boxes_padded, valid = collate_boxes(targets, device)
    masks_padded = collate_masks(targets, device, n_max=valid.shape[1], image_height=H, image_width=W)
    img_aug, boxes_aug, masks_aug = pipeline(img, boxes_padded, masks_padded)
    img_aug = normalize(img_aug)
    targets = unpack_boxes(boxes_aug, valid, targets, H, W, masks_aug=masks_aug)
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import torch
from torch import Tensor

from rfdetr.config import AugmentationBackend
from rfdetr.datasets._aug_utils import filter_keypoint_hflip_augmentations
from rfdetr.utilities.logger import get_logger

logger = get_logger()

__doctest_requires__ = {"build_kornia_pipeline": ["kornia"]}

#: ImageNet channel-wise mean (RGB order).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
#: ImageNet channel-wise standard deviation (RGB order).
IMAGENET_STD = (0.229, 0.224, 0.225)

#: Albumentations' own default ``blur_limit`` for ``A.Blur``, which is a range rather than a single kernel size
#: (``A.Blur(blur_limit=7).blur_limit`` normalises to this same pair). A configured pair equal to it is the library
#: default rather than a deliberate user choice, so :func:`_as_odd_kernel` reports its collapse at ``DEBUG``.
_ALBUMENTATIONS_BLUR_LIMIT_DEFAULT: tuple[int, int] = (3, 7)

#: Threshold applied to float32 mask values produced by Kornia augmentation.
#: Kornia forces nearest-neighbour resampling for the ``"mask"`` data key, so
#: output values are already in {0.0, 1.0}; the threshold is a defensive cast.
#: Must be updated if the pipeline is ever switched to bilinear interpolation.
_MASK_BINARIZE_THRESHOLD: float = 0.5


def _has_cuda_device() -> bool:
    """Return ``True`` when the runtime has a CUDA accelerator available.

    Uses the fork-safe global ``DEVICE`` constant from ``rfdetr.config`` so that the CUDA driver context is not created
    in the main process before forking (fork-based DDP and some notebook environments).

    Returns:
        ``True`` if at least one CUDA device is reachable; ``False`` otherwise.

    Examples:
        >>> _has_cuda_device()  # doctest: +SKIP
        False
    """
    from rfdetr.config import DEVICE

    return str(DEVICE).startswith("cuda")


def resolve_augmentation_backend(backend: str, *, has_cuda: bool | None = None) -> AugmentationBackend:
    """Resolve an ``augmentation_backend`` value to a concrete :class:`AugmentationBackend`.

    Auto-pick priority (for ``"cpu"``/``"auto"``) is implemented by
    :meth:`AugmentationBackend.from_str`; this function supplies the fork-safe CUDA check that
    gates ``"auto"``'s GPU-Kornia preference and fails fast when ``"albumentations"``/``"albu"``
    is explicitly requested but Albumentations is not installed.

    This is a pure resolution step — explicit ``"kornia"``/``"gpu"`` requests always pass through
    to :attr:`AugmentationBackend.KORNIA` regardless of *has_cuda*, so a saved/forced concrete
    backend resolves deterministically. Callers that need to fail fast when an explicit GPU
    request cannot actually run (no CUDA device, or Kornia not installed) should additionally call
    :func:`require_gpu_backend_ready` before building anything that depends on the result.

    Note:
        ``"cpu"`` and ``"auto"`` resolution depends on which optional packages happen to be
        installed (Albumentations and/or Kornia are both optional, via ``pip install
        'rfdetr[augment]'``). The same ``backend`` value can therefore resolve to a different
        concrete backend across environments — e.g. CI without ``[augment]`` installed resolves
        to ``TV`` (torchvision), while a local dev environment with Albumentations installed
        resolves to ``ALBU``. Pass ``"torchvision"`` explicitly to guarantee torchvision
        regardless of environment.

    Args:
        backend: One of ``"cpu"``, ``"auto"``, ``"torchvision"``, ``"albumentations"``,
            ``"kornia"``, or legacy ``"tv"``/``"albu"``/``"gpu"``.
        has_cuda: Whether a CUDA device is available. When ``None`` (default), computed via this
            module's fork-safe :func:`_has_cuda_device`. Callers with their own fork-safe CUDA
            check (e.g. :mod:`rfdetr.training.module_data`) may pass it explicitly so patching
            their own check affects resolution.

    Returns:
        Resolved :class:`AugmentationBackend` member.

    Raises:
        ValueError: When *backend* is not a recognised value.
        ImportError: When *backend* is explicitly ``"albumentations"``/``"albu"`` but
            Albumentations is not installed.

    Examples:
        >>> resolve_augmentation_backend("albumentations")
        <AugmentationBackend.ALBU: 'albumentations'>
        >>> resolve_augmentation_backend("kornia")
        <AugmentationBackend.KORNIA: 'kornia'>
        >>> resolve_augmentation_backend("torchvision")
        <AugmentationBackend.TV: 'torchvision'>
    """
    if backend in (AugmentationBackend.ALBU, "albumentations", "albu"):
        _require_albu()
    if has_cuda is None:
        has_cuda = _has_cuda_device()
    return AugmentationBackend.from_str(backend, has_cuda=has_cuda)


def require_gpu_backend_ready(requested_backend: str, *, has_cuda: bool) -> None:
    """Fail fast when an explicit Kornia/GPU backend request cannot run in this environment.

    Only gates explicit ``"kornia"``/``"gpu"`` requests. ``"auto"``/``"cpu"`` silently fall back to
    the best installed CPU backend elsewhere (see :func:`resolve_augmentation_backend`) and are
    never gated here — an unavailable GPU is not an error for those sentinels, only for a backend
    the caller explicitly pinned to Kornia.

    Args:
        requested_backend: Raw ``augmentation_backend`` value as configured by the caller, before
            legacy-alias/auto-pick resolution.
        has_cuda: Whether a CUDA device is available, as determined by the caller's own fork-safe
            check.

    Raises:
        RuntimeError: ``kornia``/``gpu`` is explicitly requested but no CUDA device is available.
        ImportError: ``kornia``/``gpu`` is explicitly requested, CUDA is available, but Kornia is
            not installed.

    Examples:
        >>> require_gpu_backend_ready("cpu", has_cuda=False)
    """
    if requested_backend not in (AugmentationBackend.KORNIA, "kornia", "gpu"):
        return
    if not has_cuda:
        raise RuntimeError(f"augmentation_backend={requested_backend!r} requires a CUDA device, but none is available.")
    _require_kornia()


def is_gpu_postprocess(resolved: AugmentationBackend) -> bool:
    """Return ``True`` when the resolved backend defers augmentation/normalization to the GPU.

    Kornia is the only on-device (GPU) backend, so a resolved backend of :attr:`AugmentationBackend.KORNIA`
    means the CPU dataset pipeline must skip its Albumentations augmentation wrappers and ``Normalize`` step
    (both are applied later on-device). ``TV`` and ``ALBU`` keep the full CPU pipeline.

    This is the single source of truth for the ``gpu_postprocess`` flag threaded through every dataset builder;
    call it instead of re-writing ``resolved == AugmentationBackend.KORNIA`` inline so the predicate stays in one
    place.

    Args:
        resolved: A concrete backend as returned by :func:`resolve_augmentation_backend` or
            :func:`resolve_backend_for_build`.

    Returns:
        ``True`` when *resolved* is Kornia (GPU postprocessing), ``False`` otherwise.

    Examples:
        >>> from rfdetr.config import AugmentationBackend
        >>> is_gpu_postprocess(AugmentationBackend.KORNIA)
        True
        >>> is_gpu_postprocess(AugmentationBackend.TV)
        False
    """
    return resolved == AugmentationBackend.KORNIA


def resolve_backend_for_build(
    requested_backend: str | AugmentationBackend,
    *,
    has_cuda: bool | None = None,
) -> AugmentationBackend:
    """Fail fast, then resolve, an ``augmentation_backend`` value in one call for dataset builders.

    Bundles the two steps every dataset builder must perform before wiring ``gpu_postprocess``:

    1. :func:`require_gpu_backend_ready` — raise immediately when an explicit ``"kornia"``/``"gpu"`` request
       cannot actually run in this environment (no CUDA device, or Kornia not installed).
    2. :func:`resolve_augmentation_backend` — map the (possibly sentinel/legacy) value to a concrete
       :class:`AugmentationBackend` member.

    Combining them here makes the fail-fast structural rather than something each builder must remember to call
    separately, so a builder cannot silently skip the readiness check. Both steps share a single *has_cuda*
    value, computed once.

    Args:
        requested_backend: Raw ``augmentation_backend`` value as configured by the caller, before
            legacy-alias/auto-pick resolution — e.g. ``"cpu"``, ``"auto"``, ``"torchvision"``,
            ``"kornia"``, or legacy ``"gpu"``/``"tv"``/``"albu"``.
        has_cuda: Whether a CUDA device is available. When ``None`` (default), computed once via this
            module's fork-safe :func:`_has_cuda_device` and reused for both steps.

    Returns:
        Resolved :class:`AugmentationBackend` member.

    Raises:
        RuntimeError: ``"kornia"``/``"gpu"`` is explicitly requested but no CUDA device is available.
        ImportError: ``"kornia"``/``"gpu"`` is explicitly requested (with CUDA) but Kornia is not installed,
            or ``"albumentations"``/``"albu"`` is explicitly requested but Albumentations is not installed.
        ValueError: When *requested_backend* is not a recognised value.

    Examples:
        >>> resolve_backend_for_build("torchvision", has_cuda=False)
        <AugmentationBackend.TV: 'torchvision'>
    """
    if has_cuda is None:
        has_cuda = _has_cuda_device()
    require_gpu_backend_ready(requested_backend, has_cuda=has_cuda)
    return resolve_augmentation_backend(requested_backend, has_cuda=has_cuda)


def _require_kornia() -> None:
    """Verify that Kornia is importable, raising a clear error if not.

    Raises:
        ImportError: When ``kornia`` is not installed, with an install hint.
    """
    if not AugmentationBackend.KORNIA._is_available():
        raise ImportError("GPU augmentation requires kornia. Install with: pip install 'rfdetr[augment]'")


def _require_albu() -> None:
    """Verify that Albumentations is importable, raising a clear error if not.

    Raises:
        ImportError: When ``albumentations`` is not installed, with an install hint.
    """
    if not AugmentationBackend.ALBU._is_available():
        raise ImportError(
            "Custom Albumentations augmentations require albumentations. Install with: pip install 'rfdetr[augment]'"
        )


# ---------------------------------------------------------------------------
# Registry: Albumentations key -> Kornia factory
# ---------------------------------------------------------------------------


def _as_range(value: Any) -> tuple[float, float]:
    """Normalise a scalar-or-pair config value to a ``(min, max)`` tuple.

    Albumentations accepts either form for range parameters such as ``sigma`` and ``std_range``, so the builders below
    take both rather than raising a bare ``TypeError`` from inside Kornia on a config that is valid for the CPU path.
    :func:`_make_rotate` also accepts a scalar or a pair for ``limit``, but with different scalar semantics: it
    expands a scalar ``v`` symmetrically to ``(-v, v)``, whereas this helper expands it to the degenerate ``(v, v)``.

    Args:
        value: A scalar, a 1-element sequence (a degenerate ``(v, v)`` range), or a 2-element ``(min, max)`` pair.

    Returns:
        The value as a ``(min, max)`` float pair: ``(v, v)`` for a scalar or 1-element sequence, ``(min, max)`` for a
        pair.

    Raises:
        ValueError: If ``value`` is an empty sequence or has more than two elements, rather than silently dropping
            trailing elements.
    """
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            return (float(value[0]), float(value[0]))
        if len(value) == 2:
            return (float(value[0]), float(value[1]))
        raise ValueError(
            "Range parameter must be a scalar, a 1-element sequence, or a 2-element (min, max) pair; "
            f"got a {len(value)}-element sequence: {value!r}"
        )
    return (float(value), float(value))


def _as_odd_kernel(value: Any, transform: str, default_pair: tuple[int, int] | None = None) -> int:
    """Resolve an Albumentations ``blur_limit`` to a single odd Kornia kernel size.

    Albumentations samples an odd kernel from a ``(min, max)`` range per call; Kornia takes one fixed kernel size, so a
    non-degenerate pair collapses to its upper bound and the divergence is logged. The result is forced odd and at
    least 3, which Kornia requires, and to an ``int``: a float such as ``5.0`` builds but crashes at forward time.

    Args:
        value: A scalar kernel size, a 1-element sequence (a degenerate single kernel size), or a ``(min, max)`` pair.
        transform: Name used in the log message, so the log says which augmentation collapsed.
        default_pair: The Albumentations default range for ``transform``, when its default is a range rather than a
            scalar. A ``value`` equal to it is the library default rather than a deliberate user choice, so its
            collapse is logged at ``DEBUG``; every other non-degenerate pair is an explicit request the GPU path
            cannot honor and stays at ``WARNING``.

    Returns:
        An odd ``int`` kernel size of at least 3.

    Raises:
        ValueError: If ``value`` is an empty sequence or has more than two elements, rather than silently dropping
            trailing elements.
    """
    collapsed_from: tuple[Any, Any] | None = None
    if isinstance(value, (list, tuple)):
        if len(value) == 1:
            value = value[0]
        elif len(value) == 2:
            if value[0] != value[1]:
                collapsed_from = (value[0], value[1])
            value = max(value)
        else:
            raise ValueError(
                "Kernel size parameter must be a scalar, a 1-element sequence, or a 2-element (min, max) pair; "
                f"got a {len(value)}-element sequence: {value!r}"
            )
    kernel = int(value)
    if kernel % 2 == 0:
        kernel += 1
    kernel = max(3, kernel)
    if collapsed_from is not None:
        # Report the kernel actually handed to Kornia, not the pre-rounding upper bound: (3, 6) resolves to 7.
        # A pair matching the library default is an expected, documented divergence, so only a range the user
        # chose explicitly is worth a warning.
        is_library_default = default_pair is not None and collapsed_from == tuple(default_pair)
        log = logger.debug if is_library_default else logger.warning
        log(
            "GPU augmentation (Kornia) uses a fixed kernel_size=%d for %s "
            "(Kornia does not sample the kernel size per call). "
            "CPU augmentation (albumentations) samples an odd kernel from [%s, %s].",
            kernel,
            transform,
            collapsed_from[0],
            collapsed_from[1],
        )
    return kernel


def _make_horizontal_flip(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomHorizontalFlip`` from aug_config params."""
    from kornia.augmentation import RandomHorizontalFlip

    return RandomHorizontalFlip(p=params.get("p", 0.5))


def _make_vertical_flip(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomVerticalFlip`` from aug_config params."""
    from kornia.augmentation import RandomVerticalFlip

    return RandomVerticalFlip(p=params.get("p", 0.5))


def _make_rotate(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomRotation`` from aug_config params.

    The ``limit`` parameter may be a scalar (symmetric range) or a tuple.
    """
    from kornia.augmentation import RandomRotation

    limit = params.get("limit", 15)
    degrees = tuple(limit) if isinstance(limit, (list, tuple)) else (-limit, limit)
    rotation = RandomRotation(degrees=degrees, p=params.get("p", 0.5))

    # Kornia has changed the public parameter key for rotation ranges across releases.
    # Keep the legacy ``degrees`` entry available because our tests and downstream
    # callers inspect it directly.
    flags = getattr(rotation, "flags", None)
    if isinstance(flags, dict) and "degrees" not in flags:
        flags["degrees"] = degrees

    return rotation


def _make_affine(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomAffine`` from aug_config params.

    Albumentations ``translate_percent`` is a ``(min, max)`` signed range (e.g. ``(-0.1, 0.1)``).  Kornia ``translate``
    is a non-negative per-axis max fraction ``(tx, ty)`` where translation is sampled from ``[-tx, tx]``.  The
    conversion takes ``max(|min|, |max|)`` for each axis, producing a symmetric range that matches the intent.
    """
    from kornia.augmentation import RandomAffine

    translate_percent = params.get("translate_percent")
    if translate_percent is not None:
        if isinstance(translate_percent, (list, tuple)) and len(translate_percent) == 2:
            t = max(abs(translate_percent[0]), abs(translate_percent[1]))
            translate: float | tuple[float, float] | None = (t, t)
        else:
            translate = translate_percent
    else:
        translate = None

    return RandomAffine(
        degrees=params.get("rotate", (-15, 15)),
        translate=translate,
        scale=params.get("scale"),
        shear=params.get("shear"),
        p=params.get("p", 0.5),
    )


def _make_color_jitter(params: dict[str, Any]) -> Any:
    """Build a ``K.ColorJiggle`` from aug_config ``ColorJitter`` params.

    Note: Kornia >=0.7 uses ``ColorJiggle``; the ``ColorJitter`` alias was
    added in later versions.  We use ``ColorJiggle`` for broad compatibility.
    """
    from kornia.augmentation import ColorJiggle

    return ColorJiggle(
        brightness=params.get("brightness", 0.0),
        contrast=params.get("contrast", 0.0),
        saturation=params.get("saturation", 0.0),
        hue=params.get("hue", 0.0),
        p=params.get("p", 0.5),
    )


def _make_to_gray(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomGrayscale`` from aug_config ``ToGray`` params.

    Matches Albumentations' ``ToGray``: the image is converted to grayscale and kept at three channels, so it stays a
    drop-in for an RGB pipeline. Only ``p`` is honored on this (Kornia) backend: ``method`` and ``num_output_channels``
    are accepted by the CPU (albumentations) path but have no Kornia equivalent, so they are silently ignored here.
    """
    from kornia.augmentation import RandomGrayscale

    if "method" in params or "num_output_channels" in params:
        logger.warning(
            "GPU augmentation (Kornia) ToGray ignores 'method' and 'num_output_channels' "
            "(Kornia's RandomGrayscale always uses BT.601 weights and returns 3 channels). "
            "CPU augmentation (albumentations) honors both."
        )
    return RandomGrayscale(p=params.get("p", 0.5))


def _make_random_brightness_contrast(params: dict[str, Any]) -> Any:
    """Build a ``K.ColorJiggle`` from ``RandomBrightnessContrast`` params."""
    from kornia.augmentation import ColorJiggle

    return ColorJiggle(
        brightness=params.get("brightness_limit", 0.2),
        contrast=params.get("contrast_limit", 0.2),
        p=params.get("p", 0.5),
    )


def _make_gaussian_blur(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomGaussianBlur`` from aug_config params.

    Both ``blur_limit`` and ``sigma`` accept a scalar or a ``(min, max)`` pair, since Albumentations accepts either and
    a config written for the CPU path should not fail here, but a pair resolves asymmetrically: ``blur_limit`` takes the
    pair's upper bound (Kornia uses a single kernel size), rounded up to an odd integer, while ``sigma`` is passed
    through as a real ``(min, max)`` range. A non-degenerate ``blur_limit`` pair therefore collapses to fixed maximum
    blur and logs a warning.
    """
    from kornia.augmentation import RandomGaussianBlur

    # Shared with Blur: a (min, max) pair collapses to its upper bound, forced odd and >= 3.
    blur_limit = _as_odd_kernel(params.get("blur_limit", 3), "GaussianBlur")
    if "sigma" in params:
        sigma_range = params["sigma"]
    else:
        # The supported Albumentations range has version-dependent defaults. Read the installed CPU transform so a
        # shared config stays aligned whenever both optional augmentation backends are available; otherwise retain
        # Kornia's original default for GPU-only installations. AUG_INDUSTRIAL reaches this branch.
        try:
            import albumentations
        except ModuleNotFoundError as error:
            if error.name != "albumentations":
                raise
            sigma_range = (0.1, 2.0)
        else:
            sigma_range = albumentations.GaussianBlur().sigma_limit
    blur_sigma = _as_range(sigma_range)
    return RandomGaussianBlur(
        kernel_size=(blur_limit, blur_limit),
        sigma=blur_sigma,
        p=params.get("p", 0.5),
    )


def _make_gauss_noise(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomGaussianNoise`` from aug_config params.

    Kornia takes a single ``std`` value, so the upper bound of ``std_range`` is used as a fixed standard deviation. When
    the configured range is non-degenerate this diverges from the CPU (albumentations) path, which samples a fresh std
    per call; a warning is emitted at build time so the drift is visible.
    """
    from kornia.augmentation import RandomGaussianNoise

    # Default to the CPU (albumentations) GaussNoise std_range default, (0.2, 0.44), on the 0-1 image
    # scale both backends use. Kornia still fixes its std at the upper bound, while the CPU path samples
    # the range; the warning below makes that unavoidable distribution difference visible.
    std_range = _as_range(params.get("std_range", (0.2, 0.44)))
    if std_range[0] != std_range[1]:
        logger.warning(
            "GPU augmentation (Kornia) uses fixed std=%.3f for GaussianNoise "
            "(Kornia does not support per-sample std ranges). "
            "CPU augmentation (albumentations) samples from [%.3f, %.3f].",
            std_range[1],
            std_range[0],
            std_range[1],
        )
    return RandomGaussianNoise(
        std=std_range[1],
        p=params.get("p", 0.5),
    )


def _make_blur(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomBoxBlur`` from aug_config ``Blur`` params.

    Albumentations' ``Blur`` is a box (average) blur, which is what ``RandomBoxBlur`` applies, so this is a direct
    mapping. ``blur_limit`` resolves the same way as for ``GaussianBlur``: a non-degenerate pair collapses to its upper
    bound because Kornia takes a single kernel size. Unlike ``GaussianBlur``, Albumentations' own default here is a
    range rather than a scalar, so the default collapse is expected rather than a misconfiguration and is reported at
    ``DEBUG``; a range the user set explicitly still warns.
    """
    from kornia.augmentation import RandomBoxBlur

    kernel = _as_odd_kernel(
        params.get("blur_limit", _ALBUMENTATIONS_BLUR_LIMIT_DEFAULT),
        "Blur",
        default_pair=_ALBUMENTATIONS_BLUR_LIMIT_DEFAULT,
    )
    return RandomBoxBlur(kernel_size=(kernel, kernel), p=params.get("p", 0.5))


def _make_sharpen(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomSharpness`` from aug_config ``Sharpen`` params.

    The two libraries use different origins for the same effect, so ``alpha`` is shifted rather than passed through.
    Albumentations' ``alpha`` is the visibility of the sharpened image: ``0`` leaves the image unchanged and ``1``
    shows the fully sharpened version, so it never blurs. Kornia's ``sharpness`` factor is pivoted at ``1.0`` (the
    PIL ``ImageEnhance.Sharpness`` convention): it blends from a smoothed copy at ``0`` through the untouched image
    at ``1.0`` and sharpens only above ``1.0``. Passing ``alpha`` through unchanged would therefore blur the image
    for every value below ``1``, so the resolved range is shifted with ``sharpness = 1.0 + alpha``, which keeps the
    Albumentations no-op at ``alpha = 0`` mapped to Kornia's no-op at ``sharpness = 1.0``.

    ``lightness`` and ``method`` have no Kornia equivalent and are ignored here; the CPU (albumentations) path honors
    both.
    """
    from kornia.augmentation import RandomSharpness

    if "lightness" in params or "method" in params:
        logger.warning(
            "GPU augmentation (Kornia) Sharpen ignores 'lightness' and 'method' "
            "(Kornia's RandomSharpness exposes only a sharpness factor). "
            "CPU augmentation (albumentations) honors both."
        )
    alpha_min, alpha_max = _as_range(params.get("alpha", (0.2, 0.5)))
    return RandomSharpness(
        sharpness=(1.0 + alpha_min, 1.0 + alpha_max),
        p=params.get("p", 0.5),
    )


def _make_equalize(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomEqualize`` from aug_config ``Equalize`` params.

    Only ``p`` is honored. ``mode``, ``by_channels`` and ``mask`` are accepted by the CPU (albumentations) path but
    have no Kornia equivalent: ``RandomEqualize`` always equalizes every channel and takes no mask.
    """
    from kornia.augmentation import RandomEqualize

    if any(key in params for key in ("mode", "by_channels", "mask")):
        logger.warning(
            "GPU augmentation (Kornia) Equalize ignores 'mode', 'by_channels' and 'mask' "
            "(Kornia's RandomEqualize always equalizes all channels and takes no mask). "
            "CPU augmentation (albumentations) honors them."
        )
    return RandomEqualize(p=params.get("p", 0.5))


def _make_clahe(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomClahe`` from aug_config ``CLAHE`` params.

    Both parameters map directly: Albumentations' ``clip_limit`` (a scalar or a pair) becomes Kornia's ``clip_limit``
    range, and ``tile_grid_size`` becomes ``grid_size``.
    """
    import kornia.augmentation as kornia_augmentation

    random_clahe = cast(Any, kornia_augmentation).RandomClahe
    grid = params.get("tile_grid_size", (8, 8))
    return random_clahe(
        clip_limit=_as_range(params.get("clip_limit", 4.0)),
        grid_size=(int(grid[0]), int(grid[1])),
        p=params.get("p", 0.5),
    )


#: Albumentations ``Perspective`` options with no ``K.RandomPerspective`` equivalent.
_PERSPECTIVE_IGNORED_KEYS = (
    "fit_output",
    "interpolation",
    "mask_interpolation",
    "border_mode",
    "fill",
    "fill_mask",
)


def _make_perspective(params: dict[str, Any]) -> Any:
    """Build a ``K.RandomPerspective`` from aug_config ``Perspective`` params.

    Both libraries displace the corners by a fraction of the image side, but they do not sample that fraction the same
    way, so this is an approximation rather than a parameter rename. Albumentations treats ``scale`` as the standard
    deviation of a normal and takes ``abs(N(0, sigma))`` per corner (normalizing a scalar ``v`` to ``(0, v)``), so small
    displacements dominate and large ones are possible but rare. Kornia draws uniformly from ``[0, distortion_scale]``.
    Passing the upper bound of ``scale`` as ``distortion_scale`` keeps the worst-case distortion roughly aligned while
    making the typical distortion noticeably stronger on the GPU path; there is no setting that makes the two
    distributions equal. The divergence is logged so a run does not silently change character when it moves onto the
    GPU.

    Other Albumentations ``Perspective`` options (``fit_output``, ``interpolation``, ``mask_interpolation``,
    ``border_mode``, ``fill``, ``fill_mask``) have no ``RandomPerspective`` equivalent and are ignored with a warning
    when set to a non-default value.

    ``keep_size`` is not accepted. Albumentations defaults it to ``True`` (the output keeps the input's height and
    width) and Kornia's ``RandomPerspective`` always behaves that way, so the default maps cleanly; ``keep_size=False``
    would change the output resolution, which this pipeline cannot express (see the note in
    :func:`build_kornia_pipeline` about size-preserving transforms), so it is refused rather than silently ignored.
    """
    from kornia.augmentation import RandomPerspective

    if params.get("keep_size") is False:
        raise ValueError(
            "Perspective(keep_size=False) is not supported on the Kornia GPU backend: it changes the output "
            "resolution, but the GPU augmentation path requires a fixed batch height and width. Use "
            "keep_size=True (the Albumentations default), or run this augmentation on the CPU "
            "(albumentations) backend."
        )

    ignored = [k for k in _PERSPECTIVE_IGNORED_KEYS if k in params]
    if ignored:
        logger.warning(
            "GPU augmentation (Kornia) Perspective ignores %s "
            "(Kornia's RandomPerspective exposes only distortion_scale and p). "
            "CPU augmentation (albumentations) honors them.",
            ", ".join(repr(k) for k in ignored),
        )

    # Albumentations reads a scalar ``scale`` as ``(0, v)`` rather than ``(v, v)``, so it is normalized here
    # instead of going through ``_as_range``; otherwise the reported CPU-side range would be wrong.
    raw_scale = params.get("scale", (0.05, 0.1))
    scale = (0.0, float(raw_scale)) if isinstance(raw_scale, (int, float)) else _as_range(raw_scale)
    logger.warning(
        "GPU augmentation (Kornia) Perspective uses distortion_scale=%.3f sampled uniformly from "
        "[0, %.3f]. CPU augmentation (albumentations) samples each corner offset from abs(N(0, sigma)) "
        "over sigma in [%.3f, %.3f], so the GPU path distorts more on a typical sample. "
        "Use the albumentations backend if the exact distribution matters.",
        scale[1],
        scale[1],
        scale[0],
        scale[1],
    )
    return RandomPerspective(
        distortion_scale=scale[1],
        p=params.get("p", 0.5),
    )


_REGISTRY: dict[str, Callable[[dict[str, Any]], Any]] = {
    "HorizontalFlip": _make_horizontal_flip,
    "VerticalFlip": _make_vertical_flip,
    "Rotate": _make_rotate,
    "Affine": _make_affine,
    "ColorJitter": _make_color_jitter,
    "ToGray": _make_to_gray,
    "RandomBrightnessContrast": _make_random_brightness_contrast,
    "GaussianBlur": _make_gaussian_blur,
    "GaussNoise": _make_gauss_noise,
    "Blur": _make_blur,
    "Sharpen": _make_sharpen,
    "Equalize": _make_equalize,
    "CLAHE": _make_clahe,
    "Perspective": _make_perspective,
}


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_kornia_pipeline(
    aug_config: dict[str, dict[str, Any]],
    resolution: int,
    with_masks: bool = False,
    include_keypoints: bool = False,
) -> Any:
    """Build a Kornia ``AugmentationSequential`` from an aug_config dict.

    Each key in *aug_config* is looked up in ``_REGISTRY`` and instantiated with the corresponding parameter dict.
    Unknown keys raise ``ValueError``.

    Args:
        aug_config: Mapping of augmentation names to parameter dicts, identical
            to the format accepted by the Albumentations path (e.g. ``{"HorizontalFlip": {"p": 0.5}}``).
        resolution: Target image resolution in pixels (currently reserved for
            future resolution-aware augmentations).
        with_masks: When ``True``, include ``"mask"`` in ``data_keys`` so
            auxiliary masks are augmented in sync with images and boxes. The training DataModule always enables this
            to transport its padding mask; segmentation batches concatenate instance-mask channels before the final
            padding channel. The pipeline then expects three inputs ``(img, boxes, masks)`` and returns three outputs.
            Defaults to ``False`` for direct detection-only callers.
        include_keypoints: When ``True``, keypoint-unsafe horizontal-flip
            transforms are dropped with a warning before the Kornia pipeline is built.

    Returns:
        A ``kornia.augmentation.AugmentationSequential`` instance.

    Raises:
        ValueError: If *aug_config* contains an unsupported augmentation key.

    Examples:
        >>> from rfdetr.datasets.aug_configs import AUG_CONSERVATIVE
        >>> pipeline = build_kornia_pipeline(AUG_CONSERVATIVE, resolution=560)
        >>> pipeline_seg = build_kornia_pipeline(AUG_CONSERVATIVE, resolution=560, with_masks=True)
    """
    _require_kornia()
    from kornia.augmentation import AugmentationSequential

    filtered_aug_config = filter_keypoint_hflip_augmentations(
        aug_config,
        include_keypoints=include_keypoints,
        warn=logger.warning,
    )
    assert isinstance(filtered_aug_config, dict)

    transforms: list[Any] = []
    for name, params in filtered_aug_config.items():
        factory = _REGISTRY.get(name)
        if factory is None:
            raise ValueError(
                f"Unknown augmentation key {name!r} for Kornia GPU backend. Supported keys: {sorted(_REGISTRY)}."
            )
        transforms.append(factory(params))

    data_keys = ["input", "bbox_xyxy", "mask"] if with_masks else ["input", "bbox_xyxy"]
    return AugmentationSequential(
        *transforms,
        data_keys=data_keys,
    )


def build_normalize(
    mean: tuple[float, ...] = IMAGENET_MEAN,
    std: tuple[float, ...] = IMAGENET_STD,
) -> Any:
    """Build a Kornia ``Normalize`` transform for GPU-side normalization.

    Args:
        mean: Per-channel mean values.  Defaults to ImageNet statistics.
        std: Per-channel standard deviation values.  Defaults to ImageNet
            statistics.

    Returns:
        A ``kornia.augmentation.Normalize`` instance.
    """
    _require_kornia()
    from kornia.augmentation import Normalize

    return Normalize(
        mean=mean,
        std=std,
    )


# ---------------------------------------------------------------------------
# Bounding-box utilities
# ---------------------------------------------------------------------------


def collate_boxes(
    targets: list[dict[str, Any]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Pack variable-length xyxy boxes into a padded tensor and valid mask.

    Kornia ``AugmentationSequential`` expects boxes as ``[B, N_max, 4]``. This function zero-pads each image's boxes to
    the maximum count in the batch and returns a boolean mask indicating which entries are real.

    Args:
        targets: List of target dicts (one per image), each containing a
            ``"boxes"`` key with an ``[N_i, 4]`` tensor in xyxy format.
        device: Device on which to allocate the output tensors.

    Returns:
        Tuple of:
            - ``boxes_padded`` — ``[B, N_max, 4]`` float tensor (zero-padded).
            - ``valid_mask``   — ``[B, N_max]`` bool tensor (``True`` = real box).

        When ``B == 0`` or all images have zero boxes, both tensors have ``N_max == 0``.
    """
    if len(targets) == 0:
        return (
            torch.zeros(0, 0, 4, device=device),
            torch.zeros(0, 0, dtype=torch.bool, device=device),
        )

    box_counts = [t["boxes"].shape[0] for t in targets]
    n_max = max(box_counts) if box_counts else 0
    batch_size = len(targets)

    if n_max == 0:
        return (
            torch.zeros(batch_size, 0, 4, device=device),
            torch.zeros(batch_size, 0, dtype=torch.bool, device=device),
        )

    boxes_padded = torch.zeros(batch_size, n_max, 4, device=device)
    valid_mask = torch.zeros(batch_size, n_max, dtype=torch.bool, device=device)

    for i, t in enumerate(targets):
        n = t["boxes"].shape[0]
        if n > 0:
            boxes_padded[i, :n] = t["boxes"]
            valid_mask[i, :n] = True

    return boxes_padded, valid_mask


def collate_masks(
    targets: list[dict[str, Any]],
    device: torch.device,
    n_max: int,
    image_height: int,
    image_width: int,
) -> Tensor:
    """Pack variable-length instance masks into a zero-padded ``[B, N_max, H, W]`` tensor.

    Kornia ``AugmentationSequential`` expects masks as ``[B, N_max, H, W]`` when ``data_keys`` includes ``"mask"``.
    This function zero-pads each image's masks to *n_max* channels (matching the padding used by :func:`collate_boxes`)
    and converts boolean masks to ``float32`` for Kornia compatibility.

    Args:
        targets: List of target dicts (one per image).  Each dict may optionally
            contain a ``"masks"`` key with an ``[N_i, H, W]`` boolean tensor. Dicts without the key are treated as
            having zero instances.
        device: Device on which to allocate the output tensor.
        n_max: Maximum instance count across the batch — must equal
            ``collate_boxes(targets, device)[1].shape[1]`` to keep box/mask indices in sync.
        image_height: Spatial height ``H`` of each mask (pixels).
        image_width: Spatial width ``W`` of each mask (pixels).

    Returns:
        Float32 tensor of shape ``[B, N_max, H, W]``, zero-padded where ``N_i < N_max``.  Boolean input masks are cast
        to ``float32`` (``True → 1.0``, ``False → 0.0``).

    Examples:
        >>> import torch
        >>> targets = [{"masks": torch.ones(2, 8, 8, dtype=torch.bool)}]
        >>> out = collate_masks(targets, torch.device("cpu"), n_max=2, image_height=8, image_width=8)
        >>> out.shape
        torch.Size([1, 2, 8, 8])
        >>> out.dtype
        torch.float32
    """
    batch_size = len(targets)
    masks_padded = torch.zeros(batch_size, n_max, image_height, image_width, dtype=torch.float32, device=device)
    for i, t in enumerate(targets):
        if "masks" not in t or n_max == 0:
            continue
        masks_i = t["masks"].to(dtype=torch.float32, device=device)  # [N_i, H, W]
        n = min(masks_i.shape[0], n_max)
        if n > 0:
            masks_padded[i, :n] = masks_i[:n]
    return masks_padded


def unpack_boxes(
    boxes_aug: Tensor,
    valid: Tensor,
    targets: list[dict[str, Any]],
    image_height: int,
    image_width: int,
    masks_aug: Tensor | None = None,
) -> list[dict[str, Any]]:
    """Unpack augmented boxes (and optionally masks), clamp to image bounds, remove zero-area boxes.

    After Kornia augmentation the padded ``[B, N_max, 4]`` tensor is unpacked back into per-image target dicts.  Boxes
    are clamped to ``[0, W] x [0, H]`` and any that collapse to zero area are removed along with their corresponding
    ``labels``, ``area``, ``iscrowd``, and (if provided) ``masks`` entries.

    Args:
        boxes_aug: Augmented boxes tensor ``[B, N_max, 4]`` in xyxy format.
        valid: Boolean mask ``[B, N_max]`` from :func:`collate_boxes`.
        targets: Original target dicts; each dict is shallow-copied before
            modification — the input list itself is not mutated.
        image_height: Image height in pixels (for clamping).
        image_width: Image width in pixels (for clamping).
        masks_aug: Optional augmented masks tensor ``[B, N_max, H, W]``
            (float32) from Kornia.  When provided, masks are filtered by the same ``keep`` mask as boxes, thresholded at
            ``> 0.5`` to bool, and stored under ``"masks"`` in each output target dict.  When ``None``, any existing
            ``"masks"`` entry in the target dict is preserved unchanged.

    Returns:
        A new list of target dicts with updated ``boxes``, ``labels``, ``area``, ``iscrowd``, and (when *masks_aug* is
        given) ``masks`` entries.
    """
    if masks_aug is not None:
        assert masks_aug.shape[:2] == valid.shape, (
            f"masks_aug batch/n_max dims {tuple(masks_aug.shape[:2])} must match "
            f"valid shape {tuple(valid.shape)}; ensure collate_masks is called with "
            "n_max=valid.shape[1] from collate_boxes"
        )
    new_targets: list[dict[str, Any]] = []
    for i, t in enumerate(targets):
        t = t.copy()
        n_orig = t["boxes"].shape[0]

        if n_orig == 0 or valid.shape[1] == 0:
            new_targets.append(t)
            continue

        # Extract valid boxes for this image
        v = valid[i, :n_orig]
        boxes_i = boxes_aug[i, :n_orig]

        # Clamp to image boundaries
        boxes_i = boxes_i.clone()
        boxes_i[:, 0].clamp_(min=0, max=image_width)
        boxes_i[:, 1].clamp_(min=0, max=image_height)
        boxes_i[:, 2].clamp_(min=0, max=image_width)
        boxes_i[:, 3].clamp_(min=0, max=image_height)

        # Remove zero-area boxes (after clamping)
        widths = boxes_i[:, 2] - boxes_i[:, 0]
        heights = boxes_i[:, 3] - boxes_i[:, 1]
        keep = v & (widths > 0) & (heights > 0)

        t["boxes"] = boxes_i[keep]
        if "labels" in t:
            t["labels"] = t["labels"][keep]
        if "area" in t:
            # Recompute area from clamped boxes
            kept_boxes = t["boxes"]
            t["area"] = (kept_boxes[:, 2] - kept_boxes[:, 0]) * (kept_boxes[:, 3] - kept_boxes[:, 1])
        if "iscrowd" in t:
            t["iscrowd"] = t["iscrowd"][keep]
        if masks_aug is not None:
            masks_i = masks_aug[i, :n_orig]  # [N_orig, H, W]
            t["masks"] = masks_i[keep] > _MASK_BINARIZE_THRESHOLD
        # TODO(keypoints): First public keypoint preview keeps keypoint coordinates unchanged through GPU augmentation
        # to preserve existing training paths without introducing partial geometry transforms. Add keypoint-aware
        # Kornia unpack/keep logic once augmentation parity is implemented.

        new_targets.append(t)

    return new_targets
