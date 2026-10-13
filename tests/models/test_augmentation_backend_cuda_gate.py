# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""CUDA gating of the Kornia pick in ``AugmentationBackend.from_str``."""

from unittest.mock import patch

import pytest

from rfdetr.config import AugmentationBackend


class TestFromStrKorniaCudaGate:
    """``"cpu"``/``"auto"`` pick Kornia only with CUDA, since the Kornia training path refuses to run without it."""

    @pytest.mark.parametrize("value", ["cpu", "auto"])
    def test_kornia_only_install_without_cuda_resolves_to_torchvision(self, value):
        """With Kornia installed, Albumentations missing and no CUDA, the sentinels fall back to torchvision.

        Resolving to Kornia here made dataset construction raise a CUDA ``RuntimeError`` on CPU-only hosts that happen
        to have Kornia installed, although the user never asked for GPU augmentation.
        """
        with patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.ALBU):
            resolved = AugmentationBackend.from_str(value, has_cuda=False)

        assert resolved is AugmentationBackend.TV

    @pytest.mark.parametrize("value", ["cpu", "auto"])
    def test_kornia_only_install_with_cuda_resolves_to_kornia(self, value):
        """With CUDA available, a Kornia-only install still resolves both sentinels to Kornia.

        The CUDA gate must not change what CUDA hosts already get.
        """
        with patch.object(AugmentationBackend, "_is_available", lambda self: self is not AugmentationBackend.ALBU):
            resolved = AugmentationBackend.from_str(value, has_cuda=True)

        assert resolved is AugmentationBackend.KORNIA
