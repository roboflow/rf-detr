# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the deprecated ``rfdetr.datasets.aug_config`` compatibility shim."""

import importlib
import sys
from types import ModuleType

import pytest

from rfdetr.datasets import aug_configs

#: Preset names the shim re-exports; each must resolve to the same object as in ``aug_configs``.
_PRESET_NAMES = ("AUG_CONFIG", "AUG_CONSERVATIVE", "AUG_AGGRESSIVE", "AUG_AERIAL", "AUG_INDUSTRIAL")


@pytest.fixture
def freshly_imported_shim() -> ModuleType:
    """Import ``rfdetr.datasets.aug_config`` with its module-level warning guaranteed to fire.

    A module-level ``warnings.warn`` runs once per process, at first import. Any earlier import — by another test, or by
    collection — would leave the module cached and make the warning unobservable here, so the cached entry is evicted
    before importing.
    """
    sys.modules.pop("rfdetr.datasets.aug_config", None)
    return importlib.import_module("rfdetr.datasets.aug_config")


class TestAugConfigShim:
    """The singular module path still works, but announces its own removal."""

    def test_import_warns_with_removal_version(self) -> None:
        """Importing the shim raises a ``FutureWarning`` naming the v1.12.0 removal target.

        The shim reappeared in v1.9.0 with an open-ended "a future release" wording and no changelog entry. Asserting on
        the concrete version keeps the deadline in the code rather than only in the migration guide, so bumping one
        without the other fails here.
        """
        sys.modules.pop("rfdetr.datasets.aug_config", None)

        with pytest.warns(FutureWarning, match=r"deprecated since v1\.9\.0 and will be removed in v1\.12\.0"):
            importlib.import_module("rfdetr.datasets.aug_config")

    def test_warning_points_at_the_replacement_module(self) -> None:
        """The deprecation message names ``rfdetr.datasets.aug_configs`` as the replacement.

        A deprecation that says only "this is going away" strands the reader; the plural module name is the one piece of
        information needed to act on the warning.
        """
        sys.modules.pop("rfdetr.datasets.aug_config", None)

        with pytest.warns(FutureWarning, match=r"Use rfdetr\.datasets\.aug_configs instead"):
            importlib.import_module("rfdetr.datasets.aug_config")

    @pytest.mark.parametrize("name", [pytest.param(preset, id=preset) for preset in _PRESET_NAMES])
    def test_reexports_preset_unchanged(self, freshly_imported_shim: ModuleType, name: str) -> None:
        """Each re-exported preset is the very same object exposed by ``aug_configs``.

        The shim exists so that existing user code keeps behaving identically until v1.12.0. Identity, not equality, is
        what guarantees that — a copied dict would silently drift from the real preset.
        """
        assert getattr(freshly_imported_shim, name) is getattr(aug_configs, name)

    def test_all_lists_every_reexported_preset(self, freshly_imported_shim: ModuleType) -> None:
        """``__all__`` covers exactly the presets the shim re-exports.

        ``from rfdetr.datasets.aug_config import *`` is the star-import path users on the old module may rely on; a
        preset missing from ``__all__`` would be importable by name but vanish under a star import.
        """
        assert sorted(freshly_imported_shim.__all__) == sorted(_PRESET_NAMES)
