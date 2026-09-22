# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for rfdetr.utilities.decorators — the deprecation re-export surface and module-move warning helper."""

from __future__ import annotations

import deprecate as _deprecate_module
import pytest

from rfdetr.utilities import decorators
from rfdetr.utilities.decorators import _warn_deprecated_module


class TestPublicSurface:
    """The module re-exports pyDeprecate's deprecation primitives unchanged."""

    def test_all_lists_the_reexported_names(self) -> None:
        """__all__ names exactly the three symbols re-exported from `deprecate`."""
        assert decorators.__all__ == ["TargetMode", "deprecated", "void"]

    @pytest.mark.parametrize("name", ["TargetMode", "deprecated", "void"])
    def test_reexport_is_the_deprecate_original(self, name: str) -> None:
        """Each re-exported name is the identical object from the `deprecate` package, not a wrapper."""
        assert getattr(decorators, name) is getattr(_deprecate_module, name)


class TestWarnDeprecatedModule:
    """_warn_deprecated_module emits the DeprecationWarning shown to callers of a moved module."""

    def test_emits_deprecation_warning_with_old_new_and_versions(self) -> None:
        """The warning message names the old and new module paths and both version numbers.

        A caller importing a module that has been relocated (e.g. `rfdetr.util.logger` moved to
        `rfdetr.utilities.logger`) needs the message to point at the replacement and state when the old path stops
        working, so it must be a DeprecationWarning carrying all four inputs verbatim.
        """
        with pytest.warns(DeprecationWarning) as record:
            _warn_deprecated_module("rfdetr.util.logger", "rfdetr.utilities.logger", "1.2.0", "2.0.0")

        assert len(record) == 1
        message = str(record[0].message)
        assert "rfdetr.util.logger is deprecated since v1.2.0" in message
        assert "removed in v2.0.0" in message
        assert "use rfdetr.utilities.logger instead" in message
