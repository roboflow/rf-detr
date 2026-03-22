# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for optional dependency declarations in pyproject.toml."""

import pathlib
import re


class TestOptionalDependencies:
    """Validate selected extras constraints in pyproject.toml."""

    def _read_loggers_extra_block(self) -> str:
        """Return the loggers optional-dependency block from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        content = (root / "pyproject.toml").read_text()
        match = re.search(r"loggers\s*=\s*\[(.*?)\]", content, re.DOTALL)
        assert match, "loggers extra not found in [project.optional-dependencies]"
        return match.group(1)

    def test_loggers_extra_pins_protobuf_below_4(self):
        """loggers extra must constrain protobuf for TensorBoard compatibility."""
        block = self._read_loggers_extra_block()
        assert '"protobuf>=3.20.0,<4.0.0"' in block
