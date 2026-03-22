# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for optional dependency declarations in pyproject.toml."""

import pathlib

import tomllib
from packaging.requirements import Requirement
from packaging.version import Version


class TestOptionalDependencies:
    """Validate selected extras constraints in pyproject.toml."""

    def read_loggers_extra(self) -> list[str]:
        """Return the loggers optional-dependency list from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        loggers = pyproject["project"]["optional-dependencies"].get("loggers")
        assert loggers, "loggers extra not found in [project.optional-dependencies]"
        return loggers

    def test_loggers_extra_pins_protobuf_below_4(self):
        """loggers extra must constrain protobuf for TensorBoard compatibility."""
        requirements = [Requirement(dep) for dep in self.read_loggers_extra()]
        protobuf_requirements = [req for req in requirements if req.name == "protobuf"]
        assert protobuf_requirements, "loggers extra must include protobuf dependency"

        max_allowed_version = Version("4.0.0")

        def has_upper_bound_below_4(requirement: Requirement) -> bool:
            for spec in requirement.specifier:
                spec_version = Version(spec.version)
                if spec.operator == "<" and spec_version <= max_allowed_version:
                    return True
                if spec.operator == "<=" and spec_version < max_allowed_version:
                    return True
            return False

        assert any(has_upper_bound_below_4(req) for req in protobuf_requirements), (
            "protobuf dependency must include an upper bound below 4.0.0"
        )
