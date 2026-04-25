# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

"""Tests for optional dependency declarations in pyproject.toml."""

import pathlib

import pytest
from packaging.requirements import Requirement
from packaging.version import Version

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib


class TestOptionalDependencies:
    """Validate selected extras constraints in pyproject.toml."""

    @staticmethod
    def read_extra(name: str) -> list[str]:
        """Return an optional-dependency list from pyproject.toml."""
        root = pathlib.Path(__file__).parent.parent.parent
        pyproject = tomllib.loads((root / "pyproject.toml").read_text())
        extra = pyproject["project"]["optional-dependencies"].get(name)
        assert extra, f"{name} extra not found in [project.optional-dependencies]"
        return extra

    @staticmethod
    def has_upper_bound_below_4(requirement: Requirement) -> bool:
        """Return True if the requirement specifier caps the version below 4.0.0."""
        max_allowed_version = Version("4.0.0")
        for spec in requirement.specifier:
            spec_version = Version(spec.version)
            if spec.operator == "<" and spec_version <= max_allowed_version:
                return True
            if spec.operator == "<=" and spec_version < max_allowed_version:
                return True
        return False

    def test_loggers_extra_pins_protobuf_below_4(self):
        """loggers extra must constrain protobuf for TensorBoard compatibility."""
        requirements = [Requirement(dep) for dep in self.read_extra("loggers")]
        protobuf_requirements = [req for req in requirements if req.name == "protobuf"]
        assert protobuf_requirements, "loggers extra must include protobuf dependency"

        assert any(self.has_upper_bound_below_4(req) for req in protobuf_requirements), (
            "protobuf dependency must include an upper bound below 4.0.0"
        )

    def test_headless_extra_uses_headless_opencv(self) -> None:
        """headless extra must avoid declaring the GUI OpenCV distribution."""
        requirements = [Requirement(dep) for dep in self.read_extra("headless")]
        requirement_names = {req.name for req in requirements}

        assert "opencv-python-headless" in requirement_names
        assert "opencv-python" not in requirement_names

    def test_headless_extra_constrains_supervision_before_opencv_switch(self) -> None:
        """headless extra must use the Supervision line compatible with headless OpenCV."""
        requirements = [Requirement(dep) for dep in self.read_extra("headless")]
        supervision_requirements = [req for req in requirements if req.name == "supervision"]
        assert supervision_requirements, "headless extra must constrain supervision"

        assert any(
            req.specifier.contains(Version("0.20.0")) and not req.specifier.contains(Version("0.21.0"))
            for req in supervision_requirements
        )

    @pytest.mark.parametrize(
        "dep_str,expected",
        [
            pytest.param("protobuf>=3.20.0,<4.0.0", True, id="strict-upper-bound"),
            pytest.param("protobuf>=3.20.0", False, id="lower-bound-only"),
            pytest.param("protobuf>=3.20.0,<5.0.0", False, id="too-permissive-upper"),
            pytest.param("protobuf<=3.9.0", True, id="le-bound-below-4"),
        ],
    )
    def test_has_upper_bound_below_4(self, dep_str: str, expected: bool) -> None:
        """has_upper_bound_below_4 correctly identifies specifiers bounded below 4.0.0."""
        assert self.has_upper_bound_below_4(Requirement(dep_str)) == expected
