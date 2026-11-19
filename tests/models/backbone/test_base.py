# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for the BackboneBase subclass contract."""

from __future__ import annotations

import argparse

import pytest

from rfdetr.models.backbone.base import BackboneBase


class TestBackboneBase:
    """Tests the parameter-free base module and its subclass override contract."""

    def test_is_parameter_free_module(self) -> None:
        """A bare BackboneBase is a valid nn.Module with no parameters of its own.

        BackboneBase carries no layers itself -- it only defines the get_named_param_lr_pairs contract that concrete
        backbones such as Backbone (backbone.py) override. Instantiating it directly must still succeed as a normal
        nn.Module.
        """
        base = BackboneBase()

        assert list(base.parameters()) == []

    def test_get_named_param_lr_pairs_raises_not_implemented(self) -> None:
        """The unimplemented base method signals subclasses must override it.

        Concrete backbones (e.g. Backbone.get_named_param_lr_pairs) override this method; calling it on the base class
        directly must fail loudly with NotImplementedError rather than silently returning nothing.
        """
        base = BackboneBase()

        with pytest.raises(NotImplementedError):
            base.get_named_param_lr_pairs(argparse.Namespace(), prefix="backbone.0")
