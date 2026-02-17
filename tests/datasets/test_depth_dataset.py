# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import pytest

from rfdetr.datasets.coco import ConvertCoco


class TestConvertCocoDepth:
    def test_include_depth_attribute(self):
        """ConvertCoco should have include_depth when set to True."""
        convert = ConvertCoco(include_masks=False, include_depth=True)
        assert hasattr(convert, "include_depth")
        assert convert.include_depth is True

    def test_include_depth_default_false(self):
        """include_depth should default to False."""
        convert = ConvertCoco(include_masks=False)
        assert convert.include_depth is False
