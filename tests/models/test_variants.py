# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import re
from unittest.mock import patch

import pytest

from rfdetr.variants import RFDETRBase, RFDETRSegPreview


@pytest.mark.parametrize(
    ("deprecated_class", "replacement"),
    [
        (RFDETRBase, "RFDETRSmall"),
        (RFDETRSegPreview, "RFDETRSegSmall"),
    ],
)
def test_deprecated_variant_warning_names_replacement(
    deprecated_class: type[object],
    replacement: str,
) -> None:
    """Deprecated variant warnings recommend a size-specific replacement."""
    deprecated_class._cfg.warned = 0
    with patch("rfdetr.detr.RFDETR.__init__", return_value=None):
        with pytest.warns(FutureWarning, match=re.escape("size-specific")) as warning:
            deprecated_class()

    assert replacement in str(warning[0].message)
