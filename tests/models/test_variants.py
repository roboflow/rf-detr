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
        pytest.param(RFDETRBase, "RFDETRSmall", id="base-recommends-small"),
        pytest.param(RFDETRSegPreview, "RFDETRSegSmall", id="seg-preview-recommends-small"),
    ],
)
def test_deprecated_variant_warning_names_replacement(
    deprecated_class: type[object],
    replacement: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Deprecated variant warnings recommend a size-specific replacement."""
    original_warned = deprecated_class._cfg.warned
    expected_warned = original_warned + 1
    with monkeypatch.context() as baseline_state:
        baseline_state.setattr(deprecated_class._cfg, "warned", expected_warned)
        with monkeypatch.context() as warning_state:
            warning_state.setattr(deprecated_class._cfg, "warned", 0)
            with patch("rfdetr.detr.RFDETR.__init__", return_value=None):
                with pytest.warns(FutureWarning, match=re.escape("size-specific")) as warning:
                    deprecated_class()

        assert deprecated_class._cfg.warned == expected_warned

    assert replacement in str(warning[0].message)
    assert deprecated_class._cfg.warned == original_warned
