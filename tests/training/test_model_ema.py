# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for best-metric tracking in ``rfdetr.training.model_ema``."""

import subprocess
import sys

import pytest

from rfdetr.training.model_ema import BestMetricHolder, BestMetricSingle


class TestBestMetricSingle:
    """Validation of the ``better`` argument of ``BestMetricSingle``."""

    @pytest.mark.parametrize("better", ["invalid", "LARGE", None])
    def test_rejects_invalid_better(self, better: object) -> None:
        """An unknown ``better`` value is rejected at construction."""
        with pytest.raises(ValueError, match="'better' must be 'large' or 'small'"):
            BestMetricSingle(better=better)  # type: ignore[arg-type]

    def test_holder_rejects_invalid_better(self) -> None:
        """``BestMetricHolder`` passes ``better`` through and rejects it too."""
        with pytest.raises(ValueError, match="'better' must be 'large' or 'small'"):
            BestMetricHolder(better="invalid")

    def test_rejects_invalid_better_under_optimized_mode(self) -> None:
        """The check must not rely on ``assert``, which ``python -O`` strips."""
        code = "\n".join(
            [
                "from rfdetr.training.model_ema import BestMetricSingle",
                "try:",
                "    BestMetricSingle(better='invalid')",
                "except ValueError:",
                "    print('raised')",
            ]
        )
        result = subprocess.run([sys.executable, "-O", "-c", code], check=True, text=True, capture_output=True)
        assert result.stdout.strip() == "raised"

    @pytest.mark.parametrize(("better", "expected"), [("large", True), ("small", False)])
    def test_accepts_valid_better(self, better: str, expected: bool) -> None:
        """Both supported directions are accepted and compare as documented."""
        metric = BestMetricSingle(better=better)
        assert metric.isbetter(1.0, 0.0) is expected
