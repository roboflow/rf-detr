# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Tests for best-metric tracking in ``rfdetr.training.model_ema``."""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Literal

import pytest

from rfdetr.training.model_ema import BestMetricHolder, BestMetricSingle


class TestBestMetricSingle:
    """Validation of the ``better`` argument of ``BestMetricSingle``."""

    @pytest.mark.parametrize(
        "better",
        [
            "invalid",
            "LARGE",
            None,
            pytest.param("", id="empty"),
            b"large",
            pytest.param(["large"], id="list"),
            1,
        ],
    )
    def test_rejects_invalid_better(self, better: object) -> None:
        """An unknown ``better`` value is rejected at construction."""
        with pytest.raises(ValueError, match="'better' must be 'large' or 'small'"):
            BestMetricSingle(better=better)  # type: ignore[arg-type]

    @pytest.mark.parametrize("better", ["invalid", 1])
    def test_error_message_names_offending_value(self, better: object) -> None:
        """The ``ValueError`` message shows the rejected value."""
        with pytest.raises(ValueError, match=re.escape(f"got {better!r}")):
            BestMetricSingle(better=better)  # type: ignore[arg-type]

    def test_holder_rejects_invalid_better(self) -> None:
        """``BestMetricHolder`` passes ``better`` through and rejects it too."""
        with pytest.raises(ValueError, match="'better' must be 'large' or 'small'"):
            BestMetricHolder(better="invalid")  # type: ignore[arg-type]

    def test_rejects_invalid_better_under_optimized_mode(self) -> None:
        """The check must not rely on ``assert``, which ``python -O`` strips."""
        source_root = Path(__file__).parents[2] / "src"
        code = "\n".join(
            [
                "import sys",
                "if not sys.flags.optimize:",
                "    raise SystemExit('python -O was not applied')",
                "from rfdetr.training.model_ema import BestMetricSingle",
                "try:",
                "    BestMetricSingle(better='invalid')",
                "except ValueError:",
                "    print('raised')",
            ]
        )
        environment = os.environ.copy()
        # The child does not inherit pytest's ``pythonpath``; pin it to the tree under test, not an installed copy.
        environment["PYTHONPATH"] = os.pathsep.join(filter(None, (str(source_root), environment.get("PYTHONPATH"))))
        result = subprocess.run(
            [sys.executable, "-O", "-c", code],
            check=False,
            text=True,
            capture_output=True,
            timeout=60,
            env=environment,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "raised"

    @pytest.mark.parametrize(("better", "expected"), [("large", True), ("small", False)])
    def test_accepts_valid_better(self, better: Literal["large", "small"], expected: bool) -> None:
        """Both supported directions are accepted and compare as documented."""
        metric = BestMetricSingle(better=better)
        assert metric.isbetter(1.0, 0.0) is expected
