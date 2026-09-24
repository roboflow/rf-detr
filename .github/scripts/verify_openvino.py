# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify openvino is importable before running the OpenVINO parity tests.

Purpose:
    Guard against a silently-skipped parity test: ``TestOpenVINOEndToEnd`` is
    gated by ``pytest.importorskip("openvino")``, so a broken or missing wheel
    would skip it (false green) instead of failing the job red.
Usage:
    Run ``python .github/scripts/verify_openvino.py`` after installing the
    ``[openvino]`` extra.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: openvino``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import openvino and assert it is detected as available.

    Returns:
        Process exit code; 0 once the import check passes.

    Examples:
        >>> callable(main)
        True
    """
    from rfdetr.export._openvino import _IS_OPENVINO_AVAILABLE

    assert _IS_OPENVINO_AVAILABLE, "openvino installed but not detected as available"
    print("openvino import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
