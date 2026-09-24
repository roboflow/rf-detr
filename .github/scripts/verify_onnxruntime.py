# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Verify onnx and onnxruntime are importable before the ONNX Runtime session tests.

Purpose:
    Guard against a silently-skipped test: the real-session tests are gated by
    ``pytest.importorskip("onnx")`` / ``pytest.importorskip("onnxruntime")``, so
    a broken or missing wheel would skip them (false green) instead of failing
    the job red.
Usage:
    Run ``python .github/scripts/verify_onnxruntime.py`` after installing the
    ``[onnx]`` extra.
Used by:
    ``.github/workflows/ci-integrations.yml`` (``export-parity`` matrix job,
    ``format: onnxruntime``, ``extra: onnx``).
"""

from __future__ import annotations

import sys


def main() -> int:
    """Import onnx and onnxruntime and print their versions.

    Returns:
        Process exit code; 0 once both imports succeed.

    Examples:
        >>> callable(main)
        True
    """
    import onnx
    import onnxruntime

    print(f"onnx {onnx.__version__} (IR {onnx.IR_VERSION}) / onnxruntime {onnxruntime.__version__} import OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
