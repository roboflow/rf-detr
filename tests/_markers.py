# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared collection-time skip markers reused across multiple test modules."""

from __future__ import annotations

import sys

import pytest

#: Skip a test node that invokes CPU-backend `torch.compile` (Inductor codegen). Windows CI runners
#: have no MSVC (``cl.exe``) on ``PATH``, so Inductor's CPU C++ codegen fails with
#: ``InvalidCxxCompiler`` before the test body's own assertions ever run. CUDA-parametrized variants
#: of the same test are unaffected by this marker: the Windows CPU CI workflow already runs with
#: ``-m "not gpu"``, so they are excluded from Windows runs by their own ``@pytest.mark.gpu`` marker,
#: not by this one.
requires_cpu_inductor = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CPU Inductor needs a C++ compiler; Windows CI runners have no MSVC on PATH",
)
