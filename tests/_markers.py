# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared collection-time skip markers reused across multiple test modules."""

from __future__ import annotations

import importlib.util
import sys

import pytest
import torch

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

#: Skip a test node when no CUDA device is present. Pair it with ``@pytest.mark.gpu`` so GPU CI selects it and CPU CI
#: deselects it; ``cuda_marks`` is that pair for ``pytest.param(..., marks=cuda_marks)``.
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
cuda_marks = [pytest.mark.gpu, requires_cuda]

#: Skip a test node that drives a real XLA device when ``torch_xla`` is not installed (the ``xla`` extra). Pair it with
#: ``@pytest.mark.xla`` so the XLA CI job selects it.
requires_torch_xla = pytest.mark.skipif(
    importlib.util.find_spec("torch_xla") is None,
    reason="torch_xla not installed; skip XLA device tests",
)
