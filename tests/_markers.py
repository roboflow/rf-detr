# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared collection-time skip markers reused across multiple test modules."""

from __future__ import annotations

import os
import sys

import pytest
import torch

from rfdetr.export.imports import _IS_ONNX_INSTALLED, _IS_ONNXRUNTIME_INSTALLED, _IS_OPENVINO_INSTALLED
from rfdetr.utilities.imports import (
    _IS_ALBUMENTATIONS_INSTALLED,
    _IS_FASTER_COCO_EVAL_INSTALLED,
    _IS_HOTCOCO_INSTALLED,
    _IS_KORNIA_INSTALLED,
    _IS_PYCOCOTOOLS_INSTALLED,
    _IS_TORCH_XLA_INSTALLED,
    _IS_UFCOCO_INSTALLED,
    _IS_VERNIER_INSTALLED,
)

#: Skip a test node that invokes CPU-backend `torch.compile` (Inductor codegen). Windows CI runners
#: have no MSVC (``cl.exe``) on ``PATH``, so Inductor's CPU C++ codegen fails with
#: ``InvalidCxxCompiler`` before the test body's own assertions ever run. CUDA-parametrized variants
#: of the same test are unaffected by this marker: the Windows CPU CI workflow already runs with
#: ``-m "not gpu"``, so they are excluded from Windows runs by their own ``@pytest.mark.gpu`` marker,
#: not by this one.
_requires_cpu_inductor = pytest.mark.skipif(
    sys.platform == "win32",
    reason="CPU Inductor needs a C++ compiler; Windows CI runners have no MSVC on PATH",
)

#: Skip a test node when no CUDA device is present. Pair it with ``@pytest.mark.gpu`` so GPU CI selects it and CPU CI
#: deselects it; ``_cuda_marks`` is that pair for ``pytest.param(..., marks=_cuda_marks)``.
_requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
_cuda_marks = [pytest.mark.gpu, _requires_cuda]

#: Skip a test node when no Apple MPS device is present. MPS runs on the macOS CPU CI runners, so no ``gpu`` marker.
_requires_mps = pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")

#: Skip a test node that drives a real XLA device when ``torch_xla`` is not installed (the ``xla`` extra). Pair it with
#: ``@pytest.mark.xla`` so the XLA CI job selects it.
_requires_torch_xla = pytest.mark.skipif(
    not _IS_TORCH_XLA_INSTALLED,
    reason="torch_xla not installed; skip XLA device tests",
)

#: Skip a test node that needs the ``onnx`` package (graph construction, export, checker).
_onnx_only = pytest.mark.skipif(not _IS_ONNX_INSTALLED, reason="onnx not installed; skip ONNX tests")

#: Skip a test node that needs ``onnxruntime`` alone (running a model it was handed).
_onnxruntime_only = pytest.mark.skipif(
    not _IS_ONNXRUNTIME_INSTALLED, reason="onnxruntime not installed; skip ONNX Runtime tests"
)

#: Skip a test node that builds a graph with ``onnx`` and runs it with ``onnxruntime``.
_onnx_and_onnxruntime_only = pytest.mark.skipif(
    not (_IS_ONNX_INSTALLED and _IS_ONNXRUNTIME_INSTALLED),
    reason="onnx/onnxruntime not installed; skip ONNX inference tests",
)

#: Skip a test node that needs two CUDA devices, such as a model and its input on different GPUs.
_requires_two_cuda_devices = pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two CUDA devices")

#: Skip a test node that checks POSIX file permission bits or ownership, which Windows does not have.
_requires_posix_permissions = pytest.mark.skipif(os.name == "nt", reason="POSIX file permissions")

#: Skip a test node that creates a named pipe; ``os.mkfifo`` exists only on POSIX.
_requires_mkfifo = pytest.mark.skipif(
    not hasattr(os, "mkfifo"), reason="named pipes need os.mkfifo, which only POSIX has"
)

#: Skip a test node that spawns gloo DDP processes, which the Windows CI runners do not support.
_requires_gloo_spawn = pytest.mark.skipif(sys.platform == "win32", reason="gloo DDP spawn unsupported on Windows CI")

#: Skip a test node that needs the ``openvino`` runtime.
_openvino_only = pytest.mark.skipif(not _IS_OPENVINO_INSTALLED, reason="openvino not installed")

#: Skip a test node that needs ``albumentations`` (the ``augment`` extra).
_albumentations_only = pytest.mark.skipif(not _IS_ALBUMENTATIONS_INSTALLED, reason="albumentations not installed")

#: Skip a test node that needs ``kornia`` (the ``augment`` extra).
_kornia_only = pytest.mark.skipif(not _IS_KORNIA_INSTALLED, reason="kornia not installed")

#: Skip a test node that needs a COCO evaluation backend; each gates one ``TrainConfig.eval_backend`` choice.
_pycocotools_only = pytest.mark.skipif(not _IS_PYCOCOTOOLS_INSTALLED, reason="pycocotools not installed")
_faster_coco_eval_only = pytest.mark.skipif(not _IS_FASTER_COCO_EVAL_INSTALLED, reason="faster_coco_eval not installed")
_hotcoco_only = pytest.mark.skipif(not _IS_HOTCOCO_INSTALLED, reason="hotcoco not installed")
_ufcoco_only = pytest.mark.skipif(not _IS_UFCOCO_INSTALLED, reason="ultrafast_pycocotools not installed")
_vernier_only = pytest.mark.skipif(not _IS_VERNIER_INSTALLED, reason="vernier not installed")

#: Skip a test node that creates a symbolic link, which needs extra privileges on Windows.
_requires_symlink_privilege = pytest.mark.skipif(
    os.name == "nt", reason="creating a symbolic link needs extra privileges on Windows"
)

#: Skip a test node that runs BF16 on CUDA.
_requires_cuda_bf16 = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(), reason="CUDA BF16 unavailable"
)
