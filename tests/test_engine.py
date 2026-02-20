# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------

import torch

from rfdetr.engine import _get_cuda_autocast_dtype


def test_get_cuda_autocast_dtype_prefers_bfloat16_when_supported(monkeypatch) -> None:
    """Use bfloat16 when CUDA reports BF16 support."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert _get_cuda_autocast_dtype() == torch.bfloat16


def test_get_cuda_autocast_dtype_falls_back_to_float16_when_bfloat16_unsupported(monkeypatch) -> None:
    """Use float16 on CUDA devices that do not support BF16 (e.g. T4)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    assert _get_cuda_autocast_dtype() == torch.float16
