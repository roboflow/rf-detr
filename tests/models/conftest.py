# ------------------------------------------------------------------------
# RF-DETR
# Copyright (c) 2025 Roboflow. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
"""Shared fixtures for the models test suite."""

from collections.abc import Iterator

import pytest
import torch


@pytest.fixture(autouse=True)
def reset_torch_safe_globals():
    """Reset torch serialization safe globals after each test.

    Prevents cross-test state contamination caused by ``_safe_torch_load``'s Attempt 2 path, which calls
    ``torch.serialization.add_safe_globals``. Without this reset, globals registered by one test bleed into subsequent
    tests and can mask trust-gate failures.
    """
    yield
    try:
        torch.serialization.clear_safe_globals()
    except AttributeError:
        pass  # torch <2.4 does not have clear_safe_globals


@pytest.fixture(params=["highest", "high"])
def float32_matmul_precision(request: pytest.FixtureRequest) -> Iterator[str]:
    """Run a test under each float32 matmul precision a process can be left in, then restore the previous one.

    ``"highest"`` is true fp32 and ``"high"`` allows TF32 GEMMs; ``build_trainer`` sets ``"high"`` for the process, so
    production runs under it and a bitwise comparison of two fp32 GEMM routes must hold under both.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``request``):

        >>> float32_matmul_precision  # doctest: +SKIP
        <pytest_fixture(<function float32_matmul_precision at 0x...>)>
    """
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision(request.param)
    yield request.param
    torch.set_float32_matmul_precision(previous_precision)


@pytest.fixture
def highest_float32_matmul_precision() -> Iterator[None]:
    """Pin float32 matmuls to true fp32 for one test, then restore the previous precision.

    ``build_trainer`` leaves TF32 matmuls on for the process, so a test that compares fp32 results must opt out.

    Examples:
        Skipped because a pytest fixture has no standalone call:

        >>> highest_float32_matmul_precision  # doctest: +SKIP
        <pytest_fixture(<function highest_float32_matmul_precision at 0x...>)>
    """
    previous_precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    yield
    torch.set_float32_matmul_precision(previous_precision)


@pytest.fixture
def forced_eager_cuda_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Open the shared eager-CUDA gate of the decoder rewrites on CPU with a bf16 autocast dtype.

    Tests that use it assert routing only: there is no autocast on CPU, so the rewrites are replaced by spies.

    Examples:
        Skipped because a pytest fixture has no standalone call (pytest injects ``monkeypatch``):

        >>> forced_eager_cuda_gate  # doctest: +SKIP
        <pytest_fixture(<function forced_eager_cuda_gate at 0x...>)>
    """
    monkeypatch.setattr("rfdetr.models.transformer._eager_cuda", lambda tensor: True)
    monkeypatch.setattr("rfdetr.models.transformer._cuda_autocast_dtype", lambda: torch.bfloat16)
